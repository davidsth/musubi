//! Advisory, memory-aware placement of contiguous GGUF transformer layers.
//!
//! The planner is deliberately offline: it consumes a local GGUF metadata
//! summary and an exported cluster-dashboard snapshot. It never sends commands
//! to nodes, reserves memory remotely, or starts inference.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::Path};

use crate::gguf::GgufModelSummary;

const MAX_NODE_INVENTORY_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CLUSTER_NODES: usize = 256;
const MAX_PLAN_LAYERS: u64 = 4096;
const UTILIZATION_BASIS_POINTS: u32 = 10_000;

fn default_online() -> bool {
    true
}

/// The fields needed from the manager's read-only cluster snapshot. Unknown
/// dashboard fields are ignored so the input remains forward-compatible.
#[derive(Debug, Clone, Deserialize)]
pub struct PlannerNode {
    pub node_id: String,
    pub hostname: String,
    #[serde(default)]
    pub address: String,
    /// Current authenticated process session, required for live reservations.
    #[serde(default)]
    pub session_id: Option<String>,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    #[serde(default = "default_online")]
    pub online: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NodeInventory {
    Snapshot { nodes: Vec<PlannerNode> },
    List(Vec<PlannerNode>),
}

/// Read either the dashboard's full /api/cluster JSON object or a bare node
/// array. The input is intentionally small and bounded before parsing.
pub fn load_nodes(path: &Path) -> Result<Vec<PlannerNode>> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting node inventory {}", path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > MAX_NODE_INVENTORY_BYTES
    {
        bail!(
            "node inventory must be a non-empty regular file no larger than {} bytes",
            MAX_NODE_INVENTORY_BYTES
        );
    }

    let encoded =
        fs::read(path).with_context(|| format!("reading node inventory {}", path.display()))?;
    if encoded.len() as u64 > MAX_NODE_INVENTORY_BYTES {
        bail!("node inventory exceeds the configured size limit");
    }
    let inventory: NodeInventory =
        serde_json::from_slice(&encoded).context("parsing cluster node inventory JSON")?;
    let nodes = match inventory {
        NodeInventory::Snapshot { nodes } | NodeInventory::List(nodes) => nodes,
    };
    if nodes.is_empty() {
        bail!("node inventory contains no nodes");
    }
    if nodes.len() > MAX_CLUSTER_NODES {
        bail!(
            "node inventory contains {} entries; the maximum is {}",
            nodes.len(),
            MAX_CLUSTER_NODES
        );
    }
    Ok(nodes)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePlacement {
    pub node_id: String,
    pub hostname: String,
    pub address: String,
    /// Current worker process session observed in the manager snapshot.
    #[serde(default)]
    pub session_id: Option<String>,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    /// Available RAM after the requested reserve is withheld.
    pub usable_memory_bytes: u64,
    /// Non-block tensors are charged to the largest-capacity node only.
    pub non_layer_tensor_bytes: u64,
    /// Inclusive first transformer layer, when this node receives layers.
    pub first_layer: Option<u64>,
    /// Exclusive end of the contiguous transformer-layer range.
    pub end_layer_exclusive: Option<u64>,
    pub layer_count: u64,
    pub layer_weight_bytes: u64,
    pub assigned_weight_bytes: u64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub schema_version: u8,
    pub model_filename: String,
    pub architecture: Option<String>,
    pub file_bytes: u64,
    pub layer_count: u64,
    pub tensor_count: u64,
    /// Sum of all known GGUF tensor storage sizes.
    pub model_tensor_bytes: u64,
    /// Tensor bytes not associated with a canonical blk.N.* layer name.
    pub non_layer_tensor_bytes: u64,
    pub non_layer_tensor_count: u64,
    pub reserve_percent: u8,
    pub target_max_utilization_percent: f64,
    pub online_nodes_considered: usize,
    pub offline_nodes_ignored: usize,
    pub zero_capacity_nodes_ignored: usize,
    pub used_nodes: usize,
    pub node_assignments: Vec<NodePlacement>,
    pub caveats: Vec<String>,
}

#[derive(Debug, Clone)]
struct CapacityNode {
    node: PlannerNode,
    usable_memory_bytes: u64,
}

#[derive(Debug, Default)]
struct Allocation {
    first_layer: Option<usize>,
    end_layer_exclusive: Option<usize>,
    layer_weight_bytes: u64,
    non_layer_tensor_bytes: u64,
}

/// Build a deterministic, contiguous layer placement using only current free
/// system RAM. reserve_percent is withheld from each node's available RAM.
pub fn create_plan(
    model: &GgufModelSummary,
    inventory: &[PlannerNode],
    reserve_percent: u8,
) -> Result<PlacementPlan> {
    if reserve_percent > 90 {
        bail!("--reserve-percent must be between 0 and 90");
    }
    if inventory.is_empty() {
        bail!("no cluster nodes were supplied");
    }
    if inventory.len() > MAX_CLUSTER_NODES {
        bail!(
            "cluster inventory has {} nodes; the maximum is {}",
            inventory.len(),
            MAX_CLUSTER_NODES
        );
    }

    let (layer_weights, non_layer_tensor_bytes, non_layer_tensor_count, model_tensor_bytes) =
        inspect_model_weights(model)?;

    let mut node_ids = HashSet::with_capacity(inventory.len());
    let mut active_nodes = Vec::new();
    let mut offline_nodes_ignored = 0usize;
    let mut zero_capacity_nodes_ignored = 0usize;

    for node in inventory {
        if node.node_id.trim().is_empty() || node.node_id.len() > 64 {
            bail!("cluster inventory contains an invalid node ID");
        }
        if node.hostname.trim().is_empty() || node.hostname.len() > 128 {
            bail!("cluster inventory contains an invalid node hostname");
        }
        if !node_ids.insert(node.node_id.as_str()) {
            bail!(
                "cluster inventory contains duplicate node ID '{}'",
                node.node_id
            );
        }
        if !node.online {
            offline_nodes_ignored += 1;
            continue;
        }
        if node.available_memory_bytes > node.total_memory_bytes {
            bail!(
                "node '{}' reports more available RAM than total RAM",
                node.node_id
            );
        }

        let usable_memory_bytes = ((node.available_memory_bytes as u128)
            * ((100 - reserve_percent) as u128)
            / 100) as u64;
        if usable_memory_bytes == 0 {
            zero_capacity_nodes_ignored += 1;
            continue;
        }
        active_nodes.push(CapacityNode {
            node: node.clone(),
            usable_memory_bytes,
        });
    }

    if active_nodes.is_empty() {
        bail!("no online cluster node has usable RAM after the requested reserve");
    }

    // Memory capacity is the only placement signal at this stage. Stable
    // tie-breaking makes repeated previews produce identical assignments.
    active_nodes.sort_by(|left, right| {
        right
            .usable_memory_bytes
            .cmp(&left.usable_memory_bytes)
            .then_with(|| left.node.node_id.cmp(&right.node.node_id))
    });

    let total_usable_bytes: u128 = active_nodes
        .iter()
        .map(|node| node.usable_memory_bytes as u128)
        .sum();
    if model_tensor_bytes as u128 > total_usable_bytes {
        bail!(
            "model tensor weights require {} bytes, but online nodes provide only {} usable bytes after reserve",
            model_tensor_bytes,
            total_usable_bytes
        );
    }
    if non_layer_tensor_bytes > active_nodes[0].usable_memory_bytes {
        bail!(
            "non-layer tensors require {} bytes, exceeding the largest node's {} usable bytes",
            non_layer_tensor_bytes,
            active_nodes[0].usable_memory_bytes
        );
    }
    let largest_node_bytes = active_nodes[0].usable_memory_bytes;
    if let Some(largest_layer_bytes) = layer_weights.iter().max() {
        if *largest_layer_bytes > largest_node_bytes {
            bail!(
                "one transformer layer requires {} bytes, exceeding every node's usable RAM (largest node: {} bytes)",
                largest_layer_bytes,
                largest_node_bytes
            );
        }
    }

    if allocate_layers(
        &layer_weights,
        &active_nodes,
        non_layer_tensor_bytes,
        UTILIZATION_BASIS_POINTS,
    )
    .is_none()
    {
        bail!(
            "no contiguous layer partition fits within online-node RAM after the requested reserve"
        );
    }

    // Find the smallest maximum-utilization threshold (0.01% precision) for
    // which all ordered layers fit in contiguous ranges across the nodes.
    let mut low = 0u32;
    let mut high = UTILIZATION_BASIS_POINTS;
    while low < high {
        let middle = low + (high - low) / 2;
        if allocate_layers(
            &layer_weights,
            &active_nodes,
            non_layer_tensor_bytes,
            middle,
        )
        .is_some()
        {
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    let allocations = allocate_layers(&layer_weights, &active_nodes, non_layer_tensor_bytes, low)
        .context("could not reconstruct the validated placement")?;

    let mut node_assignments = Vec::with_capacity(active_nodes.len());
    let mut max_utilization_percent = 0.0f64;
    for (capacity, allocation) in active_nodes.iter().zip(allocations) {
        let assigned_weight_bytes = allocation
            .layer_weight_bytes
            .checked_add(allocation.non_layer_tensor_bytes)
            .context("node assigned-byte total overflow")?;
        let utilization_percent =
            assigned_weight_bytes as f64 * 100.0 / capacity.usable_memory_bytes as f64;
        max_utilization_percent = max_utilization_percent.max(utilization_percent);

        node_assignments.push(NodePlacement {
            node_id: capacity.node.node_id.clone(),
            hostname: capacity.node.hostname.clone(),
            address: capacity.node.address.clone(),
            session_id: capacity.node.session_id.clone(),
            total_memory_bytes: capacity.node.total_memory_bytes,
            available_memory_bytes: capacity.node.available_memory_bytes,
            usable_memory_bytes: capacity.usable_memory_bytes,
            non_layer_tensor_bytes: allocation.non_layer_tensor_bytes,
            first_layer: allocation.first_layer.map(|value| value as u64),
            end_layer_exclusive: allocation.end_layer_exclusive.map(|value| value as u64),
            layer_count: allocation
                .end_layer_exclusive
                .zip(allocation.first_layer)
                .map(|(end, start)| (end - start) as u64)
                .unwrap_or(0),
            layer_weight_bytes: allocation.layer_weight_bytes,
            assigned_weight_bytes,
            utilization_percent,
        });
    }

    let used_nodes = node_assignments
        .iter()
        .filter(|node| node.assigned_weight_bytes > 0)
        .count();

    Ok(PlacementPlan {
        schema_version: 1,
        model_filename: model.filename.clone(),
        architecture: model.architecture.clone(),
        file_bytes: model.file_bytes,
        layer_count: model.layer_count.unwrap_or(0),
        tensor_count: model.tensor_count,
        model_tensor_bytes,
        non_layer_tensor_bytes,
        non_layer_tensor_count,
        reserve_percent,
        target_max_utilization_percent: max_utilization_percent,
        online_nodes_considered: active_nodes.len(),
        offline_nodes_ignored,
        zero_capacity_nodes_ignored,
        used_nodes,
        node_assignments,
        caveats: vec![
            "Advisory dry-run only: no worker is contacted, no memory is reserved, and no model is loaded.".into(),
            "Estimates cover GGUF tensor storage bytes only; KV cache, activations, runtime workspaces, allocator overhead, model replicas, and transfer buffers are excluded.".into(),
            "All tensors outside canonical blk.<layer>.* names (including embeddings and output tensors) are charged to the largest-RAM node.".into(),
            "Placement uses current available system RAM and a reserve percentage; accelerator memory, CPU throughput, network bandwidth, and topology are not measured.".into(),
            "Only GGUF models with complete, known tensor sizes and canonical blk.<layer>.* transformer tensor names are accepted.".into(),
        ],
    })
}

fn inspect_model_weights(model: &GgufModelSummary) -> Result<(Vec<u64>, u64, u64, u64)> {
    let layer_count = model
        .layer_count
        .context("GGUF model metadata does not declare a transformer layer count")?;
    if layer_count == 0 || layer_count > MAX_PLAN_LAYERS {
        bail!(
            "GGUF layer count must be between 1 and {} for placement planning",
            MAX_PLAN_LAYERS
        );
    }
    if model.tensor_count != model.tensors.len() as u64 {
        bail!("GGUF tensor count does not match the parsed tensor table");
    }
    if model.unsupported_tensor_count != 0 {
        bail!(
            "GGUF contains {} tensors with unsupported storage sizes; refusing an incomplete memory estimate",
            model.unsupported_tensor_count
        );
    }

    let mut layer_weights = vec![0u64; layer_count as usize];
    let mut non_layer_tensor_bytes = 0u64;
    let mut non_layer_tensor_count = 0u64;
    let mut model_tensor_bytes = 0u64;

    for tensor in &model.tensors {
        let byte_len = tensor
            .byte_len
            .with_context(|| format!("tensor '{}' has no verified storage size", tensor.name))?;
        if byte_len == 0 {
            bail!("tensor '{}' has a zero-byte storage size", tensor.name);
        }
        model_tensor_bytes = model_tensor_bytes
            .checked_add(byte_len)
            .context("total GGUF tensor byte count overflow")?;

        match parse_block_layer(&tensor.name)? {
            Some(layer_index) => {
                if layer_index >= layer_weights.len() {
                    bail!(
                        "tensor '{}' refers to layer {}, outside the declared layer count {}",
                        tensor.name,
                        layer_index,
                        layer_count
                    );
                }
                layer_weights[layer_index] = layer_weights[layer_index]
                    .checked_add(byte_len)
                    .with_context(|| format!("weight size overflow for layer {layer_index}"))?;
            }
            None => {
                non_layer_tensor_bytes = non_layer_tensor_bytes
                    .checked_add(byte_len)
                    .context("non-layer tensor byte count overflow")?;
                non_layer_tensor_count = non_layer_tensor_count
                    .checked_add(1)
                    .context("non-layer tensor count overflow")?;
            }
        }
    }

    if model_tensor_bytes != model.known_tensor_bytes {
        bail!(
            "parsed tensor bytes ({}) do not match GGUF known tensor bytes ({})",
            model_tensor_bytes,
            model.known_tensor_bytes
        );
    }
    for (index, bytes) in layer_weights.iter().enumerate() {
        if *bytes == 0 {
            bail!(
                "GGUF layer {} has no recognized blk.{}.* tensors; refusing a partial layer plan",
                index,
                index
            );
        }
    }
    Ok((
        layer_weights,
        non_layer_tensor_bytes,
        non_layer_tensor_count,
        model_tensor_bytes,
    ))
}

fn parse_block_layer(name: &str) -> Result<Option<usize>> {
    let Some(rest) = name.strip_prefix("blk.") else {
        return Ok(None);
    };
    let (index, suffix) = rest
        .split_once('.')
        .with_context(|| format!("malformed GGUF transformer tensor name '{name}'"))?;
    if index.is_empty() || suffix.is_empty() {
        bail!("malformed GGUF transformer tensor name '{name}'");
    }
    let index = index
        .parse::<usize>()
        .with_context(|| format!("invalid GGUF transformer layer index in '{name}'"))?;
    Ok(Some(index))
}

fn allocate_layers(
    layer_weights: &[u64],
    nodes: &[CapacityNode],
    non_layer_tensor_bytes: u64,
    utilization_basis_points: u32,
) -> Option<Vec<Allocation>> {
    let mut next_layer = 0usize;
    let mut allocations = Vec::with_capacity(nodes.len());

    for (node_index, node) in nodes.iter().enumerate() {
        let budget = ((node.usable_memory_bytes as u128) * (utilization_basis_points as u128)
            / (UTILIZATION_BASIS_POINTS as u128)) as u64;
        let fixed_bytes = if node_index == 0 {
            non_layer_tensor_bytes
        } else {
            0
        };
        if fixed_bytes > budget {
            return None;
        }

        let mut used_bytes = fixed_bytes;
        let first_layer = next_layer;
        let mut layer_weight_bytes = 0u64;

        while next_layer < layer_weights.len() {
            let layer_bytes = layer_weights[next_layer];
            if layer_bytes > budget - used_bytes {
                break;
            }
            used_bytes += layer_bytes;
            layer_weight_bytes += layer_bytes;
            next_layer += 1;
        }

        let received_layers = next_layer > first_layer;
        allocations.push(Allocation {
            first_layer: received_layers.then_some(first_layer),
            end_layer_exclusive: received_layers.then_some(next_layer),
            layer_weight_bytes,
            non_layer_tensor_bytes: fixed_bytes,
        });
    }

    if next_layer == layer_weights.len() {
        Some(allocations)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufModelSummary, GgufTensorInfo};
    use std::collections::BTreeMap;

    fn sample_model(layer_bytes: &[u64], shared_bytes: u64) -> GgufModelSummary {
        let mut tensors = Vec::new();
        let mut offset = 0u64;
        for (index, bytes) in layer_bytes.iter().enumerate() {
            tensors.push(GgufTensorInfo {
                name: format!("blk.{index}.attn_q.weight"),
                dimensions: vec![*bytes],
                ggml_type_id: 1,
                type_name: "F16".into(),
                relative_offset: offset,
                file_offset: offset,
                byte_len: Some(*bytes),
            });
            offset += *bytes;
        }
        if shared_bytes > 0 {
            tensors.push(GgufTensorInfo {
                name: "token_embd.weight".into(),
                dimensions: vec![shared_bytes],
                ggml_type_id: 1,
                type_name: "F16".into(),
                relative_offset: offset,
                file_offset: offset,
                byte_len: Some(shared_bytes),
            });
            offset += shared_bytes;
        }

        GgufModelSummary {
            filename: "tiny.gguf".into(),
            file_bytes: offset + 4096,
            version: 3,
            alignment: 32,
            data_offset: 0,
            architecture: Some("llama".into()),
            model_name: None,
            model_basename: None,
            model_size_label: None,
            model_version: None,
            file_type: None,
            quantization_version: None,
            layer_count: Some(layer_bytes.len() as u64),
            context_length: None,
            embedding_length: None,
            feed_forward_length: None,
            parameter_count_metadata: None,
            estimated_tensor_elements: 0,
            tensor_count: tensors.len() as u64,
            metadata_count: 0,
            known_tensor_bytes: offset,
            unsupported_tensor_count: 0,
            tensor_type_counts: BTreeMap::new(),
            tensors,
        }
    }

    fn node(id: &str, available_memory_bytes: u64, online: bool) -> PlannerNode {
        PlannerNode {
            node_id: id.into(),
            hostname: format!("host-{id}"),
            address: format!("192.0.2.{}:9001", id.len() + 1),
            session_id: Some("00000000-0000-4000-8000-000000000042".into()),
            total_memory_bytes: available_memory_bytes.saturating_mul(2),
            available_memory_bytes,
            online,
        }
    }

    #[test]
    fn assigns_ordered_contiguous_layers_and_charges_shared_tensors_once() {
        let model = sample_model(&[40, 30, 20, 10], 5);
        let inventory = vec![node("a", 100, true), node("b", 100, true)];
        let plan = create_plan(&model, &inventory, 0).unwrap();
        assert!(plan.node_assignments.iter().all(|assignment| {
            assignment.session_id.as_deref() == Some("00000000-0000-4000-8000-000000000042")
        }));

        let mut next_layer = 0u64;
        let mut assigned_total = 0u64;
        for assignment in &plan.node_assignments {
            assert!(assignment.assigned_weight_bytes <= assignment.usable_memory_bytes);
            assigned_total += assignment.assigned_weight_bytes;
            if let (Some(start), Some(end)) =
                (assignment.first_layer, assignment.end_layer_exclusive)
            {
                assert_eq!(start, next_layer);
                assert!(end > start);
                next_layer = end;
            }
        }

        assert_eq!(next_layer, 4);
        assert_eq!(assigned_total, 105);
        assert_eq!(plan.node_assignments[0].non_layer_tensor_bytes, 5);
        assert_eq!(plan.node_assignments[1].non_layer_tensor_bytes, 0);
        assert_eq!(plan.used_nodes, 2);
        assert!(plan.target_max_utilization_percent <= 100.0);
    }

    #[test]
    fn excludes_offline_nodes_and_orders_by_safe_capacity() {
        let model = sample_model(&[20, 20], 0);
        let inventory = vec![
            node("offline", 1024, false),
            node("small", 64, true),
            node("large", 128, true),
        ];
        let plan = create_plan(&model, &inventory, 0).unwrap();

        assert_eq!(plan.offline_nodes_ignored, 1);
        assert_eq!(plan.online_nodes_considered, 2);
        assert_eq!(plan.node_assignments[0].node_id, "large");
        assert_eq!(plan.node_assignments[1].node_id, "small");
    }

    #[test]
    fn rejects_a_model_that_exceeds_aggregate_safe_ram() {
        let model = sample_model(&[80, 80], 0);
        let inventory = vec![node("only", 100, true)];
        let error = create_plan(&model, &inventory, 0).unwrap_err();
        assert!(error.to_string().contains("provide only"));
    }

    #[test]
    fn rejects_unknown_tensor_sizes_instead_of_underestimating() {
        let mut model = sample_model(&[20, 20], 0);
        model.unsupported_tensor_count = 1;
        model.tensors[0].byte_len = None;

        let error = create_plan(&model, &[node("a", 100, true)], 0).unwrap_err();
        assert!(error.to_string().contains("unsupported storage sizes"));
    }

    #[test]
    fn rejects_missing_or_out_of_range_layer_indices() {
        let mut model = sample_model(&[20, 20], 0);
        model.tensors[0].name = "blk.2.attn_q.weight".into();

        let error = create_plan(&model, &[node("a", 100, true)], 0).unwrap_err();
        assert!(error
            .to_string()
            .contains("outside the declared layer count"));
    }

    #[test]
    fn rejects_duplicate_node_ids_and_invalid_reserve() {
        let model = sample_model(&[20], 0);
        let duplicate = vec![node("same", 100, true), node("same", 100, true)];
        assert!(create_plan(&model, &duplicate, 0).is_err());
        assert!(create_plan(&model, &[node("a", 100, true)], 91).is_err());
    }

    #[test]
    fn reads_the_cluster_dashboard_snapshot_shape() {
        let path = std::env::temp_dir().join(format!("aitp-planner-{}.json", uuid::Uuid::new_v4()));
        let json = serde_json::json!({
            "nodes": [{
                "node_id": "stable-node",
                "hostname": "mini",
                "address": "192.0.2.10:9001",
                "session_id": "00000000-0000-4000-8000-000000000042",
                "total_memory_bytes": 16_000_000_000u64,
                "available_memory_bytes": 8_000_000_000u64,
                "online": true,
                "last_seen_secs": 1
            }],
            "online_nodes": 1,
            "offline_nodes": 0
        });
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let nodes = load_nodes(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "stable-node");
        assert!(nodes[0].online);
    }
}
