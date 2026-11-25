//! Authenticated, expiring logical memory reservations for GGUF layer placement.
//!
//! Reservations are accounting leases only. They do not allocate or lock OS
//! memory, load weights, transfer payloads, or execute model code.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    net::SocketAddr,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    auth,
    control::{AllocationPrepare, AllocationRelease},
    planner::PlacementPlan,
};

pub const DEFAULT_LEASE_TTL_SECS: u64 = 120;
pub const MAX_LEASE_TTL_SECS: u64 = 300;
pub const MAX_ACTIVE_LEASES: usize = 64;
pub const MAX_TRACKED_LEASES: usize = 512;
pub const MAX_LEASE_FILE_BYTES: u64 = 256 * 1024;
pub const MAX_PLACEMENT_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_RESERVED_BYTES_PER_LEASE: u64 = 1_u64 << 40;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationLeaseTarget {
    pub node_id: String,
    pub node_session_id: String,
    pub address: String,
    pub first_layer: Option<u64>,
    pub end_layer_exclusive: Option<u64>,
    pub reserved_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationLeaseRecord {
    pub schema_version: u8,
    pub lease_id: String,
    pub coordinator_node_id: String,
    pub model_filename: String,
    pub model_sha256: String,
    pub created_at_unix: u64,
    pub expires_at_unix: u64,
    pub targets: Vec<AllocationLeaseTarget>,
}

#[derive(Debug, Clone)]
struct ActiveLease {
    request: AllocationPrepare,
    expires_at: Instant,
}

/// Per-node logical reservations, scoped to the current process session.
#[derive(Debug, Default)]
pub struct NodeAllocationBook {
    active: HashMap<String, ActiveLease>,
    released: HashMap<String, Instant>,
}

impl NodeAllocationBook {
    /// Authenticate a prepare request and account for its memory without
    /// allocating an OS buffer. Repeating an identical active request is safe.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        request: &AllocationPrepare,
        node_id: &str,
        session_id: &str,
        key: &[u8],
        available_memory_bytes: u64,
        now: Instant,
        now_unix: u64,
    ) -> Result<(), &'static str> {
        self.expire(now);

        if !auth::verify_allocation_prepare(request, key)
            .map_err(|_| "allocation authentication could not be verified")?
        {
            return Err("invalid allocation authentication");
        }
        if request.target_node_id != node_id || request.target_session_id != session_id {
            return Err("allocation targets a different node process session");
        }

        let ttl_secs = request.expires_at_unix.saturating_sub(now_unix);
        if ttl_secs == 0 || ttl_secs > MAX_LEASE_TTL_SECS {
            return Err("allocation lease is expired or exceeds the maximum TTL");
        }

        if self.released.contains_key(&request.lease_id) {
            return Err("allocation lease has already been released");
        }
        if let Some(existing) = self.active.get(&request.lease_id) {
            return if same_prepare(&existing.request, request) {
                Ok(())
            } else {
                Err("lease ID is already bound to a different allocation")
            };
        }
        if self.active.len() >= MAX_ACTIVE_LEASES {
            return Err("node allocation lease capacity is full");
        }
        if self.active.len() + self.released.len() >= MAX_TRACKED_LEASES {
            return Err("node allocation replay-tracking capacity is full");
        }

        let requested_total = (self.reserved_bytes() as u128)
            .checked_add(request.reserved_bytes as u128)
            .ok_or("reserved memory total overflow")?;
        if requested_total > available_memory_bytes as u128 {
            return Err("requested allocation exceeds currently available unreserved RAM");
        }

        self.active.insert(
            request.lease_id.clone(),
            ActiveLease {
                request: request.clone(),
                expires_at: now + Duration::from_secs(ttl_secs),
            },
        );
        Ok(())
    }

    /// Release a lease. Repeated releases are idempotent. A bounded tombstone
    /// also prevents a delayed prepare datagram from resurrecting a released
    /// lease during the maximum lease lifetime.
    pub fn release(
        &mut self,
        request: &AllocationRelease,
        node_id: &str,
        session_id: &str,
        key: &[u8],
        now: Instant,
    ) -> Result<bool, &'static str> {
        self.expire(now);

        if !auth::verify_allocation_release(request, key)
            .map_err(|_| "allocation release authentication could not be verified")?
        {
            return Err("invalid allocation release authentication");
        }
        if request.target_node_id != node_id || request.target_session_id != session_id {
            return Err("allocation release targets a different node process session");
        }

        if self.released.contains_key(&request.lease_id) {
            return Ok(true);
        }

        if let Some(existing) = self.active.get(&request.lease_id) {
            if existing.request.coordinator_node_id != request.coordinator_node_id {
                return Err("allocation lease belongs to a different coordinator");
            }
            self.active.remove(&request.lease_id);
        } else if self.active.len() + self.released.len() >= MAX_TRACKED_LEASES {
            return Err("node allocation replay-tracking capacity is full");
        }

        self.released.insert(
            request.lease_id.clone(),
            now + Duration::from_secs(MAX_LEASE_TTL_SECS),
        );
        Ok(true)
    }

    /// Expire monotonic lease deadlines and old replay tombstones.
    pub fn expire(&mut self, now: Instant) -> usize {
        let expired: Vec<String> = self
            .active
            .iter()
            .filter(|(_, lease)| lease.expires_at <= now)
            .map(|(lease_id, _)| lease_id.clone())
            .collect();

        for lease_id in &expired {
            self.active.remove(lease_id);
            self.released.insert(
                lease_id.clone(),
                now + Duration::from_secs(MAX_LEASE_TTL_SECS),
            );
        }
        self.released.retain(|_, until| *until > now);
        expired.len()
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.active
            .values()
            .map(|lease| lease.request.reserved_bytes)
            .sum()
    }

    pub fn lease_count(&self) -> usize {
        self.active.len()
    }
}

fn same_prepare(left: &AllocationPrepare, right: &AllocationPrepare) -> bool {
    left.version == right.version
        && left.coordinator_node_id == right.coordinator_node_id
        && left.target_node_id == right.target_node_id
        && left.target_session_id == right.target_session_id
        && left.lease_id == right.lease_id
        && left.model_sha256 == right.model_sha256
        && left.first_layer == right.first_layer
        && left.end_layer_exclusive == right.end_layer_exclusive
        && left.reserved_bytes == right.reserved_bytes
        && left.expires_at_unix == right.expires_at_unix
}

pub fn unix_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

pub fn load_placement_plan(path: &Path) -> Result<PlacementPlan> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting placement plan {}", path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > MAX_PLACEMENT_FILE_BYTES
    {
        bail!(
            "placement plan must be a non-empty regular file no larger than {} bytes",
            MAX_PLACEMENT_FILE_BYTES
        );
    }

    let encoded =
        fs::read(path).with_context(|| format!("reading placement plan {}", path.display()))?;
    if encoded.len() as u64 > MAX_PLACEMENT_FILE_BYTES {
        bail!("placement plan exceeds the configured size limit");
    }

    let plan: PlacementPlan =
        serde_json::from_slice(&encoded).context("parsing GGUF placement plan JSON")?;
    validate_plan(&plan)?;
    Ok(plan)
}

fn validate_plan(plan: &PlacementPlan) -> Result<()> {
    if plan.schema_version != 1
        || plan.layer_count == 0
        || plan.layer_count > 4096
        || plan.model_tensor_bytes == 0
        || plan.non_layer_tensor_bytes > plan.model_tensor_bytes
        || plan.model_filename.is_empty()
        || plan.model_filename.len() > 512
        || plan.model_filename.contains('/')
        || plan.model_filename.contains('\\')
    {
        bail!("unsupported or incomplete GGUF placement plan");
    }
    if plan.node_assignments.is_empty() || plan.node_assignments.len() > 256 {
        bail!("placement plan must contain between 1 and 256 node assignments");
    }

    let mut node_ids = HashSet::new();
    let mut ranges = Vec::new();
    let mut assigned_bytes = 0u128;
    let mut assigned_layer_bytes = 0u128;
    let mut assigned_non_layer_bytes = 0u128;
    let mut used_nodes = 0usize;

    for node in &plan.node_assignments {
        if uuid::Uuid::parse_str(&node.node_id).is_err() || !node_ids.insert(node.node_id.as_str())
        {
            bail!("placement plan contains an invalid or duplicate node ID");
        }
        if node.assigned_weight_bytes == 0 {
            continue;
        }

        used_nodes += 1;
        if node.usable_memory_bytes == 0
            || node.available_memory_bytes > node.total_memory_bytes
            || node.usable_memory_bytes > node.available_memory_bytes
            || node.assigned_weight_bytes > node.usable_memory_bytes
            || node.layer_weight_bytes
                .checked_add(node.non_layer_tensor_bytes)
                != Some(node.assigned_weight_bytes)
        {
            bail!("node '{}' has an invalid assigned memory total", node.node_id);
        }

        let session_id = node.session_id.as_deref().context(
            "placement plan is missing a node process session; export a fresh manager snapshot and re-run plan-gguf",
        )?;
        if uuid::Uuid::parse_str(session_id).is_err() {
            bail!("node '{}' has an invalid process session ID", node.node_id);
        }
        let address: SocketAddr = node.address.parse().with_context(|| {
            format!(
                "node '{}' has an invalid control-plane address",
                node.node_id
            )
        })?;
        if address.port() == 0 || address.ip().is_unspecified() {
            bail!("node '{}' has an unusable control-plane address", node.node_id);
        }

        match (node.first_layer, node.end_layer_exclusive) {
            (Some(start), Some(end)) if start < end && end <= plan.layer_count => {
                if end - start != node.layer_count {
                    bail!(
                        "node '{}' has an inconsistent transformer layer range",
                        node.node_id
                    );
                }
                ranges.push((start, end));
            }
            (None, None) if node.layer_count == 0 => {}
            _ => bail!(
                "node '{}' has an invalid transformer layer range",
                node.node_id
            ),
        }

        assigned_bytes = assigned_bytes
            .checked_add(node.assigned_weight_bytes as u128)
            .context("placement plan assigned-byte total overflow")?;
        assigned_layer_bytes = assigned_layer_bytes
            .checked_add(node.layer_weight_bytes as u128)
            .context("placement plan layer-byte total overflow")?;
        assigned_non_layer_bytes = assigned_non_layer_bytes
            .checked_add(node.non_layer_tensor_bytes as u128)
            .context("placement plan non-layer-byte total overflow")?;
    }

    if used_nodes == 0 || used_nodes != plan.used_nodes {
        bail!("placement plan used-node count does not match its assignments");
    }
    if assigned_bytes != plan.model_tensor_bytes as u128
        || assigned_layer_bytes != (plan.model_tensor_bytes - plan.non_layer_tensor_bytes) as u128
        || assigned_non_layer_bytes != plan.non_layer_tensor_bytes as u128
    {
        bail!("placement plan assignments do not match the complete GGUF layer and non-layer byte totals");
    }

    ranges.sort_unstable();
    let mut next_layer = 0u64;
    for (start, end) in ranges {
        if start != next_layer {
            bail!("placement plan has overlapping or missing transformer layers");
        }
        next_layer = end;
    }
    if next_layer != plan.layer_count {
        bail!("placement plan does not assign every transformer layer");
    }
    Ok(())
}

pub fn targets_from_plan(
    plan: &PlacementPlan,
) -> Result<Vec<AllocationLeaseTarget>> {
    validate_plan(plan)?;
    let mut targets = Vec::new();
    for node in &plan.node_assignments {
        if node.assigned_weight_bytes == 0 {
            continue;
        }
        targets.push(AllocationLeaseTarget {
            node_id: node.node_id.clone(),
            node_session_id: node
                .session_id
                .clone()
                .context("placement plan is missing a node process session")?,
            address: node.address.clone(),
            first_layer: node.first_layer,
            end_layer_exclusive: node.end_layer_exclusive,
            reserved_bytes: node.assigned_weight_bytes,
        });
    }
    Ok(targets)
}

/// Hash the exact model file referenced by a placement. It is streamed in a
/// fixed 1 MiB buffer; no model-sized allocation is made.
pub fn sha256_model(
    path: &Path,
    expected_bytes: u64,
    expected_filename: &str,
) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting GGUF model {}", path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() != expected_bytes
    {
        bail!("GGUF model must be a regular file matching the placement plan's exact file size");
    }
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("GGUF model filename is not valid UTF-8")?;
    if filename != expected_filename {
        bail!(
            "placement plan refers to '{}', but the supplied model is '{}'",
            expected_filename,
            filename
        );
    }

    let mut file =
        File::open(path).with_context(|| format!("opening GGUF model {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn load_lease_record(path: &Path) -> Result<AllocationLeaseRecord> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting allocation lease {}", path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > MAX_LEASE_FILE_BYTES
    {
        bail!(
            "lease record must be a non-empty regular file no larger than {} bytes",
            MAX_LEASE_FILE_BYTES
        );
    }

    let encoded =
        fs::read(path).with_context(|| format!("reading allocation lease {}", path.display()))?;
    if encoded.len() as u64 > MAX_LEASE_FILE_BYTES {
        bail!("allocation lease record exceeds the configured size limit");
    }
    let record: AllocationLeaseRecord =
        serde_json::from_slice(&encoded).context("parsing allocation lease record")?;
    validate_lease_record(&record)?;
    Ok(record)
}

pub fn validate_lease_record(record: &AllocationLeaseRecord) -> Result<()> {
    if record.schema_version != 1
        || uuid::Uuid::parse_str(&record.lease_id).is_err()
        || uuid::Uuid::parse_str(&record.coordinator_node_id).is_err()
        || record.model_filename.is_empty()
        || record.model_filename.len() > 512
        || record.model_filename.contains('/')
        || record.model_filename.contains('\\')
        || record.model_sha256.len() != 64
        || !record.model_sha256.chars().all(|c| c.is_ascii_hexdigit())
        || record.expires_at_unix <= record.created_at_unix
        || record.expires_at_unix - record.created_at_unix > MAX_LEASE_TTL_SECS
        || record.targets.is_empty()
        || record.targets.len() > 256
    {
        bail!("allocation lease record contains invalid metadata");
    }

    let mut node_ids = HashSet::new();
    for target in &record.targets {
        if uuid::Uuid::parse_str(&target.node_id).is_err()
            || !node_ids.insert(target.node_id.as_str())
            || uuid::Uuid::parse_str(&target.node_session_id).is_err()
            || target.reserved_bytes == 0
            || target.reserved_bytes > MAX_RESERVED_BYTES_PER_LEASE
        {
            bail!("allocation lease record contains an invalid node target");
        }
        let address: SocketAddr = target
            .address
            .parse()
            .with_context(|| format!("invalid lease target address '{}'", target.address))?;
        if address.port() == 0 || address.ip().is_unspecified() {
            bail!("allocation lease record contains an unusable target address");
        }
        match (target.first_layer, target.end_layer_exclusive) {
            (Some(start), Some(end)) if start < end && end <= 4096 => {}
            (None, None) => {}
            _ => bail!("allocation lease record contains an invalid layer range"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        auth::{sign_allocation_prepare, sign_allocation_release},
        control::{AllocationPrepare, AllocationRelease, CONTROL_VERSION},
    };

    fn signed_prepare(
        key: &[u8],
        lease_id: &str,
        bytes: u64,
        expiry: u64,
    ) -> AllocationPrepare {
        let mut request = AllocationPrepare {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: lease_id.into(),
            model_sha256: "a".repeat(64),
            first_layer: Some(0),
            end_layer_exclusive: Some(4),
            reserved_bytes: bytes,
            expires_at_unix: expiry,
            auth_tag: Vec::new(),
        };
        sign_allocation_prepare(&mut request, key).unwrap();
        request
    }

    fn signed_release(key: &[u8], lease_id: &str) -> AllocationRelease {
        let mut request = AllocationRelease {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: lease_id.into(),
            auth_tag: Vec::new(),
        };
        sign_allocation_release(&mut request, key).unwrap();
        request
    }

    #[test]
    fn placement_plan_preserves_fresh_worker_session_and_checks_layer_cover() {
        let plan = PlacementPlan {
            schema_version: 1,
            model_filename: "model.gguf".into(),
            architecture: Some("llama".into()),
            file_bytes: 2048,
            layer_count: 1,
            tensor_count: 2,
            model_tensor_bytes: 1024,
            non_layer_tensor_bytes: 128,
            non_layer_tensor_count: 1,
            reserve_percent: 20,
            target_max_utilization_percent: 50.0,
            online_nodes_considered: 1,
            offline_nodes_ignored: 0,
            zero_capacity_nodes_ignored: 0,
            used_nodes: 1,
            node_assignments: vec![crate::planner::NodePlacement {
                node_id: "00000000-0000-4000-8000-000000000002".into(),
                hostname: "worker".into(),
                address: "127.0.0.1:9001".into(),
                session_id: Some("00000000-0000-4000-8000-000000000003".into()),
                total_memory_bytes: 4096,
                available_memory_bytes: 2048,
                usable_memory_bytes: 1800,
                non_layer_tensor_bytes: 128,
                first_layer: Some(0),
                end_layer_exclusive: Some(1),
                layer_count: 1,
                layer_weight_bytes: 896,
                assigned_weight_bytes: 1024,
                utilization_percent: 56.8,
            }],
            caveats: Vec::new(),
        };
        let targets = targets_from_plan(&plan).unwrap();
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0].node_session_id,
            "00000000-0000-4000-8000-000000000003"
        );

        let mut stale = plan.clone();
        stale.node_assignments[0].session_id = None;
        assert!(targets_from_plan(&stale).is_err());

        let mut incomplete = plan;
        incomplete.node_assignments[0].end_layer_exclusive = Some(2);
        assert!(targets_from_plan(&incomplete).is_err());
    }

    #[test]
    fn admits_and_accounts_for_authenticated_memory_leases() {
        let key = [0x45; 32];
        let now = Instant::now();
        let request = signed_prepare(
            &key,
            "00000000-0000-4000-8000-000000000004",
            512,
            10_120,
        );
        let mut book = NodeAllocationBook::default();
        book.prepare(
            &request,
            &request.target_node_id,
            &request.target_session_id,
            &key,
            1024,
            now,
            10_000,
        )
        .unwrap();
        assert_eq!(book.lease_count(), 1);
        assert_eq!(book.reserved_bytes(), 512);
    }

    #[test]
    fn duplicate_prepare_is_idempotent_but_does_not_double_reserve() {
        let key = [0x46; 32];
        let now = Instant::now();
        let request = signed_prepare(
            &key,
            "00000000-0000-4000-8000-000000000005",
            512,
            10_120,
        );
        let mut book = NodeAllocationBook::default();
        for _ in 0..2 {
            book.prepare(
                &request,
                &request.target_node_id,
                &request.target_session_id,
                &key,
                1024,
                now,
                10_000,
            )
            .unwrap();
        }
        assert_eq!(book.lease_count(), 1);
        assert_eq!(book.reserved_bytes(), 512);
    }

    #[test]
    fn rejects_wrong_session_and_overcommitted_memory() {
        let key = [0x47; 32];
        let now = Instant::now();
        let request = signed_prepare(
            &key,
            "00000000-0000-4000-8000-000000000006",
            512,
            10_120,
        );
        let mut book = NodeAllocationBook::default();
        assert!(book
            .prepare(
                &request,
                &request.target_node_id,
                "00000000-0000-4000-8000-000000000099",
                &key,
                1024,
                now,
                10_000,
            )
            .is_err());
        assert!(book
            .prepare(
                &request,
                &request.target_node_id,
                &request.target_session_id,
                &key,
                400,
                now,
                10_000,
            )
            .is_err());
        assert_eq!(book.lease_count(), 0);
    }

    #[test]
    fn expires_leases_and_rejects_stale_requests() {
        let key = [0x48; 32];
        let now = Instant::now();
        let request = signed_prepare(
            &key,
            "00000000-0000-4000-8000-000000000007",
            256,
            10_010,
        );
        let mut book = NodeAllocationBook::default();
        book.prepare(
            &request,
            &request.target_node_id,
            &request.target_session_id,
            &key,
            1024,
            now,
            10_000,
        )
        .unwrap();
        assert_eq!(book.expire(now + Duration::from_secs(11)), 1);
        assert_eq!(book.reserved_bytes(), 0);
        assert!(book
            .prepare(
                &request,
                &request.target_node_id,
                &request.target_session_id,
                &key,
                1024,
                now + Duration::from_secs(11),
                10_011,
            )
            .is_err());
    }

    #[test]
    fn release_is_idempotent_and_prevents_prepare_replay() {
        let key = [0x49; 32];
        let now = Instant::now();
        let lease_id = "00000000-0000-4000-8000-000000000008";
        let prepare = signed_prepare(&key, lease_id, 256, 10_120);
        let release = signed_release(&key, lease_id);
        let mut book = NodeAllocationBook::default();
        book.prepare(
            &prepare,
            &prepare.target_node_id,
            &prepare.target_session_id,
            &key,
            1024,
            now,
            10_000,
        )
        .unwrap();
        assert!(book
            .release(
                &release,
                &prepare.target_node_id,
                &prepare.target_session_id,
                &key,
                now,
            )
            .unwrap());
        assert!(book
            .release(
                &release,
                &prepare.target_node_id,
                &prepare.target_session_id,
                &key,
                now,
            )
            .unwrap());
        assert_eq!(book.reserved_bytes(), 0);
        assert!(book
            .prepare(
                &prepare,
                &prepare.target_node_id,
                &prepare.target_session_id,
                &key,
                1024,
                now,
                10_000,
            )
            .is_err());
    }
}
