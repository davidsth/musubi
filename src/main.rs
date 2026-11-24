mod auth;
mod control;
mod dashboard;
mod gguf;
mod hardware;
mod planner;
mod protocol;
mod runtime;
mod storage;
mod tensor;
mod tensor_transfer;
mod worker;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use control::{
    ControlMessage, NodeAnnouncement, TensorOfferMessage, TensorOfferReply, CONTROL_VERSION,
    MAX_CONTROL_DATAGRAM,
};
use protocol::*;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use sysinfo::System;
use tensor::TensorDescriptor;
use tokio::{fs, net::UdpSocket, sync::Mutex, time::timeout};
use worker::OfferAdmission;

const CHUNK_SIZE: usize = MAX_PAYLOAD;
const WINDOW_PACKETS: usize = 512;
const RETRY_LIMIT: usize = 20;
const ACK_WAIT: Duration = Duration::from_millis(250);
const MAX_CLUSTER_NODES: usize = 256;

#[derive(Parser)]
#[command(
    name = "aitp",
    version,
    about = "Experimental distributed-inference transport"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Listen for incoming AITP transfers and write received payloads to a file.
    Receive {
        #[arg(long, default_value = "0.0.0.0:9000")]
        bind: SocketAddr,
        #[arg(long, default_value = "received.tensor")]
        output: PathBuf,
    },
    /// Send a file as tensor chunks to an AITP receiver.
    Send {
        #[arg(long)]
        target: SocketAddr,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 1)]
        tensor_id: u32,
    },
    /// Submit authenticated tensor metadata to a worker. No tensor bytes are sent.
    Offer {
        #[arg(long)]
        target: SocketAddr,
        #[arg(long)]
        descriptor: PathBuf,
    },
    /// Offer a tensor and transfer its bytes to an authenticated AITP node.
    SendTensor {
        /// Worker control-plane address (normally UDP port 9001).
        #[arg(long)]
        target: SocketAddr,
        /// Worker tensor data-plane address (defaults to the same IP on UDP port 9000).
        #[arg(long)]
        data_target: Option<SocketAddr>,
        #[arg(long)]
        descriptor: PathBuf,
        #[arg(long)]
        input: PathBuf,
    },
    /// Stream one supported tensor directly from a GGUF model file to a worker.
    SendGgufTensor {
        /// Worker control-plane address (normally UDP port 9001).
        #[arg(long)]
        target: SocketAddr,
        /// Worker tensor data-plane address (defaults to the same IP on UDP port 9000).
        #[arg(long)]
        data_target: Option<SocketAddr>,
        /// Source GGUF model file.
        #[arg(long)]
        model: PathBuf,
        /// Exact tensor name from the GGUF tensor table.
        #[arg(long)]
        tensor_name: String,
        /// AITP tensor ID for this transfer.
        #[arg(long, default_value_t = 1)]
        tensor_id: u32,
    },
    /// Run a LAN node agent with a persistent identity.
    Node {
        #[arg(long, default_value = "0.0.0.0:9001")]
        bind: SocketAddr,
        #[arg(long, default_value_t = 9000)]
        data_port: u16,
        #[arg(long)]
        name: Option<String>,
        /// Optional authenticated manager endpoint; requires AITP_CLUSTER_KEY.
        #[arg(long)]
        manager: Option<SocketAddr>,
        #[arg(long, default_value_t = 1000)]
        heartbeat_ms: u64,
    },
    /// Discover AITP nodes using IPv4 UDP broadcast.
    Discover {
        #[arg(long, default_value = "255.255.255.255:9001")]
        target: SocketAddr,
        #[arg(long, default_value_t = 1500)]
        wait_ms: u64,
    },
    /// List catalogued tensor artifacts in the local AITP tensor store.
    Catalog {
        #[arg(long)]
        directory: Option<PathBuf>,
        /// Recompute SHA-256 for each displayed payload (slower for large tensors).
        #[arg(long, default_value_t = false)]
        verify_hashes: bool,
    },
    /// Prepare one committed tensor for a runtime adapter after same-handle SHA-256 verification.
    PrepareTensor {
        /// Path to the tensor's JSON sidecar manifest.
        #[arg(long)]
        manifest: PathBuf,
        /// Directory containing the payload (defaults to the manifest's parent directory).
        #[arg(long)]
        directory: Option<PathBuf>,
    },
    /// Inspect GGUF model metadata and tensor descriptors without loading model weights.
    InspectGguf {
        #[arg(long)]
        input: PathBuf,
        /// Number of tensor descriptors to print (maximum 128).
        #[arg(long, default_value_t = 20)]
        tensor_limit: usize,
    },
    /// List GGUF models in AITP_MODEL_DIR (default: ./aitp-models).
    GgufCatalog {
        #[arg(long)]
        directory: Option<PathBuf>,
    },
    /// Create a read-only memory-aware layer-placement preview from a GGUF model and cluster snapshot.
    PlanGguf {
        /// Source GGUF model file.
        #[arg(long)]
        model: PathBuf,
        /// JSON exported from the manager's /api/cluster endpoint (or a bare node array).
        #[arg(long)]
        nodes: PathBuf,
        /// Percentage of each node's currently available RAM to keep unused.
        #[arg(long, default_value_t = 20)]
        reserve_percent: u8,
        /// Optional new JSON output file; existing files are never overwritten.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run a cluster manager that authenticates node registrations and heartbeats.
    Manager {
        #[arg(long, default_value = "0.0.0.0:9002")]
        bind: SocketAddr,
        #[arg(long, default_value_t = 5000)]
        timeout_ms: u64,
        /// HTTP dashboard bind address. Defaults to localhost for safety.
        #[arg(long, default_value = "127.0.0.1:9003")]
        dashboard_bind: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Receive { bind, output } => receive(bind, output).await,
        Command::Send {
            target,
            input,
            tensor_id,
        } => send(target, input, tensor_id).await,
        Command::Offer { target, descriptor } => offer_tensor(target, descriptor).await,
        Command::SendTensor {
            target,
            data_target,
            descriptor,
            input,
        } => send_tensor(target, data_target, descriptor, input).await,
        Command::SendGgufTensor {
            target,
            data_target,
            model,
            tensor_name,
            tensor_id,
        } => send_gguf_tensor(target, data_target, model, tensor_name, tensor_id).await,
        Command::Node {
            bind,
            data_port,
            name,
            manager,
            heartbeat_ms,
        } => run_node(bind, data_port, name, manager, heartbeat_ms).await,
        Command::Discover { target, wait_ms } => {
            discover(target, Duration::from_millis(wait_ms)).await
        }
        Command::Catalog {
            directory,
            verify_hashes,
        } => catalog_tensors(directory, verify_hashes).await,
        Command::PrepareTensor {
            manifest,
            directory,
        } => prepare_tensor(manifest, directory).await,
        Command::InspectGguf {
            input,
            tensor_limit,
        } => inspect_gguf(input, tensor_limit),
        Command::GgufCatalog { directory } => catalog_gguf(directory),
        Command::PlanGguf {
            model,
            nodes,
            reserve_percent,
            output,
        } => plan_gguf(model, nodes, reserve_percent, output),
        Command::Manager {
            bind,
            timeout_ms,
            dashboard_bind,
        } => {
            run_manager(
                bind,
                Duration::from_millis(timeout_ms.max(500)),
                dashboard_bind,
            )
            .await
        }
    }
}

fn make_header(kind: u8, message_id: u32, tensor_id: u32, sequence: u32, payload: &[u8]) -> Header {
    Header {
        kind,
        flags: 0,
        message_id,
        tensor_id,
        sequence,
        payload_len: payload.len() as u16,
        checksum: checksum(payload),
        worker_id: 0,
    }
}

async fn catalog_tensors(directory: Option<PathBuf>, verify_hashes: bool) -> Result<()> {
    let directory = directory.unwrap_or_else(storage::configured_store_dir);
    let snapshot = storage::scan_store(&directory).await?;
    println!(
        "AITP tensor store: {} valid tensor(s), {:.2} GiB in {}",
        snapshot.count,
        snapshot.total_bytes as f64 / 1024.0_f64.powi(3),
        directory.display()
    );
    let mut failed_hashes = 0usize;
    for manifest in &snapshot.recent {
        let integrity = if verify_hashes {
            if storage::verify_payload(&directory, manifest).await? {
                "SHA-256 OK"
            } else {
                failed_hashes += 1;
                "SHA-256 MISMATCH"
            }
        } else {
            "not rehashed"
        };
        println!(
            "- {} | {:?} | {:?} | shape {:?} | {} bytes | {} | sha256 {}… | {}",
            manifest.descriptor.name,
            manifest.descriptor.dtype,
            manifest.descriptor.encoding,
            manifest.descriptor.shape,
            manifest.byte_len,
            manifest.payload_file,
            &manifest.sha256[..16],
            integrity
        );
    }
    if snapshot.count > snapshot.recent.len() {
        println!(
            "Showing the {} most recent manifests of {} valid tensor(s).",
            snapshot.recent.len(),
            snapshot.count
        );
    }
    if failed_hashes > 0 {
        bail!("{failed_hashes} tensor payload(s) failed SHA-256 verification");
    }
    if verify_hashes {
        println!(
            "Verified SHA-256 for {} displayed tensor(s).",
            snapshot.recent.len()
        );
    }
    Ok(())
}

async fn prepare_tensor(manifest_path: PathBuf, directory: Option<PathBuf>) -> Result<()> {
    let metadata = fs::symlink_metadata(&manifest_path)
        .await
        .with_context(|| format!("inspecting manifest {}", manifest_path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.len() == 0
        || metadata.len() > storage::MAX_MANIFEST_BYTES
    {
        bail!(
            "manifest must be a non-empty regular file no larger than {} bytes",
            storage::MAX_MANIFEST_BYTES
        );
    }

    let encoded = fs::read(&manifest_path)
        .await
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let manifest: storage::TensorManifest =
        serde_json::from_slice(&encoded).context("parsing tensor manifest JSON")?;

    let directory = directory.unwrap_or_else(|| {
        manifest_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    let prepared = runtime::open_verified_tensor(&directory, &manifest).await?;
    println!(
        "Verified tensor '{}' (ID {})",
        prepared.descriptor().name,
        prepared.descriptor().tensor_id
    );
    println!(
        "Shape: {:?} | dtype: {:?} | encoding: {:?}",
        prepared.descriptor().shape,
        prepared.descriptor().dtype,
        prepared.descriptor().encoding
    );
    println!(
        "Payload: {} ({} bytes)",
        prepared.manifest().payload_file,
        prepared.byte_len()
    );
    println!("SHA-256: {}", prepared.manifest().sha256);
    println!("A read-only streaming handle is open at byte 0. No bytes were copied into a second buffer or passed to a model runtime.");
    drop(prepared);
    Ok(())
}

fn plan_gguf(
    model_path: PathBuf,
    nodes_path: PathBuf,
    reserve_percent: u8,
    output: Option<PathBuf>,
) -> Result<()> {
    let model = gguf::inspect_file(&model_path)
        .with_context(|| format!("inspecting GGUF model {}", model_path.display()))?;
    let nodes = planner::load_nodes(&nodes_path)?;
    let plan = planner::create_plan(&model, &nodes, reserve_percent)?;
    let encoded = serde_json::to_string_pretty(&plan)?;

    if let Some(output_path) = output {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
            .with_context(|| {
                format!(
                    "creating placement plan {} without overwriting an existing file",
                    output_path.display()
                )
            })?;
        file.write_all(encoded.as_bytes())
            .with_context(|| format!("writing placement plan {}", output_path.display()))?;
        file.write_all(b"\n")?;
        println!("Wrote advisory placement plan to {}", output_path.display());
        println!(
            "Assigned {} transformer layers across {} node(s); maximum planned weight utilization {:.2}%.",
            plan.layer_count, plan.used_nodes, plan.target_max_utilization_percent
        );
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn inspect_gguf(input: PathBuf, tensor_limit: usize) -> Result<()> {
    if tensor_limit > 128 {
        bail!("--tensor-limit may not exceed 128");
    }
    let model = gguf::inspect_file(&input)?;
    println!(
        "GGUF model: {}",
        model.model_name.as_deref().unwrap_or(&model.filename)
    );
    println!(
        "File: {} ({:.2} GiB)",
        model.filename,
        model.file_bytes as f64 / 1024.0_f64.powi(3)
    );
    println!(
        "Format: v{} little-endian | alignment {} | data offset {}",
        model.version, model.alignment, model.data_offset
    );
    println!(
        "Architecture: {}",
        model.architecture.as_deref().unwrap_or("unknown")
    );
    println!(
        "Model version: {} | size label: {}",
        model.model_version.as_deref().unwrap_or("unknown"),
        model.model_size_label.as_deref().unwrap_or("unknown")
    );
    println!(
        "Layers: {} | context: {} | embedding: {} | feed-forward: {}",
        display_optional(model.layer_count),
        display_optional(model.context_length),
        display_optional(model.embedding_length),
        display_optional(model.feed_forward_length)
    );
    println!(
        "Tensors: {} | estimated elements: {} | metadata keys: {}",
        model.tensor_count, model.estimated_tensor_elements, model.metadata_count
    );
    println!(
        "Known tensor bytes: {:.2} GiB | unsupported type layouts: {}",
        model.known_tensor_bytes as f64 / 1024.0_f64.powi(3),
        model.unsupported_tensor_count
    );
    if let Some(value) = model.parameter_count_metadata {
        println!("Metadata parameter count: {value}");
    }
    if let Some(value) = model.file_type {
        println!("GGUF file type code: {value}");
    }
    if let Some(value) = model.quantization_version {
        println!("Quantization version: {value}");
    }

    println!("Tensor type counts:");
    for (type_name, count) in &model.tensor_type_counts {
        println!("  {type_name}: {count}");
    }

    if tensor_limit > 0 {
        println!(
            "Tensor descriptors (first {}):",
            tensor_limit.min(model.tensors.len())
        );
        for tensor in model.tensors.iter().take(tensor_limit) {
            let bytes = tensor
                .byte_len
                .map(|value| format!("{value} bytes"))
                .unwrap_or_else(|| "size unsupported by this inspector".into());
            println!(
                "  {} | {:?} | {} | offset {} | {}",
                tensor.name, tensor.dimensions, tensor.type_name, tensor.file_offset, bytes
            );
        }
    }
    if tensor_limit < model.tensors.len() {
        println!(
            "Showing {tensor_limit} of {} tensor descriptors.",
            model.tensors.len()
        );
    }
    println!("No tensor payloads were read, allocated, or loaded into an inference runtime.");
    Ok(())
}

fn catalog_gguf(directory: Option<PathBuf>) -> Result<()> {
    let directory = directory.unwrap_or_else(gguf::configured_model_dir);
    let catalog = gguf::scan_model_dir(&directory)?;
    println!(
        "AITP GGUF model catalog: {} valid model(s) in {}",
        catalog.models.len(),
        directory.display()
    );
    println!(
        "Scanned {} file(s), skipped {} invalid file(s), omitted {} file(s) above the scan limit.",
        catalog.scanned_files, catalog.invalid_files, catalog.omitted_files
    );
    for model in &catalog.models {
        let identity = model.model_name.as_deref().unwrap_or(&model.filename);
        println!(
            "- {} | {} | {} | {:.2} GiB | {} tensors | layers {} | context {}",
            identity,
            model.architecture.as_deref().unwrap_or("unknown"),
            model.model_size_label.as_deref().unwrap_or("size unknown"),
            model.file_bytes as f64 / 1024.0_f64.powi(3),
            model.tensor_count,
            display_optional(model.layer_count),
            display_optional(model.context_length)
        );
    }
    Ok(())
}

fn display_optional(value: Option<u64>) -> String {
    value
        .map(|number| number.to_string())
        .unwrap_or_else(|| "unknown".into())
}

async fn send(target: SocketAddr, input: PathBuf, tensor_id: u32) -> Result<()> {
    let data = fs::read(&input)
        .await
        .with_context(|| format!("reading {}", input.display()))?;
    if data.is_empty() {
        bail!("input file is empty");
    }
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.connect(target).await?;
    let message_id = (std::process::id() ^ (data.len() as u32)) | 1;
    let chunks: Vec<&[u8]> = data.chunks(CHUNK_SIZE).collect();
    println!(
        "AITP: sending {} bytes in {} chunks to {}",
        data.len(),
        chunks.len(),
        target
    );

    let mut acked_total = 0usize;
    for (base, batch) in chunks.chunks(WINDOW_PACKETS).enumerate() {
        let first_seq = base * WINDOW_PACKETS;
        let mut acked = vec![false; batch.len()];
        let mut attempts = 0usize;
        while acked.iter().any(|v| !v) {
            for (i, payload) in batch.iter().enumerate() {
                if !acked[i] {
                    let seq = (first_seq + i) as u32;
                    let h = make_header(KIND_DATA, message_id, tensor_id, seq, payload);
                    socket.send(&packet(h, payload)).await?;
                }
            }
            let deadline = tokio::time::Instant::now() + ACK_WAIT;
            let mut buf = [0u8; 2048];
            while tokio::time::Instant::now() < deadline && acked.iter().any(|v| !v) {
                match timeout(
                    deadline - tokio::time::Instant::now(),
                    socket.recv(&mut buf),
                )
                .await
                {
                    Ok(Ok(n)) => {
                        if let Ok(h) = Header::decode(&buf[..n]) {
                            if h.kind == KIND_ACK
                                && h.message_id == message_id
                                && h.tensor_id == tensor_id
                            {
                                let seq = h.sequence as usize;
                                if seq >= first_seq && seq < first_seq + batch.len() {
                                    let idx = seq - first_seq;
                                    if !acked[idx] {
                                        acked[idx] = true;
                                        acked_total += 1;
                                    }
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => break,
                }
            }
            attempts += 1;
            if attempts > RETRY_LIMIT {
                bail!(
                    "retry limit exceeded; {acked_total}/{} chunks acknowledged",
                    chunks.len()
                );
            }
        }
        println!("progress: {}/{} chunks", acked_total, chunks.len());
    }

    let fin_payload = [
        (chunks.len() as u32).to_be_bytes().as_slice(),
        (data.len() as u64).to_be_bytes().as_slice(),
    ]
    .concat();
    let fin = packet(
        make_header(
            KIND_FIN,
            message_id,
            tensor_id,
            chunks.len() as u32,
            &fin_payload,
        ),
        &fin_payload,
    );
    let mut buf = [0u8; 2048];
    for _ in 0..RETRY_LIMIT {
        socket.send(&fin).await?;
        if let Ok(Ok(n)) = timeout(Duration::from_secs(1), socket.recv(&mut buf)).await {
            if let Ok(h) = Header::decode(&buf[..n]) {
                if h.kind == KIND_FIN_ACK && h.message_id == message_id {
                    println!("transfer complete");
                    return Ok(());
                }
            }
        }
    }
    bail!("receiver did not confirm transfer completion")
}

async fn prepare_tensor_offer(
    target: SocketAddr,
    descriptor_path: PathBuf,
) -> Result<(UdpSocket, TensorOfferMessage, Vec<u8>)> {
    let key = auth::cluster_key_from_env()?
        .context("set AITP_CLUSTER_KEY to the same 32+ byte secret used by the worker")?;
    let descriptor_bytes = fs::read(&descriptor_path)
        .await
        .with_context(|| format!("reading {}", descriptor_path.display()))?;
    let descriptor: TensorDescriptor =
        serde_json::from_slice(&descriptor_bytes).context("parsing tensor descriptor JSON")?;
    prepare_tensor_offer_with_descriptor(target, descriptor, key).await
}

async fn prepare_tensor_offer_with_descriptor(
    target: SocketAddr,
    descriptor: TensorDescriptor,
    key: Vec<u8>,
) -> Result<(UdpSocket, TensorOfferMessage, Vec<u8>)> {
    descriptor.validate()?;
    let sender_node_id = persistent_node_id()?;
    let sequence = next_offer_sequence()?;
    let random_id = uuid::Uuid::new_v4();
    let message_id = u32::from_be_bytes(random_id.as_bytes()[0..4].try_into()?) | 1;
    let mut offer = TensorOfferMessage {
        version: CONTROL_VERSION,
        sender_node_id,
        sequence,
        message_id,
        descriptor,
        auth_tag: Vec::new(),
    };
    auth::sign_tensor_offer(&mut offer, &key)?;

    let encoded = serde_json::to_vec(&ControlMessage::TensorOffer {
        offer: offer.clone(),
    })?;
    if encoded.len() > MAX_CONTROL_DATAGRAM {
        bail!(
            "tensor offer is {} bytes; control-plane limit is {}",
            encoded.len(),
            MAX_CONTROL_DATAGRAM
        );
    }

    let bind = if target.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    };
    let socket = UdpSocket::bind(bind).await?;
    socket.send_to(&encoded, target).await?;
    Ok((socket, offer, key))
}

async fn await_tensor_admission(
    socket: &UdpSocket,
    target: SocketAddr,
    offer: &TensorOfferMessage,
    key: &[u8],
) -> Result<TensorOfferReply> {
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    let (n, peer) = timeout(Duration::from_secs(3), socket.recv_from(&mut buf))
        .await
        .context("timed out waiting for tensor-offer admission reply")??;
    if peer != target {
        bail!("tensor-offer reply came from unexpected peer {peer}");
    }
    let response: ControlMessage = serde_json::from_slice(&buf[..n])?;
    let ControlMessage::TensorOfferReply { reply } = response else {
        bail!("worker returned an unexpected control message");
    };
    if !auth::verify_tensor_offer_reply(&reply, key)? {
        bail!("worker reply authentication failed; verify that both nodes share the same AITP_CLUSTER_KEY");
    }
    if reply.sender_node_id != offer.sender_node_id
        || reply.sequence != offer.sequence
        || reply.message_id != offer.message_id
        || reply.tensor_id != offer.descriptor.tensor_id
    {
        bail!("worker reply does not match this tensor offer");
    }
    if !reply.accepted {
        bail!(
            "worker rejected tensor offer: {}",
            reply.reason.as_deref().unwrap_or("no reason supplied")
        );
    }
    Ok(reply)
}

async fn offer_tensor(target: SocketAddr, descriptor_path: PathBuf) -> Result<()> {
    let (socket, offer, key) = prepare_tensor_offer(target, descriptor_path).await?;
    let reply = await_tensor_admission(&socket, target, &offer, &key).await?;
    println!(
        "Worker {} admitted metadata for tensor '{}' ({} bytes).",
        reply.receiver_node_id, offer.descriptor.name, offer.descriptor.byte_len
    );
    println!("No tensor bytes were sent; use send-tensor to complete the authenticated transfer.");
    Ok(())
}

async fn send_tensor(
    target: SocketAddr,
    data_target: Option<SocketAddr>,
    descriptor_path: PathBuf,
    input: PathBuf,
) -> Result<()> {
    let (socket, offer, key) = prepare_tensor_offer(target, descriptor_path).await?;
    let reply = await_tensor_admission(&socket, target, &offer, &key).await?;
    let data_target = data_target.unwrap_or_else(|| SocketAddr::new(target.ip(), 9000));
    if data_target.ip() != target.ip() {
        bail!("control and data targets must use the same worker IP so the admitted UDP peer stays bound to this transfer");
    }
    println!(
        "Worker {} admitted tensor '{}'; starting payload transfer.",
        reply.receiver_node_id, offer.descriptor.name
    );
    tensor_transfer::send_tensor_data(&socket, data_target, &input, &offer, &key).await
}

async fn send_gguf_tensor(
    target: SocketAddr,
    data_target: Option<SocketAddr>,
    model_path: PathBuf,
    tensor_name: String,
    tensor_id: u32,
) -> Result<()> {
    let data_target = data_target.unwrap_or_else(|| SocketAddr::new(target.ip(), 9000));
    if data_target.ip() != target.ip() {
        bail!("control and data targets must use the same worker IP so the admitted UDP peer stays bound to this transfer");
    }

    let source_path = model_path.clone();
    let source_name = tensor_name.clone();
    let source = tokio::task::spawn_blocking(move || {
        gguf::open_tensor_source(&source_path, &source_name, tensor_id)
    })
    .await
    .context("joining GGUF tensor source inspection task")??;
    let (source_file, descriptor, source_offset) = source.into_parts();

    if descriptor.byte_len > worker::DEFAULT_MAX_OFFER_BYTES {
        bail!(
            "GGUF tensor '{}' is {} bytes; the current worker limit is {} bytes",
            descriptor.name,
            descriptor.byte_len,
            worker::DEFAULT_MAX_OFFER_BYTES
        );
    }

    let key = auth::cluster_key_from_env()?
        .context("set AITP_CLUSTER_KEY to the same 32+ byte secret used by the worker")?;
    let (socket, offer, key) =
        prepare_tensor_offer_with_descriptor(target, descriptor, key).await?;
    let reply = await_tensor_admission(&socket, target, &offer, &key).await?;

    println!(
        "Worker {} admitted GGUF tensor '{}'; streaming {} bytes from {} at offset {} without extracting a copy.",
        reply.receiver_node_id,
        offer.descriptor.name,
        offer.descriptor.byte_len,
        model_path.display(),
        source_offset
    );
    let mut source_file = fs::File::from_std(source_file);
    tensor_transfer::send_tensor_range(
        &socket,
        data_target,
        &mut source_file,
        source_offset,
        offer.descriptor.byte_len,
        &offer,
        &key,
    )
    .await
}

async fn receive(bind: SocketAddr, output: PathBuf) -> Result<()> {
    let socket = UdpSocket::bind(bind).await?;
    println!("AITP receiver listening on {}", socket.local_addr()?);
    let mut buf = [0u8; 2048];
    let mut active: Option<(u32, u32, SocketAddr, tokio::fs::File, HashSet<u32>)> = None;

    loop {
        let (n, peer) = socket.recv_from(&mut buf).await?;
        let h = match Header::decode(&buf[..n]) {
            Ok(h) => h,
            Err(_) => continue,
        };
        if HEADER_LEN + h.payload_len as usize != n {
            continue;
        }
        let payload = &buf[HEADER_LEN..n];
        if checksum(payload) != h.checksum {
            continue;
        }

        if h.kind == KIND_DATA {
            let needs_new = active
                .as_ref()
                .map(|a| a.0 != h.message_id || a.1 != h.tensor_id || a.2 != peer)
                .unwrap_or(true);
            if needs_new {
                let file = fs::File::create(&output).await?;
                active = Some((h.message_id, h.tensor_id, peer, file, HashSet::new()));
                println!(
                    "receiving message {} tensor {} from {}",
                    h.message_id, h.tensor_id, peer
                );
            }
            if let Some((msg, tensor, addr, file, received)) = active.as_mut() {
                if *msg == h.message_id && *tensor == h.tensor_id && *addr == peer {
                    use tokio::io::{AsyncSeekExt, AsyncWriteExt};
                    if !received.contains(&h.sequence) {
                        file.seek(std::io::SeekFrom::Start(
                            h.sequence as u64 * CHUNK_SIZE as u64,
                        ))
                        .await?;
                        file.write_all(payload).await?;
                        received.insert(h.sequence);
                    }
                    let ack = packet(
                        make_header(KIND_ACK, h.message_id, h.tensor_id, h.sequence, &[]),
                        &[],
                    );
                    socket.send_to(&ack, peer).await?;
                }
            }
        } else if h.kind == KIND_FIN {
            if payload.len() != 12 {
                continue;
            }
            let total_chunks = u32::from_be_bytes(payload[0..4].try_into()?);
            let total_bytes = u64::from_be_bytes(payload[4..12].try_into()?);
            if let Some((msg, tensor, addr, file, received)) = active.as_mut() {
                if *msg == h.message_id
                    && *tensor == h.tensor_id
                    && *addr == peer
                    && received.len() == total_chunks as usize
                {
                    file.set_len(total_bytes).await?;
                    file.sync_all().await?;
                    let ack = packet(
                        make_header(KIND_FIN_ACK, h.message_id, h.tensor_id, total_chunks, &[]),
                        &[],
                    );
                    socket.send_to(&ack, peer).await?;
                    println!("saved {} bytes to {}", total_bytes, output.display());
                    active = None;
                }
            }
        }
    }
}

fn aitp_state_dir() -> Result<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("HOME").map(PathBuf::from)
    }
    .unwrap_or(std::env::current_dir()?);
    let dir = if cfg!(windows) {
        base.join("aitp")
    } else {
        base.join(".aitp")
    };
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn persistent_node_id() -> Result<String> {
    let path = aitp_state_dir()?.join("node-id");
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let id = existing.trim();
        if uuid::Uuid::parse_str(id).is_ok() {
            return Ok(id.to_owned());
        }
    }
    let id = uuid::Uuid::new_v4().to_string();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    file.write_all(id.as_bytes())?;
    file.sync_all()?;
    Ok(id)
}

fn next_offer_sequence() -> Result<u64> {
    let path = aitp_state_dir()?.join("offer-sequence");
    let previous = match std::fs::read_to_string(&path) {
        Ok(value) => value
            .trim()
            .parse::<u64>()
            .context("parsing persisted tensor-offer sequence")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    let next = previous
        .checked_add(1)
        .context("tensor-offer sequence exhausted")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    file.write_all(next.to_string().as_bytes())?;
    file.sync_all()?;
    Ok(next)
}

async fn run_node(
    bind: SocketAddr,
    data_port: u16,
    name: Option<String>,
    manager: Option<SocketAddr>,
    heartbeat_ms: u64,
) -> Result<()> {
    let cluster_key = auth::cluster_key_from_env()?;
    if manager.is_some() && cluster_key.is_none() {
        bail!("joining a managed cluster requires AITP_CLUSTER_KEY (at least 32 bytes), matching the manager");
    }
    let session_id = uuid::Uuid::new_v4().to_string();
    if data_port == 0 {
        bail!("data port must be nonzero");
    }
    let socket = UdpSocket::bind(bind).await?;
    let node_id = persistent_node_id()?;
    let offer_admission = Arc::new(Mutex::new(OfferAdmission::default()));
    if let Some(key) = cluster_key.clone() {
        let data_bind = SocketAddr::new(bind.ip(), data_port);
        let data_socket = UdpSocket::bind(data_bind)
            .await
            .with_context(|| format!("binding authenticated tensor data port {data_bind}"))?;
        let admission_state = offer_admission.clone();
        tokio::spawn(async move {
            if let Err(error) =
                tensor_transfer::run_receiver(data_socket, key, admission_state).await
            {
                eprintln!("AITP authenticated tensor receiver stopped: {error:#}");
            }
        });
    }
    let hostname = name.unwrap_or_else(local_hostname);
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut system = System::new_all();
    system.refresh_memory();
    let total_memory_bytes = system.total_memory();
    let available_memory_bytes = system.available_memory();
    let cpu_model = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".into());
    let accelerators = hardware::detect_accelerators();
    let announcement = NodeAnnouncement {
        version: CONTROL_VERSION,
        nonce: String::new(),
        node_id: node_id.clone(),
        hostname: hostname.clone(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus,
        cpu_model: cpu_model.clone(),
        accelerators: accelerators.clone(),
        total_memory_bytes,
        available_memory_bytes,
        inference_port: data_port,
    };

    println!(
        "AITP node '{}' [{}] listening on {}",
        hostname,
        node_id,
        socket.local_addr()?
    );
    println!(
        "Persistent identity stored locally; {} logical CPUs, {:.1} GiB total RAM, {:.1} GiB available RAM.",
        logical_cpus,
        total_memory_bytes as f64 / 1024.0_f64.powi(3),
        available_memory_bytes as f64 / 1024.0_f64.powi(3)
    );
    println!("CPU: {}", cpu_model);
    for gpu in &accelerators {
        println!(
            "Accelerator: {} ({}, {})",
            gpu.name,
            gpu.vendor,
            hardware::memory_label(gpu)
        );
    }
    if accelerators.is_empty() {
        println!("No GPU probe found. Apple Silicon GPU shares unified system memory.");
    }
    if cluster_key.is_some() {
        println!("Authenticated tensor offers and bounded tensor reception are enabled.");
    } else {
        println!("Tensor offers and tensor reception are disabled; set AITP_CLUSTER_KEY on trusted cluster nodes to enable them.");
    }
    println!("No remote command execution is supported.");

    if let Some(manager_addr) = manager {
        let key = cluster_key
            .as_deref()
            .context("joining a managed cluster requires AITP_CLUSTER_KEY")?;
        let auth_tag = auth::sign_node_registration(&announcement, &session_id, key)?;
        let registration = serde_json::to_vec(&ControlMessage::Register {
            node: announcement.clone(),
            session_id: session_id.clone(),
            auth_tag,
        })?;
        if registration.len() > MAX_CONTROL_DATAGRAM {
            bail!(
                "authenticated node registration is {} bytes; control-plane limit is {}",
                registration.len(),
                MAX_CONTROL_DATAGRAM
            );
        }
        socket.send_to(&registration, manager_addr).await?;
        println!("Sent authenticated registration to cluster manager at {}", manager_addr);
    }

    let mut heartbeat = tokio::time::interval(Duration::from_millis(heartbeat_ms.max(100)));
    let mut sequence = 0u64;
    let mut offer_reap = tokio::time::interval(Duration::from_secs(1));
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];

    loop {
        tokio::select! {
            _ = heartbeat.tick(), if manager.is_some() => {
                sequence = sequence.saturating_add(1);
                let available_memory_bytes = {
                    system.refresh_memory();
                    system.available_memory().min(total_memory_bytes)
                };
                let key = cluster_key
                    .as_deref()
                    .context("managed-cluster heartbeat requires AITP_CLUSTER_KEY")?;
                let auth_tag = auth::sign_node_heartbeat(
                    CONTROL_VERSION,
                    &node_id,
                    &session_id,
                    sequence,
                    available_memory_bytes,
                    key,
                )?;
                let message = ControlMessage::Heartbeat {
                    version: CONTROL_VERSION,
                    node_id: node_id.clone(),
                    session_id: session_id.clone(),
                    sequence,
                    available_memory_bytes,
                    auth_tag,
                };
                let encoded = serde_json::to_vec(&message)?;
                if encoded.len() > MAX_CONTROL_DATAGRAM {
                    bail!("authenticated heartbeat exceeds the control-plane datagram limit");
                }
                if let Some(manager_addr) = manager {
                    socket.send_to(&encoded, manager_addr).await?;
                }
            }
            _ = offer_reap.tick() => {
                offer_admission.lock().await.expire(Instant::now());

            }
            received = socket.recv_from(&mut buf) => {
                let (n, peer) = received?;
                if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
                let Ok(message) = serde_json::from_slice::<ControlMessage>(&buf[..n]) else { continue };
                match message {
                    ControlMessage::Discover { version, nonce } if version == CONTROL_VERSION => {
                        let mut response_node = announcement.clone();
                        response_node.nonce = nonce;
                        let response = serde_json::to_vec(&ControlMessage::Announce(response_node))?;
                        if response.len() <= MAX_CONTROL_DATAGRAM {
                            socket.send_to(&response, peer).await?;
                        }
                    }
                    ControlMessage::TensorOffer { offer } => {
                        let Some(key) = cluster_key.as_deref() else {
                            println!("Ignoring tensor offer from {}: AITP_CLUSTER_KEY is not configured.", peer);
                            continue;
                        };
                        let admission = offer_admission.lock().await.admit(&offer, peer, key, Instant::now());
                        let (accepted, reason) = match admission {
                            Ok(()) => (true, None),
                            Err(reason) => (false, Some(reason.to_string())),
                        };
                        let mut reply = TensorOfferReply {
                            version: CONTROL_VERSION,
                            receiver_node_id: node_id.clone(),
                            sender_node_id: offer.sender_node_id.clone(),
                            sequence: offer.sequence,
                            message_id: offer.message_id,
                            tensor_id: offer.descriptor.tensor_id,
                            accepted,
                            reason,
                            auth_tag: Vec::new(),
                        };
                        auth::sign_tensor_offer_reply(&mut reply, key)?;
                        let response = serde_json::to_vec(&ControlMessage::TensorOfferReply { reply })?;
                        if response.len() <= MAX_CONTROL_DATAGRAM {
                            socket.send_to(&response, peer).await?;
                        }
                        if accepted {
                            println!(
                                "Admitted tensor metadata '{}' from {} ({} bytes); awaiting authenticated payload transfer.",
                                offer.descriptor.name, peer, offer.descriptor.byte_len
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[derive(Debug)]
struct ManagedNode {
    announcement: NodeAnnouncement,
    address: SocketAddr,
    last_seen: Instant,
    last_sequence: u64,
    session_id: String,
}

async fn run_manager(
    bind: SocketAddr,
    offline_after: Duration,
    dashboard_bind: SocketAddr,
) -> Result<()> {
    let cluster_key = auth::cluster_key_from_env()?
        .context("cluster manager requires AITP_CLUSTER_KEY (at least 32 bytes)")?;
    let socket = UdpSocket::bind(bind).await?;
    let dashboard_snapshot = dashboard::shared_snapshot();
    let dashboard_state = dashboard_snapshot.clone();
    tokio::spawn(async move {
        if let Err(error) = dashboard::serve(dashboard_bind, dashboard_state).await {
            eprintln!("AITP dashboard stopped: {error:#}");
        }
    });
    println!("AITP cluster manager listening on {}", socket.local_addr()?);
    println!(
        "Nodes are marked OFFLINE after {} ms without a heartbeat.",
        offline_after.as_millis()
    );
    println!("Dashboard is read-only and defaults to localhost; use --dashboard-bind to expose it on a trusted LAN.");
    println!("Manager membership requires HMAC authentication with AITP_CLUSTER_KEY.");

    let mut nodes: HashMap<String, ManagedNode> = HashMap::new();
    let tensor_store_dir = storage::configured_store_dir();
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    let mut refresh = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            _ = refresh.tick() => {
                let retention = offline_after.saturating_mul(10);
                nodes.retain(|_, node| node.last_seen.elapsed() <= retention);
                if !nodes.is_empty() {
                    println!("\nCLUSTER INVENTORY ({} known nodes)", nodes.len());
                    let mut ordered: BTreeMap<String, &ManagedNode> = BTreeMap::new();
                    for (id, node) in &nodes { ordered.insert(id.clone(), node); }
                    for (id, node) in ordered {
                        let age = node.last_seen.elapsed();
                        let status = if age <= offline_after { "ONLINE" } else { "OFFLINE" };
                        println!("- {} [{}] {} {} | {} | {} CPUs | RAM {:.1}/{:.1} GiB available/total | {} | last heartbeat {:.1}s ago",
                            node.announcement.hostname, id, node.announcement.os,
                            node.announcement.arch, node.announcement.cpu_model, node.announcement.logical_cpus,
                            node.announcement.available_memory_bytes as f64 / 1024.0_f64.powi(3),
                            node.announcement.total_memory_bytes as f64 / 1024.0_f64.powi(3),
                            status, age.as_secs_f32());
                        for gpu in &node.announcement.accelerators { println!("    GPU: {} ({}, {})", gpu.name, gpu.vendor, hardware::memory_label(gpu)); }
                    }
                }
                let mut snapshot_nodes = Vec::with_capacity(nodes.len());
                let mut online_nodes = 0usize;
                for (id, node) in &nodes {
                    let age = node.last_seen.elapsed();
                    let online = age <= offline_after;
                    if online { online_nodes += 1; }
                    snapshot_nodes.push(dashboard::DashboardNode {
                        node_id: id.clone(),
                        hostname: node.announcement.hostname.clone(),
                        address: node.address.to_string(),
                        os: node.announcement.os.clone(),
                        arch: node.announcement.arch.clone(),
                        cpu_model: node.announcement.cpu_model.clone(),
                        logical_cpus: node.announcement.logical_cpus,
                        total_memory_bytes: node.announcement.total_memory_bytes,
                        available_memory_bytes: node.announcement.available_memory_bytes,
                        accelerators: node.announcement.accelerators.clone(),
                        online,
                        last_seen_secs: age.as_secs(),
                    });
                }
                snapshot_nodes.sort_by(|a, b| a.hostname.cmp(&b.hostname));
                let tensor_store = match storage::scan_store(&tensor_store_dir).await {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        eprintln!("Could not scan local tensor store {}: {error:#}", tensor_store_dir.display());
                        storage::TensorStoreSnapshot::default()
                    }
                };
                {
                    let mut snapshot = dashboard_snapshot.write().await;
                    snapshot.nodes = snapshot_nodes;
                    snapshot.online_nodes = online_nodes;
                    snapshot.offline_nodes = nodes.len().saturating_sub(online_nodes);
                    snapshot.tensor_store = tensor_store;
                    snapshot.updated_at_unix = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                }
            }
            received = socket.recv_from(&mut buf) => {
                let (n, peer) = received?;
                if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
                let Ok(message) = serde_json::from_slice::<ControlMessage>(&buf[..n]) else { continue };
                match message {
                    ControlMessage::Register { node, session_id, auth_tag } => {
                        if auth::verify_node_registration(&node, &session_id, &auth_tag, &cluster_key)?
                            && (nodes.contains_key(&node.node_id) || nodes.len() < MAX_CLUSTER_NODES)
                        {
                            let id = node.node_id.clone();
                            let is_new = !nodes.contains_key(&id);
                            nodes.insert(id.clone(), ManagedNode {
                                announcement: node,
                                address: peer,
                                last_seen: Instant::now(),
                                last_sequence: 0,
                                session_id,
                            });
                            println!(
                                "{} authenticated node registration: {} from {}",
                                if is_new { "New" } else { "Refreshed" },
                                id,
                                peer
                            );
                        }
                    }
                    ControlMessage::Heartbeat {
                        version,
                        node_id,
                        session_id,
                        sequence,
                        available_memory_bytes,
                        auth_tag,
                    } => {
                        if auth::verify_node_heartbeat(
                            version,
                            &node_id,
                            &session_id,
                            sequence,
                            available_memory_bytes,
                            &auth_tag,
                            &cluster_key,
                        )? {
                            if let Some(node) = nodes.get_mut(&node_id) {
                                if node.session_id == session_id && sequence > node.last_sequence {
                                    node.last_sequence = sequence;
                                    node.last_seen = Instant::now();
                                    node.address = peer;
                                    node.announcement.available_memory_bytes = available_memory_bytes
                                        .min(node.announcement.total_memory_bytes);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

async fn discover(target: SocketAddr, wait: Duration) -> Result<()> {
    if !target.is_ipv4() {
        bail!("this discovery implementation currently requires an IPv4 broadcast target");
    }
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.set_broadcast(true)?;
    let nonce = uuid::Uuid::new_v4().to_string();
    let query = serde_json::to_vec(&ControlMessage::Discover {
        version: CONTROL_VERSION,
        nonce: nonce.clone(),
    })?;
    socket.send_to(&query, target).await?;
    println!(
        "Searching for AITP nodes via {} for {} ms...",
        target,
        wait.as_millis()
    );

    let deadline = tokio::time::Instant::now() + wait;
    let mut nodes = BTreeMap::<String, NodeAnnouncement>::new();
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline - tokio::time::Instant::now();
        match timeout(remaining, socket.recv_from(&mut buf)).await {
            Ok(Ok((n, _peer))) => {
                if n == 0 || n > MAX_CONTROL_DATAGRAM {
                    continue;
                }
                let Ok(ControlMessage::Announce(node)) = serde_json::from_slice(&buf[..n]) else {
                    continue;
                };
                if node.version != CONTROL_VERSION || node.nonce != nonce {
                    continue;
                }
                nodes.entry(node.node_id.clone()).or_insert(node);
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => break,
        }
    }

    if nodes.is_empty() {
        println!(
            "No AITP nodes found. Check that node agents are running and UDP broadcast is allowed."
        );
    } else {
        println!("Found {} node(s):", nodes.len());
        for node in nodes.values() {
            println!("- {} [{}] {} / {} | {} | {} CPUs | {:.1}/{:.1} GiB available/total RAM | inference port {}",
                node.hostname, node.node_id, node.os, node.arch, node.cpu_model, node.logical_cpus,
                node.available_memory_bytes as f64 / 1024.0_f64.powi(3),
                node.total_memory_bytes as f64 / 1024.0_f64.powi(3), node.inference_port);
            for gpu in &node.accelerators {
                println!(
                    "    GPU: {} ({}, {})",
                    gpu.name,
                    gpu.vendor,
                    hardware::memory_label(gpu)
                );
            }
        }
    }
    Ok(())
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "aitp-node".to_string())
}
