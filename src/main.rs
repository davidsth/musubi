mod auth;
mod control;
mod dashboard;
mod hardware;
mod protocol;
mod tensor;
mod worker;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use control::{ControlMessage, NodeAnnouncement, TensorOfferMessage, TensorOfferReply, CONTROL_VERSION, MAX_CONTROL_DATAGRAM};
use protocol::*;
use tensor::TensorDescriptor;
use worker::OfferAdmission;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};
use sysinfo::System;
use tokio::{fs, net::UdpSocket, time::timeout};

const CHUNK_SIZE: usize = MAX_PAYLOAD;
const WINDOW_PACKETS: usize = 512;
const RETRY_LIMIT: usize = 20;
const ACK_WAIT: Duration = Duration::from_millis(250);
const MAX_CLUSTER_NODES: usize = 256;

#[derive(Parser)]
#[command(name = "aitp", version, about = "Experimental distributed-inference transport")]
struct Cli { #[command(subcommand)] command: Command }

#[derive(Subcommand)]
enum Command {
    /// Listen for incoming AITP transfers and write received payloads to a file.
    Receive { #[arg(long, default_value = "0.0.0.0:9000")] bind: SocketAddr,
              #[arg(long, default_value = "received.tensor")] output: PathBuf },
    /// Send a file as tensor chunks to an AITP receiver.
    Send { #[arg(long)] target: SocketAddr, #[arg(long)] input: PathBuf,
           #[arg(long, default_value_t = 1)] tensor_id: u32 },
    /// Submit authenticated tensor metadata to a worker. No tensor bytes are sent.
    Offer { #[arg(long)] target: SocketAddr, #[arg(long)] descriptor: PathBuf },
    /// Run a LAN node agent with a persistent identity.
    Node { #[arg(long, default_value = "0.0.0.0:9001")] bind: SocketAddr,
           #[arg(long, default_value_t = 9000)] data_port: u16,
           #[arg(long)] name: Option<String>,
           /// Optional manager control endpoint, for example 192.168.1.10:9002.
           #[arg(long)] manager: Option<SocketAddr>,
           #[arg(long, default_value_t = 1000)] heartbeat_ms: u64 },
    /// Discover AITP nodes using IPv4 UDP broadcast.
    Discover { #[arg(long, default_value = "255.255.255.255:9001")] target: SocketAddr,
               #[arg(long, default_value_t = 1500)] wait_ms: u64 },
    /// Run a cluster manager that tracks registrations and node liveness.
    Manager { #[arg(long, default_value = "0.0.0.0:9002")] bind: SocketAddr,
              #[arg(long, default_value_t = 5000)] timeout_ms: u64,
              /// HTTP dashboard bind address. Defaults to localhost for safety.
              #[arg(long, default_value = "127.0.0.1:9003")] dashboard_bind: SocketAddr },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Receive { bind, output } => receive(bind, output).await,
        Command::Send { target, input, tensor_id } => send(target, input, tensor_id).await,
        Command::Offer { target, descriptor } => offer_tensor(target, descriptor).await,
        Command::Node { bind, data_port, name, manager, heartbeat_ms } =>
            run_node(bind, data_port, name, manager, heartbeat_ms).await,
        Command::Discover { target, wait_ms } =>
            discover(target, Duration::from_millis(wait_ms)).await,
        Command::Manager { bind, timeout_ms, dashboard_bind } =>
            run_manager(bind, Duration::from_millis(timeout_ms.max(500)), dashboard_bind).await,
    }
}

fn make_header(kind: u8, message_id: u32, tensor_id: u32, sequence: u32, payload: &[u8]) -> Header {
    Header { kind, flags: 0, message_id, tensor_id, sequence,
        payload_len: payload.len() as u16, checksum: checksum(payload), worker_id: 0 }
}

async fn send(target: SocketAddr, input: PathBuf, tensor_id: u32) -> Result<()> {
    let data = fs::read(&input).await.with_context(|| format!("reading {}", input.display()))?;
    if data.is_empty() { bail!("input file is empty"); }
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.connect(target).await?;
    let message_id = (std::process::id() ^ (data.len() as u32)) | 1;
    let chunks: Vec<&[u8]> = data.chunks(CHUNK_SIZE).collect();
    println!("AITP: sending {} bytes in {} chunks to {}", data.len(), chunks.len(), target);

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
                match timeout(deadline - tokio::time::Instant::now(), socket.recv(&mut buf)).await {
                    Ok(Ok(n)) => {
                        if let Ok(h) = Header::decode(&buf[..n]) {
                            if h.kind == KIND_ACK && h.message_id == message_id && h.tensor_id == tensor_id {
                                let seq = h.sequence as usize;
                                if seq >= first_seq && seq < first_seq + batch.len() {
                                    let idx = seq - first_seq;
                                    if !acked[idx] { acked[idx] = true; acked_total += 1; }
                                }
                            }
                        }
                    }
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => break,
                }
            }
            attempts += 1;
            if attempts > RETRY_LIMIT { bail!("retry limit exceeded; {acked_total}/{} chunks acknowledged", chunks.len()); }
        }
        println!("progress: {}/{} chunks", acked_total, chunks.len());
    }

    let fin_payload = [(chunks.len() as u32).to_be_bytes().as_slice(), (data.len() as u64).to_be_bytes().as_slice()].concat();
    let fin = packet(make_header(KIND_FIN, message_id, tensor_id, chunks.len() as u32, &fin_payload), &fin_payload);
    let mut buf = [0u8; 2048];
    for _ in 0..RETRY_LIMIT {
        socket.send(&fin).await?;
        if let Ok(Ok(n)) = timeout(Duration::from_secs(1), socket.recv(&mut buf)).await {
            if let Ok(h) = Header::decode(&buf[..n]) {
                if h.kind == KIND_FIN_ACK && h.message_id == message_id { println!("transfer complete"); return Ok(()); }
            }
        }
    }
    bail!("receiver did not confirm transfer completion")
}


async fn offer_tensor(target: SocketAddr, descriptor_path: PathBuf) -> Result<()> {
    let key = auth::cluster_key_from_env()?
        .context("set AITP_CLUSTER_KEY to the same 32+ byte secret used by the worker")?;
    let descriptor_bytes = fs::read(&descriptor_path)
        .await
        .with_context(|| format!("reading {}", descriptor_path.display()))?;
    let descriptor: TensorDescriptor = serde_json::from_slice(&descriptor_bytes)
        .context("parsing tensor descriptor JSON")?;
    descriptor.validate()?;

    let sender_node_id = persistent_node_id()?;
    let sequence = next_offer_sequence()?;
    let random_id = uuid::Uuid::new_v4();
    let message_id = u32::from_be_bytes(random_id.as_bytes()[0..4].try_into()?) | 1;
    let mut offer = TensorOfferMessage {
        version: CONTROL_VERSION,
        sender_node_id: sender_node_id.clone(),
        sequence,
        message_id,
        descriptor,
        auth_tag: Vec::new(),
    };
    auth::sign_tensor_offer(&mut offer, &key)?;

    let encoded = serde_json::to_vec(&ControlMessage::TensorOffer { offer: offer.clone() })?;
    if encoded.len() > MAX_CONTROL_DATAGRAM {
        bail!("tensor offer is {} bytes; control-plane limit is {}", encoded.len(), MAX_CONTROL_DATAGRAM);
    }

    let bind = if target.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" };
    let socket = UdpSocket::bind(bind).await?;
    socket.connect(target).await?;
    socket.send(&encoded).await?;

    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    let n = timeout(Duration::from_secs(3), socket.recv(&mut buf))
        .await
        .context("timed out waiting for tensor-offer admission reply")??;
    let response: ControlMessage = serde_json::from_slice(&buf[..n])?;
    let ControlMessage::TensorOfferReply { reply } = response else {
        bail!("worker returned an unexpected control message");
    };
    if !auth::verify_tensor_offer_reply(&reply, &key)? {
        bail!("worker reply authentication failed; verify that both nodes share the same AITP_CLUSTER_KEY");
    }
    if reply.sender_node_id != sender_node_id
        || reply.sequence != sequence
        || reply.message_id != message_id
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

    println!(
        "Worker {} admitted metadata for tensor '{}' ({} bytes).",
        reply.receiver_node_id,
        offer.descriptor.name,
        offer.descriptor.byte_len
    );
    println!("No tensor bytes were sent; authenticated data-plane transfer is the next milestone.");
    Ok(())
}

async fn receive(bind: SocketAddr, output: PathBuf) -> Result<()> {
    let socket = UdpSocket::bind(bind).await?;
    println!("AITP receiver listening on {}", socket.local_addr()?);
    let mut buf = [0u8; 2048];
    let mut active: Option<(u32, u32, SocketAddr, tokio::fs::File, HashSet<u32>)> = None;

    loop {
        let (n, peer) = socket.recv_from(&mut buf).await?;
        let h = match Header::decode(&buf[..n]) { Ok(h) => h, Err(_) => continue };
        if HEADER_LEN + h.payload_len as usize != n { continue; }
        let payload = &buf[HEADER_LEN..n];
        if checksum(payload) != h.checksum { continue; }

        if h.kind == KIND_DATA {
            let needs_new = active.as_ref().map(|a| a.0 != h.message_id || a.1 != h.tensor_id || a.2 != peer).unwrap_or(true);
            if needs_new {
                let file = fs::File::create(&output).await?;
                active = Some((h.message_id, h.tensor_id, peer, file, HashSet::new()));
                println!("receiving message {} tensor {} from {}", h.message_id, h.tensor_id, peer);
            }
            if let Some((msg, tensor, addr, file, received)) = active.as_mut() {
                if *msg == h.message_id && *tensor == h.tensor_id && *addr == peer {
                    use tokio::io::{AsyncSeekExt, AsyncWriteExt};
                    if !received.contains(&h.sequence) {
                        file.seek(std::io::SeekFrom::Start(h.sequence as u64 * CHUNK_SIZE as u64)).await?;
                        file.write_all(payload).await?;
                        received.insert(h.sequence);
                    }
                    let ack = packet(make_header(KIND_ACK, h.message_id, h.tensor_id, h.sequence, &[]), &[]);
                    socket.send_to(&ack, peer).await?;
                }
            }
        } else if h.kind == KIND_FIN {
            if payload.len() != 12 { continue; }
            let total_chunks = u32::from_be_bytes(payload[0..4].try_into()?);
            let total_bytes = u64::from_be_bytes(payload[4..12].try_into()?);
            if let Some((msg, tensor, addr, file, received)) = active.as_mut() {
                if *msg == h.message_id && *tensor == h.tensor_id && *addr == peer && received.len() == total_chunks as usize {
                    file.set_len(total_bytes).await?;
                    file.sync_all().await?;
                    let ack = packet(make_header(KIND_FIN_ACK, h.message_id, h.tensor_id, total_chunks, &[]), &[]);
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
    }.unwrap_or(std::env::current_dir()?);
    let dir = if cfg!(windows) { base.join("aitp") } else { base.join(".aitp") };
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
        Ok(value) => value.trim().parse::<u64>()
            .context("parsing persisted tensor-offer sequence")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    let next = previous.checked_add(1).context("tensor-offer sequence exhausted")?;
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
    let socket = UdpSocket::bind(bind).await?;
    let node_id = persistent_node_id()?;
    let hostname = name.unwrap_or_else(local_hostname);
    let logical_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut system = System::new_all();
    system.refresh_memory();
    let total_memory_bytes = system.total_memory();
    let available_memory_bytes = system.available_memory();
    let cpu_model = system.cpus().first().map(|cpu| cpu.brand().trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| "Unknown CPU".into());
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

    println!("AITP node '{}' [{}] listening on {}", hostname, node_id, socket.local_addr()?);
    println!(
        "Persistent identity stored locally; {} logical CPUs, {:.1} GiB total RAM, {:.1} GiB available RAM.",
        logical_cpus,
        total_memory_bytes as f64 / 1024.0_f64.powi(3),
        available_memory_bytes as f64 / 1024.0_f64.powi(3)
    );
    println!("CPU: {}", cpu_model);
    for gpu in &accelerators { println!("Accelerator: {} ({}, {})", gpu.name, gpu.vendor, hardware::memory_label(gpu)); }
    if accelerators.is_empty() { println!("No GPU probe found. Apple Silicon GPU shares unified system memory."); }
    if cluster_key.is_some() {
        println!("Authenticated tensor-offer admission is enabled (metadata only).");
    } else {
        println!("Tensor offers are disabled; set AITP_CLUSTER_KEY on trusted cluster nodes to enable them.");
    }
    println!("No remote command execution is supported.");

    if let Some(manager_addr) = manager {
        let registration = serde_json::to_vec(&ControlMessage::Register { node: announcement.clone() })?;
        socket.send_to(&registration, manager_addr).await?;
        println!("Registered with cluster manager at {}", manager_addr);
    }

    let mut heartbeat = tokio::time::interval(Duration::from_millis(heartbeat_ms.max(100)));
    let mut sequence = 0u64;
    let mut offer_admission = OfferAdmission::default();
    let mut offer_reap = tokio::time::interval(Duration::from_secs(1));
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];

    loop {
        tokio::select! {
            _ = heartbeat.tick(), if manager.is_some() => {
                sequence = sequence.saturating_add(1);
                let message = ControlMessage::Heartbeat {
                    version: CONTROL_VERSION,
                    node_id: node_id.clone(),
                    sequence,
                    available_memory_bytes: {
                        system.refresh_memory();
                        system.available_memory()
                    },
                };
                let encoded = serde_json::to_vec(&message)?;
                if let Some(manager_addr) = manager {
                    socket.send_to(&encoded, manager_addr).await?;
                }
            }
            _ = offer_reap.tick() => {
                offer_admission.expire(Instant::now());
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
                *dashboard_snapshot.write().await = dashboard::ClusterSnapshot {
                    nodes: snapshot_nodes,
                    online_nodes,
                    offline_nodes: nodes.len().saturating_sub(online_nodes),
                    updated_at_unix: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                };
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
                        let admission = offer_admission.admit(&offer, peer, key, Instant::now());
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
                                "Admitted tensor metadata '{}' from {} ({} bytes); no payload allocated.",
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
}

async fn run_manager(bind: SocketAddr, offline_after: Duration, dashboard_bind: SocketAddr) -> Result<()> {
    let socket = UdpSocket::bind(bind).await?;
    let dashboard_snapshot = dashboard::shared_snapshot();
    let dashboard_state = dashboard_snapshot.clone();
    tokio::spawn(async move {
        if let Err(error) = dashboard::serve(dashboard_bind, dashboard_state).await {
            eprintln!("AITP dashboard stopped: {error:#}");
        }
    });
    println!("AITP cluster manager listening on {}", socket.local_addr()?);
    println!("Nodes are marked OFFLINE after {} ms without a heartbeat.", offline_after.as_millis());
    println!("Dashboard is read-only and defaults to localhost; use --dashboard-bind to expose it on a trusted LAN.");

    let mut nodes: HashMap<String, ManagedNode> = HashMap::new();
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
            }
            received = socket.recv_from(&mut buf) => {
                let (n, peer) = received?;
                if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
                let Ok(message) = serde_json::from_slice::<ControlMessage>(&buf[..n]) else { continue };
                match message {
                    ControlMessage::Register { node } if node.is_valid() && node.nonce.is_empty()
                        && (nodes.contains_key(&node.node_id) || nodes.len() < MAX_CLUSTER_NODES) => {
                        let id = node.node_id.clone();
                        let is_new = !nodes.contains_key(&id);
                        nodes.insert(id.clone(), ManagedNode {
                            announcement: node,
                            address: peer,
                            last_seen: Instant::now(),
                            last_sequence: 0,
                        });
                        println!("{} node registration: {} from {}", if is_new { "New" } else { "Refreshed" }, id, peer);
                    }
                    ControlMessage::Heartbeat { version, node_id, sequence, available_memory_bytes } if version == CONTROL_VERSION => {
                        if let Some(node) = nodes.get_mut(&node_id) {
                            if sequence > node.last_sequence {
                                node.last_sequence = sequence;
                                node.last_seen = Instant::now();
                                node.address = peer;
                                node.announcement.available_memory_bytes = available_memory_bytes
                                    .min(node.announcement.total_memory_bytes);
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
    if !target.is_ipv4() { bail!("this discovery implementation currently requires an IPv4 broadcast target"); }
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.set_broadcast(true)?;
    let nonce = uuid::Uuid::new_v4().to_string();
    let query = serde_json::to_vec(&ControlMessage::Discover { version: CONTROL_VERSION, nonce: nonce.clone() })?;
    socket.send_to(&query, target).await?;
    println!("Searching for AITP nodes via {} for {} ms...", target, wait.as_millis());

    let deadline = tokio::time::Instant::now() + wait;
    let mut nodes = BTreeMap::<String, NodeAnnouncement>::new();
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline - tokio::time::Instant::now();
        match timeout(remaining, socket.recv_from(&mut buf)).await {
            Ok(Ok((n, _peer))) => {
                if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
                let Ok(ControlMessage::Announce(node)) = serde_json::from_slice(&buf[..n]) else { continue };
                if node.version != CONTROL_VERSION || node.nonce != nonce { continue; }
                nodes.entry(node.node_id.clone()).or_insert(node);
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => break,
        }
    }

    if nodes.is_empty() {
        println!("No AITP nodes found. Check that node agents are running and UDP broadcast is allowed.");
    } else {
        println!("Found {} node(s):", nodes.len());
        for node in nodes.values() {
            println!("- {} [{}] {} / {} | {} | {} CPUs | {:.1}/{:.1} GiB available/total RAM | inference port {}",
                node.hostname, node.node_id, node.os, node.arch, node.cpu_model, node.logical_cpus,
                node.available_memory_bytes as f64 / 1024.0_f64.powi(3),
                node.total_memory_bytes as f64 / 1024.0_f64.powi(3), node.inference_port);
            for gpu in &node.accelerators { println!("    GPU: {} ({}, {})", gpu.name, gpu.vendor, hardware::memory_label(gpu)); }
        }
    }
    Ok(())
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME").or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "aitp-node".to_string())
}
