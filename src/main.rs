mod control;
mod protocol;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use control::{ControlMessage, NodeAnnouncement, CONTROL_VERSION, MAX_CONTROL_DATAGRAM};
use protocol::*;
use std::{collections::HashSet, net::SocketAddr, path::PathBuf, time::Duration};
use tokio::{fs, net::UdpSocket, time::timeout};

const CHUNK_SIZE: usize = MAX_PAYLOAD;
const WINDOW_PACKETS: usize = 512;
const RETRY_LIMIT: usize = 20;
const ACK_WAIT: Duration = Duration::from_millis(250);

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
    /// Run a LAN discovery agent. This advertises metadata only; it cannot execute workloads.
    Node { #[arg(long, default_value = "0.0.0.0:9001")] bind: SocketAddr,
           #[arg(long, default_value_t = 9000)] data_port: u16,
           #[arg(long)] name: Option<String> },
    /// Discover AITP nodes using IPv4 UDP broadcast.
    Discover { #[arg(long, default_value = "255.255.255.255:9001")] target: SocketAddr,
               #[arg(long, default_value_t = 1500)] wait_ms: u64 },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Receive { bind, output } => receive(bind, output).await,
        Command::Send { target, input, tensor_id } => send(target, input, tensor_id).await,
        Command::Node { bind, data_port, name } => run_node(bind, data_port, name).await,
        Command::Discover { target, wait_ms } => discover(target, Duration::from_millis(wait_ms)).await,
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

async fn run_node(bind: SocketAddr, data_port: u16, name: Option<String>) -> Result<()> {
    let socket = UdpSocket::bind(bind).await?;
    let node_id = uuid::Uuid::new_v4().to_string();
    let hostname = name.unwrap_or_else(local_hostname);
    let logical_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("AITP node agent '{}' ({}) listening on {}", hostname, node_id, socket.local_addr()?);
    println!("Discovery only: no remote command execution is supported.");

    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    loop {
        let (n, peer) = socket.recv_from(&mut buf).await?;
        if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
        let Ok(ControlMessage::Discover { version, nonce }) = serde_json::from_slice(&buf[..n]) else { continue };
        if version != CONTROL_VERSION { continue; }

        let announcement = NodeAnnouncement {
            version: CONTROL_VERSION,
            nonce,
            node_id: node_id.clone(),
            hostname: hostname.clone(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cpus,
            inference_port: data_port,
        };
        let response = serde_json::to_vec(&ControlMessage::Announce(announcement))?;
        if response.len() <= MAX_CONTROL_DATAGRAM {
            socket.send_to(&response, peer).await?;
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
    let mut nodes = std::collections::BTreeMap::<String, NodeAnnouncement>::new();
    let mut buf = [0u8; MAX_CONTROL_DATAGRAM];
    while tokio::time::Instant::now() < deadline {
        let remaining = deadline - tokio::time::Instant::now();
        match timeout(remaining, socket.recv_from(&mut buf)).await {
            Ok(Ok((n, peer))) => {
                if n == 0 || n > MAX_CONTROL_DATAGRAM { continue; }
                let Ok(ControlMessage::Announce(node)) = serde_json::from_slice(&buf[..n]) else { continue };
                if node.version != CONTROL_VERSION || node.nonce != nonce { continue; }
                nodes.entry(node.node_id.clone()).or_insert(node);
                let _ = peer;
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
            println!("- {} [{}] {} / {} | {} logical CPUs | inference port {}",
                node.hostname, node.node_id, node.os, node.arch, node.logical_cpus, node.inference_port);
        }
    }
    Ok(())
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME").or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "aitp-node".to_string())
}
