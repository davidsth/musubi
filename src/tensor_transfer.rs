//! Authenticated, bounded tensor payload transfer over the existing binary data plane.
//!
//! The 32-byte v0.1 header is unchanged. A 32-byte HMAC tag is appended after
//! each payload, so authenticated datagrams remain at or below 1200 bytes.
//! This fixed-window UDP prototype is LAN-only and is not congestion control.
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as FmtWrite,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    net::UdpSocket,
    sync::Mutex,
    time::timeout,
};

use crate::{
    auth,
    control::TensorOfferMessage,
    storage,
    protocol::{
        self, Header, AUTH_TAG_LEN, KIND_ACK, KIND_DATA, KIND_FIN, KIND_FIN_ACK,
        MAX_AUTH_PAYLOAD, MAX_DATAGRAM, HEADER_LEN,
    },
    worker::{OfferAdmission, PendingTensorOffer},
};

const SEND_WINDOW: usize = 32;
const SEND_PACING: Duration = Duration::from_micros(200);
const ACK_WAIT: Duration = Duration::from_millis(300);
const RETRY_LIMIT: usize = 20;
const FIN_ACK_WAIT: Duration = Duration::from_secs(1);
const TRANSFER_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const COMPLETED_RETRY_TTL: Duration = Duration::from_secs(60);
/// Bound duplicate-FIN response state even under a high transfer completion rate.
const MAX_COMPLETED_TRANSFERS: usize = 256;
const MAX_WORKER_TENSOR_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
struct ActiveTransfer {
    offer: PendingTensorOffer,
    file: fs::File,
    partial_path: PathBuf,
    final_path: PathBuf,
    received: Vec<bool>,
    total_chunks: u32,
    last_activity: Instant,
}

#[derive(Debug)]
struct CompletedTransfer {
    sender_node_id: String,
    offer_sequence: u64,
    message_id: u32,
    tensor_id: u32,
    peer: SocketAddr,
    total_chunks: u32,
    byte_len: u64,
    expires_at: Instant,
}

/// Bind this socket before spawning the task so port errors are reported to the node CLI.
pub async fn run_receiver(
    socket: UdpSocket,
    key: Vec<u8>,
    admission: Arc<Mutex<OfferAdmission>>,
) -> Result<()> {
    let output_dir = storage::configured_store_dir();
    run_receiver_in(socket, key, admission, output_dir).await
}

/// Receiver implementation with an explicit output directory for isolated tests
/// and embedders. Production callers use the AITP_TENSOR_DIR environment setting.
async fn run_receiver_in(
    socket: UdpSocket,
    key: Vec<u8>,
    admission: Arc<Mutex<OfferAdmission>>,
    output_dir: PathBuf,
) -> Result<()> {
    fs::create_dir_all(&output_dir)
        .await
        .with_context(|| format!("creating tensor output directory {}", output_dir.display()))?;
    println!(
        "Authenticated tensor receiver listening on {} (output: {})",
        socket.local_addr()?,
        output_dir.display()
    );

    let mut active: Option<ActiveTransfer> = None;
    let mut completed: Vec<CompletedTransfer> = Vec::new();
    let mut buffer = [0u8; MAX_DATAGRAM];
    let mut maintenance = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            _ = maintenance.tick() => {
                let now = Instant::now();
                completed.retain(|entry| entry.expires_at > now);
                admission.lock().await.expire(now);

                let expired = active.as_ref()
                    .is_some_and(|transfer| now.saturating_duration_since(transfer.last_activity) > TRANSFER_IDLE_TIMEOUT);
                if expired {
                    if let Some(transfer) = active.take() {
                        let sender = transfer.offer.offer.sender_node_id.clone();
                        let message_id = transfer.offer.offer.message_id;
                        drop(transfer.file);
                        let _ = fs::remove_file(&transfer.partial_path).await;
                        admission.lock().await.remove_pending(&sender, message_id);
                        eprintln!("Expired inactive tensor transfer {} from {}", message_id, transfer.offer.peer);
                    }
                }
            }
            received = socket.recv_from(&mut buffer) => {
                let (n, peer) = received?;
                if n < HEADER_LEN + AUTH_TAG_LEN || n > MAX_DATAGRAM { continue; }
                let Some((header, payload, tag)) = decode_authenticated_datagram(&buffer[..n]) else { continue };
                if header.flags != 0 || header.worker_id != 0 || header.message_id == 0 { continue; }
                if protocol::checksum(payload) != header.checksum { continue; }

                if header.kind == KIND_DATA {
                    if payload.is_empty() { continue; }

                    let pending = if let Some(current) = active.as_ref() {
                        if same_transfer(current, peer, &header) {
                            Some(current.offer.clone())
                        } else {
                            None
                        }
                    } else {
                        admission.lock().await.pending_for_peer(peer, header.message_id, header.tensor_id)
                    };
                    let Some(pending) = pending else { continue };
                    let offer = &pending.offer;
                    if offer.descriptor.byte_len == 0
                        || offer.descriptor.byte_len > MAX_WORKER_TENSOR_BYTES
                        || !auth::verify_tensor_packet(
                            &key, &offer.sender_node_id, offer.sequence, &header, payload, tag
                        )? {
                        continue;
                    }

                    let total_chunks = chunk_count(offer.descriptor.byte_len)?;
                    if header.sequence >= total_chunks { continue; }
                    let expected_len = expected_chunk_len(offer.descriptor.byte_len, header.sequence)?;
                    if payload.len() != expected_len { continue; }

                    if active.is_none() {
                        let (partial_path, final_path) = tensor_paths(
                            &output_dir,
                            &offer.sender_node_id,
                            offer.sequence,
                            header.message_id,
                            header.tensor_id,
                        );
                        let file = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&partial_path)
                            .await
                            .with_context(|| format!("creating bounded tensor staging file {}", partial_path.display()))?;
                        active = Some(ActiveTransfer {
                            offer: pending,
                            file,
                            partial_path,
                            final_path,
                            received: vec![false; total_chunks as usize],
                            total_chunks,
                            last_activity: Instant::now(),
                        });
                    }

                    let Some(current) = active.as_mut() else { continue };
                    if !same_transfer(current, peer, &header) { continue; }
                    let index = header.sequence as usize;
                    if !current.received[index] {
                        let offset = u64::from(header.sequence) * MAX_AUTH_PAYLOAD as u64;
                        current.file.seek(std::io::SeekFrom::Start(offset)).await?;
                        current.file.write_all(payload).await?;
                        current.received[index] = true;
                    }
                    current.last_activity = Instant::now();
                    let ack = authenticated_control_packet(
                        &key, &current.offer.offer.sender_node_id,
                        current.offer.offer.sequence, KIND_ACK,
                        header.message_id, header.tensor_id, header.sequence,
                    )?;
                    socket.send_to(&ack, peer).await?;
                } else if header.kind == KIND_FIN {
                    // A duplicate FIN after a lost FIN_ACK is answered idempotently.
                    if let Some(done) = completed.iter().find(|done| {
                        done.peer == peer
                            && done.message_id == header.message_id
                            && done.tensor_id == header.tensor_id
                    }) {
                        if verify_fin(done, &key, &header, payload, tag)? {
                            let reply = authenticated_control_packet(
                                &key, &done.sender_node_id, done.offer_sequence, KIND_FIN_ACK,
                                done.message_id, done.tensor_id, done.total_chunks,
                            )?;
                            socket.send_to(&reply, peer).await?;
                        }
                        continue;
                    }

                    let Some(current) = active.as_ref() else { continue };
                    if !same_transfer(current, peer, &header) { continue; }
                    let offer = &current.offer.offer;
                    if !auth::verify_tensor_packet(
                        &key, &offer.sender_node_id, offer.sequence, &header, payload, tag
                    )? {
                        continue;
                    }
                    if payload.len() != 12 || header.sequence != current.total_chunks { continue; }
                    let declared_chunks = u32::from_be_bytes(payload[0..4].try_into()?);
                    let declared_bytes = u64::from_be_bytes(payload[4..12].try_into()?);
                    if declared_chunks != current.total_chunks
                        || declared_bytes != offer.descriptor.byte_len
                        || !current.received.iter().all(|received| *received) {
                        continue;
                    }

                    let mut finished = active.take().expect("active transfer checked above");
                    finished.file.set_len(declared_bytes).await?;
                    finished.file.sync_all().await?;
                    drop(finished.file);
                    let manifest = match storage::commit_tensor_file(
                        &finished.partial_path,
                        &finished.final_path,
                        &finished.offer.offer,
                    ).await {
                        Ok(manifest) => manifest,
                        Err(error) => {
                            let _ = fs::remove_file(&finished.partial_path).await;
                            admission.lock().await.remove_pending(
                                &finished.offer.offer.sender_node_id,
                                finished.offer.offer.message_id,
                            );
                            return Err(error).context("publishing committed tensor manifest");
                        }
                    };

                    let done = CompletedTransfer {
                        sender_node_id: finished.offer.offer.sender_node_id.clone(),
                        offer_sequence: finished.offer.offer.sequence,
                        message_id: finished.offer.offer.message_id,
                        tensor_id: finished.offer.offer.descriptor.tensor_id,
                        peer,
                        total_chunks: finished.total_chunks,
                        byte_len: finished.offer.offer.descriptor.byte_len,
                        expires_at: Instant::now() + COMPLETED_RETRY_TTL,
                    };
                    let reply = authenticated_control_packet(
                        &key, &done.sender_node_id, done.offer_sequence, KIND_FIN_ACK,
                        done.message_id, done.tensor_id, done.total_chunks,
                    )?;
                    socket.send_to(&reply, peer).await?;
                    admission.lock().await.remove_pending(&done.sender_node_id, done.message_id);
                    println!(
                        "Committed tensor '{}' ({} bytes, {} chunks) to {}",
                        finished.offer.offer.descriptor.name,
                        done.byte_len,
                        done.total_chunks,
                        finished.final_path.display()
                    );
                    println!("Tensor manifest published at {}", storage::manifest_path_for_payload(&finished.final_path).display());
                    debug_assert_eq!(manifest.byte_len, done.byte_len);
                    remember_completed(&mut completed, done);
                }
            }
        }
    }
}

/// Send the file bytes for an already-admitted tensor offer using the same UDP socket
/// that submitted the offer. This keeps the observed source endpoint bound to admission.
pub async fn send_tensor_data(
    socket: &UdpSocket,
    data_target: SocketAddr,
    input: &Path,
    offer: &TensorOfferMessage,
    key: &[u8],
) -> Result<()> {
    offer.descriptor.validate()?;
    let byte_len = offer.descriptor.byte_len;
    if byte_len > MAX_WORKER_TENSOR_BYTES {
        bail!("tensor exceeds the current 1 GiB transfer ceiling");
    }
    let mut file = fs::File::open(input)
        .await
        .with_context(|| format!("opening tensor file {}", input.display()))?;
    let actual_len = file.metadata().await?.len();
    if actual_len != byte_len {
        bail!("input file is {actual_len} bytes but descriptor expects {byte_len}");
    }

    let total_chunks = chunk_count(byte_len)?;
    let mut acknowledged_total = 0u32;
    println!(
        "Sending authenticated tensor '{}' ({} bytes, {} chunks) to {}",
        offer.descriptor.name, byte_len, total_chunks, data_target
    );

    let mut first_sequence = 0u32;
    while first_sequence < total_chunks {
        let batch_end = total_chunks.min(first_sequence.saturating_add(SEND_WINDOW as u32));
        let mut packets = Vec::with_capacity((batch_end - first_sequence) as usize);
        for sequence in first_sequence..batch_end {
            let payload_len = expected_chunk_len(byte_len, sequence)?;
            let offset = u64::from(sequence) * MAX_AUTH_PAYLOAD as u64;
            file.seek(std::io::SeekFrom::Start(offset)).await?;
            let mut payload = vec![0u8; payload_len];
            file.read_exact(&mut payload).await?;
            let header = Header {
                kind: KIND_DATA,
                flags: 0,
                message_id: offer.message_id,
                tensor_id: offer.descriptor.tensor_id,
                sequence,
                payload_len: payload.len() as u16,
                checksum: protocol::checksum(&payload),
                worker_id: 0,
            };
            packets.push((
                sequence,
                make_authenticated_packet(
                    key, &offer.sender_node_id, offer.sequence, header, &payload
                )?,
            ));
        }

        let mut acknowledged = vec![false; packets.len()];
        let mut attempts = 0usize;
        while acknowledged.iter().any(|value| !value) {
            for (index, (sequence, packet)) in packets.iter().enumerate() {
                if acknowledged[index] { continue; }
                socket.send_to(packet, data_target).await?;
                tokio::time::sleep(SEND_PACING).await;
                let _ = sequence;
            }

            let deadline = tokio::time::Instant::now() + ACK_WAIT;
            let mut buffer = [0u8; MAX_DATAGRAM];
            while tokio::time::Instant::now() < deadline && acknowledged.iter().any(|value| !value) {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                match timeout(remaining, socket.recv_from(&mut buffer)).await {
                    Ok(Ok((n, peer))) if peer == data_target => {
                        let Some((header, payload, tag)) = decode_authenticated_datagram(&buffer[..n]) else { continue };
                        if header.kind != KIND_ACK
                            || header.message_id != offer.message_id
                            || header.tensor_id != offer.descriptor.tensor_id
                            || !payload.is_empty()
                            || header.flags != 0
                            || header.worker_id != 0
                            || protocol::checksum(payload) != header.checksum
                            || !auth::verify_tensor_packet(
                                key, &offer.sender_node_id, offer.sequence, &header, payload, tag
                            )? {
                            continue;
                        }
                        let seq = header.sequence;
                        if seq >= first_sequence && seq < batch_end {
                            let index = (seq - first_sequence) as usize;
                            if !acknowledged[index] {
                                acknowledged[index] = true;
                                acknowledged_total += 1;
                            }
                        }
                    }
                    Ok(Ok((_n, _peer))) => continue,
                    Ok(Err(error)) => return Err(error.into()),
                    Err(_) => break,
                }
            }
            attempts += 1;
            if attempts > RETRY_LIMIT {
                bail!(
                    "authenticated tensor transfer retry limit exceeded; acknowledged {acknowledged_total}/{total_chunks} chunks"
                );
            }
        }
        println!("tensor progress: {acknowledged_total}/{total_chunks} chunks");
        first_sequence = batch_end;
    }

    let fin_payload = [
        total_chunks.to_be_bytes().as_slice(),
        byte_len.to_be_bytes().as_slice(),
    ].concat();
    let fin_header = Header {
        kind: KIND_FIN,
        flags: 0,
        message_id: offer.message_id,
        tensor_id: offer.descriptor.tensor_id,
        sequence: total_chunks,
        payload_len: fin_payload.len() as u16,
        checksum: protocol::checksum(&fin_payload),
        worker_id: 0,
    };
    let fin = make_authenticated_packet(
        key, &offer.sender_node_id, offer.sequence, fin_header, &fin_payload
    )?;
    let mut buffer = [0u8; MAX_DATAGRAM];
    for _ in 0..RETRY_LIMIT {
        socket.send_to(&fin, data_target).await?;
        match timeout(FIN_ACK_WAIT, socket.recv_from(&mut buffer)).await {
            Ok(Ok((n, peer))) if peer == data_target => {
                let Some((header, payload, tag)) = decode_authenticated_datagram(&buffer[..n]) else { continue };
                if header.kind == KIND_FIN_ACK
                    && header.message_id == offer.message_id
                    && header.tensor_id == offer.descriptor.tensor_id
                    && header.sequence == total_chunks
                    && payload.is_empty()
                    && header.flags == 0
                    && header.worker_id == 0
                    && protocol::checksum(payload) == header.checksum
                    && auth::verify_tensor_packet(
                        key, &offer.sender_node_id, offer.sequence, &header, payload, tag
                    )? {
                    println!("authenticated tensor transfer complete");
                    return Ok(());
                }
            }
            Ok(Ok((_n, _peer))) => continue,
            Ok(Err(error)) => return Err(error.into()),
            Err(_) => continue,
        }
    }
    bail!("worker did not confirm authenticated tensor completion")
}

fn make_authenticated_packet(
    key: &[u8],
    sender_node_id: &str,
    offer_sequence: u64,
    header: Header,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let mut bytes = protocol::packet(header, payload);
    let tag = auth::tensor_packet_tag(key, sender_node_id, offer_sequence, &header, payload)?;
    bytes.extend_from_slice(&tag);
    debug_assert!(bytes.len() <= MAX_DATAGRAM);
    Ok(bytes)
}

fn authenticated_control_packet(
    key: &[u8],
    sender_node_id: &str,
    offer_sequence: u64,
    kind: u8,
    message_id: u32,
    tensor_id: u32,
    sequence: u32,
) -> Result<Vec<u8>> {
    let payload = [];
    let header = Header {
        kind,
        flags: 0,
        message_id,
        tensor_id,
        sequence,
        payload_len: 0,
        checksum: protocol::checksum(&payload),
        worker_id: 0,
    };
    make_authenticated_packet(key, sender_node_id, offer_sequence, header, &payload)
}

fn decode_authenticated_datagram(buffer: &[u8]) -> Option<(Header, &[u8], &[u8])> {
    let header = Header::decode(buffer).ok()?;
    let payload_len = usize::from(header.payload_len);
    if payload_len > MAX_AUTH_PAYLOAD
        || buffer.len() != HEADER_LEN + payload_len + AUTH_TAG_LEN {
        return None;
    }
    let payload_end = HEADER_LEN + payload_len;
    Some((
        header,
        &buffer[HEADER_LEN..payload_end],
        &buffer[payload_end..],
    ))
}

fn chunk_count(byte_len: u64) -> Result<u32> {
    if byte_len == 0 || byte_len > MAX_WORKER_TENSOR_BYTES {
        bail!("tensor byte length must be between 1 and {MAX_WORKER_TENSOR_BYTES}");
    }
    let chunk_size = MAX_AUTH_PAYLOAD as u64;
    let count = byte_len / chunk_size + u64::from(byte_len % chunk_size != 0);
    u32::try_from(count).context("tensor requires too many data-plane chunks")
}

fn expected_chunk_len(byte_len: u64, sequence: u32) -> Result<usize> {
    let count = chunk_count(byte_len)?;
    if sequence >= count {
        bail!("tensor chunk sequence is outside the descriptor byte range");
    }
    let offset = u64::from(sequence) * MAX_AUTH_PAYLOAD as u64;
    usize::try_from((byte_len - offset).min(MAX_AUTH_PAYLOAD as u64))
        .context("chunk length does not fit this platform")
}

fn same_transfer(active: &ActiveTransfer, peer: SocketAddr, header: &Header) -> bool {
    active.offer.peer == peer
        && active.offer.offer.message_id == header.message_id
        && active.offer.offer.descriptor.tensor_id == header.tensor_id
}

fn tensor_paths(
    output_dir: &Path,
    sender_node_id: &str,
    offer_sequence: u64,
    message_id: u32,
    tensor_id: u32,
) -> (PathBuf, PathBuf) {
    let digest = Sha256::digest(sender_node_id.as_bytes());
    let mut sender_prefix = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(&mut sender_prefix, "{byte:02x}");
    }
    let unique = uuid::Uuid::new_v4();
    let stem = format!("{sender_prefix}-{offer_sequence}-{message_id}-{tensor_id}-{unique}");
    (
        output_dir.join(format!("{stem}.part")),
        output_dir.join(format!("{stem}.tensor")),
    )
}

fn remember_completed(completed: &mut Vec<CompletedTransfer>, transfer: CompletedTransfer) {
    if completed.len() >= MAX_COMPLETED_TRANSFERS {
        completed.remove(0);
    }
    completed.push(transfer);
}

fn verify_fin(
    done: &CompletedTransfer,
    key: &[u8],
    header: &Header,
    payload: &[u8],
    tag: &[u8],
) -> Result<bool> {
    if payload.len() != 12
        || header.sequence != done.total_chunks
        || u32::from_be_bytes(payload[0..4].try_into()?) != done.total_chunks
        || u64::from_be_bytes(payload[4..12].try_into()?) != done.byte_len {
        return Ok(false);
    }
    auth::verify_tensor_packet(
        key, &done.sender_node_id, done.offer_sequence, header, payload, tag
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn authenticated_loopback_transfer_commits_exact_bytes() {
        use crate::{
            auth::sign_tensor_offer,
            control::{TensorOfferMessage, CONTROL_VERSION},
            tensor::{TensorDescriptor, TensorDType, TensorEncoding, TensorLayout},
        };

        let root = std::env::temp_dir().join(format!("aitp-e2e-{}", uuid::Uuid::new_v4()));
        let output_dir = root.join("out");
        fs::create_dir_all(&root).await.unwrap();

        let data: Vec<u8> = (0..(MAX_AUTH_PAYLOAD * 3 + 17))
            .map(|index| (index % 251) as u8)
            .collect();
        let input = root.join("input.bin");
        fs::write(&input, &data).await.unwrap();

        let receiver_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let receiver_addr = receiver_socket.local_addr().unwrap();
        let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let sender_addr = sender_socket.local_addr().unwrap();
        let key = vec![0x5c; 32];

        let mut offer = TensorOfferMessage {
            version: CONTROL_VERSION,
            sender_node_id: "loopback-sender".into(),
            sequence: 1,
            message_id: 0x4154_5001,
            descriptor: TensorDescriptor {
                tensor_id: 17,
                name: "integration.test.tensor".into(),
                dtype: TensorDType::U8,
                shape: vec![data.len() as u64],
                layout: TensorLayout::RowMajor,
                encoding: TensorEncoding::Raw,
                byte_len: data.len() as u64,
            },
            auth_tag: Vec::new(),
        };
        sign_tensor_offer(&mut offer, &key).unwrap();

        let mut admission_state = OfferAdmission::default();
        admission_state
            .admit(&offer, sender_addr, &key, Instant::now())
            .unwrap();
        let admission = Arc::new(Mutex::new(admission_state));

        let receiver = tokio::spawn(run_receiver_in(
            receiver_socket,
            key.clone(),
            admission.clone(),
            output_dir.clone(),
        ));

        let transfer_result = timeout(
            Duration::from_secs(10),
            send_tensor_data(&sender_socket, receiver_addr, &input, &offer, &key),
        )
        .await;
        if !matches!(transfer_result, Ok(Ok(()))) {
            receiver.abort();
            let _ = receiver.await;
            let _ = fs::remove_dir_all(&root).await;
            panic!("loopback tensor transfer failed: {transfer_result:?}");
        }
        receiver.abort();
        let _ = receiver.await;

        let mut entries = fs::read_dir(&output_dir).await.unwrap();
        let mut committed_path = None;
        while let Some(entry) = entries.next_entry().await.unwrap() {
            if entry.path().extension().and_then(|value| value.to_str()) == Some("tensor") {
                committed_path = Some(entry.path());
            }
        }
        let committed_path = committed_path.expect("receiver should commit a .tensor file");
        let received = fs::read(&committed_path).await.unwrap();
        assert_eq!(received, data);

        let manifest_path = storage::manifest_path_for_payload(&committed_path);
        let manifest_bytes = fs::read(&manifest_path).await.unwrap();
        let manifest: storage::TensorManifest = serde_json::from_slice(&manifest_bytes).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.descriptor.name, "integration.test.tensor");
        assert_eq!(manifest.byte_len, data.len() as u64);
        assert_eq!(manifest.payload_file, committed_path.file_name().unwrap().to_string_lossy().to_string());
        assert_eq!(storage::sha256_file(&committed_path).await.unwrap(), manifest.sha256);
        fs::remove_dir_all(&root).await.unwrap();
    }

    #[test]
    fn completed_retry_cache_is_bounded() {
        let mut completed = Vec::new();
        let now = Instant::now();
        for message_id in 0..=(MAX_COMPLETED_TRANSFERS as u32) {
            remember_completed(
                &mut completed,
                CompletedTransfer {
                    sender_node_id: "sender".into(),
                    offer_sequence: u64::from(message_id),
                    message_id,
                    tensor_id: 1,
                    peer: "127.0.0.1:40000".parse().unwrap(),
                    total_chunks: 1,
                    byte_len: 1,
                    expires_at: now + COMPLETED_RETRY_TTL,
                },
            );
        }
        assert_eq!(completed.len(), MAX_COMPLETED_TRANSFERS);
        assert_eq!(completed.first().unwrap().message_id, 1);
        assert_eq!(completed.last().unwrap().message_id, MAX_COMPLETED_TRANSFERS as u32);
    }

    #[test]
    fn auth_datagram_encodes_unchanged_header_and_fits_mtu() {
        let key = [0x21; 32];
        let payload = vec![7u8; MAX_AUTH_PAYLOAD];
        let header = Header {
            kind: KIND_DATA, flags: 0, message_id: 2, tensor_id: 3,
            sequence: 4, payload_len: payload.len() as u16,
            checksum: protocol::checksum(&payload), worker_id: 0,
        };
        let packet = make_authenticated_packet(&key, "sender", 5, header, &payload).unwrap();
        assert_eq!(packet.len(), MAX_DATAGRAM);
        let (decoded, decoded_payload, tag) = decode_authenticated_datagram(&packet).unwrap();
        assert_eq!(decoded.encode(), header.encode());
        assert_eq!(decoded_payload, payload);
        assert!(auth::verify_tensor_packet(&key, "sender", 5, &decoded, decoded_payload, tag).unwrap());
    }

    #[test]
    fn rejects_truncated_or_oversized_authenticated_datagrams() {
        let key = [0x21; 32];
        let payload = vec![1u8; MAX_AUTH_PAYLOAD];
        let header = Header {
            kind: KIND_DATA, flags: 0, message_id: 2, tensor_id: 3,
            sequence: 0, payload_len: payload.len() as u16,
            checksum: protocol::checksum(&payload), worker_id: 0,
        };
        let packet = make_authenticated_packet(&key, "sender", 1, header, &payload).unwrap();
        assert!(decode_authenticated_datagram(&packet[..packet.len() - 1]).is_none());
        let mut oversized = packet;
        oversized.push(0);
        assert!(decode_authenticated_datagram(&oversized).is_none());
    }

    #[test]
    fn chunk_boundaries_cover_exact_tensor_length() {
        assert_eq!(chunk_count(1).unwrap(), 1);
        assert_eq!(chunk_count(MAX_AUTH_PAYLOAD as u64).unwrap(), 1);
        assert_eq!(chunk_count(MAX_AUTH_PAYLOAD as u64 + 1).unwrap(), 2);
        assert_eq!(expected_chunk_len(MAX_AUTH_PAYLOAD as u64 + 1, 0).unwrap(), MAX_AUTH_PAYLOAD);
        assert_eq!(expected_chunk_len(MAX_AUTH_PAYLOAD as u64 + 1, 1).unwrap(), 1);
        assert!(expected_chunk_len(12, 1).is_err());
    }

    #[test]
    fn transfer_paths_do_not_use_untrusted_tensor_names() {
        let (partial, final_path) = tensor_paths(
            Path::new("aitp-tensors"), "../../etc/passwd", 1, 2, 3
        );
        assert!(partial.starts_with("aitp-tensors"));
        assert!(final_path.starts_with("aitp-tensors"));
        assert!(final_path.to_string_lossy().ends_with(".tensor"));
        assert!(!final_path.to_string_lossy().contains("passwd"));
    }
}
