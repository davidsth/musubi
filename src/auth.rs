//! Shared-secret authentication for cluster membership and tensor control messages.
//! HMAC authenticates metadata; it does not encrypt it.

use anyhow::{bail, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::control::{
    AllocationPrepare, AllocationRelease, AllocationReleaseReply, AllocationReply,
    AllocationStatusReply, AllocationStatusRequest, NodeAnnouncement, TensorOfferMessage,
    TensorOfferReply, CONTROL_VERSION,
};

type HmacSha256 = Hmac<Sha256>;
pub const MIN_CLUSTER_KEY_BYTES: usize = 32;
pub const HMAC_TAG_BYTES: usize = 32;

fn valid_session_id(session_id: &str) -> bool {
    uuid::Uuid::parse_str(session_id).is_ok()
}

fn node_registration_mac(
    node: &NodeAnnouncement,
    session_id: &str,
    key: &[u8],
) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/NODE_REGISTER/V1\0")?;
    update_blob(&mut mac, session_id.as_bytes());
    let encoded_node = serde_json::to_vec(node)?;
    update_blob(&mut mac, &encoded_node);
    Ok(mac)
}

/// Sign a node registration. Registration announcements must have an empty
/// discovery nonce; discovery replies remain metadata-only and unauthenticated.
pub fn sign_node_registration(
    node: &NodeAnnouncement,
    session_id: &str,
    key: &[u8],
) -> Result<Vec<u8>> {
    if !valid_session_id(session_id) {
        bail!("invalid node session ID");
    }
    if !node.is_valid() || !node.nonce.is_empty() {
        bail!("invalid node announcement for registration");
    }
    Ok(node_registration_mac(node, session_id, key)?
        .finalize()
        .into_bytes()
        .to_vec())
}

/// Verify registration authentication before trusting node metadata or the
/// source endpoint advertised by the UDP peer.
pub fn verify_node_registration(
    node: &NodeAnnouncement,
    session_id: &str,
    tag: &[u8],
    key: &[u8],
) -> Result<bool> {
    if tag.len() != HMAC_TAG_BYTES
        || !valid_session_id(session_id)
        || !node.is_valid()
        || !node.nonce.is_empty()
    {
        return Ok(false);
    }
    Ok(node_registration_mac(node, session_id, key)?
        .verify_slice(tag)
        .is_ok())
}

fn node_heartbeat_mac(
    version: u8,
    node_id: &str,
    session_id: &str,
    sequence: u64,
    available_memory_bytes: u64,
    reserved_memory_bytes: u64,
    key: &[u8],
) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/NODE_HEARTBEAT/V2\0")?;
    mac.update(&[version]);
    update_blob(&mut mac, node_id.as_bytes());
    update_blob(&mut mac, session_id.as_bytes());
    mac.update(&sequence.to_be_bytes());
    mac.update(&available_memory_bytes.to_be_bytes());
    mac.update(&reserved_memory_bytes.to_be_bytes());
    Ok(mac)
}

pub fn sign_node_heartbeat(
    version: u8,
    node_id: &str,
    session_id: &str,
    sequence: u64,
    available_memory_bytes: u64,
    reserved_memory_bytes: u64,
    key: &[u8],
) -> Result<Vec<u8>> {
    if version != CONTROL_VERSION
        || node_id.is_empty()
        || node_id.len() > 64
        || !valid_session_id(session_id)
        || sequence == 0
        || reserved_memory_bytes > (1_u64 << 50)
    {
        bail!("invalid node heartbeat fields");
    }
    Ok(node_heartbeat_mac(
        version,
        node_id,
        session_id,
        sequence,
        available_memory_bytes,
        reserved_memory_bytes,
        key,
    )?
    .finalize()
    .into_bytes()
    .to_vec())
}

pub fn verify_node_heartbeat(
    version: u8,
    node_id: &str,
    session_id: &str,
    sequence: u64,
    available_memory_bytes: u64,
    reserved_memory_bytes: u64,
    tag: &[u8],
    key: &[u8],
) -> Result<bool> {
    if tag.len() != HMAC_TAG_BYTES
        || version != CONTROL_VERSION
        || node_id.is_empty()
        || node_id.len() > 64
        || !valid_session_id(session_id)
        || sequence == 0
        || reserved_memory_bytes > (1_u64 << 50)
    {
        return Ok(false);
    }
    Ok(node_heartbeat_mac(
        version,
        node_id,
        session_id,
        sequence,
        available_memory_bytes,
        reserved_memory_bytes,
        key,
    )?
    .verify_slice(tag)
    .is_ok())
}

pub fn cluster_key_from_env() -> Result<Option<Vec<u8>>> {
    match std::env::var("AITP_CLUSTER_KEY") {
        Ok(value) => {
            if value.as_bytes().len() < MIN_CLUSTER_KEY_BYTES {
                bail!("AITP_CLUSTER_KEY must contain at least {MIN_CLUSTER_KEY_BYTES} bytes");
            }
            Ok(Some(value.into_bytes()))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn new_mac(key: &[u8], domain: &[u8]) -> Result<HmacSha256> {
    if key.len() < MIN_CLUSTER_KEY_BYTES {
        bail!("cluster key is too short");
    }
    let mut mac =
        HmacSha256::new_from_slice(key).map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(domain);
    Ok(mac)
}

fn update_blob(mac: &mut HmacSha256, bytes: &[u8]) {
    mac.update(&(bytes.len() as u64).to_be_bytes());
    mac.update(bytes);
}

fn offer_mac(key: &[u8], offer: &TensorOfferMessage) -> Result<Vec<u8>> {
    let mut mac = new_mac(key, b"AITP/TENSOR_OFFER/V1\0")?;
    mac.update(&[offer.version]);
    update_blob(&mut mac, offer.sender_node_id.as_bytes());
    mac.update(&offer.sequence.to_be_bytes());
    mac.update(&offer.message_id.to_be_bytes());
    let descriptor = serde_json::to_vec(&offer.descriptor)?;
    update_blob(&mut mac, &descriptor);
    Ok(mac.finalize().into_bytes().to_vec())
}

pub fn sign_tensor_offer(offer: &mut TensorOfferMessage, key: &[u8]) -> Result<()> {
    offer.auth_tag = offer_mac(key, offer)?;
    Ok(())
}

pub fn verify_tensor_offer(offer: &TensorOfferMessage, key: &[u8]) -> Result<bool> {
    if offer.auth_tag.len() != HMAC_TAG_BYTES {
        return Ok(false);
    }
    let mut mac = new_mac(key, b"AITP/TENSOR_OFFER/V1\0")?;
    mac.update(&[offer.version]);
    update_blob(&mut mac, offer.sender_node_id.as_bytes());
    mac.update(&offer.sequence.to_be_bytes());
    mac.update(&offer.message_id.to_be_bytes());
    let descriptor = serde_json::to_vec(&offer.descriptor)?;
    update_blob(&mut mac, &descriptor);
    Ok(mac.verify_slice(&offer.auth_tag).is_ok())
}

fn reply_mac(key: &[u8], reply: &TensorOfferReply) -> Result<Vec<u8>> {
    let mut mac = new_mac(key, b"AITP/TENSOR_OFFER_REPLY/V1\0")?;
    mac.update(&[reply.version]);
    update_blob(&mut mac, reply.receiver_node_id.as_bytes());
    update_blob(&mut mac, reply.sender_node_id.as_bytes());
    mac.update(&reply.sequence.to_be_bytes());
    mac.update(&reply.message_id.to_be_bytes());
    mac.update(&reply.tensor_id.to_be_bytes());
    mac.update(&[u8::from(reply.accepted)]);
    match reply.reason.as_deref() {
        Some(reason) => {
            mac.update(&[1]);
            update_blob(&mut mac, reason.as_bytes());
        }
        None => mac.update(&[0]),
    }
    Ok(mac.finalize().into_bytes().to_vec())
}

pub fn sign_tensor_offer_reply(reply: &mut TensorOfferReply, key: &[u8]) -> Result<()> {
    reply.auth_tag = reply_mac(key, reply)?;
    Ok(())
}

/// Produce a per-datagram tag for the binary tensor data plane. The fixed
/// 32-byte header is authenticated as-is; the tag is appended after payload.
pub fn tensor_packet_tag(
    key: &[u8],
    sender_node_id: &str,
    offer_sequence: u64,
    header: &crate::protocol::Header,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let mut mac = new_mac(key, b"AITP/TENSOR_PACKET/V1\0")?;
    update_blob(&mut mac, sender_node_id.as_bytes());
    mac.update(&offer_sequence.to_be_bytes());
    mac.update(&header.encode());
    update_blob(&mut mac, payload);
    Ok(mac.finalize().into_bytes().to_vec())
}

pub fn verify_tensor_packet(
    key: &[u8],
    sender_node_id: &str,
    offer_sequence: u64,
    header: &crate::protocol::Header,
    payload: &[u8],
    tag: &[u8],
) -> Result<bool> {
    if tag.len() != HMAC_TAG_BYTES {
        return Ok(false);
    }
    let mut mac = new_mac(key, b"AITP/TENSOR_PACKET/V1\0")?;
    update_blob(&mut mac, sender_node_id.as_bytes());
    mac.update(&offer_sequence.to_be_bytes());
    mac.update(&header.encode());
    update_blob(&mut mac, payload);
    Ok(mac.verify_slice(tag).is_ok())
}

pub fn verify_tensor_offer_reply(reply: &TensorOfferReply, key: &[u8]) -> Result<bool> {
    if reply.auth_tag.len() != HMAC_TAG_BYTES {
        return Ok(false);
    }
    let mut mac = new_mac(key, b"AITP/TENSOR_OFFER_REPLY/V1\0")?;
    mac.update(&[reply.version]);
    update_blob(&mut mac, reply.receiver_node_id.as_bytes());
    update_blob(&mut mac, reply.sender_node_id.as_bytes());
    mac.update(&reply.sequence.to_be_bytes());
    mac.update(&reply.message_id.to_be_bytes());
    mac.update(&reply.tensor_id.to_be_bytes());
    mac.update(&[u8::from(reply.accepted)]);
    match reply.reason.as_deref() {
        Some(reason) => {
            mac.update(&[1]);
            update_blob(&mut mac, reason.as_bytes());
        }
        None => mac.update(&[0]),
    }
    Ok(mac.verify_slice(&reply.auth_tag).is_ok())
}

fn valid_allocation_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok()
}

fn valid_model_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_layer_range(first: Option<u64>, end: Option<u64>) -> bool {
    match (first, end) {
        (Some(start), Some(end)) => start < end && end <= 4096,
        (None, None) => true,
        _ => false,
    }
}

fn valid_allocation_prepare(request: &AllocationPrepare) -> bool {
    request.version == CONTROL_VERSION
        && valid_allocation_uuid(&request.coordinator_node_id)
        && valid_allocation_uuid(&request.target_node_id)
        && valid_allocation_uuid(&request.target_session_id)
        && valid_allocation_uuid(&request.lease_id)
        && valid_model_hash(&request.model_sha256)
        && valid_layer_range(request.first_layer, request.end_layer_exclusive)
        && request.reserved_bytes > 0
        && request.reserved_bytes <= (1_u64 << 40)
        && request.expires_at_unix > 0
}

fn allocation_prepare_mac(key: &[u8], request: &AllocationPrepare) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_PREPARE/V1\0")?;
    mac.update(&[request.version]);
    update_blob(&mut mac, request.coordinator_node_id.as_bytes());
    update_blob(&mut mac, request.target_node_id.as_bytes());
    update_blob(&mut mac, request.target_session_id.as_bytes());
    update_blob(&mut mac, request.lease_id.as_bytes());
    update_blob(&mut mac, request.model_sha256.as_bytes());
    update_optional_u64(&mut mac, request.first_layer);
    update_optional_u64(&mut mac, request.end_layer_exclusive);
    mac.update(&request.reserved_bytes.to_be_bytes());
    mac.update(&request.expires_at_unix.to_be_bytes());
    Ok(mac)
}

pub fn sign_allocation_prepare(request: &mut AllocationPrepare, key: &[u8]) -> Result<()> {
    if !valid_allocation_prepare(request) {
        bail!("invalid allocation prepare fields");
    }
    request.auth_tag = allocation_prepare_mac(key, request)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_prepare(request: &AllocationPrepare, key: &[u8]) -> Result<bool> {
    if request.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_prepare(request) {
        return Ok(false);
    }
    Ok(allocation_prepare_mac(key, request)?
        .verify_slice(&request.auth_tag)
        .is_ok())
}

fn valid_allocation_reply(reply: &AllocationReply) -> bool {
    reply.version == CONTROL_VERSION
        && valid_allocation_uuid(&reply.coordinator_node_id)
        && valid_allocation_uuid(&reply.node_id)
        && valid_allocation_uuid(&reply.session_id)
        && valid_allocation_uuid(&reply.lease_id)
        && reply.reserved_bytes > 0
        && reply.reserved_bytes <= (1_u64 << 40)
        && reply.expires_at_unix > 0
        && match reply.reason.as_deref() {
            Some(reason) => !reason.is_empty() && reason.len() <= 128 && !reply.accepted,
            None => reply.accepted,
        }
}

fn allocation_reply_mac(key: &[u8], reply: &AllocationReply) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_REPLY/V1\0")?;
    mac.update(&[reply.version]);
    update_blob(&mut mac, reply.coordinator_node_id.as_bytes());
    update_blob(&mut mac, reply.node_id.as_bytes());
    update_blob(&mut mac, reply.session_id.as_bytes());
    update_blob(&mut mac, reply.lease_id.as_bytes());
    mac.update(&[u8::from(reply.accepted)]);
    match reply.reason.as_deref() {
        Some(reason) => {
            mac.update(&[1]);
            update_blob(&mut mac, reason.as_bytes());
        }
        None => mac.update(&[0]),
    }
    mac.update(&reply.reserved_bytes.to_be_bytes());
    mac.update(&reply.expires_at_unix.to_be_bytes());
    Ok(mac)
}

pub fn sign_allocation_reply(reply: &mut AllocationReply, key: &[u8]) -> Result<()> {
    if !valid_allocation_reply(reply) {
        bail!("invalid allocation reply fields");
    }
    reply.auth_tag = allocation_reply_mac(key, reply)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_reply(reply: &AllocationReply, key: &[u8]) -> Result<bool> {
    if reply.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_reply(reply) {
        return Ok(false);
    }
    Ok(allocation_reply_mac(key, reply)?
        .verify_slice(&reply.auth_tag)
        .is_ok())
}

fn valid_allocation_release(request: &AllocationRelease) -> bool {
    request.version == CONTROL_VERSION
        && valid_allocation_uuid(&request.coordinator_node_id)
        && valid_allocation_uuid(&request.target_node_id)
        && valid_allocation_uuid(&request.target_session_id)
        && valid_allocation_uuid(&request.lease_id)
}

fn allocation_release_mac(key: &[u8], request: &AllocationRelease) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_RELEASE/V1\0")?;
    mac.update(&[request.version]);
    update_blob(&mut mac, request.coordinator_node_id.as_bytes());
    update_blob(&mut mac, request.target_node_id.as_bytes());
    update_blob(&mut mac, request.target_session_id.as_bytes());
    update_blob(&mut mac, request.lease_id.as_bytes());
    Ok(mac)
}

pub fn sign_allocation_release(request: &mut AllocationRelease, key: &[u8]) -> Result<()> {
    if !valid_allocation_release(request) {
        bail!("invalid allocation release fields");
    }
    request.auth_tag = allocation_release_mac(key, request)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_release(request: &AllocationRelease, key: &[u8]) -> Result<bool> {
    if request.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_release(request) {
        return Ok(false);
    }
    Ok(allocation_release_mac(key, request)?
        .verify_slice(&request.auth_tag)
        .is_ok())
}

fn valid_allocation_release_reply(reply: &AllocationReleaseReply) -> bool {
    reply.version == CONTROL_VERSION
        && valid_allocation_uuid(&reply.coordinator_node_id)
        && valid_allocation_uuid(&reply.node_id)
        && valid_allocation_uuid(&reply.session_id)
        && valid_allocation_uuid(&reply.lease_id)
}

fn allocation_release_reply_mac(key: &[u8], reply: &AllocationReleaseReply) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_RELEASE_REPLY/V1\0")?;
    mac.update(&[reply.version]);
    update_blob(&mut mac, reply.coordinator_node_id.as_bytes());
    update_blob(&mut mac, reply.node_id.as_bytes());
    update_blob(&mut mac, reply.session_id.as_bytes());
    update_blob(&mut mac, reply.lease_id.as_bytes());
    mac.update(&[u8::from(reply.released)]);
    Ok(mac)
}

pub fn sign_allocation_release_reply(reply: &mut AllocationReleaseReply, key: &[u8]) -> Result<()> {
    if !valid_allocation_release_reply(reply) {
        bail!("invalid allocation release reply fields");
    }
    reply.auth_tag = allocation_release_reply_mac(key, reply)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_release_reply(reply: &AllocationReleaseReply, key: &[u8]) -> Result<bool> {
    if reply.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_release_reply(reply) {
        return Ok(false);
    }
    Ok(allocation_release_reply_mac(key, reply)?
        .verify_slice(&reply.auth_tag)
        .is_ok())
}

fn valid_allocation_status(request: &AllocationStatusRequest) -> bool {
    request.version == CONTROL_VERSION
        && valid_allocation_uuid(&request.coordinator_node_id)
        && valid_allocation_uuid(&request.target_node_id)
        && valid_allocation_uuid(&request.target_session_id)
        && valid_allocation_uuid(&request.lease_id)
        && valid_allocation_uuid(&request.request_id)
}

fn allocation_status_mac(key: &[u8], request: &AllocationStatusRequest) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_STATUS/V1\0")?;
    mac.update(&[request.version]);
    update_blob(&mut mac, request.coordinator_node_id.as_bytes());
    update_blob(&mut mac, request.target_node_id.as_bytes());
    update_blob(&mut mac, request.target_session_id.as_bytes());
    update_blob(&mut mac, request.lease_id.as_bytes());
    update_blob(&mut mac, request.request_id.as_bytes());
    Ok(mac)
}

pub fn sign_allocation_status(request: &mut AllocationStatusRequest, key: &[u8]) -> Result<()> {
    if !valid_allocation_status(request) {
        bail!("invalid allocation status request fields");
    }
    request.auth_tag = allocation_status_mac(key, request)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_status(request: &AllocationStatusRequest, key: &[u8]) -> Result<bool> {
    if request.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_status(request) {
        return Ok(false);
    }
    Ok(allocation_status_mac(key, request)?
        .verify_slice(&request.auth_tag)
        .is_ok())
}

fn valid_allocation_status_reply(reply: &AllocationStatusReply) -> bool {
    reply.version == CONTROL_VERSION
        && valid_allocation_uuid(&reply.coordinator_node_id)
        && valid_allocation_uuid(&reply.node_id)
        && valid_allocation_uuid(&reply.session_id)
        && valid_allocation_uuid(&reply.lease_id)
        && valid_allocation_uuid(&reply.request_id)
        && (if reply.active {
            reply.reserved_bytes > 0 && reply.expires_at_unix > 0
        } else {
            reply.reserved_bytes == 0 && reply.expires_at_unix == 0
        })
}

fn allocation_status_reply_mac(key: &[u8], reply: &AllocationStatusReply) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/ALLOCATION_STATUS_REPLY/V1\0")?;
    mac.update(&[reply.version]);
    update_blob(&mut mac, reply.coordinator_node_id.as_bytes());
    update_blob(&mut mac, reply.node_id.as_bytes());
    update_blob(&mut mac, reply.session_id.as_bytes());
    update_blob(&mut mac, reply.lease_id.as_bytes());
    update_blob(&mut mac, reply.request_id.as_bytes());
    mac.update(&[u8::from(reply.active)]);
    mac.update(&reply.reserved_bytes.to_be_bytes());
    mac.update(&reply.expires_at_unix.to_be_bytes());
    Ok(mac)
}

pub fn sign_allocation_status_reply(reply: &mut AllocationStatusReply, key: &[u8]) -> Result<()> {
    if !valid_allocation_status_reply(reply) {
        bail!("invalid allocation status reply fields");
    }
    reply.auth_tag = allocation_status_reply_mac(key, reply)?
        .finalize()
        .into_bytes()
        .to_vec();
    Ok(())
}

pub fn verify_allocation_status_reply(reply: &AllocationStatusReply, key: &[u8]) -> Result<bool> {
    if reply.auth_tag.len() != HMAC_TAG_BYTES || !valid_allocation_status_reply(reply) {
        return Ok(false);
    }
    Ok(allocation_status_reply_mac(key, reply)?
        .verify_slice(&reply.auth_tag)
        .is_ok())
}

fn update_optional_u64(mac: &mut HmacSha256, value: Option<u64>) {
    match value {
        Some(value) => {
            mac.update(&[1]);
            mac.update(&value.to_be_bytes());
        }
        None => mac.update(&[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        control::CONTROL_VERSION,
        tensor::{TensorDType, TensorDescriptor, TensorEncoding, TensorLayout},
    };

    fn sample_node() -> NodeAnnouncement {
        NodeAnnouncement {
            version: CONTROL_VERSION,
            nonce: String::new(),
            node_id: "stable-node".into(),
            hostname: "worker-a".into(),
            os: "linux".into(),
            arch: "x86_64".into(),
            logical_cpus: 16,
            cpu_model: "Example CPU".into(),
            accelerators: Vec::new(),
            total_memory_bytes: 32 * 1024 * 1024 * 1024,
            available_memory_bytes: 24 * 1024 * 1024 * 1024,
            inference_port: 9000,
        }
    }

    fn sample_offer(key: &[u8]) -> TensorOfferMessage {
        let mut offer = TensorOfferMessage {
            version: CONTROL_VERSION,
            sender_node_id: "sender".into(),
            sequence: 1,
            message_id: 2,
            descriptor: TensorDescriptor {
                tensor_id: 3,
                name: "layer.0.weight".into(),
                dtype: TensorDType::F16,
                shape: vec![4],
                layout: TensorLayout::RowMajor,
                encoding: TensorEncoding::Raw,
                byte_len: 8,
            },
            auth_tag: Vec::new(),
        };
        sign_tensor_offer(&mut offer, key).unwrap();
        offer
    }

    #[test]
    fn node_registration_tag_binds_metadata_and_process_session() {
        let key = [0x51; 32];
        let node = sample_node();
        let session = uuid::Uuid::new_v4().to_string();
        let tag = sign_node_registration(&node, &session, &key).unwrap();

        assert!(verify_node_registration(&node, &session, &tag, &key).unwrap());
        assert!(
            !verify_node_registration(&node, &uuid::Uuid::new_v4().to_string(), &tag, &key)
                .unwrap()
        );

        let mut changed = node.clone();
        changed.hostname.push_str("-spoofed");
        assert!(!verify_node_registration(&changed, &session, &tag, &key).unwrap());

        let mut invalid_tag = tag;
        invalid_tag[0] ^= 0x80;
        assert!(!verify_node_registration(&node, &session, &invalid_tag, &key).unwrap());
        assert!(!verify_node_registration(&node, "not-a-session", &[0; 32], &key).unwrap());
    }

    #[test]
    fn heartbeat_tag_binds_node_session_sequence_and_memory() {
        let key = [0x52; 32];
        let session = uuid::Uuid::new_v4().to_string();
        let tag = sign_node_heartbeat(
            CONTROL_VERSION,
            "stable-node",
            &session,
            7,
            12_345,
            256,
            &key,
        )
        .unwrap();

        assert!(verify_node_heartbeat(
            CONTROL_VERSION,
            "stable-node",
            &session,
            7,
            12_345,
            256,
            &tag,
            &key
        )
        .unwrap());
        assert!(!verify_node_heartbeat(
            CONTROL_VERSION,
            "stable-node",
            &session,
            8,
            12_345,
            256,
            &tag,
            &key
        )
        .unwrap());
        assert!(!verify_node_heartbeat(
            CONTROL_VERSION,
            "stable-node",
            &session,
            7,
            12_346,
            256,
            &tag,
            &key
        )
        .unwrap());
        assert!(!verify_node_heartbeat(
            CONTROL_VERSION,
            "stable-node",
            &session,
            7,
            12_345,
            257,
            &tag,
            &key
        )
        .unwrap());
        assert!(!verify_node_heartbeat(
            CONTROL_VERSION,
            "other-node",
            &session,
            7,
            12_345,
            256,
            &tag,
            &key
        )
        .unwrap());
    }

    #[test]
    fn offer_tag_detects_descriptor_tampering() {
        let key = [0x5a; 32];
        let mut offer = sample_offer(&key);
        assert!(verify_tensor_offer(&offer, &key).unwrap());
        offer.descriptor.byte_len = 10;
        assert!(!verify_tensor_offer(&offer, &key).unwrap());
    }

    #[test]
    fn reply_tag_covers_acceptance_and_reason() {
        let key = [0x33; 32];
        let mut reply = TensorOfferReply {
            version: CONTROL_VERSION,
            receiver_node_id: "receiver".into(),
            sender_node_id: "sender".into(),
            sequence: 1,
            message_id: 2,
            tensor_id: 3,
            accepted: false,
            reason: Some("over limit".into()),
            auth_tag: Vec::new(),
        };
        sign_tensor_offer_reply(&mut reply, &key).unwrap();
        assert!(verify_tensor_offer_reply(&reply, &key).unwrap());
        reply.accepted = true;
        assert!(!verify_tensor_offer_reply(&reply, &key).unwrap());
    }

    #[test]
    fn tensor_packet_tag_binds_header_payload_and_offer() {
        let key = [0x44; 32];
        let header = crate::protocol::Header {
            kind: crate::protocol::KIND_DATA,
            flags: 0,
            message_id: 9,
            tensor_id: 7,
            sequence: 1,
            payload_len: 3,
            checksum: crate::protocol::checksum(b"abc"),
            worker_id: 0,
        };
        let tag = tensor_packet_tag(&key, "sender", 12, &header, b"abc").unwrap();
        assert!(verify_tensor_packet(&key, "sender", 12, &header, b"abc", &tag).unwrap());
        assert!(!verify_tensor_packet(&key, "other", 12, &header, b"abc", &tag).unwrap());
        assert!(!verify_tensor_packet(&key, "sender", 13, &header, b"abc", &tag).unwrap());
        assert!(!verify_tensor_packet(&key, "sender", 12, &header, b"abd", &tag).unwrap());
        let mut changed = header;
        changed.sequence = 2;
        assert!(!verify_tensor_packet(&key, "sender", 12, &changed, b"abc", &tag).unwrap());
    }

    #[test]
    fn allocation_signatures_bind_targets_and_memory() {
        let key = [0x65; 32];
        let mut request = AllocationPrepare {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            model_sha256: "a".repeat(64),
            first_layer: Some(0),
            end_layer_exclusive: Some(4),
            reserved_bytes: 4096,
            expires_at_unix: 1_800_000_000,
            auth_tag: Vec::new(),
        };
        sign_allocation_prepare(&mut request, &key).unwrap();
        assert!(verify_allocation_prepare(&request, &key).unwrap());
        request.reserved_bytes += 1;
        assert!(!verify_allocation_prepare(&request, &key).unwrap());

        let mut release = AllocationRelease {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            auth_tag: Vec::new(),
        };
        sign_allocation_release(&mut release, &key).unwrap();
        assert!(verify_allocation_release(&release, &key).unwrap());
        release.target_session_id = "00000000-0000-4000-8000-000000000099".into();
        assert!(!verify_allocation_release(&release, &key).unwrap());
    }

    #[test]
    fn allocation_status_signatures_bind_query_and_reported_state() {
        let key = [0x66; 32];
        let mut request = AllocationStatusRequest {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            request_id: "00000000-0000-4000-8000-000000000005".into(),
            auth_tag: Vec::new(),
        };
        sign_allocation_status(&mut request, &key).unwrap();
        assert!(verify_allocation_status(&request, &key).unwrap());
        request.request_id = "00000000-0000-4000-8000-000000000006".into();
        assert!(!verify_allocation_status(&request, &key).unwrap());
        request.request_id = "00000000-0000-4000-8000-000000000005".into();
        request.lease_id = "00000000-0000-4000-8000-000000000007".into();
        assert!(!verify_allocation_status(&request, &key).unwrap());

        let mut reply = AllocationStatusReply {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            node_id: "00000000-0000-4000-8000-000000000002".into(),
            session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            request_id: "00000000-0000-4000-8000-000000000005".into(),
            active: true,
            reserved_bytes: 4096,
            expires_at_unix: 1_800_000_000,
            auth_tag: Vec::new(),
        };
        sign_allocation_status_reply(&mut reply, &key).unwrap();
        assert!(verify_allocation_status_reply(&reply, &key).unwrap());
        let mut replayed = reply.clone();
        replayed.request_id = "00000000-0000-4000-8000-000000000006".into();
        assert!(!verify_allocation_status_reply(&replayed, &key).unwrap());
        reply.reserved_bytes += 1;
        assert!(!verify_allocation_status_reply(&reply, &key).unwrap());

        let mut inactive = reply;
        inactive.active = false;
        inactive.reserved_bytes = 0;
        inactive.expires_at_unix = 0;
        inactive.auth_tag.clear();
        sign_allocation_status_reply(&mut inactive, &key).unwrap();
        assert!(verify_allocation_status_reply(&inactive, &key).unwrap());
    }

    #[test]
    fn cluster_key_requires_at_least_32_bytes() {
        assert!(new_mac(b"short", b"test").is_err());
        assert!(new_mac(&[0; 32], b"test").is_ok());
    }
}
