//! Shared-secret authentication for tensor-offer control messages.
//! HMAC authenticates metadata; it does not encrypt it.

use anyhow::{bail, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::control::{TensorOfferMessage, TensorOfferReply};

type HmacSha256 = Hmac<Sha256>;
pub const MIN_CLUSTER_KEY_BYTES: usize = 32;
pub const HMAC_TAG_BYTES: usize = 32;

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
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        control::CONTROL_VERSION,
        tensor::{TensorDescriptor, TensorDType, TensorEncoding, TensorLayout},
    };

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
    fn cluster_key_requires_at_least_32_bytes() {
        assert!(new_mac(b"short", b"test").is_err());
        assert!(new_mac(&[0; 32], b"test").is_ok());
    }
}
