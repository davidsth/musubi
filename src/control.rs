//! Versioned control-plane messages. Tensor payloads remain on the binary data plane.

use crate::tensor::TensorDescriptor;
use serde::{Deserialize, Serialize};

pub const CONTROL_VERSION: u8 = 6;
pub const MAX_CONTROL_DATAGRAM: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    Discover {
        version: u8,
        nonce: String,
    },
    Announce(NodeAnnouncement),
    Register {
        node: NodeAnnouncement,
    },
    TensorOffer {
        offer: TensorOfferMessage,
    },
    TensorOfferReply {
        reply: TensorOfferReply,
    },
    Heartbeat {
        version: u8,
        node_id: String,
        sequence: u64,
        available_memory_bytes: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOfferMessage {
    pub version: u8,
    pub sender_node_id: String,
    pub sequence: u64,
    pub message_id: u32,
    pub descriptor: TensorDescriptor,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOfferReply {
    pub version: u8,
    pub receiver_node_id: String,
    pub sender_node_id: String,
    pub sequence: u64,
    pub message_id: u32,
    pub tensor_id: u32,
    pub accepted: bool,
    pub reason: Option<String>,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorInfo {
    pub name: String,
    pub vendor: String,
    pub memory_bytes: Option<u64>,
    pub unified_memory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnnouncement {
    pub version: u8,
    pub nonce: String,
    pub node_id: String,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub logical_cpus: usize,
    pub cpu_model: String,
    pub accelerators: Vec<AcceleratorInfo>,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub inference_port: u16,
}

impl NodeAnnouncement {
    pub fn is_valid(&self) -> bool {
        self.version == CONTROL_VERSION
            && !self.node_id.is_empty()
            && self.node_id.len() <= 64
            && !self.hostname.is_empty()
            && self.hostname.len() <= 128
            && !self.os.is_empty()
            && self.os.len() <= 32
            && !self.arch.is_empty()
            && self.arch.len() <= 32
            && self.nonce.len() <= 64
            && (1..=4096).contains(&self.logical_cpus)
            && !self.cpu_model.is_empty()
            && self.cpu_model.len() <= 256
            && self.accelerators.len() <= 8
            && self
                .accelerators
                .iter()
                .all(|gpu| !gpu.name.is_empty() && gpu.name.len() <= 256 && gpu.vendor.len() <= 64)
            && self.total_memory_bytes <= (1_u64 << 60)
            && self.available_memory_bytes <= self.total_memory_bytes
            && self.inference_port != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node() -> NodeAnnouncement {
        NodeAnnouncement {
            version: CONTROL_VERSION,
            nonce: String::new(),
            node_id: "stable-id".into(),
            hostname: "mini".into(),
            os: "macos".into(),
            arch: "aarch64".into(),
            logical_cpus: 10,
            cpu_model: "Apple M3".into(),
            accelerators: vec![AcceleratorInfo {
                name: "Apple M3 GPU".into(),
                vendor: "Apple".into(),
                memory_bytes: None,
                unified_memory: true,
            }],
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            inference_port: 9000,
        }
    }

    #[test]
    fn discovery_and_registration_round_trip() {
        let node = sample_node();
        assert!(node.is_valid());

        for message in [
            ControlMessage::Discover {
                version: CONTROL_VERSION,
                nonce: "test-nonce".into(),
            },
            ControlMessage::Announce(node.clone()),
            ControlMessage::Register { node },
        ] {
            let encoded = serde_json::to_vec(&message).unwrap();
            assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
            let _: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        }
    }

    #[test]
    fn authenticated_tensor_offer_and_reply_round_trip() {
        let offer = TensorOfferMessage {
            version: CONTROL_VERSION,
            sender_node_id: "sender-node".into(),
            sequence: 42,
            message_id: 1234,
            descriptor: TensorDescriptor {
                tensor_id: 9,
                name: "embedding.weight".into(),
                dtype: crate::tensor::TensorDType::F16,
                shape: vec![4096, 4096],
                layout: crate::tensor::TensorLayout::RowMajor,
                encoding: crate::tensor::TensorEncoding::Raw,
                byte_len: 33_554_432,
            },
            auth_tag: vec![7; 32],
        };
        let encoded = serde_json::to_vec(&ControlMessage::TensorOffer { offer }).unwrap();
        assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
        let _: ControlMessage = serde_json::from_slice(&encoded).unwrap();

        let reply = TensorOfferReply {
            version: CONTROL_VERSION,
            receiver_node_id: "receiver-node".into(),
            sender_node_id: "sender-node".into(),
            sequence: 42,
            message_id: 1234,
            tensor_id: 9,
            accepted: true,
            reason: None,
            auth_tag: vec![8; 32],
        };
        let encoded = serde_json::to_vec(&ControlMessage::TensorOfferReply { reply }).unwrap();
        assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
        let _: ControlMessage = serde_json::from_slice(&encoded).unwrap();
    }

    #[test]
    fn rejects_unbounded_or_inconsistent_node_metadata() {
        let mut node = sample_node();
        node.hostname = "x".repeat(129);
        assert!(!node.is_valid());

        let mut node = sample_node();
        node.available_memory_bytes = node.total_memory_bytes + 1;
        assert!(!node.is_valid());

        let mut node = sample_node();
        node.logical_cpus = 0;
        assert!(!node.is_valid());
    }
}
