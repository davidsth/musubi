//! Versioned control-plane messages. Tensor payloads remain on the binary data plane.

use crate::tensor::TensorDescriptor;
use serde::{Deserialize, Serialize};

pub const CONTROL_VERSION: u8 = 9;
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
        /// Random UUID generated for this process lifetime.
        session_id: String,
        /// HMAC-SHA256 over the announcement and session ID.
        auth_tag: Vec<u8>,
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
        /// Must match the authenticated registration's process session.
        session_id: String,
        sequence: u64,
        /// Raw OS-reported available RAM before logical AITP reservations.
        available_memory_bytes: u64,
        /// Sum of active, unexpired AITP memory-reservation leases.
        reserved_memory_bytes: u64,
        /// HMAC-SHA256 over all heartbeat fields.
        auth_tag: Vec<u8>,
    },
    AllocationPrepare {
        request: AllocationPrepare,
    },
    AllocationReply {
        reply: AllocationReply,
    },
    AllocationRelease {
        request: AllocationRelease,
    },
    AllocationReleaseReply {
        reply: AllocationReleaseReply,
    },
    AllocationStatus {
        request: AllocationStatusRequest,
    },
    AllocationStatusReply {
        reply: AllocationStatusReply,
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

/// Authenticated request to reserve logical RAM for a contiguous GGUF layer range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPrepare {
    pub version: u8,
    pub coordinator_node_id: String,
    pub target_node_id: String,
    pub target_session_id: String,
    pub lease_id: String,
    pub model_sha256: String,
    pub first_layer: Option<u64>,
    pub end_layer_exclusive: Option<u64>,
    pub reserved_bytes: u64,
    pub expires_at_unix: u64,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationReply {
    pub version: u8,
    pub coordinator_node_id: String,
    pub node_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub accepted: bool,
    pub reason: Option<String>,
    pub reserved_bytes: u64,
    pub expires_at_unix: u64,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRelease {
    pub version: u8,
    pub coordinator_node_id: String,
    pub target_node_id: String,
    pub target_session_id: String,
    pub lease_id: String,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationReleaseReply {
    pub version: u8,
    pub coordinator_node_id: String,
    pub node_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub released: bool,
    pub auth_tag: Vec<u8>,
}

/// Authenticated read-only query for the state of one logical allocation lease.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStatusRequest {
    pub version: u8,
    pub coordinator_node_id: String,
    pub target_node_id: String,
    pub target_session_id: String,
    pub lease_id: String,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStatusReply {
    pub version: u8,
    pub coordinator_node_id: String,
    pub node_id: String,
    /// Current worker process session, which may differ from the recorded target session.
    pub session_id: String,
    pub lease_id: String,
    pub active: bool,
    /// Zero when the lease is not active on this process session.
    pub reserved_bytes: u64,
    /// Zero when the lease is not active on this process session.
    pub expires_at_unix: u64,
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
    fn discovery_registration_and_heartbeat_round_trip() {
        let node = sample_node();
        assert!(node.is_valid());

        let mut registered_node = node.clone();
        registered_node.nonce.clear();
        let session_id = "d290f1ee-6c54-4b01-90e6-d701748f0851".to_string();
        for message in [
            ControlMessage::Discover {
                version: CONTROL_VERSION,
                nonce: "test-nonce".into(),
            },
            ControlMessage::Announce(node.clone()),
            ControlMessage::Register {
                node: registered_node,
                session_id: session_id.clone(),
                auth_tag: vec![7; 32],
            },
            ControlMessage::Heartbeat {
                version: CONTROL_VERSION,
                node_id: "stable-id".into(),
                session_id,
                sequence: 1,
                available_memory_bytes: 8 * 1024 * 1024 * 1024,
                reserved_memory_bytes: 1024,
                auth_tag: vec![8; 32],
            },
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
    fn allocation_control_messages_fit_the_control_datagram() {
        let request = AllocationPrepare {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            model_sha256: "a".repeat(64),
            first_layer: Some(0),
            end_layer_exclusive: Some(8),
            reserved_bytes: 1024 * 1024,
            expires_at_unix: 1_800_000_000,
            auth_tag: vec![9; 32],
        };
        let reply = AllocationReply {
            version: CONTROL_VERSION,
            coordinator_node_id: request.coordinator_node_id.clone(),
            node_id: request.target_node_id.clone(),
            session_id: request.target_session_id.clone(),
            lease_id: request.lease_id.clone(),
            accepted: true,
            reason: None,
            reserved_bytes: request.reserved_bytes,
            expires_at_unix: request.expires_at_unix,
            auth_tag: vec![10; 32],
        };
        let release = AllocationRelease {
            version: CONTROL_VERSION,
            coordinator_node_id: request.coordinator_node_id.clone(),
            target_node_id: request.target_node_id.clone(),
            target_session_id: request.target_session_id.clone(),
            lease_id: request.lease_id.clone(),
            auth_tag: vec![11; 32],
        };
        let release_reply = AllocationReleaseReply {
            version: CONTROL_VERSION,
            coordinator_node_id: request.coordinator_node_id.clone(),
            node_id: request.target_node_id.clone(),
            session_id: request.target_session_id.clone(),
            lease_id: request.lease_id.clone(),
            released: true,
            auth_tag: vec![12; 32],
        };

        for message in [
            ControlMessage::AllocationPrepare { request },
            ControlMessage::AllocationReply { reply },
            ControlMessage::AllocationRelease { request: release },
            ControlMessage::AllocationReleaseReply {
                reply: release_reply,
            },
        ] {
            let encoded = serde_json::to_vec(&message).unwrap();
            assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
            let _: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        }
    }

    #[test]
    fn allocation_status_messages_fit_the_control_datagram() {
        let request = AllocationStatusRequest {
            version: CONTROL_VERSION,
            coordinator_node_id: "00000000-0000-4000-8000-000000000001".into(),
            target_node_id: "00000000-0000-4000-8000-000000000002".into(),
            target_session_id: "00000000-0000-4000-8000-000000000003".into(),
            lease_id: "00000000-0000-4000-8000-000000000004".into(),
            auth_tag: vec![1; 32],
        };
        let reply = AllocationStatusReply {
            version: CONTROL_VERSION,
            coordinator_node_id: request.coordinator_node_id.clone(),
            node_id: request.target_node_id.clone(),
            session_id: request.target_session_id.clone(),
            lease_id: request.lease_id.clone(),
            active: true,
            reserved_bytes: 1024 * 1024,
            expires_at_unix: 1_800_000_000,
            auth_tag: vec![2; 32],
        };
        for message in [
            ControlMessage::AllocationStatus { request },
            ControlMessage::AllocationStatusReply { reply },
        ] {
            let encoded = serde_json::to_vec(&message).unwrap();
            assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
            let _: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        }
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
