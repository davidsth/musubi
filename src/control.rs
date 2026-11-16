//! Versioned control-plane messages. Tensor payloads remain on the binary data plane.

use serde::{Deserialize, Serialize};

pub const CONTROL_VERSION: u8 = 2;
pub const MAX_CONTROL_DATAGRAM: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    Discover { version: u8, nonce: String },
    Announce(NodeAnnouncement),
    Register { node: NodeAnnouncement },
    Heartbeat {
        version: u8,
        node_id: String,
        sequence: u64,
        available_memory_bytes: u64,
    },
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
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub inference_port: u16,
}

impl NodeAnnouncement {
    /// Validate bounded metadata before a manager accepts an untrusted LAN registration.
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
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            inference_port: 9000,
        }
    }

    #[test]
    fn discovery_message_round_trips() {
        let message = ControlMessage::Discover {
            version: CONTROL_VERSION,
            nonce: "test-nonce".into(),
        };
        let encoded = serde_json::to_vec(&message).unwrap();
        let decoded: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(
            decoded,
            ControlMessage::Discover {
                version: CONTROL_VERSION,
                ..
            }
        ));
    }

    #[test]
    fn announcement_and_registration_round_trip() {
        let node = sample_node();
        assert!(node.is_valid());
        let encoded = serde_json::to_vec(&ControlMessage::Announce(node.clone())).unwrap();
        let decoded: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(decoded, ControlMessage::Announce(_)));
        assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);

        let registration = serde_json::to_vec(&ControlMessage::Register { node }).unwrap();
        assert!(matches!(
            serde_json::from_slice::<ControlMessage>(&registration).unwrap(),
            ControlMessage::Register { .. }
        ));
    }

    #[test]
    fn heartbeat_carries_live_available_memory() {
        let heartbeat = ControlMessage::Heartbeat {
            version: CONTROL_VERSION,
            node_id: "stable-id".into(),
            sequence: 42,
            available_memory_bytes: 4 * 1024 * 1024 * 1024,
        };
        let encoded = serde_json::to_vec(&heartbeat).unwrap();
        let decoded: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(
            decoded,
            ControlMessage::Heartbeat {
                sequence: 42,
                available_memory_bytes,
                ..
            } if available_memory_bytes == 4 * 1024 * 1024 * 1024
        ));
    }

    #[test]
    fn rejects_unbounded_or_inconsistent_metadata() {
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
