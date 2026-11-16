//! AITP control-plane messages. Kept separate from the binary tensor data plane.

use serde::{Deserialize, Serialize};

pub const CONTROL_VERSION: u8 = 1;
pub const MAX_CONTROL_DATAGRAM: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlMessage {
    Discover { version: u8, nonce: String },
    Announce(NodeAnnouncement),
    Register { node: NodeAnnouncement },
    Heartbeat { version: u8, node_id: String, sequence: u64 },
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
    pub inference_port: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node() -> NodeAnnouncement {
        NodeAnnouncement {
            version: CONTROL_VERSION,
            nonce: "n".into(),
            node_id: "stable-id".into(),
            hostname: "mini".into(),
            os: "macos".into(),
            arch: "aarch64".into(),
            logical_cpus: 10,
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
        assert!(matches!(decoded, ControlMessage::Discover { version: CONTROL_VERSION, .. }));
    }

    #[test]
    fn announcement_round_trips() {
        let encoded = serde_json::to_vec(&ControlMessage::Announce(sample_node())).unwrap();
        let decoded: ControlMessage = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(decoded, ControlMessage::Announce(_)));
        assert!(encoded.len() <= MAX_CONTROL_DATAGRAM);
    }

    #[test]
    fn registration_and_heartbeat_round_trip() {
        let register = ControlMessage::Register { node: sample_node() };
        let encoded = serde_json::to_vec(&register).unwrap();
        assert!(matches!(serde_json::from_slice::<ControlMessage>(&encoded).unwrap(), ControlMessage::Register { .. }));

        let heartbeat = ControlMessage::Heartbeat {
            version: CONTROL_VERSION,
            node_id: "stable-id".into(),
            sequence: 42,
        };
        let encoded = serde_json::to_vec(&heartbeat).unwrap();
        assert!(matches!(serde_json::from_slice::<ControlMessage>(&encoded).unwrap(), ControlMessage::Heartbeat { sequence: 42, .. }));
    }
}
