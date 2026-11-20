//! AITP v0.1 wire format.
//!
//! All integers are network byte order (big endian). Header is exactly 32 bytes.

use anyhow::{bail, Result};

pub const MAGIC: [u8; 4] = *b"AITP";
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 32;
pub const MAX_DATAGRAM: usize = 1200;
pub const MAX_PAYLOAD: usize = MAX_DATAGRAM - HEADER_LEN;
/// Binary tensor datagrams append a 32-byte HMAC-SHA256 tag after the payload.
pub const AUTH_TAG_LEN: usize = 32;
pub const MAX_AUTH_PAYLOAD: usize = MAX_DATAGRAM - HEADER_LEN - AUTH_TAG_LEN;

pub const KIND_DATA: u8 = 1;
pub const KIND_ACK: u8 = 2;
pub const KIND_FIN: u8 = 3;
pub const KIND_FIN_ACK: u8 = 4;

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub kind: u8,
    pub flags: u16,
    pub message_id: u32,
    pub tensor_id: u32,
    pub sequence: u32,
    pub payload_len: u16,
    pub checksum: u32,
    pub worker_id: u32,
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = VERSION;
        out[5] = self.kind;
        out[6..8].copy_from_slice(&self.flags.to_be_bytes());
        out[8..12].copy_from_slice(&self.message_id.to_be_bytes());
        out[12..16].copy_from_slice(&self.tensor_id.to_be_bytes());
        out[16..20].copy_from_slice(&self.sequence.to_be_bytes());
        out[20..22].copy_from_slice(&self.payload_len.to_be_bytes());
        // bytes 22..24 reserved for future use
        out[24..28].copy_from_slice(&self.checksum.to_be_bytes());
        out[28..32].copy_from_slice(&self.worker_id.to_be_bytes());
        out
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_LEN {
            bail!("short AITP packet");
        }
        if buf[0..4] != MAGIC {
            bail!("invalid AITP magic");
        }
        if buf[4] != VERSION {
            bail!("unsupported AITP version: {}", buf[4]);
        }
        Ok(Self {
            kind: buf[5],
            flags: u16::from_be_bytes([buf[6], buf[7]]),
            message_id: u32::from_be_bytes(buf[8..12].try_into()?),
            tensor_id: u32::from_be_bytes(buf[12..16].try_into()?),
            sequence: u32::from_be_bytes(buf[16..20].try_into()?),
            payload_len: u16::from_be_bytes(buf[20..22].try_into()?),
            checksum: u32::from_be_bytes(buf[24..28].try_into()?),
            worker_id: u32::from_be_bytes(buf[28..32].try_into()?),
        })
    }
}

pub fn packet(header: Header, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(payload);
    out
}

pub fn checksum(payload: &[u8]) -> u32 {
    crc32fast::hash(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_payload_stays_within_safe_datagram_size() {
        assert_eq!(HEADER_LEN, 32);
        assert_eq!(HEADER_LEN + MAX_AUTH_PAYLOAD + AUTH_TAG_LEN, MAX_DATAGRAM);
        assert!(MAX_AUTH_PAYLOAD < MAX_PAYLOAD);
    }

    #[test]
    fn header_is_exactly_32_bytes_and_round_trips() {
        let h = Header {
            kind: KIND_DATA,
            flags: 7,
            message_id: 10,
            tensor_id: 20,
            sequence: 30,
            payload_len: 99,
            checksum: 1234,
            worker_id: 40,
        };
        let bytes = h.encode();
        assert_eq!(bytes.len(), 32);
        let decoded = Header::decode(&bytes).unwrap();
        assert_eq!(decoded.message_id, 10);
        assert_eq!(decoded.tensor_id, 20);
        assert_eq!(decoded.sequence, 30);
        assert_eq!(decoded.checksum, 1234);
    }
}
