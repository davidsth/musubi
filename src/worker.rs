//! Bounded, authenticated admission for remote tensor metadata offers.
//! This module reserves metadata slots only; it does not allocate tensor buffers.

use std::{
    collections::HashMap,
    net::SocketAddr,
    time::{Duration, Instant},
};

use crate::{
    auth,
    control::{TensorOfferMessage, CONTROL_VERSION},
};

pub const DEFAULT_MAX_OFFER_BYTES: u64 = 1024 * 1024 * 1024;
pub const DEFAULT_MAX_PENDING_OFFERS: usize = 32;
pub const MAX_AUTHENTICATED_SENDERS: usize = 256;
pub const DEFAULT_OFFER_TTL: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct PendingTensorOffer {
    pub offer: TensorOfferMessage,
    pub peer: SocketAddr,
    pub accepted_at: Instant,
}

#[derive(Debug)]
pub struct OfferAdmission {
    pending: HashMap<(String, u32), PendingTensorOffer>,
    last_sequence: HashMap<String, u64>,
    max_tensor_bytes: u64,
    max_pending: usize,
    ttl: Duration,
}

impl Default for OfferAdmission {
    fn default() -> Self {
        Self::new(
            DEFAULT_MAX_OFFER_BYTES,
            DEFAULT_MAX_PENDING_OFFERS,
            DEFAULT_OFFER_TTL,
        )
    }
}

impl OfferAdmission {
    pub fn new(max_tensor_bytes: u64, max_pending: usize, ttl: Duration) -> Self {
        Self {
            pending: HashMap::new(),
            last_sequence: HashMap::new(),
            max_tensor_bytes,
            max_pending,
            ttl,
        }
    }

    /// Authenticate and admit an offer without allocating any tensor storage.
    pub fn admit(
        &mut self,
        offer: &TensorOfferMessage,
        peer: SocketAddr,
        key: &[u8],
        now: Instant,
    ) -> Result<(), &'static str> {
        self.expire(now);

        if offer.version != CONTROL_VERSION {
            return Err("unsupported tensor-offer version");
        }
        if offer.sender_node_id.is_empty() || offer.sender_node_id.len() > 64 {
            return Err("invalid sender node identity");
        }
        if offer.sequence == 0 || offer.message_id == 0 {
            return Err("offer sequence and message ID must be nonzero");
        }
        if !auth::verify_tensor_offer(offer, key)
            .map_err(|_| "could not verify tensor-offer authentication")?
        {
            return Err("invalid tensor-offer authentication");
        }
        offer
            .descriptor
            .validate()
            .map_err(|_| "invalid tensor descriptor")?;
        if offer.descriptor.byte_len > self.max_tensor_bytes {
            return Err("tensor exceeds worker admission limit");
        }

        let sender = offer.sender_node_id.clone();
        let pending_key = (sender.clone(), offer.message_id);
        if self.pending.contains_key(&pending_key) {
            return Err("duplicate pending message ID");
        }
        if self
            .last_sequence
            .get(&sender)
            .is_some_and(|last| offer.sequence <= *last)
        {
            return Err("replayed or out-of-order offer sequence");
        }
        if self.pending.len() >= self.max_pending {
            return Err("worker pending-offer capacity is full");
        }
        if !self.last_sequence.contains_key(&sender)
            && self.last_sequence.len() >= MAX_AUTHENTICATED_SENDERS
        {
            return Err("worker authenticated-sender capacity is full");
        }

        self.pending.insert(
            pending_key,
            PendingTensorOffer {
                offer: offer.clone(),
                peer,
                accepted_at: now,
            },
        );
        self.last_sequence.insert(sender, offer.sequence);
        Ok(())
    }

    /// Remove timed-out metadata reservations. No tensor payload files are involved.
    pub fn expire(&mut self, now: Instant) -> usize {
        let before = self.pending.len();
        self.pending
            .retain(|_, pending| now.saturating_duration_since(pending.accepted_at) < self.ttl);
        before - self.pending.len()
    }

    /// Locate a pending reservation by the observed UDP peer and binary transfer IDs.
    /// Ambiguous matches are rejected rather than guessing which offer owns the bytes.
    pub fn pending_for_peer(
        &self,
        peer: SocketAddr,
        message_id: u32,
        tensor_id: u32,
    ) -> Option<PendingTensorOffer> {
        let mut matches = self.pending.values().filter(|pending| {
            pending.peer == peer
                && pending.offer.message_id == message_id
                && pending.offer.descriptor.tensor_id == tensor_id
        });
        let first = matches.next()?.clone();
        if matches.next().is_some() {
            None
        } else {
            Some(first)
        }
    }

    /// Consume the metadata reservation once the corresponding tensor is committed.
    pub fn remove_pending(&mut self, sender_node_id: &str, message_id: u32) -> Option<PendingTensorOffer> {
        self.pending.remove(&(sender_node_id.to_owned(), message_id))
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn pending_for(&self, sender_node_id: &str, message_id: u32) -> Option<&PendingTensorOffer> {
        self.pending.get(&(sender_node_id.to_owned(), message_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        auth::sign_tensor_offer,
        control::TensorOfferMessage,
        tensor::{TensorDescriptor, TensorDType, TensorEncoding, TensorLayout},
    };

    fn signed_offer(key: &[u8], sender: &str, sequence: u64, message_id: u32, bytes: u64) -> TensorOfferMessage {
        let mut offer = TensorOfferMessage {
            version: CONTROL_VERSION,
            sender_node_id: sender.into(),
            sequence,
            message_id,
            descriptor: TensorDescriptor {
                tensor_id: 7,
                name: "hidden_states".into(),
                dtype: TensorDType::U8,
                shape: vec![bytes],
                layout: TensorLayout::RowMajor,
                encoding: TensorEncoding::Raw,
                byte_len: bytes,
            },
            auth_tag: Vec::new(),
        };
        sign_tensor_offer(&mut offer, key).unwrap();
        offer
    }

    fn peer() -> SocketAddr {
        "127.0.0.1:40000".parse().unwrap()
    }

    #[test]
    fn admits_authenticated_offer_and_remembers_peer() {
        let key = [0x42; 32];
        let now = Instant::now();
        let offer = signed_offer(&key, "sender-a", 1, 101, 64);
        let mut admission = OfferAdmission::new(1024, 4, Duration::from_secs(30));
        assert!(admission.admit(&offer, peer(), &key, now).is_ok());
        assert_eq!(admission.pending_count(), 1);
        assert_eq!(admission.pending_for("sender-a", 101).unwrap().peer, peer());
    }

    #[test]
    fn rejects_bad_authentication_and_replayed_sequences() {
        let key = [0x42; 32];
        let now = Instant::now();
        let mut admission = OfferAdmission::default();
        let mut tampered = signed_offer(&key, "sender-a", 1, 101, 64);
        tampered.descriptor.name = "tampered".into();
        assert!(admission.admit(&tampered, peer(), &key, now).is_err());

        let valid = signed_offer(&key, "sender-a", 2, 102, 64);
        admission.admit(&valid, peer(), &key, now).unwrap();
        assert!(admission.admit(&valid, peer(), &key, now).is_err());
    }

    #[test]
    fn enforces_tensor_and_pending_limits() {
        let key = [0x42; 32];
        let now = Instant::now();
        let mut admission = OfferAdmission::new(64, 1, Duration::from_secs(30));
        let too_large = signed_offer(&key, "sender-a", 1, 101, 128);
        assert!(admission.admit(&too_large, peer(), &key, now).is_err());

        let first = signed_offer(&key, "sender-a", 2, 102, 32);
        admission.admit(&first, peer(), &key, now).unwrap();
        let second = signed_offer(&key, "sender-b", 1, 103, 32);
        assert!(admission.admit(&second, peer(), &key, now).is_err());
    }

    #[test]
    fn pending_offer_is_bound_to_peer_and_transfer_ids() {
        let key = [0x42; 32];
        let now = Instant::now();
        let offer = signed_offer(&key, "sender-a", 1, 101, 64);
        let mut admission = OfferAdmission::default();
        admission.admit(&offer, peer(), &key, now).unwrap();
        assert!(admission.pending_for_peer(peer(), 101, 7).is_some());
        assert!(admission.pending_for_peer(peer(), 102, 7).is_none());
        assert!(admission.pending_for_peer(peer(), 101, 8).is_none());
        assert!(admission.pending_for_peer("127.0.0.1:40001".parse().unwrap(), 101, 7).is_none());
        assert!(admission.remove_pending("sender-a", 101).is_some());
        assert_eq!(admission.pending_count(), 0);
    }

    #[test]
    fn expires_pending_offer_but_retains_replay_counter() {
        let key = [0x42; 32];
        let now = Instant::now();
        let mut admission = OfferAdmission::new(1024, 1, Duration::from_secs(1));
        let offer = signed_offer(&key, "sender-a", 10, 110, 32);
        admission.admit(&offer, peer(), &key, now).unwrap();
        assert_eq!(admission.expire(now + Duration::from_secs(2)), 1);
        assert_eq!(admission.pending_count(), 0);
        assert!(admission.admit(&offer, peer(), &key, now + Duration::from_secs(2)).is_err());
    }
}
