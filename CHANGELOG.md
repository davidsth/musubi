# Changelog

## v0.3.0 — Persistent identity and cluster heartbeats

- Persisted a UUID node identity locally across agent restarts.
- Added manager registration and sequence-numbered heartbeat control messages.
- Added a cluster manager that tracks registered nodes and marks them OFFLINE after a configurable heartbeat timeout.
- Kept control-plane messages separate from the binary tensor data plane.
- Added control-message round-trip tests for registration and heartbeat messages.
- Documented manager/node startup, identity storage locations, firewall requirements, and security limitations.

## v0.2 — LAN discovery

- Added a separate Serde/JSON control-plane message model in src/control.rs.
- Added the node command as a lightweight UDP discovery responder.
- Added the discover command for IPv4 broadcast discovery with nonce matching and duplicate suppression.
- Added node metadata for hostname, OS, architecture, logical CPU count, and inference port.
- Added control-message serialization tests and documented firewall, broadcast, and security limitations.
- Preserved the v0.1 UDP tensor data-plane wire format and sender/receiver commands.

## v0.1 — Initial transport prototype

- Added a fixed 32-byte binary packet header.
- Added chunked UDP file/tensor-stand-in transfers.
- Added per-chunk CRC32 checks, acknowledgments, retries, and a final transfer confirmation.
- Added sender and receiver CLI commands.
