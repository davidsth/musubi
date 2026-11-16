# Changelog

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
