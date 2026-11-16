# Changelog

## v0.4.0 — CPU and accelerator inventory

- Added CPU model reporting.
- Added best-effort NVIDIA GPU detection via nvidia-smi.
- Added macOS system_profiler GPU detection and Apple Silicon unified-memory labeling.
- Added Windows video-controller detection via PowerShell CIM and Linux display-controller detection via lspci.
- Included accelerator inventory in node registration/discovery and manager output.
- Added tests for unified-memory reporting.
- Probes fail gracefully when optional vendor tools are unavailable.

## v0.3.1 — Hardware memory inventory

- Added cross-platform total and available system RAM reporting through sysinfo.
- Included available RAM in sequence-numbered heartbeats so the manager sees changing memory pressure.
- Bumped the control-plane protocol version to 2 for the expanded node/heartbeat schema.
- Added metadata validation and a 256-node registration cap to limit accidental or hostile LAN registry growth.
- Retained offline node entries for a bounded period, then pruned them.
- Added serialization and validation tests for memory metadata and heartbeats.

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
