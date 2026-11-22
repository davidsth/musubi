# Changelog

## v0.13.0 — Expanded GGML quantized tensor transfer descriptors

- Added exact AITP block encodings for Q8_1, Q2_K through Q8_K, and IQ1/IQ2/IQ3/IQ4 variants, including IQ1_M.
- Added raw I16 and F64 descriptor widths and GGUF type-ID mappings.
- Added cross-coverage tests that open synthetic GGUF files, validate supported type descriptors and exact byte lengths, and read selected source ranges back byte-for-byte.
- Bumped the control-plane schema to version 6 for the expanded descriptor variants and the package to 0.13.0.
- Preserved the 32-byte binary data-plane header, authenticated range streaming, bounded transfer path, and fail-closed behavior for unverified types.
- No dequantization, arbitrary strided views, llama.cpp layer execution, workload scheduler, or remote code execution added.

## v0.12.0 — Direct GGUF tensor range transfer

- Added a secure GGUF tensor-source API that parses and validates metadata on the same opened file handle returned for streaming.
- Added exact AITP descriptor mappings for F32/F16/BF16/I8/I32/I64 and Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 tensor formats; unsupported encodings fail closed.
- Added `send-gguf-tensor` to authenticate a worker tensor offer and stream only the selected tensor's validated byte range directly from the original GGUF file.
- Generalized the authenticated sender with a bounded file-range API; ordinary `send-tensor` remains supported.
- Kept the 32-byte binary packet header, authenticated payload framing, worker admission, and 1 GiB per-tensor ceiling unchanged.
- Added tests for raw GGUF range reading, quantized descriptor mapping, unsupported format rejection, and multi-chunk ranged loopback transfer.
- No complete-model loading, tensor extraction copy, llama.cpp execution, layer scheduling, or arbitrary remote execution added.

## v0.11.0 — GGUF metadata inspector and dashboard model catalog

- Added a bounded little-endian GGUF v2/v3 reader for headers, selected architecture metadata, and tensor descriptors.
- Skips tokenizer arrays and tensor payload data without allocating model-sized buffers.
- Calculates storage sizes for a documented subset of GGML tensor formats and reports unknown/newer types without guessing.
- Validates file bounds, metadata/tensor count limits, dimensions, alignment, duplicate names, known tensor ranges, and overlapping known-size tensors.
- Added `inspect-gguf` and `gguf-catalog` CLI commands.
- Added a local `AITP_MODEL_DIR` scanner (16 top-level files per scan) and model cards in the read-only dashboard, refreshed every 30 seconds.
- Added synthetic GGUF fixtures for metadata, tokenizer string arrays, supported/unknown types, metadata-only files, truncation, invalid alignment, and overlapping ranges.
- No GGUF weights are loaded; no model allocator, llama.cpp adapter, workload scheduler, or remote execution was added. The 32-byte tensor data-plane header remains unchanged.

## v0.10.0 — Verified streaming runtime handoff

- Added a read-only runtime handoff API that validates a tensor manifest and opens its payload as a streaming reader.
- Verifies regular-file type, exact byte length, and SHA-256 using the same open file handle returned to the adapter; rewinds the reader to byte zero.
- Rejects final-component symbolic links on Unix and Windows reparse points on Windows.
- Avoids copying the complete tensor into a second in-memory buffer.
- Added the `prepare-tensor` CLI command for bounded sidecar parsing and targeted pre-handoff verification.
- Added tests for reader contents, digest mismatch, and symbolic-link rejection on Unix.
- No llama.cpp integration, model loading, workload scheduler, or remote execution added; the 32-byte data-plane header is unchanged.

## v0.9.0 — Durable tensor manifests and local catalog

- Added versioned JSON sidecars for completed authenticated tensor payloads, including the typed descriptor, sender identity, offer sequence, transfer ID, exact byte length, commit timestamp, payload basename, and SHA-256 digest.
- Publish payload and manifest through same-directory staging/rename steps; the catalog ignores manifests whose payload is absent, not a regular file, or has the wrong length.
- Added a bounded local tensor-store scanner and an explicit SHA-256 verification helper.
- Added the `catalog` CLI command with optional `--verify-hashes` and custom `--directory`.
- Expanded the existing read-only dashboard with recent local tensor artifacts, tensor count, total bytes, descriptor metadata, and digest.
- Clarified that the dashboard catalog scans only the dashboard host's local AITP_TENSOR_DIR; remote node stores are not aggregated.
- Added manifest validation, path-safety, catalog filtering, post-commit mutation detection, and end-to-end manifest assertions.
- No runtime loader, model execution, arbitrary remote commands, or change to the 32-byte data-plane header.

## v0.8.0 — Authenticated tensor payload transfer

- Connected authenticated tensor offers to a worker-side UDP data receiver.
- Added the `send-tensor` CLI path: signed metadata admission followed by authenticated chunk transfer using the same sender socket/peer identity.
- Preserved the existing 32-byte binary header; appended HMAC-SHA256 tags after payloads and kept authenticated datagrams at or below 1200 bytes.
- Added exact file-size, per-chunk length, source endpoint, transfer ID, CRC32, HMAC, sequence-coverage, and final-length checks.
- Added bounded temporary-file staging and final-name commit after complete verification; retained a 1 GiB per-tensor cap and one active transfer per worker.
- Bounded duplicate-FIN response state to 256 completed transfers.
- Added a multi-chunk loopback integration test that verifies committed bytes match the sender input.
- No model loading, tensor allocator, or remote execution was introduced; UDP fixed-window retry is not congestion control.

## v0.7.0 — Read-only cluster dashboard

- Added a responsive web dashboard served by the cluster manager over HTTP.
- Added a read-only JSON snapshot endpoint at /api/cluster with node status, addresses, CPU, RAM, accelerator inventory, and heartbeat age.
- Dashboard refreshes every two seconds and reports online/offline node counts and aggregate CPU/RAM.
- Dashboard binds to 127.0.0.1:9003 by default; LAN exposure requires an explicit --dashboard-bind override.
- Kept dashboard/control-plane HTTP separate from the UDP tensor data plane.
- Added dashboard snapshot serialization and static UI smoke tests.
- No remote execution, workload-control endpoints, or tensor transfer behavior added.

## v0.6.0 — Authenticated tensor-offer admission

- Added HMAC-SHA256 authentication for tensor-offer requests and replies using a shared 32+ byte AITP_CLUSTER_KEY.
- Added worker-side descriptor validation, a 1 GiB per-offer ceiling, a 32-entry pending queue, a 30-second expiry, and per-sender replay-sequence checks.
- Added an offer CLI command that submits a JSON descriptor and verifies the worker's signed reply.
- Persisted offer sequence counters alongside the persistent node identity.
- Kept tensor bytes off the control plane; no payload buffers are allocated and the 32-byte data-plane header is unchanged.
- Bumped the control-plane version to 5.
- Added HMAC, replay, admission-limit, expiry, and control-message serialization tests.

## v0.5.0 — Typed tensor descriptors

- Added bounded tensor metadata with dtype, shape, layout, storage encoding, tensor ID, and exact byte length.
- Added raw tensor sizing plus GGML Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 block-size calculations.
- Added checked arithmetic, rank/name/size limits, and descriptor validation.
- Added a versioned TensorOffer control-plane message; the existing 32-byte binary data-plane header is unchanged.
- Added tests for raw and quantized sizing, invalid dimensions, overflow, incompatible layout/dtype, and control-message round trips.

## v0.4.0 — CPU and accelerator inventory

- Added CPU model reporting.
- Added best-effort NVIDIA GPU detection via nvidia-smi.
- Added macOS system_profiler GPU detection and Apple Silicon unified-memory labeling.
- Added Windows video-controller detection via PowerShell CIM and Linux display-controller detection via lspci.
- Included accelerator inventory in node registration/discovery and manager output.
- Added tests for unified-memory reporting.
- Bumped the control-plane version to 3 for the expanded registration schema.
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
