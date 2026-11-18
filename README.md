# AITP — AI Transport Protocol

AITP is an experimental Rust prototype for transporting tensor-like payloads between machines for distributed LLM inference. It separates a binary UDP data plane from a JSON control plane for LAN discovery and node liveness.

This is an early prototype, not production-ready networking software. Do not expose its unauthenticated UDP services to untrusted networks.

## Requirements

- Stable Rust toolchain
- Machines reachable on the same LAN

## Build and test

    cargo build --release
    cargo test

## Run a cluster manager

Start the manager on one machine:

    cargo run --release -- manager --bind 0.0.0.0:9002 --timeout-ms 5000

Open the read-only cluster dashboard at http://127.0.0.1:9003. The HTTP listener binds to localhost by default. To view it from another trusted LAN machine, explicitly set `--dashboard-bind 0.0.0.0:9003` and allow TCP port 9003 through the manager firewall. Do not expose the dashboard to the public internet.

The dashboard polls `/api/cluster` every two seconds and displays registered nodes, online/offline state, CPU model/core count, total and available RAM, accelerator inventory, node address, and heartbeat age. It has no workload-control or remote-execution endpoints.

The manager accepts node registrations and periodic heartbeats. It prints a cluster inventory and marks nodes OFFLINE when no heartbeat has arrived within the configured timeout.

## Join a node to the manager

On each Mac, Windows, or Linux machine:

    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000 --manager 192.168.1.10:9002 --heartbeat-ms 1000

The node reports CPU model, logical CPU count, total RAM, current available RAM, and best-effort GPU inventory. NVIDIA devices are queried through nvidia-smi; macOS uses system_profiler (Apple Silicon GPUs are marked as sharing unified system memory); Windows uses PowerShell CIM; Linux falls back to lspci. Available RAM is refreshed in each sequence-numbered heartbeat.

Replace 192.168.1.10 with the manager machine's LAN address.

Each node creates a UUID once and stores it locally:
- macOS/Linux: ~/.aitp/node-id
- Windows: %APPDATA%\aitp\node-id

Restarting the agent preserves its node ID. Deleting this file creates a new identity. The node sends a registration on startup and then sequence-numbered heartbeats at the configured interval.

The manager keeps its inventory in memory; restarting the manager clears the current inventory. The manager does not issue remote commands or execute workloads.

## Discover nodes without a manager

Start the node agent:

    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000

Discover from another machine:

    cargo run --release -- discover --target 255.255.255.255:9001 --wait-ms 1500

For networks with a specific subnet broadcast address, use that address instead (for example 192.168.1.255:9001).

Discovery advertises hostname, operating system, CPU architecture, logical CPU count, and inference/data port. Discovery is IPv4 broadcast only. Some Wi-Fi access points block client-to-client traffic or broadcast packets.

## Transfer a tensor stand-in

Start a receiver:

    cargo run --release -- receive --bind 0.0.0.0:9000 --output received.tensor

Send a file from another machine:

    cargo run --release -- send --target 192.168.1.50:9000 --input test-tensor.bin --tensor-id 1

Replace the address with the receiver's LAN IP. Allow inbound UDP port 9000 through the receiver's firewall.

## Data-plane wire format v0.1

All integers are big-endian. Header size is 32 bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | Magic AITP |
| 4 | 1 | Version |
| 5 | 1 | Packet kind: DATA=1, ACK=2, FIN=3, FIN_ACK=4 |
| 6 | 2 | Flags |
| 8 | 4 | Message ID |
| 12 | 4 | Tensor ID |
| 16 | 4 | Sequence number |
| 20 | 2 | Payload length |
| 22 | 2 | Reserved |
| 24 | 4 | CRC32 of payload |
| 28 | 4 | Worker ID |

Datagrams are capped at 1200 bytes to reduce IP fragmentation risk. Payload chunks are at most 1168 bytes.

## Tensor descriptor metadata v0.5

AITP now defines a bounded `TensorDescriptor` schema for the worker-execution milestone. A descriptor carries the tensor ID (matching the existing 32-byte data-plane header), name, logical dtype, shape, layout, storage encoding, and exact byte length. Raw F32/F16/BF16/integer storage is supported, along with initial GGML Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 block-size calculations. Descriptors are validated with checked arithmetic, a maximum rank of 8, and a 1 TiB safety ceiling.

`ControlMessage::TensorOffer` carries an authenticated metadata offer. The node agent verifies its HMAC, descriptor size, sender sequence, and bounded pending-queue capacity, then returns a signed admission reply. The data plane remains separate and the 32-byte header is unchanged; this milestone does not allocate tensor buffers or receive payload bytes.

## Authenticated tensor-offer admission v0.6

A node validates incoming tensor metadata offers when the AITP_CLUSTER_KEY environment variable is configured. Use the same random secret (at least 32 bytes) on the sender and worker. Generate one with openssl rand -hex 32 and set it in both process environments.

macOS/Linux:

    export AITP_CLUSTER_KEY="paste-the-same-secret-on-each-machine"
    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000

Windows PowerShell:

    $env:AITP_CLUSTER_KEY = "paste-the-same-secret-on-each-machine"
    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000

Submit a descriptor from another machine for metadata-only admission:

    cargo run --release -- offer --target 192.168.1.50:9001 --descriptor examples/tensor-descriptor.json

Transfer a matching tensor file in one authenticated operation:

    cargo run --release -- send-tensor --target 192.168.1.50:9001 --descriptor examples/tensor-descriptor.json --input layer-weights.bin

The `send-tensor` command first obtains a signed admission response, then streams the file over the authenticated binary data plane. The descriptor's `byte_len` must exactly match the input file size. The worker's data port defaults to UDP 9000; override it with `--data-target 192.168.1.50:9000` when needed. The sender reuses the same UDP socket for the offer and payload transfer so the worker can bind the data packets to the peer endpoint recorded during admission.

When `AITP_CLUSTER_KEY` is configured, the node binds its data receiver on `--data-port` and stores completed tensors in `AITP_TENSOR_DIR` (default: `aitp-tensors/`). Each binary datagram keeps the existing 32-byte header and appends a 32-byte HMAC-SHA256 tag after the payload. Authenticated payload chunks are sized so the complete datagram remains at or below 1200 bytes.

The worker validates the HMAC, source endpoint, message ID, tensor ID, sequence, exact per-chunk length, CRC32, full chunk coverage, and declared final length. It writes to a bounded temporary file and renames it to a `.tensor` file only after all chunks verify. The current receiver admits one active transfer at a time, enforces a 1 GiB tensor ceiling, and keeps a bounded 256-entry duplicate-FIN response cache for 60 seconds.

This transfers tensor bytes to disk; it does **not** load the tensor into an inference runtime or execute a model layer. The older generic `send`/`receive` commands remain unauthenticated compatibility paths and should only be used on a trusted network.

## Cluster dashboard v0.7

The manager serves a responsive, read-only web dashboard over HTTP. The dashboard is intentionally separate from the UDP tensor data plane and reads a snapshot of the manager's in-memory node inventory. It does not issue commands to nodes, allocate workloads, or execute code. The default bind is localhost; LAN exposure requires an explicit bind-address override. The dashboard currently has no authentication or TLS, so keep it on localhost or a trusted isolated LAN.

## Current limitations

- The legacy generic send/receive path uses batches of up to 512 chunks. The authenticated tensor path uses a fixed 32-packet window with bounded pacing and retries; it is not congestion control.
- No congestion control, adaptive rate control, selective ACK ranges, adaptive MTU, encryption, or multi-transfer multiplexing. The authenticated tensor path uses HMAC-SHA256 per datagram, but UDP retransmission and fixed pacing are still experimental and LAN-only.
- UDP reliability is experimental; this implementation is not a replacement for a mature congestion-controlled transport.
- The authenticated tensor receiver handles one active transfer per worker. The legacy generic receiver writes to one configured output path.
- The legacy generic sender reads its input file into memory. The authenticated `send-tensor` path reads bounded windows from disk.
- Discovery, registration, and heartbeat messages remain unauthenticated. HMAC does not encrypt metadata or protect against a compromised endpoint. Offer replay counters are held in worker memory and reset when that worker restarts. Use only on a trusted LAN.
- Manager inventory is in-memory and is lost when the manager restarts.
- The manager does not authenticate node IDs; a device on the LAN can attempt to register a claimed ID.
- Heartbeat liveness is based on sequence-numbered messages and a configurable timeout, not cryptographic identity.
- The authenticated tensor receiver stages completed payloads to disk; there is not yet a tensor allocator, mmap-backed runtime handoff, model loader, or inference execution path.
- No inference-runtime integration, GPU-direct networking, RDMA, or zero-copy path.
- Hardware inventory includes CPU model, OS, architecture, logical CPU count, total RAM, periodically refreshed available RAM, and best-effort accelerator names/vendors/memory. NVIDIA VRAM is reported when nvidia-smi is installed. Apple Silicon is represented as an integrated GPU sharing unified system memory. Some platform probes identify devices without reliable VRAM data. Compute capability, utilization, and measured throughput are not yet reported.

## Roadmap

1. Add congestion control, pacing, selective acknowledgments, and robust transfer IDs.
2. Add persistent node identities, heartbeats, and disconnect detection. (Implemented in v0.3.)
3. Report CPU model and best-effort GPU/accelerator inventory. (Initial probes added in v0.4.)
4. Define and validate tensor descriptors and control-plane offers. (Schema added in v0.5.)
5. Authenticate and bound worker tensor-offer admission. (Initial metadata-only path added in v0.6.)
6. Bind accepted offers to an authenticated, bounded tensor receive path without changing the 32-byte data-plane header. (Initial disk-staging transfer path and loopback integration test added in v0.8.)
7. Add a runtime handoff and integrate one inference backend (llama.cpp on CPU first), then execute a remote layer on a worker.
8. Implement pipeline partitioning: place contiguous transformer layer ranges on workers, transfer activations, and stream generated tokens.
9. Add automatic memory-aware placement, then GPU/Metal/CUDA acceleration and performance tuning.
10. Add control-plane authentication, congestion control, and benchmarking.
11. Add a cluster dashboard for live node inventory. (Initial read-only dashboard added in v0.7.)

## Security

Discovery, registration, heartbeats, the manager, dashboard, and legacy generic send/receive remain unauthenticated. The tensor offer and authenticated tensor payload path use HMAC-SHA256 with a shared environment-provided key; this provides integrity/authentication but not confidentiality. The receiver enforces a 1 GiB per-tensor ceiling and writes to bounded temporary files. Use a trusted LAN; do not expose these services to the public internet.

## License

MIT
