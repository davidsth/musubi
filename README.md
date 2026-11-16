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

The manager accepts node registrations and periodic heartbeats. It prints a cluster inventory and marks nodes OFFLINE when no heartbeat has arrived within the configured timeout.

## Join a node to the manager

On each Mac, Windows, or Linux machine:

    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000 --manager 192.168.1.10:9002 --heartbeat-ms 1000

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

## Current limitations

- ACK/retry is basic and uses batches of up to 512 chunks.
- No congestion control, packet pacing, authentication, encryption, selective ACK ranges, adaptive MTU, or multi-transfer multiplexing.
- UDP reliability is experimental; this implementation is not a replacement for a mature congestion-controlled transport.
- Receiver handles one active transfer and writes to one output path.
- Sender reads the input file into memory.
- Discovery, registration, and heartbeat messages are unauthenticated. Use only on trusted LANs.
- Manager inventory is in-memory and is lost when the manager restarts.
- The manager does not authenticate node IDs; a device on the LAN can attempt to register a claimed ID.
- Heartbeat liveness is based on sequence-numbered messages and a configurable timeout, not cryptographic identity.
- No tensor schema negotiation, inference-runtime integration, GPU-direct networking, RDMA, or zero-copy path.
- Hardware inventory currently includes only OS, architecture, hostname, and logical CPU count.

## Roadmap

1. Add congestion control, pacing, selective acknowledgments, and robust transfer IDs.
2. Add persistent node identities, heartbeats, and disconnect detection. (Implemented in v0.3.)
3. Report RAM and accelerator capabilities.
4. Persist cluster inventory and add authenticated node enrollment.
5. Add tensor descriptors (dtype, shape, layout, quantization, offsets).
6. Add platform-specific fast paths behind a transport abstraction.
7. Integrate with llama.cpp first, then evaluate MLX on Apple Silicon.
8. Benchmark against TCP, QUIC, and existing distributed inference approaches on 1/2.5/10 GbE.

## Security

The current control plane is unauthenticated and exposes basic machine metadata to devices on the local network. The transfer receiver is also a prototype and should only be used on trusted networks. Authentication, encryption, rate limiting, and robust resource limits are future work.

## License

MIT
