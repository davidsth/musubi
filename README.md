# AITP — AI Transport Protocol

AITP is an experimental Rust prototype for transporting tensor-like payloads between machines for distributed LLM inference. The initial implementation separates a binary UDP data plane from a small JSON control plane for LAN discovery.

This is an early prototype, not production-ready networking software. Do not expose its unauthenticated UDP services to untrusted networks.

## Requirements

- Stable Rust toolchain
- Machines reachable on the same LAN

## Build

    cargo build --release
    cargo test

## Transfer a tensor stand-in

Start a receiver:

    cargo run --release -- receive --bind 0.0.0.0:9000 --output received.tensor

Send a file from another machine:

    cargo run --release -- send --target 192.168.1.50:9000 --input test-tensor.bin --tensor-id 1

Replace the address with the receiver's LAN IP. Allow inbound UDP port 9000 through the receiver's firewall.

## Discover nodes

Start the node agent on each machine:

    cargo run --release -- node --bind 0.0.0.0:9001 --data-port 9000

Discover nodes from another machine:

    cargo run --release -- discover --target 255.255.255.255:9001 --wait-ms 1500

For networks with a specific subnet broadcast address, use that address instead (for example 192.168.1.255:9001).

The node agent advertises a per-process node ID, hostname, operating system, CPU architecture, logical CPU count, and inference/data port. It only answers discovery requests. It does not accept commands, execute code, or transfer model data.

Allow inbound UDP port 9001 on each node's firewall. Some Wi-Fi access points block client-to-client traffic or broadcast packets. Discovery is IPv4-only.

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
- Discovery is unauthenticated and IPv4 broadcast only. The nonce correlates responses but is not authentication.
- No tensor schema negotiation, inference-runtime integration, GPU-direct networking, RDMA, or zero-copy path.
- No hardware inventory beyond basic OS, architecture, hostname, and logical CPU count.

## Roadmap

1. Add congestion control, pacing, selective acknowledgments, and robust transfer IDs.
2. Add persistent node identities, heartbeats, and disconnect detection.
3. Report RAM and accelerator capabilities.
4. Add tensor descriptors (dtype, shape, layout, quantization, offsets).
5. Add platform-specific fast paths behind a transport abstraction.
6. Integrate with llama.cpp first, then evaluate MLX on Apple Silicon.
7. Benchmark against TCP, QUIC, and existing distributed inference approaches on 1/2.5/10 GbE.

## Security

The current node discovery service is unauthenticated and exposes basic machine metadata to devices on the local broadcast domain. The transfer receiver is also a prototype and should only be used on trusted networks. Authentication, encryption, rate limiting, and robust resource limits are future work.

## License

MIT
