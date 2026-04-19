# yenc-simd

Fast yEnc encoder/decoder with SIMD acceleration (SSE2/AVX2).

yEnc is the encoding used to transfer binary files over Usenet (NNTP). This crate provides a high-throughput decoder with CRC32 verification, falling back to scalar code on platforms without SIMD support.

## Features

- SIMD-accelerated decoding (SSE2/AVX2 on x86/x86_64)
- Scalar fallback for all other platforms
- CRC32 verification
- Supports single-part and multi-part yEnc

## Usage

```toml
[dependencies]
yenc-simd = "0.1"
```

## License

MIT
