//! Functional tests: round-trip encode/decode, edge cases, error handling.

use yenc_simd::{decode_yenc, encode_article};

// ---------------------------------------------------------------------------
// Round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn test_roundtrip_16_bytes() {
    let data: Vec<u8> = (0..16).collect();
    let (encoded, crc) = encode_article(&data, "small.bin", 1, 1, 0, data.len() as u64);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data, data);
    assert_eq!(result.crc32, crc);
    assert_eq!(result.filename.as_deref(), Some("small.bin"));
}

#[test]
fn test_roundtrip_simd_boundary_32() {
    // Exactly 32 bytes — one full SSE2 vector width.
    let data: Vec<u8> = (0..32).collect();
    let (encoded, crc) = encode_article(&data, "sse2.bin", 1, 1, 0, data.len() as u64);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data, data);
    assert_eq!(result.crc32, crc);
}

#[test]
fn test_roundtrip_simd_boundary_64() {
    // Exactly 64 bytes — one full AVX2 vector width.
    let data: Vec<u8> = (0..64).collect();
    let (encoded, crc) = encode_article(&data, "avx2.bin", 1, 1, 0, data.len() as u64);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data, data);
    assert_eq!(result.crc32, crc);
}

#[test]
fn test_roundtrip_1mb() {
    // 1 MiB payload — exercises sustained SIMD path.
    let data: Vec<u8> = (0..1_048_576).map(|i| (i % 256) as u8).collect();
    let (encoded, crc) = encode_article(&data, "large.bin", 1, 1, 0, data.len() as u64);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data.len(), data.len());
    assert_eq!(result.data, data);
    assert_eq!(result.crc32, crc);
}

#[test]
fn test_roundtrip_all_byte_values() {
    // All 256 byte values — ensures every value round-trips including critical chars.
    let data: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
    let (encoded, crc) = encode_article(&data, "all_bytes.bin", 1, 1, 0, data.len() as u64);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data, data, "all 256 byte values must round-trip");
    assert_eq!(result.crc32, crc);
}

#[test]
fn test_encode_decode_multipart() {
    // Multi-part article: uses =ypart header with begin/end.
    let data = b"Hello, multipart world!";
    let (encoded, crc) = encode_article(data, "multi.bin", 1, 3, 0, 1000);
    let result = decode_yenc(&encoded).unwrap();
    assert_eq!(result.data, data);
    assert_eq!(result.crc32, crc);
    assert_eq!(result.part_number, Some(1));
    // =ypart begin= is 1-indexed in the article, decoder preserves raw value
    assert!(result.part_begin.is_some());
    assert!(result.part_end.is_some());
    assert_eq!(result.file_size, Some(1000));
}

#[test]
fn test_decode_empty_body_between_headers() {
    // Minimal valid yEnc article with empty body.
    let article = b"=ybegin line=128 size=0 name=empty.bin\r\n=yend size=0 crc32=00000000\r\n";
    let result = decode_yenc(article).unwrap();
    assert!(result.data.is_empty());
    assert_eq!(result.filename.as_deref(), Some("empty.bin"));
}

#[test]
fn test_decode_missing_header_errors() {
    let article = b"This is just some random text\r\nwith no yEnc headers\r\n";
    let err = decode_yenc(article).unwrap_err();
    assert!(
        err.to_string().contains("Missing =ybegin"),
        "expected MissingHeader error, got: {err}"
    );
}

#[test]
fn test_decode_missing_footer_errors() {
    let article = b"=ybegin line=128 size=5 name=test.bin\r\nHello\r\n";
    let err = decode_yenc(article).unwrap_err();
    assert!(
        err.to_string().contains("Missing =yend"),
        "expected MissingFooter error, got: {err}"
    );
}

#[test]
fn test_decode_crc_mismatch_errors() {
    // Encode valid article, then corrupt the CRC in the =yend line.
    let data = b"Test data for CRC";
    let (mut encoded, _crc) = encode_article(data, "crc.bin", 1, 1, 0, data.len() as u64);

    // Find "crc32=" in the encoded output (search as bytes, not string — body may be non-UTF8).
    let needle = b"crc32=";
    if let Some(pos) = encoded.windows(needle.len()).position(|w| w == needle) {
        let hex_start = pos + needle.len();
        // Overwrite with a known-bad CRC.
        let bad_crc = b"DEADBEEF";
        encoded[hex_start..hex_start + 8].copy_from_slice(bad_crc);
    }

    let err = decode_yenc(&encoded).unwrap_err();
    assert!(
        err.to_string().contains("CRC32 mismatch"),
        "expected CrcMismatch error, got: {err}"
    );
}
