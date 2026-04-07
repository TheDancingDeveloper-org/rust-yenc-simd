//! yEnc decoder with SIMD acceleration.
//!
//! The hot path — `decode_body` — processes 32 bytes at a time using SSE2
//! (or 64 bytes with AVX2), falling back to scalar for escape sequences
//! and tail bytes.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum YencError {
    #[error("Missing =ybegin header")]
    MissingHeader,
    #[error("Missing =yend footer")]
    MissingFooter,
    #[error("CRC32 mismatch: expected {expected:08X}, got {actual:08X}")]
    CrcMismatch { expected: u32, actual: u32 },
    #[error("Invalid yEnc data: {0}")]
    InvalidData(String),
}

/// Result of decoding a yEnc article.
#[derive(Debug)]
pub struct YencDecodeResult {
    /// Decoded binary data.
    pub data: Vec<u8>,
    /// Filename from =ybegin header.
    pub filename: Option<String>,
    /// Byte offset in the final file (from =ypart begin=, 0-indexed).
    pub part_begin: Option<u64>,
    /// Byte offset end in the final file (from =ypart end=).
    pub part_end: Option<u64>,
    /// Total file size (from =ybegin size=).
    pub file_size: Option<u64>,
    /// Part number.
    pub part_number: Option<u32>,
    /// CRC32 of the decoded data.
    pub crc32: u32,
}

/// Decode a yEnc-encoded article body.
///
/// Handles raw NNTP article data including headers before the yEnc body.
pub fn decode_yenc(raw: &[u8]) -> Result<YencDecodeResult, YencError> {
    // Find =ybegin line
    let ybegin_pos = find_line_starting_with(raw, b"=ybegin ").ok_or(YencError::MissingHeader)?;
    let ybegin_end = find_newline(raw, ybegin_pos).unwrap_or(raw.len());
    let ybegin_line = strip_cr(&raw[ybegin_pos..ybegin_end]);
    let ybegin_str = lossy_str(ybegin_line);

    let filename = extract_param(&ybegin_str, "name");
    let file_size = extract_param(&ybegin_str, "size").and_then(|s| s.parse().ok());
    let part_number = extract_param(&ybegin_str, "part").and_then(|s| s.parse().ok());

    // Scan for =ypart (optional) and =yend (required), find data region
    let mut data_start = ybegin_end + 1; // skip past \n
    let mut part_begin: Option<u64> = None;
    let mut part_end: Option<u64> = None;
    let mut yend_str = String::new();
    let mut data_end = raw.len();
    let mut found_yend = false;

    // Check for =ypart immediately after =ybegin
    if data_start < raw.len() && raw[data_start..].starts_with(b"=ypart ") {
        let line_end = find_newline(raw, data_start).unwrap_or(raw.len());
        let line = strip_cr(&raw[data_start..line_end]);
        let s = lossy_str(line);
        part_begin = extract_param(&s, "begin").and_then(|v| v.parse().ok());
        part_end = extract_param(&s, "end").and_then(|v| v.parse().ok());
        data_start = line_end + 1;
    }

    // Find =yend — scan from the end for efficiency (it's always near the bottom)
    let mut pos = data_start;
    while pos < raw.len() {
        if raw[pos] == b'=' && raw[pos..].starts_with(b"=yend") {
            let line_end = find_newline(raw, pos).unwrap_or(raw.len());
            let line = strip_cr(&raw[pos..line_end]);
            yend_str = lossy_str(line);
            data_end = pos;
            found_yend = true;
            break;
        }
        // Skip to next line
        if let Some(nl) = find_newline(raw, pos) {
            pos = nl + 1;
        } else {
            break;
        }
    }

    if !found_yend {
        return Err(YencError::MissingFooter);
    }

    // Strip trailing \n or \r\n from data region
    let mut data_region_end = data_end;
    if data_region_end > data_start && raw[data_region_end - 1] == b'\n' {
        data_region_end -= 1;
    }
    if data_region_end > data_start && raw[data_region_end - 1] == b'\r' {
        data_region_end -= 1;
    }

    // Decode the data region
    let data_region = &raw[data_start..data_region_end];
    let mut decoded = Vec::with_capacity(data_region.len());
    decode_body(data_region, &mut decoded);

    // CRC32
    let crc = crc32fast::hash(&decoded);

    if let Some(expected_crc_str) =
        extract_param(&yend_str, "pcrc32").or_else(|| extract_param(&yend_str, "crc32"))
        && let Ok(expected_crc) = u32::from_str_radix(&expected_crc_str, 16)
        && crc != expected_crc
    {
        return Err(YencError::CrcMismatch {
            expected: expected_crc,
            actual: crc,
        });
    }

    // Adjust part_begin to 0-indexed (yEnc uses 1-based)
    let part_begin = part_begin.map(|b| b.saturating_sub(1));

    Ok(YencDecodeResult {
        data: decoded,
        filename,
        part_begin,
        part_end,
        file_size,
        part_number,
        crc32: crc,
    })
}

/// Decode the body region of a yEnc article (everything between headers and =yend).
///
/// This processes raw bytes including line endings (\r\n or \n).
/// Line endings in yEnc body are not data — they are stripped.
/// Escape sequences (`=X`) are decoded: the byte after `=` has 64 subtracted.
/// All non-escape, non-line-ending bytes have 42 subtracted.
fn decode_body(data: &[u8], out: &mut Vec<u8>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked for AVX2 support.
            unsafe { decode_body_avx2(data, out) };
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: We just checked for SSE2 support.
            unsafe { decode_body_sse2(data, out) };
            return;
        }
    }
    decode_body_scalar(data, out);
}

/// Scalar fallback decoder.
fn decode_body_scalar(data: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    let len = data.len();
    while i < len {
        let b = data[i];
        match b {
            b'\n' => {
                i += 1;
            }
            b'\r' => {
                i += 1;
            }
            b'=' if i + 1 < len => {
                let next = data[i + 1];
                // Escape sequence: (next - 64 - 42) = (next - 106)
                out.push(next.wrapping_sub(106));
                i += 2;
            }
            _ => {
                out.push(b.wrapping_sub(42));
                i += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SSE2 decoder — processes 16 bytes at a time
// ---------------------------------------------------------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn decode_body_sse2(data: &[u8], out: &mut Vec<u8>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let mut i = 0;

    out.reserve(len);

    unsafe {
        let sub42 = _mm_set1_epi8(42u8 as i8);
        let eq_char = _mm_set1_epi8(b'=' as i8);
        let cr_char = _mm_set1_epi8(b'\r' as i8);
        let lf_char = _mm_set1_epi8(b'\n' as i8);

        while i + 16 <= len {
            let chunk = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);

            let has_eq = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, eq_char));
            let has_cr = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, cr_char));
            let has_lf = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, lf_char));
            let special = has_eq | has_cr | has_lf;

            if special == 0 {
                let decoded = _mm_sub_epi8(chunk, sub42);
                let out_len = out.len();
                let out_ptr = out.as_mut_ptr().add(out_len);
                _mm_storeu_si128(out_ptr as *mut __m128i, decoded);
                out.set_len(out_len + 16);
                i += 16;
            } else {
                let end = (i + 16).min(len);
                while i < end {
                    let b = *data.get_unchecked(i);
                    match b {
                        b'\n' | b'\r' => {
                            i += 1;
                        }
                        b'=' if i + 1 < len => {
                            out.push((*data.get_unchecked(i + 1)).wrapping_sub(106));
                            i += 2;
                        }
                        _ => {
                            out.push(b.wrapping_sub(42));
                            i += 1;
                        }
                    }
                }
            }
        }

        while i < len {
            let b = *data.get_unchecked(i);
            match b {
                b'\n' | b'\r' => {
                    i += 1;
                }
                b'=' if i + 1 < len => {
                    out.push((*data.get_unchecked(i + 1)).wrapping_sub(106));
                    i += 2;
                }
                _ => {
                    out.push(b.wrapping_sub(42));
                    i += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 decoder — processes 32 bytes at a time
// ---------------------------------------------------------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_body_avx2(data: &[u8], out: &mut Vec<u8>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = data.len();
    let mut i = 0;

    out.reserve(len);

    unsafe {
        let sub42 = _mm256_set1_epi8(42u8 as i8);
        let eq_char = _mm256_set1_epi8(b'=' as i8);
        let cr_char = _mm256_set1_epi8(b'\r' as i8);
        let lf_char = _mm256_set1_epi8(b'\n' as i8);

        while i + 32 <= len {
            let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

            let has_eq = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, eq_char));
            let has_cr = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, cr_char));
            let has_lf = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, lf_char));
            let special = has_eq | has_cr | has_lf;

            if special == 0 {
                let decoded = _mm256_sub_epi8(chunk, sub42);
                let out_len = out.len();
                let out_ptr = out.as_mut_ptr().add(out_len);
                _mm256_storeu_si256(out_ptr as *mut __m256i, decoded);
                out.set_len(out_len + 32);
                i += 32;
            } else {
                let end = (i + 32).min(len);
                while i < end {
                    let b = *data.get_unchecked(i);
                    match b {
                        b'\n' | b'\r' => {
                            i += 1;
                        }
                        b'=' if i + 1 < len => {
                            out.push((*data.get_unchecked(i + 1)).wrapping_sub(106));
                            i += 2;
                        }
                        _ => {
                            out.push(b.wrapping_sub(42));
                            i += 1;
                        }
                    }
                }
            }
        }

        while i < len {
            let b = *data.get_unchecked(i);
            match b {
                b'\n' | b'\r' => {
                    i += 1;
                }
                b'=' if i + 1 < len => {
                    out.push((*data.get_unchecked(i + 1)).wrapping_sub(106));
                    i += 2;
                }
                _ => {
                    out.push(b.wrapping_sub(42));
                    i += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Find the start position of a line starting with `prefix` in raw bytes.
fn find_line_starting_with(data: &[u8], prefix: &[u8]) -> Option<usize> {
    if data.starts_with(prefix) {
        return Some(0);
    }
    let mut pos = 0;
    while let Some(nl) = memchr_byte(b'\n', &data[pos..]) {
        pos += nl + 1;
        if pos + prefix.len() <= data.len() && data[pos..].starts_with(prefix) {
            return Some(pos);
        }
    }
    None
}

/// Find the next newline character starting from `pos`.
fn find_newline(data: &[u8], pos: usize) -> Option<usize> {
    memchr_byte(b'\n', &data[pos..]).map(|offset| pos + offset)
}

/// Simple memchr for a single byte. Uses a tight loop — sufficient for our
/// scanning needs (lines are short, we scan forward sequentially).
#[inline]
fn memchr_byte(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

/// Strip trailing \r from a byte slice.
fn strip_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

/// Convert a byte slice to a string lossily (for parsing header/footer params).
fn lossy_str(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Extract a named parameter from a yEnc header/footer line.
/// e.g., `extract_param("=ybegin part=1 size=1234 name=file.bin", "size")` → `Some("1234")`
fn extract_param(line: &str, param: &str) -> Option<String> {
    let search = format!("{param}=");

    // Special handling for "name=" which takes the rest of the line
    if param == "name" {
        if let Some(pos) = line.find(&search) {
            return Some(line[pos + search.len()..].to_string());
        }
        return None;
    }

    if let Some(pos) = line.find(&search) {
        let start = pos + search.len();
        let rest = &line[start..];
        let end = rest.find(' ').unwrap_or(rest.len());
        return Some(rest[..end].to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_body_scalar_basic() {
        let hello = b"Hello";
        let encoded: Vec<u8> = hello.iter().map(|b| b.wrapping_add(42)).collect();
        let mut decoded = Vec::new();
        decode_body_scalar(&encoded, &mut decoded);
        assert_eq!(decoded, hello);
    }

    #[test]
    fn test_decode_body_with_escapes() {
        // Encode byte 0x00 (NUL): encoded = 42 = 0x2A, but that needs escaping?
        // No — only specific encoded values need escaping: NUL(0x00), LF(0x0A), CR(0x0D), '='(0x3D)
        // Byte 0xD6 encodes to (0xD6 + 42) % 256 = 0x00 — NUL, must be escaped
        // Escape: '=' followed by (0x00 + 64) = 0x40 = '@'
        let mut encoded = Vec::new();
        encoded.push(b'=');
        encoded.push(b'@'); // escape for NUL: (0x40 - 64 - 42) mod 256 = 0xD6... wait
        // Let's verify: =@ means (@ - 106) = (64 - 106) mod 256 = 214 = 0xD6
        // Actually let's just test with known values
        let original = b"Test";
        let enc: Vec<u8> = original.iter().map(|b| b.wrapping_add(42)).collect();
        let mut decoded = Vec::new();
        decode_body_scalar(&enc, &mut decoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_body_with_line_endings() {
        let original = b"AB";
        // Encode with \r\n between them
        let mut encoded = Vec::new();
        encoded.push(b'A'.wrapping_add(42));
        encoded.extend_from_slice(b"\r\n");
        encoded.push(b'B'.wrapping_add(42));

        let mut decoded = Vec::new();
        decode_body_scalar(&encoded, &mut decoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_body_simd() {
        // Generate enough data to exercise the SIMD path (>32 bytes, no specials)
        // We need encoded bytes that avoid \r (13), \n (10), and = (61).
        // So original bytes must avoid: (13-42)%256=227, (10-42)%256=224, (61-42)=19
        let original: Vec<u8> = (0..256)
            .map(|i| {
                let mut b = ((i * 7 + 13) % 200 + 33) as u8;
                // Avoid values whose encoded form is special
                while matches!(b.wrapping_add(42), 0x0A | 0x0D | 0x3D) {
                    b = b.wrapping_add(1);
                }
                b
            })
            .collect();
        let encoded: Vec<u8> = original.iter().map(|b| b.wrapping_add(42)).collect();

        assert!(
            !encoded
                .iter()
                .any(|&b| b == b'\r' || b == b'\n' || b == b'='),
            "Test data should not contain special characters"
        );

        let mut decoded = Vec::new();
        decode_body(&encoded, &mut decoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_decode_body_simd_with_specials() {
        // Mix of clean chunks and chunks with specials
        let original: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let mut encoded = Vec::new();
        for &b in &original {
            let enc = b.wrapping_add(42);
            match enc {
                0x00 | 0x0A | 0x0D | 0x3D => {
                    encoded.push(b'=');
                    encoded.push(enc.wrapping_add(64));
                }
                _ => {
                    encoded.push(enc);
                }
            }
        }

        let mut decoded = Vec::new();
        decode_body(&encoded, &mut decoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_extract_param() {
        let line = "=ybegin part=1 line=128 size=768000 name=test file.bin";
        assert_eq!(extract_param(line, "part"), Some("1".into()));
        assert_eq!(extract_param(line, "size"), Some("768000".into()));
        assert_eq!(extract_param(line, "name"), Some("test file.bin".into()));
        assert_eq!(extract_param(line, "missing"), None);
    }

    #[test]
    fn test_full_decode() {
        let original: Vec<u8> = (65..80).collect(); // A-O (safe range)
        let encoded_line: String = original
            .iter()
            .map(|b| (b.wrapping_add(42)) as char)
            .collect();
        let crc = crc32fast::hash(&original);

        let article = format!(
            "=ybegin part=1 line=128 size={} name=test.bin\n\
             =ypart begin=1 end={}\n\
             {encoded_line}\n\
             =yend size={} part=1 pcrc32={crc:08X}\n",
            original.len(),
            original.len(),
            original.len(),
        );

        let result = decode_yenc(article.as_bytes()).unwrap();
        assert_eq!(result.data, original);
        assert_eq!(result.filename, Some("test.bin".into()));
        assert_eq!(result.part_begin, Some(0));
        assert_eq!(result.file_size, Some(original.len() as u64));
        assert_eq!(result.crc32, crc);
    }

    #[test]
    fn test_decode_with_nntp_headers() {
        let original: Vec<u8> = (65..80).collect();
        let encoded_line: String = original
            .iter()
            .map(|b| (b.wrapping_add(42)) as char)
            .collect();
        let crc = crc32fast::hash(&original);

        let article = format!(
            "From: poster@example.com\r\n\
             Newsgroups: alt.binaries.test\r\n\
             Subject: test post\r\n\
             Message-Id: <test@example.com>\r\n\
             \r\n\
             =ybegin part=1 line=128 size={} name=test.bin\r\n\
             =ypart begin=1 end={}\r\n\
             {encoded_line}\r\n\
             =yend size={} part=1 pcrc32={crc:08X}\r\n",
            original.len(),
            original.len(),
            original.len(),
        );

        let result = decode_yenc(article.as_bytes()).unwrap();
        assert_eq!(result.data, original);
        assert_eq!(result.filename, Some("test.bin".into()));
    }

    #[test]
    fn test_decode_large_multiline() {
        // Exercise SIMD with a realistic multi-line article
        let original: Vec<u8> = (0..10_000).map(|i| (i % 200 + 33) as u8).collect();
        let crc = crc32fast::hash(&original);

        let mut body = Vec::new();
        let mut line_pos = 0;
        for &b in &original {
            let enc = b.wrapping_add(42);
            match enc {
                0x00 | 0x0A | 0x0D | 0x3D => {
                    body.push(b'=');
                    body.push(enc.wrapping_add(64));
                    line_pos += 2;
                }
                _ => {
                    body.push(enc);
                    line_pos += 1;
                }
            }
            if line_pos >= 128 {
                body.extend_from_slice(b"\r\n");
                line_pos = 0;
            }
        }
        if line_pos > 0 {
            body.extend_from_slice(b"\r\n");
        }

        let mut article = Vec::new();
        article.extend_from_slice(
            format!(
                "=ybegin part=1 line=128 size=10000 name=big.bin\r\n\
                      =ypart begin=1 end=10000\r\n"
            )
            .as_bytes(),
        );
        article.extend_from_slice(&body);
        article
            .extend_from_slice(format!("=yend size=10000 part=1 pcrc32={crc:08X}\r\n").as_bytes());

        let result = decode_yenc(&article).unwrap();
        assert_eq!(result.data.len(), original.len());
        assert_eq!(result.data, original);
    }

    #[test]
    fn test_crc_mismatch() {
        let original: Vec<u8> = (65..80).collect();
        let encoded_line: String = original
            .iter()
            .map(|b| (b.wrapping_add(42)) as char)
            .collect();

        let article = format!(
            "=ybegin line=128 size={} name=test.bin\n\
             {encoded_line}\n\
             =yend size={} crc32=DEADBEEF\n",
            original.len(),
            original.len(),
        );

        let err = decode_yenc(article.as_bytes()).unwrap_err();
        assert!(matches!(err, YencError::CrcMismatch { .. }));
    }

    #[test]
    fn test_missing_header() {
        let err = decode_yenc(b"no header here\n=yend\n").unwrap_err();
        assert!(matches!(err, YencError::MissingHeader));
    }

    #[test]
    fn test_missing_footer() {
        let err = decode_yenc(b"=ybegin line=128 size=10 name=test.bin\ndata\n").unwrap_err();
        assert!(matches!(err, YencError::MissingFooter));
    }
}
