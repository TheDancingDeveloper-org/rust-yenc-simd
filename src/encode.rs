//! yEnc encoder.

use crc32fast::Hasher;

const LINE_WIDTH: usize = 128;

/// yEnc encode a raw data block into a complete article body.
///
/// Returns `(encoded_article_bytes, crc32)`.
///
/// The output includes `=ybegin`, optional `=ypart`, encoded body, and `=yend`.
pub fn encode_article(
    raw: &[u8],
    filename: &str,
    part: u32,
    total_parts: u32,
    file_offset: u64,
    total_file_size: u64,
) -> (Vec<u8>, u32) {
    let mut hasher = Hasher::new();
    hasher.update(raw);
    let crc = hasher.finalize();

    let mut out = Vec::with_capacity(raw.len() * 11 / 10 + 256);

    // =ybegin header
    if total_parts > 1 {
        out.extend_from_slice(
            format!(
                "=ybegin part={part} line={LINE_WIDTH} size={total_file_size} name={filename}\r\n"
            )
            .as_bytes(),
        );
        let begin = file_offset + 1;
        let end = file_offset + raw.len() as u64;
        out.extend_from_slice(format!("=ypart begin={begin} end={end}\r\n").as_bytes());
    } else {
        out.extend_from_slice(
            format!("=ybegin line={LINE_WIDTH} size={total_file_size} name={filename}\r\n")
                .as_bytes(),
        );
    }

    // Encode body
    let mut line_pos: usize = 0;
    for &byte in raw {
        let encoded = byte.wrapping_add(42);

        // Escape critical bytes, plus TAB/SPACE/DOT at line start
        let escape = matches!(encoded, 0x00 | 0x0A | 0x0D | 0x3D)
            || (line_pos == 0 && matches!(encoded, 0x09 | 0x20 | 0x2E));

        if escape {
            out.push(b'=');
            out.push(encoded.wrapping_add(64));
            line_pos += 2;
        } else {
            out.push(encoded);
            line_pos += 1;
        }

        if line_pos >= LINE_WIDTH {
            out.extend_from_slice(b"\r\n");
            line_pos = 0;
        }
    }
    if line_pos > 0 {
        out.extend_from_slice(b"\r\n");
    }

    // =yend footer
    if total_parts > 1 {
        out.extend_from_slice(
            format!("=yend size={} pcrc32={crc:08X}\r\n", raw.len()).as_bytes(),
        );
    } else {
        out.extend_from_slice(
            format!("=yend size={} crc32={crc:08X}\r\n", raw.len()).as_bytes(),
        );
    }

    (out, crc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::decode_yenc;

    #[test]
    fn test_encode_single_part() {
        let data = b"Hello, world!";
        let (encoded, crc) = encode_article(data, "test.txt", 1, 1, 0, data.len() as u64);
        assert_eq!(crc, crc32fast::hash(data));

        let result = decode_yenc(&encoded).unwrap();
        assert_eq!(result.data, data);
        assert_eq!(result.filename, Some("test.txt".into()));
    }

    #[test]
    fn test_encode_multipart() {
        let data = b"Part one data";
        let (encoded, _) = encode_article(data, "big.bin", 1, 3, 0, 1000);

        let result = decode_yenc(&encoded).unwrap();
        assert_eq!(result.data, data);
        assert_eq!(result.part_begin, Some(0));
    }

    #[test]
    fn test_roundtrip_all_bytes() {
        // Encode and decode every possible byte value
        let data: Vec<u8> = (0..=255).collect();
        let (encoded, _) = encode_article(&data, "allbytes.bin", 1, 1, 0, 256);

        let result = decode_yenc(&encoded).unwrap();
        assert_eq!(result.data, data);
    }

    #[test]
    fn test_roundtrip_large() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let (encoded, _) = encode_article(&data, "large.bin", 1, 1, 0, data.len() as u64);

        let result = decode_yenc(&encoded).unwrap();
        assert_eq!(result.data, data);
    }
}
