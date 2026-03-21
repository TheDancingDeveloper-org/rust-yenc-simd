//! Fast yEnc encoder/decoder with SIMD acceleration.
//!
//! yEnc encoding: each byte is `(original_byte + 42) % 256`.
//! Escape sequences: `=char` means `(char - 64 - 42) mod 256`.
//! Critical characters that must be escaped: NUL, LF, CR, `=`.
//!
//! # Example
//!
//! ```
//! use yenc_simd::{decode_yenc, encode_article};
//!
//! let data = b"Hello, world!";
//! let (encoded, crc) = encode_article(data, "test.bin", 1, 1, 0, data.len() as u64);
//! let result = decode_yenc(&encoded).unwrap();
//! assert_eq!(result.data, data);
//! ```

mod decode;
mod encode;

pub use decode::{YencDecodeResult, YencError, decode_yenc};
pub use encode::encode_article;
