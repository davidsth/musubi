//! Typed tensor metadata shared by the control and data planes.
//!
//! Descriptors travel over the JSON control plane. Tensor bytes continue to use
//! the existing fixed 32-byte binary data-plane header and chunk transfer path.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const MAX_TENSOR_RANK: usize = 8;
pub const MAX_TENSOR_NAME_BYTES: usize = 128;
pub const MAX_TENSOR_BYTES: u64 = 1_u64 << 40; // 1 TiB safety ceiling.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    U8,
    I16,
    I32,
    I64,
}

impl TensorDType {
    fn byte_width(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
            Self::F64 | Self::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    RowMajor,
    ColumnMajor,
    /// GGML storage order: dimension zero is the innermost/contiguous axis.
    Ggml,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorEncoding {
    Raw,
    GgmlQ4_0,
    GgmlQ4_1,
    GgmlQ5_0,
    GgmlQ5_1,
    GgmlQ8_0,
    GgmlQ8_1,
    GgmlQ2K,
    GgmlQ3K,
    GgmlQ4K,
    GgmlQ5K,
    GgmlQ6K,
    GgmlQ8K,
    GgmlIq2Xxs,
    GgmlIq2Xs,
    GgmlIq3Xxs,
    GgmlIq1S,
    GgmlIq4Nl,
    GgmlIq3S,
    GgmlIq2S,
    GgmlIq4Xs,
    GgmlIq1M,
}

impl TensorEncoding {
    fn ggml_block(self) -> Option<(u64, u64)> {
        match self {
            Self::Raw => None,
            Self::GgmlQ4_0 => Some((32, 18)),
            Self::GgmlQ4_1 => Some((32, 20)),
            Self::GgmlQ5_0 => Some((32, 22)),
            Self::GgmlQ5_1 => Some((32, 24)),
            Self::GgmlQ8_0 => Some((32, 34)),
            Self::GgmlQ8_1 => Some((32, 36)),
            Self::GgmlQ2K => Some((256, 84)),
            Self::GgmlQ3K => Some((256, 110)),
            Self::GgmlQ4K => Some((256, 144)),
            Self::GgmlQ5K => Some((256, 176)),
            Self::GgmlQ6K => Some((256, 210)),
            Self::GgmlQ8K => Some((256, 292)),
            Self::GgmlIq2Xxs => Some((256, 66)),
            Self::GgmlIq2Xs => Some((256, 74)),
            Self::GgmlIq3Xxs => Some((256, 98)),
            Self::GgmlIq1S => Some((256, 50)),
            Self::GgmlIq4Nl => Some((32, 18)),
            Self::GgmlIq3S => Some((256, 110)),
            Self::GgmlIq2S => Some((256, 82)),
            Self::GgmlIq4Xs => Some((256, 136)),
            Self::GgmlIq1M => Some((256, 56)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Matches tensor_id in the existing binary data-plane header.
    pub tensor_id: u32,
    pub name: String,
    /// Logical element type. Initial GGML block encodings represent F32 values.
    pub dtype: TensorDType,
    /// Dimensions in storage order. For GGML, shape[0] is the block axis.
    pub shape: Vec<u64>,
    pub layout: TensorLayout,
    pub encoding: TensorEncoding,
    /// Exact number of bytes expected in the following data-plane transfer.
    pub byte_len: u64,
}

impl TensorDescriptor {
    /// Compute encoded storage size with overflow and layout checks.
    pub fn expected_byte_len(&self) -> Result<u64> {
        if self.shape.is_empty() || self.shape.len() > MAX_TENSOR_RANK {
            bail!("tensor rank must be between 1 and {MAX_TENSOR_RANK}");
        }
        if self.shape.iter().any(|dimension| *dimension == 0) {
            bail!("tensor dimensions must be non-zero");
        }

        let elements = self.shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow::anyhow!("tensor element count overflow"))
        })?;

        let bytes = match self.encoding.ggml_block() {
            // Raw GGUF tensors still use GGML's dimension ordering. Layout
            // does not change the number of bytes in an unquantized tensor.
            None => elements
                .checked_mul(self.dtype.byte_width())
                .ok_or_else(|| anyhow::anyhow!("tensor byte length overflow"))?,
            Some((block_elements, block_bytes)) => {
                if self.layout != TensorLayout::Ggml {
                    bail!("GGML block encoding requires GGML layout");
                }
                if self.dtype != TensorDType::F32 {
                    bail!("GGML block encodings currently require logical F32 dtype");
                }
                let inner = self.shape[0];
                if inner % block_elements != 0 {
                    bail!("GGML innermost dimension must be divisible by {block_elements}");
                }
                let outer_rows = self.shape[1..].iter().try_fold(1_u64, |total, dimension| {
                    total
                        .checked_mul(*dimension)
                        .ok_or_else(|| anyhow::anyhow!("tensor row count overflow"))
                })?;
                (inner / block_elements)
                    .checked_mul(block_bytes)
                    .and_then(|row_bytes| row_bytes.checked_mul(outer_rows))
                    .ok_or_else(|| anyhow::anyhow!("quantized tensor byte length overflow"))?
            }
        };

        if bytes == 0 || bytes > MAX_TENSOR_BYTES {
            bail!("tensor byte length must be between 1 and {MAX_TENSOR_BYTES}");
        }
        Ok(bytes)
    }

    /// Reject malformed descriptors before allocating or receiving tensor data.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() || self.name.len() > MAX_TENSOR_NAME_BYTES {
            bail!("tensor name must be non-empty and at most {MAX_TENSOR_NAME_BYTES} UTF-8 bytes");
        }
        let expected = self.expected_byte_len()?;
        if self.byte_len != expected {
            bail!(
                "descriptor byte_len {} does not match expected size {expected}",
                self.byte_len
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        encoding: TensorEncoding,
        layout: TensorLayout,
        shape: Vec<u64>,
        byte_len: u64,
    ) -> TensorDescriptor {
        TensorDescriptor {
            tensor_id: 7,
            name: "hidden_states".into(),
            dtype: TensorDType::F16,
            shape,
            layout,
            encoding,
            byte_len,
        }
    }

    #[test]
    fn raw_f16_descriptor_calculates_exact_size() {
        let value = descriptor(
            TensorEncoding::Raw,
            TensorLayout::RowMajor,
            vec![2, 4, 8],
            128,
        );
        assert_eq!(value.expected_byte_len().unwrap(), 128);
        assert!(value.validate().is_ok());
    }

    #[test]
    fn ggml_q4_0_uses_block_storage_size() {
        let mut value = descriptor(
            TensorEncoding::GgmlQ4_0,
            TensorLayout::Ggml,
            vec![64, 2],
            72,
        );
        value.dtype = TensorDType::F32;
        assert_eq!(value.expected_byte_len().unwrap(), 72);
        assert!(value.validate().is_ok());
    }

    #[test]
    fn rejects_mismatched_size_and_invalid_quantized_shape() {
        let value = descriptor(TensorEncoding::Raw, TensorLayout::RowMajor, vec![4, 4], 31);
        assert!(value.validate().is_err());

        let mut quantized = descriptor(
            TensorEncoding::GgmlQ4_0,
            TensorLayout::Ggml,
            vec![33, 2],
            38,
        );
        quantized.dtype = TensorDType::F32;
        assert!(quantized.validate().is_err());
    }

    #[test]
    fn rejects_zero_dimensions_and_overflow() {
        let zero = descriptor(TensorEncoding::Raw, TensorLayout::RowMajor, vec![0, 4], 0);
        assert!(zero.validate().is_err());

        let overflow = descriptor(
            TensorEncoding::Raw,
            TensorLayout::RowMajor,
            vec![u64::MAX, 8],
            1,
        );
        assert!(overflow.validate().is_err());
    }

    #[test]
    fn raw_gguf_tensor_accepts_ggml_dimension_order() {
        let value = descriptor(TensorEncoding::Raw, TensorLayout::Ggml, vec![2, 3], 12);
        assert_eq!(value.expected_byte_len().unwrap(), 12);
        assert!(value.validate().is_ok());
    }

    #[test]
    fn rejects_incompatible_layout_and_dtype() {
        let mut wrong_layout = descriptor(
            TensorEncoding::GgmlQ4_0,
            TensorLayout::RowMajor,
            vec![32],
            18,
        );
        wrong_layout.dtype = TensorDType::F32;
        assert!(wrong_layout.validate().is_err());

        let wrong_dtype = descriptor(TensorEncoding::GgmlQ4_0, TensorLayout::Ggml, vec![32], 18);
        assert!(wrong_dtype.validate().is_err());
    }

    #[test]
    fn common_k_and_iq_encodings_use_exact_block_storage_sizes() {
        let cases = [
            (TensorEncoding::GgmlQ8_1, 32, 36),
            (TensorEncoding::GgmlQ2K, 256, 84),
            (TensorEncoding::GgmlQ3K, 256, 110),
            (TensorEncoding::GgmlQ4K, 256, 144),
            (TensorEncoding::GgmlQ5K, 256, 176),
            (TensorEncoding::GgmlQ6K, 256, 210),
            (TensorEncoding::GgmlQ8K, 256, 292),
            (TensorEncoding::GgmlIq2Xxs, 256, 66),
            (TensorEncoding::GgmlIq2Xs, 256, 74),
            (TensorEncoding::GgmlIq3Xxs, 256, 98),
            (TensorEncoding::GgmlIq1S, 256, 50),
            (TensorEncoding::GgmlIq4Nl, 32, 18),
            (TensorEncoding::GgmlIq3S, 256, 110),
            (TensorEncoding::GgmlIq2S, 256, 82),
            (TensorEncoding::GgmlIq4Xs, 256, 136),
            (TensorEncoding::GgmlIq1M, 256, 56),
        ];

        for (encoding, block_elements, block_bytes) in cases {
            let mut value = descriptor(
                encoding,
                TensorLayout::Ggml,
                vec![block_elements * 2, 3],
                block_bytes * 6,
            );
            value.dtype = TensorDType::F32;
            assert_eq!(value.expected_byte_len().unwrap(), value.byte_len);
            assert!(value.validate().is_ok(), "invalid descriptor for {encoding:?}");

            let encoded = serde_json::to_vec(&value).unwrap();
            let decoded: TensorDescriptor = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn raw_i16_and_f64_descriptors_use_their_full_element_width() {
        for (dtype, width) in [(TensorDType::I16, 2), (TensorDType::F64, 8)] {
            let mut value = descriptor(
                TensorEncoding::Raw,
                TensorLayout::Ggml,
                vec![3, 2],
                width * 6,
            );
            value.dtype = dtype;
            assert_eq!(value.expected_byte_len().unwrap(), width * 6);
            assert!(value.validate().is_ok());
        }
    }

}
