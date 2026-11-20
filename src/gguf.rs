//! Bounded GGUF metadata and tensor-table inspection.
//!
//! This module reads only the GGUF header, metadata and tensor descriptors. It
//! seeks over tokenizer arrays and tensor payloads; it never loads model weights.
//! The initial reader supports little-endian GGUF v2/v3 and reports byte sizes
//! only for GGML tensor types whose block layouts are known here.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const DEFAULT_ALIGNMENT: u32 = 32;
pub const MAX_ALIGNMENT: u32 = 4096;
pub const MAX_METADATA_ENTRIES: u64 = 65_536;
pub const MAX_TENSORS: u64 = 262_144;
pub const MAX_TENSOR_DIMS: u32 = 8;
pub const MAX_TENSOR_NAME_BYTES: u64 = 64;
pub const MAX_METADATA_KEY_BYTES: u64 = 1024;
pub const MAX_SELECTED_STRING_BYTES: u64 = 4096;
pub const MAX_ARRAY_ELEMENTS_TO_SKIP: u64 = 2_000_000;
pub const MAX_ARRAY_DEPTH: u32 = 8;
pub const MAX_MODELS_TO_SCAN: usize = 16;

#[derive(Debug, Clone, Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    /// GGML order: dimensions[0] is the innermost/contiguous axis.
    pub dimensions: Vec<u64>,
    pub ggml_type_id: u32,
    pub type_name: String,
    /// Offset relative to the beginning of the tensor-data section.
    pub relative_offset: u64,
    /// Absolute byte offset from the beginning of the GGUF file.
    pub file_offset: u64,
    /// None means this reader does not yet know the type's block layout.
    pub byte_len: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufModelSummary {
    pub filename: String,
    pub file_bytes: u64,
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub model_basename: Option<String>,
    pub model_size_label: Option<String>,
    pub model_version: Option<String>,
    pub file_type: Option<u32>,
    pub quantization_version: Option<u32>,
    pub layer_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub parameter_count_metadata: Option<u64>,
    pub estimated_tensor_elements: u64,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub known_tensor_bytes: u64,
    pub unsupported_tensor_count: u64,
    pub tensor_type_counts: BTreeMap<String, u64>,
    pub tensors: Vec<GgufTensorInfo>,
}

impl GgufModelSummary {
    fn into_card(self) -> GgufModelCard {
        GgufModelCard {
            filename: self.filename,
            file_bytes: self.file_bytes,
            version: self.version,
            alignment: self.alignment,
            architecture: self.architecture,
            model_name: self.model_name,
            model_basename: self.model_basename,
            model_size_label: self.model_size_label,
            model_version: self.model_version,
            file_type: self.file_type,
            quantization_version: self.quantization_version,
            layer_count: self.layer_count,
            context_length: self.context_length,
            embedding_length: self.embedding_length,
            feed_forward_length: self.feed_forward_length,
            parameter_count_metadata: self.parameter_count_metadata,
            estimated_tensor_elements: self.estimated_tensor_elements,
            tensor_count: self.tensor_count,
            known_tensor_bytes: self.known_tensor_bytes,
            unsupported_tensor_count: self.unsupported_tensor_count,
            tensor_type_counts: self.tensor_type_counts,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufModelCard {
    pub filename: String,
    pub file_bytes: u64,
    pub version: u32,
    pub alignment: u32,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub model_basename: Option<String>,
    pub model_size_label: Option<String>,
    pub model_version: Option<String>,
    pub file_type: Option<u32>,
    pub quantization_version: Option<u32>,
    pub layer_count: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub feed_forward_length: Option<u64>,
    pub parameter_count_metadata: Option<u64>,
    pub estimated_tensor_elements: u64,
    pub tensor_count: u64,
    pub known_tensor_bytes: u64,
    pub unsupported_tensor_count: u64,
    pub tensor_type_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GgufModelCatalog {
    pub scanned_files: usize,
    pub invalid_files: usize,
    pub omitted_files: usize,
    pub models: Vec<GgufModelCard>,
    pub scanned_at_unix: u64,
}

pub fn configured_model_dir() -> PathBuf {
    std::env::var_os("AITP_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("aitp-models"))
}

/// Inspect a GGUF file without reading its tensor payloads into memory.
pub fn inspect_file(path: &Path) -> Result<GgufModelSummary> {
    let path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting GGUF path {}", path.display()))?;
    if !path_metadata.file_type().is_file() {
        bail!("GGUF path must be a regular file, not a directory or symbolic link");
    }

    let mut options = OpenOptions::new();
    options.read(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }

    let file = options
        .open(path)
        .with_context(|| format!("opening GGUF file {}", path.display()))?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        bail!("opened GGUF path is not a regular file");
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!("refusing Windows reparse-point GGUF input");
        }
    }

    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.gguf")
        .to_owned();
    let file_len = metadata.len();
    let mut reader = BufReader::new(file);
    parse_reader(&mut reader, file_len, filename)
}

/// Scan only the top-level .gguf files in a model directory. At most sixteen
/// files are parsed per scan; malformed files are counted and skipped.
pub fn scan_model_dir(directory: &Path) -> Result<GgufModelCatalog> {
    let scanned_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let directory_metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(GgufModelCatalog {
                scanned_at_unix,
                ..GgufModelCatalog::default()
            });
        }
        Err(error) => return Err(error.into()),
    };
    if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
        bail!("GGUF model directory must be a real directory, not a symbolic link");
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("reading GGUF model directory {}", directory.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_file()
            && path.extension().and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
        {
            paths.push(path);
        }
    }
    paths.sort_by(|left, right| {
        left.file_name().cmp(&right.file_name())
    });

    let omitted_files = paths.len().saturating_sub(MAX_MODELS_TO_SCAN);
    paths.truncate(MAX_MODELS_TO_SCAN);

    let mut catalog = GgufModelCatalog {
        scanned_files: paths.len(),
        invalid_files: 0,
        omitted_files,
        models: Vec::new(),
        scanned_at_unix,
    };

    for path in paths {
        match inspect_file(&path) {
            Ok(summary) => catalog.models.push(summary.into_card()),
            Err(_) => catalog.invalid_files += 1,
        }
    }

    Ok(catalog)
}

pub fn parse_reader<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    filename: String,
) -> Result<GgufModelSummary> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).context("reading GGUF magic")?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file: invalid magic");
    }

    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes).context("reading GGUF version")?;
    let version = u32::from_le_bytes(version_bytes);
    if !matches!(version, 2 | 3) {
        let big_endian_version = u32::from_be_bytes(version_bytes);
        if big_endian_version == 3 {
            bail!("big-endian GGUF v3 is not supported by this reader");
        }
        bail!("unsupported GGUF version {version}; supported little-endian versions are 2 and 3");
    }

    let tensor_count = read_u64(reader).context("reading GGUF tensor count")?;
    let metadata_count = read_u64(reader).context("reading GGUF metadata count")?;
    if tensor_count > MAX_TENSORS {
        bail!("GGUF tensor count {tensor_count} exceeds the safety limit {MAX_TENSORS}");
    }
    if metadata_count > MAX_METADATA_ENTRIES {
        bail!("GGUF metadata count {metadata_count} exceeds the safety limit {MAX_METADATA_ENTRIES}");
    }

    let mut architecture = None;
    let mut model_name = None;
    let mut model_basename = None;
    let mut model_size_label = None;
    let mut model_version = None;
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut file_type = None;
    let mut quantization_version = None;
    let mut layer_count = None;
    let mut context_length = None;
    let mut embedding_length = None;
    let mut feed_forward_length = None;
    let mut parameter_count_metadata = None;
    let mut seen_keys = HashSet::new();
    let mut skip_budget = MAX_ARRAY_ELEMENTS_TO_SKIP;

    for _ in 0..metadata_count {
        let key = read_string(reader, file_len, MAX_METADATA_KEY_BYTES)
            .context("reading GGUF metadata key")?;
        if key.is_empty() || !key.is_ascii() {
            bail!("GGUF metadata keys must be non-empty ASCII strings");
        }
        if !seen_keys.insert(key.clone()) {
            bail!("duplicate GGUF metadata key '{key}'");
        }
        let value_type = read_u32(reader).context("reading GGUF metadata value type")?;

        match key.as_str() {
            "general.architecture" => {
                architecture = Some(read_selected_string(reader, file_len, value_type, &key)?);
            }
            "general.name" => {
                model_name = Some(read_selected_string(reader, file_len, value_type, &key)?);
            }
            "general.basename" => {
                model_basename = Some(read_selected_string(reader, file_len, value_type, &key)?);
            }
            "general.size_label" => {
                model_size_label = Some(read_selected_string(reader, file_len, value_type, &key)?);
            }
            "general.version" => {
                model_version = Some(read_selected_string(reader, file_len, value_type, &key)?);
            }
            "general.alignment" => {
                if value_type != 4 {
                    bail!("general.alignment must be a uint32 GGUF metadata value");
                }
                alignment = read_u32(reader)?;
            }
            "general.file_type" => {
                file_type = Some(read_selected_u32(reader, file_len, value_type, &key, &mut skip_budget)?);
            }
            "general.quantization_version" => {
                quantization_version = Some(read_selected_u32(reader, file_len, value_type, &key, &mut skip_budget)?);
            }
            "general.parameter_count" => {
                parameter_count_metadata = read_selected_optional_u64(reader, file_len, value_type, &key, &mut skip_budget)?;
            }
            _ if key.ends_with(".block_count") => {
                layer_count = read_selected_optional_u64(reader, file_len, value_type, &key, &mut skip_budget)?;
            }
            _ if key.ends_with(".context_length") => {
                context_length = read_selected_optional_u64(reader, file_len, value_type, &key, &mut skip_budget)?;
            }
            _ if key.ends_with(".embedding_length") => {
                embedding_length = read_selected_optional_u64(reader, file_len, value_type, &key, &mut skip_budget)?;
            }
            _ if key.ends_with(".feed_forward_length") => {
                feed_forward_length = read_selected_optional_u64(reader, file_len, value_type, &key, &mut skip_budget)?;
            }
            _ => skip_value(reader, file_len, value_type, 0, &mut skip_budget)?,
        }
    }

    if !(8..=MAX_ALIGNMENT).contains(&alignment) || alignment % 8 != 0 {
        bail!("GGUF alignment must be a multiple of 8 between 8 and {MAX_ALIGNMENT}, got {alignment}");
    }

    let descriptor_end = reader.stream_position()?;
    let data_offset = if tensor_count == 0 {
        descriptor_end
    } else {
        align_up(descriptor_end, u64::from(alignment))?
    };
    if data_offset > file_len {
        bail!("GGUF tensor-data offset is beyond the end of the file");
    }

    let mut tensors = Vec::with_capacity((tensor_count as usize).min(4096));
    let mut tensor_names = HashSet::new();
    let mut ranges = Vec::new();
    let mut known_tensor_bytes = 0u64;
    let mut unsupported_tensor_count = 0u64;
    let mut estimated_tensor_elements = 0u64;
    let mut tensor_type_counts = BTreeMap::<String, u64>::new();

    for _ in 0..tensor_count {
        let name = read_string(reader, file_len, MAX_TENSOR_NAME_BYTES)
            .context("reading GGUF tensor name")?;
        if name.is_empty() || !tensor_names.insert(name.clone()) {
            bail!("GGUF tensor names must be non-empty and unique (duplicate '{name}')");
        }

        let dimensions_count = read_u32(reader).context("reading GGUF tensor rank")?;
        if dimensions_count == 0 || dimensions_count > MAX_TENSOR_DIMS {
            bail!("GGUF tensor '{name}' has unsupported rank {dimensions_count}");
        }

        let mut dimensions = Vec::with_capacity(dimensions_count as usize);
        let mut elements = 1u64;
        for _ in 0..dimensions_count {
            let dimension = read_u64(reader).context("reading GGUF tensor dimension")?;
            if dimension == 0 {
                bail!("GGUF tensor '{name}' has a zero dimension");
            }
            elements = elements
                .checked_mul(dimension)
                .context("GGUF tensor element count overflow")?;
            dimensions.push(dimension);
        }
        estimated_tensor_elements = estimated_tensor_elements
            .checked_add(elements)
            .context("GGUF total parameter estimate overflow")?;

        let ggml_type_id = read_u32(reader).context("reading GGUF tensor type")?;
        let relative_offset = read_u64(reader).context("reading GGUF tensor offset")?;
        if relative_offset % u64::from(alignment) != 0 {
            bail!("GGUF tensor '{name}' offset {relative_offset} is not aligned to {alignment}");
        }

        let (type_name, layout) = ggml_type_info(ggml_type_id);
        let byte_len = match layout {
            Some((block_elements, bytes_per_block)) => {
                let inner = dimensions[0];
                if inner % block_elements != 0 {
                    bail!("GGUF tensor '{name}' dimension 0 is not divisible by type block size {block_elements}");
                }
                let rows = dimensions.iter().skip(1).try_fold(1u64, |total, value| {
                    total.checked_mul(*value).context("GGUF tensor row count overflow")
                })?;
                let bytes = (inner / block_elements)
                    .checked_mul(bytes_per_block)
                    .and_then(|row_bytes| row_bytes.checked_mul(rows))
                    .context("GGUF tensor byte length overflow")?;
                let absolute_start = data_offset
                    .checked_add(relative_offset)
                    .context("GGUF absolute tensor offset overflow")?;
                let absolute_end = absolute_start
                    .checked_add(bytes)
                    .context("GGUF tensor end offset overflow")?;
                if absolute_end > file_len {
                    bail!("GGUF tensor '{name}' extends beyond the file length");
                }
                ranges.push((absolute_start, absolute_end, name.clone()));
                known_tensor_bytes = known_tensor_bytes
                    .checked_add(bytes)
                    .context("GGUF known tensor byte total overflow")?;
                Some(bytes)
            }
            None => {
                unsupported_tensor_count += 1;
                let absolute_start = data_offset
                    .checked_add(relative_offset)
                    .context("GGUF absolute tensor offset overflow")?;
                if absolute_start >= file_len {
                    bail!("GGUF tensor '{name}' offset is beyond the tensor data");
                }
                None
            }
        };

        *tensor_type_counts.entry(type_name.clone()).or_insert(0) += 1;
        let file_offset = data_offset
            .checked_add(relative_offset)
            .context("GGUF absolute tensor offset overflow")?;
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type_id,
            type_name,
            relative_offset,
            file_offset,
            byte_len,
        });
    }

    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            bail!("GGUF known-size tensors '{}' and '{}' overlap", pair[0].2, pair[1].2);
        }
    }

    Ok(GgufModelSummary {
        filename,
        file_bytes: file_len,
        version,
        alignment,
        data_offset,
        architecture,
        model_name,
        model_basename,
        model_size_label,
        model_version,
        file_type,
        quantization_version,
        layer_count,
        context_length,
        embedding_length,
        feed_forward_length,
        parameter_count_metadata,
        estimated_tensor_elements,
        tensor_count,
        metadata_count,
        known_tensor_bytes,
        unsupported_tensor_count,
        tensor_type_counts,
        tensors,
    })
}

fn read_selected_string<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    value_type: u32,
    key: &str,
) -> Result<String> {
    if value_type != 8 {
        bail!("GGUF metadata key '{key}' must have a string value");
    }
    read_string(reader, file_len, MAX_SELECTED_STRING_BYTES)
        .with_context(|| format!("reading GGUF metadata string '{key}'"))
}

fn read_selected_u32<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    value_type: u32,
    key: &str,
    budget: &mut u64,
) -> Result<u32> {
    let value = read_selected_optional_u64(reader, file_len, value_type, key, budget)?
        .context(format!("GGUF metadata key '{key}' must be an unsigned integer"))?;
    u32::try_from(value).with_context(|| format!("GGUF metadata key '{key}' exceeds uint32"))
}

fn read_selected_optional_u64<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    value_type: u32,
    key: &str,
    budget: &mut u64,
) -> Result<Option<u64>> {
    if is_integer_metadata_type(value_type) {
        read_metadata_unsigned(reader, value_type)
            .with_context(|| format!("reading numeric GGUF metadata '{key}'"))
            .map(Some)
    } else {
        skip_value(reader, file_len, value_type, 0, budget)?;
        Ok(None)
    }
}

fn read_metadata_unsigned<R: Read>(reader: &mut R, value_type: u32) -> Result<u64> {
    let value = match value_type {
        0 => u64::from(read_u8(reader)?),
        1 => {
            let value = read_u8(reader)? as i8;
            u64::try_from(value).context("negative GGUF integer metadata value")?
        }
        2 => u64::from(read_u16(reader)?),
        3 => {
            let value = read_u16(reader)? as i16;
            u64::try_from(value).context("negative GGUF integer metadata value")?
        }
        4 => u64::from(read_u32(reader)?),
        5 => {
            let value = read_u32(reader)? as i32;
            u64::try_from(value).context("negative GGUF integer metadata value")?
        }
        10 => read_u64(reader)?,
        11 => {
            let value = read_u64(reader)? as i64;
            u64::try_from(value).context("negative GGUF integer metadata value")?
        }
        _ => bail!("GGUF metadata value type {value_type} is not an integer"),
    };
    Ok(value)
}

fn is_integer_metadata_type(value_type: u32) -> bool {
    matches!(value_type, 0 | 1 | 2 | 3 | 4 | 5 | 10 | 11)
}

fn skip_value<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    value_type: u32,
    depth: u32,
    budget: &mut u64,
) -> Result<()> {
    let width = match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        8 => {
            skip_string(reader, file_len)?;
            return Ok(());
        }
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                bail!("GGUF metadata array nesting exceeds {MAX_ARRAY_DEPTH}");
            }
            let element_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > *budget {
                bail!("GGUF metadata arrays exceed the skip-element budget");
            }
            *budget -= count;
            if let Some(element_width) = metadata_fixed_width(element_type) {
                let bytes = count
                    .checked_mul(element_width)
                    .context("GGUF metadata array size overflow")?;
                skip_bytes(reader, file_len, bytes)?;
            } else {
                for _ in 0..count {
                    skip_value(reader, file_len, element_type, depth + 1, budget)?;
                }
            }
            return Ok(());
        }
        _ => bail!("unknown GGUF metadata value type {value_type}"),
    };

    if let Some(width) = width {
        skip_bytes(reader, file_len, width)?;
    }
    Ok(())
}

fn metadata_fixed_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

fn skip_string<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<()> {
    let length = read_u64(reader)?;
    skip_bytes(reader, file_len, length)
}

fn read_string<R: Read + Seek>(reader: &mut R, file_len: u64, max_len: u64) -> Result<String> {
    let length = read_u64(reader)?;
    if length > max_len {
        bail!("GGUF string length {length} exceeds the limit {max_len}");
    }
    let position = reader.stream_position()?;
    let end = position.checked_add(length).context("GGUF string offset overflow")?;
    if end > file_len {
        bail!("GGUF string extends beyond the file");
    }
    let size = usize::try_from(length).context("GGUF string length exceeds platform limits")?;
    let mut bytes = vec![0u8; size];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn skip_bytes<R: Read + Seek>(reader: &mut R, file_len: u64, length: u64) -> Result<()> {
    let position = reader.stream_position()?;
    let end = position.checked_add(length).context("GGUF skip offset overflow")?;
    if end > file_len {
        bail!("GGUF metadata value extends beyond the file");
    }
    reader.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .context("GGUF alignment offset overflow")
    }
}

fn ggml_type_info(type_id: u32) -> (String, Option<(u64, u64)>) {
    let entry = match type_id {
        0 => Some(("F32", 1, 4)),
        1 => Some(("F16", 1, 2)),
        2 => Some(("Q4_0", 32, 18)),
        3 => Some(("Q4_1", 32, 20)),
        6 => Some(("Q5_0", 32, 22)),
        7 => Some(("Q5_1", 32, 24)),
        8 => Some(("Q8_0", 32, 34)),
        9 => Some(("Q8_1", 32, 36)),
        10 => Some(("Q2_K", 256, 84)),
        11 => Some(("Q3_K", 256, 110)),
        12 => Some(("Q4_K", 256, 144)),
        13 => Some(("Q5_K", 256, 176)),
        14 => Some(("Q6_K", 256, 210)),
        15 => Some(("Q8_K", 256, 292)),
        16 => Some(("IQ2_XXS", 256, 66)),
        17 => Some(("IQ2_XS", 256, 74)),
        18 => Some(("IQ3_XXS", 256, 98)),
        19 => Some(("IQ1_S", 256, 50)),
        20 => Some(("IQ4_NL", 32, 18)),
        21 => Some(("IQ3_S", 256, 110)),
        22 => Some(("IQ2_S", 256, 82)),
        23 => Some(("IQ4_XS", 256, 136)),
        24 => Some(("I8", 1, 1)),
        25 => Some(("I16", 1, 2)),
        26 => Some(("I32", 1, 4)),
        27 => Some(("I64", 1, 8)),
        28 => Some(("F64", 1, 8)),
        29 => Some(("IQ1_M", 256, 56)),
        30 => Some(("BF16", 1, 2)),
        _ => None,
    };
    if let Some((name, block, bytes)) = entry {
        (name.to_owned(), Some((block, bytes)))
    } else {
        let name = match type_id {
            34 => "TQ1_0".to_owned(),
            35 => "TQ2_0".to_owned(),
            39 => "MXFP4".to_owned(),
            40 => "NVFP4".to_owned(),
            41 => "Q1_0".to_owned(),
            42 => "Q2_0".to_owned(),
            43 => "F8_E4M3".to_owned(),
            _ => format!("unknown({type_id})"),
        };
        (name, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn put_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_string(bytes: &mut Vec<u8>, value: &[u8]) {
        put_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value);
    }

    fn put_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        put_string(bytes, key.as_bytes());
        put_u32(bytes, 8);
        put_string(bytes, value.as_bytes());
    }

    fn put_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        put_string(bytes, key.as_bytes());
        put_u32(bytes, 4);
        put_u32(bytes, value);
    }

    fn put_u64_metadata(bytes: &mut Vec<u8>, key: &str, value: u64) {
        put_string(bytes, key.as_bytes());
        put_u32(bytes, 10);
        put_u64(bytes, value);
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 2);
        put_u64(&mut bytes, 6);

        put_string_metadata(&mut bytes, "general.architecture", "llama");
        put_string_metadata(&mut bytes, "general.name", "Tiny fixture");
        put_u32_metadata(&mut bytes, "general.alignment", 32);
        put_u32_metadata(&mut bytes, "llama.block_count", 2);
        put_u64_metadata(&mut bytes, "llama.context_length", 2048);

        // Tokenizer string arrays may contain byte sequences that are not UTF-8.
        put_string(&mut bytes, b"tokenizer.ggml.tokens");
        put_u32(&mut bytes, 9);
        put_u32(&mut bytes, 8);
        put_u64(&mut bytes, 2);
        put_string(&mut bytes, b"hello");
        put_string(&mut bytes, &[0xff, 0x00]);

        put_string(&mut bytes, b"blk.0.attn_q.weight");
        put_u32(&mut bytes, 2);
        put_u64(&mut bytes, 32);
        put_u64(&mut bytes, 2);
        put_u32(&mut bytes, 2); // Q4_0
        put_u64(&mut bytes, 0);

        put_string(&mut bytes, b"output.weight");
        put_u32(&mut bytes, 2);
        put_u64(&mut bytes, 2);
        put_u64(&mut bytes, 2);
        put_u32(&mut bytes, 1); // F16
        put_u64(&mut bytes, 64);

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&vec![0u8; 72]);
        bytes
    }

    #[test]
    fn parses_v3_metadata_and_tensor_table_without_decoding_tokenizer_bytes() {
        let bytes = fixture();
        let mut cursor = Cursor::new(bytes.clone());
        let model = parse_reader(&mut cursor, bytes.len() as u64, "tiny.gguf".into()).unwrap();

        assert_eq!(model.version, 3);
        assert_eq!(model.architecture.as_deref(), Some("llama"));
        assert_eq!(model.model_name.as_deref(), Some("Tiny fixture"));
        assert_eq!(model.layer_count, Some(2));
        assert_eq!(model.context_length, Some(2048));
        assert_eq!(model.tensor_count, 2);
        assert_eq!(model.known_tensor_bytes, 44);
        assert_eq!(model.unsupported_tensor_count, 0);
        assert_eq!(model.estimated_tensor_elements, 68);
        assert_eq!(model.tensors[0].type_name, "Q4_0");
        assert_eq!(model.tensors[0].byte_len, Some(36));
        assert_eq!(model.tensors[1].file_offset, model.data_offset + 64);
    }

    #[test]
    fn reports_unknown_tensor_types_without_guessing_their_sizes() {
        let mut bytes = fixture();
        let name = b"blk.0.attn_q.weight";
        let position = bytes.windows(name.len()).position(|window| window == name).unwrap();
        let type_pos = position + name.len() + 4 + 8 + 8;
        bytes[type_pos..type_pos + 4].copy_from_slice(&999u32.to_le_bytes());

        let mut cursor = Cursor::new(bytes.clone());
        let model = parse_reader(&mut cursor, bytes.len() as u64, "tiny.gguf".into()).unwrap();
        assert_eq!(model.tensors[0].type_name, "unknown(999)");
        assert_eq!(model.tensors[0].byte_len, None);
        assert_eq!(model.unsupported_tensor_count, 1);
    }

    #[test]
    fn accepts_metadata_only_gguf_without_alignment_padding() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 2);
        put_string_metadata(&mut bytes, "general.architecture", "llama");
        put_u32_metadata(&mut bytes, "general.alignment", 32);
        let mut cursor = Cursor::new(bytes.clone());
        let model = parse_reader(&mut cursor, bytes.len() as u64, "vocab.gguf".into()).unwrap();
        assert_eq!(model.tensor_count, 0);
        assert_eq!(model.data_offset, bytes.len() as u64);
    }

    #[test]
    fn rejects_truncated_payload_ranges_and_invalid_alignment() {
        let mut bytes = fixture();
        bytes.truncate(bytes.len() - 8);
        let mut cursor = Cursor::new(bytes.clone());
        assert!(parse_reader(&mut cursor, bytes.len() as u64, "truncated.gguf".into()).is_err());

        let mut invalid = fixture();
        // The alignment metadata value is after its type tag; find the encoded key.
        let key = b"general.alignment";
        let position = invalid.windows(key.len()).position(|window| window == key).unwrap();
        let value_pos = position + key.len() + 4;
        invalid[value_pos..value_pos + 4].copy_from_slice(&7u32.to_le_bytes());
        let mut cursor = Cursor::new(invalid.clone());
        assert!(parse_reader(&mut cursor, invalid.len() as u64, "bad-alignment.gguf".into()).is_err());
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let mut bytes = fixture();
        let name = b"output.weight";
        let position = bytes.windows(name.len()).position(|window| window == name).unwrap();
        let offset_pos = position + name.len() + 4 + 8 + 8 + 4;
        bytes[offset_pos..offset_pos + 8].copy_from_slice(&32u64.to_le_bytes());
        let mut cursor = Cursor::new(bytes.clone());
        assert!(parse_reader(&mut cursor, bytes.len() as u64, "overlap.gguf".into()).is_err());
    }

    #[test]
    fn rejects_bad_magic_unsupported_version_and_excessive_counts() {
        let mut bad_magic = fixture();
        bad_magic[0] = b'X';
        let mut cursor = Cursor::new(bad_magic.clone());
        assert!(parse_reader(&mut cursor, bad_magic.len() as u64, "bad.gguf".into()).is_err());

        let mut bad_version = fixture();
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        let mut cursor = Cursor::new(bad_version.clone());
        assert!(parse_reader(&mut cursor, bad_version.len() as u64, "version.gguf".into()).is_err());

        let mut excessive = fixture();
        excessive[8..16].copy_from_slice(&(MAX_TENSORS + 1).to_le_bytes());
        let mut cursor = Cursor::new(excessive.clone());
        assert!(parse_reader(&mut cursor, excessive.len() as u64, "count.gguf".into()).is_err());
    }
}
