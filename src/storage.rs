//! Durable tensor metadata and a bounded local artifact catalog.
//!
//! A manifest is published only with a fully received payload. It is a runtime
//! handoff description, not an instruction to execute code.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt},
};

use crate::{control::TensorOfferMessage, tensor::TensorDescriptor};

pub const MANIFEST_VERSION: u16 = 1;
pub const MAX_MANIFEST_BYTES: u64 = 16 * 1024;
pub const MAX_MANIFESTS_TO_SCAN: usize = 4096;
pub const MAX_RECENT_TENSORS: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorManifest {
    pub version: u16,
    pub sender_node_id: String,
    pub offer_sequence: u64,
    pub message_id: u32,
    pub descriptor: TensorDescriptor,
    pub byte_len: u64,
    /// Lowercase hexadecimal SHA-256 of the committed payload.
    pub sha256: String,
    /// Safe basename of the payload file, never an arbitrary path.
    pub payload_file: String,
    pub stored_at_unix: u64,
}

impl TensorManifest {
    pub fn validate(&self) -> Result<()> {
        if self.version != MANIFEST_VERSION {
            bail!("unsupported tensor manifest version {}", self.version);
        }
        if self.sender_node_id.trim().is_empty() || self.sender_node_id.len() > 64 {
            bail!("invalid tensor manifest sender identity");
        }
        if self.offer_sequence == 0 || self.message_id == 0 {
            bail!("tensor manifest offer sequence and message ID must be nonzero");
        }
        self.descriptor.validate()?;
        if self.byte_len != self.descriptor.byte_len {
            bail!("manifest byte length does not match its tensor descriptor");
        }
        if self.sha256.len() != 64
            || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || self.sha256.bytes().any(|byte| byte.is_ascii_uppercase())
        {
            bail!("manifest SHA-256 must be 64 lowercase hexadecimal characters");
        }
        if !is_safe_payload_basename(&self.payload_file) {
            bail!("manifest payload_file must be a safe .tensor basename");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorStoreSnapshot {
    pub count: usize,
    pub total_bytes: u64,
    /// Most recent valid manifests. The aggregate count and size cover all
    /// valid entries discovered during the bounded scan.
    pub recent: Vec<TensorManifest>,
}

pub fn configured_store_dir() -> PathBuf {
    std::env::var_os("AITP_TENSOR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("aitp-tensors"))
}

pub fn manifest_path_for_payload(payload_path: &Path) -> PathBuf {
    payload_path.with_extension("json")
}

/// Publish a manifest and payload as a pair. The manifest is staged first,
/// then the payload is renamed into place, and finally the sidecar is published.
/// A crash can leave an unreferenced file, but the catalog never exposes a
/// manifest whose payload is missing or has the wrong length.
pub async fn commit_tensor_file(
    partial_path: &Path,
    final_path: &Path,
    offer: &TensorOfferMessage,
) -> Result<TensorManifest> {
    offer.descriptor.validate()?;
    if offer.sequence == 0 || offer.message_id == 0 {
        bail!("tensor offer sequence and message ID must be nonzero");
    }
    let payload_name = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("tensor payload path has no UTF-8 filename")?
        .to_owned();
    if !is_safe_payload_basename(&payload_name) {
        bail!("generated tensor payload filename is invalid");
    }

    let payload_metadata = fs::symlink_metadata(partial_path)
        .await
        .with_context(|| format!("stat staging tensor {}", partial_path.display()))?;
    if !payload_metadata.is_file() || payload_metadata.len() != offer.descriptor.byte_len {
        bail!("staged tensor length does not match the admitted descriptor");
    }
    if fs::try_exists(final_path).await? {
        bail!("refusing to overwrite existing tensor payload {}", final_path.display());
    }

    let digest = sha256_file(partial_path).await?;
    let stored_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let manifest = TensorManifest {
        version: MANIFEST_VERSION,
        sender_node_id: offer.sender_node_id.clone(),
        offer_sequence: offer.sequence,
        message_id: offer.message_id,
        descriptor: offer.descriptor.clone(),
        byte_len: payload_metadata.len(),
        sha256: digest,
        payload_file: payload_name,
        stored_at_unix,
    };
    manifest.validate()?;

    let manifest_path = manifest_path_for_payload(final_path);
    if fs::try_exists(&manifest_path).await? {
        bail!("refusing to overwrite existing tensor manifest {}", manifest_path.display());
    }
    let manifest_name = manifest_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("tensor manifest path has no UTF-8 filename")?;
    let temporary_manifest = manifest_path.with_file_name(format!(
        "{manifest_name}.{}.part",
        uuid::Uuid::new_v4()
    ));
    let encoded = serde_json::to_vec_pretty(&manifest)?;
    if encoded.len() as u64 > MAX_MANIFEST_BYTES {
        bail!("tensor manifest exceeds the local size limit");
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_manifest)
        .await
        .with_context(|| format!("creating manifest staging file {}", temporary_manifest.display()))?;
    if let Err(error) = file.write_all(&encoded).await {
        drop(file);
        let _ = fs::remove_file(&temporary_manifest).await;
        return Err(error).context("writing tensor manifest staging file");
    }
    if let Err(error) = file.sync_all().await {
        drop(file);
        let _ = fs::remove_file(&temporary_manifest).await;
        return Err(error).context("syncing tensor manifest staging file");
    }
    drop(file);

    if let Err(error) = fs::rename(partial_path, final_path).await {
        let _ = fs::remove_file(&temporary_manifest).await;
        return Err(error).with_context(|| format!("committing tensor payload {}", final_path.display()));
    }
    if let Err(error) = fs::rename(&temporary_manifest, &manifest_path).await {
        let _ = fs::remove_file(&temporary_manifest).await;
        let _ = fs::remove_file(final_path).await;
        return Err(error).with_context(|| format!("publishing tensor manifest {}", manifest_path.display()));
    }

    Ok(manifest)
}

/// Scan the local store without hashing payloads on each dashboard refresh.
/// The SHA-256 digest is retained for an explicit runtime/loader verification.
pub async fn scan_store(directory: &Path) -> Result<TensorStoreSnapshot> {
    if !fs::try_exists(directory).await? {
        return Ok(TensorStoreSnapshot::default());
    }

    let mut entries = fs::read_dir(directory)
        .await
        .with_context(|| format!("reading tensor store {}", directory.display()))?;
    let mut manifest_paths = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if file_type.is_file()
            && entry.path().extension().and_then(|value| value.to_str()) == Some("json")
        {
            manifest_paths.push(entry.path());
            if manifest_paths.len() >= MAX_MANIFESTS_TO_SCAN {
                break;
            }
        }
    }
    manifest_paths.sort();

    let mut snapshot = TensorStoreSnapshot::default();
    for manifest_path in manifest_paths {
        let metadata = match fs::symlink_metadata(&manifest_path).await {
            Ok(metadata) if metadata.file_type().is_file() => metadata,
            _ => continue,
        };
        if metadata.len() == 0 || metadata.len() > MAX_MANIFEST_BYTES {
            continue;
        }
        let encoded = match fs::read(&manifest_path).await {
            Ok(encoded) => encoded,
            Err(_) => continue,
        };
        let manifest: TensorManifest = match serde_json::from_slice(&encoded) {
            Ok(manifest) if manifest.validate().is_ok() => manifest,
            _ => continue,
        };

        let payload_path = directory.join(&manifest.payload_file);
        let payload_metadata = match fs::symlink_metadata(&payload_path).await {
            Ok(metadata) if metadata.file_type().is_file() => metadata,
            _ => continue,
        };
        if payload_metadata.len() != manifest.byte_len {
            continue;
        }

        snapshot.count = snapshot.count.saturating_add(1);
        snapshot.total_bytes = snapshot.total_bytes.saturating_add(manifest.byte_len);
        snapshot.recent.push(manifest);
    }

    snapshot.recent.sort_by(|left, right| {
        right.stored_at_unix.cmp(&left.stored_at_unix)
            .then_with(|| left.payload_file.cmp(&right.payload_file))
    });
    snapshot.recent.truncate(MAX_RECENT_TENSORS);
    Ok(snapshot)
}

/// Recompute the payload digest before handing an artifact to a runtime.
pub async fn verify_payload(directory: &Path, manifest: &TensorManifest) -> Result<bool> {
    manifest.validate()?;
    let payload_path = directory.join(&manifest.payload_file);
    let metadata = match fs::symlink_metadata(&payload_path).await {
        Ok(metadata) if metadata.file_type().is_file() => metadata,
        _ => return Ok(false),
    };
    if metadata.len() != manifest.byte_len {
        return Ok(false);
    }
    Ok(sha256_file(&payload_path).await? == manifest.sha256)
}

pub async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("opening tensor payload for SHA-256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}")?;
    }
    Ok(hex)
}

fn is_safe_payload_basename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value.ends_with(".tensor")
        && value != ".tensor"
        && value != ".."
        && !value.contains('/')
        && !value.contains('\\')
        && !value.contains(':')
        && !value.chars().any(|character| character.is_control())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorDType, TensorEncoding, TensorLayout};

    fn sample_manifest(payload_file: &str, byte_len: u64) -> TensorManifest {
        TensorManifest {
            version: MANIFEST_VERSION,
            sender_node_id: "sender-node".into(),
            offer_sequence: 1,
            message_id: 9,
            descriptor: TensorDescriptor {
                tensor_id: 5,
                name: "layers.0.attention.weight".into(),
                dtype: TensorDType::U8,
                shape: vec![byte_len],
                layout: TensorLayout::RowMajor,
                encoding: TensorEncoding::Raw,
                byte_len,
            },
            byte_len,
            sha256: "a".repeat(64),
            payload_file: payload_file.into(),
            stored_at_unix: 1_800_000_000,
        }
    }

    #[test]
    fn validates_manifest_and_rejects_unsafe_payload_paths() {
        assert!(sample_manifest("weights.tensor", 32).validate().is_ok());
        for path in ["../outside.tensor", "subdir/file.tensor", r"..\\outside.tensor", "C:outside.tensor"] {
            assert!(sample_manifest(path, 32).validate().is_err(), "accepted {path}");
        }
    }

    #[tokio::test]
    async fn detects_payload_changes_after_manifest_creation() {
        let root = std::env::temp_dir().join(format!("aitp-verify-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).await.unwrap();
        let payload = root.join("weights.tensor");
        fs::write(&payload, b"good").await.unwrap();
        let mut manifest = sample_manifest("weights.tensor", 4);
        manifest.sha256 = sha256_file(&payload).await.unwrap();
        assert!(verify_payload(&root, &manifest).await.unwrap());
        fs::write(&payload, b"evil").await.unwrap();
        assert!(!verify_payload(&root, &manifest).await.unwrap());
        fs::remove_dir_all(&root).await.unwrap();
    }

    #[test]
    fn rejects_manifest_with_descriptor_length_mismatch() {
        let mut manifest = sample_manifest("weights.tensor", 32);
        manifest.byte_len = 31;
        assert!(manifest.validate().is_err());
    }

    #[tokio::test]
    async fn catalog_ignores_missing_or_wrong_length_payloads() {
        let root = std::env::temp_dir().join(format!("aitp-catalog-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).await.unwrap();

        let valid = sample_manifest("valid.tensor", 4);
        fs::write(root.join("valid.tensor"), b"data").await.unwrap();
        fs::write(root.join("valid.json"), serde_json::to_vec(&valid).unwrap()).await.unwrap();

        let missing = sample_manifest("missing.tensor", 4);
        fs::write(root.join("missing.json"), serde_json::to_vec(&missing).unwrap()).await.unwrap();

        let wrong_length = sample_manifest("short.tensor", 4);
        fs::write(root.join("short.tensor"), b"bad").await.unwrap();
        fs::write(root.join("short.json"), serde_json::to_vec(&wrong_length).unwrap()).await.unwrap();

        let snapshot = scan_store(&root).await.unwrap();
        assert_eq!(snapshot.count, 1);
        assert_eq!(snapshot.total_bytes, 4);
        assert_eq!(snapshot.recent[0].payload_file, "valid.tensor");

        fs::remove_dir_all(&root).await.unwrap();
    }
}
