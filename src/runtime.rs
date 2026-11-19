//! Verified, streaming handoff of committed tensor artifacts to local runtimes.
//!
//! This module validates a manifest and opens its payload read-only. SHA-256 is
//! computed through that same open file handle, which is rewound before it is
//! returned. Runtime adapters should consume the returned reader instead of
//! reopening a path from the manifest.
use anyhow::{bail, Context, Result};
use std::path::Path;
use tokio::{fs, io::AsyncSeekExt};

use crate::{storage::TensorManifest, tensor::TensorDescriptor};

/// A committed tensor whose manifest, size, file type, and SHA-256 were checked.
///
/// The payload remains on disk and is exposed as a streaming Tokio file. No
/// tensor bytes are copied into a second in-memory buffer during preparation.
#[derive(Debug)]
pub struct VerifiedTensor {
    manifest: TensorManifest,
    payload: fs::File,
}

impl VerifiedTensor {
    pub fn manifest(&self) -> &TensorManifest {
        &self.manifest
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.manifest.descriptor
    }

    pub fn byte_len(&self) -> u64 {
        self.manifest.byte_len
    }

    /// Consume the prepared artifact and hand its already-verified reader to an
    /// adapter. The reader starts at byte offset zero.
    pub fn into_parts(self) -> (TensorManifest, fs::File) {
        (self.manifest, self.payload)
    }
}

/// Open a manifest's payload for streaming after checking it from the same
/// handle that is returned to the caller. The manifest filename is a validated
/// basename; symbolic links/reparse points are not accepted as payload files.
pub async fn open_verified_tensor(
    directory: &Path,
    manifest: &TensorManifest,
) -> Result<VerifiedTensor> {
    manifest.validate()?;
    let payload_path = directory.join(&manifest.payload_file);

    // Opening with no-follow flags is a blocking OS operation. Keep it off the
    // async executor, then hash and rewind the resulting handle asynchronously.
    let std_file = tokio::task::spawn_blocking(move || open_payload_readonly(&payload_path))
        .await
        .context("joining tensor payload open task")??;
    let mut payload = fs::File::from_std(std_file);

    let metadata = payload
        .metadata()
        .await
        .context("reading opened tensor payload metadata")?;
    if !metadata.is_file() {
        bail!("tensor payload is not a regular file");
    }
    if metadata.len() != manifest.byte_len {
        bail!(
            "tensor payload length {} does not match manifest length {}",
            metadata.len(),
            manifest.byte_len
        );
    }

    let actual_sha256 = crate::storage::sha256_open_file(&mut payload).await?;
    if actual_sha256 != manifest.sha256 {
        bail!("tensor payload SHA-256 does not match its manifest");
    }
    payload
        .seek(std::io::SeekFrom::Start(0))
        .await
        .context("rewinding verified tensor payload")?;

    Ok(VerifiedTensor {
        manifest: manifest.clone(),
        payload,
    })
}

fn open_payload_readonly(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Win32 FILE_FLAG_OPEN_REPARSE_POINT: open the reparse point itself
        // rather than following it. The file attributes are checked below.
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }

    #[cfg(not(any(unix, windows)))]
    {
        let metadata = std::fs::symlink_metadata(path)
            .with_context(|| format!("inspecting tensor payload {}", path.display()))?;
        if metadata.file_type().is_symlink() {
            bail!("refusing symbolic-link tensor payload");
        }
    }

    let file = options
        .open(path)
        .with_context(|| format!("opening tensor payload {}", path.display()))?;
    let metadata = file
        .metadata()
        .context("reading opened tensor payload metadata")?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        bail!("tensor payload must be a regular, non-symlink file");
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!("refusing Windows reparse-point tensor payload");
        }
    }

    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        storage::{sha256_file, MANIFEST_VERSION},
        tensor::{TensorDType, TensorEncoding, TensorLayout},
    };
    use tokio::io::AsyncReadExt;

    fn sample_manifest(payload_file: &str, byte_len: u64, sha256: String) -> TensorManifest {
        TensorManifest {
            version: MANIFEST_VERSION,
            sender_node_id: "sender-node".into(),
            offer_sequence: 1,
            message_id: 41,
            descriptor: TensorDescriptor {
                tensor_id: 7,
                name: "layers.0.attention.weight".into(),
                dtype: TensorDType::U8,
                shape: vec![byte_len],
                layout: TensorLayout::RowMajor,
                encoding: TensorEncoding::Raw,
                byte_len,
            },
            byte_len,
            sha256,
            payload_file: payload_file.into(),
            stored_at_unix: 1_800_000_000,
        }
    }

    async fn test_store() -> (std::path::PathBuf, Vec<u8>, TensorManifest) {
        let root = std::env::temp_dir().join(format!("aitp-runtime-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).await.unwrap();
        let data = b"verified tensor bytes for runtime handoff".to_vec();
        let payload_path = root.join("weights.tensor");
        fs::write(&payload_path, &data).await.unwrap();
        let digest = sha256_file(&payload_path).await.unwrap();
        let manifest = sample_manifest("weights.tensor", data.len() as u64, digest);
        (root, data, manifest)
    }

    #[tokio::test]
    async fn opens_verified_artifact_as_stream_without_loading_a_second_copy() {
        let (root, expected, manifest) = test_store().await;
        let prepared = open_verified_tensor(&root, &manifest).await.unwrap();
        assert_eq!(prepared.descriptor(), &manifest.descriptor);
        assert_eq!(prepared.byte_len(), expected.len() as u64);

        let (returned_manifest, mut reader) = prepared.into_parts();
        assert_eq!(returned_manifest, manifest);
        let mut actual = Vec::new();
        reader.read_to_end(&mut actual).await.unwrap();
        assert_eq!(actual, expected);
        drop(reader);
        fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn rejects_payload_modified_after_manifest_creation() {
        let (root, _, manifest) = test_store().await;
        fs::write(root.join("weights.tensor"), b"tampered").await.unwrap();
        assert!(open_verified_tensor(&root, &manifest).await.is_err());
        fs::remove_dir_all(root).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_symbolic_link_payloads() {
        use std::os::unix::fs::symlink;

        let (root, expected, mut manifest) = test_store().await;
        let original = root.join("weights.tensor");
        let linked = root.join("linked.tensor");
        symlink(&original, &linked).unwrap();
        manifest.payload_file = "linked.tensor".into();
        manifest.sha256 = sha256_file(&original).await.unwrap();
        manifest.byte_len = expected.len() as u64;
        manifest.descriptor.byte_len = expected.len() as u64;
        manifest.descriptor.shape = vec![expected.len() as u64];

        assert!(open_verified_tensor(&root, &manifest).await.is_err());
        fs::remove_dir_all(root).await.unwrap();
    }
}
