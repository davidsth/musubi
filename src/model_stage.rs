//! Lease-bound GGUF artifact staging over a dedicated TCP stream.
//!
//! This artifact-staging channel is distinct from both the JSON UDP membership/
//! lease control plane and the optimized fixed-header UDP tensor data plane.
//! TCP provides ordered delivery and transport flow control for bulk files;
//! staged model files are published only after exact-length and SHA-256 checks.
use crate::allocation::NodeAllocationBook;
use anyhow::{bail, Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    time::timeout,
};
type HmacSha256 = Hmac<Sha256>;

pub const STAGE_PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_STAGE_PORT: u16 = 9004;
/// A conservative upper bound for a single GGUF transfer (large enough for
/// current very-large quantized models while limiting authenticated disk use).
pub const MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
const MAX_FRAME_BYTES: usize = 16 * 1024;
const HMAC_TAG_BYTES: usize = 32;
const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const IO_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_MANIFEST_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Ready,
    AlreadyPresent,
    Rejected,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRequest {
    pub version: u16,
    pub coordinator_node_id: String,
    pub target_node_id: String,
    pub target_session_id: String,
    pub lease_id: String,
    pub request_id: String,
    pub model_filename: String,
    pub model_sha256: String,
    pub model_bytes: u64,
    pub first_layer: Option<u64>,
    pub end_layer_exclusive: Option<u64>,
    pub auth_tag: Vec<u8>,
}

impl StageRequest {
    pub fn validate(&self) -> Result<()> {
        if self.version != STAGE_PROTOCOL_VERSION
            || !valid_uuid(&self.coordinator_node_id)
            || !valid_uuid(&self.target_node_id)
            || !valid_uuid(&self.target_session_id)
            || !valid_uuid(&self.lease_id)
            || !valid_uuid(&self.request_id)
            || !is_safe_gguf_name(&self.model_filename)
            || !is_lower_hex_digest(&self.model_sha256)
            || self.model_bytes == 0
            || self.model_bytes > MAX_MODEL_BYTES
        {
            bail!("model-stage request contains invalid or out-of-range metadata");
        }
        match (self.first_layer, self.end_layer_exclusive) {
            (Some(start), Some(end)) if start < end && end <= 4096 => {}
            (None, None) => {}
            _ => bail!("model-stage request contains an invalid layer range"),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReply {
    pub version: u16,
    pub coordinator_node_id: String,
    pub node_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub request_id: String,
    pub model_sha256: String,
    pub model_bytes: u64,
    pub status: StageStatus,
    pub payload_file: Option<String>,
    pub manifest_file: Option<String>,
    pub reason: Option<String>,
    pub auth_tag: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StagedModelManifest {
    pub version: u16,
    pub coordinator_node_id: String,
    pub node_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub model_filename: String,
    pub model_sha256: String,
    pub model_bytes: u64,
    pub payload_file: String,
    pub first_layer: Option<u64>,
    pub end_layer_exclusive: Option<u64>,
    pub staged_at_unix: u64,
}

impl StagedModelManifest {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || !valid_uuid(&self.coordinator_node_id)
            || !valid_uuid(&self.node_id)
            || !valid_uuid(&self.session_id)
            || !valid_uuid(&self.lease_id)
            || !is_safe_gguf_name(&self.model_filename)
            || !is_lower_hex_digest(&self.model_sha256)
            || self.model_bytes == 0
            || self.model_bytes > MAX_MODEL_BYTES
            || self.payload_file != format!("{}.gguf", self.model_sha256)
        {
            bail!("staged-model manifest contains invalid metadata");
        }
        match (self.first_layer, self.end_layer_exclusive) {
            (Some(start), Some(end)) if start < end && end <= 4096 => {}
            (None, None) => {}
            _ => bail!("staged-model manifest contains an invalid layer range"),
        }
        Ok(())
    }
}

pub fn configured_model_dir() -> PathBuf {
    std::env::var_os("AITP_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("aitp-models"))
}

pub async fn run_listener(
    listener: TcpListener,
    node_id: String,
    session_id: String,
    key: Vec<u8>,
    allocation_book: Arc<Mutex<NodeAllocationBook>>,
    model_dir: PathBuf,
) -> Result<()> {
    loop {
        let (stream, peer) = listener.accept().await?;
        // Serve one upload at a time. This bounds concurrent disk writers and
        // avoids an unbounded task queue for authenticated TCP connections.
        if let Err(error) = serve_connection(
            stream,
            peer,
            &node_id,
            &session_id,
            &key,
            allocation_book.clone(),
            &model_dir,
        )
        .await
        {
            eprintln!("AITP model-stage connection from {peer} ended: {error:#}");
        }
    }
}

async fn serve_connection(
    mut stream: TcpStream,
    _peer: SocketAddr,
    node_id: &str,
    session_id: &str,
    key: &[u8],
    allocation_book: Arc<Mutex<NodeAllocationBook>>,
    model_dir: &Path,
) -> Result<()> {
    stream.set_nodelay(true)?;
    // The listener handles one upload at a time; keep unauthenticated peers
    // from occupying it indefinitely with a partial JSON header.
    let request: StageRequest = timeout(Duration::from_secs(5), read_frame(&mut stream))
        .await
        .context("timed out waiting for an authenticated model-stage request")??;
    if !verify_stage_request(&request, key)? {
        bail!("model-stage request authentication failed");
    }
    request.validate()?;
    if request.target_node_id != node_id || request.target_session_id != session_id {
        return send_status(
            &mut stream,
            &request,
            node_id,
            session_id,
            key,
            StageStatus::Rejected,
            Some("request target does not match this worker process identity".into()),
            None,
            None,
        )
        .await;
    }

    let authorized = allocation_book.lock().await.authorize_model_stage(
        &request.coordinator_node_id,
        node_id,
        session_id,
        &request.lease_id,
        &request.model_sha256,
        request.first_layer,
        request.end_layer_exclusive,
        Instant::now(),
    );
    if let Err(reason) = authorized {
        send_status(
            &mut stream,
            &request,
            node_id,
            session_id,
            key,
            StageStatus::Rejected,
            Some(reason.to_string()),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    fs::create_dir_all(model_dir)
        .await
        .with_context(|| format!("creating model store {}", model_dir.display()))?;
    let payload_file = format!("{}.gguf", request.model_sha256);
    let manifest_file = format!("{}.{}.aitp.json", request.model_sha256, request.lease_id);
    let payload_path = model_dir.join(&payload_file);

    match verify_existing_payload(&payload_path, &request).await? {
        Some(true) => {
            let still_authorized = allocation_book.lock().await.authorize_model_stage(
                &request.coordinator_node_id,
                node_id,
                session_id,
                &request.lease_id,
                &request.model_sha256,
                request.first_layer,
                request.end_layer_exclusive,
                Instant::now(),
            );
            if let Err(reason) = still_authorized {
                return send_status(
                    &mut stream,
                    &request,
                    node_id,
                    session_id,
                    key,
                    StageStatus::Rejected,
                    Some(reason.to_string()),
                    None,
                    None,
                )
                .await;
            }
            publish_manifest(
                model_dir,
                &request,
                node_id,
                session_id,
                &payload_file,
                &manifest_file,
            )
            .await?;
            return send_status(
                &mut stream,
                &request,
                node_id,
                session_id,
                key,
                StageStatus::AlreadyPresent,
                None,
                Some(payload_file),
                Some(manifest_file),
            )
            .await;
        }
        Some(false) => {
            return send_status(
                &mut stream,
                &request,
                node_id,
                session_id,
                key,
                StageStatus::Rejected,
                Some("content-addressed model path exists but fails file-type, size, or SHA-256 verification".into()),
                None,
                None,
            )
            .await;
        }
        None => {}
    }

    let partial_path = model_dir.join(format!(
        ".{}.{}.part",
        request.model_sha256, request.request_id
    ));
    let mut staging_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)
        .await
        .with_context(|| format!("creating staging file {}", partial_path.display()))?;

    if let Err(error) = send_status(
        &mut stream,
        &request,
        node_id,
        session_id,
        key,
        StageStatus::Ready,
        None,
        None,
        None,
    )
    .await
    {
        drop(staging_file);
        let _ = fs::remove_file(&partial_path).await;
        return Err(error).context("notifying coordinator that model staging is ready");
    }

    if let Err(error) = receive_payload(&mut stream, &mut staging_file, &request).await {
        drop(staging_file);
        let _ = fs::remove_file(&partial_path).await;
        let _ = send_status(
            &mut stream,
            &request,
            node_id,
            session_id,
            key,
            StageStatus::Rejected,
            Some("payload transfer was incomplete or failed verification".into()),
            None,
            None,
        )
        .await;
        return Err(error).context("receiving GGUF artifact");
    }
    drop(staging_file);

    let still_authorized = allocation_book.lock().await.authorize_model_stage(
        &request.coordinator_node_id,
        node_id,
        session_id,
        &request.lease_id,
        &request.model_sha256,
        request.first_layer,
        request.end_layer_exclusive,
        Instant::now(),
    );
    if let Err(reason) = still_authorized {
        let _ = fs::remove_file(&partial_path).await;
        send_status(
            &mut stream,
            &request,
            node_id,
            session_id,
            key,
            StageStatus::Rejected,
            Some(format!("allocation lease is no longer active: {reason}")),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    // A single listener processes uploads sequentially. The existence check and
    // same-directory rename therefore prevent an AITP transfer from replacing
    // an already-published model artifact.
    if fs::try_exists(&payload_path).await? {
        let same = verify_existing_payload(&payload_path, &request).await? == Some(true);
        let _ = fs::remove_file(&partial_path).await;
        if !same {
            return send_status(
                &mut stream,
                &request,
                node_id,
                session_id,
                key,
                StageStatus::Rejected,
                Some("model destination appeared during staging and did not verify".into()),
                None,
                None,
            )
            .await;
        }
    } else {
        if let Err(error) = fs::rename(&partial_path, &payload_path).await {
            let _ = fs::remove_file(&partial_path).await;
            return Err(error)
                .with_context(|| format!("publishing model {}", payload_path.display()));
        }
    }

    publish_manifest(
        model_dir,
        &request,
        node_id,
        session_id,
        &payload_file,
        &manifest_file,
    )
    .await?;
    send_status(
        &mut stream,
        &request,
        node_id,
        session_id,
        key,
        StageStatus::Complete,
        None,
        Some(payload_file),
        Some(manifest_file),
    )
    .await
}

async fn receive_payload(
    stream: &mut TcpStream,
    output: &mut fs::File,
    request: &StageRequest,
) -> Result<()> {
    let mut remaining = request.model_bytes;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();

    while remaining > 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        timeout(IO_IDLE_TIMEOUT, stream.read_exact(&mut buffer[..count]))
            .await
            .context("timed out waiting for model bytes")??;
        output.write_all(&buffer[..count]).await?;
        hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    output.sync_all().await?;

    let actual = format!("{:x}", hasher.finalize());
    if actual != request.model_sha256 {
        bail!("received GGUF SHA-256 does not match the active allocation lease");
    }
    let metadata = output.metadata().await?;
    if !metadata.is_file() || metadata.len() != request.model_bytes {
        bail!("received GGUF file type or byte length is invalid");
    }
    Ok(())
}

async fn verify_existing_payload(path: &Path, request: &StageRequest) -> Result<Option<bool>> {
    let metadata = match fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_file() || metadata.len() != request.model_bytes {
        return Ok(Some(false));
    }
    Ok(Some(sha256_file(path).await? == request.model_sha256))
}

/// Open a regular file without following symbolic links or Windows reparse
/// points. The returned handle is then used for metadata checks and reads.
fn open_regular_file_no_follow(path: &Path) -> Result<std::fs::File> {
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
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }

    let file = options.open(path).with_context(|| {
        format!(
            "opening model file without following links {}",
            path.display()
        )
    })?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        bail!("model artifact path is not a regular file");
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!("refusing a Windows reparse-point model artifact");
        }
    }

    Ok(file)
}

async fn sha256_file(path: &Path) -> Result<String> {
    let source = open_regular_file_no_follow(path)
        .with_context(|| format!("opening model payload {}", path.display()))?;
    let mut file = fs::File::from_std(source);
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

async fn publish_manifest(
    directory: &Path,
    request: &StageRequest,
    node_id: &str,
    session_id: &str,
    payload_file: &str,
    manifest_file: &str,
) -> Result<()> {
    let manifest = StagedModelManifest {
        version: 1,
        coordinator_node_id: request.coordinator_node_id.clone(),
        node_id: node_id.to_owned(),
        session_id: session_id.to_owned(),
        lease_id: request.lease_id.clone(),
        model_filename: request.model_filename.clone(),
        model_sha256: request.model_sha256.clone(),
        model_bytes: request.model_bytes,
        payload_file: payload_file.to_owned(),
        first_layer: request.first_layer,
        end_layer_exclusive: request.end_layer_exclusive,
        staged_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    manifest.validate()?;

    let final_path = directory.join(manifest_file);
    match fs::symlink_metadata(&final_path).await {
        Ok(metadata) => {
            if !metadata.file_type().is_file()
                || metadata.len() == 0
                || metadata.len() > MAX_MANIFEST_BYTES as u64
            {
                bail!("existing staged-model manifest is not a bounded regular file");
            }
            let encoded = fs::read(&final_path).await?;
            let existing: StagedModelManifest = serde_json::from_slice(&encoded)
                .context("parsing existing staged-model manifest")?;
            existing.validate()?;
            if !same_manifest_binding(&existing, &manifest) {
                bail!("existing staged-model manifest is bound to different artifact or lease metadata");
            }
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("checking staged-model manifest {}", final_path.display())
            });
        }
    }

    let encoded = serde_json::to_vec_pretty(&manifest)?;
    if encoded.len() > MAX_MANIFEST_BYTES {
        bail!("staged-model manifest exceeds the size limit");
    }
    let temporary_path = directory.join(format!(".{}.{}.part", manifest_file, request.request_id));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .await
        .with_context(|| {
            format!(
                "creating manifest staging file {}",
                temporary_path.display()
            )
        })?;
    if let Err(error) = file.write_all(&encoded).await {
        drop(file);
        let _ = fs::remove_file(&temporary_path).await;
        return Err(error).context("writing staged-model manifest");
    }
    if let Err(error) = file.sync_all().await {
        drop(file);
        let _ = fs::remove_file(&temporary_path).await;
        return Err(error).context("syncing staged-model manifest");
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary_path, &final_path).await {
        let _ = fs::remove_file(&temporary_path).await;
        return Err(error)
            .with_context(|| format!("publishing staged-model manifest {}", final_path.display()));
    }
    Ok(())
}

fn same_manifest_binding(left: &StagedModelManifest, right: &StagedModelManifest) -> bool {
    left.version == right.version
        && left.coordinator_node_id == right.coordinator_node_id
        && left.node_id == right.node_id
        && left.session_id == right.session_id
        && left.lease_id == right.lease_id
        && left.model_filename == right.model_filename
        && left.model_sha256 == right.model_sha256
        && left.model_bytes == right.model_bytes
        && left.payload_file == right.payload_file
        && left.first_layer == right.first_layer
        && left.end_layer_exclusive == right.end_layer_exclusive
}

async fn send_status(
    stream: &mut TcpStream,
    request: &StageRequest,
    node_id: &str,
    session_id: &str,
    key: &[u8],
    status: StageStatus,
    reason: Option<String>,
    payload_file: Option<String>,
    manifest_file: Option<String>,
) -> Result<()> {
    let mut reply = StageReply {
        version: STAGE_PROTOCOL_VERSION,
        coordinator_node_id: request.coordinator_node_id.clone(),
        node_id: node_id.to_owned(),
        session_id: session_id.to_owned(),
        lease_id: request.lease_id.clone(),
        request_id: request.request_id.clone(),
        model_sha256: request.model_sha256.clone(),
        model_bytes: request.model_bytes,
        status,
        payload_file,
        manifest_file,
        reason,
        auth_tag: Vec::new(),
    };
    sign_stage_reply(&mut reply, key)?;
    write_frame(stream, &reply).await
}

pub async fn send_to_worker(
    target: SocketAddr,
    model_path: &Path,
    mut request: StageRequest,
    key: &[u8],
) -> Result<StageReply> {
    // Open the source once without following symbolic links/reparse points, then
    // keep this handle for the entire transfer so validation and reads use the
    // same file object.
    let source_file = open_regular_file_no_follow(model_path)
        .with_context(|| format!("inspecting source model {}", model_path.display()))?;
    let mut source = fs::File::from_std(source_file);
    let metadata = source.metadata().await?;
    if !metadata.is_file() || metadata.len() != request.model_bytes {
        bail!(
            "source model must be a regular file with the exact byte length in the staging request"
        );
    }
    request.auth_tag.clear();
    sign_stage_request(&mut request, key)?;

    let mut stream = timeout(CONNECT_TIMEOUT, TcpStream::connect(target))
        .await
        .context("timed out connecting to worker model-stage port")??;
    stream.set_nodelay(true)?;
    write_frame(&mut stream, &request).await?;

    let ready: StageReply = read_frame(&mut stream).await?;
    verify_reply_for_request(&ready, &request, key)?;
    match ready.status {
        StageStatus::AlreadyPresent => return Ok(ready),
        StageStatus::Rejected => bail!(
            "worker {} rejected model staging: {}",
            ready.node_id,
            ready.reason.as_deref().unwrap_or("no reason supplied")
        ),
        StageStatus::Ready => {}
        StageStatus::Complete => bail!("worker returned COMPLETE before receiving the model bytes"),
    }

    let mut remaining = request.model_bytes;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    while remaining > 0 {
        let capacity = remaining.min(buffer.len() as u64) as usize;
        let count = source.read(&mut buffer[..capacity]).await?;
        if count == 0 {
            bail!("source model became shorter while staging");
        }
        timeout(IO_IDLE_TIMEOUT, stream.write_all(&buffer[..count]))
            .await
            .context("timed out sending model bytes to worker")??;
        hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    let source_digest = format!("{:x}", hasher.finalize());
    if source_digest != request.model_sha256 {
        bail!("source model changed after lease verification; worker will not commit mismatched bytes");
    }

    let completed: StageReply = read_frame(&mut stream).await?;
    verify_reply_for_request(&completed, &request, key)?;
    match completed.status {
        StageStatus::Complete | StageStatus::AlreadyPresent => Ok(completed),
        StageStatus::Rejected => bail!(
            "worker {} rejected model staging: {}",
            completed.node_id,
            completed.reason.as_deref().unwrap_or("no reason supplied")
        ),
        StageStatus::Ready => bail!("worker did not complete model staging"),
    }
}

fn verify_reply_for_request(reply: &StageReply, request: &StageRequest, key: &[u8]) -> Result<()> {
    if !verify_stage_reply(reply, key)? {
        bail!("worker returned an invalid model-stage response signature");
    }
    if reply.version != STAGE_PROTOCOL_VERSION
        || reply.coordinator_node_id != request.coordinator_node_id
        || reply.node_id != request.target_node_id
        || reply.session_id != request.target_session_id
        || reply.lease_id != request.lease_id
        || reply.request_id != request.request_id
        || reply.model_sha256 != request.model_sha256
        || reply.model_bytes != request.model_bytes
    {
        bail!("worker model-stage response does not match the authenticated request");
    }
    if matches!(
        reply.status,
        StageStatus::Complete | StageStatus::AlreadyPresent
    ) {
        let expected_payload = format!("{}.gguf", request.model_sha256);
        let expected_manifest = format!("{}.{}.aitp.json", request.model_sha256, request.lease_id);
        if reply.payload_file.as_deref() != Some(expected_payload.as_str())
            || reply.manifest_file.as_deref() != Some(expected_manifest.as_str())
        {
            bail!("worker model-stage receipt contains unexpected artifact paths");
        }
    }
    Ok(())
}

pub fn sign_stage_request(request: &mut StageRequest, key: &[u8]) -> Result<()> {
    request.validate()?;
    request.auth_tag = request_mac(request, key)?.finalize().into_bytes().to_vec();
    Ok(())
}

pub fn verify_stage_request(request: &StageRequest, key: &[u8]) -> Result<bool> {
    if request.auth_tag.len() != HMAC_TAG_BYTES || request.validate().is_err() {
        return Ok(false);
    }
    Ok(request_mac(request, key)?
        .verify_slice(&request.auth_tag)
        .is_ok())
}

pub fn sign_stage_reply(reply: &mut StageReply, key: &[u8]) -> Result<()> {
    reply.auth_tag = reply_mac(reply, key)?.finalize().into_bytes().to_vec();
    Ok(())
}

pub fn verify_stage_reply(reply: &StageReply, key: &[u8]) -> Result<bool> {
    if reply.auth_tag.len() != HMAC_TAG_BYTES {
        return Ok(false);
    }
    Ok(reply_mac(reply, key)?.verify_slice(&reply.auth_tag).is_ok())
}

fn request_mac(request: &StageRequest, key: &[u8]) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/MODEL_STAGE_REQUEST/V1\0")?;
    mac.update(&request.version.to_be_bytes());
    update_blob(&mut mac, request.coordinator_node_id.as_bytes());
    update_blob(&mut mac, request.target_node_id.as_bytes());
    update_blob(&mut mac, request.target_session_id.as_bytes());
    update_blob(&mut mac, request.lease_id.as_bytes());
    update_blob(&mut mac, request.request_id.as_bytes());
    update_blob(&mut mac, request.model_filename.as_bytes());
    update_blob(&mut mac, request.model_sha256.as_bytes());
    mac.update(&request.model_bytes.to_be_bytes());
    update_option_u64(&mut mac, request.first_layer);
    update_option_u64(&mut mac, request.end_layer_exclusive);
    Ok(mac)
}

fn reply_mac(reply: &StageReply, key: &[u8]) -> Result<HmacSha256> {
    let mut mac = new_mac(key, b"AITP/MODEL_STAGE_REPLY/V1\0")?;
    mac.update(&reply.version.to_be_bytes());
    update_blob(&mut mac, reply.coordinator_node_id.as_bytes());
    update_blob(&mut mac, reply.node_id.as_bytes());
    update_blob(&mut mac, reply.session_id.as_bytes());
    update_blob(&mut mac, reply.lease_id.as_bytes());
    update_blob(&mut mac, reply.request_id.as_bytes());
    update_blob(&mut mac, reply.model_sha256.as_bytes());
    mac.update(&reply.model_bytes.to_be_bytes());
    mac.update(&[match reply.status {
        StageStatus::Ready => 1,
        StageStatus::AlreadyPresent => 2,
        StageStatus::Rejected => 3,
        StageStatus::Complete => 4,
    }]);
    update_option_blob(&mut mac, reply.payload_file.as_deref().map(str::as_bytes));
    update_option_blob(&mut mac, reply.manifest_file.as_deref().map(str::as_bytes));
    update_option_blob(&mut mac, reply.reason.as_deref().map(str::as_bytes));
    Ok(mac)
}

fn new_mac(key: &[u8], domain: &[u8]) -> Result<HmacSha256> {
    if key.len() < 32 {
        bail!("AITP cluster key must contain at least 32 bytes");
    }
    let mut mac =
        HmacSha256::new_from_slice(key).map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(domain);
    Ok(mac)
}

fn update_blob(mac: &mut HmacSha256, bytes: &[u8]) {
    mac.update(&(bytes.len() as u64).to_be_bytes());
    mac.update(bytes);
}

fn update_option_blob(mac: &mut HmacSha256, bytes: Option<&[u8]>) {
    match bytes {
        Some(bytes) => {
            mac.update(&[1]);
            update_blob(mac, bytes);
        }
        None => mac.update(&[0]),
    }
}

fn update_option_u64(mac: &mut HmacSha256, value: Option<u64>) {
    match value {
        Some(value) => {
            mac.update(&[1]);
            mac.update(&value.to_be_bytes());
        }
        None => mac.update(&[0]),
    }
}

async fn write_frame<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let encoded = serde_json::to_vec(value)?;
    if encoded.is_empty() || encoded.len() > MAX_FRAME_BYTES {
        bail!("model-stage control frame exceeds the bounded header size");
    }
    let length = u32::try_from(encoded.len()).context("model-stage control frame is too large")?;
    stream.write_all(&length.to_be_bytes()).await?;
    stream.write_all(&encoded).await?;
    stream.flush().await?;
    Ok(())
}

async fn read_frame<T: for<'de> Deserialize<'de>>(stream: &mut TcpStream) -> Result<T> {
    let mut length = [0u8; 4];
    timeout(IO_IDLE_TIMEOUT, stream.read_exact(&mut length))
        .await
        .context("timed out waiting for model-stage control frame length")??;
    let length = u32::from_be_bytes(length) as usize;
    if length == 0 || length > MAX_FRAME_BYTES {
        bail!("model-stage control frame length is invalid");
    }
    let mut encoded = vec![0u8; length];
    timeout(IO_IDLE_TIMEOUT, stream.read_exact(&mut encoded))
        .await
        .context("timed out waiting for model-stage control frame")??;
    Ok(serde_json::from_slice(&encoded).context("parsing model-stage control frame")?)
}

fn valid_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok()
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_safe_gguf_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && !value.starts_with('.')
        && value.to_ascii_lowercase().ends_with(".gguf")
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'.' | b'_' | b'-' | b' ' | b'(' | b')' | b'[' | b']' | b'+'
                )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocation::unix_now,
        auth::{sign_allocation_prepare, sign_allocation_release},
        control::{AllocationPrepare, AllocationRelease, CONTROL_VERSION},
    };

    fn signed_prepare(
        key: &[u8],
        coordinator: &str,
        node: &str,
        session: &str,
        lease: &str,
        digest: &str,
        first_layer: Option<u64>,
        end_layer_exclusive: Option<u64>,
    ) -> AllocationPrepare {
        let mut prepare = AllocationPrepare {
            version: CONTROL_VERSION,
            coordinator_node_id: coordinator.to_owned(),
            target_node_id: node.to_owned(),
            target_session_id: session.to_owned(),
            lease_id: lease.to_owned(),
            model_sha256: digest.to_owned(),
            first_layer,
            end_layer_exclusive,
            reserved_bytes: 1024,
            expires_at_unix: unix_now().unwrap() + 120,
            auth_tag: Vec::new(),
        };
        sign_allocation_prepare(&mut prepare, key).unwrap();
        prepare
    }

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("aitp-model-stage-{}", uuid::Uuid::new_v4()))
    }

    fn make_request(
        coordinator: &str,
        node: &str,
        session: &str,
        lease: &str,
        digest: &str,
        bytes: u64,
        first_layer: Option<u64>,
        end_layer_exclusive: Option<u64>,
    ) -> StageRequest {
        StageRequest {
            version: STAGE_PROTOCOL_VERSION,
            coordinator_node_id: coordinator.to_owned(),
            target_node_id: node.to_owned(),
            target_session_id: session.to_owned(),
            lease_id: lease.to_owned(),
            request_id: uuid::Uuid::new_v4().to_string(),
            model_filename: "test-model.gguf".into(),
            model_sha256: digest.to_owned(),
            model_bytes: bytes,
            first_layer,
            end_layer_exclusive,
            auth_tag: Vec::new(),
        }
    }

    fn make_book(
        key: &[u8],
        coordinator: &str,
        node: &str,
        session: &str,
        lease: &str,
        digest: &str,
        first_layer: Option<u64>,
        end_layer_exclusive: Option<u64>,
    ) -> NodeAllocationBook {
        let now = Instant::now();
        let mut book = NodeAllocationBook::default();
        let prepare = signed_prepare(
            key,
            coordinator,
            node,
            session,
            lease,
            digest,
            first_layer,
            end_layer_exclusive,
        );
        book.prepare(&prepare, node, session, key, 4096, now, unix_now().unwrap())
            .unwrap();
        book
    }

    #[test]
    fn stage_request_authentication_binds_model_lease_and_layer_range() {
        let key = [0x73; 32];
        let coordinator = uuid::Uuid::new_v4().to_string();
        let node = uuid::Uuid::new_v4().to_string();
        let session = uuid::Uuid::new_v4().to_string();
        let lease = uuid::Uuid::new_v4().to_string();
        let mut request = make_request(
            &coordinator,
            &node,
            &session,
            &lease,
            &"a".repeat(64),
            2048,
            Some(0),
            Some(4),
        );
        sign_stage_request(&mut request, &key).unwrap();
        assert!(verify_stage_request(&request, &key).unwrap());
        request.model_bytes += 1;
        assert!(!verify_stage_request(&request, &key).unwrap());
    }

    #[tokio::test]
    async fn stage_request_must_name_the_actual_worker_and_session() {
        let key = vec![0x44; 32];
        let coordinator = uuid::Uuid::new_v4().to_string();
        let node = uuid::Uuid::new_v4().to_string();
        let session = uuid::Uuid::new_v4().to_string();
        let lease = uuid::Uuid::new_v4().to_string();
        let bytes = b"worker identity binding".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let book = Arc::new(Mutex::new(make_book(
            &key,
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            Some(0),
            Some(2),
        )));
        let root = temp_root();
        let model_dir = root.join("models");

        for (target_node_id, target_session_id) in [
            (uuid::Uuid::new_v4().to_string(), session.clone()),
            (node.clone(), uuid::Uuid::new_v4().to_string()),
        ] {
            let mut request = make_request(
                &coordinator,
                &node,
                &session,
                &lease,
                &digest,
                bytes.len() as u64,
                Some(0),
                Some(2),
            );
            request.target_node_id = target_node_id;
            request.target_session_id = target_session_id;
            sign_stage_request(&mut request, &key).unwrap();

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server_book = book.clone();
            let server_key = key.clone();
            let server_node = node.clone();
            let server_session = session.clone();
            let server_dir = model_dir.clone();
            let server = tokio::spawn(async move {
                let (stream, peer) = listener.accept().await.unwrap();
                serve_connection(
                    stream,
                    peer,
                    &server_node,
                    &server_session,
                    &server_key,
                    server_book,
                    &server_dir,
                )
                .await
                .unwrap();
            });

            let mut stream = TcpStream::connect(address).await.unwrap();
            write_frame(&mut stream, &request).await.unwrap();
            let reply: StageReply = read_frame(&mut stream).await.unwrap();
            assert_eq!(reply.status, StageStatus::Rejected);
            assert!(verify_stage_reply(&reply, &key).unwrap());
            server.await.unwrap();
        }

        assert!(!fs::try_exists(&model_dir).await.unwrap());
    }

    #[tokio::test]
    async fn tcp_staging_verifies_and_publishes_content_addressed_model() {
        let key = vec![0x38; 32];
        let coordinator = uuid::Uuid::new_v4().to_string();
        let node = uuid::Uuid::new_v4().to_string();
        let session = uuid::Uuid::new_v4().to_string();
        let lease = uuid::Uuid::new_v4().to_string();
        let bytes = vec![0x5a; 2 * 1024 * 1024 + 37];
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let request = make_request(
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            bytes.len() as u64,
            Some(0),
            Some(2),
        );
        let book = Arc::new(Mutex::new(make_book(
            &key,
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            Some(0),
            Some(2),
        )));
        let root = temp_root();
        fs::create_dir_all(&root).await.unwrap();
        let source = root.join("source.gguf");
        fs::write(&source, &bytes).await.unwrap();
        let model_dir = root.join("models");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_book = book.clone();
        let server_key = key.clone();
        let server_node = node.clone();
        let server_session = session.clone();
        let server_dir = model_dir.clone();
        let server = tokio::spawn(async move {
            let (stream, peer) = listener.accept().await.unwrap();
            serve_connection(
                stream,
                peer,
                &server_node,
                &server_session,
                &server_key,
                server_book,
                &server_dir,
            )
            .await
            .unwrap();
        });

        let reply = send_to_worker(address, &source, request.clone(), &key)
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(reply.status, StageStatus::Complete);
        let payload = model_dir.join(format!("{}.gguf", digest));
        assert_eq!(sha256_file(&payload).await.unwrap(), digest);
        let manifest_path = model_dir.join(format!("{}.{}.aitp.json", digest, lease));
        let encoded = fs::read(&manifest_path).await.unwrap();
        let manifest: StagedModelManifest = serde_json::from_slice(&encoded).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.model_sha256, digest);
        assert_eq!(manifest.model_bytes, bytes.len() as u64);

        let mut retry = request;
        retry.request_id = uuid::Uuid::new_v4().to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_book = book.clone();
        let server_key = key.clone();
        let server_node = node.clone();
        let server_session = session.clone();
        let server_dir = model_dir.clone();
        let server = tokio::spawn(async move {
            let (stream, peer) = listener.accept().await.unwrap();
            serve_connection(
                stream,
                peer,
                &server_node,
                &server_session,
                &server_key,
                server_book,
                &server_dir,
            )
            .await
            .unwrap();
        });
        let repeated = send_to_worker(address, &source, retry, &key).await.unwrap();
        server.await.unwrap();
        assert_eq!(repeated.status, StageStatus::AlreadyPresent);
        fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn released_lease_during_upload_prevents_publication() {
        let key = vec![0x43; 32];
        let coordinator = uuid::Uuid::new_v4().to_string();
        let node = uuid::Uuid::new_v4().to_string();
        let session = uuid::Uuid::new_v4().to_string();
        let lease = uuid::Uuid::new_v4().to_string();
        let bytes = b"payload is complete but its lease is revoked".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let request = make_request(
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            bytes.len() as u64,
            Some(0),
            Some(2),
        );
        let book = Arc::new(Mutex::new(make_book(
            &key,
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            Some(0),
            Some(2),
        )));
        let root = temp_root();
        fs::create_dir_all(&root).await.unwrap();
        let model_dir = root.join("models");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_book = book.clone();
        let server_key = key.clone();
        let server_node = node.clone();
        let server_session = session.clone();
        let server_dir = model_dir.clone();
        let server = tokio::spawn(async move {
            let (stream, peer) = listener.accept().await.unwrap();
            serve_connection(
                stream,
                peer,
                &server_node,
                &server_session,
                &server_key,
                server_book,
                &server_dir,
            )
            .await
            .unwrap();
        });

        let mut signed = request.clone();
        sign_stage_request(&mut signed, &key).unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        write_frame(&mut stream, &signed).await.unwrap();
        let ready: StageReply = read_frame(&mut stream).await.unwrap();
        assert_eq!(ready.status, StageStatus::Ready);

        let mut release = AllocationRelease {
            version: CONTROL_VERSION,
            coordinator_node_id: coordinator,
            target_node_id: node,
            target_session_id: session,
            lease_id: lease,
            auth_tag: Vec::new(),
        };
        sign_allocation_release(&mut release, &key).unwrap();
        assert!(book
            .lock()
            .await
            .release(
                &release,
                &request.target_node_id,
                &request.target_session_id,
                &key,
                Instant::now(),
            )
            .unwrap());

        stream.write_all(&bytes).await.unwrap();
        let rejected: StageReply = read_frame(&mut stream).await.unwrap();
        assert_eq!(rejected.status, StageStatus::Rejected);
        assert!(verify_stage_reply(&rejected, &key).unwrap());
        server.await.unwrap();

        let mut entries = fs::read_dir(&model_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(!name.ends_with(".gguf"));
            assert!(!name.ends_with(".aitp.json"));
            assert!(!name.ends_with(".part"));
        }
        fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn interrupted_upload_does_not_publish_a_model_file() {
        let key = vec![0x42; 32];
        let coordinator = uuid::Uuid::new_v4().to_string();
        let node = uuid::Uuid::new_v4().to_string();
        let session = uuid::Uuid::new_v4().to_string();
        let lease = uuid::Uuid::new_v4().to_string();
        let bytes = b"model bytes that should not be committed".to_vec();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let request = make_request(
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            bytes.len() as u64,
            Some(0),
            Some(2),
        );
        let book = Arc::new(Mutex::new(make_book(
            &key,
            &coordinator,
            &node,
            &session,
            &lease,
            &digest,
            Some(0),
            Some(2),
        )));
        let root = temp_root();
        fs::create_dir_all(&root).await.unwrap();
        let model_dir = root.join("models");
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_book = book.clone();
        let server_key = key.clone();
        let server_node = node.clone();
        let server_session = session.clone();
        let server_dir = model_dir.clone();
        let server = tokio::spawn(async move {
            let (stream, peer) = listener.accept().await.unwrap();
            let _ = serve_connection(
                stream,
                peer,
                &server_node,
                &server_session,
                &server_key,
                server_book,
                &server_dir,
            )
            .await;
        });

        let mut signed = request.clone();
        sign_stage_request(&mut signed, &key).unwrap();
        let mut stream = TcpStream::connect(address).await.unwrap();
        write_frame(&mut stream, &signed).await.unwrap();
        let ready: StageReply = read_frame(&mut stream).await.unwrap();
        assert_eq!(ready.status, StageStatus::Ready);
        stream.write_all(&bytes[..8]).await.unwrap();
        drop(stream);
        server.await.unwrap();

        let mut entries = fs::read_dir(&model_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(!name.ends_with(".gguf"));
            assert!(!name.ends_with(".aitp.json"));
            assert!(!name.ends_with(".part"));
        }
        fs::remove_dir_all(root).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn staged_model_hash_refuses_symbolic_links() {
        let root = temp_root();
        fs::create_dir_all(&root).await.unwrap();
        let target = root.join("real-model.gguf");
        let link = root.join("linked-model.gguf");
        fs::write(&target, b"verified model bytes").await.unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert!(sha256_file(&link).await.is_err());

        fs::remove_dir_all(root).await.unwrap();
    }
}
