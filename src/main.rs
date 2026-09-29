use anyhow::{bail, Context, Result};
use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use clap::{Parser, Subcommand};
use image::imageops::FilterType;
use ndarray::{ArrayD, IxDyn};
use ort::{inputs, session::Session, value::TensorRef};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{net::TcpListener, sync::Mutex};

#[derive(Parser)]
#[command(name = "distributed-resnet")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start a worker that owns one pipeline stage.
    Worker {
        #[arg(long)]
        config: PathBuf,

        #[arg(long)]
        stage: String,

        /// Supply an approved local listen address at runtime.
        #[arg(long)]
        listen: String,
    },

    /// Send an image through the configured pipeline.
    Predict {
        #[arg(long)]
        config: PathBuf,

        #[arg(long)]
        image: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
struct PipelineConfig {
    model_version: String,
    stages: Vec<StageSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct StageSpec {
    stage: String,
    layers: Vec<String>,
    worker_group: String,
    endpoint: String,
    model_path: PathBuf,
    input_name: String,
    output_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorPacket {
    model_version: String,
    request_id: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

struct WorkerState {
    model_version: String,
    stage: StageSpec,
    session: Mutex<Session>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Worker {
            config,
            stage,
            listen,
        } => start_worker(&config, &stage, &listen).await,

        Command::Predict { config, image } => predict(&config, &image).await,
    }
}

async fn start_worker(config_path: &Path, stage_name: &str, listen: &str) -> Result<()> {
    let config = load_config(config_path)?;

    let stage = config
        .stages
        .iter()
        .find(|s| s.stage == stage_name)
        .cloned()
        .with_context(|| format!("Stage '{stage_name}' was not found in the configuration"))?;

    let session = Session::builder()?
        .commit_from_file(&stage.model_path)
        .with_context(|| {
            format!(
                "Could not load ONNX model for {} from {}",
                stage.stage,
                stage.model_path.display()
            )
        })?;

    println!(
        "Starting {} for layers [{}]",
        stage.stage,
        stage.layers.join(", ")
    );

    let state = Arc::new(WorkerState {
        model_version: config.model_version,
        stage,
        session: Mutex::new(session),
    });

    let app = Router::new()
        .route("/health", post(health))
        .route("/run", post(run_stage))
        .with_state(state);

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("Could not bind worker to '{listen}'"))?;

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "healthy")
}

async fn run_stage(
    State(state): State<Arc<WorkerState>>,
    body: Bytes,
) -> Result<Vec<u8>, (StatusCode, String)> {
    let packet: TensorPacket = bincode::deserialize(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid tensor packet: {e}")))?;

    if packet.model_version != state.model_version {
        return Err((
            StatusCode::CONFLICT,
            format!(
                "Model-version mismatch. Worker has '{}', request has '{}'",
                state.model_version, packet.model_version
            ),
        ));
    }

    let input = ArrayD::from_shape_vec(IxDyn(&packet.shape), packet.values)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid tensor shape: {e}")))?;

    let tensor = TensorRef::from_array_view(&input)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Cannot create ONNX tensor: {e}")))?;

    let mut session = state.session.lock().await;

    let outputs = session
        .run(inputs! {
            state.stage.input_name.as_str() => tensor
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference failed in {}: {e}", state.stage.stage),
            )
        })?;

    let output = outputs
        .get(state.stage.output_name.as_str())
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "Expected output '{}' was not returned by {}",
                    state.stage.output_name, state.stage.stage
                ),
            )
        })?;

    let (shape, values) = output.try_extract_tensor::<f32>().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Could not read model output: {e}"),
        )
    })?;

    let response = TensorPacket {
        model_version: state.model_version.clone(),
        request_id: packet.request_id,
        shape: shape.iter().map(|v| *v as usize).collect(),
        values: values.to_vec(),
    };

    bincode::serialize(&response).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Could not serialize output tensor: {e}"),
        )
    })
}

async fn predict(config_path: &Path, image_path: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let client = reqwest::Client::new();

    let initial_tensor = preprocess_image(image_path)?;

    let mut packet = TensorPacket {
        model_version: config.model_version.clone(),
        request_id: format!("request-{}", chrono_like_request_id()),
        shape: vec![1, 3, 224, 224],
        values: initial_tensor,
    };

    for stage in &config.stages {
        let request_body = bincode::serialize(&packet)?;

        let response = client
            .post(format!("{}/run", stage.endpoint))
            .body(request_body)
            .send()
            .await
            .with_context(|| format!("Could not reach {}", stage.stage))?;

        if !response.status().is_success() {
            bail!(
                "{} returned {}",
                stage.stage,
                response.status()
            );
        }

        let bytes = response.bytes().await?;
        packet = bincode::deserialize(&bytes)
            .with_context(|| format!("Invalid response from {}", stage.stage))?;

        println!(
            "{} complete: output shape {:?}",
            stage.stage, packet.shape
        );
    }

    let (class_index, score) = top_prediction(&packet.values)?;

    println!("Top ImageNet class index: {class_index}");
    println!("Logit score: {score:.4}");
    println!("Note: map the class index to an approved local ImageNet label file if labels are needed.");

    Ok(())
}

fn load_config(path: &Path) -> Result<PipelineConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("Could not read {}", path.display()))?;

    Ok(serde_yaml::from_str(&contents)?)
}

fn preprocess_image(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("Could not open image {}", path.display()))?
        .resize_exact(224, 224, FilterType::Triangle)
        .to_rgb8();

    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];

    let mut nchw = vec![0.0_f32; 3 * 224 * 224];

    for y in 0..224 {
        for x in 0..224 {
            let pixel = image.get_pixel(x, y);

            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                let normalized = (value - mean[channel]) / std[channel];
                let index = channel * 224 * 224 + y as usize * 224 + x as usize;
                nchw[index] = normalized;
            }
        }
    }

    Ok(nchw)
}

fn top_prediction(logits: &[f32]) -> Result<(usize, f32)> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| anyhow::anyhow!("The model returned no logits"))
}

fn chrono_like_request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .to_string()
}
