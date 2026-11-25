//! Read-only HTTP dashboard for the cluster manager. No control actions are exposed.
use axum::{extract::State, response::Html, routing::get, Json, Router};
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::RwLock};

pub const DASHBOARD_HTML: &str = include_str!("../web/dashboard.html");
pub type SharedSnapshot = Arc<RwLock<ClusterSnapshot>>;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ClusterSnapshot {
    pub nodes: Vec<DashboardNode>,
    pub online_nodes: usize,
    pub offline_nodes: usize,
    /// Manifests and payloads visible in the dashboard host's local tensor directory.
    pub tensor_store: crate::storage::TensorStoreSnapshot,
    /// Local GGUF headers/tensor tables found under AITP_MODEL_DIR.
    pub model_store: crate::gguf::GgufModelCatalog,
    pub updated_at_unix: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardNode {
    pub node_id: String,
    pub hostname: String,
    pub address: String,
    /// Authenticated worker process-session UUID used to reject stale plans.
    pub session_id: String,
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub total_memory_bytes: u64,
    /// Raw OS-available RAM minus current AITP logical reservations.
    pub available_memory_bytes: u64,
    pub reserved_memory_bytes: u64,
    pub accelerators: Vec<crate::control::AcceleratorInfo>,
    pub online: bool,
    pub last_seen_secs: u64,
}

pub fn shared_snapshot() -> SharedSnapshot {
    Arc::new(RwLock::new(ClusterSnapshot::default()))
}

pub async fn serve(bind: SocketAddr, snapshot: SharedSnapshot) -> anyhow::Result<()> {
    let model_snapshot = snapshot.clone();
    tokio::spawn(async move {
        let mut refresh = tokio::time::interval(Duration::from_secs(30));
        loop {
            refresh.tick().await;
            let directory = crate::gguf::configured_model_dir();
            let scanned =
                tokio::task::spawn_blocking(move || crate::gguf::scan_model_dir(&directory)).await;
            match scanned {
                Ok(Ok(catalog)) => {
                    model_snapshot.write().await.model_store = catalog;
                }
                Ok(Err(error)) => {
                    eprintln!("AITP GGUF catalog scan failed: {error:#}");
                }
                Err(error) => {
                    eprintln!("AITP GGUF catalog task failed: {error}");
                }
            }
        }
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/cluster", get(cluster))
        .with_state(snapshot);
    let listener = TcpListener::bind(bind).await?;
    println!(
        "AITP dashboard listening at http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn cluster(State(snapshot): State<SharedSnapshot>) -> Json<ClusterSnapshot> {
    Json(snapshot.read().await.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dashboard_snapshot_is_serializable_and_starts_empty() {
        let snapshot = shared_snapshot();
        let value = serde_json::to_value(snapshot.read().await.clone()).unwrap();
        assert_eq!(value["nodes"].as_array().unwrap().len(), 0);
        assert_eq!(value["online_nodes"], 0);
        assert_eq!(value["tensor_store"]["count"], 0);
        assert_eq!(value["tensor_store"]["recent"].as_array().unwrap().len(), 0);
        assert_eq!(value["model_store"]["models"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn dashboard_snapshot_reports_session_and_reserved_memory() {
        let node = DashboardNode {
            node_id: "node-id".into(),
            hostname: "worker".into(),
            address: "192.0.2.10:9001".into(),
            session_id: "00000000-0000-4000-8000-000000000042".into(),
            os: "linux".into(),
            arch: "x86_64".into(),
            cpu_model: "Test CPU".into(),
            logical_cpus: 8,
            total_memory_bytes: 4096,
            available_memory_bytes: 1024,
            reserved_memory_bytes: 512,
            accelerators: Vec::new(),
            online: true,
            last_seen_secs: 0,
        };
        let value = serde_json::to_value(node).unwrap();
        assert_eq!(value["session_id"], "00000000-0000-4000-8000-000000000042");
        assert_eq!(value["available_memory_bytes"], 1024);
        assert_eq!(value["reserved_memory_bytes"], 512);
        assert!(DASHBOARD_HTML.contains("Memory free / reserved"));
    }

    #[test]
    fn dashboard_contains_read_only_routes_only() {
        assert!(DASHBOARD_HTML.contains("/api/cluster"));
        assert!(!DASHBOARD_HTML.contains("eval("));
        assert!(DASHBOARD_HTML.contains("Stored tensor artifacts"));
        assert!(DASHBOARD_HTML.contains("s.tensor_store"));
        assert!(DASHBOARD_HTML.contains("Local GGUF models"));
        assert!(DASHBOARD_HTML.contains("s.model_store"));
    }
}
