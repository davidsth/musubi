//! Read-only HTTP dashboard for the cluster manager. No control actions are exposed.
use axum::{extract::State, response::Html, routing::get, Json, Router};
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc};
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
    pub updated_at_unix: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardNode {
    pub node_id: String,
    pub hostname: String,
    pub address: String,
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub accelerators: Vec<crate::control::AcceleratorInfo>,
    pub online: bool,
    pub last_seen_secs: u64,
}

pub fn shared_snapshot() -> SharedSnapshot {
    Arc::new(RwLock::new(ClusterSnapshot::default()))
}

pub async fn serve(bind: SocketAddr, snapshot: SharedSnapshot) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/api/cluster", get(cluster))
        .with_state(snapshot);
    let listener = TcpListener::bind(bind).await?;
    println!("AITP dashboard listening at http://{}", listener.local_addr()?);
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
    }

    #[test]
    fn dashboard_contains_read_only_routes_only() {
        assert!(DASHBOARD_HTML.contains("/api/cluster"));
        assert!(!DASHBOARD_HTML.contains("eval("));
        assert!(DASHBOARD_HTML.contains("Stored tensor artifacts"));
        assert!(DASHBOARD_HTML.contains("s.tensor_store"));
    }
}
