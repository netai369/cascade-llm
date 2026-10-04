use rusqlite::{params, Connection};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::types::ProviderConfig;
use crate::registry::{RegisterUpstreamRequest, UpstreamNode};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug)]
pub struct Db {
    conn: Mutex<Connection>,
}

#[allow(dead_code)]
impl Db {
    pub fn new(db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;
        let db = Self {
            conn: Mutex::new(conn),
        };
        db.init_tables()?;
        Ok(db)
    }

    pub fn new_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Mutex::new(conn),
        };
        db.init_tables()?;
        Ok(db)
    }

    fn init_tables(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                labels TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics_history(metric_name);
            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                config TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS upstreams (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                config TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_upstreams_role ON upstreams(role);
            ",
        )?;
        Ok(())
    }

    pub fn save_config(&self, key: &str, value: &str) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?1, ?2, ?3)",
            params![key, value, now],
        )?;
        Ok(())
    }

    pub fn load_config(&self, key: &str) -> Result<Option<String>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value FROM config WHERE key = ?1")?;
        let mut rows = stmt.query_map(params![key], |row| row.get::<_, String>(0))?;
        match rows.next() {
            Some(Ok(val)) => Ok(Some(val)),
            _ => Ok(None),
        }
    }

    pub fn record_metric(
        &self,
        name: &str,
        value: f64,
        labels: Option<&str>,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        conn.execute(
            "INSERT INTO metrics_history (timestamp, metric_name, metric_value, labels) VALUES (?1, ?2, ?3, ?4)",
            params![now, name, value, labels],
        )?;
        Ok(())
    }

    pub fn query_metrics(
        &self,
        name: &str,
        since_secs: u64,
    ) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let since = now - since_secs as i64;
        let mut stmt = conn.prepare(
            "SELECT timestamp, metric_value FROM metrics_history WHERE metric_name = ?1 AND timestamp >= ?2 ORDER BY timestamp",
        )?;
        let rows = stmt.query_map(params![name, since], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
    }

    pub fn cleanup_old_metrics(&self, retention_secs: u64) -> Result<usize, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let cutoff = now - retention_secs as i64;
        let deleted = conn.execute("DELETE FROM metrics_history WHERE timestamp < ?1", params![cutoff])?;
        if deleted > 0 {
            info!("Cleaned up {} old metric records (48h retention)", deleted);
        }
        Ok(deleted)
    }

    pub fn save_provider(&self, provider: &ProviderConfig) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let json = serde_json::to_string(provider)
            .map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO providers (id, config, updated_at) VALUES (?1, ?2, ?3)",
            params![provider.id, json, now],
        )?;
        Ok(())
    }

    pub fn delete_provider(&self, id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute("DELETE FROM providers WHERE id = ?1", params![id]);
    }

    pub fn load_providers(&self) -> Result<Vec<ProviderConfig>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT config FROM providers")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut providers = Vec::new();
        for row in rows {
            if let Ok(json) = row {
                if let Ok(provider) = serde_json::from_str::<ProviderConfig>(&json) {
                    providers.push(provider);
                }
            }
        }
        Ok(providers)
    }

    /// Persist the user-managed dynamic registry. Health/latency are runtime
    /// observations and are deliberately reset on restore; endpoint identity,
    /// role, credentials and activation settings survive gateway restarts.
    pub fn save_upstreams(&self, nodes: &[UpstreamNode]) -> Result<(), rusqlite::Error> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM upstreams", [])?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        for node in nodes {
            let stored = StoredUpstream::from(node);
            let json = serde_json::to_string(&stored)
                .map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))?;
            tx.execute(
                "INSERT INTO upstreams (id, role, config, updated_at) VALUES (?1, ?2, ?3, ?4)",
                params![node.id, node.role, json, now],
            )?;
        }
        tx.commit()
    }

    pub fn load_upstreams(
        &self,
    ) -> Result<Vec<(String, RegisterUpstreamRequest)>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT role, config FROM upstreams ORDER BY role, id")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut result = Vec::new();
        for row in rows {
            let (role, json) = row?;
            if let Ok(stored) = serde_json::from_str::<StoredUpstream>(&json) {
                result.push((role, stored.into_request()));
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredUpstream {
    id: String,
    endpoint_url: String,
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    bearer_token: Option<String>,
    weight: u32,
    #[serde(default)]
    max_context_length: Option<u64>,
    active: bool,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    cost_per_hour: Option<f64>,
    #[serde(default)]
    health_url: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

impl From<&UpstreamNode> for StoredUpstream {
    fn from(node: &UpstreamNode) -> Self {
        Self {
            id: node.id.clone(), endpoint_url: node.endpoint_url.clone(),
            engine: node.engine.clone(), bearer_token: node.bearer_token.clone(),
            weight: node.weight, max_context_length: node.max_context_length,
            active: node.active, label: node.label.clone(), provider: node.provider.clone(),
            cost_per_hour: node.cost_per_hour, health_url: node.health_url.clone(),
            model: node.model.clone(),
        }
    }
}

impl StoredUpstream {
    fn into_request(self) -> RegisterUpstreamRequest {
        RegisterUpstreamRequest {
            endpoint_url: Some(self.endpoint_url), engine: self.engine,
            bearer_token: self.bearer_token, weight: Some(self.weight),
            max_context_length: self.max_context_length, id: Some(self.id),
            label: self.label, provider: self.provider, cost_per_hour: self.cost_per_hour,
            health_url: self.health_url, enabled: Some(self.active), model: self.model,
        }
    }
}

#[cfg(test)]
mod upstream_tests {
    use super::*;
    use crate::registry::UpstreamRegistry;

    #[test]
    fn upstreams_round_trip_with_engine_credentials_and_disabled_state() {
        let db = Db::new_in_memory().unwrap();
        let mut registry = UpstreamRegistry::default();
        let node = registry.upsert("main", RegisterUpstreamRequest {
            endpoint_url: Some("http://host.docker.internal:8080/v1/chat/completions".into()),
            engine: Some("vmlx".into()),
            bearer_token: Some("secret-token".into()),
            id: Some("mac-main".into()),
            enabled: Some(false),
            weight: Some(3),
            ..Default::default()
        }).node;
        db.save_upstreams(&[node]).unwrap();
        let rows = db.load_upstreams().unwrap();
        assert_eq!(rows.len(), 1);
        let (role, request) = &rows[0];
        assert_eq!(role, "main");
        assert_eq!(request.engine.as_deref(), Some("vmlx"));
        assert_eq!(request.endpoint_url.as_deref(), Some("http://host.docker.internal:8080/v1/chat/completions"));
        assert_eq!(request.bearer_token.as_deref(), Some("secret-token"));
        assert_eq!(request.enabled, Some(false));
        assert_eq!(request.weight, Some(3));
    }
}
