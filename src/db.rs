use rusqlite::{params, Connection};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::types::ProviderConfig;
use crate::registry::{RegisterUpstreamRequest, UpstreamNode};
use crate::usage::UsageEvent;
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
            CREATE TABLE IF NOT EXISTS usage_events (
                id TEXT PRIMARY KEY,
                ts_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                route TEXT NOT NULL,
                model TEXT NOT NULL,
                tenant_id TEXT,
                user_id TEXT,
                source TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                audio_seconds REAL NOT NULL DEFAULT 0,
                characters INTEGER NOT NULL DEFAULT 0,
                images INTEGER NOT NULL DEFAULT 0,
                estimated INTEGER NOT NULL DEFAULT 0,
                metered_upstream INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                delivered INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts_ms);
            CREATE INDEX IF NOT EXISTS idx_usage_undelivered ON usage_events(delivered, ts_ms);
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

/// Usage metering persistence (see `usage.rs`).
#[allow(dead_code)]
impl Db {
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Store one usage event as undelivered. Duplicate ids are ignored.
    pub fn insert_usage(&self, ev: &UsageEvent) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO usage_events (
                id, ts_ms, kind, route, model, tenant_id, user_id, source,
                prompt_tokens, completion_tokens, audio_seconds, characters, images,
                estimated, metered_upstream, status, latency_ms, delivered
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, 0)",
            params![
                ev.id,
                ev.ts_ms as i64,
                ev.kind,
                ev.route,
                ev.model,
                ev.tenant_id,
                ev.user_id,
                ev.source,
                ev.prompt_tokens as i64,
                ev.completion_tokens as i64,
                ev.audio_seconds,
                ev.characters as i64,
                ev.images as i64,
                ev.estimated as i64,
                ev.metered_upstream as i64,
                ev.status as i64,
                ev.latency_ms as i64,
            ],
        )?;
        Ok(())
    }

    /// Oldest undelivered events first, at most `limit`.
    pub fn undelivered_usage(&self, limit: usize) -> Result<Vec<UsageEvent>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, ts_ms, kind, route, model, tenant_id, user_id, source,
                    prompt_tokens, completion_tokens, audio_seconds, characters, images,
                    estimated, metered_upstream, status, latency_ms
             FROM usage_events WHERE delivered = 0
             ORDER BY ts_ms, id LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(UsageEvent {
                id: row.get(0)?,
                ts_ms: row.get::<_, i64>(1)? as u64,
                kind: row.get(2)?,
                route: row.get(3)?,
                model: row.get(4)?,
                tenant_id: row.get(5)?,
                user_id: row.get(6)?,
                source: row.get(7)?,
                prompt_tokens: row.get::<_, i64>(8)? as u64,
                completion_tokens: row.get::<_, i64>(9)? as u64,
                audio_seconds: row.get(10)?,
                characters: row.get::<_, i64>(11)? as u64,
                images: row.get::<_, i64>(12)? as u64,
                estimated: row.get::<_, i64>(13)? != 0,
                metered_upstream: row.get::<_, i64>(14)? != 0,
                status: row.get::<_, i64>(15)? as u16,
                latency_ms: row.get::<_, i64>(16)? as u64,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
    }

    /// Number of events not yet acknowledged by the billing webhook.
    pub fn pending_usage_count(&self) -> Result<i64, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM usage_events WHERE delivered = 0",
            [],
            |row| row.get(0),
        )
    }

    /// Flag events as delivered in one transaction. Returns rows updated.
    pub fn mark_usage_delivered(&self, ids: &[String]) -> Result<usize, rusqlite::Error> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        let mut updated = 0;
        {
            let mut stmt = tx.prepare("UPDATE usage_events SET delivered = 1 WHERE id = ?1")?;
            for id in ids {
                updated += stmt.execute(params![id])?;
            }
        }
        tx.commit()?;
        Ok(updated)
    }

    /// Aggregate usage since `since_ms` (epoch ms). `group_by` is one of
    /// `tenant | model | kind | source | day`. Each row carries `key` plus
    /// `count`, `prompt_tokens`, `completion_tokens`, `audio_seconds`,
    /// `characters`, `images`.
    pub fn usage_summary(
        &self,
        since_ms: u64,
        group_by: &str,
    ) -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let key_expr = match group_by {
            "tenant" => "COALESCE(tenant_id, '(none)')",
            "model" => "model",
            "kind" => "kind",
            "source" => "COALESCE(source, '(none)')",
            "day" => "strftime('%Y-%m-%d', ts_ms / 1000, 'unixepoch')",
            other => {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "unknown group_by '{other}'"
                )))
            }
        };
        let sql = format!(
            "SELECT {key_expr} AS k,
                    COUNT(*),
                    COALESCE(SUM(prompt_tokens), 0),
                    COALESCE(SUM(completion_tokens), 0),
                    COALESCE(SUM(audio_seconds), 0.0),
                    COALESCE(SUM(characters), 0),
                    COALESCE(SUM(images), 0)
             FROM usage_events WHERE ts_ms >= ?1
             GROUP BY k ORDER BY k"
        );
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![since_ms as i64], |row| {
            Ok(serde_json::json!({
                "key": row.get::<_, String>(0)?,
                "count": row.get::<_, i64>(1)?,
                "prompt_tokens": row.get::<_, i64>(2)?,
                "completion_tokens": row.get::<_, i64>(3)?,
                "audio_seconds": row.get::<_, f64>(4)?,
                "characters": row.get::<_, i64>(5)?,
                "images": row.get::<_, i64>(6)?,
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
    }

    /// Delete DELIVERED events older than `retention_ms`. Undelivered rows are
    /// never removed, so nothing unbilled is lost.
    /// `include_undelivered`: also drop rows that were never delivered (used
    /// when no webhook is configured, so nothing will ever deliver them).
    pub fn cleanup_usage(&self, retention_ms: u64, include_undelivered: bool) -> Result<usize, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let cutoff = Self::now_ms().saturating_sub(retention_ms) as i64;
        let sql = if include_undelivered {
            "DELETE FROM usage_events WHERE ts_ms < ?1"
        } else {
            "DELETE FROM usage_events WHERE delivered = 1 AND ts_ms < ?1"
        };
        conn.execute(sql, params![cutoff])
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;
    use crate::usage::UsageEvent;

    fn event(id: &str, ts_ms: u64, tenant: Option<&str>, model: &str, kind: &str) -> UsageEvent {
        UsageEvent {
            id: id.into(),
            ts_ms,
            kind: kind.into(),
            route: "inference".into(),
            model: model.into(),
            tenant_id: tenant.map(String::from),
            prompt_tokens: 10,
            completion_tokens: 5,
            audio_seconds: 1.5,
            characters: 20,
            images: 1,
            status: 200,
            ..Default::default()
        }
    }

    #[test]
    fn insert_and_summary_group_by_tenant_and_model() {
        let db = Db::new_in_memory().unwrap();
        db.insert_usage(&event("a", 1_000, Some("acme"), "m1", "chat")).unwrap();
        db.insert_usage(&event("b", 2_000, Some("acme"), "m2", "chat")).unwrap();
        db.insert_usage(&event("c", 3_000, None, "m1", "embeddings")).unwrap();
        // Duplicate id is ignored.
        db.insert_usage(&event("a", 1_000, Some("acme"), "m1", "chat")).unwrap();

        let rows = db.usage_summary(0, "tenant").unwrap();
        assert_eq!(rows.len(), 2);
        let acme = rows.iter().find(|r| r["key"] == "acme").unwrap();
        assert_eq!(acme["count"], 2);
        assert_eq!(acme["prompt_tokens"], 20);
        assert_eq!(acme["completion_tokens"], 10);
        assert_eq!(acme["audio_seconds"], 3.0);
        assert_eq!(acme["characters"], 40);
        assert_eq!(acme["images"], 2);
        let none = rows.iter().find(|r| r["key"] == "(none)").unwrap();
        assert_eq!(none["count"], 1);

        let by_model = db.usage_summary(0, "model").unwrap();
        assert_eq!(by_model.iter().find(|r| r["key"] == "m1").unwrap()["count"], 2);

        let by_kind = db.usage_summary(0, "kind").unwrap();
        assert_eq!(by_kind.iter().find(|r| r["key"] == "chat").unwrap()["count"], 2);

        // since filter excludes older rows.
        let recent = db.usage_summary(2_500, "tenant").unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0]["count"], 1);

        assert!(db.usage_summary(0, "bogus").is_err());
    }

    #[test]
    fn undelivered_then_mark_delivered_then_cleanup() {
        let db = Db::new_in_memory().unwrap();
        db.insert_usage(&event("a", 1_000, Some("t"), "m", "chat")).unwrap();
        db.insert_usage(&event("b", 2_000, Some("t"), "m", "chat")).unwrap();
        db.insert_usage(&event("c", 3_000, Some("t"), "m", "chat")).unwrap();
        assert_eq!(db.pending_usage_count().unwrap(), 3);

        let batch = db.undelivered_usage(2).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, "a");
        assert_eq!(batch[0], event("a", 1_000, Some("t"), "m", "chat"));

        let updated = db
            .mark_usage_delivered(&["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(updated, 2);
        assert_eq!(db.pending_usage_count().unwrap(), 1);

        let rest = db.undelivered_usage(10).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].id, "c");

        // Cleanup only removes delivered rows; retention of 0 ages out "a" and "b".
        let removed = db.cleanup_usage(0, false).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(db.pending_usage_count().unwrap(), 1);
        assert_eq!(db.usage_summary(0, "tenant").unwrap()[0]["count"], 1);
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
