use rusqlite::{params, Connection};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use crate::types::ProviderConfig;

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
}
