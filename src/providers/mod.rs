use crate::types::{ProviderConfig, ProviderType};
use std::collections::HashMap;
use tracing::info;

#[allow(dead_code)]
pub struct ProviderRegistry {
    providers: HashMap<String, ProviderConfig>,
}

#[allow(dead_code)]
impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, config: ProviderConfig) {
        info!("Registered provider: {} ({:?})", config.name, config.provider_type);
        self.providers.insert(config.id.clone(), config);
    }

    pub fn unregister(&mut self, id: &str) -> Option<ProviderConfig> {
        self.providers.remove(id)
    }

    pub fn get(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers.get(id)
    }

    pub fn list(&self) -> Vec<&ProviderConfig> {
        self.providers.values().collect()
    }

    pub fn resolve_model(&self, model: &str) -> Option<&ProviderConfig> {
        self.providers
            .values()
            .filter(|p| p.enabled)
            .find(|p| p.models.iter().any(|m| m == model))
    }

    pub fn resolve_by_type(&self, provider_type: &ProviderType) -> Vec<&ProviderConfig> {
        self.providers
            .values()
            .filter(|p| p.enabled && p.provider_type == *provider_type)
            .collect()
    }

    pub fn all_models(&self) -> Vec<String> {
        let mut models: Vec<String> = self
            .providers
            .values()
            .filter(|p| p.enabled)
            .flat_map(|p| p.models.clone())
            .collect();
        models.sort();
        models.dedup();
        models
    }
}
