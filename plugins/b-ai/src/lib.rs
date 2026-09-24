//! B.AI integration for Kinetix.
//!
//! Kinetix core owns OpenAI protocol translation and API-key storage. This
//! plugin owns B.AI provider defaults, authenticated `/models` discovery, and
//! the small amount of B.AI-specific model metadata not carried by that list.

use kinetix::plugin::types::*;
use kinetix_plugin_sdk::{
    export, exports, kinetix,
    model_capabilities::{
        ModelCapabilitiesV1, ReasoningCapability, ReasoningLevel, ReasoningMode,
        SupportCapability, TransportCapability, VisionCapability,
    },
};
use serde_json::Value;

use kinetix_plugin_sdk::model_source as model_world;

type ModelPluginError = model_world::kinetix::plugin::types::PluginError;
type ModelHttpRequest = model_world::kinetix::plugin::types::HttpRequest;
type ModelAccountRef = model_world::kinetix::plugin::types::AccountRef;
type ModelCredentialRef = model_world::kinetix::plugin::types::CredentialRef;
type ModelDiscoveredModel = model_world::kinetix::plugin::types::DiscoveredModel;

const DEFAULT_BASE_URL: &str = "https://api.b.ai/v1";
const DEFAULT_MODELS_PATH: &str = "/models";
const MODEL_CATALOG: &str = include_str!("../models.json");

struct Component;

fn unsupported() -> PluginError {
    kinetix_plugin_sdk::helpers::error("unknown", "capability not provided by this plugin")
}

fn model_error(code: &str, message: impl Into<String>, retryable: bool) -> ModelPluginError {
    ModelPluginError {
        code: code.into(),
        message: message.into(),
        retryable,
        retry_after: None,
        reset_at: None,
    }
}

fn join_models_url(base_url: &str, models_path: &str) -> String {
    let base = if base_url.trim().is_empty() {
        DEFAULT_BASE_URL
    } else {
        base_url.trim()
    };
    let path = if models_path.trim().is_empty() {
        DEFAULT_MODELS_PATH
    } else {
        models_path.trim()
    };
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn catalog_entry(id: &str) -> Option<Value> {
    let catalog: Value = serde_json::from_str(MODEL_CATALOG).ok()?;
    let entry = catalog.get(id)?.clone();
    let Some(canonical) = entry.get("canonical").and_then(Value::as_str) else {
        return Some(entry);
    };
    let mut canonical_entry = catalog.get(canonical)?.clone();
    if let Some(object) = canonical_entry.as_object_mut() {
        object.insert("provider_alias".into(), Value::String(id.to_string()));
        object.insert("canonical_id".into(), Value::String(canonical.to_string()));
    }
    Some(canonical_entry)
}

fn reasoning_from_catalog(entry: &Value) -> Option<ReasoningCapability> {
    let raw = entry.get("reasoning")?;
    let supported = raw.get("supported")?.as_bool()?;
    if !supported {
        return Some(ReasoningCapability::unsupported());
    }

    let mode = raw.get("mode").and_then(Value::as_str);
    let can_disable = raw
        .get("can_disable")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if mode != Some("level") {
        let mut reasoning = ReasoningCapability::supported_unknown();
        reasoning.can_disable = raw.get("can_disable").and_then(Value::as_bool);
        return Some(reasoning);
    }

    let levels: Vec<ReasoningLevel> = raw
        .get("levels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(ReasoningLevel::parse)
        .collect();
    if levels.is_empty() {
        return Some(ReasoningCapability::supported_unknown());
    }
    let default = raw
        .get("default")
        .and_then(Value::as_str)
        .and_then(ReasoningLevel::parse)
        .filter(|level| levels.contains(level));
    Some(ReasoningCapability {
        supported: true,
        mode: Some(ReasoningMode::Level),
        levels: Some(levels),
        default,
        can_disable: Some(can_disable),
    })
}

fn normalized_capabilities(entry: Option<&Value>) -> Result<String, ModelPluginError> {
    let mut capabilities = ModelCapabilitiesV1::default();
    capabilities.transport = Some(TransportCapability::new("openai"));
    if let Some(entry) = entry {
        capabilities.reasoning = reasoning_from_catalog(entry);
        capabilities.tools = entry
            .get("tools")
            .and_then(Value::as_bool)
            .map(SupportCapability::new);
        capabilities.vision = entry
            .get("vision")
            .and_then(Value::as_bool)
            .map(VisionCapability::new);
        capabilities.structured_output = entry
            .get("structured_output")
            .and_then(Value::as_bool)
            .map(SupportCapability::new);
    }
    capabilities.to_json().map_err(|error| {
        model_error(
            "plugin_internal",
            format!("invalid normalized B.AI capabilities: {error}"),
            false,
        )
    })
}

fn parse_model_list(value: &Value) -> Result<Vec<ModelDiscoveredModel>, ModelPluginError> {
    let Some(data) = value.get("data").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut models = Vec::new();
    for item in data {
        let Some(id) = item
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let catalog = catalog_entry(id);
        let display_name = item
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or_else(|| {
                catalog
                    .as_ref()
                    .and_then(|entry| entry.get("display_name"))
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .or_else(|| Some(id.to_string()));
        let context_window = item
            .get("context_length")
            .or_else(|| item.get("contextWindow"))
            .and_then(Value::as_u64)
            .or_else(|| {
                catalog
                    .as_ref()
                    .and_then(|entry| entry.get("context_window"))
                    .and_then(Value::as_u64)
            });
        let max_output_tokens = item
            .get("max_output_tokens")
            .or_else(|| item.get("maxOutputTokens"))
            .and_then(Value::as_u64)
            .or_else(|| {
                catalog
                    .as_ref()
                    .and_then(|entry| entry.get("max_output_tokens"))
                    .and_then(Value::as_u64)
            });

        models.push(ModelDiscoveredModel {
            id: id.to_string(),
            display_name,
            context_window,
            max_output_tokens,
            capabilities_json: Some(normalized_capabilities(catalog.as_ref())?),
            raw_metadata: serde_json::to_string(item).ok(),
        });
    }
    models.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(models)
}

impl model_world::exports::account_model_source::Guest for Component {
    fn discover(
        provider_id: String,
        account: ModelAccountRef,
        base_url: String,
        models_path: String,
    ) -> Result<Vec<ModelDiscoveredModel>, ModelPluginError> {
        if account.provider_id != provider_id {
            return Err(model_error(
                "invalid_configuration",
                "model discovery account does not belong to the requested provider",
                false,
            ));
        }

        let req = ModelHttpRequest {
            method: "GET".into(),
            url: join_models_url(&base_url, &models_path),
            headers: vec![("accept".into(), "application/json".into())],
            body: Vec::new(),
            credential: None,
        };
        let credential = ModelCredentialRef::Account(account);
        let req = model_world::kinetix::plugin::host_credential::sign(&req, &credential)
            .map_err(|error| model_error(&error.code, error.message, error.retryable))?;
        let resp = model_world::kinetix::plugin::host_http::send(&req)
            .map_err(|error| model_error(&error.code, error.message, error.retryable))?;

        if resp.body_truncated {
            return Err(model_error(
                "upstream_unavailable",
                "B.AI model catalog response was truncated",
                true,
            ));
        }
        let body = String::from_utf8(resp.body)
            .map_err(|_| model_error("protocol_error", "B.AI model catalog is not UTF-8", false))?;
        if resp.status != 200 {
            return Err(model_error(
                if resp.status == 429 {
                    "rate_limited"
                } else if resp.status == 401 || resp.status == 403 {
                    "credential_expired"
                } else {
                    "upstream_unavailable"
                },
                format!("B.AI model catalog returned HTTP {}", resp.status),
                resp.status == 429 || resp.status >= 500,
            ));
        }
        let value: Value = serde_json::from_str(&body).map_err(|error| {
            model_error(
                "protocol_error",
                format!("invalid B.AI model catalog JSON: {error}"),
                false,
            )
        })?;
        parse_model_list(&value)
    }
}

model_world::export!(Component with_types_in kinetix_plugin_sdk::model_source);

impl exports::credential_strategy::Guest for Component {
    fn resolve(
        _provider_id: String,
        _account_id: String,
        _account_label: String,
    ) -> Result<CredentialLease, PluginError> {
        Err(unsupported())
    }
    fn health(_provider_id: String, _account_id: String) -> Result<String, PluginError> {
        Err(unsupported())
    }
    fn rotate(_provider_id: String, _account_id: String) -> Result<(), PluginError> {
        Err(unsupported())
    }
}
impl exports::model_source::Guest for Component {
    fn discover(
        _provider_id: String,
        _base_url: String,
        _models_path: String,
    ) -> Result<Vec<DiscoveredModel>, PluginError> {
        Err(unsupported())
    }
}
impl exports::health_probe::Guest for Component {
    fn probe(_provider_id: String, _account_id: String) -> Result<HealthObservation, PluginError> {
        Err(unsupported())
    }
}
impl exports::routing_facts::Guest for Component {
    fn facts(_request_json: String) -> Result<Vec<RoutingFact>, PluginError> {
        Ok(Vec::new())
    }
}
impl exports::hooks::Guest for Component {
    fn on_request_normalized(_request_json: String) -> Result<(), PluginError> {
        Ok(())
    }
    fn on_target_candidate(_target_json: String) -> Result<(), PluginError> {
        Ok(())
    }
    fn on_usage_finalized(_usage_json: String) -> Result<(), PluginError> {
        Ok(())
    }
}

export!(Component with_types_in kinetix_plugin_sdk);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enriches_deepseek_v41_flash_from_explicit_provider_catalog() {
        let value = serde_json::json!({
            "object": "list",
            "data": [{"id": "DeepSeek-V4.1-Flash", "object": "model", "created": 1790000000}]
        });
        let models = parse_model_list(&value).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].context_window, Some(1_000_000));
        assert_eq!(models[0].max_output_tokens, Some(384_000));
        let caps = ModelCapabilitiesV1::from_json(models[0].capabilities_json.as_deref().unwrap())
            .unwrap();
        assert_eq!(caps.vision.as_ref().map(|value| value.input), Some(true));
        assert_eq!(caps.tools.as_ref().map(|value| value.supported), Some(true));
        assert_eq!(
            caps.structured_output.as_ref().map(|value| value.supported),
            Some(true)
        );
        assert_eq!(
            caps.reasoning.as_ref().and_then(|value| value.default),
            Some(ReasoningLevel::High)
        );
    }

    #[test]
    fn explicit_bai_aliases_share_metadata_without_fuzzy_matching() {
        let alias = catalog_entry("DeepSeek-V4-Flash").unwrap();
        assert_eq!(
            alias.get("canonical_id").and_then(Value::as_str),
            Some("DeepSeek-V4.1-Flash")
        );
        assert!(catalog_entry("deepseek-v4.1-flash").is_none());
    }

    #[test]
    fn unknown_bai_models_keep_unknown_capabilities() {
        let value = serde_json::json!({"data": [{"id": "future-model"}]});
        let models = parse_model_list(&value).unwrap();
        assert_eq!(models[0].context_window, None);
        assert_eq!(models[0].max_output_tokens, None);
        let caps = ModelCapabilitiesV1::from_json(models[0].capabilities_json.as_deref().unwrap())
            .unwrap();
        assert!(caps.reasoning.is_none());
        assert!(caps.tools.is_none());
        assert!(caps.vision.is_none());
        assert!(caps.structured_output.is_none());
    }

    #[test]
    fn manifest_reuses_core_openai_adapter() {
        let manifest = include_str!("../plugin.toml");
        assert!(manifest.contains("wire_format = \"openai\""));
        assert!(manifest.contains("account_model_sources = [\"b-ai-models\"]"));
        assert!(!manifest.contains("provider_adapters"));
    }
}
