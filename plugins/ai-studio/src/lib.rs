//! Google AI Studio integration for Kinetix.
//!
//! Kinetix core owns the Gemini wire adapter and API-key storage. This plugin
//! only provides provider defaults and authenticated model discovery through
//! Google's native `models.list` endpoint.

use kinetix::plugin::types::*;
use kinetix_plugin_sdk::{
    export, exports, kinetix,
    model_capabilities::{ModelCapabilitiesV1, ReasoningCapability, TransportCapability},
};
use serde_json::Value;

use kinetix_plugin_sdk::model_source as model_world;

type ModelPluginError = model_world::kinetix::plugin::types::PluginError;
type ModelHttpRequest = model_world::kinetix::plugin::types::HttpRequest;
type ModelAccountRef = model_world::kinetix::plugin::types::AccountRef;
type ModelCredentialRef = model_world::kinetix::plugin::types::CredentialRef;
type ModelDiscoveredModel = model_world::kinetix::plugin::types::DiscoveredModel;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODELS_PATH: &str = "/models";
const MAX_PAGES: usize = 4;

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

fn join_models_url(base_url: &str, models_path: &str, page_token: Option<&str>) -> String {
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
    let mut url = format!(
        "{}/{}?pageSize=1000",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    );
    if let Some(token) = page_token.filter(|value| !value.is_empty()) {
        url.push_str("&pageToken=");
        url.push_str(&urlencode(token));
    }
    url
}

fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn supports_generate_content(item: &Value) -> bool {
    item.get("supportedGenerationMethods")
        .and_then(Value::as_array)
        .is_some_and(|methods| {
            methods
                .iter()
                .filter_map(Value::as_str)
                .any(|method| method == "generateContent")
        })
}

fn model_id(item: &Value) -> Option<String> {
    item.get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.strip_prefix("models/").unwrap_or(value).to_string())
        .or_else(|| {
            item.get("baseModelId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        })
}

fn normalized_capabilities(item: &Value) -> Result<String, ModelPluginError> {
    let mut capabilities = ModelCapabilitiesV1::default();
    capabilities.transport = Some(TransportCapability::new("gemini"));
    capabilities.reasoning = item.get("thinking").and_then(Value::as_bool).map(|supported| {
        if supported {
            ReasoningCapability::supported_unknown()
        } else {
            ReasoningCapability::unsupported()
        }
    });
    capabilities.to_json().map_err(|error| {
        model_error(
            "plugin_internal",
            format!("invalid normalized AI Studio capabilities: {error}"),
            false,
        )
    })
}

fn parse_model_page(value: &Value) -> Result<Vec<ModelDiscoveredModel>, ModelPluginError> {
    let Some(models) = value.get("models").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut discovered = Vec::new();
    for item in models {
        if !supports_generate_content(item) {
            continue;
        }
        let Some(id) = model_id(item) else {
            continue;
        };
        let display_name = item
            .get("displayName")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or_else(|| Some(id.clone()));

        discovered.push(ModelDiscoveredModel {
            id,
            display_name,
            context_window: item.get("inputTokenLimit").and_then(Value::as_u64),
            max_output_tokens: item.get("outputTokenLimit").and_then(Value::as_u64),
            capabilities_json: Some(normalized_capabilities(item)?),
            raw_metadata: serde_json::to_string(item).ok(),
        });
    }
    Ok(discovered)
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

        let credential = ModelCredentialRef::Account(account);
        let mut page_token: Option<String> = None;
        let mut discovered = Vec::new();

        for page in 0..MAX_PAGES {
            let req = ModelHttpRequest {
                method: "GET".into(),
                url: join_models_url(&base_url, &models_path, page_token.as_deref()),
                headers: vec![("accept".into(), "application/json".into())],
                body: Vec::new(),
                credential: None,
            };
            let req = model_world::kinetix::plugin::host_credential::sign(&req, &credential)
                .map_err(|error| model_error(&error.code, error.message, error.retryable))?;
            let resp = model_world::kinetix::plugin::host_http::send(&req)
                .map_err(|error| model_error(&error.code, error.message, error.retryable))?;

            if resp.body_truncated {
                return Err(model_error(
                    "upstream_unavailable",
                    "Google model catalog response was truncated",
                    true,
                ));
            }
            let body = String::from_utf8(resp.body).map_err(|_| {
                model_error("protocol_error", "Google model catalog is not UTF-8", false)
            })?;
            if resp.status != 200 {
                return Err(model_error(
                    if resp.status == 429 {
                        "rate_limited"
                    } else if resp.status == 401 || resp.status == 403 {
                        "credential_expired"
                    } else {
                        "upstream_unavailable"
                    },
                    format!("Google model catalog returned HTTP {}", resp.status),
                    resp.status == 429 || resp.status >= 500,
                ));
            }

            let value: Value = serde_json::from_str(&body).map_err(|error| {
                model_error(
                    "protocol_error",
                    format!("invalid Google model catalog JSON: {error}"),
                    false,
                )
            })?;
            discovered.extend(parse_model_page(&value)?);
            page_token = value
                .get("nextPageToken")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string);

            if page_token.is_none() {
                discovered.sort_by(|left, right| left.id.cmp(&right.id));
                discovered.dedup_by(|left, right| left.id == right.id);
                return Ok(discovered);
            }
            if page + 1 == MAX_PAGES {
                return Err(model_error(
                    "upstream_unavailable",
                    "Google model catalog exceeded the plugin pagination bound",
                    true,
                ));
            }
        }

        Ok(discovered)
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
    fn parses_gemini_model_metadata_without_inventing_unknown_capabilities() {
        let value = serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-3.8-flash",
                    "baseModelId": "gemini-3.8-flash",
                    "displayName": "Gemini 3.8 Flash",
                    "inputTokenLimit": 1000000,
                    "outputTokenLimit": 65536,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                    "thinking": true
                },
                {
                    "name": "models/text-embedding-004",
                    "supportedGenerationMethods": ["embedContent"]
                }
            ]
        });

        let models = parse_model_page(&value).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-3.8-flash");
        assert_eq!(models[0].context_window, Some(1_000_000));
        assert_eq!(models[0].max_output_tokens, Some(65_536));

        let caps = ModelCapabilitiesV1::from_json(models[0].capabilities_json.as_deref().unwrap())
            .unwrap();
        assert_eq!(caps.transport.as_ref().map(|value| value.format.as_str()), Some("gemini"));
        assert_eq!(caps.reasoning.as_ref().map(|value| value.supported), Some(true));
        assert!(caps.tools.is_none());
        assert!(caps.vision.is_none());
        assert!(caps.structured_output.is_none());
    }

    #[test]
    fn manifest_uses_native_gemini_and_account_discovery_only() {
        let manifest = include_str!("../plugin.toml");
        assert!(manifest.contains("wire_format = \"gemini\""));
        assert!(manifest.contains("custom_header_name = \"x-goog-api-key\""));
        assert!(manifest.contains("account_model_sources = [\"ai-studio-models\"]"));
        assert!(!manifest.contains("provider_adapters"));
    }
}
