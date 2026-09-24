# Google AI Studio

First-party Kinetix integration for the Gemini API exposed through Google AI Studio.

- Base URL: `https://generativelanguage.googleapis.com/v1beta`
- Wire format: Kinetix core `gemini`
- Authentication: user-supplied API key via `x-goog-api-key`
- Discovery: authenticated `GET /models` (`models.list`)
- Custom provider adapter: none

## Model discovery

The plugin uses Google's live model catalog as the availability source. It preserves provider metadata, imports `inputTokenLimit`, `outputTokenLimit`, and explicit `thinking` support, and only returns models that advertise `generateContent`.

Fields the Google catalog does not explicitly expose stay unknown. Kinetix can then apply its normal lower-priority `models.dev` enrichment; this plugin does not ship a bundled Gemini catalog.

The API key remains owned by Kinetix. Account-aware discovery asks the host to sign the model-list request, so the guest never reads plaintext credential bytes.

## Build

```sh
rustup target add wasm32-unknown-unknown
bash scripts/build-plugin.sh plugins/ai-studio
```
