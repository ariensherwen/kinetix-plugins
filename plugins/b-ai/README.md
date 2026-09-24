# B.AI

First-party Kinetix integration for B.AI's OpenAI-compatible API.

- Base URL: `https://api.b.ai/v1`
- Wire format: Kinetix core `openai`
- Authentication: user-supplied B.AI API key via bearer auth
- Discovery: authenticated `GET /models`
- Custom provider adapter: none

## Model discovery and metadata

The live B.AI model list is authoritative for availability. `models.json` only enriches models that B.AI actually returns; it never creates availability by itself.

B.AI's `/models` response is intentionally sparse, so the plugin carries a small sourced catalog for provider-specific facts. It currently covers `DeepSeek-V4.1-Flash` and the documented `DeepSeek-V4-Flash` / `DeepSeek-V4-Flash-Vision-Exp` aliases. Unknown fields remain unknown.

The API key remains owned by Kinetix. Account-aware discovery uses host-side credential signing and does not read plaintext credential bytes.

## Build

```sh
rustup target add wasm32-unknown-unknown
bash scripts/build-plugin.sh plugins/b-ai
```
