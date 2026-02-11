# API Provider Configuration

The SODA MMQC project now supports both OpenAI and Anthropic APIs with structured output capabilities.

## Environment Setup

### Using .env File (Recommended)

Create a `.env` file in the project root with your configuration:

```bash
# API Provider Configuration
API_PROVIDER=openai  # or 'anthropic'

# OpenAI Configuration (required if API_PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (required if API_PROVIDER=anthropic)  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Device configuration
DEVICE=cpu  # or 'cuda', 'mps' for GPU acceleration
```

### Using Environment Variables

Alternatively, export environment variables directly:

#### OpenAI (default)
```bash
export API_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key
```

#### Anthropic
```bash
export API_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Configuration Validation

The system automatically validates your configuration on startup:
- ✅ **Valid setup**: Provider and API key are properly configured
- ⚠️ **Missing API key**: Provider selected but API key not found
- ⚠️ **Invalid provider**: Falls back to OpenAI with warning

## Supported Models

### OpenAI Models
- `gpt-4o-2024-08-06` (default)
- `gpt-4o-mini` 
- `gpt-4` variants with structured output support

### Anthropic Models
- `claude-3-5-sonnet-20241022` (default)
- `claude-3-7-sonnet-20250219`
- `claude-4-sonnet` and `claude-4-opus` (when available)

## Implementation Details

### OpenAI Structured Output
Uses the native `response_format` parameter with JSON schema enforcement:
```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "structured_output",
        "schema": your_json_schema
    }
}
```

### Anthropic Structured Output
Uses tool calling to enforce structured output:
- Converts JSON schema to an Anthropic tool definition
- Forces the model to use the structured output tool
- Extracts the structured response from tool use

### Per-check model config (OpenAI)

Checks can include a `model_config.json` to enable tools (e.g. web search). The file is passed through to `client.responses.create()` as `tools`, `tool_choice`, etc.

**Valid tool types** (OpenAI Responses API): `web_search_preview`, `web_search_preview_2025_03_11`, `file_search`, `code_interpreter`, `function`, `mcp`, `image_generation`, `shell`, `computer_use_preview`, `apply_patch`, `custom`. Do **not** use `web_search` or `web_fetch` (unsupported; will return 400).

Example for a check that may use web search:
```json
{
  "tools": [{ "type": "web_search_preview" }],
  "tool_choice": "auto"
}
```

**More agentic / coding (same endpoint):** You can enable multi-step thinking and code execution without changing the API:

- **Reasoning** (gpt-5, o-series): Add `"reasoning": { "effort": "medium" }` (`low` / `medium` / `high`) so the model spends more tokens on chain-of-thought before answering. Optional: `"max_output_tokens": 16384` to cap total output (reasoning + reply).
- **Code interpreter:** Add a tool `{ "type": "code_interpreter", "container": { "type": "auto", "memory_limit": "4g" } }` so the model can write and run Python in a sandbox (e.g. for data or image analysis). Use with a reasoning model for best results.
- **Multi-turn:** The Responses API supports `conversation` and `previous_response_id` for multi-round chats; that would require run-loop changes (multiple API calls per example) and is not yet wired through `model_config`.

Example agentic + coding config (reasoning model + web search + code interpreter):
```json
{
  "tools": [
    { "type": "web_search_preview" },
    { "type": "code_interpreter", "container": { "type": "auto", "memory_limit": "4g" } }
  ],
  "tool_choice": "auto",
  "max_tool_calls": 20,
  "reasoning": { "effort": "medium" },
  "max_output_tokens": 16384
}
```
Use a reasoning-capable model (e.g. `gpt-5`, `gpt-5-mini`, or o-series) when setting `reasoning`.

## Usage

The API automatically routes to the appropriate provider based on the `API_PROVIDER` environment variable. Both providers return the same response format:

```python
response, metadata = generate_response(
    model_input=model_input,
    model="your-preferred-model",  # Optional
    metadata={"custom": "metadata"}  # Optional
)
```

## Benefits

1. **Multimodal Support**: Both providers support text and image inputs
2. **Structured Output**: JSON schema enforcement for consistent responses
3. **Provider Flexibility**: Easy switching between OpenAI and Anthropic
4. **Unified Interface**: Same API surface regardless of provider
5. **Error Handling**: Consistent retry logic and error handling