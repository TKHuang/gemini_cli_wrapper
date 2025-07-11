# Gemini CLI Wrapper

An OpenAI-compatible API wrapper for the Gemini CLI that provides REST endpoints following the OpenAI API specification.

## Features

- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- 🚀 **Fast**: Built with FastAPI for high performance
- 🛠️ **Flexible**: Supports both chat completions and text completions
- 📦 **Easy Installation**: Install as a command-line tool via pip

## Prerequisites

- Python 3.13+
- Gemini CLI installed and available in PATH

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/tkhuang/gemini-cli-wrapper.git
cd gemini-cli-wrapper

# Install dependencies with uv
make install

# Or manually with uv
uv sync
```

### From PyPI

```bash
# Install the package
pip install gemini-cli-wrapper

# Or install from source
pip install git+https://github.com/tkhuang/gemini-cli-wrapper.git
```

## Usage

### Running the Server

**Start the server:**

```bash
gemini-cli-wrapper
```

**With custom host and port:**

```bash
gemini-cli-wrapper --host 127.0.0.1 --port 9000
```

**View all options:**

```bash
gemini-cli-wrapper --help
```

The server will be available at `http://localhost:8000` by default.

### API Endpoints

- `POST /v1/chat/completions` - OpenAI Chat Completions API
- `POST /v1/completions` - OpenAI Completions API
- `GET /v1/models` - List available models
- `GET /health` - Health check endpoint

### Examples

**Chat Completion:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

**Text Completion:**

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "prompt": "The weather today is"
  }'
```

**Using with OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not used, but required
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Configuration

### Available Models

- `gemini-2.5-pro`
- `gemini-2.5-flash` (default)

The wrapper automatically:

- Filters out MCP STDERR messages from Gemini CLI output
- Uses the `-m` flag to specify models to Gemini CLI
- Uses the `-p` flag to pass prompts to Gemini CLI
- Adds `--yolo` flag for chat completions (auto-accept actions)

## License

MIT License
