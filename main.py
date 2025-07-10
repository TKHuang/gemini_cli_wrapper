import asyncio
import json
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Gemini CLI Wrapper",
    description="OpenAI-compatible API wrapper for Gemini CLI",
    version="1.0.0",
)


# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "gemini"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Helper functions
async def call_gemini_cli(model: str, prompt: str, **kwargs) -> str:
    """Call the Gemini CLI with the specified parameters"""
    cmd = ["gemini", "-m", model, "-p", prompt]

    # Add additional CLI options if needed
    if kwargs.get("debug"):
        cmd.append("-d")
    if kwargs.get("sandbox"):
        cmd.append("-s")
    if kwargs.get("yolo"):
        cmd.append("-y")

    try:
        # Run the Gemini CLI command
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout_bytes, stderr_bytes = await result.communicate()
        stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Gemini CLI error: {stderr}")

        # Clean MCP messages from the output
        cleaned_output = clean_mcp_messages(stdout)
        return cleaned_output

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Gemini CLI not found. Please ensure 'gemini' is installed and in PATH.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling Gemini CLI: {str(e)}"
        )


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages format to a single prompt"""
    prompt_parts = []

    for message in messages:
        role = message.role
        content = message.content

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    return "\n".join(prompt_parts)


def clean_mcp_messages(text: str) -> str:
    """Remove MCP STDERR messages from the output"""
    # Pattern to match MCP STDERR messages like "MCP STDERR (context7): Context7 Documentation MCP Server running on stdio"
    mcp_pattern = r"MCP STDERR \([^)]+\): [^\n]*\n?"
    cleaned_text = re.sub(mcp_pattern, "", text, flags=re.MULTILINE)

    # Remove any standalone "Loaded cached credentials." messages that might appear
    cleaned_text = re.sub(r"Loaded cached credentials\.\s*\n?", "", cleaned_text)

    # Remove any extra empty lines that might be left
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)

    return cleaned_text.strip()


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (1 token ≈ 4 characters)"""
    return len(text) // 4


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Gemini CLI Wrapper - OpenAI Compatible API"}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    # Default Gemini models based on the CLI help
    models = [
        ModelInfo(id="gemini-2.5-pro", created=int(time.time()), owned_by="gemini"),
        ModelInfo(id="gemini-2.5-flash", created=int(time.time()), owned_by="gemini"),
    ]

    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""

    # Convert messages to prompt format
    prompt = messages_to_prompt(request.messages)

    # Call Gemini CLI
    response_text = await call_gemini_cli(model=request.model, prompt=prompt, yolo=True)

    # Create response
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:
        # For streaming responses
        async def generate_stream():
            # Split response into chunks for streaming effect
            words = response_text.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming response
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_text)

        response = ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion"""

    # Call Gemini CLI
    response_text = await call_gemini_cli(model=request.model, prompt=request.prompt)

    # Create response
    completion_id = f"cmpl-{uuid.uuid4().hex}"

    if request.stream:
        # For streaming responses
        async def generate_stream():
            words = response_text.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": word + " " if i < len(words) - 1 else word,
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming response
        prompt_tokens = estimate_tokens(request.prompt)
        completion_tokens = estimate_tokens(response_text)

        response = CompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(index=0, text=response_text, finish_reason="stop")
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gemini-cli-wrapper"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
