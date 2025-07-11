import asyncio
import json
import logging
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .__about__ import __version__

# Configure logging
logger = logging.getLogger(__name__)

def configure_logging(level: str = "INFO"):
    """Configure logging for the application"""
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )

    # Configure uvicorn access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.setLevel(numeric_level)

    logger.info(f"Logging configured at {level.upper()} level")

# Configure logging at module level for development
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
configure_logging(LOG_LEVEL)

app = FastAPI(
    title="Gemini CLI Wrapper",
    description="OpenAI-compatible API wrapper for Gemini CLI",
    version=__version__,
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
    logger.debug(f"Calling Gemini CLI with model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Additional kwargs: {kwargs}")

    cmd = ["gemini", "-m", model, "-p", prompt]

    # Add additional CLI options if needed
    if kwargs.get("debug"):
        cmd.append("-d")
        logger.debug("Added debug flag to Gemini CLI")
    if kwargs.get("sandbox"):
        cmd.append("-s")
        logger.debug("Added sandbox flag to Gemini CLI")
    if kwargs.get("yolo"):
        cmd.append("-y")
        logger.debug("Added yolo flag to Gemini CLI")

    logger.debug(f"Full Gemini CLI command: {' '.join(cmd[:4])} [prompt truncated]")

    try:
        # Run the Gemini CLI command
        logger.debug("Starting Gemini CLI subprocess")
        result = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout_bytes, stderr_bytes = await result.communicate()
        stdout = stdout_bytes.decode("utf-8") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8") if stderr_bytes else ""

        logger.debug(f"Gemini CLI return code: {result.returncode}")
        logger.debug(f"Gemini CLI stdout length: {len(stdout)} characters")
        if stderr:
            logger.debug(f"Gemini CLI stderr: {stderr}")

        if result.returncode != 0:
            logger.error(
                f"Gemini CLI failed with return code {result.returncode}: {stderr}"
            )
            raise HTTPException(status_code=500, detail=f"Gemini CLI error: {stderr}")

        # Clean MCP messages from the output
        logger.debug(f"Cleaning MCP messages from output: \n{stdout}")
        cleaned_output = clean_mcp_messages(stdout)
        logger.debug(
            f"Cleaned output length: {len(cleaned_output)} characters: \n{cleaned_output}"
        )
        
        return cleaned_output

    except FileNotFoundError as e:
        logger.error("Gemini CLI not found in PATH")
        raise HTTPException(
            status_code=500,
            detail="Gemini CLI not found. Please ensure 'gemini' is installed and in PATH.",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error calling Gemini CLI: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error calling Gemini CLI: {str(e)}"
        ) from e


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages format to a single prompt"""
    logger.debug(f"Converting {len(messages)} messages to prompt format")
    prompt_parts = []

    for i, message in enumerate(messages):
        role = message.role
        content = message.content
        logger.debug(f"Message {i}: role={role}, content_length={len(content)}")

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt = "\n".join(prompt_parts)
    logger.debug(f"Generated prompt length: {len(prompt)} characters")
    return prompt


def clean_mcp_messages(text: str) -> str:
    """Remove MCP STDERR messages from the output"""
    logger.debug(f"Cleaning MCP messages from text of length: {len(text)}")

    # Pattern to match MCP STDERR messages like "MCP STDERR (context7): Context7 Documentation MCP Server running on stdio"
    mcp_pattern = r"MCP STDERR \([^)]+\): [^\n]*\n?"
    cleaned_text = re.sub(mcp_pattern, "", text, flags=re.MULTILINE)

    # Remove any standalone "Loaded cached credentials." messages that might appear
    cleaned_text = re.sub(r"Loaded cached credentials\.\s*\n?", "", cleaned_text)

    # Remove any extra empty lines that might be left
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)

    result = cleaned_text.strip()
    logger.debug(f"Cleaned text length: {len(result)} characters")
    return result


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (1 token ≈ 4 characters)"""
    tokens = len(text) // 4
    logger.debug(f"Estimated {tokens} tokens for text of length {len(text)}")
    return tokens


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Gemini CLI Wrapper - OpenAI Compatible API"}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    logger.debug("Listing available models")
    # Default Gemini models based on the CLI help
    models = [
        ModelInfo(id="gemini-2.5-pro", created=int(time.time()), owned_by="gemini"),
        ModelInfo(id="gemini-2.5-flash", created=int(time.time()), owned_by="gemini"),
    ]

    logger.debug(f"Returning {len(models)} models")
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    logger.info(
        f"Chat completion request: model={request.model}, messages={len(request.messages)}, stream={request.stream}"
    )
    logger.debug(
        f"Chat completion request details: temperature={request.temperature}, max_tokens={request.max_tokens}"
    )

    # Convert messages to prompt format
    prompt = messages_to_prompt(request.messages)

    # Call Gemini CLI
    logger.debug("Calling Gemini CLI for chat completion")
    response_text = await call_gemini_cli(model=request.model, prompt=prompt, yolo=True)

    # Create response
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.debug(f"Generated completion ID: {completion_id}")

    if request.stream:
        logger.debug("Generating streaming response")

        # For streaming responses
        async def generate_stream():
            # Stream by characters to preserve formatting
            chunk_size = 10  # Stream characters in small chunks
            logger.debug(f"Streaming {len(response_text)} characters in chunks of {chunk_size}")
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                is_last = i + chunk_size >= len(response_text)
                
                chunk_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.02)  # Small delay for smooth streaming

            logger.debug("Finished streaming response")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming response
        logger.debug("Generating non-streaming response")
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

        logger.info(
            f"Chat completion completed: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens"
        )
        return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion"""
    logger.info(
        f"Text completion request: model={request.model}, prompt_length={len(request.prompt)}, stream={request.stream}"
    )
    logger.debug(
        f"Text completion request details: temperature={request.temperature}, max_tokens={request.max_tokens}"
    )

    # Call Gemini CLI
    logger.debug("Calling Gemini CLI for text completion")
    response_text = await call_gemini_cli(model=request.model, prompt=request.prompt)

    # Create response
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    logger.debug(f"Generated completion ID: {completion_id}")

    if request.stream:
        logger.debug("Generating streaming response")

        # For streaming responses
        async def generate_stream():
            # Stream by characters to preserve formatting
            chunk_size = 10  # Stream characters in small chunks
            logger.debug(f"Streaming {len(response_text)} characters in chunks of {chunk_size}")
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                is_last = i + chunk_size >= len(response_text)
                
                chunk_data = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk,
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.02)

            logger.debug("Finished streaming response")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming response
        logger.debug("Generating non-streaming response")
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

        logger.info(
            f"Text completion completed: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens"
        )
        return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {"status": "healthy", "service": "gemini-cli-wrapper"}


def main():
    """Main entry point for the application."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Gemini CLI Wrapper - OpenAI-compatible API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--version", action="version", version=f"gemini-cli-wrapper {__version__}"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level)

    logger.info(f"Starting Gemini CLI Wrapper v{__version__}")
    logger.info(f"Log level set to: {args.log_level}")

    print(f"Starting Gemini CLI Wrapper v{__version__}")
    print(f"Server will be available at http://{args.host}:{args.port}")
    print("OpenAI-compatible endpoints:")
    print(f"  - Chat completions: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - Text completions: http://{args.host}:{args.port}/v1/completions")
    print(f"  - Models: http://{args.host}:{args.port}/v1/models")
    print(f"  - Health check: http://{args.host}:{args.port}/health")
    print(f"Logging level: {args.log_level}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
