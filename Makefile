.PHONY: help dev run install clean test lint format check health

# Default target
help:
	@echo "Gemini CLI Wrapper - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make dev       - Run development server with hot reload (INFO level)"
	@echo "  make dev-debug - Run development server with debug logging (DEBUG level)"
	@echo "  make dev-custom HOST=127.0.0.1 PORT=8080 LOG_LEVEL=DEBUG - Custom development server"
	@echo ""
	@echo "Production:"
	@echo "  make run       - Run production server"
	@echo "  make run-prod  - Run production server with optimized settings"
	@echo ""
	@echo "Dependencies:"
	@echo "  make install   - Install dependencies with uv"
	@echo "  make clean     - Clean cache and temporary files"
	@echo ""
	@echo "Quality:"
	@echo "  make lint      - Run linting checks"
	@echo "  make format    - Format code with black"
	@echo "  make check     - Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test      - Run tests"
	@echo "  make health    - Check API health"

# Development commands
dev:
	@echo "Starting development server with hot reload..."
	LOG_LEVEL=INFO uvicorn gemini_cli_wrapper.main:app --reload --host 0.0.0.0 --port 8000

dev-debug:
	@echo "Starting development server with debug logging..."
	LOG_LEVEL=DEBUG uvicorn gemini_cli_wrapper.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# Production commands  
run:
	@echo "Starting production server..."
	python -m gemini_cli_wrapper

run-prod:
	@echo "Starting production server with optimized settings..."
	python -m gemini_cli_wrapper --host 0.0.0.0 --port 8000 --log-level INFO

# Custom host/port
run-custom:
	@echo "Starting server on custom host/port..."
	@echo "Usage: make run-custom HOST=127.0.0.1 PORT=8080"
	python -m gemini_cli_wrapper --host $(or $(HOST),0.0.0.0) --port $(or $(PORT),8000)

dev-custom:
	@echo "Starting development server on custom host/port..."
	@echo "Usage: make dev-custom HOST=127.0.0.1 PORT=8080 LOG_LEVEL=DEBUG"
	LOG_LEVEL=$(or $(LOG_LEVEL),INFO) uvicorn gemini_cli_wrapper.main:app --reload --host $(or $(HOST),0.0.0.0) --port $(or $(PORT),8000)

# Dependencies
install:
	@echo "Installing dependencies with uv..."
	uv sync

install-dev:
	@echo "Installing development dependencies..."
	uv add --dev black ruff pytest httpx

# Quality assurance
lint:
	@echo "Running linting checks..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check gemini_cli_wrapper/; \
	else \
		echo "ruff not installed. Run 'make install-dev' first."; \
	fi

format:
	@echo "Formatting code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black gemini_cli_wrapper/; \
	else \
		echo "black not installed. Run 'make install-dev' first."; \
	fi

check: lint
	@echo "Running all quality checks..."
	@echo "✓ Linting completed"

# Testing
test:
	@echo "Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest; \
	else \
		echo "pytest not installed. Run 'make install-dev' first."; \
	fi

health:
	@echo "Checking API health..."
	@echo "Make sure the server is running first with 'make dev' or 'make run'"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Server not responding"

# Utility commands
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

logs:
	@echo "Showing recent logs (if using systemd or similar)..."
	@echo "For development, logs appear in terminal"

# Docker commands (optional)
docker-build:
	@echo "Building Docker image..."
	docker build -t gemini-cli-wrapper .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 gemini-cli-wrapper

# Show version
version:
	@python -c "from gemini_cli_wrapper.__about__ import __version__; print(f'Gemini CLI Wrapper v{__version__}')" 