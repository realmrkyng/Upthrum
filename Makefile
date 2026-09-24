.PHONY: help install install-gpu install-server install-dev models test lint fmt cov benchmark docker-cpu docker-gpu clean

PY ?= python3
VENV ?= .venv

help:
	@echo "PixelBoost development targets"
	@echo ""
	@echo "  make install          CPU runtime (numpy + pillow + onnxruntime)"
	@echo "  make install-gpu      GPU runtime (onnxruntime-gpu)"
	@echo "  make install-server   add the HTTP API extras"
	@echo "  make install-dev      everything needed to develop and test"
	@echo "  make models           download the default model set"
	@echo "  make test             run the test suite"
	@echo "  make lint             ruff check"
	@echo "  make fmt              ruff format"
	@echo "  make cov              tests with coverage"
	@echo "  make benchmark        tile-size sweep on a synthetic image"
	@echo "  make docker-cpu       build the CPU container"
	@echo "  make docker-gpu       build the CUDA container"

install:
	$(PY) -m pip install -e ".[onnx-cpu,yaml]"

install-gpu:
	$(PY) -m pip install -e ".[onnx-gpu,yaml]"

install-server:
	$(PY) -m pip install -e ".[server,yaml]"

install-dev:
	$(PY) -m pip install -e ".[onnx-cpu,server,yaml,dev]"

models:
	$(PY) scripts/download_models.py --all

test:
	$(PY) -m pytest tests -q

cov:
	$(PY) -m pytest tests --cov=pixelboost --cov-report=term-missing --cov-report=html

lint:
	$(PY) -m ruff check src tests scripts

fmt:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

benchmark:
	$(PY) scripts/benchmark.py --size 512 --scale 4

docker-cpu:
	docker build -f docker/Dockerfile.cpu -t pixelboost:cpu .

docker-gpu:
	docker build -f docker/Dockerfile.gpu -t pixelboost:gpu .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.py[co]' -delete
