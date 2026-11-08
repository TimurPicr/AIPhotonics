FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY config.py ./
COPY src/ ./src/
COPY weights/h100.pt ./weights/h100.pt
COPY examples/ ./examples/

ENTRYPOINT ["python", "-m", "src.infer"]
CMD ["--help"]

FROM runtime AS test
COPY requirements-dev.txt ./
RUN python -m pip install --no-cache-dir -r requirements-dev.txt
COPY tests/ ./tests/
RUN python -m pytest -q -p no:cacheprovider

FROM runtime AS final
