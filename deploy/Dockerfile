# GlitchHunter container: OpenCV 5 + headless Chromium + agent + dashboard.
# Multi-arch: builds natively on AWS Graviton (linux/arm64) and on x86_64.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    GH_WORK_DIR=/data/work \
    GH_EMF=stdout \
    GH_APPROVAL_MODE=web

# git drives the game repo (bisect, revert); Liberation fonts keep HUD text metrics close to Arial.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git fonts-liberation fonts-dejavu-core ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -m playwright install --with-deps chromium

COPY glitchhunter ./glitchhunter
COPY games ./games
COPY bridges ./bridges
COPY deploy/entrypoint.sh /entrypoint.sh

RUN useradd --create-home --uid 10001 gh \
 && mkdir -p /data /app/models \
 && chown -R gh /data /app \
 && chmod +x /entrypoint.sh
USER gh

# OpenCV Zoo PP-OCRv3 + CRNN text models, pinned by SHA-256.
RUN python -m glitchhunter fetch-models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
