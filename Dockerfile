# trnscrb server container — serves transcripts + MCP (``trnscrb serve``).
#
# Build:  docker build -t trnscrb-server:latest .
# Run:    docker run -d -p 8765:8765 -v trnscrb_home:/home/trnscrb trnscrb-server:latest
#
# The dependency set mirrors install-server.sh: mcp 2.2.0 (ships starlette,
# uvicorn and sse-starlette), numpy + sounddevice for the import chain, and
# trnscrb itself with --no-deps — its macOS-only deps (rumps/pyobjc, MLX)
# are neither installed nor needed by the store toolset.
#
# State (settings incl. the server token, and the transcript store) lives in
# /home/trnscrb — mount a volume there so restarts and redeploys keep it.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY trnscrb/ trnscrb/
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir "mcp==2.2.0" numpy sounddevice click

ENV HOME=/home/trnscrb \
    TRNSCRB_LOG_DIR=/tmp/trnscrb-logs
VOLUME ["/home/trnscrb"]
EXPOSE 8765

CMD ["trnscrb", "serve", "--host", "0.0.0.0"]