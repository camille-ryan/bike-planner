FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-requests \
        osmium-tool libsqlite3-mod-spatialite \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV DATA_DIR=/data
ENTRYPOINT ["python3", "main.py"]
