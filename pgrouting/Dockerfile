FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-mod-spatialite \
        ca-certificates \
        libgeos-c1v5 \
        libexpat1 \
        libbz2-1.0 \
        zlib1g \
        postgresql-client \
        osmium-tool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY . .

ENV DATA_DIR=/data
ENV PGHOST=postgres
ENV PGUSER=bike
ENV PGPASSWORD=bike
ENV PGDATABASE=bike
# Make /app (== pgrouting/) importable regardless of the invocation
# entry point — so both `python3 main.py <sub>` and
# `python3 /app/chain/build_ways_paved.py` can resolve `import config`
# and the intra-package `from ingest import ...` imports.
ENV PYTHONPATH=/app
ENTRYPOINT ["python3", "main.py"]
