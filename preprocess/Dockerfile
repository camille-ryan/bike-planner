FROM python:3.11-slim-bookworm

# pyrosm pulls in pyrobuf which builds C extensions; keep build-essential
# only for the install step and drop afterward.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-mod-spatialite \
        ca-certificates \
        libgeos-c1v5 \
        libexpat1 \
        libbz2-1.0 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY . .

ENV DATA_DIR=/data
ENTRYPOINT ["python3", "main.py"]
