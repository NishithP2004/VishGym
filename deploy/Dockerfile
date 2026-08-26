FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/workspace/.cache/huggingface
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip nginx && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY app ./app
RUN python3 -m pip install --break-system-packages --no-cache-dir --upgrade pip && \
    python3 -m pip install --break-system-packages --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch && \
    python3 -m pip install --break-system-packages --no-cache-dir '.[app,audio,training]'

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 7860
CMD ["/workspace/start.sh"]
