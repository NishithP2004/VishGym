FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
RUN python3 -m pip install --no-cache-dir \
      'pydantic>=2.8' 'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'streamlit>=1.39'

COPY app ./app
RUN python3 -m pip install --no-cache-dir .

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 7860
CMD ["/workspace/start.sh"]
