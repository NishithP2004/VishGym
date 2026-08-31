#!/bin/sh
set -eu

if [ "${VISHGYM_FRONTEND_LOCAL_API:-0}" = "1" ]; then
  uvicorn vishgym.api.main:app --host 127.0.0.1 --port 8000 &
fi
streamlit run app/vishgym/ui/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true &
exec nginx -g 'daemon off;'
