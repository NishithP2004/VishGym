#!/bin/sh
set -eu

uvicorn vishgym.api.main:app --host 127.0.0.1 --port 8000 &
streamlit run app/vishgym/ui/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true &
exec nginx -g 'daemon off;'
