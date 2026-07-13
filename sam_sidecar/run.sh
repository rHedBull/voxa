#!/usr/bin/env bash
cd "$(dirname "$0")"
exec /home/hendrik/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8011
