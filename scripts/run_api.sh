#!/usr/bin/env bash
echo "Starting Fraud Detection API..."
uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --reload
