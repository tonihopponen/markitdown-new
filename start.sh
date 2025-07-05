#!/bin/bash
echo "Starting PDF to Markdown converter..."
echo "Environment check:"
echo "- R2_ENDPOINT_URL: ${R2_ENDPOINT_URL:+SET}"
echo "- R2_BUCKET_NAME: ${R2_BUCKET_NAME:+SET}"
echo "- R2_ACCOUNT_ID: ${R2_ACCOUNT_ID:+SET}"
echo "- R2_ACCESS_KEY_ID: ${R2_ACCESS_KEY_ID:+SET}"
echo "- R2_SECRET_ACCESS_KEY: ${R2_SECRET_ACCESS_KEY:+SET}"

# Start the FastAPI server
exec uvicorn app:app --host 0.0.0.0 --port $PORT 