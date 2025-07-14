#!/bin/bash
echo "Starting PDF to Markdown converter..."
echo "Environment check:"
# Removed R2 environment variable checks
echo ""
echo "Testing imports..."
python test_imports.py

if [ $? -ne 0 ]; then
    echo "❌ Import test failed - check requirements.txt"
    exit 1
fi

echo ""
echo "✅ All tests passed, starting server..."
# Start the FastAPI server
# Set default port if not provided
PORT=${PORT:-8000}

# Start the FastAPI app with uvicorn
uvicorn app:app --host 0.0.0.0 --port $PORT --reload

echo "\nApp running!"
echo "- Main page:     http://127.0.0.1:$PORT/"
echo "- Upload API:    http://127.0.0.1:$PORT/upload (POST)"
echo "- Health check:  http://127.0.0.1:$PORT/health" 