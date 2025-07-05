#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality
"""
import sys
import os
import time

print("Testing imports...")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")
    sys.exit(1)

try:
    import boto3
    print("✅ boto3 imported successfully")
except ImportError as e:
    print(f"❌ boto3 import failed: {e}")
    sys.exit(1)

print("Loading apify-client...")
start_time = time.time()
try:
    from apify_client import ApifyClient
    load_time = time.time() - start_time
    print(f"✅ apify-client imported successfully (took {load_time:.2f}s)")
except ImportError as e:
    print(f"❌ apify-client import failed: {e}")
    sys.exit(1)

try:
    import uvicorn
    print("✅ uvicorn imported successfully")
except ImportError as e:
    print(f"❌ uvicorn import failed: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("\nEnvironment variables check:")
env_vars = [
    "R2_ENDPOINT_URL",
    "R2_BUCKET_NAME", 
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "APIFY_API_TOKEN"
]

for var in env_vars:
    value = os.getenv(var)
    status = "✅ SET" if value else "❌ NOT SET"
    print(f"  {var}: {status}")

print("\nReady to run!") 