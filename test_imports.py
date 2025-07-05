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
    import openai
    print("✅ OpenAI imported successfully")
except ImportError as e:
    print(f"❌ OpenAI import failed: {e}")
    sys.exit(1)

print("Loading PyMuPDF...")
start_time = time.time()
try:
    import fitz
    load_time = time.time() - start_time
    print(f"✅ PyMuPDF imported successfully (took {load_time:.2f}s)")
except ImportError as e:
    print(f"❌ PyMuPDF import failed: {e}")
    sys.exit(1)

try:
    from pptx import Presentation
    print("✅ python-pptx imported successfully")
except ImportError as e:
    print(f"❌ python-pptx import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("✅ Pillow imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")
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
    "OPENAI_API_KEY"
]

for var in env_vars:
    value = os.getenv(var)
    status = "✅ SET" if value else "❌ NOT SET"
    print(f"  {var}: {status}")

print("\nReady to run!") 