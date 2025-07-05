#!/usr/bin/env python3
"""
Test script to verify all required dependencies can be imported.
Run this to check if your environment is properly set up.
"""

import sys
import traceback

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = __import__(package_name)
        else:
            module = __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    print("=" * 50)
    print("Testing all required dependencies...")
    print("=" * 50)
    
    # Core FastAPI dependencies
    print("\n--- Core Dependencies ---")
    test_import("fastapi")
    test_import("uvicorn")
    test_import("multipart", "python-multipart")
    
    # OpenAI
    print("\n--- AI/ML Dependencies ---")
    test_import("openai")
    
    # Document processing
    print("\n--- Document Processing ---")
    test_import("fitz", "PyMuPDF")  # PDF processing
    test_import("pptx", "python-pptx")  # PowerPoint processing
    test_import("docx", "python-docx")  # Word document processing
    test_import("PIL", "Pillow")  # Image processing
    
    # Data processing
    print("\n--- Data Processing ---")
    test_import("pandas")
    test_import("openpyxl")
    test_import("markdownify")
    test_import("tabulate")
    
    print("\n" + "=" * 50)
    print("Import testing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main() 