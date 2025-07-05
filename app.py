import os, uuid, io
import logging
import tempfile
import base64
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 50)
logger.info("Starting Document to Markdown converter application...")
logger.info("=" * 50)

# Try to import required libraries with error handling
logger.info("Importing required libraries...")

try:
    import openai
    logger.info("✅ OpenAI imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import openai: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    import fitz  # PyMuPDF for PDFs
    logger.info("✅ PyMuPDF imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import PyMuPDF: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    from pptx import Presentation
    logger.info("✅ python-pptx imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import python-pptx: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    from PIL import Image
    logger.info("✅ Pillow imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Pillow: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

logger.info("All libraries imported successfully!")

app = FastAPI(title="Document to Markdown Converter")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
openai_client = None

def initialize_openai():
    """Initialize OpenAI client with proper error handling"""
    global openai_client
    
    logger.info("Initializing OpenAI client...")
    
    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        logger.info(f"OPENAI_API_KEY configured: {OPENAI_API_KEY is not None}")
        
        if not OPENAI_API_KEY:
            logger.error("❌ Missing OPENAI_API_KEY environment variable")
            raise ValueError("Missing required OPENAI_API_KEY environment variable")
        
        logger.info("Creating OpenAI client...")
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Test the client with a simple call
        logger.info("Testing OpenAI client...")
        test_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logger.info("✅ OpenAI client test successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 50)
    logger.info("Application starting up...")
    logger.info("=" * 50)
    
    try:
        if not initialize_openai():
            logger.error("❌ Failed to initialize OpenAI - application may not work properly")
        else:
            logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

@app.get("/", response_class=HTMLResponse)
def index():
    logger.info("Serving index page")
    try:
        with open("static/index.html", encoding="utf8") as f:
            content = f.read()
            logger.info(f"✅ Index page loaded successfully ({len(content)} characters)")
            return content
    except Exception as e:
        logger.error(f"❌ Error reading index.html: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, "Internal server error")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"✅ Image converted to base64 ({len(base64_str)} characters)")
        return base64_str
    except Exception as e:
        logger.error(f"❌ Error converting image to base64: {e}")
        raise

def pdf_to_images(pdf_data: bytes):
    """Convert PDF bytes to list of PIL Images"""
    try:
        logger.info(f"Opening PDF with {len(pdf_data)} bytes...")
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        images = []
        
        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}...")
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.pil_tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
            logger.info(f"✅ Page {page_num + 1} converted to image")
        
        doc.close()
        logger.info(f"✅ PDF converted to {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"❌ Error converting PDF to images: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def pptx_slide_texts(pptx_data: bytes):
    """Extract text from PowerPoint bytes"""
    temp_file_path = None
    try:
        logger.info("Creating temporary PPTX file...")
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_file:
            temp_file.write(pptx_data)
            temp_file_path = temp_file.name
        
        logger.info("Opening PowerPoint presentation...")
        prs = Presentation(temp_file_path)
        slides_text = []
        
        for slide_num, slide in enumerate(prs.slides):
            logger.info(f"Processing slide {slide_num + 1}/{len(prs.slides)}...")
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
            slides_text.append("\n".join(text))
            logger.info(f"✅ Slide {slide_num + 1} text extracted")
        
        logger.info(f"✅ PowerPoint converted to {len(slides_text)} slides")
        return slides_text
    except Exception as e:
        logger.error(f"❌ Error extracting PowerPoint text: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if temp_file_path:
            try:
                os.unlink(temp_file_path)
                logger.info("✅ Temporary PPTX file cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up temporary PPTX file: {e}")

def call_openai_with_text(text: str) -> str:
    """Call OpenAI API with text content"""
    try:
        logger.info(f"Calling OpenAI with text ({len(text)} characters)...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a document-to-Markdown converter. Convert the given content to clean, well-formatted Markdown while preserving structure, headings, lists, and formatting."},
                {"role": "user", "content": f"Convert this to Markdown:\n\n{text}"}
            ],
            max_tokens=4000
        )
        result = response.choices[0].message.content
        logger.info(f"✅ OpenAI text conversion successful ({len(result)} characters)")
        return result
    except Exception as e:
        logger.error(f"❌ Error calling OpenAI with text: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def call_openai_with_image(base64_image: str) -> str:
    """Call OpenAI API with image content"""
    try:
        logger.info(f"Calling OpenAI with image ({len(base64_image)} base64 characters)...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all visible text and format it as clean Markdown. Preserve structure, headings, lists, and formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=4000
        )
        result = response.choices[0].message.content
        logger.info(f"✅ OpenAI image conversion successful ({len(result)} characters)")
        return result
    except Exception as e:
        logger.error(f"❌ Error calling OpenAI with image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    logger.info("=" * 50)
    logger.info(f"Upload request received for file: {file.filename}")
    logger.info("=" * 50)
    
    # Check if OpenAI is properly initialized
    if not openai_client:
        logger.error("❌ OpenAI client not initialized")
        raise HTTPException(500, "AI service not available")
    
    try:
        # Read file data
        logger.info("Reading file data...")
        data = await file.read()
        logger.info(f"✅ File read successfully ({len(data)} bytes)")
        
        # Determine file type and process accordingly
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        content_type = file.content_type.lower()
        
        logger.info(f"File type: {content_type}, extension: {file_extension}")
        
        if content_type == "application/pdf" or file_extension == "pdf":
            # Process PDF
            logger.info("Processing PDF file...")
            images = pdf_to_images(data)
            logger.info(f"Extracted {len(images)} pages from PDF")
            
            # Process each page with OpenAI
            markdown_parts = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)} with OpenAI...")
                base64_image = image_to_base64(image)
                page_markdown = call_openai_with_image(base64_image)
                markdown_parts.append(f"## Page {i+1}\n\n{page_markdown}")
            
            md = "\n\n---\n\n".join(markdown_parts)
            
        elif content_type in ["application/vnd.openxmlformats-officedocument.presentationml.presentation", 
                             "application/vnd.ms-powerpoint"] or file_extension in ["pptx", "ppt"]:
            # Process PowerPoint
            logger.info("Processing PowerPoint file...")
            slides_text = pptx_slide_texts(data)
            logger.info(f"Extracted {len(slides_text)} slides from PowerPoint")
            
            # Combine all slide text and convert to markdown
            combined_text = "\n\n---\n\n".join([f"Slide {i+1}:\n{text}" for i, text in enumerate(slides_text)])
            md = call_openai_with_text(combined_text)
            
        else:
            # For other file types, try to extract text and convert
            logger.info("Processing as text file...")
            try:
                text_content = data.decode('utf-8')
                md = call_openai_with_text(text_content)
            except UnicodeDecodeError:
                logger.error(f"❌ Unsupported file type: {content_type}")
                raise HTTPException(400, "Unsupported file type. Please upload PDF, PowerPoint, or text files.")
        
        logger.info(f"✅ Conversion successful, markdown length: {len(md)} characters")
        
        # Generate a unique file ID for reference
        file_id = str(uuid.uuid4())
        
        response_data = {
            "file_id": file_id,
            "filename": file.filename,
            "markdown": md,
            "pages_processed": len(images) if 'images' in locals() else 1
        }
        
        logger.info("✅ Returning successful response")
        return JSONResponse(response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        logger.info("Re-raising HTTP exception")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during upload: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Internal server error: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "openai_configured": openai_client is not None,
        "environment_vars": {
            "openai_api_key": os.getenv("OPENAI_API_KEY") is not None
        }
    }
