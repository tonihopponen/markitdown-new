import os, uuid, io
import logging
import tempfile
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting PDF to Markdown converter application...")

# Try to import required libraries with error handling
try:
    import openai
    logger.info("OpenAI imported successfully")
except ImportError as e:
    logger.error(f"Failed to import openai: {e}")
    raise

try:
    import fitz  # PyMuPDF for PDFs
    logger.info("PyMuPDF imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PyMuPDF: {e}")
    raise

try:
    from pptx import Presentation
    logger.info("python-pptx imported successfully")
except ImportError as e:
    logger.error(f"Failed to import python-pptx: {e}")
    raise

try:
    from PIL import Image
    logger.info("Pillow imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Pillow: {e}")
    raise

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
    
    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        logger.info(f"OPENAI_API_KEY configured: {OPENAI_API_KEY is not None}")
        
        if not OPENAI_API_KEY:
            raise ValueError("Missing required OPENAI_API_KEY environment variable")
        
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Application starting up...")
    if not initialize_openai():
        logger.error("Failed to initialize OpenAI - application may not work properly")

@app.get("/", response_class=HTMLResponse)
def index():
    logger.info("Serving index page")
    try:
        with open("static/index.html", encoding="utf8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(500, "Internal server error")

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def pdf_to_images(pdf_data: bytes):
    """Convert PDF bytes to list of PIL Images"""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img_data = pix.pil_tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        images.append(image)
    doc.close()
    return images

def pptx_slide_texts(pptx_data: bytes):
    """Extract text from PowerPoint bytes"""
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_file:
        temp_file.write(pptx_data)
        temp_file_path = temp_file.name
    
    try:
        prs = Presentation(temp_file_path)
        slides_text = []
        for slide in prs.slides:
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
            slides_text.append("\n".join(text))
        return slides_text
    finally:
        try:
            os.unlink(temp_file_path)
        except:
            pass

def call_openai_with_text(text: str) -> str:
    """Call OpenAI API with text content"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You're a document-to-Markdown converter. Convert the given content to clean, well-formatted Markdown while preserving structure, headings, lists, and formatting."},
            {"role": "user", "content": f"Convert this to Markdown:\n\n{text}"}
        ],
        max_tokens=4000
    )
    return response.choices[0].message.content

def call_openai_with_image(base64_image: str) -> str:
    """Call OpenAI API with image content"""
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
    return response.choices[0].message.content

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    logger.info(f"Upload request received for file: {file.filename}")
    
    # Check if OpenAI is properly initialized
    if not openai_client:
        logger.error("OpenAI client not initialized")
        raise HTTPException(500, "AI service not available")
    
    try:
        # Read file data
        data = await file.read()
        logger.info(f"File size: {len(data)} bytes")
        
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
                logger.info(f"Processing page {i+1}/{len(images)}...")
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
                raise HTTPException(400, "Unsupported file type. Please upload PDF, PowerPoint, or text files.")
        
        logger.info(f"Conversion successful, markdown length: {len(md)} characters")
        
        # Generate a unique file ID for reference
        file_id = str(uuid.uuid4())
        
        return JSONResponse({
            "file_id": file_id,
            "filename": file.filename,
            "markdown": md,
            "pages_processed": len(images) if 'images' in locals() else 1
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": openai_client is not None,
        "environment_vars": {
            "openai_api_key": os.getenv("OPENAI_API_KEY") is not None
        }
    }
