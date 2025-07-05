import os, uuid, io
import logging
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting PDF to Markdown converter application...")

# Try to import docling with error handling
try:
    from docling.document_converter import DocumentConverter
    logger.info("Docling imported successfully")
except ImportError as e:
    logger.error(f"Failed to import docling: {e}")
    raise

app = FastAPI(title="PDF to Markdown Converter")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for R2 configuration
R2_ENDPOINT = None
R2_BUCKET = None
R2_ACCOUNT_ID = None
s3_client = None

def initialize_r2():
    """Initialize R2 client with proper error handling"""
    global R2_ENDPOINT, R2_BUCKET, R2_ACCOUNT_ID, s3_client
    
    try:
        R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
        R2_BUCKET = os.getenv("R2_BUCKET_NAME")
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
        
        # Log configuration (without sensitive data)
        logger.info(f"R2_ENDPOINT configured: {R2_ENDPOINT is not None}")
        logger.info(f"R2_BUCKET configured: {R2_BUCKET is not None}")
        logger.info(f"R2_ACCOUNT_ID configured: {R2_ACCOUNT_ID is not None}")
        logger.info(f"R2_ACCESS_KEY_ID configured: {R2_ACCESS_KEY_ID is not None}")
        logger.info(f"R2_SECRET_ACCESS_KEY configured: {R2_SECRET_ACCESS_KEY is not None}")
        
        # Validate required environment variables
        if not all([R2_ENDPOINT, R2_BUCKET, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
            missing_vars = []
            if not R2_ENDPOINT: missing_vars.append("R2_ENDPOINT_URL")
            if not R2_BUCKET: missing_vars.append("R2_BUCKET_NAME")
            if not R2_ACCOUNT_ID: missing_vars.append("R2_ACCOUNT_ID")
            if not R2_ACCESS_KEY_ID: missing_vars.append("R2_ACCESS_KEY_ID")
            if not R2_SECRET_ACCESS_KEY: missing_vars.append("R2_SECRET_ACCESS_KEY")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=boto3.session.Config(s3={"addressing_style": "virtual"}),
        )
        
        logger.info("R2 client initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize R2 client: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Application starting up...")
    if not initialize_r2():
        logger.error("Failed to initialize R2 - application may not work properly")

@app.get("/", response_class=HTMLResponse)
def index():
    logger.info("Serving index page")
    try:
        with open("static/index.html", encoding="utf8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    logger.info(f"Upload request received for file: {file.filename}")
    
    # Check if R2 is properly initialized
    if not s3_client:
        logger.error("R2 client not initialized")
        raise HTTPException(500, "Storage service not available")
    
    try:
        # Validate file type
        if file.content_type != "application/pdf":
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(400, "Only PDF files allowed")
        
        # Generate unique key
        key = f"{uuid.uuid4()}.pdf"
        logger.info(f"Generated key: {key}")
        
        # Read file data
        data = await file.read()
        logger.info(f"File size: {len(data)} bytes")
        
        # Upload to R2 using the bucket name in the upload call
        logger.info(f"Uploading to R2 bucket: {R2_BUCKET}")
        s3_client.upload_fileobj(io.BytesIO(data), R2_BUCKET, key)
        logger.info("File uploaded to R2 successfully")
        
        # Get the file back from R2 for processing (avoids SSL issues with public URL)
        logger.info("Retrieving file from R2 for processing...")
        response = s3_client.get_object(Bucket=R2_BUCKET, Key=key)
        pdf_data = response['Body'].read()
        logger.info(f"Retrieved {len(pdf_data)} bytes from R2")
        
        # Save to temporary file for Docling (it expects a file path)
        logger.info("Creating temporary file for Docling processing...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_file_path = temp_file.name
        
        try:
            # Convert PDF to markdown using the temporary file path
            logger.info("Starting PDF to markdown conversion using temporary file")
            converter = DocumentConverter()
            md = converter.convert(temp_file_path).document.export_to_markdown()
            logger.info(f"Conversion successful, markdown length: {len(md)} characters")
            
            # Construct public URL for response
            public_url = f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.dev/{key}"
            logger.info(f"Public URL: {public_url}")
            
            return JSONResponse({"file_url": public_url, "markdown": md})
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
        
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
        "r2_configured": s3_client is not None,
        "environment_vars": {
            "r2_endpoint": R2_ENDPOINT is not None,
            "r2_bucket": R2_BUCKET is not None,
            "r2_account_id": R2_ACCOUNT_ID is not None
        }
    }
