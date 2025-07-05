import os, uuid, io
import logging
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting PDF to Markdown converter application...")

# Try to import apify client with error handling
try:
    from apify_client import ApifyClient
    logger.info("Apify client imported successfully")
except ImportError as e:
    logger.error(f"Failed to import apify-client: {e}")
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
apify_client = None

def initialize_services():
    """Initialize R2 client and Apify client with proper error handling"""
    global R2_ENDPOINT, R2_BUCKET, R2_ACCOUNT_ID, s3_client, apify_client
    
    try:
        # Initialize R2
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
        
        # Validate required R2 environment variables
        if not all([R2_ENDPOINT, R2_BUCKET, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
            missing_vars = []
            if not R2_ENDPOINT: missing_vars.append("R2_ENDPOINT_URL")
            if not R2_BUCKET: missing_vars.append("R2_BUCKET_NAME")
            if not R2_ACCOUNT_ID: missing_vars.append("R2_ACCOUNT_ID")
            if not R2_ACCESS_KEY_ID: missing_vars.append("R2_ACCESS_KEY_ID")
            if not R2_SECRET_ACCESS_KEY: missing_vars.append("R2_SECRET_ACCESS_KEY")
            raise ValueError(f"Missing required R2 environment variables: {', '.join(missing_vars)}")
        
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
        
        # Initialize Apify
        APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
        logger.info(f"APIFY_API_TOKEN configured: {APIFY_API_TOKEN is not None}")
        
        if not APIFY_API_TOKEN:
            raise ValueError("Missing required APIFY_API_TOKEN environment variable")
        
        apify_client = ApifyClient(APIFY_API_TOKEN)
        logger.info("Apify client initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Application starting up...")
    if not initialize_services():
        logger.error("Failed to initialize services - application may not work properly")

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
    
    # Check if services are properly initialized
    if not s3_client or not apify_client:
        logger.error("Services not initialized")
        raise HTTPException(500, "Services not available")
    
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
        
        # Construct the public URL for the uploaded file
        public_url = f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.dev/{key}"
        logger.info(f"Public URL: {public_url}")
        
        # Convert PDF to markdown using Apify Docling API
        logger.info("Starting PDF to markdown conversion using Apify Docling API")
        
        # Prepare the Apify Actor input
        run_input = {
            "http_sources": [{ "url": public_url }],
            "options": { "to_formats": ["md"] },
        }
        
        # Run the Apify Actor and wait for it to finish
        logger.info("Calling Apify Docling Actor...")
        run = apify_client.actor("vancura/docling").call(run_input=run_input)
        logger.info(f"Apify run completed with ID: {run['id']}")
        
        # Fetch results from the run's dataset
        logger.info("Fetching conversion results...")
        results = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
        
        if not results:
            raise HTTPException(500, "No results returned from Apify Docling")
        
        # Extract markdown from the first result
        result = results[0]
        if "markdown" not in result:
            raise HTTPException(500, "Markdown not found in Apify results")
        
        md = result["markdown"]
        logger.info(f"Conversion successful, markdown length: {len(md)} characters")
        
        return JSONResponse({"file_url": public_url, "markdown": md})
        
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
        "apify_configured": apify_client is not None,
        "environment_vars": {
            "r2_endpoint": R2_ENDPOINT is not None,
            "r2_bucket": R2_BUCKET is not None,
            "r2_account_id": R2_ACCOUNT_ID is not None,
            "apify_token": os.getenv("APIFY_API_TOKEN") is not None
        }
    }
