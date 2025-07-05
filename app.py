import os, uuid, io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
from docling.document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Cloudflare R2 configuration (env vars set on Render) ----
# Use the jurisdiction-specific endpoint instead of the generic one
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")  # This should be your jurisdiction-specific URL
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")

# Log configuration (without sensitive data)
logger.info(f"R2_ENDPOINT configured: {R2_ENDPOINT is not None}")
logger.info(f"R2_BUCKET configured: {R2_BUCKET is not None}")
logger.info(f"R2_ACCOUNT_ID configured: {R2_ACCOUNT_ID is not None}")

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=boto3.session.Config(s3={"addressing_style": "virtual"}),
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # Upload to R2
        logger.info(f"Uploading to R2 bucket: {R2_BUCKET}")
        s3.upload_fileobj(io.BytesIO(data), R2_BUCKET, key)
        logger.info("File uploaded to R2 successfully")
        
        # Construct the public URL for the uploaded file
        public_url = f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.dev/{key}"
        logger.info(f"Public URL: {public_url}")
        
        # Convert PDF to markdown
        logger.info("Starting PDF to markdown conversion")
        converter = DocumentConverter()
        md = converter.convert(public_url).document.export_to_markdown()
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
    return {"status": "healthy"}
