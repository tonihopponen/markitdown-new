import os, uuid, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import boto3
from docling.document_converter import DocumentConverter

# ---- Cloudflare R2 configuration (env vars set on Render) ----
# Use the jurisdiction-specific endpoint instead of the generic one
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")  # This should be your jurisdiction-specific URL
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=boto3.session.Config(s3={"addressing_style": "virtual"}),
)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf8") as f:
        return f.read()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files allowed")
    
    key = f"{uuid.uuid4()}.pdf"
    data = await file.read()
    
    # Upload to R2 using the bucket name in the upload call
    s3.upload_fileobj(io.BytesIO(data), R2_BUCKET, key)
    
    # Construct the public URL for the uploaded file
    public_url = f"https://{R2_BUCKET}.{R2_ACCOUNT_ID}.r2.dev/{key}"
    
    # Convert PDF to markdown
    converter = DocumentConverter()
    md = converter.convert(public_url).document.export_to_markdown()
    
    return JSONResponse({"file_url": public_url, "markdown": md})
