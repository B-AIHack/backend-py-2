
from fastapi import FastAPI, Body
from pydantic import BaseModel
from processor import process_pdf, process_text, read_pdf
from compliance import compliance_validation
from fastapi import FastAPI, UploadFile, File, HTTPException
from egrul import get_owners 

app = FastAPI()

class Document(BaseModel):
    text: str

class PdfTextRequest(BaseModel):
    file_text: str

@app.post("/process/")
async def process(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(400, "Нужен PDF-файл")

    pdf_bytes = await file.read()           
    return process_pdf(pdf_bytes)           


@app.get("/egrul/")
def process(bin: str):
    return get_owners(bin)


@app.post("/ocr/")
async def process(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(400, "Нужен PDF-файл")
    pdf_bytes = await file.read()           
    return { 'result' : read_pdf(pdf_bytes)}   



@app.post("/processText/")
async def process(request: PdfTextRequest):
    return process_text(request.file_text)

@app.post("/compliance/")
async def process(request: PdfTextRequest):
    return compliance_validation(request.file_text)

    
