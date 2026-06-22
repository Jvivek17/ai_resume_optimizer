import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader 
import google.generativeai as genai
from dotenv import load_dotenv

# loading the secret variables. 
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize the FastAPI app
app = FastAPI()

# Allow the react frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Message": "Welcome to ATS optimizer! The Server is Running"}

@app.get("/check-env")
def check_env():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return {"staus": "API Key Found"}
    return {"Status" : "API key is not Found"}


@app.post("/api/parse-pdf")
async def parse_pdf(resume: UploadFile = File(...)):
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are suported")
    
    try:
        reader = PdfReader(resume.file)

        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
        
        return{
            "filename": resume.filename,
            "text": extracted_text.strip()
        }
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail=f"Failed to read the PDF file: {str(e)}")


@app.post("/api/optimize")
async def optimize_reumse(
    job_title : str = Form(...),
    job_description : str = Form(...),
    resume: UploadFile = File(...)
    ):

    """
    Takes a PDF resume, a job title, and a job description.
    # Extracts the text and sends it to Gemini for ATS optimization
    """

    # 1. parse the PDF
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail = "Only PDF files are supported")
    
    try: 
        reader = PdfReader(resume.file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to read the PDF file: {str(e)}")


    # 2. Construct the AI Prompt
    prompt = f""" 
    You are an expert ATS Resume Optimizer. I will provide you with a candidate's original resume, a target job title, and a job description.
    Your task is to rewrite the resume to highly align with the job description for Applicant Tracking System (ATS) optimization.
    
    Rules:
    1. Keep the facts accurate; DO NOT invent prior experience, skills, or degrees.
    2. Incorporate keywords seamlessly from the job description.
    3. Start bullet points with strong action verbs and quantify achievements if possible.
    4. Return ONLY the finalized optimized resume in clean Markdown format. Do not include introductory chatter.

    Target Job Title: {job_title}
    
    Job Description: 
    {job_description}

    Original Resume Text:
    {extracted_text}
    """

    # 3. Call the Gemini AI.

    try: 
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)

        return {"optimized_resume": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing failed: {str(e)}")
