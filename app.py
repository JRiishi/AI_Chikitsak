from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from main import get_medical_analysis_from_file
import tempfile
import os

# Create the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Root GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}

@app.post("/MedicalHistoryPdf")
async def analyze_medical_history(
    file: UploadFile = File(...),
    problem: str = Form(...)
):
    # Validate file type
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400, 
            content={"error": "Only PDF files are allowed"}
        )
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Read and write file content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        analysis = get_medical_analysis_from_file(temp_file_path, problem)
        
        # Clean up - delete temporary file
        os.unlink(temp_file_path)
        
        print("Analysis Result:", analysis)  # Debugging line
        return {"analysis": analysis}
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing file: {str(e)}"}
        )