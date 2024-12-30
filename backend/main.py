from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from training.trainer_pipeline import TrainingPipeline
import uvicorn
import json
import logging
import sys
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs/qure.log"))
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

app = FastAPI(
    title="Qure Medical AI API",
    description="Enterprise-grade medical language model API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration with environment-based settings
ALLOWED_ORIGINS = {
    "development": ["*"],
    "production": [
        "https://qure.ai",
        "https://api.qure.ai",
        "https://dashboard.qure.ai"
    ]
}

ENV = "development"  # Change to "production" in production environment

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS[ENV],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System information for health check
SYSTEM_INFO = {
    "os": platform.system(),
    "python_version": platform.python_version(),
    "processor": platform.processor(),
    "machine": platform.machine()
}

# Initialize model with error handling
try:
    model = TrainingPipeline()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    confidence_threshold: float = Field(0.85, ge=0.0, le=1.0)
    context: Optional[Dict] = None

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the contraindications for metformin?",
                "confidence_threshold": 0.85,
                "context": {"patient_age": 45, "existing_conditions": ["diabetes", "hypertension"]}
            }
        }

class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=150)
    conditions: List[str] = Field(..., min_items=0, max_items=50)
    medications: List[str] = Field(..., min_items=0, max_items=50)
    vitals: Optional[Dict[str, float]] = None
    lab_results: Optional[Dict[str, Union[float, str]]] = None

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "conditions": ["type_2_diabetes", "hypertension"],
                "medications": ["metformin", "lisinopril"],
                "vitals": {"blood_pressure": "120/80", "heart_rate": 72},
                "lab_results": {"hba1c": 7.2, "glucose": 140}
            }
        }

class RecommendationRequest(BaseModel):
    condition: str = Field(..., min_length=1, max_length=100)
    patient_data: PatientData
    include_references: bool = True

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred",
            "detail": str(exc) if ENV == "development" else "Please contact support"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint with system information"""
    try:
        return {
            "status": "healthy",
            "model_version": model.version,
            "system_info": SYSTEM_INFO,
            "gpu_info": model.get_gpu_info()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/v2/query")
async def process_query(request: QueryRequest):
    """Process a medical query with enhanced error handling"""
    logger.info(f"Processing query: {request.query[:100]}...")
    try:
        response = model.process_query(
            query=request.query,
            confidence_threshold=request.confidence_threshold,
            context=request.context
        )
        logger.info(f"Query processed successfully with confidence: {response.get('confidence', 0)}")
        return response
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e)
            }
        )

@app.post("/api/v2/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get medical recommendations with enhanced validation"""
    logger.info(f"Generating recommendations for condition: {request.condition}")
    try:
        recommendations = model.get_recommendations(
            condition=request.condition,
            patient_data=request.patient_data.dict(),
            include_references=request.include_references
        )
        logger.info("Recommendations generated successfully")
        return recommendations
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate recommendations",
                "message": str(e)
            }
        )

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time medical chat interface with improved error handling"""
    await websocket.accept()
    logger.info("New WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            query = json.loads(data)
            logger.debug(f"Received WebSocket query: {query}")
            
            response = model.process_query(
                query=query["text"],
                confidence_threshold=query.get("confidence_threshold", 0.85)
            )
            
            await websocket.send_json(response)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {str(e)}")
        await websocket.send_json({
            "status": "error",
            "message": "Invalid JSON format"
        })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        await websocket.close(code=1000, reason=str(e))

@app.post("/api/v2/analyze/interactions")
async def analyze_drug_interactions(medications: List[str] = Field(..., min_items=1, max_items=20)):
    """Analyze drug interactions with input validation"""
    logger.info(f"Analyzing interactions for medications: {medications}")
    try:
        interactions = model.analyze_interactions(medications)
        logger.info("Drug interaction analysis completed")
        return interactions
    except Exception as e:
        logger.error(f"Drug interaction analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to analyze drug interactions",
                "message": str(e)
            }
        )

@app.post("/api/v2/analyze/symptoms")
async def analyze_symptoms(
    symptoms: List[str] = Field(..., min_items=1, max_items=20),
    patient_data: Optional[PatientData] = None
):
    """Analyze symptoms with enhanced validation and error handling"""
    logger.info(f"Analyzing symptoms: {symptoms}")
    try:
        analysis = model.analyze_symptoms(
            symptoms,
            patient_data.dict() if patient_data else None
        )
        logger.info("Symptom analysis completed")
        return analysis
    except Exception as e:
        logger.error(f"Symptom analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to analyze symptoms",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    # Determine the appropriate host based on the environment
    host = "0.0.0.0" if ENV == "production" else "127.0.0.1"
    port = 8000
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=ENV == "development",
        log_level="info",
        access_log=True
    ) 