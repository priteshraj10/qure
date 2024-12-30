from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
from training.trainer_pipeline import TrainingPipeline
import uvicorn
import json

app = FastAPI(
    title="Qure Medical AI API",
    description="Enterprise-grade medical language model API",
    version="2.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = TrainingPipeline()

class QueryRequest(BaseModel):
    query: str
    confidence_threshold: float = 0.85
    context: Optional[Dict] = None

class PatientData(BaseModel):
    age: int
    conditions: List[str]
    medications: List[str]
    vitals: Optional[Dict[str, float]] = None
    lab_results: Optional[Dict[str, Union[float, str]]] = None

class RecommendationRequest(BaseModel):
    condition: str
    patient_data: PatientData
    include_references: bool = True

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_version": model.version}

@app.post("/api/v2/query")
async def process_query(request: QueryRequest):
    """Process a medical query"""
    try:
        response = model.process_query(
            query=request.query,
            confidence_threshold=request.confidence_threshold,
            context=request.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get medical recommendations"""
    try:
        recommendations = model.get_recommendations(
            condition=request.condition,
            patient_data=request.patient_data.dict(),
            include_references=request.include_references
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time medical chat interface"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            query = json.loads(data)
            
            # Process the query through the model
            response = model.process_query(
                query=query["text"],
                confidence_threshold=query.get("confidence_threshold", 0.85)
            )
            
            await websocket.send_json(response)
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))

@app.post("/api/v2/analyze/interactions")
async def analyze_drug_interactions(medications: List[str]):
    """Analyze drug interactions"""
    try:
        interactions = model.analyze_interactions(medications)
        return interactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analyze/symptoms")
async def analyze_symptoms(symptoms: List[str], patient_data: Optional[PatientData] = None):
    """Analyze symptoms and suggest possible conditions"""
    try:
        analysis = model.analyze_symptoms(symptoms, patient_data.dict() if patient_data else None)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 