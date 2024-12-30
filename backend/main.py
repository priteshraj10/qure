from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from training.config import TrainingConfig
from training.trainer_pipeline import TrainingPipeline
from training.events import TrainingEventEmitter
import logging
import asyncio
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                await self.disconnect(connection)

manager = ConnectionManager()

@app.post("/api/training/start")
async def start_training(config: dict):
    try:
        # Initialize training configuration
        training_config = TrainingConfig(
            model_name=config["modelName"],
            batch_size=config["batchSize"],
            learning_rate=config["learningRate"],
            num_epochs=config["maxEpochs"],
            device=config["deviceType"]
        )
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(training_config)
        
        # Start training in background task
        asyncio.create_task(pipeline.train(
            event_emitter=TrainingEventEmitter(manager)
        ))
        
        return {"status": "Training started successfully"}
        
    except Exception as e:
        logger.error(f"Training failed to start: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000) 