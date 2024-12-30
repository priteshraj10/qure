from typing import Dict, Any
import json
from fastapi import WebSocket

class TrainingEventEmitter:
    def __init__(self, connection_manager):
        self.manager = connection_manager

    async def emit_metrics(self, metrics):
        await self.manager.broadcast({
            "type": "metrics",
            "payload": metrics
        })

    async def emit_status(self, status, details=None):
        await self.manager.broadcast({
            "type": "status",
            "payload": {
                "status": status,
                **(details or {})
            }
        }) 