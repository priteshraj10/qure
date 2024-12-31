from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional
import logging
from ..models.vision_model import EnhancedVisionModelHandler

router = APIRouter()
logger = logging.getLogger(__name__)
model_handler = EnhancedVisionModelHandler()

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    modality: Optional[str] = "xray"
):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
            
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(400, "Invalid image file")
            
        # Get prediction
        result = model_handler.analyze_image(image, modality)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500) 