from typing import Dict, List, Optional, Union
import torch
from .model import MedicalLLM
from transformers import AutoTokenizer
import logging
import platform
import subprocess
import json
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs/pipeline.log"))
    ]
)
logger = logging.getLogger(__name__)

class SystemInfo:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_version = platform.python_version()
        self.cuda_version = None
        self._detect_cuda()

    def _detect_cuda(self):
        """Detect CUDA version if available"""
        try:
            if torch.cuda.is_available():
                self.cuda_version = torch.version.cuda
        except Exception as e:
            logger.warning(f"Error detecting CUDA version: {str(e)}")

    def to_dict(self) -> Dict:
        """Convert system information to dictionary"""
        return {
            "os_type": self.os_type,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "cuda_version": self.cuda_version
        }

class GPUInfo:
    def __init__(self):
        self.available = False
        self.device_name = "CPU"
        self.cuda_version = None
        self.device_count = 0
        self.memory_info = {}
        self.device_type = self._detect_device_type()
        self._detect_gpu()

    def _detect_device_type(self) -> str:
        """Detect the type of compute device available"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _detect_gpu(self):
        """Detect GPU and CUDA information across different platforms"""
        try:
            if self.device_type == "cuda":
                self.available = True
                self.device_count = torch.cuda.device_count()
                self.device_name = torch.cuda.get_device_name(0)
                self.cuda_version = torch.version.cuda
                
                # Get memory info for each GPU
                for i in range(self.device_count):
                    props = torch.cuda.get_device_properties(i)
                    self.memory_info[i] = {
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count
                    }
            elif self.device_type == "mps":
                self.available = True
                self.device_name = "Apple Silicon"
                self.device_count = 1
            
        except Exception as e:
            logger.warning(f"Error detecting GPU: {str(e)}")
            self.device_type = "cpu"

    def get_optimal_settings(self) -> Dict:
        """Get optimal settings for model initialization based on device"""
        settings = {
            "device_map": None,
            "torch_dtype": torch.float32,
            "use_mixed_precision": False
        }

        if self.device_type == "cuda":
            # For CUDA devices, use automatic device mapping and mixed precision
            settings.update({
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "use_mixed_precision": True
            })
        elif self.device_type == "mps":
            # For Apple Silicon, use specific optimizations
            settings.update({
                "device_map": None,
                "torch_dtype": torch.float16,
                "use_mixed_precision": False
            })

        return settings

    def to_dict(self) -> Dict:
        """Convert GPU information to dictionary"""
        return {
            "available": self.available,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "cuda_version": self.cuda_version,
            "device_count": self.device_count,
            "memory_info": self.memory_info
        }

class TrainingPipeline:
    def __init__(self):
        self.version = "2.0.0"
        self.system_info = SystemInfo()
        self.gpu_info = GPUInfo()
        self.device = self.gpu_info.device_type
        
        logger.info(f"System Info: {self.system_info.to_dict()}")
        logger.info(f"GPU Info: {self.gpu_info.to_dict()}")
        logger.info(f"Using device: {self.device}")
        
        # Create necessary directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize model and tokenizer with error handling
        try:
            self.model = self._initialize_model()
            self.tokenizer = self._initialize_tokenizer()
            self._verify_model_loading()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _initialize_model(self) -> MedicalLLM:
        """Initialize the model with appropriate settings based on device"""
        try:
            settings = self.gpu_info.get_optimal_settings()
            logger.info(f"Initializing model with settings: {settings}")
            
            model = MedicalLLM.from_pretrained(
                "qure-ai/medical-model-v2",
                device_map=settings["device_map"],
                torch_dtype=settings["torch_dtype"]
            )
            
            # Move model to device if not using device_map="auto"
            if settings["device_map"] is None:
                model.to(self.device)
            
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize the tokenizer with appropriate settings"""
        try:
            return AutoTokenizer.from_pretrained(
                "qure-ai/medical-model-v2",
                use_fast=True
            )
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

    def _verify_model_loading(self):
        """Verify model loaded correctly by running a test inference"""
        try:
            test_input = "Test medical query"
            inputs = self.tokenizer(
                test_input,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                self.model.generate(**inputs, max_length=20)
            logger.info("Model verification successful")
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            raise

    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "system": self.system_info.to_dict(),
            "gpu": self.gpu_info.to_dict()
        }

    def get_gpu_info(self) -> Dict:
        """Get GPU information"""
        return self.gpu_info.to_dict()

    def process_query(
        self,
        query: str,
        confidence_threshold: float = 0.85,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process a medical query with enhanced error handling and logging"""
        try:
            # Input validation
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Prepare input with progress logging
            logger.debug("Tokenizing input")
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            if context:
                inputs["context"] = context

            # Generate response with memory optimization
            logger.debug("Generating response")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            # Process response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            confidence = self.model.get_confidence(outputs)

            # Memory cleanup for GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Confidence threshold check
            if confidence < confidence_threshold:
                logger.warning(f"Low confidence response: {confidence}")
                return {
                    "status": "low_confidence",
                    "message": "Response confidence below threshold",
                    "confidence": confidence
                }

            logger.info(f"Query processed successfully with confidence: {confidence}")
            return {
                "status": "success",
                "response": response_text,
                "confidence": confidence,
                "references": self.model.get_references(response_text)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    def get_recommendations(
        self,
        condition: str,
        patient_data: Dict,
        include_references: bool = True
    ) -> Dict:
        """Generate medical recommendations with enhanced validation and error handling"""
        try:
            # Input validation
            if not condition.strip():
                raise ValueError("Condition cannot be empty")
            
            # Prepare input prompt
            prompt = self._format_recommendation_prompt(condition, patient_data)
            
            logger.debug("Tokenizing recommendation prompt")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            logger.debug("Generating recommendations")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.3
                )

            recommendations = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Memory cleanup for GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            response = {
                "status": "success",
                "recommendations": self._parse_recommendations(recommendations),
                "confidence": self.model.get_confidence(outputs)
            }

            if include_references:
                response["references"] = self.model.get_references(recommendations)

            logger.info("Recommendations generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            raise

    def analyze_interactions(self, medications: List[str]) -> Dict:
        """Analyze drug interactions with input validation"""
        try:
            # Input validation
            if not medications:
                raise ValueError("Medications list cannot be empty")
            
            # Prepare interaction analysis prompt
            prompt = self._format_interaction_prompt(medications)
            
            logger.debug("Tokenizing interaction prompt")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            logger.debug("Generating interaction analysis")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1
                )

            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Memory cleanup for GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Drug interaction analysis completed")
            return {
                "status": "success",
                "interactions": self._parse_interactions(analysis),
                "severity_levels": self._get_severity_levels(analysis),
                "references": self.model.get_references(analysis)
            }

        except Exception as e:
            logger.error(f"Error analyzing interactions: {str(e)}", exc_info=True)
            raise

    def analyze_symptoms(
        self,
        symptoms: List[str],
        patient_data: Optional[Dict] = None
    ) -> Dict:
        """Analyze symptoms with enhanced validation and error handling"""
        try:
            # Input validation
            if not symptoms:
                raise ValueError("Symptoms list cannot be empty")
            
            # Prepare symptom analysis prompt
            prompt = self._format_symptom_prompt(symptoms, patient_data)
            
            logger.debug("Tokenizing symptom prompt")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            logger.debug("Generating symptom analysis")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=3
                )

            analysis = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Memory cleanup for GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Symptom analysis completed")
            return {
                "status": "success",
                "possible_conditions": self._parse_conditions(analysis),
                "confidence_scores": self.model.get_confidence(outputs),
                "recommendations": self._get_recommendations(analysis),
                "references": self.model.get_references(str(analysis))
            }

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {str(e)}", exc_info=True)
            raise

    def _format_recommendation_prompt(self, condition: str, patient_data: Dict) -> str:
        """Format prompt for recommendation generation with validation"""
        try:
            prompt = f"""Provide evidence-based recommendations for {condition}.
Patient Information:
- Age: {patient_data['age']}
- Existing Conditions: {', '.join(patient_data['conditions'])}
- Current Medications: {', '.join(patient_data['medications'])}
{self._format_additional_data(patient_data)}
"""
            return prompt.strip()
        except KeyError as e:
            raise ValueError(f"Missing required patient data field: {str(e)}")

    def _format_interaction_prompt(self, medications: List[str]) -> str:
        """Format prompt for drug interaction analysis"""
        return f"Analyze potential interactions between the following medications: {', '.join(medications)}"

    def _format_symptom_prompt(self, symptoms: List[str], patient_data: Optional[Dict]) -> str:
        """Format prompt for symptom analysis"""
        prompt = f"Analyze the following symptoms: {', '.join(symptoms)}"
        if patient_data:
            prompt += f"\nPatient Context: {self._format_additional_data(patient_data)}"
        return prompt

    def _format_additional_data(self, data: Dict) -> str:
        """Format additional patient data with validation"""
        try:
            additional = []
            if data.get("vitals"):
                additional.append("Vitals: " + ", ".join(f"{k}: {v}" for k, v in data["vitals"].items()))
            if data.get("lab_results"):
                additional.append("Lab Results: " + ", ".join(f"{k}: {v}" for k, v in data["lab_results"].items()))
            return "\n".join(additional) if additional else ""
        except Exception as e:
            logger.error(f"Error formatting additional data: {str(e)}")
            return ""

    def _parse_recommendations(self, text: str) -> List[Dict]:
        """Parse model output into structured recommendations"""
        # Implementation depends on model output format
        pass

    def _parse_interactions(self, text: str) -> List[Dict]:
        """Parse model output into structured interaction data"""
        # Implementation depends on model output format
        pass

    def _get_severity_levels(self, text: str) -> Dict:
        """Extract interaction severity levels"""
        # Implementation depends on model output format
        pass

    def _parse_conditions(self, analyses: List[str]) -> List[Dict]:
        """Parse model output into structured condition data"""
        # Implementation depends on model output format
        pass

    def _get_recommendations(self, analyses: List[str]) -> List[Dict]:
        """Extract recommendations from analyses"""
        # Implementation depends on model output format
        pass 