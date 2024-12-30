from typing import Dict, List, Optional, Union
import torch
from .model import MedicalLLM
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.version = "2.0.0"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model = MedicalLLM.from_pretrained("qure-ai/medical-model-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("qure-ai/medical-model-v2")
        
        # Move model to appropriate device
        self.model.to(self.device)
        self.model.eval()

    def process_query(
        self,
        query: str,
        confidence_threshold: float = 0.85,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process a medical query and return response with confidence scores"""
        try:
            # Prepare input
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Add context if provided
            if context:
                inputs["context"] = context

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get confidence scores
            confidence = self.model.get_confidence(outputs)

            # Filter based on confidence threshold
            if confidence < confidence_threshold:
                return {
                    "status": "low_confidence",
                    "message": "Response confidence below threshold",
                    "confidence": confidence
                }

            return {
                "status": "success",
                "response": response_text,
                "confidence": confidence,
                "references": self.model.get_references(response_text)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def get_recommendations(
        self,
        condition: str,
        patient_data: Dict,
        include_references: bool = True
    ) -> Dict:
        """Generate medical recommendations based on condition and patient data"""
        try:
            # Prepare input prompt
            prompt = self._format_recommendation_prompt(condition, patient_data)
            
            # Generate recommendations
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.3  # Lower temperature for more focused recommendations
                )

            recommendations = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = {
                "status": "success",
                "recommendations": self._parse_recommendations(recommendations),
                "confidence": self.model.get_confidence(outputs)
            }

            if include_references:
                response["references"] = self.model.get_references(recommendations)

            return response

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def analyze_interactions(self, medications: List[str]) -> Dict:
        """Analyze potential drug interactions"""
        try:
            # Prepare interaction analysis prompt
            prompt = self._format_interaction_prompt(medications)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=1
                )

            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "status": "success",
                "interactions": self._parse_interactions(analysis),
                "severity_levels": self._get_severity_levels(analysis),
                "references": self.model.get_references(analysis)
            }

        except Exception as e:
            logger.error(f"Error analyzing interactions: {str(e)}")
            raise

    def analyze_symptoms(
        self,
        symptoms: List[str],
        patient_data: Optional[Dict] = None
    ) -> Dict:
        """Analyze symptoms and suggest possible conditions"""
        try:
            # Prepare symptom analysis prompt
            prompt = self._format_symptom_prompt(symptoms, patient_data)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    num_return_sequences=3  # Get multiple possible conditions
                )

            analysis = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            return {
                "status": "success",
                "possible_conditions": self._parse_conditions(analysis),
                "confidence_scores": self.model.get_confidence(outputs),
                "recommendations": self._get_recommendations(analysis),
                "references": self.model.get_references(str(analysis))
            }

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {str(e)}")
            raise

    def _format_recommendation_prompt(self, condition: str, patient_data: Dict) -> str:
        """Format prompt for recommendation generation"""
        return f"""Provide evidence-based recommendations for {condition}.
Patient Information:
- Age: {patient_data['age']}
- Existing Conditions: {', '.join(patient_data['conditions'])}
- Current Medications: {', '.join(patient_data['medications'])}
{self._format_additional_data(patient_data)}
"""

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
        """Format additional patient data if available"""
        additional = []
        if data.get("vitals"):
            additional.append("Vitals: " + ", ".join(f"{k}: {v}" for k, v in data["vitals"].items()))
        if data.get("lab_results"):
            additional.append("Lab Results: " + ", ".join(f"{k}: {v}" for k, v in data["lab_results"].items()))
        return "\n".join(additional) if additional else ""

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