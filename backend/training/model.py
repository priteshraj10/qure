from transformers import AutoModelForCausalLM, PreTrainedModel
from typing import List, Dict, Union
import torch
import re
import logging

logger = logging.getLogger(__name__)

class MedicalLLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        self.medical_token_patterns = self._compile_medical_patterns()
        self.reference_pattern = re.compile(r'\[(\d+)\]|\(([^)]+)\)')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """Load a pretrained medical language model"""
        try:
            config = kwargs.pop('config', None)
            if config is None:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            
            model = cls(config)
            model.base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs
            )
            return model
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            raise

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """Forward pass of the model"""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_length: int = 1024,
        num_return_sequences: int = 1,
        **kwargs
    ):
        """Generate medical text with enhanced medical context awareness"""
        try:
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.base_model.config.pad_token_id,
                eos_token_id=self.base_model.config.eos_token_id,
                **kwargs
            )
            
            # Apply medical post-processing
            if num_return_sequences == 1:
                outputs = self._post_process_medical_text(outputs)
            else:
                outputs = [self._post_process_medical_text(output) for output in outputs]
            
            return outputs
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise

    def get_confidence(self, outputs: Union[torch.Tensor, List[torch.Tensor]]) -> Union[float, List[float]]:
        """Calculate confidence scores for generated outputs"""
        try:
            if isinstance(outputs, list):
                return [self._calculate_confidence(output) for output in outputs]
            return self._calculate_confidence(outputs)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def get_references(self, text: str) -> List[Dict[str, str]]:
        """Extract medical references from generated text"""
        try:
            references = []
            matches = self.reference_pattern.finditer(text)
            
            for match in matches:
                ref_num = match.group(1)
                ref_text = match.group(2)
                
                if ref_num:
                    # Numbered reference
                    references.append({
                        "type": "numbered",
                        "id": ref_num,
                        "text": self._find_reference_text(text, ref_num)
                    })
                elif ref_text:
                    # Inline reference
                    references.append({
                        "type": "inline",
                        "text": ref_text
                    })
            
            return references
        except Exception as e:
            logger.error(f"Error extracting references: {str(e)}")
            return []

    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """Calculate confidence score based on output probabilities"""
        try:
            # Get logits for the generated sequence
            with torch.no_grad():
                logits = self.base_model(output).logits
            
            # Calculate token probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get maximum probability for each token
            token_confidences = torch.max(probs, dim=-1).values
            
            # Calculate overall confidence
            confidence = float(torch.mean(token_confidences).item())
            
            # Apply medical term weighting
            medical_term_weight = self._get_medical_term_weight(output)
            weighted_confidence = confidence * medical_term_weight
            
            return min(weighted_confidence, 1.0)
        except Exception as e:
            logger.error(f"Error in confidence calculation: {str(e)}")
            return 0.0

    def _compile_medical_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for medical term recognition"""
        return {
            'drugs': re.compile(r'\b[A-Z][a-z]*(?:mab|nib|zib|lin|nin|tide|kine|fene|vec|cel|tune|mide|zide|olol|oxin|arin)\b'),
            'conditions': re.compile(r'\b(?:syndrome|disease|disorder|itis|osis|oma|emia|pathy)\b', re.IGNORECASE),
            'measurements': re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|L|mmol|μg|ng|IU|mEq|mmHg)\b'),
            'anatomical': re.compile(r'\b(?:anterior|posterior|lateral|medial|proximal|distal|superior|inferior)\b', re.IGNORECASE)
        }

    def _get_medical_term_weight(self, output: torch.Tensor) -> float:
        """Calculate weight based on medical term density"""
        try:
            text = self.base_model.tokenizer.decode(output, skip_special_tokens=True)
            total_terms = 0
            
            for pattern in self.medical_token_patterns.values():
                total_terms += len(pattern.findall(text))
            
            # Calculate density of medical terms
            words = text.split()
            if not words:
                return 1.0
                
            density = total_terms / len(words)
            
            # Apply sigmoid-like scaling
            weight = 1.0 + (0.5 * (2.0 / (1.0 + torch.exp(-density)) - 1.0))
            
            return float(weight)
        except Exception as e:
            logger.error(f"Error calculating medical term weight: {str(e)}")
            return 1.0

    def _post_process_medical_text(self, output: torch.Tensor) -> torch.Tensor:
        """Apply medical-specific post-processing to generated text"""
        try:
            text = self.base_model.tokenizer.decode(output, skip_special_tokens=True)
            
            # Apply medical term standardization
            for pattern_name, pattern in self.medical_token_patterns.items():
                text = self._standardize_terms(text, pattern, pattern_name)
            
            # Re-encode processed text
            processed = self.base_model.tokenizer.encode(
                text,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            return processed.squeeze()
        except Exception as e:
            logger.error(f"Error in medical text post-processing: {str(e)}")
            return output

    def _standardize_terms(self, text: str, pattern: re.Pattern, term_type: str) -> str:
        """Standardize medical terminology"""
        try:
            def replace_term(match):
                term = match.group(0)
                # Add specific standardization rules based on term_type
                if term_type == 'drugs':
                    return term.capitalize()
                elif term_type == 'measurements':
                    return term.lower()
                return term
            
            return pattern.sub(replace_term, text)
        except Exception as e:
            logger.error(f"Error standardizing {term_type} terms: {str(e)}")
            return text

    def _find_reference_text(self, text: str, ref_num: str) -> str:
        """Find the full text of a numbered reference"""
        try:
            # Look for reference list at the end of the text
            ref_list_match = re.search(
                rf"{ref_num}\.\s+([^[]+?)(?=\d+\.|$)",
                text.split("References:")[-1]
            )
            
            if ref_list_match:
                return ref_list_match.group(1).strip()
            return ""
        except Exception as e:
            logger.error(f"Error finding reference text: {str(e)}")
            return "" 