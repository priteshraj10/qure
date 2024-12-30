from unsloth import FastLanguageModel
import torch

class MedicalLLM:
    def __init__(self, config):
        self.config = config
        self.model, self.tokenizer = self._initialize_model()
        self.model = self._apply_lora()

    def _initialize_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            dtype=None
        )
        return model, tokenizer

    def _apply_lora(self):
        return FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407
        )

    def format_prompt(self, instruction, input_text, response=""):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}""" 