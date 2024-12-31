from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 1.5
    min_p: float = 0.1
    cors_origins: list = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings() 