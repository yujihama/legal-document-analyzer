
import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'azure_openai_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'azure_openai_api_key': os.getenv('AZURE_OPENAI_API_KEY'),
        'azure_openai_api_version': os.getenv('AZURE_OPENAI_API_VERSION'),
        'azure_openai_gpt_deployment': os.getenv('AZURE_OPENAI_GPT_DEPLOYMENT'),
        'azure_openai_embedding_deployment': os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
    }
