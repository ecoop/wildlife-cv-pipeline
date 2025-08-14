import os
from dotenv import load_dotenv

# Load .env file if it exists (for local development)
load_dotenv()

class GetiConfig:
    def __init__(self):
        # Try .env first, then environment variables (like your .zshrc)
        self.api_token = os.getenv('GETI_API_TOKEN')
        self.server_url = os.getenv('GETI_SERVER_URL')
        
        if not self.api_token:
            raise ValueError("GETI_API_TOKEN not found in environment")
        
    @property
    def headers(self):
        return {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
