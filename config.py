#CONFIGURATION AND ENVIRONMENT VARIABLES

import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file or as an environment variable.")