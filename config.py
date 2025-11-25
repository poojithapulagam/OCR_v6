"""Configuration module for OCR Label Processing Pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Ollama Configuration
# You can configure these in two ways:
# 1. Create a .env file with: OLLAMA_BASE_URL=http://localhost:11434 and OLLAMA_MODEL=phi3
# 2. Or modify the default values below
#
# RECOMMENDED MODELS FOR MOBILE/iOS:
# - "phi3"      : BEST - Fastest + most accurate, optimized for mobile (2.3GB)
# - "phi"       : GOOD - Fast, lightweight (1.7GB) - current default
# - "gemma:2b"  : Alternative - Google's mobile model (1.7GB)
# - "tinyllama" : FASTEST - Ultra-lightweight but lower accuracy (0.6GB)
#
# NOTE: For sub-2-second responses on CPU/mobile, use phi3 or phi
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")  # Using Phi-2 for faster speed

# Matching Configuration
FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "70"))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# File paths
RECIPIENT_DB_PATH = DATA_DIR / "recipient_database.csv"
OCR_TEST_DATA_PATH = DATA_DIR / "ocr_test_data.csv"
TEST_RESULTS_PATH = OUTPUT_DIR / "test_results.csv"
REVIEW_LOG_PATH = OUTPUT_DIR / "review_log.csv"
