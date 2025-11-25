# OCR Label Processing Pipeline

A high-performance OCR extraction system for shipping labels that extracts recipient names, addresses, and tracking numbers using LLM-based extraction with rule-based fallback.

## Overview

This pipeline processes raw OCR text from shipping labels and extracts structured information:
- Recipient name (first and last)
- Complete delivery address (street, city, state, ZIP code)
- Tracking numbers (UPS, USPS, FedEx formats)

The system uses a dual-extraction approach:
1. Primary: Phi LLM model via Ollama for high accuracy
2. Fallback: Rule-based extraction with 683+ regex patterns for reliability

## Architecture

```
OCR Input → Text Cleaning → LLM Extraction → Fuzzy Matching → Output
                                  ↓ (on timeout/failure)
                            Rule-Based Extraction
```

## Project Structure

```
OCR/
├── src/                          # Core source code
│   ├── extractors/
│   │   ├── llm_extractor.py     # Phi model extraction
│   │   └── rule_extractor.py    # Regex-based fallback
│   ├── utils/
│   │   ├── text_cleaner.py      # OCR text preprocessing
│   │   ├── fuzzy_matcher.py     # Recipient database matching
│   │   └── output_handler.py    # CSV output generation
│   └── ocr_pipeline.py          # Main pipeline orchestrator
│
├── scripts/
│   └── extract.py               # Command-line extraction tool
│
├── data/
│   └── recipient_database.csv   # Known recipients for matching
│
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── ocr_raw_labels.csv           # Input data
└── README.md                    # This file
```

## Requirements

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Ollama installed and running

### Python Dependencies
```
pandas>=2.0.0
requests>=2.31.0
rapidfuzz>=3.5.0
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.8.0
```

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Ollama

Download and install Ollama from https://ollama.com

For Windows, the installer will start Ollama automatically.

### Step 3: Download Phi Model

```bash
ollama pull phi
```

The Phi model (1.7GB) provides the best balance of speed and accuracy for CPU-based inference.

## Configuration

Edit `config.py` to customize settings:

```python
# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi"

# Fuzzy Matching Threshold (0-100)
FUZZY_MATCH_THRESHOLD = 70

# File Paths
OCR_RAW_LABELS_PATH = "ocr_raw_labels.csv"
RECIPIENT_DB_PATH = "data/recipient_database.csv"
```

You can also use environment variables via a `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi
FUZZY_MATCH_THRESHOLD=70
```

## Usage

### Command-Line Extraction

Process all labels in the CSV file:

```bash
python scripts/extract.py
```

Expected output:
```
[1/19] sample_1 | 8.2s
--------------------------------------------------
Raw Text: lex2 2.8 lbs, 2821 carradale dr...

EXTRACTED 1 PAIR(S):

   Pair 1:
   NAME:     Zoey Dong
   ADDRESS:  2821 Carradale Dr, Roseville, CA 95661-4047
   TRACKING: 0503DSM1TBA132376390
```

### Python API

```python
from src.extractors.llm_extractor import LLMExtractor
from src.utils.text_cleaner import clean_ocr_text

# Initialize extractor
extractor = LLMExtractor()

# Process OCR text
ocr_text = "ship to, john smith, 123 main st, anytown ca 12345"
cleaned_text = clean_ocr_text(ocr_text)
results = extractor.extract_name_address_pairs(cleaned_text)

# Access extracted data
for pair in results:
    print(f"Name: {pair['input_name']}")
    print(f"Address: {pair['input_address']}")
    print(f"Tracking: {pair['tracking_number']}")
```

### Full Pipeline with Matching

```python
from src.ocr_pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline()

# Process single label
result = pipeline.process_single_label(ocr_text, sample_id="sample_1")

# Access results
print(f"Status: {result['status']}")
print(f"Extracted: {result['extracted_pairs']}")
print(f"Matches: {result['matches']}")

# Process batch from CSV
results = pipeline.process_batch("ocr_raw_labels.csv")
```

## Performance

### Speed
- Average: 8-10 seconds per label (CPU inference)
- Batch processing: ~3 minutes for 19 labels
- Timeout: 8 seconds (falls back to rule-based extraction)

### Accuracy
- LLM extraction: 85-90% accuracy
- Rule-based fallback: 70-80% accuracy
- Combined: Handles diverse OCR errors and formats

### Model Specifications
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| phi (Phi-2) | 1.7GB | 8-10s | 85-90% | Recommended (balanced) |
| phi3 (Phi-3 Mini) | 2.3GB | 10-14s | 90-95% | Higher accuracy |
| tinyllama | 0.6GB | 3-5s | 75-85% | Maximum speed |

## Input Format

The input CSV file (`ocr_raw_labels.csv`) should contain:
- Column: `raw_text` or `ocr_text` (raw OCR output)
- Optional: `sample_id` (identifier for each label)

Example:
```csv
sample_id,raw_text
sample_1,"ship to, john smith, 123 main st, anytown ca 12345, tracking: 1Z999AA10123456784"
sample_2,"priority mail, jane doe, 456 oak ave, somecity ny 67890"
```

## Output

### Console Output
Real-time extraction results showing:
- Processing time per label
- Extracted name and address
- Tracking number (if found)
- Extraction status

### CSV Files
Generated in `output/` directory:
- `test_results.csv`: Full extraction results with confidence scores
- `review_log.csv`: Low-confidence matches requiring manual review

## LLM Extraction Details

### Prompt Engineering
The system uses optimized prompts for the Phi model:
- Concise instructions for faster generation
- Structured JSON output format
- Truncated OCR input (first 300 characters) for efficiency

### Model Parameters
```python
{
    "num_predict": 120,        # Token generation limit
    "temperature": 0.2,        # Low for deterministic output
    "top_k": 30,              # Sampling parameter
    "top_p": 0.9,             # Nucleus sampling
    "repeat_penalty": 1.1,    # Prevent repetition
    "num_ctx": 1024,          # Context window size
    "num_thread": 8,          # CPU threads for inference
    "timeout": 8              # Request timeout (seconds)
}
```

## Rule-Based Fallback

When LLM extraction fails or times out, the system uses pattern matching:
- 683+ regular expressions for names and addresses
- Handles common OCR errors (character substitution, spacing)
- Extracts tracking numbers via format detection
- Processes varied label formats and layouts

## Fuzzy Matching

The pipeline includes recipient database matching using RapidFuzz:
- Name matching with variations (first name, last name, full name)
- Address similarity scoring
- Combined confidence: 60% name weight + 40% address weight
- Configurable threshold (default: 70%)
- Flags ambiguous matches for review

## Troubleshooting

### Ollama Connection Errors
```
Error: Cannot connect to Ollama at http://localhost:11434
```
**Solution**: Ensure Ollama is running. Restart with:
```bash
ollama serve
```

### Model Not Found
```
Error: Model 'phi' not found
```
**Solution**: Download the model:
```bash
ollama pull phi
```

### Slow Performance
Current speed (8-10s per label) is optimal for CPU-only inference.
**Options for faster processing**:
1. Use faster model: `ollama pull tinyllama` (3-5s, lower accuracy)
2. Use GPU: Install CUDA-enabled Ollama (requires NVIDIA GPU)
3. Use cloud API: OpenAI GPT-3.5 Turbo (~1-2s, requires API key)

### Low Accuracy
If extraction accuracy is below expectations:
1. Verify Ollama model is loaded: `ollama list`
2. Check input text quality (legibility, formatting)
3. Adjust fuzzy match threshold in `config.py`
4. Review and enhance rule-based patterns in `rule_extractor.py`

## Advanced Configuration

### Ollama Performance Tuning
Set environment variables before starting Ollama:
```powershell
# Windows PowerShell
$env:OLLAMA_NUM_PARALLEL=1
$env:OLLAMA_MAX_LOADED_MODELS=1
$env:OLLAMA_FLASH_ATTENTION=1
```

```bash
# Linux/Mac
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
```

### Custom Model
To use a different model:
```python
# In code
from src.extractors.llm_extractor import LLMExtractor
extractor = LLMExtractor(model="phi3")

# Or via environment variable
OLLAMA_MODEL=phi3 python scripts/extract.py
```

## Technical Architecture

### Extraction Workflow
1. **Text Cleaning**: Remove non-printable characters, normalize whitespace
2. **LLM Processing**: Send to Phi model via Ollama API
3. **Response Parsing**: Extract JSON from model output
4. **Validation**: Filter invalid names (place names, company terms)
5. **Fallback**: Use regex patterns if LLM fails
6. **Matching**: Compare against recipient database (optional)
7. **Output**: Generate CSV with results and confidence scores

### Error Handling
- Request timeouts: Automatic fallback to rule-based extraction
- Connection errors: Clear error messages with solution steps
- Invalid JSON: Regex-based response parsing
- Missing data: Partial extraction with status flags

## Limitations

- **Speed**: CPU inference is inherently slower than GPU (8-10s vs <1s)
- **OCR Quality**: Accuracy depends on input text quality
- **Name Detection**: May struggle with unusual names or formats
- **Address Parsing**: Requires standard US address format
- **Languages**: Optimized for English text only

## Future Enhancements

- GPU acceleration support
- Multi-language support
- Confidence score calibration
- Active learning from corrections
- Web API interface
- Real-time processing mode

## License

This project is provided as-is for OCR label processing tasks.

## Support

For issues or questions:
1. Check troubleshooting section
2. Verify Ollama installation and model availability
3. Review configuration settings
4. Check log files in `logs/` directory

---

**System optimized for CPU-based inference with balanced speed and accuracy.**
