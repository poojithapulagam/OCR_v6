"""Fast extraction script - Phi model extraction without emoji output."""
import sys
import os
import pandas as pd
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.text_cleaner import clean_ocr_text
from src.extractors.llm_extractor import LLMExtractor
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Path to the user's OCR file
OCR_RAW_LABELS_PATH = Path(__file__).parent.parent / "ocr_raw_labels.csv"


def main():
    """Run extraction pipeline with Phi model."""
    print("=" * 80)
    print("OCR EXTRACTION - Phi Model")
    print("=" * 80)
    print(f"Input file: {OCR_RAW_LABELS_PATH}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print("=" * 80 + "\n")
    
    # Check if input file exists
    if not OCR_RAW_LABELS_PATH.exists():
        print(f"ERROR: Input file not found: {OCR_RAW_LABELS_PATH}")
        sys.exit(1)
    
    # Initialize LLM extractor
    print("Initializing Phi extractor...")
    extractor = LLMExtractor()
    
    # Load CSV
    print(f"Loading OCR data from: {OCR_RAW_LABELS_PATH}\n")
    df = pd.read_csv(OCR_RAW_LABELS_PATH)
    
    # Determine text column
    if 'raw_text' in df.columns:
        text_column = 'raw_text'
    elif 'ocr_text' in df.columns:
        text_column = 'ocr_text'
    else:
        print("ERROR: CSV must contain 'raw_text' or 'ocr_text' column")
        sys.exit(1)
    
    has_sample_id = 'sample_id' in df.columns
    total = len(df)
    total_time = 0
    
    print("=" * 80)
    print("EXTRACTING NAME & ADDRESS (Using Phi Model)")
    print("=" * 80)
    
    for idx, row in df.iterrows():
        sample_id = row.get('sample_id', f"sample_{idx+1}") if has_sample_id else f"sample_{idx+1}"
        ocr_text = str(row[text_column])
        
        # Preprocess
        cleaned_text = clean_ocr_text(ocr_text)
        
        # Extract with timing
        start_time = time.time()
        try:
            extracted_pairs = extractor.extract_name_address_pairs(cleaned_text)
            elapsed = time.time() - start_time
            total_time += elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            total_time += elapsed
            extracted_pairs = []
            error_msg = str(e)
        
        # Print results immediately
        print(f"\n[{idx+1}/{total}] {sample_id} | Time: {elapsed:.2f}s")
        print("-" * 80)
        print(f"Raw Text: {ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}")
        print()
        
        if extracted_pairs:
            print(f"EXTRACTED {len(extracted_pairs)} PAIR(S):")
            for i, pair in enumerate(extracted_pairs, 1):
                name = pair.get('input_name', 'N/A')
                address = pair.get('input_address', 'N/A')
                tracking = pair.get('tracking_number', None)
                print(f"\n   Pair {i}:")
                print(f"   NAME:     {name}")
                print(f"   ADDRESS:  {address}")
                if tracking:
                    print(f"   TRACKING: {tracking}")
        else:
            print("No name-address pairs extracted")
            if 'error_msg' in locals():
                print(f"   Error: {error_msg}")
        
        print("=" * 80)
        sys.stdout.flush()
    
    # Summary
    avg_time = total_time / total if total > 0 else 0
    print(f"\n{'='*80}")
    print(f"SUMMARY: Processed {total} samples with Phi Model")
    print(f"Total time: {total_time:.2f}s | Average: {avg_time:.2f}s per sample")
    if avg_time < 10.0:
        print(f"Performance: Average response time is {avg_time:.2f}s")
    else:
        print(f"Notice: Average response time is {avg_time:.2f}s")
        print(f"   Tip: This is normal for CPU-based inference")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
