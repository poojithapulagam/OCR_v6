"""Main OCR extraction pipeline."""
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from config import (
    RECIPIENT_DB_PATH,
    OCR_TEST_DATA_PATH,
    TEST_RESULTS_PATH,
    REVIEW_LOG_PATH
)
from src.utils.text_cleaner import clean_ocr_text
from src.extractors.llm_extractor import LLMExtractor
from src.utils.fuzzy_matcher import RecipientMatcher
from src.utils.output_handler import OutputHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRPipeline:
    """Main pipeline for processing OCR labels with Phi-3."""
    
    def __init__(
        self,
        recipient_db_path: str = str(RECIPIENT_DB_PATH),
        ollama_model: str = None
    ):
        """
        Initialize the pipeline with Phi-3 model.
        
        Args:
            recipient_db_path: Path to recipient database CSV
            ollama_model: Optional model name override (default: phi3)
        """
        self.llm_extractor = LLMExtractor(model=ollama_model) if ollama_model else LLMExtractor()
        self.matcher = RecipientMatcher(recipient_db_path)
        self.output_handler = OutputHandler()
        logger.info(f"Pipeline initialized with model: {self.llm_extractor.model}")
    
    def process_single_label(
        self, 
        ocr_text: str, 
        sample_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a single OCR label.
        
        Args:
            ocr_text: Raw OCR text from label
            sample_id: Optional sample identifier
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing sample: {sample_id or 'unknown'}")
        
        # Preprocess
        cleaned_text = clean_ocr_text(ocr_text)
        
        if not cleaned_text:
            logger.warning(f"Empty text after preprocessing for sample: {sample_id}")
            return {
                'sample_id': sample_id,
                'original_text': ocr_text,
                'cleaned_text': '',
                'extracted_pairs': [],
                'matches': [],
                'status': 'error',
                'error_message': 'Empty text after preprocessing'
            }
        
        # Extract pairs using Phi-3
        try:
            extracted_pairs = self.llm_extractor.extract_name_address_pairs(cleaned_text)
            logger.info(f"Extracted {len(extracted_pairs)} name-address pairs")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                'sample_id': sample_id,
                'original_text': ocr_text,
                'cleaned_text': cleaned_text,
                'extracted_pairs': [],
                'matches': [],
                'status': 'error',
                'error_message': str(e)
            }
        
        # Match each pair against database
        all_matches = []
        for pair in extracted_pairs:
            matches = self.matcher.match_pair(
                pair['input_name'],
                pair['input_address']
            )
            
            for match in matches:
                match['extracted_name'] = pair['input_name']
                match['extracted_address'] = pair['input_address']
                all_matches.append(match)
        
        # Determine status
        status = 'success'
        if not extracted_pairs:
            status = 'no_pairs_extracted'
        elif not all_matches:
            status = 'no_matches'
        elif any(m['combined_confidence'] < 80 for m in all_matches):
            status = 'low_confidence'
        elif any(m['is_ambiguous'] for m in all_matches):
            status = 'ambiguous'
        
        return {
            'sample_id': sample_id,
            'original_text': ocr_text,
            'cleaned_text': cleaned_text,
            'extracted_pairs': extracted_pairs,
            'matches': all_matches,
            'status': status,
            'error_message': None
        }
    
    def process_batch(
        self, 
        ocr_data_path: str = str(OCR_TEST_DATA_PATH)
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of OCR labels from CSV.
        
        Args:
            ocr_data_path: Path to CSV with OCR data
            
        Returns:
            List of processing results
        """
        logger.info(f"Loading OCR data from: {ocr_data_path}")
        
        try:
            df = pd.read_csv(ocr_data_path)
            
            # Check for required column (support both 'ocr_text' and 'raw_text')
            if 'ocr_text' in df.columns:
                text_column = 'ocr_text'
            elif 'raw_text' in df.columns:
                text_column = 'raw_text'
            else:
                raise ValueError("CSV must contain 'ocr_text' or 'raw_text' column")
            
            # Get sample_id if available
            has_sample_id = 'sample_id' in df.columns
            
            results = []
            total = len(df)
            
            for idx, row in df.iterrows():
                sample_id = row.get('sample_id', f"sample_{idx+1}") if has_sample_id else f"sample_{idx+1}"
                ocr_text = str(row[text_column])
                
                logger.info(f"Processing {idx+1}/{total}: {sample_id}")
                result = self.process_single_label(ocr_text, sample_id)
                results.append(result)
            
            logger.info(f"Processed {total} samples")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def run_full_pipeline(self):
        """Run the complete pipeline and save outputs."""
        logger.info("Starting full pipeline execution with Phi-3")
        
        # Process batch
        results = self.process_batch()
        
        # Save results
        self.output_handler.save_results(results, str(TEST_RESULTS_PATH))
        self.output_handler.save_review_log(results, str(REVIEW_LOG_PATH))
        
        # Print summary
        self._print_summary(results)
        
        logger.info("Pipeline execution completed")
        logger.info(f"Results saved to: {TEST_RESULTS_PATH}")
        logger.info(f"Review log saved to: {REVIEW_LOG_PATH}")
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics."""
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        no_pairs = sum(1 for r in results if r['status'] == 'no_pairs_extracted')
        no_matches = sum(1 for r in results if r['status'] == 'no_matches')
        low_confidence = sum(1 for r in results if r['status'] == 'low_confidence')
        ambiguous = sum(1 for r in results if r['status'] == 'ambiguous')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        total_pairs = sum(len(r['extracted_pairs']) for r in results)
        total_matches = sum(len(r['matches']) for r in results)
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY (Phi-3 Model)")
        print("="*60)
        print(f"Total samples processed: {total}")
        print(f"  - Successful: {successful}")
        print(f"  - No pairs extracted: {no_pairs}")
        print(f"  - No matches found: {no_matches}")
        print(f"  - Low confidence: {low_confidence}")
        print(f"  - Ambiguous: {ambiguous}")
        print(f"  - Errors: {errors}")
        print(f"\nTotal pairs extracted: {total_pairs}")
        print(f"Total matches found: {total_matches}")
        print("="*60 + "\n")


if __name__ == "__main__":
    pipeline = OCRPipeline()
    pipeline.run_full_pipeline()
