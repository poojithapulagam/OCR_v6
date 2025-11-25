"""Output handler for saving pipeline results."""
import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class OutputHandler:
    """Handler for saving pipeline outputs."""
    
    def save_results(
        self, 
        results: List[Dict[str, Any]], 
        output_path: str
    ):
        """
        Save processing results to CSV.
        
        Args:
            results: List of processing result dictionaries
            output_path: Path to save CSV
        """
        rows = []
        
        for result in results:
            sample_id = result.get('sample_id', 'unknown')
            status = result.get('status', 'unknown')
            extracted_pairs = result.get('extracted_pairs', [])
            matches = result.get('matches', [])
            
            # If no pairs extracted, create one row
            if not extracted_pairs:
                rows.append({
                    'sample_id': sample_id,
                    'status': status,
                    'extracted_name': '',
                    'extracted_address': '',
                    'recipient_id': '',
                    'matched_name': '',
                    'matched_address': '',
                    'name_match_score': '',
                    'address_match_score': '',
                    'combined_confidence': '',
                    'is_ambiguous': '',
                    'error_message': result.get('error_message', '')
                })
            else:
                # Create a row for each extracted pair
                for pair in extracted_pairs:
                    # Find matches for this pair
                    pair_matches = [
                        m for m in matches 
                        if m.get('extracted_name') == pair['input_name'] 
                        and m.get('extracted_address') == pair['input_address']
                    ]
                    
                    if pair_matches:
                        # One row per match
                        for match in pair_matches:
                            rows.append({
                                'sample_id': sample_id,
                                'status': status,
                                'extracted_name': pair['input_name'],
                                'extracted_address': pair['input_address'],
                                'recipient_id': match.get('recipient_id', ''),
                                'matched_name': f"{match.get('first_name', '')} {match.get('last_name', '')}".strip(),
                                'matched_address': match.get('address', ''),
                                'name_match_score': match.get('name_match_score', ''),
                                'address_match_score': match.get('address_match_score', ''),
                                'combined_confidence': match.get('combined_confidence', ''),
                                'is_ambiguous': match.get('is_ambiguous', False),
                                'error_message': ''
                            })
                    else:
                        # No match found for this pair
                        rows.append({
                            'sample_id': sample_id,
                            'status': status,
                            'extracted_name': pair['input_name'],
                            'extracted_address': pair['input_address'],
                            'recipient_id': '',
                            'matched_name': '',
                            'matched_address': '',
                            'name_match_score': '',
                            'address_match_score': '',
                            'combined_confidence': '',
                            'is_ambiguous': '',
                            'error_message': ''
                        })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(rows)} result rows to {output_path}")
    
    def save_review_log(
        self, 
        results: List[Dict[str, Any]], 
        log_path: str
    ):
        """
        Save review log for low-confidence and ambiguous results.
        
        Args:
            results: List of processing result dictionaries
            log_path: Path to save review log CSV
        """
        review_rows = []
        
        for result in results:
            sample_id = result.get('sample_id', 'unknown')
            status = result.get('status', 'unknown')
            matches = result.get('matches', [])
            
            # Flag for review if:
            # - Low confidence (status == 'low_confidence')
            # - Ambiguous matches
            # - No matches found
            # - Errors
            needs_review = (
                status in ['low_confidence', 'ambiguous', 'no_matches', 'error', 'no_pairs_extracted'] or
                any(m.get('combined_confidence', 100) < 80 for m in matches) or
                any(m.get('is_ambiguous', False) for m in matches)
            )
            
            if needs_review:
                extracted_pairs = result.get('extracted_pairs', [])
                
                if extracted_pairs:
                    for pair in extracted_pairs:
                        pair_matches = [
                            m for m in matches 
                            if m.get('extracted_name') == pair['input_name'] 
                            and m.get('extracted_address') == pair['input_address']
                        ]
                        
                        if pair_matches:
                            for match in pair_matches:
                                review_rows.append({
                                    'sample_id': sample_id,
                                    'status': status,
                                    'extracted_name': pair['input_name'],
                                    'extracted_address': pair['input_address'],
                                    'recipient_id': match.get('recipient_id', ''),
                                    'matched_name': f"{match.get('first_name', '')} {match.get('last_name', '')}".strip(),
                                    'combined_confidence': match.get('combined_confidence', ''),
                                    'is_ambiguous': match.get('is_ambiguous', False),
                                    'review_reason': self._get_review_reason(status, match),
                                    'original_text': result.get('original_text', '')[:200]  # Truncate for readability
                                })
                        else:
                            review_rows.append({
                                'sample_id': sample_id,
                                'status': status,
                                'extracted_name': pair['input_name'],
                                'extracted_address': pair['input_address'],
                                'recipient_id': '',
                                'matched_name': '',
                                'combined_confidence': '',
                                'is_ambiguous': '',
                                'review_reason': status,
                                'original_text': result.get('original_text', '')[:200]
                            })
                else:
                    review_rows.append({
                        'sample_id': sample_id,
                        'status': status,
                        'extracted_name': '',
                        'extracted_address': '',
                        'recipient_id': '',
                        'matched_name': '',
                        'combined_confidence': '',
                        'is_ambiguous': '',
                        'review_reason': status,
                        'original_text': result.get('original_text', '')[:200]
                    })
        
        if review_rows:
            df = pd.DataFrame(review_rows)
            df.to_csv(log_path, index=False)
            logger.info(f"Saved {len(review_rows)} review entries to {log_path}")
        else:
            # Create empty file with headers
            df = pd.DataFrame(columns=[
                'sample_id', 'status', 'extracted_name', 'extracted_address',
                'recipient_id', 'matched_name', 'combined_confidence',
                'is_ambiguous', 'review_reason', 'original_text'
            ])
            df.to_csv(log_path, index=False)
            logger.info(f"No items requiring review. Created empty log at {log_path}")
    
    def _get_review_reason(
        self, 
        status: str, 
        match: Dict[str, Any]
    ) -> str:
        """Get reason for review."""
        reasons = []
        
        if status == 'low_confidence':
            reasons.append('low_confidence')
        if match.get('is_ambiguous', False):
            reasons.append('ambiguous_match')
        if match.get('combined_confidence', 100) < 80:
            reasons.append('confidence_below_80')
        
        return ', '.join(reasons) if reasons else status
