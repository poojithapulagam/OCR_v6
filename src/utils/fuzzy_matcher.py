"""Fuzzy matching module for name-address pairs against recipient database."""
import logging
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz, process
import pandas as pd
from config import FUZZY_MATCH_THRESHOLD

logger = logging.getLogger(__name__)


class RecipientMatcher:
    """Fuzzy matcher for matching extracted pairs against recipient database."""
    
    def __init__(self, recipient_db_path: str, threshold: int = FUZZY_MATCH_THRESHOLD):
        """
        Initialize the matcher with recipient database.
        
        Args:
            recipient_db_path: Path to recipient database CSV
            threshold: Minimum similarity score (0-100) for accepting a match
        """
        self.threshold = threshold
        self.recipient_db = self._load_recipient_db(recipient_db_path)
        logger.info(f"Loaded {len(self.recipient_db)} recipients from database")
    
    def _load_recipient_db(self, db_path: str) -> pd.DataFrame:
        """Load recipient database from CSV."""
        try:
            df = pd.read_csv(db_path)
            required_columns = ['recipient_id', 'first_name', 'last_name', 'address']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return df
        except Exception as e:
            logger.error(f"Failed to load recipient database: {e}")
            raise
    
    def match_pair(
        self, 
        input_name: str, 
        input_address: str
    ) -> List[Dict[str, Any]]:
        """
        Match a name-address pair against the recipient database.
        
        Args:
            input_name: Extracted name from OCR
            input_address: Extracted address from OCR
            
        Returns:
            List of match dictionaries with recipient info and confidence scores
        """
        matches = []
        
        # Create full name variations from database
        name_variations = self._create_name_variations()
        
        # Match name
        name_matches = self._fuzzy_match_name(input_name, name_variations)
        
        # For each name match, check address similarity
        for name_match in name_matches:
            recipient_id = name_match['recipient_id']
            recipient = self.recipient_db[
                self.recipient_db['recipient_id'] == recipient_id
            ].iloc[0]
            
            # Match address
            address_score = self._fuzzy_match_address(
                input_address, 
                recipient
            )
            
            # Calculate combined confidence
            name_score = name_match['score']
            combined_score = (name_score * 0.6 + address_score * 0.4)
            
            if combined_score >= self.threshold:
                match = {
                    'recipient_id': recipient_id,
                    'first_name': recipient.get('first_name', ''),
                    'last_name': recipient.get('last_name', ''),
                    'preferred_first_name': recipient.get('preferred_first_name', ''),
                    'preferred_full_name': recipient.get('preferred_full_name', ''),
                    'address': recipient.get('address', ''),
                    'city': recipient.get('city', ''),
                    'state': recipient.get('state', ''),
                    'zip_code': recipient.get('zip_code', ''),
                    'unit_number': recipient.get('unit_number', ''),
                    'name_match_score': name_score,
                    'address_match_score': address_score,
                    'combined_confidence': round(combined_score, 2),
                    'is_ambiguous': len(name_matches) > 1
                }
                matches.append(match)
        
        # Sort by combined confidence (highest first)
        matches.sort(key=lambda x: x['combined_confidence'], reverse=True)
        
        return matches
    
    def _create_name_variations(self) -> Dict[str, Dict[str, Any]]:
        """
        Create name variations from recipient database for matching.
        
        Returns:
            Dictionary mapping name strings to recipient info
        """
        variations = {}
        
        for _, row in self.recipient_db.iterrows():
            recipient_id = row['recipient_id']
            first_name = str(row.get('first_name', '')).strip()
            last_name = str(row.get('last_name', '')).strip()
            preferred_first = str(row.get('preferred_first_name', '')).strip()
            preferred_full = str(row.get('preferred_full_name', '')).strip()
            
            # Create various name combinations
            if first_name and last_name:
                # Full name: "First Last"
                full_name = f"{first_name} {last_name}"
                variations[full_name.lower()] = {
                    'recipient_id': recipient_id,
                    'name': full_name
                }
                
                # Last, First
                last_first = f"{last_name}, {first_name}"
                variations[last_first.lower()] = {
                    'recipient_id': recipient_id,
                    'name': last_first
                }
            
            # Preferred first name variations
            if preferred_first:
                if last_name:
                    pref_full = f"{preferred_first} {last_name}"
                    variations[pref_full.lower()] = {
                        'recipient_id': recipient_id,
                        'name': pref_full
                    }
            
            # Preferred full name
            if preferred_full:
                variations[preferred_full.lower()] = {
                    'recipient_id': recipient_id,
                    'name': preferred_full
                }
        
        return variations
    
    def _fuzzy_match_name(
        self, 
        input_name: str, 
        name_variations: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy match input name against name variations.
        
        Args:
            input_name: Name to match
            name_variations: Dictionary of name variations
            
        Returns:
            List of matches with scores above threshold
        """
        if not input_name or not name_variations:
            return []
        
        input_name_lower = input_name.lower().strip()
        matches = []
        
        # Use rapidfuzz to find best matches
        for name_key, name_info in name_variations.items():
            score = fuzz.ratio(input_name_lower, name_key)
            if score >= self.threshold:
                matches.append({
                    'recipient_id': name_info['recipient_id'],
                    'score': score
                })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches
    
    def _fuzzy_match_address(
        self, 
        input_address: str, 
        recipient: pd.Series
    ) -> float:
        """
        Fuzzy match input address against recipient address.
        
        Args:
            input_address: Address to match
            recipient: Recipient database row
            
        Returns:
            Similarity score (0-100)
        """
        if not input_address:
            return 0.0
        
        # Build full address from recipient
        address_parts = []
        if recipient.get('address'):
            address_parts.append(str(recipient['address']))
        if recipient.get('unit_number'):
            address_parts.append(str(recipient['unit_number']))
        if recipient.get('city'):
            address_parts.append(str(recipient['city']))
        if recipient.get('state'):
            address_parts.append(str(recipient['state']))
        if recipient.get('zip_code'):
            address_parts.append(str(recipient['zip_code']))
        
        full_address = ", ".join(address_parts).lower().strip()
        input_address_lower = input_address.lower().strip()
        
        # Use token sort ratio for better address matching
        # (handles word order differences)
        score = fuzz.token_sort_ratio(input_address_lower, full_address)
        
        return float(score)
