"""LLM-based OCR extraction using Ollama with Phi-3 model."""
import json
import logging
import re
from typing import List, Dict, Any, Optional
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class LLMExtractor:
    """LLM-based extractor using Ollama API with Phi model."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        """
        Initialize LLM extractor with Phi model.
        
        Args:
            base_url: Ollama API base URL
            model: Model name (default: phi)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        logger.info(f"Initialized LLM Extractor with model: {self.model}")
    
    def extract_name_address_pairs(self, ocr_text: str, use_fallback: bool = True) -> List[Dict[str, str]]:
        """
        Extract name-address pairs from OCR text using Phi.
        Falls back to rule-based extraction if LLM fails.
        Args:
            ocr_text: Raw OCR text from shipping label
            use_fallback: If True, use rule-based extraction on failure
            
        Returns:
            List of dictionaries with 'input_name' and 'input_address' keys
        """
        import time
        start_time = time.time()
        
        prompt = self._build_extraction_prompt(ocr_text)
        
        try:
            response = self._call_ollama(prompt)
            extracted_pairs = self._parse_response(response)
            elapsed = time.time() - start_time
            logger.info(f"Extraction completed in {elapsed:.2f} seconds")
            
            # Validate extracted pairs
            if extracted_pairs and len(extracted_pairs) > 0:
                return extracted_pairs
            else:
                logger.warning("LLM returned no pairs, trying fallback extraction")
                if use_fallback:
                    return self._fallback_rule_based(ocr_text)
                return []
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            logger.warning(f"Ollama request timed out (took {elapsed:.2f}s), using fallback extraction")
            if use_fallback:
                fallback_result = self._fallback_rule_based(ocr_text)
                if fallback_result:
                    logger.info(f"Fallback extraction found {len(fallback_result)} pair(s)")
                return fallback_result
            return []
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting from OCR text (took {elapsed:.2f}s): {e}")
            if use_fallback:
                logger.info("Attempting rule-based fallback extraction")
                return self._fallback_rule_based(ocr_text)
            return []
    
    def _fallback_rule_based(self, ocr_text: str) -> List[Dict[str, str]]:
        """Use rule-based extraction as fallback."""
        try:
            from src.extractors.rule_extractor import extract_name_address_rule_based
            return extract_name_address_rule_based(ocr_text)
        except ImportError:
            logger.warning("Rule-based extractor not available")
            return []
    
    def _build_extraction_prompt(self, ocr_text: str) -> str:
        """
        Balanced prompt - concise but accurate.
        """
        # Optimized prompt: short but clear
        prompt = f"""Extract recipient info from shipping label OCR.

Need:
1. Full name (first + last)
2. Complete address (street, city, state, zip)
3. Tracking number (if any)

OCR text:
{ocr_text[:300]}

Return JSON:
{{"name": "", "address": "", "tracking": ""}}"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to Phi-3
            
        Returns:
            Response text from the LLM
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 120,  # Balanced for speed + completeness
                "temperature": 0.2,  # Lower for accuracy
                "top_k": 30,
                "top_p": 0.9,
                "repeat_penalty": 1.1,  # Prevent repetition
                "num_ctx": 1024,  # Enough context
                "num_thread": 8,  # Use more CPU threads
            }
        }
        
        try:
            # First check if Ollama is reachable
            try:
                health_check = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if health_check.status_code != 200:
                    raise ConnectionError(
                        f"Ollama service not accessible at {self.base_url}. "
                        f"Status: {health_check.status_code}. "
                        f"Make sure Ollama is running: 'ollama serve' or check if it's installed."
                    )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Please ensure Ollama is running. "
                    f"Start it with: 'ollama serve' or check your OLLAMA_BASE_URL in config.py or .env file."
                )
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=8  # Balanced timeout
            )
            
            if response.status_code == 404:
                # Check if model exists
                try:
                    models_resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                    if models_resp.status_code == 200:
                        models = models_resp.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        raise ValueError(
                            f"Model '{self.model}' not found. "
                            f"Available models: {', '.join(model_names) if model_names else 'none'}. "
                            f"Download the model with: 'ollama pull {self.model}' or update OLLAMA_MODEL in config.py/.env"
                        )
                except:
                    pass
                raise ValueError(
                    f"Ollama API endpoint not found (404). "
                    f"Model: {self.model}, URL: {self.api_url}. "
                    f"Check if model exists: 'ollama list'"
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Ollama returns the response in 'response' field
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def _parse_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        Parse LLM response into structured name-address pairs.
        Improved parsing to handle various response formats.
        
        Args:
            response_text: Raw response from Ollama
            
        Returns:
            List of dictionaries with extracted pairs
        """
        # Clean the response text
        response_text = response_text.strip()
        
        # Try to extract JSON from the response
        # Sometimes LLMs wrap JSON in markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end > start:
                response_text = response_text[start:end].strip()
            else:
                response_text = response_text[start:].strip()
        
        try:
            data = json.loads(response_text)
            
            # Handle both list and dict responses
            if isinstance(data, list):
                pairs = data
            elif isinstance(data, dict):
                # If it's a dict, try to find a list field
                if "pairs" in data:
                    pairs = data["pairs"]
                elif "results" in data:
                    pairs = data["results"]
                elif "input_name" in data or "input_address" in data:
                    # If it's a single pair as dict
                    pairs = [data]
                else:
                    # Try to find any list-like structure
                    pairs = []
                    for key, value in data.items():
                        if isinstance(value, list):
                            pairs = value
                            break
                    if not pairs:
                        pairs = [data]
            else:
                logger.warning(f"Unexpected response format: {type(data)}")
                return []
            
            # Validate and normalize pairs
            normalized_pairs = []
            for pair in pairs:
                if isinstance(pair, dict):
                    # Get name - handle various field names
                    name = (pair.get("input_name") or pair.get("name") or 
                           pair.get("recipient_name") or pair.get("full_name") or "")
                    
                    # Get address - handle various field names
                    address = (pair.get("input_address") or pair.get("address") or 
                              pair.get("recipient_address") or pair.get("full_address") or "")
                    
                    # Get tracking number
                    tracking = (pair.get("tracking") or pair.get("tracking_number") or 
                               pair.get("tracking_id") or None)
                    
                    # Clean up - handle if address is a list
                    if isinstance(address, list):
                        address = ", ".join(str(x) for x in address if x)
                    
                    name = str(name).strip() if name else ""
                    address = str(address).strip() if address else ""
                    tracking = str(tracking).strip() if tracking and str(tracking).lower() != "null" else None
                    
                    # Validate name - be more lenient for speed
                    name_words = name.split()
                    if len(name_words) < 2:
                        logger.warning(f"Short name '{name}', might be incomplete")
                        # Continue anyway - don't skip
                    
                    # Filter out place names that were incorrectly extracted as names
                    place_names = ['pompano', 'pamgiano', 'pumgiana', 'beach', 'roseville', 'williston', 
                                  'tampa', 'norcross', 'chicago', 'webster', 'bethesda', 'folsom',
                                  'cleveland', 'harvey', 'euless', 'mishawaka', 'fort', 'collins']
                    if any(place in name.lower() for place in place_names):
                        logger.warning(f"Filtered out place name as person name: {name}")
                        continue
                    
                    # Filter out common non-name words
                    non_name_words = ['ship', 'priority', 'mail', 'ground', 'ups', 'fedex', 'usps', 
                                     'tracking', 'postage', 'fees', 'paid', 'sender', 'shipper',
                                     'notifii', 'llc', 'corporation', 'company', 'apartments']
                    if any(word in name.lower() for word in non_name_words):
                        logger.warning(f"Filtered out non-name word: {name}")
                        continue
                    
                    normalized = {
                        "input_name": name,
                        "input_address": address,
                        "tracking_number": tracking
                    }
                    
                    # Add if we have at least name or address
                    if (name and len(name) > 2) or (address and len(address) > 5):
                        normalized_pairs.append(normalized)
                    else:
                        logger.warning(f"Skipping pair: {normalized}")
            
            return normalized_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            # Try to extract name and address using regex as fallback
            return self._fallback_extraction(response_text)
    
    def _fallback_extraction(self, text: str) -> List[Dict[str, str]]:
        """
        Fallback extraction using regex patterns when JSON parsing fails.
        
        Args:
            text: Response text to parse
            
        Returns:
            List of extracted pairs
        """
        pairs = []
        # Try to find name and address patterns in the text
        # This is a simple fallback - not as accurate as LLM extraction
        name_pattern = r'(?:name|Name|NAME)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        address_pattern = r'(\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|way|court|ct)[^,]*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)'
        
        name_match = re.search(name_pattern, text)
        address_match = re.search(address_pattern, text, re.IGNORECASE)
        
        if name_match and address_match:
            pairs.append({
                "input_name": name_match.group(1).strip(),
                "input_address": address_match.group(1).strip()
            })
        
        return pairs
