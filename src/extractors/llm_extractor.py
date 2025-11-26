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
        """Build the prompt for the LLM."""
        return f"""Instruction: Extract the recipient name, full address, and tracking number from the shipping label text below.
Return ONLY a JSON object with keys: "name", "address", "tracking".
If a value is not found, use null. Do not add any other text.

Text:
{ocr_text}

JSON:"""

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 150,
                "stop": ["\\n\\n", "User:", "Instruction:"]
            }
        }
        
        try:
            # First check if Ollama is reachable
            try:
                health_check = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if health_check.status_code != 200:
                    raise ConnectionError(f"Ollama service not accessible. Status: {health_check.status_code}")
            except requests.exceptions.ConnectionError:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 404:
                logger.warning(f"Model {self.model} not found, attempting to pull...")
                import subprocess
                subprocess.run(f"ollama pull {self.model}", shell=True, check=True)
                response = requests.post(self.api_url, json=payload, timeout=30)
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def _parse_response(self, response: Dict[str, Any]) -> List[Dict[str, str]]:
        """Parse the LLM response."""
        try:
            response_text = response.get("response", "").strip()
            logger.debug(f"LLM Response: {response_text}")
            
            # Try to find JSON object using regex if direct parse fails
            import re
            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up potential common JSON errors
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\\s*\\}', '}', json_str)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extracted JSON: {json_str}")
                    return []
            else:
                try:
                    data = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning("No JSON object found in response")
                    return []
            
            # Normalize to list
            if isinstance(data, dict):
                pairs = [data]
            elif isinstance(data, list):
                pairs = data
            else:
                return []
                
            valid_pairs = []
            for pair in pairs:
                # Get name
                name = (pair.get("input_name") or pair.get("name") or 
                       pair.get("recipient_name") or pair.get("full_name") or "")
                
                # Get address
                address = (pair.get("input_address") or pair.get("address") or 
                          pair.get("recipient_address") or pair.get("full_address") or "")
                
                # Get tracking number
                tracking = (pair.get("tracking") or pair.get("tracking_number") or 
                           pair.get("tracking_id") or None)
                
                # Clean up address if it's a dict
                if isinstance(address, dict):
                    street = address.get('street', '').strip()
                    city = address.get('city', '').strip()
                    state = address.get('state', '').strip()
                    zip_code = address.get('zip', '').strip()
                    
                    parts = []
                    if street: parts.append(street)
                    if city: parts.append(city)
                    if state and zip_code: parts.append(f"{state} {zip_code}")
                    elif state: parts.append(state)
                    elif zip_code: parts.append(zip_code)
                    
                    address = ", ".join(parts) if parts else ""
                elif isinstance(address, list):
                    address = ", ".join(str(x) for x in address if x)
                
                name = str(name).strip() if name else ""
                address = str(address).strip() if address else ""
                tracking = str(tracking).strip() if tracking and str(tracking).lower() != "null" else None
                
                # Validate name
                name_words = name.split()
                if len(name_words) < 2:
                    logger.warning(f"Short name '{name}', might be incomplete")
                
                # Filter out place names
                place_names = ['pompano', 'pamgiano', 'pumgiana', 'beach', 'roseville', 'williston', 
                              'tampa', 'norcross', 'chicago', 'webster', 'bethesda', 'folsom',
                              'cleveland', 'harvey', 'euless', 'mishawaka', 'fort', 'collins']
                if any(place in name.lower() for place in place_names):
                    logger.warning(f"Filtered out place name as person name: {name}")
                    continue
                
                # Filter out non-name words
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
                
                if (name and len(name) > 2) or (address and len(address) > 5):
                    valid_pairs.append(normalized)
            
            return valid_pairs
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
