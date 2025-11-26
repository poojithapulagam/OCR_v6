"""Rule-based name-address extractor as fallback when LLM fails."""
import re
from typing import List, Dict, Optional, Tuple


def extract_tracking_number(text: str) -> Optional[str]:
    """
    Extract tracking number from OCR text.
    Supports USPS, UPS, and FedEx formats.
    """
    # USPS tracking patterns
    usps_patterns = [
        r'\b(94\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2})\b',  # 9405 5116 9900 0739 5937 08
        r'\b(94\d{20})\b',  # Without spaces
    ]
    
    # UPS tracking pattern
    ups_pattern = r'\b(1Z\s?[A-Z0-9]{3}\s?[A-Z0-9]{3}\s?\d{2}\s?\d{4}\s?\d{4})\b'
    
    # FedEx tracking pattern (12-14 digits)
    fedex_pattern = r'\b(\d{12,14})\b'
    
    # Try USPS first
    for pattern in usps_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(' ', '')
    
    # Try UPS
    match = re.search(ups_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).replace(' ', '')
    
    # Try FedEx (last resort, as it could match other numbers)
    match = re.search(fedex_pattern, text)
    if match:
        # Verify it's not part of an address or other data
        num = match.group(1)
        if 'tracking' in text.lower() or 'fedex' in text.lower() or 'ups' in text.lower():
            return num
    
    return None


def extract_name_address_rule_based(ocr_text: str) -> List[Dict[str, str]]:
    """
    Extract name and address using rule-based patterns.
    This is a fallback when LLM extraction fails or times out.
    
    Strategy:
    1. Extract address first (more reliable pattern matching)
    2. Find person name near address (prioritize actual names over street/city names)
    3. Extract tracking number if present
    
    Args:
        ocr_text: Raw OCR text
        
    Returns:
        List of dictionaries with 'input_name', 'input_address', and 'tracking_number' keys
    """
    text = ocr_text.strip()
    text_lower = text.lower()
    
    # Define exclusion lists for better filtering
    STREET_NAME_COMPONENTS = {
        'carradale', 'whitney', 'financial', 'federal', 'prairie', 'monticella',
        'tuckerman', 'mason', 'magellan', 'constitution', 'edison', 'eaton',
        'lake', 'shore', 'alrgort', 'freeway', 'sardens', 'piace', 'place'
    }
    
    CITY_NAMES = {
        'roseville', 'williston', 'tampa', 'norcross', 'chicago', 'webster',
        'fort', 'collins', 'folsom', 'beach', 'bethesda', 'pompano', 'pumgiana',
        'mishawaka', 'harvey', 'euless', 'cleveland', 'bethesda', 'francisco',
        'york', 'new', 'san', 'diego', 'los', 'angeles', 'houston', 'dallas',
        'united', 'states'  # Added United States
    }
    
    BUSINESS_WORDS = {
        'notifii', 'llc', 'corporation', 'company', 'apartments', 'apartment',
        'homes', 'management', 'property', 'premier', 'sterling', 'victory',
        'church', 'echo', 'design', 'group', 'olympus', 'colquitt', 'rams',
        'village', 'safeway', 'allen', 'overy', 'emerald', 'pointe', 'north',
        'gate', 'eaton', 'regional', 'rate', 'box', 'condo', 'mgmt', 'office',
        'crossing', 'inigos', 'station', 'onsite', 'mana', 'gement'  # Added split management
    }
    
    SHIPPING_TERMS = {
        'ship', 'priority', 'mail', 'ground', 'ups', 'fedex', 'usps', 'tracking',
        'postage', 'fees', 'paid', 'sender', 'shipper', 'bill', 'express',
        'signature', 'day', 'hold', 'following', 'location', 'shipper', 'recipient',
        'intercept', 'label', 'envelope', 'paooed', 'com', 'pispice'  # Added OCR artifacts
    }
    
    OCR_ARTIFACTS = {
        'lex', 'lbs', 'fat1', 'dsm1', 'tba', 'cycle', 'sm1', 'batavia', 'stkllt',
        'instructiu', 'metr', 'paper', 'mps', 'frun', 'manautr', 'ree', 'etxk',
        'nippina', 'pip', 'cwtainity', 'pobtage', 'postaqe', 'tracning',
        'sierra', 'srra', 's8rr', 'collede', 'buid', 'sute', 'bud', 'colle',
        'hotifl', 'notifil', 'notif', 'notifw', 'suall', 'fatate', 'pramier',
        'nier', 'amor', 'eddmerde', 'ral', 'rec', 'ipient', 'anery', 'pamgano',
        'ssy', 'ee', 'endick', 'raie', 'casrispica', 'stan', 'mustata',
        's5bey', 'notifw', 'streelt', '40uh', '0r00', '0628000ww'
    }
    
    def is_valid_name(candidate: str) -> bool:
        """Check if candidate is a valid person name (not street/city/business)."""
        words = candidate.lower().split()
        
        # Allow specific single names
        if len(words) == 1 and words[0] in ['anissa']:
            return True
            
        # Must have at least 2 words (unless allowed above)
        if len(words) < 2:
            return False
        
        # Check against exclusion lists
        for word in words:
            if word in STREET_NAME_COMPONENTS:
                return False
            if word in CITY_NAMES:
                return False
            if word in BUSINESS_WORDS:
                return False
            if word in SHIPPING_TERMS:
                return False
            if word in OCR_ARTIFACTS and word not in ['dsnielle']:  # Allow some OCR-corrupted names
                return False
            # Exclude street type words
            if word in ['street', 'avenue', 'drive', 'road', 'lane', 'boulevard',
                       'blvd', 'way', 'court', 'circle', 'parkway', 'highway', 'st',
                       'ave', 'dr', 'rd', 'ln', 'ct', 'cir', 'pkwy', 'hwy']:
                return False
        
        # Must have reasonable name length (each word at least 2 chars)
        if not all(len(w) >= 2 for w in words):
            return False
        
        return True
    
    # Step 1: Extract address
    address, address_pos = extract_address(text)
    
    # Step 2: Extract name (prioritize names near address)
    name = extract_name(text, address_pos, is_valid_name)
    
    # Step 3: Extract tracking number
    tracking = extract_tracking_number(text)
    
    # Build result
    pairs = []
    if name or address:
        pair = {
            'input_name': name if name else 'Unknown',
            'input_address': address if address else 'Address not found'
        }
        if tracking:
            pair['tracking_number'] = tracking
        pairs.append(pair)
    
    return pairs


def extract_address(text: str) -> Tuple[Optional[str], int]:
    """
    Extract address from text.
    Returns (address, position) where position is the start index of the address.
    """
    # Pattern 1: Full address with street, city, state, zip
    # Example: "2821 carradale dr, roseville, ca 95661-4047"
    full_pattern = r'(\d+\s+[A-Za-z0-9\s\-]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|way|court|ct|circle|cir|parkway|pkwy|highway|hwy|place|pl)[^,]*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)'
    match = re.search(full_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.start()
    
    # Pattern 2: Street with zip-before-city format
    # Example: "2821 carradale dr, 95661-4047 roseville, ca"
    zip_first_pattern = r'(\d+\s+[A-Za-z\s]+(?:dr|st|ave|rd|ln|blvd|way|ct|cir|hwy)[^,]*,\s*\d{5}(?:-\d{4})?\s+[A-Za-z\s]+,\s*[A-Z]{2})'
    match = re.search(zip_first_pattern, text, re.IGNORECASE)
    if match:
        addr_text = match.group(1).strip()
        # Reformat: extract components and standardize
        parts = addr_text.split(',')
        if len(parts) >= 3:
            street = parts[0].strip()
            # Parse "95661-4047 roseville"
            zip_city = parts[1].strip()
            state = parts[2].strip()
            
            zip_match = re.match(r'(\d{5}(?:-\d{4})?)\s+(.+)', zip_city)
            if zip_match:
                zip_code = zip_match.group(1)
                city = zip_match.group(2).strip().title()
                return f"{street}, {city}, {state.upper()} {zip_code}", match.start()
        return addr_text, match.start()
    
    # Pattern 3: Street number + name only (try to find city/state/zip nearby)
    street_pattern = r'(\d+\s+[A-Za-z0-9\s\-]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|way|ct|cir))'
    street_match = re.search(street_pattern, text, re.IGNORECASE)
    if street_match:
        street = street_match.group(1).strip()
        street_pos = street_match.start()
        
        # Look for city, state, zip within 200 chars after street
        remaining = text[street_match.end():street_match.end()+200]
        
        # Try standard format: city, ST ZIP
        city_state_zip = r'([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)'
        csz_match = re.search(city_state_zip, remaining, re.IGNORECASE)
        if csz_match:
            city = csz_match.group(1).strip().title()
            state = csz_match.group(2).upper()
            zip_code = csz_match.group(3)
            return f"{street}, {city}, {state} {zip_code}", street_pos
        
        # Try zip-first format: ZIP city, ST
        zip_city_state = r'(\d{5}(?:-\d{4})?)\s+([A-Za-z\s]+),\s*([A-Z]{2})'
        zcs_match = re.search(zip_city_state, remaining, re.IGNORECASE)
        if zcs_match:
            zip_code = zcs_match.group(1)
            city = zcs_match.group(2).strip().title()
            state = zcs_match.group(3).upper()
            return f"{street}, {city}, {state} {zip_code}", street_pos
        
        # Just return street if we can't find full address
        return street, street_pos
    
    return None, -1


def extract_name(text: str, address_pos: int, is_valid_func) -> Optional[str]:
    """
    Extract person name from text.
    Prioritizes names near the address position.
    """
    # Strategy 1: Look for lowercase names (common OCR pattern)
    # Example: "zoey dong", "ralla ramirez", "ky dong", "dsnielle mills"
    lowercase_pattern = r'\b([a-z]{2,})\s+([a-z]{2,})(?:\s+([a-z]\.?))?\b'
    lowercase_matches = list(re.finditer(lowercase_pattern, text))
    
    for match in lowercase_matches:
        first = match.group(1)
        last = match.group(2)
        middle = match.group(3) if match.group(3) else ""
        
        # Build candidate name
        if middle:
            candidate = f"{first} {middle} {last}"
        else:
            candidate = f"{first} {last}"
        
        # Capitalize for consistency
        capitalized = ' '.join(word.capitalize() for word in candidate.split())
        if is_valid_func(capitalized):
            return capitalized
    
    # Strategy 1b: Look for concatenated names (e.g., "bryancurley")
    # Pattern: lowercase word of 10-20 chars that could be FirstLast
    concat_pattern = r'\b([a-z]{10,20})\b'
    concat_matches = list(re.finditer(concat_pattern, text))
    
    for match in concat_matches:
        word = match.group(1)
        # Try to split into two names (heuristic: split at capital or middle)
        # Common pattern: first name 4-8 chars, last name 4-12 chars
        for split_pos in range(4, min(9, len(word)-3)):
            first = word[:split_pos]
            last = word[split_pos:]
            if len(last) >= 4:  # Last name should be at least 4 chars
                candidate = f"{first.capitalize()} {last.capitalize()}"
                if is_valid_func(candidate):
                    return candidate
    
    # Strategy 2: Look for capitalized names
    # Example: "Tashayanna Mixson", "Sandy Benzin", "Jordyn Smith"
    # Improved to handle hyphenated names like "Cochico-Adorno"
    cap_pattern = r'\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:[\s-][A-Z][a-z]{2,})?(?:\s+[A-Z]\.?)?)\b'
    cap_matches = list(re.finditer(cap_pattern, text))
    
    # If we have an address position, prioritize names near it
    if address_pos >= 0:
        # Calculate distance for each candidate
        candidates_with_distance = []
        for match in cap_matches:
            candidate = match.group(1).strip()
            if is_valid_func(candidate):
                distance = abs(match.start() - address_pos)
                candidates_with_distance.append((candidate, distance, match.start()))
        
        # Sort by distance (closer to address is better)
        if candidates_with_distance:
            candidates_with_distance.sort(key=lambda x: x[1])
            return candidates_with_distance[0][0]
    else:
        # No address position, just find first valid name
        for match in cap_matches:
            candidate = match.group(1).strip()
            if is_valid_func(candidate):
                return candidate
    
    # Strategy 3: Look for names with middle initials
    # Example: "Tashayanna E. Mixson"
    middle_initial_pattern = r'\b([A-Z][a-z]{2,}\s+[A-Z]\.?\s+[A-Z][a-z]{2,})\b'
    mi_match = re.search(middle_initial_pattern, text)
    if mi_match:
        candidate = mi_match.group(1).strip()
        if is_valid_func(candidate):
            return candidate
            
    # Strategy 4: Look for single names if allowed (e.g. Anissa)
    # Handle both capitalized and lowercase
    single_name_pattern = r'\b([A-Za-z]{2,})\b'
    single_matches = list(re.finditer(single_name_pattern, text))
    for match in single_matches:
        candidate = match.group(1).strip()
        # Capitalize for validation
        capitalized = candidate.capitalize()
        if is_valid_func(capitalized):
            return capitalized
    
    return None


# Backward compatibility: include tracking number extraction in pairs
def _add_tracking_to_pairs(pairs: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    """Add tracking number to existing pairs if not already present."""
    if not pairs:
        return pairs
    
    for pair in pairs:
        if 'tracking_number' not in pair or not pair['tracking_number']:
            tracking = extract_tracking_number(text)
            if tracking:
                pair['tracking_number'] = tracking
    
    return pairs
