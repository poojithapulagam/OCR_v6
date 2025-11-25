"""Text preprocessing module for OCR data."""
import re


def clean_ocr_text(ocr_text: str) -> str:
    """
    Preprocess OCR text by cleaning and standardizing.
    
    Args:
        ocr_text: Raw OCR text
        
    Returns:
        Cleaned and standardized text
    """
    if not ocr_text:
        return ""
    
    # Remove non-printable characters (keep newlines, tabs, spaces)
    text = re.sub(r'[^\x20-\x7E\n\t]', '', ocr_text)
    
    # Standardize whitespace (multiple spaces to single space)
    text = re.sub(r' +', ' ', text)
    
    # Standardize newlines (multiple newlines to single)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Final strip
    text = text.strip()
    
    return text
