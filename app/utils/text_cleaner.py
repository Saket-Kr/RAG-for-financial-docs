"""
Text cleaning utilities to improve PDF extraction quality.

Handles common PDF extraction issues like stray characters,
poor formatting, and enhances important information like scores.
"""
import re
from typing import List


class TextCleaner:
    """Cleans and enhances text extracted from PDFs."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean PDF-extracted text and enhance score-related information.
        
        Args:
            text: Raw text from PDF extraction
            
        Returns:
            Cleaned and enhanced text
        """
        if not text:
            return text
        
        # First pass: enhance score sections (before cleanup to preserve patterns)
        text = TextCleaner._enhance_scores(text)
        
        # Second pass: basic cleanup
        text = TextCleaner._basic_cleanup(text)
        
        return text

    @staticmethod
    def _basic_cleanup(text: str) -> str:
        """Remove common PDF extraction artifacts."""
        # Clean up excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    @staticmethod
    def _enhance_scores(text: str) -> str:
        """
        Enhance score-related text to make it more structured and readable.
        
        Specifically handles patterns like:
        "Generic Risk Score p VANTAGESCORE 3.0 Insight Score\n604 m 609 586"
        """
        lines = text.split('\n')
        enhanced_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line contains score labels
            if re.search(r'VANTAGESCORE|Generic\s+Risk\s+Score|Insight\s+Score', line, re.IGNORECASE):
                # Look ahead for the next line which might contain score values
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # Extract score values (3-4 digit numbers in the 300-850 range)
                    score_values = re.findall(r'\b(\d{3,4})\b', next_line)
                    valid_scores = [
                        s for s in score_values 
                        if 300 <= int(s) <= 850
                    ]
                    
                    if valid_scores:
                        # Check what score types are in the current line
                        enhanced_line = TextCleaner._format_score_line(line, valid_scores)
                        enhanced_lines.append(enhanced_line)
                        
                        # Skip the next line if we've processed it
                        i += 2
                        continue
            
            enhanced_lines.append(line)
            i += 1
        
        return '\n'.join(enhanced_lines)

    @staticmethod
    def _format_score_line(line: str, scores: List[str]) -> str:
        """
        Format a line containing score labels with their corresponding values.
        
        Handles patterns like:
        - "Generic Risk Score p VANTAGESCORE 3.0 Insight Score" with scores [604, 609, 586]
        """
        # Clean up stray characters in the line first
        cleaned_line = re.sub(r'\s+([a-z])\s+', ' ', line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
        
        results = []
        
        # Extract score types in the order they appear in the line
        score_types = []
        
        # Find positions of each score type in the original line
        generic_match = re.search(r'Generic\s+Risk\s+Score', cleaned_line, re.IGNORECASE)
        vantage_match = re.search(r'VANTAGESCORE\s+([\d.]+)', cleaned_line, re.IGNORECASE)
        insight_match = re.search(r'Insight\s+Score', cleaned_line, re.IGNORECASE)
        
        # Build list of (score_type, position_in_line) tuples
        type_positions = []
        if generic_match:
            type_positions.append(('Generic Risk Score', generic_match.start()))
        if vantage_match:
            version = vantage_match.group(1)
            type_positions.append((f'VANTAGESCORE {version}', vantage_match.start()))
        if insight_match:
            type_positions.append(('Insight Score', insight_match.start()))
        
        # Sort by position in line to get the correct order
        type_positions.sort(key=lambda x: x[1])
        
        # Match score types with values in order
        for idx, (score_type, _) in enumerate(type_positions):
            if idx < len(scores):
                results.append(f"{score_type}: {scores[idx]}")
        
        if results:
            return '\n'.join(results)
        
        # Fallback: if we can't parse it properly, just clean it up
        return cleaned_line
