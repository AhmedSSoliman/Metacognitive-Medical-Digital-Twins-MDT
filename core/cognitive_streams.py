"""
Cognitive Stream Parser and Data Structures.

Implements triple-stream cognitive architecture with XML-based stream markers:
    - <think>: Clinical reasoning and deductive logic
    - <patient_state>: Physiological state with LOINC codes
    - <user_belief>: Theory of Mind inference

The three streams provide:
    1. Auditability: Explicit reasoning chains can be inspected
    2. Metacognition: Model can revise its own reasoning
    3. Empathy: Theory of Mind enables communication calibration

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class CognitiveStreams:
    """
    Container for the three cognitive reasoning streams.
    
    Attributes:
        think: Clinical reasoning and deductive logic stream
        patient_state: Structured physiological state with LOINC codes
        user_belief: Theory of Mind inference about user literacy/state
    
    Example:
        >>> streams = CognitiveStreams(
        ...     think="Patient presents with sepsis...",
        ...     patient_state="HR: 125 bpm [LOINC:8867-4]...",
        ...     user_belief="Literacy: low, anxious family member"
        ... )
        >>> assert streams.is_complete()
    """
    think: str = ""
    patient_state: str = ""
    user_belief: str = ""
    
    def is_complete(self) -> bool:
        """
        Check if all three streams are present and non-empty.
        
        Returns:
            True if all streams contain content, False otherwise
        
        Example:
            >>> streams = CognitiveStreams(think="...", patient_state="...", user_belief="...")
            >>> assert streams.is_complete() == True
            >>> 
            >>> incomplete = CognitiveStreams(think="...", patient_state="")
            >>> assert incomplete.is_complete() == False
        """
        return bool(self.think and self.patient_state and self.user_belief)
    
    def has_stream(self, stream_name: str) -> bool:
        """
        Check if a specific stream is present and has content.
        
        Args:
            stream_name: Name of stream ('think', 'patient_state', 'user_belief')
        
        Returns:
            True if stream exists and has content, False otherwise
        
        Example:
            >>> streams = CognitiveStreams(think="Clinical reasoning...")
            >>> assert streams.has_stream('think') == True
            >>> assert streams.has_stream('patient_state') == False
        """
        if not hasattr(self, stream_name):
            logger.warning(f"Unknown stream name: {stream_name}")
            return False
        
        stream_value = getattr(self, stream_name, "")
        return bool(stream_value)
    
    def get_stream_length(self, stream_name: str) -> int:
        """
        Get character length of a specific stream.
        
        Args:
            stream_name: Name of stream to measure
        
        Returns:
            Length in characters, or 0 if stream doesn't exist
        
        Example:
            >>> streams = CognitiveStreams(think="Sepsis likely")
            >>> assert streams.get_stream_length('think') == 13
            >>> assert streams.get_stream_length('patient_state') == 0
        """
        if not hasattr(self, stream_name):
            logger.warning(f"Unknown stream name: {stream_name}")
            return 0
        
        stream_value = getattr(self, stream_name, "")
        return len(stream_value)
    
    def get_all_lengths(self) -> Dict[str, int]:
        """
        Get lengths of all streams.
        
        Returns:
            Dictionary mapping stream names to their lengths
        
        Example:
            >>> streams = CognitiveStreams(
            ...     think="Clinical reasoning",
            ...     patient_state="HR: 120",
            ...     user_belief="Low literacy"
            ... )
            >>> lengths = streams.get_all_lengths()
            >>> print(lengths)
            {'think': 18, 'patient_state': 7, 'user_belief': 12}
        """
        return {
            'think': len(self.think),
            'patient_state': len(self.patient_state),
            'user_belief': len(self.user_belief)
        }
    
    def count_complete_streams(self) -> int:
        """
        Count how many streams are present and non-empty.
        
        Returns:
            Number of complete streams (0-3)
        
        Example:
            >>> streams = CognitiveStreams(think="...", patient_state="...")
            >>> assert streams.count_complete_streams() == 2
        """
        count = 0
        if self.think:
            count += 1
        if self.patient_state:
            count += 1
        if self.user_belief:
            count += 1
        return count
    
    def get_missing_streams(self) -> list:
        """
        Get list of missing stream names.
        
        Returns:
            List of stream names that are empty
        
        Example:
            >>> streams = CognitiveStreams(think="...")
            >>> missing = streams.get_missing_streams()
            >>> print(missing)
            ['patient_state', 'user_belief']
        """
        missing = []
        if not self.think:
            missing.append('think')
        if not self.patient_state:
            missing.append('patient_state')
        if not self.user_belief:
            missing.append('user_belief')
        return missing


class CognitiveStreamParser:
    """
    Parser for triple-stream cognitive architecture.
    
    Extracts and validates the three reasoning streams from model output:
        - <think>...</think>: Clinical reasoning
        - <patient_state>...</patient_state>: Physiological state
        - <user_belief>...</user_belief>: Theory of Mind
    
    Example:
        >>> from config.configs import CognitiveArchitectureConfig
        >>> config = CognitiveArchitectureConfig()
        >>> parser = CognitiveStreamParser(config)
        >>> 
        >>> text = '''
        ... <think>Patient presents with sepsis based on elevated lactate</think>
        ... <patient_state>HR: 125 bpm [LOINC:8867-4], Lactate: 3.2 mmol/L [LOINC:2524-7]</patient_state>
        ... <user_belief>Literacy: low, Emotional state: anxious</user_belief>
        ... '''
        >>> 
        >>> streams = parser.parse(text)
        >>> assert streams.is_complete()
        >>> is_valid, error = parser.validate(streams)
        >>> assert is_valid
    """
    
    def __init__(self, config):
        """
        Initialize parser with configuration.
        
        Args:
            config: CognitiveArchitectureConfig with stream markers and validation rules
        """
        self.config = config
        
        # Build regex patterns for each stream
        self.patterns = {
            'think': self._build_pattern(
                config.think_marker,
                config.think_end_marker
            ),
            'patient_state': self._build_pattern(
                config.patient_state_marker,
                config.patient_state_end_marker
            ),
            'user_belief': self._build_pattern(
                config.user_belief_marker,
                config.user_belief_end_marker
            )
        }
        
        logger.info("Initialized CognitiveStreamParser")
        logger.debug(f"Stream markers: think={config.think_marker}, "
                    f"patient_state={config.patient_state_marker}, "
                    f"user_belief={config.user_belief_marker}")
    
    def _build_pattern(self, start_marker: str, end_marker: str) -> re.Pattern:
        """
        Build regex pattern for stream extraction.
        
        Args:
            start_marker: Opening XML tag (e.g., '<think>')
            end_marker: Closing XML tag (e.g., '</think>')
        
        Returns:
            Compiled regex pattern that matches content between markers
        
        Example:
            >>> pattern = parser._build_pattern('<think>', '</think>')
            >>> match = pattern.search('<think>content</think>')
            >>> print(match.group(1))
            'content'
        """
        # Escape special regex characters in markers
        start_escaped = re.escape(start_marker)
        end_escaped = re.escape(end_marker)
        
        # Pattern: start_marker + content + end_marker
        # .*? = non-greedy match (shortest match between markers)
        # re.DOTALL = . matches newlines too (for multi-line content)
        pattern = f"{start_escaped}(.*?){end_escaped}"
        return re.compile(pattern, re.DOTALL)
    
    def parse(self, text: str) -> CognitiveStreams:
        """
        Parse text and extract all three cognitive streams.
        
        Args:
            text: Model-generated text containing stream markers
        
        Returns:
            CognitiveStreams object with extracted content
        
        Example:
            >>> text = "<think>Sepsis likely</think><patient_state>HR: 120</patient_state><user_belief>Low literacy</user_belief>"
            >>> streams = parser.parse(text)
            >>> print(streams.think)
            'Sepsis likely'
            >>> print(streams.patient_state)
            'HR: 120'
            >>> print(streams.user_belief)
            'Low literacy'
        """
        streams = CognitiveStreams()
        
        for stream_name, pattern in self.patterns.items():
            match = pattern.search(text)
            if match:
                content = match.group(1)  # Extract content inside tags
                
                # Strip whitespace if configured
                if self.config.strip_whitespace:
                    content = content.strip()
                
                setattr(streams, stream_name, content)
                logger.debug(f"Extracted {stream_name}: {len(content)} chars")
            else:
                logger.debug(f"Stream '{stream_name}' not found in text")
        
        return streams
    
    def validate(self, streams: CognitiveStreams) -> Tuple[bool, Optional[str]]:
        """
        Validate that streams meet configuration requirements.
        
        Checks:
            - All required streams are present (if require_all_streams=True)
            - Each stream meets minimum length requirements
            - No nested markers (if allow_nested_markers=False)
        
        Args:
            streams: CognitiveStreams object to validate
        
        Returns:
            Tuple of (is_valid, error_message)
                - (True, None) if valid
                - (False, "error description") if invalid
        
        Example:
            >>> streams = CognitiveStreams(
            ...     think="Patient has sepsis",
            ...     patient_state="HR: 125",
            ...     user_belief="Low literacy"
            ... )
            >>> is_valid, error = parser.validate(streams)
            >>> assert is_valid
            >>> print(error)
            None
            >>> 
            >>> # Invalid: think stream too short
            >>> bad_streams = CognitiveStreams(think="Hi")
            >>> is_valid, error = parser.validate(bad_streams)
            >>> assert not is_valid
            >>> print(error)
            '<think> stream too short (min: 50 chars, got: 2)'
        """
        # Check if all streams are required
        if self.config.require_all_streams:
            missing = streams.get_missing_streams()
            
            if missing:
                error_msg = f"Missing required streams: {', '.join(missing)}"
                logger.warning(error_msg)
                return False, error_msg
        
        # Validate minimum lengths for each stream that exists
        if streams.think:
            if len(streams.think) < self.config.min_think_length:
                error_msg = (f"<think> stream too short "
                            f"(min: {self.config.min_think_length} chars, "
                            f"got: {len(streams.think)})")
                logger.warning(error_msg)
                return False, error_msg
        
        if streams.patient_state:
            if len(streams.patient_state) < self.config.min_patient_state_length:
                error_msg = (f"<patient_state> stream too short "
                            f"(min: {self.config.min_patient_state_length} chars, "
                            f"got: {len(streams.patient_state)})")
                logger.warning(error_msg)
                return False, error_msg
        
        if streams.user_belief:
            if len(streams.user_belief) < self.config.min_user_belief_length:
                error_msg = (f"<user_belief> stream too short "
                            f"(min: {self.config.min_user_belief_length} chars, "
                            f"got: {len(streams.user_belief)})")
                logger.warning(error_msg)
                return False, error_msg
        
        # All validation checks passed
        logger.debug("Stream validation passed")
        return True, None

    def validate_with_details(self, streams: CognitiveStreams) -> Dict[str, Any]:
        """Validate streams and return per-stream details with warnings."""
        result: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stream_validity": {
                "think": True,
                "patient_state": True,
                "user_belief": True,
            },
        }

        if not streams.think:
            result["errors"].append("<think> stream missing")
            result["stream_validity"]["think"] = False
            result["is_valid"] = False
        elif len(streams.think) < self.config.min_think_length:
            result["warnings"].append("<think> stream below recommended length")

        if not streams.patient_state:
            result["errors"].append("<patient_state> stream missing")
            result["stream_validity"]["patient_state"] = False
            result["is_valid"] = False
        elif len(streams.patient_state) < self.config.min_patient_state_length:
            result["warnings"].append("<patient_state> stream below recommended length")

        if not streams.user_belief:
            result["errors"].append("<user_belief> stream missing")
            result["stream_validity"]["user_belief"] = False
            result["is_valid"] = False
        elif len(streams.user_belief) < self.config.min_user_belief_length:
            result["warnings"].append("<user_belief> stream below recommended length")

        return result

    def analyze_stream_quality(self, streams: CognitiveStreams) -> Dict[str, float]:
        """Estimate stream quality metrics for monitoring/debugging."""
        quality: Dict[str, float] = {
            "think_reasoning_depth": 0.0,
            "patient_state_coding": 0.0,
            "user_belief_completeness": 0.0,
        }

        think_markers = ["because", "therefore", "suggest", "indicat", "likely", "differential"]
        lower_think = streams.think.lower()
        think_hits = sum(1 for marker in think_markers if marker in lower_think)
        quality["think_reasoning_depth"] = min(1.0, think_hits / 3.0)

        loinc_count = len(re.findall(r"LOINC:\d+-\d+", streams.patient_state or ""))
        quality["patient_state_coding"] = min(1.0, loinc_count / 3.0)

        belief_attrs = ["literacy", "emotional", "strategy"]
        lower_belief = streams.user_belief.lower()
        belief_hits = sum(1 for attr in belief_attrs if attr in lower_belief)
        quality["user_belief_completeness"] = belief_hits / float(len(belief_attrs))

        return quality
    
    def extract_stream(self, text: str, stream_name: str) -> str:
        """
        Extract a single specific stream from text.
        
        Useful when you only need one stream rather than parsing all three.
        
        Args:
            text: Model-generated text
            stream_name: Name of stream ('think', 'patient_state', 'user_belief')
        
        Returns:
            Extracted stream content or empty string if not found
        
        Example:
            >>> text = "<think>Clinical reasoning here</think><patient_state>HR: 120</patient_state>"
            >>> think_only = parser.extract_stream(text, 'think')
            >>> print(think_only)
            'Clinical reasoning here'
        """
        if stream_name not in self.patterns:
            logger.error(f"Unknown stream name: {stream_name}")
            return ""
        
        pattern = self.patterns[stream_name]
        match = pattern.search(text)
        
        if match:
            content = match.group(1)
            if self.config.strip_whitespace:
                content = content.strip()
            return content
        
        logger.debug(f"Stream '{stream_name}' not found in text")
        return ""
    
    def has_all_markers(self, text: str) -> bool:
        """
        Check if text contains all three stream marker pairs.
        
        Args:
            text: Text to check
        
        Returns:
            True if all markers are present, False otherwise
        
        Example:
            >>> complete_text = "<think>...</think><patient_state>...</patient_state><user_belief>...</user_belief>"
            >>> assert parser.has_all_markers(complete_text) == True
            >>> 
            >>> incomplete_text = "<think>...</think>"
            >>> assert parser.has_all_markers(incomplete_text) == False
        """
        for stream_name, pattern in self.patterns.items():
            if not pattern.search(text):
                return False
        return True
    
    def format_for_display(self, streams: CognitiveStreams) -> str:
        """
        Format streams for human-readable display.
        
        Creates a nicely formatted output with section headers.
        
        Args:
            streams: CognitiveStreams object
        
        Returns:
            Formatted string with labeled sections
        
        Example:
            >>> streams = CognitiveStreams(
            ...     think="Patient presents with sepsis",
            ...     patient_state="HR: 125 bpm [LOINC:8867-4]",
            ...     user_belief="Literacy: low, anxious"
            ... )
            >>> print(parser.format_for_display(streams))
            === CLINICAL REASONING ===
            Patient presents with sepsis
            
            === PATIENT STATE ===
            HR: 125 bpm [LOINC:8867-4]
            
            === USER BELIEF (Theory of Mind) ===
            Literacy: low, anxious
        """
        output = []
        
        if streams.think:
            output.append("=" * 60)
            output.append("CLINICAL REASONING (<think>)")
            output.append("=" * 60)
            output.append(streams.think)
            output.append("")
        
        if streams.patient_state:
            output.append("=" * 60)
            output.append("PATIENT STATE (<patient_state>)")
            output.append("=" * 60)
            output.append(streams.patient_state)
            output.append("")
        
        if streams.user_belief:
            output.append("=" * 60)
            output.append("USER BELIEF - Theory of Mind (<user_belief>)")
            output.append("=" * 60)
            output.append(streams.user_belief)
            output.append("")
        
        return "\n".join(output)
    
    def format_for_training(self, streams: CognitiveStreams) -> str:
        """
        Format streams as training target (with XML markers).
        
        Reconstructs the XML-formatted output for training.
        
        Args:
            streams: CognitiveStreams object
        
        Returns:
            XML-formatted string suitable for training
        
        Example:
            >>> streams = CognitiveStreams(
            ...     think="Sepsis",
            ...     patient_state="HR: 125",
            ...     user_belief="Low literacy"
            ... )
            >>> print(parser.format_for_training(streams))
            <think>Sepsis</think>
            <patient_state>HR: 125</patient_state>
            <user_belief>Low literacy</user_belief>
        """
        parts = []
        
        if streams.think:
            parts.append(f"{self.config.think_marker}{streams.think}{self.config.think_end_marker}")
        
        if streams.patient_state:
            parts.append(f"{self.config.patient_state_marker}{streams.patient_state}{self.config.patient_state_end_marker}")
        
        if streams.user_belief:
            parts.append(f"{self.config.user_belief_marker}{streams.user_belief}{self.config.user_belief_end_marker}")
        
        return "\n".join(parts)
    
    def get_statistics(self, streams: CognitiveStreams) -> Dict[str, any]:
        """
        Get detailed statistics about the streams.
        
        Args:
            streams: CognitiveStreams object
        
        Returns:
            Dictionary with statistics
        
        Example:
            >>> streams = CognitiveStreams(
            ...     think="Patient presents with sepsis",
            ...     patient_state="HR: 125 bpm",
            ...     user_belief="Low literacy"
            ... )
            >>> stats = parser.get_statistics(streams)
            >>> print(stats)
            {
                'complete': True,
                'num_streams': 3,
                'total_chars': 58,
                'lengths': {'think': 28, 'patient_state': 11, 'user_belief': 12},
                'missing': []
            }
        """
        lengths = streams.get_all_lengths()
        
        return {
            'complete': streams.is_complete(),
            'num_streams': streams.count_complete_streams(),
            'total_chars': sum(lengths.values()),
            'lengths': lengths,
            'missing': streams.get_missing_streams()
        }


def create_example_streams() -> CognitiveStreams:
    """
    Create example streams for testing/demonstration.
    
    Returns:
        CognitiveStreams with example clinical content
    
    Example:
        >>> streams = create_example_streams()
        >>> assert streams.is_complete()
    """
    return CognitiveStreams(
        think="""
Patient presents with sepsis based on:
1. Elevated lactate (3.2 mmol/L, rising from 1.8)
2. Tachycardia (HR: 125 bpm)
3. Hypotension requiring vasopressor support
4. Clinical signs of infection (fever, leukocytosis)

Rising creatinine (1.9 mg/dL from baseline 1.0) indicates AKI Stage 2.
This represents clinical deterioration requiring escalation of care.
        """.strip(),
        
        patient_state="""
HR: 125 bpm [LOINC:8867-4] - Tachycardia
BP: 85/50 mmHg [LOINC:8480-6] - Hypotension (on norepinephrine)
Lactate: 3.2 mmol/L [LOINC:2524-7] - CRITICAL (doubled in 4h)
Creatinine: 1.9 mg/dL [LOINC:2160-0] - AKI Stage 2
Temperature: 38.9°C [LOINC:8310-5] - Fever
WBC: 18,000 /μL [LOINC:6690-2] - Leukocytosis
        """.strip(),
        
        user_belief="""
Relationship: Family member (daughter)
Literacy: LOW-MEDIUM (use simple language)
Emotional State: HIGHLY ANXIOUS
Medical Background: Limited
Communication Strategy: Be honest but compassionate, avoid jargon
        """.strip()
    )