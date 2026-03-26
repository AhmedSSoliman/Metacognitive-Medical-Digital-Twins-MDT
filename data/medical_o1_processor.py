"""
Medical-O1 Reasoning Dataset Processor.

Processes Medical-O1-Reasoning-SFT dataset for cognitive stream training.
This dataset contains medical reasoning chains with chain-of-thought explanations.

Dataset: FreedomIntelligence/medical-o1-reasoning-SFT
Configs: 'en', 'zh', 'en_mix', 'zh_mix'

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MedicalO1Processor:
    """
    Processor for Medical-O1 reasoning dataset.
    
    Formats Medical-O1 examples into triple-stream cognitive architecture
    compatible with the Medical Digital Twin training pipeline.
    """
    
    def __init__(self):
        """Initialize Medical-O1 processor."""
        logger.info("Initialized MedicalO1Processor")
    
    def load_dataset(self, split: str = 'train', config: str = 'en') -> Optional[object]:
        """
        Load Medical-O1 dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            config: Dataset config:
                - 'en': English medical questions
                - 'zh': Chinese medical questions
                - 'en_mix': English mixed-domain questions
                - 'zh_mix': Chinese mixed-domain questions
        
        Returns:
            HuggingFace dataset object or None if failed
        
        Example:
            >>> processor = MedicalO1Processor()
            >>> dataset = processor.load_dataset('train', 'en')
            >>> print(f"Loaded {len(dataset)} examples")
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading FreedomIntelligence/medical-o1-reasoning-SFT (config={config}) from HuggingFace...")
            
            dataset = load_dataset(
                "FreedomIntelligence/medical-o1-reasoning-SFT",
                config,  # Specify config here (en, zh, en_mix, zh_mix)
                split=split
            )
            
            logger.info(f"✓ Loaded {len(dataset)} examples from Medical-O1 dataset")
            return dataset
            
        except ImportError:
            logger.error("datasets library not available. Install with: pip install datasets")
            return None
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def format_for_training(
        self,
        dataset: object,
        max_examples: int = 5000
    ) -> List[Dict]:
        """
        Format Medical-O1 examples for training.
        
        Converts Medical-O1 dataset format into triple-stream architecture:
            - prompt: Clinical question/scenario
            - think: Chain-of-thought reasoning
            - patient_state: Extracted clinical state (if available)
            - user_belief: Inferred user expertise level
        
        Args:
            dataset: HuggingFace dataset object
            max_examples: Maximum number of examples to process
        
        Returns:
            List of formatted training examples
        
        Example:
            >>> processor = MedicalO1Processor()
            >>> dataset = processor.load_dataset('train', 'en')
            >>> examples = processor.format_for_training(dataset, max_examples=100)
            >>> print(f"Formatted {len(examples)} examples")
        """
        training_examples = []
        
        # Limit number of examples
        num_examples = min(len(dataset), max_examples)
        
        logger.info(f"Formatting {num_examples} Medical-O1 examples...")
        
        for i in range(num_examples):
            try:
                example = dataset[i]
                
                # Extract fields from Medical-O1 dataset
                # Common field names: 'input', 'question', 'output', 'answer', 'reasoning', 'thought'
                prompt = self._extract_field(example, ['input', 'question', 'query', 'Question'])
                reasoning = self._extract_field(example, ['reasoning', 'thought', 'chain_of_thought', 'Complex_CoT'])
                answer = self._extract_field(example, ['output', 'answer', 'response', 'Response'])
                
                # Skip if missing critical fields
                if not prompt or not reasoning:
                    logger.debug(f"Skipping example {i}: missing prompt or reasoning")
                    continue
                
                # Format into training example with triple-stream architecture
                formatted = {
                    'prompt': prompt,
                    'think': reasoning[:1500],  # Limit reasoning length
                    'patient_state': self._extract_patient_state(prompt, reasoning),
                    'user_belief': self._infer_user_belief(prompt)
                }
                
                training_examples.append(formatted)
                
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"✓ Formatted {len(training_examples)} Medical-O1 examples")
        return training_examples
    
    def _extract_field(self, example: Dict, field_names: List[str]) -> str:
        """
        Extract field from example using multiple possible field names.
        
        Args:
            example: Dataset example dictionary
            field_names: List of possible field names to try
        
        Returns:
            Field value or empty string if not found
        """
        for field_name in field_names:
            if field_name in example and example[field_name]:
                return str(example[field_name])
        return ""
    
    def _extract_patient_state(self, prompt: str, reasoning: str) -> str:
        """
        Extract patient state from prompt/reasoning.
        
        Medical-O1 dataset may not have explicit vital signs or lab values.
        This method attempts to extract any clinical state information or
        returns a generic placeholder.
        
        Args:
            prompt: Input question/scenario
            reasoning: Chain-of-thought reasoning
        
        Returns:
            Patient state string (may be generic for Medical-O1)
        
        Example:
            >>> processor = MedicalO1Processor()
            >>> state = processor._extract_patient_state(
            ...     "Patient with chest pain",
            ...     "Considering differential diagnosis..."
            ... )
        """
        # Look for vital signs or lab values in prompt/reasoning
        clinical_terms = [
            'blood pressure', 'BP', 'heart rate', 'HR', 'temperature', 'temp',
            'SpO2', 'oxygen', 'lactate', 'creatinine', 'glucose', 'WBC'
        ]
        
        found_terms = []
        text = (prompt + " " + reasoning).lower()
        
        for term in clinical_terms:
            if term.lower() in text:
                found_terms.append(term)
        
        if found_terms:
            return f"Clinical case with documented: {', '.join(found_terms[:3])}"
        else:
            # Generic patient state for Medical-O1 examples
            return "Clinical case under evaluation, state assessment based on provided history"
    
    def _infer_user_belief(self, prompt: str) -> str:
        """
        Infer user belief from prompt characteristics.
        
        Estimates user's medical literacy and expertise based on:
            - Question complexity
            - Use of medical terminology
            - Question length and structure
        
        Args:
            prompt: Input question
        
        Returns:
            User belief string with literacy and expertise assessment
        
        Example:
            >>> processor = MedicalO1Processor()
            >>> belief = processor._infer_user_belief(
            ...     "What is the mechanism of action of metformin?"
            ... )
            >>> # Returns: "Literacy: medium, ..."
        """
        # Count medical terms (simple heuristic)
        medical_terms = [
            'diagnosis', 'treatment', 'pathophysiology', 'etiology',
            'prognosis', 'differential', 'mechanism', 'pharmacology',
            'contraindication', 'complication', 'manifestation'
        ]
        
        prompt_lower = prompt.lower()
        term_count = sum(1 for term in medical_terms if term in prompt_lower)
        
        # Estimate complexity
        word_count = len(prompt.split())
        
        # Determine literacy level
        if term_count >= 3 or word_count > 100:
            literacy = "high"
            background = "professional or advanced student"
            expectation = "seeking detailed technical explanation"
        elif term_count >= 1 or word_count > 50:
            literacy = "medium"
            background = "some medical knowledge"
            expectation = "seeking clear explanation with context"
        else:
            literacy = "low"
            background = "layperson"
            expectation = "seeking simple, accessible explanation"
        
        return f"Literacy: {literacy}, Medical background: {background}, {expectation}"
    
    def validate_example(self, example: Dict) -> bool:
        """
        Validate that example has required fields.
        
        Args:
            example: Formatted training example
        
        Returns:
            True if example is valid, False otherwise
        """
        required_fields = ['prompt', 'think', 'patient_state', 'user_belief']
        
        for field in required_fields:
            if field not in example or not example[field]:
                return False
        
        # Check minimum lengths
        if len(example['think']) < 50:
            return False
        
        return True
    
    def get_dataset_info(self) -> Dict[str, str]:
        """
        Get information about Medical-O1 dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'name': 'Medical-O1-Reasoning-SFT',
            'source': 'FreedomIntelligence/medical-o1-reasoning-SFT',
            'configs': ['en', 'zh', 'en_mix', 'zh_mix'],
            'description': 'Medical reasoning dataset with chain-of-thought explanations',
            'recommended_config': 'en (English medical questions)'
        }