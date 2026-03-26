"""
Ontology Validation Utilities.

Validates clinical data against standardized terminologies.

Author: Ahmed Soliman
Institution: University of Florida, HOBI
"""

import logging
from typing import Dict, List, Tuple, Optional
import re

from config.ontology import (
    LOINCCodes,
    SNOMEDCodes,
    ClinicalReferenceRanges,
    MIMICItemIDs,
    ICD10Codes
)






logger = logging.getLogger(__name__)


class OntologyValidator:
    """Validate clinical data against ontology standards."""
    
    def __init__(self):
        """Initialize validator with ontology."""

        

        




        self.loinc_codes = LOINCCodes.get_all_codes()
        self.snomed_codes = SNOMEDCodes.get_all_codes()
        self.icd10_codes = ICD10Codes.get_all_codes()  # NEW
        self.reference_ranges = ClinicalReferenceRanges.RANGES
        self.mimic_chart_items = MIMICItemIDs.get_all_chart_ids()
        self.mimic_lab_items = MIMICItemIDs.get_all_lab_ids()
        
        logger.info(f"Initialized OntologyValidator")
        logger.info(f"  LOINC codes: {len(self.loinc_codes)}")
        logger.info(f"  SNOMED codes: {len(self.snomed_codes)}")
        logger.info(f"  ICD-10 codes: {len(self.icd10_codes)}")  # NEW
        logger.info(f"  Reference ranges: {len(self.reference_ranges)}")
    
    def validate_loinc_code(self, code: str) -> bool:
        """Check if LOINC code is valid."""
        return code in self.loinc_codes.values()
    
    def validate_snomed_code(self, code: str) -> bool:
        """Check if SNOMED-CT code is valid."""
        return code in self.snomed_codes.values()
    
    def validate_biomarker_value(
        self,
        biomarker: str,
        value: float
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate biomarker value against reference range.
        
        Args:
            biomarker: Biomarker name
            value: Measured value
        
        Returns:
            Tuple of (is_valid, status, message)
        """
        ref_range = self.reference_ranges.get(biomarker.lower())
        
        if not ref_range:
            return True, "unknown", f"No reference range for {biomarker}"
        
        if ref_range.is_critical(value):
            return (
                False,
                "critical",
                f"{biomarker} = {value} {ref_range.unit} is CRITICAL "
                f"(normal: {ref_range.min_value}-{ref_range.max_value})"
            )
        elif not ref_range.is_normal(value):
            return (
                False,
                "abnormal",
                f"{biomarker} = {value} {ref_range.unit} is abnormal "
                f"(normal: {ref_range.min_value}-{ref_range.max_value})"
            )
        else:
            return (
                True,
                "normal",
                f"{biomarker} = {value} {ref_range.unit} is within normal range"
            )
    
    def extract_and_validate_patient_state(
        self,
        patient_state: str
    ) -> Dict[str, Dict]:
        """
        Extract and validate all biomarkers from patient state stream.
        
        Args:
            patient_state: <patient_state> stream content
        
        Returns:
            Dictionary with biomarker validations
        """
        results = {}
        
        # Pattern: "NAME: VALUE UNIT [LOINC:CODE]"
        pattern = r'(\w+):\s*([\d.]+)\s*([a-zA-Z/%°]+)\s*\[LOINC:([\d-]+)\]'
        matches = re.findall(pattern, patient_state)
        
        for biomarker, value_str, unit, loinc_code in matches:
            try:
                value = float(value_str)
                
                # Validate LOINC code
                loinc_valid = self.validate_loinc_code(loinc_code)
                
                # Validate value
                is_valid, status, message = self.validate_biomarker_value(
                    biomarker,
                    value
                )
                
                results[biomarker] = {
                    'value': value,
                    'unit': unit,
                    'loinc_code': loinc_code,
                    'loinc_valid': loinc_valid,
                    'value_valid': is_valid,
                    'status': status,
                    'message': message
                }
            except ValueError as e:
                logger.warning(f"Could not parse {biomarker}: {e}")
                continue
        
        return results
    
    def generate_validation_report(
        self,
        patient_state: str
    ) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            patient_state: <patient_state> stream content
        
        Returns:
            Formatted validation report
        """
        validations = self.extract_and_validate_patient_state(patient_state)
        
        if not validations:
            return "No biomarkers found to validate"
        
        report = "Biomarker Validation Report:\n"
        report += "=" * 60 + "\n\n"
        
        for biomarker, result in validations.items():
            report += f"{biomarker.upper()}:\n"
            report += f"  Value: {result['value']} {result['unit']}\n"
            report += f"  LOINC: {result['loinc_code']} "
            report += f"({'✓ valid' if result['loinc_valid'] else '✗ invalid'})\n"
            report += f"  Status: {result['status'].upper()}\n"
            report += f"  {result['message']}\n\n"
        
        return report