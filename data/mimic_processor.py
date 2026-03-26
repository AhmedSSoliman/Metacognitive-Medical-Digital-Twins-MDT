"""
MIMIC-IV Data Processor - Memory Optimized.

Processes MIMIC-IV Critical Care Database v3.1 for Medical Digital Twin training.
Uses chunked reading to handle large files without exceeding memory limits.

Supports both compressed (.csv.gz) and uncompressed (.csv) file formats.

Author: Ahmed Soliman
Institution: University of Florida, HOBI
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

from config.configs import DataConfig
from config.ontology import (
    LOINCCodes,
    MIMICItemIDs,
    ClinicalReferenceRanges
)

logger = logging.getLogger(__name__)


class MIMICProcessor:
    """
    MIMIC-IV v3.1 data processor for critical care trajectories.
    
    Memory-efficient implementation using chunked file reading.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize MIMIC processor.
        
        Args:
            config: Data configuration with MIMIC paths and settings
        """
        self.config = config
        self.root_dir = Path(config.mimic_root_dir)
        
        # Use ontology from config/ontology.py
        self.item_ids = MIMICItemIDs.CHART_ITEMS
        self.lab_ids = MIMICItemIDs.LAB_ITEMS
        self.loinc_codes = LOINCCodes.get_all_codes()
        self.reference_ranges = ClinicalReferenceRanges.RANGES
        
        logger.info(f"Initialized MIMIC-IV v{config.mimic_version} processor")
        logger.info(f"Root directory: {self.root_dir}")
        logger.info(f"Loaded {len(self.loinc_codes)} LOINC codes from ontology")
        logger.info(f"Loaded {len(self.item_ids)} chartevents item categories")
        logger.info(f"Loaded {len(self.reference_ranges)} reference ranges")
    
    
        
        # MIMIC-IV item IDs for vital signs and labs
        self.item_ids = {
            'heart_rate': [220045],
            'sbp': [220050, 220179],  # Non-invasive and invasive
            'dbp': [220051, 220180],
            'spo2': [220277],
            'temperature': [223761, 223762],  # F and C
            'respiratory_rate': [220210, 224690],
            'lactate': [50813],
            'creatinine': [50912],
            'wbc': [51301],
            'hemoglobin': [51222],
            'platelets': [51265]
        }
    
    def _find_file(self, relative_path: str) -> Optional[Path]:
        """
        Find file with either .csv or .csv.gz extension.
        
        Args:
            relative_path: Path relative to root (e.g., 'icu/icustays.csv.gz')
        
        Returns:
            Full path to existing file, or None if not found
        """
        # Try with .gz extension first
        gz_path = self.root_dir / relative_path
        if gz_path.exists():
            return gz_path
        
        # Try without .gz extension (uncompressed)
        if relative_path.endswith('.gz'):
            uncompressed_path = self.root_dir / relative_path[:-3]  # Remove '.gz'
            if uncompressed_path.exists():
                return uncompressed_path
        
        return None
    
    def check_availability(self) -> bool:
        """
        Check if required MIMIC-IV files are accessible.
        
        Returns:
            True if all required files exist (compressed or uncompressed)
        """
        required_files = [
            'icu/icustays.csv.gz',
            'icu/chartevents.csv.gz',
            'hosp/patients.csv.gz',
            'hosp/admissions.csv.gz'
        ]
        
        for file_path in required_files:
            found_path = self._find_file(file_path)
            if found_path is None:
                logger.error(f"Required file not found: {file_path}")
                logger.error(f"Searched for both .csv.gz and .csv formats")
                return False
        
        logger.info("All required MIMIC-IV files found")
        return True
    
    def load_icustays(self) -> pd.DataFrame:
        """
        Load ICU stays table.
        
        Returns:
            DataFrame with ICU stay information
        """
        file_path = self._find_file('icu/icustays.csv.gz')
        if file_path is None:
            raise FileNotFoundError("icustays file not found")
        
        logger.info(f"Loading ICU stays from: {file_path}")
        
        # ICU stays is small enough to load entirely
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} ICU stays")
        
        return df
    
    def load_chartevents_for_stays(
        self,
        stay_ids: List[int],
        item_ids: List[int],
        chunk_size: int = 1000000
    ) -> pd.DataFrame:
        """
        Load chart events for specific stays using chunked reading.
        
        MEMORY EFFICIENT: Reads large file in chunks to avoid OOM errors.
        
        Args:
            stay_ids: ICU stay IDs to extract
            item_ids: Item IDs to extract
            chunk_size: Number of rows per chunk
        
        Returns:
            DataFrame with filtered chart events
        """
        file_path = self._find_file('icu/chartevents.csv.gz')
        if file_path is None:
            raise FileNotFoundError("chartevents file not found")
        
        logger.info(f"Loading chart events from: {file_path}")
        logger.info(f"Reading in chunks of {chunk_size:,} rows...")
        
        # Convert to sets for faster lookup
        stay_ids_set = set(stay_ids)
        item_ids_set = set(item_ids)
        
        filtered_chunks = []
        total_rows_read = 0
        total_rows_kept = 0
        
        # Read file in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            total_rows_read += len(chunk)
            
            # Filter chunk
            filtered = chunk[
                chunk['stay_id'].isin(stay_ids_set) &
                chunk['itemid'].isin(item_ids_set)
            ]
            
            if len(filtered) > 0:
                filtered_chunks.append(filtered)
                total_rows_kept += len(filtered)
            
            # Progress logging every 10M rows
            if (chunk_num + 1) % 10 == 0:
                logger.info(
                    f"  Processed {total_rows_read:,} rows, "
                    f"kept {total_rows_kept:,} relevant measurements"
                )
            
            # Early exit if we have enough data
            if total_rows_kept > 100000:  # Reasonable amount for training
                logger.info(f"  Collected sufficient data, stopping early")
                break
        
        # Combine filtered chunks
        if filtered_chunks:
            result = pd.concat(filtered_chunks, ignore_index=True)
            logger.info(
                f"✓ Loaded {len(result):,} chart events "
                f"from {total_rows_read:,} total rows"
            )
            return result
        else:
            logger.warning("No chart events found for specified stays/items")
            return pd.DataFrame()
    
    def load_patients(self) -> pd.DataFrame:
        """
        Load patients table.
        
        Returns:
            DataFrame with patient demographics
        """
        file_path = self._find_file('hosp/patients.csv.gz')
        if file_path is None:
            raise FileNotFoundError("patients file not found")
        
        logger.info(f"Loading patients from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} patients")
        
        return df
    
    def load_admissions(self) -> pd.DataFrame:
        """
        Load admissions table.
        
        Returns:
            DataFrame with admission information
        """
        file_path = self._find_file('hosp/admissions.csv.gz')
        if file_path is None:
            raise FileNotFoundError("admissions file not found")
        
        logger.info(f"Loading admissions from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} admissions")
        
        return df
    
    def extract_vital_signs(
        self,
        stay_id: int,
        chartevents: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract vital signs for a specific ICU stay.
        
        Args:
            stay_id: ICU stay identifier
            chartevents: Chart events DataFrame (pre-filtered)
        
        Returns:
            DataFrame with vital signs time series
        """
        # Filter for this stay
        stay_vitals = chartevents[chartevents['stay_id'] == stay_id].copy()
        
        # Convert charttime to datetime
        stay_vitals['charttime'] = pd.to_datetime(stay_vitals['charttime'])
        
        # Sort by time
        stay_vitals = stay_vitals.sort_values('charttime')
        
        return stay_vitals
    
    def process_single_patient(
        self,
        stay_id: int,
        chartevents: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Process single ICU stay into training example.
        
        Args:
            stay_id: ICU stay identifier
            chartevents: Chart events DataFrame (pre-filtered)
        
        Returns:
            Dictionary with training example or None if insufficient data
        """
        # Extract vital signs
        vitals = self.extract_vital_signs(stay_id, chartevents)
        
        if len(vitals) < 10:  # Need minimum number of measurements
            return None
        
        # Create temporal snapshot
        # Use middle of stay for demo
        mid_idx = len(vitals) // 2
        window_size = min(5, mid_idx, len(vitals) - mid_idx)
        current_vitals = vitals.iloc[mid_idx-window_size:mid_idx+window_size]
        
        # Format as training example
        example = {
            'stay_id': int(stay_id),
            'prompt': self._create_prompt(current_vitals),
            'think': self._create_think_stream(current_vitals),
            'patient_state': self._create_patient_state_stream(current_vitals),
            'user_belief': self._create_user_belief_stream()
        }
        
        return example
    
    def _create_prompt(self, vitals: pd.DataFrame) -> str:
        """Create clinical prompt from vitals."""
        # Extract key values
        hr_rows = vitals[vitals['itemid'].isin(self.item_ids['heart_rate'])]
        sbp_rows = vitals[vitals['itemid'].isin(self.item_ids['sbp'])]
        temp_rows = vitals[vitals['itemid'].isin(self.item_ids['temperature'])]
        
        hr = hr_rows['valuenum'].mean() if len(hr_rows) > 0 else None
        sbp = sbp_rows['valuenum'].mean() if len(sbp_rows) > 0 else None
        temp = temp_rows['valuenum'].mean() if len(temp_rows) > 0 else None
        
        prompt = "ICU Patient Assessment Required:\n\n"
        prompt += "Current Vital Signs:\n"
        
        if hr:
            prompt += f"- Heart Rate: {hr:.0f} bpm\n"
        if sbp:
            prompt += f"- Systolic Blood Pressure: {sbp:.0f} mmHg\n"
        if temp:
            prompt += f"- Temperature: {temp:.1f}°F\n"
        
        prompt += "\nProvide clinical assessment with reasoning, patient state, and communication strategy."
        
        return prompt
    
    def _create_think_stream(self, vitals: pd.DataFrame) -> str:
        """Create <think> stream from vitals."""
        hr_rows = vitals[vitals['itemid'].isin(self.item_ids['heart_rate'])]
        sbp_rows = vitals[vitals['itemid'].isin(self.item_ids['sbp'])]
        temp_rows = vitals[vitals['itemid'].isin(self.item_ids['temperature'])]
        
        hr = hr_rows['valuenum'].mean() if len(hr_rows) > 0 else 80
        sbp = sbp_rows['valuenum'].mean() if len(sbp_rows) > 0 else 120
        temp = temp_rows['valuenum'].mean() if len(temp_rows) > 0 else 98.6
        
        reasoning = "Clinical Assessment: "
        
        if hr > 100:
            reasoning += "Patient demonstrates tachycardia with heart rate elevated above normal limits. "
        else:
            reasoning += "Heart rate within normal limits. "
        
        if sbp < 90:
            reasoning += "Concerning hypotension noted requiring urgent intervention. "
        elif sbp > 140:
            reasoning += "Elevated blood pressure requiring monitoring and possible treatment. "
        else:
            reasoning += "Blood pressure stable within acceptable range. "
        
        if temp > 100.4:
            reasoning += "Fever present suggesting possible infectious process or inflammatory response requiring further investigation."
        else:
            reasoning += "Temperature within normal limits suggesting no acute febrile illness at this time."
        
        return reasoning
    
    def _create_patient_state_stream(self, vitals: pd.DataFrame) -> str:
        """Create <patient_state> stream with LOINC codes."""
        state = []
        
        # Heart rate
        hr_rows = vitals[vitals['itemid'].isin(self.item_ids['heart_rate'])]
        if len(hr_rows) > 0:
            hr = hr_rows['valuenum'].mean()
            state.append(f"HR: {hr:.0f} bpm [LOINC:{self.config.loinc_codes['heart_rate']}]")
        
        # Blood pressure
        sbp_rows = vitals[vitals['itemid'].isin(self.item_ids['sbp'])]
        dbp_rows = vitals[vitals['itemid'].isin(self.item_ids['dbp'])]
        if len(sbp_rows) > 0:
            sbp = sbp_rows['valuenum'].mean()
            if len(dbp_rows) > 0:
                dbp = dbp_rows['valuenum'].mean()
                state.append(f"BP: {sbp:.0f}/{dbp:.0f} mmHg [LOINC:{self.config.loinc_codes['sbp']}]")
            else:
                state.append(f"SBP: {sbp:.0f} mmHg [LOINC:{self.config.loinc_codes['sbp']}]")
        
        # Temperature
        temp_rows = vitals[vitals['itemid'].isin(self.item_ids['temperature'])]
        if len(temp_rows) > 0:
            temp = temp_rows['valuenum'].mean()
            state.append(f"Temp: {temp:.1f}°F [LOINC:{self.config.loinc_codes['temperature']}]")
        
        # SpO2
        spo2_rows = vitals[vitals['itemid'].isin(self.item_ids['spo2'])]
        if len(spo2_rows) > 0:
            spo2 = spo2_rows['valuenum'].mean()
            state.append(f"SpO2: {spo2:.0f}% [LOINC:{self.config.loinc_codes['spo2']}]")
        
        return ", ".join(state) if state else "Vital signs pending"
    
    def _create_user_belief_stream(self) -> str:
        """Create <user_belief> stream."""
        return "Literacy: medium, Emotional state: concerned about patient condition, Medical background: limited layperson knowledge"
    
    def process_all_patients(
        self,
        max_patients: int = 1000,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple patients into training dataset.
        
        MEMORY EFFICIENT: Uses chunked reading for large files.
        
        Args:
            max_patients: Maximum number of patients to process
            save_path: Optional path to save processed data
        
        Returns:
            List of training examples
        """
        logger.info(f"Processing up to {max_patients} patients...")
        
        # Load base tables
        icustays = self.load_icustays()
        
        # Filter to valid stays (at least min_icu_stay_hours)
        icustays = icustays[icustays['los'] >= self.config.min_icu_stay_hours / 24]
        icustays = icustays.head(max_patients)
        
        logger.info(f"Selected {len(icustays)} ICU stays")
        
        # Get all vital sign item IDs we need
        vital_item_ids = []
        for vital_name in ['heart_rate', 'sbp', 'dbp', 'spo2', 'temperature', 'respiratory_rate']:
            vital_item_ids.extend(self.item_ids[vital_name])
        
        # Get stay IDs
        stay_ids = icustays['stay_id'].tolist()
        
        logger.info(f"Extracting vital signs for {len(stay_ids)} stays...")
        logger.info(f"This may take several minutes for large datasets...")
        
        # Load chart events using chunked reading (MEMORY EFFICIENT)
        chartevents = self.load_chartevents_for_stays(
            stay_ids=stay_ids,
            item_ids=vital_item_ids,
            chunk_size=1000000  # 1M rows per chunk
        )
        
        if len(chartevents) == 0:
            logger.error("No chart events found for selected stays")
            return []
        
        # Process each patient
        training_examples = []
        
        logger.info(f"Creating training examples...")
        for idx, stay_id in enumerate(stay_ids):
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processing stay {idx+1}/{len(stay_ids)}")
            
            example = self.process_single_patient(stay_id, chartevents)
            
            if example:
                training_examples.append(example)
        
        logger.info(f"✓ Created {len(training_examples)} training examples from {len(stay_ids)} stays")
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(training_examples, f, indent=2)
            logger.info(f"✓ Saved training examples to: {save_path}")
        
        return training_examples