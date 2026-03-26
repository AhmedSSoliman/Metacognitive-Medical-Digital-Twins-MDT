"""
Clinical Ontology and Terminology Standards.

Contains standardized medical codes (LOINC, SNOMED-CT) and reference ranges
for clinical biomarkers, diagnoses, and procedures.

Standards:
    - LOINC: Logical Observation Identifiers Names and Codes
    - SNOMED-CT: Systematized Nomenclature of Medicine - Clinical Terms
    - MIMIC-IV: Item ID mappings for chartevents and labevents

Author: Ahmed Soliman
Institution: University of Florida, HOBI
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ============================================================================
# LOINC CODE MAPPINGS
# ============================================================================

class LOINCCodes:
    """LOINC codes for common clinical observations."""
    
    # Vital Signs
    VITAL_SIGNS = {
        'heart_rate': '8867-4',
        'sbp': '8480-6',
        'dbp': '8462-4',
        'mean_arterial_pressure': '8478-0',
        'spo2': '59408-5',
        'temperature_oral': '8310-5',
        'temperature_core': '8329-5',
        'respiratory_rate': '9279-1',
        'weight': '29463-7',
        'height': '8302-2',
        'bmi': '39156-5'
    }
    
    # Chemistry Panel
    CHEMISTRY = {
        'lactate': '2524-7',
        'creatinine': '2160-0',
        'glucose': '2345-7',
        'sodium': '2951-2',
        'potassium': '2823-3',
        'chloride': '2075-0',
        'bicarbonate': '2028-9',
        'bun': '3094-0',
        'calcium': '17861-6',
        'magnesium': '2601-3',
        'phosphate': '2777-1'
    }
    
    # Complete Blood Count (CBC)
    HEMATOLOGY = {
        'wbc': '6690-2',
        'hemoglobin': '718-7',
        'hematocrit': '4544-3',
        'platelets': '777-3',
        'rbc': '789-8',
        'neutrophils': '751-8',
        'lymphocytes': '731-0',
        'monocytes': '742-7',
        'eosinophils': '711-2',
        'basophils': '704-7'
    }
    
    # Coagulation
    COAGULATION = {
        'inr': '6301-6',
        'pt': '5902-2',
        'ptt': '3173-2',
        'fibrinogen': '3255-7',
        'd_dimer': '48065-7'
    }
    
    # Liver Function Tests (LFTs)
    LIVER = {
        'alt': '1742-6',
        'ast': '1920-8',
        'bilirubin_total': '1975-2',
        'bilirubin_direct': '1968-7',
        'albumin': '1751-7',
        'alkaline_phosphatase': '6768-6',
        'ggt': '2324-2'
    }
    
    # Arterial Blood Gas (ABG)
    BLOOD_GAS = {
        'ph': '2744-1',
        'pco2': '2019-8',
        'po2': '2703-7',
        'base_excess': '1925-7',
        'bicarbonate_abg': '1960-4',
        'lactate_abg': '2524-7'
    }
    
    # Cardiac Markers
    CARDIAC = {
        'troponin_i': '10839-9',
        'troponin_t': '6598-7',
        'bnp': '30934-4',
        'nt_probnp': '33762-6',
        'ck_mb': '13969-1'
    }
    
    # Inflammatory Markers
    INFLAMMATORY = {
        'crp': '1988-5',
        'esr': '4537-7',
        'procalcitonin': '33959-8',
        'ferritin': '2276-4'
    }
    
    # Renal Function
    RENAL = {
        'creatinine': '2160-0',
        'bun': '3094-0',
        'egfr': '33914-3',
        'urine_output': '9187-6'
    }
    
    @classmethod
    def get_all_codes(cls) -> Dict[str, str]:
        """Get all LOINC codes as flat dictionary."""
        all_codes = {}
        for category in [
            cls.VITAL_SIGNS, cls.CHEMISTRY, cls.HEMATOLOGY,
            cls.COAGULATION, cls.LIVER, cls.BLOOD_GAS,
            cls.CARDIAC, cls.INFLAMMATORY, cls.RENAL
        ]:
            all_codes.update(category)
        return all_codes
    
    @classmethod
    def get_code_by_name(cls, name: str) -> Optional[str]:
        """Get LOINC code by biomarker name."""
        all_codes = cls.get_all_codes()
        return all_codes.get(name.lower())


# ============================================================================
# SNOMED-CT CODE MAPPINGS
# ============================================================================

class SNOMEDCodes:
    """SNOMED-CT codes for clinical conditions and procedures."""
    
    # Common ICU Diagnoses
    DIAGNOSES = {
        'sepsis': '91302008',
        'septic_shock': '76571007',
        'pneumonia': '233604007',
        'ards': '67782005',
        'acute_kidney_injury': '14669001',
        'heart_failure': '84114007',
        'myocardial_infarction': '22298006',
        'stroke': '230690007',
        'pulmonary_embolism': '59282003',
        'diabetic_ketoacidosis': '420422005',
        'gastrointestinal_bleeding': '74474003',
        'acute_pancreatitis': '197456007',
        'copd_exacerbation': '195967001',
        'asthma_exacerbation': '195967001',
        'cardiogenic_shock': '89052004',
        'hypovolemic_shock': '39419009',
        'anaphylactic_shock': '39579001',
        'respiratory_failure': '65710008',
        'liver_failure': '59927004',
        'multi_organ_failure': '57653000'
    }
    
    # Procedures
    PROCEDURES = {
        'mechanical_ventilation': '40617009',
        'central_venous_catheter': '392248005',
        'arterial_line': '392247000',
        'hemodialysis': '265764009',
        'continuous_renal_replacement': '714749008',
        'vasopressor_therapy': '225368008',
        'blood_transfusion': '116859006',
        'intubation': '112798008',
        'extubation': '271280005',
        'tracheostomy': '48387007',
        'chest_tube_insertion': '48387007',
        'paracentesis': '277762005',
        'thoracentesis': '91602002'
    }
    
    # Clinical Findings
    FINDINGS = {
        'hypotension': '45007003',
        'hypertension': '38341003',
        'tachycardia': '3424008',
        'bradycardia': '48867003',
        'fever': '386661006',
        'hypothermia': '95281009',
        'hypoxemia': '389086002',
        'hyperglycemia': '80394007',
        'hypoglycemia': '302866003',
        'metabolic_acidosis': '441742003',
        'respiratory_acidosis': '51387008',
        'metabolic_alkalosis': '447121000124109',
        'respiratory_alkalosis': '49899001',
        'hyperkalemia': '14140009',
        'hypokalemia': '43339004',
        'hypernatremia': '33889001',
        'hyponatremia': '89627008'
    }
    
    @classmethod
    def get_all_codes(cls) -> Dict[str, str]:
        """Get all SNOMED-CT codes as flat dictionary."""
        all_codes = {}
        for category in [cls.DIAGNOSES, cls.PROCEDURES, cls.FINDINGS]:
            all_codes.update(category)
        return all_codes
    
    @classmethod
    def get_code_by_name(cls, name: str) -> Optional[str]:
        """Get SNOMED-CT code by condition/procedure name."""
        all_codes = cls.get_all_codes()
        return all_codes.get(name.lower())


class ICD10Codes:
    """
    ICD-10-CM (Clinical Modification) codes for common diagnoses.
    
    Provides standardized diagnostic codes for clinical documentation
    and billing. Used alongside SNOMED-CT for comprehensive diagnosis coding.
    
    Categories:
        - Infectious diseases
        - Cardiovascular diseases
        - Respiratory diseases
        - Endocrine/metabolic diseases
        - Neurological diseases
        - Renal diseases
        - Cancer/oncology
        - Mental health
        
    Example:
        >>> sepsis_code = ICD10Codes.get_code_by_name('sepsis')
        >>> print(sepsis_code)  # 'A41.9'
    """
    
    # Infectious Diseases
    INFECTIOUS = {
        'sepsis': 'A41.9',                              # Sepsis, unspecified
        'septic_shock': 'R65.21',                       # Severe sepsis with septic shock
        'pneumonia': 'J18.9',                           # Pneumonia, unspecified
        'covid19': 'U07.1',                             # COVID-19
        'influenza': 'J11.1',                           # Influenza with respiratory manifestations
        'uti': 'N39.0',                                 # Urinary tract infection
        'clostridium_difficile': 'A04.7',               # C. difficile colitis
    }
    
    # Cardiovascular Diseases
    CARDIOVASCULAR = {
        'acute_mi': 'I21.9',                            # Acute myocardial infarction
        'heart_failure': 'I50.9',                       # Heart failure, unspecified
        'atrial_fibrillation': 'I48.91',                # Atrial fibrillation
        'hypertension': 'I10',                          # Essential hypertension
        'cardiogenic_shock': 'R57.0',                   # Cardiogenic shock
        'acute_chf': 'I50.21',                          # Acute systolic heart failure
        'ventricular_tachycardia': 'I47.2',             # Ventricular tachycardia
    }
    
    # Respiratory Diseases
    RESPIRATORY = {
        'ards': 'J80',                                  # Acute respiratory distress syndrome
        'respiratory_failure': 'J96.00',                # Acute respiratory failure
        'copd': 'J44.9',                                # COPD, unspecified
        'asthma': 'J45.909',                            # Asthma, unspecified
        'pulmonary_embolism': 'I26.99',                 # Pulmonary embolism
        'pleural_effusion': 'J90',                      # Pleural effusion
    }
    
    # Endocrine/Metabolic
    ENDOCRINE_METABOLIC = {
        'diabetes_type2': 'E11.9',                      # Type 2 diabetes
        'diabetes_type1': 'E10.9',                      # Type 1 diabetes
        'diabetic_ketoacidosis': 'E10.10',              # DKA
        'hypoglycemia': 'E16.2',                        # Hypoglycemia, unspecified
        'hypothyroidism': 'E03.9',                      # Hypothyroidism
        'hyperthyroidism': 'E05.90',                    # Hyperthyroidism
    }
    
    # Neurological
    NEUROLOGICAL = {
        'stroke_ischemic': 'I63.9',                     # Ischemic stroke
        'stroke_hemorrhagic': 'I61.9',                  # Hemorrhagic stroke
        'seizure': 'R56.9',                             # Seizure, unspecified
        'altered_mental_status': 'R41.82',              # Altered mental status
        'encephalopathy': 'G93.40',                     # Encephalopathy
        'alzheimers': 'G30.9',                          # Alzheimer's disease
        'parkinsons': 'G20',                            # Parkinson's disease
        'dementia': 'F03.90',                           # Dementia, unspecified
    }
    
    # Renal
    RENAL = {
        'aki': 'N17.9',                                 # Acute kidney injury
        'ckd_stage3': 'N18.3',                          # CKD stage 3
        'ckd_stage4': 'N18.4',                          # CKD stage 4
        'ckd_stage5': 'N18.5',                          # CKD stage 5
        'esrd': 'N18.6',                                # End-stage renal disease
    }
    
    # Oncology
    ONCOLOGY = {
        'lung_cancer': 'C34.90',                        # Lung cancer
        'breast_cancer': 'C50.919',                     # Breast cancer
        'colon_cancer': 'C18.9',                        # Colon cancer
        'prostate_cancer': 'C61',                       # Prostate cancer
        'lymphoma': 'C85.90',                           # Non-Hodgkin lymphoma
    }
    
    # Mental Health
    MENTAL_HEALTH = {
        'depression': 'F32.9',                          # Major depressive disorder
        'anxiety': 'F41.9',                             # Anxiety disorder
        'delirium': 'F05',                              # Delirium
        'substance_abuse': 'F19.10',                    # Substance use disorder
    }
    
    # Gastrointestinal
    GASTROINTESTINAL = {
        'gi_bleeding': 'K92.2',                         # GI hemorrhage
        'pancreatitis': 'K85.90',                       # Acute pancreatitis
        'cirrhosis': 'K74.60',                          # Cirrhosis
        'hepatic_encephalopathy': 'K72.90',             # Hepatic failure
    }
    
    @classmethod
    def get_all_codes(cls) -> Dict[str, str]:
        """
        Get all ICD-10 codes.
        
        Returns:
            Dictionary mapping condition names to ICD-10 codes
        """
        all_codes = {}
        for category in [
            cls.INFECTIOUS, cls.CARDIOVASCULAR, cls.RESPIRATORY,
            cls.ENDOCRINE_METABOLIC, cls.NEUROLOGICAL, cls.RENAL,
            cls.ONCOLOGY, cls.MENTAL_HEALTH, cls.GASTROINTESTINAL
        ]:
            all_codes.update(category)
        return all_codes
    
    @classmethod
    def get_code_by_name(cls, name: str) -> Optional[str]:
        """
        Get ICD-10 code by condition name.
        
        Args:
            name: Condition name (e.g., 'sepsis', 'heart_failure')
        
        Returns:
            ICD-10 code or None if not found
        
        Example:
            >>> ICD10Codes.get_code_by_name('sepsis')
            'A41.9'
        """
        all_codes = cls.get_all_codes()
        return all_codes.get(name.lower())
    
    @classmethod
    def get_category(cls, code: str) -> Optional[str]:
        """
        Get category for an ICD-10 code.
        
        Args:
            code: ICD-10 code (e.g., 'A41.9')
        
        Returns:
            Category name or None
        """
        categories = {
            'INFECTIOUS': cls.INFECTIOUS,
            'CARDIOVASCULAR': cls.CARDIOVASCULAR,
            'RESPIRATORY': cls.RESPIRATORY,
            'ENDOCRINE_METABOLIC': cls.ENDOCRINE_METABOLIC,
            'NEUROLOGICAL': cls.NEUROLOGICAL,
            'RENAL': cls.RENAL,
            'ONCOLOGY': cls.ONCOLOGY,
            'MENTAL_HEALTH': cls.MENTAL_HEALTH,
            'GASTROINTESTINAL': cls.GASTROINTESTINAL,
        }
        
        for category_name, codes in categories.items():
            if code in codes.values():
                return category_name
        
        return None

# ============================================================================
# REFERENCE RANGES
# ============================================================================

@dataclass
class ReferenceRange:
    """Clinical reference range for a biomarker."""
    min_value: float
    max_value: float
    unit: str
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    
    def is_normal(self, value: float) -> bool:
        """Check if value is within normal range."""
        return self.min_value <= value <= self.max_value
    
    def is_critical(self, value: float) -> bool:
        """Check if value is in critical range."""
        if self.critical_low and value < self.critical_low:
            return True
        if self.critical_high and value > self.critical_high:
            return True
        return False
    
    def get_status(self, value: float) -> str:
        """Get status string for value."""
        if self.is_critical(value):
            return "CRITICAL"
        elif not self.is_normal(value):
            return "ABNORMAL"
        else:
            return "NORMAL"


class ClinicalReferenceRanges:
    """Reference ranges for common clinical biomarkers."""
    
    RANGES = {
        # Vital Signs
        'heart_rate': ReferenceRange(60, 100, 'bpm', critical_low=40, critical_high=140),
        'sbp': ReferenceRange(90, 140, 'mmHg', critical_low=70, critical_high=180),
        'dbp': ReferenceRange(60, 90, 'mmHg', critical_low=40, critical_high=110),
        'mean_arterial_pressure': ReferenceRange(70, 100, 'mmHg', critical_low=60, critical_high=120),
        'spo2': ReferenceRange(95, 100, '%', critical_low=88, critical_high=None),
        'temperature': ReferenceRange(36.1, 37.2, '°C', critical_low=35.0, critical_high=40.0),
        'respiratory_rate': ReferenceRange(12, 20, 'breaths/min', critical_low=8, critical_high=30),
        
        # Chemistry
        'lactate': ReferenceRange(0.5, 2.0, 'mmol/L', critical_low=None, critical_high=4.0),
        'creatinine': ReferenceRange(0.7, 1.3, 'mg/dL', critical_low=None, critical_high=3.0),
        'glucose': ReferenceRange(70, 110, 'mg/dL', critical_low=50, critical_high=400),
        'sodium': ReferenceRange(135, 145, 'mmol/L', critical_low=120, critical_high=160),
        'potassium': ReferenceRange(3.5, 5.0, 'mmol/L', critical_low=2.5, critical_high=6.5),
        'chloride': ReferenceRange(96, 106, 'mmol/L', critical_low=80, critical_high=120),
        'bicarbonate': ReferenceRange(22, 29, 'mmol/L', critical_low=15, critical_high=35),
        'bun': ReferenceRange(7, 20, 'mg/dL', critical_low=None, critical_high=100),
        'calcium': ReferenceRange(8.5, 10.5, 'mg/dL', critical_low=6.5, critical_high=13.0),
        'magnesium': ReferenceRange(1.7, 2.2, 'mg/dL', critical_low=1.0, critical_high=4.0),
        
        # Hematology
        'wbc': ReferenceRange(4.0, 11.0, '10^9/L', critical_low=1.0, critical_high=30.0),
        'hemoglobin': ReferenceRange(12.0, 17.0, 'g/dL', critical_low=7.0, critical_high=20.0),
        'hematocrit': ReferenceRange(36, 48, '%', critical_low=20, critical_high=60),
        'platelets': ReferenceRange(150, 400, '10^9/L', critical_low=20, critical_high=1000),
        
        # Coagulation
        'inr': ReferenceRange(0.8, 1.2, '', critical_low=None, critical_high=5.0),
        'pt': ReferenceRange(11, 13.5, 'seconds', critical_low=None, critical_high=30),
        'ptt': ReferenceRange(25, 35, 'seconds', critical_low=None, critical_high=100),
        
        # Liver
        'alt': ReferenceRange(7, 56, 'U/L', critical_low=None, critical_high=500),
        'ast': ReferenceRange(10, 40, 'U/L', critical_low=None, critical_high=500),
        'bilirubin_total': ReferenceRange(0.1, 1.2, 'mg/dL', critical_low=None, critical_high=10.0),
        'albumin': ReferenceRange(3.5, 5.5, 'g/dL', critical_low=2.0, critical_high=None),
        
        # Blood Gas
        'ph': ReferenceRange(7.35, 7.45, '', critical_low=7.20, critical_high=7.60),
        'pco2': ReferenceRange(35, 45, 'mmHg', critical_low=20, critical_high=80),
        'po2': ReferenceRange(75, 100, 'mmHg', critical_low=50, critical_high=None),
        'base_excess': ReferenceRange(-2, 2, 'mmol/L', critical_low=-10, critical_high=10),
    }
    
    @classmethod
    def get_range(cls, biomarker: str) -> Optional[ReferenceRange]:
        """Get reference range for biomarker."""
        return cls.RANGES.get(biomarker.lower())
    
    @classmethod
    def validate_value(cls, biomarker: str, value: float) -> Tuple[bool, str]:
        """
        Validate biomarker value against reference range.
        
        Returns:
            Tuple of (is_valid, status_message)
        """
        ref_range = cls.get_range(biomarker)
        if not ref_range:
            return True, "unknown"
        
        if ref_range.is_critical(value):
            return False, "critical"
        elif not ref_range.is_normal(value):
            return False, "abnormal"
        else:
            return True, "normal"


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

class UnitConverter:
    """Convert between different clinical units."""
    
    CONVERSIONS = {
        # Temperature
        ('celsius', 'fahrenheit'): lambda c: c * 9/5 + 32,
        ('fahrenheit', 'celsius'): lambda f: (f - 32) * 5/9,
        
        # Glucose
        ('mg_dl', 'mmol_l'): lambda mg: mg / 18.0,
        ('mmol_l', 'mg_dl'): lambda mmol: mmol * 18.0,
        
        # Creatinine
        ('mg_dl', 'umol_l'): lambda mg: mg * 88.4,
        ('umol_l', 'mg_dl'): lambda umol: umol / 88.4,
        
        # Hemoglobin
        ('g_dl', 'g_l'): lambda g_dl: g_dl * 10,
        ('g_l', 'g_dl'): lambda g_l: g_l / 10,
    }
    
    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units."""
        key = (from_unit, to_unit)
        if key in cls.CONVERSIONS:
            return cls.CONVERSIONS[key](value)
        else:
            raise ValueError(f"No conversion available from {from_unit} to {to_unit}")


# ============================================================================
# MIMIC-IV ITEM ID MAPPINGS
# ============================================================================

class MIMICItemIDs:
    """MIMIC-IV chartevents and labevents item IDs."""
    
    # Chartevents (Vital Signs and Monitoring)
    CHART_ITEMS = {
        'heart_rate': [220045],
        'sbp': [220050, 220179],  # Non-invasive and invasive
        'dbp': [220051, 220180],
        'mean_bp': [220052, 220181],
        'spo2': [220277],
        'temperature': [223761, 223762],  # F and C
        'respiratory_rate': [220210, 224690],
        'fio2': [223835],
        'peep': [220339],
        'tidal_volume': [224685, 224684],
        'minute_ventilation': [224687],
        'gcs_total': [220739],
        'gcs_eye': [220739],
        'gcs_verbal': [223900],
        'gcs_motor': [223901],
        'urine_output': [226559, 226560, 226561, 226584, 226563],
        'central_venous_pressure': [220074],
        'arterial_bp_systolic': [220050],
        'arterial_bp_diastolic': [220051]
    }
    
    # Labevents (Laboratory Values)
    LAB_ITEMS = {
        # Chemistry
        'lactate': [50813],
        'creatinine': [50912],
        'glucose': [50809, 50931],
        'sodium': [50824, 50983],
        'potassium': [50822, 50971],
        'chloride': [50806, 50902],
        'bicarbonate': [50803, 50882],
        'bun': [51006],
        'calcium': [50893],
        'magnesium': [50960],
        'phosphate': [50970],
        
        # Hematology
        'wbc': [51300, 51301],
        'hemoglobin': [51222],
        'hematocrit': [51221],
        'platelets': [51265],
        'rbc': [51279],
        'neutrophils': [51256],
        'lymphocytes': [51244],
        
        # Coagulation
        'inr': [51237],
        'pt': [51274],
        'ptt': [51275],
        
        # Liver
        'alt': [50861],
        'ast': [50878],
        'bilirubin_total': [50885],
        'bilirubin_direct': [50883],
        'albumin': [50862],
        
        # Cardiac
        'troponin_t': [51002],
        'troponin_i': [51003],
        'bnp': [50963],
        
        # Blood Gas
        'ph': [50820],
        'pco2': [50818],
        'po2': [50821],
        'base_excess': [50802]
    }
    
    @classmethod
    def get_all_chart_ids(cls) -> list:
        """Get all chartevents item IDs."""
        all_ids = []
        for ids in cls.CHART_ITEMS.values():
            all_ids.extend(ids)
        return all_ids
    
    @classmethod
    def get_all_lab_ids(cls) -> list:
        """Get all labevents item IDs."""
        all_ids = []
        for ids in cls.LAB_ITEMS.values():
            all_ids.extend(ids)
        return all_ids
    
    @classmethod
    def get_chart_id_by_name(cls, name: str) -> Optional[list]:
        """Get chartevents item IDs by biomarker name."""
        return cls.CHART_ITEMS.get(name.lower())
    
    @classmethod
    def get_lab_id_by_name(cls, name: str) -> Optional[list]:
        """Get labevents item IDs by biomarker name."""
        return cls.LAB_ITEMS.get(name.lower())