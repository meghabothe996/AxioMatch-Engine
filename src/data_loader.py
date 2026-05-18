import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 1. Configure RSE-Standard Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MTSamplesAdapter:
    """
    Adapter to load, sanitize, and preprocess clinical notes from the MTSamples dataset.
    This replaces the MIMIC-III loader to ensure high-fidelity, real-world clinical text.
    """
    
    def __init__(self, file_path: str = "data/mtsamples.csv"):
        # Use pathlib for cross-platform robustness (Windows/Mac/Linux)
        self.file_path = Path(file_path)
        self.df = self._load_data()

    def _load_data(self) -> Optional[pd.DataFrame]:
        """Safely loads the CSV and standardizes column formats (Private Method)."""
        if not self.file_path.exists():
            logger.error(f"Dataset not found at {self.file_path}. Please ensure it is in the data folder.")
            return None
        
        try:
            df = pd.read_csv(self.file_path)
            # Standardize column names: lowercase, strip whitespace
            df.columns = df.columns.str.lower().str.strip()
            logger.info(f"Successfully ingested {len(df)} raw records from MTSamples.")
            return df
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            return None

    def get_specialty_notes(self, specialty: str = "Oncology", limit: int = 5) -> List[Dict]:
        """
        Retrieves valid clinical notes filtered by a specific medical specialty.
        """
        if self.df is None or self.df.empty:
            logger.warning("No data available to query.")
            return []

        # 2. Data Validation & Filtering
        # We must ensure the transcription actually exists and isn't just blank space
        filtered_df = self.df[
            (self.df['medical_specialty'].str.contains(specialty, case=False, na=False)) &
            (self.df['transcription'].notna()) & 
            (self.df['transcription'].str.strip() != "")
        ]

        if filtered_df.empty:
            logger.warning(f"No valid notes found for specialty query: '{specialty}'")
            return []

        # 3. Structure the Output payload
        results = []
        for idx, row in filtered_df.head(limit).iterrows():
            results.append({
                "patient_id": f"MTS-{idx}",  # Generate a pseudo-ID for tracking
                "specialty": row['medical_specialty'].strip(),
                "case_name": row.get('sample_name', 'Unnamed Case').strip(),
                "clinical_note": row['transcription'].strip()
            })
        
        logger.info(f"Successfully processed {len(results)} notes for downstream LLM inference.")
        return results

# ---------------------------------------------------------
# Execution Block (Only runs if you execute this file directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*45)
    print(" AXIOMATCH ENGINE: DATA PIPELINE INITIALIZED")
    print("="*45)
    
    adapter = MTSamplesAdapter()
    
    # Test pulling 1 real Oncology note
    oncology_notes = adapter.get_specialty_notes("Oncology", limit=1)
    
    if oncology_notes:
        sample = oncology_notes[0]
        print("\n--- SAMPLE PAYLOAD FOR GEMMA ---")
        print(f"Patient ID:  {sample['patient_id']}")
        print(f"Specialty:   {sample['specialty']}")
        print(f"Case Title:  {sample['case_name']}")
        print(f"Note Snippet:\n{sample['clinical_note'][:300]}...\n")
        print("STATUS: PIPELINE READY FOR INFERENCE")
    else:
        print("\nSTATUS: PIPELINE FAILED. Check dataset.")