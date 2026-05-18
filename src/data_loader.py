import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MTSamplesAdapter:
    """
    Adapter to load, sanitize, and preprocess clinical notes from the MTSamples dataset.
    This replaces the MIMIC-III loader to ensure high-fidelity, real-world clinical text.
    """
    
    def __init__(self, file_path: str = "data/mtsamples.csv"):
        self.file_path = Path(file_path)
        self.df = self._load_data()

    def _load_data(self) -> Optional[pd.DataFrame]:
        if not self.file_path.exists():
            logger.error(f"Dataset not found at {self.file_path}. Please ensure it is in the data folder.")
            return None
        
        try:
            df = pd.read_csv(self.file_path)
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

        filtered_df = self.df[
            (self.df['medical_specialty'].str.contains(specialty, case=False, na=False)) &
            (self.df['transcription'].notna()) & 
            (self.df['transcription'].str.strip() != "")
        ]

        if filtered_df.empty:
            logger.warning(f"No valid notes found for specialty query: '{specialty}'")
            return []

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

if __name__ == "__main__":
    adapter = MTSamplesAdapter()
    notes = adapter.get_specialty_notes("Oncology", limit=1)
    if notes:
        logger.info(f"Test successful. Loaded: {notes[0]['patient_id']}")