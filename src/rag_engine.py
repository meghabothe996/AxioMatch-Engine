import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import MTSamplesAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ClinicalRetriever:
    """
    Offline RAG Engine: Converts clinical notes into mathematical vectors 
    to instantly search 5,000+ records for trial criteria matches.
    """
    def __init__(self, data_path: str = "data/mtsamples.csv"):
        logger.info("Initializing Vector Database...")
        
        # 1. Load all valid patients from our existing adapter
        self.adapter = MTSamplesAdapter(file_path=data_path)
        
        # We'll pull a subset (e.g., 500) to keep RAM usage low for the i5, 
        # but in a production server, this could be millions.
        self.patients = self._load_corpus(limit=500)
        
        # 2. Initialize the Mathematical Vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_df=0.85,  # Ignore words that appear in 85% of documents (e.g., "patient")
            min_df=2      # Ignore words that only appear once (typos)
        )
        
        # 3. Build the Vector Index (The RAG Database)
        self.tfidf_matrix = self._build_index()

    def _load_corpus(self, limit: int):
        # We pull from multiple specialties to prove the search works
        df = self.adapter.df.dropna(subset=['transcription'])
        corpus = []
        for idx, row in df.head(limit).iterrows():
            corpus.append({
                "patient_id": f"MTS-{idx}",
                "specialty": row['medical_specialty'].strip(),
                "case_name": row.get('sample_name', 'Unnamed Case').strip(),
                "clinical_note": row['transcription'].strip()
            })
        return corpus

    def _build_index(self):
        logger.info(f"Vectorizing {len(self.patients)} clinical records...")
        texts = [p['clinical_note'] for p in self.patients]
        return self.vectorizer.fit_transform(texts)

    def search_patients(self, trial_criteria: str, top_k: int = 3):
        """
        Takes the trial criteria, converts it to a vector, and finds the closest patients.
        """
        logger.info(f"Executing Vector Search for Top {top_k} matches...")
        
        # Convert the user's trial criteria into the same math space
        query_vector = self.vectorizer.transform([trial_criteria])
        
        # Calculate Cosine Similarity (How close is the query to each patient?)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            score = round(similarities[i] * 100, 2) # Convert to a percentage
            patient = self.patients[i]
            patient['search_score'] = score
            results.append(patient)
            
        return results

# --- TEST BLOCK ---
if __name__ == "__main__":
    retriever = ClinicalRetriever()
    
    test_criteria = "Looking for patients with breast cancer who need chemotherapy."
    print(f"\nQUERY: '{test_criteria}'\n")
    
    matches = retriever.search_patients(test_criteria, top_k=2)
    
    for match in matches:
        print(f"[{match['search_score']}% Match] {match['patient_id']} - {match['specialty']}")
        print(f"Preview: {match['clinical_note'][:100]}...\n")