import logging
import json
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AxioInferenceEngine:
    def __init__(self, model_path: str, n_ctx: int = 4096):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.llm = self._initialize_model()
        
        self.json_schema = {
            "type": "object",
            "properties": {
                "is_match": {"type": "boolean"},
                "confidence_score": {"type": "integer"},
                "matched_criteria": {"type": "array", "items": {"type": "string"}},
                "failed_criteria": {"type": "array", "items": {"type": "string"}},
                "evidence_quote": {"type": "string"}
            },
            "required": ["is_match", "confidence_score", "matched_criteria", "failed_criteria", "evidence_quote"]
        }
        
        logger.info("[SYSTEM] Compiling JSON schema into GBNF Grammar constraints...")
        self.grammar = LlamaGrammar.from_json_schema(json.dumps(self.json_schema))

    def _initialize_model(self) -> Llama:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Core model missing at {self.model_path}")
            
        logger.info(f"Booting AxioMatch Core: Gemma 4 ({self.model_path.name})")
        return Llama(
            model_path=str(self.model_path),
            chat_format="gemma",
            n_ctx=self.n_ctx,
            n_threads=4, # Optimized for i5
            n_batch=512,
            verbose=False
        )

    def analyze_note(self, patient_note: str, trial_criteria: str) -> str:
        prompt = f"""<start_of_turn>user
        You are a clinical data extraction engine. Analyze this patient text against the trial criteria.
        Output your analysis strictly inside the requested JSON format.
        
        CRITICAL RULES:
        1. If the patient does NOT match the criteria, you MUST extract the specific medical reasons and list them in the `failed_criteria` array. Do not leave it empty.
        2. If the patient DOES match, list the exact rules they met in the `matched_criteria` array.

        PATIENT NOTE:
        {patient_note}
        
        TRIAL CRITERIA:
        {trial_criteria}
        <end_of_turn>
        <start_of_turn>model
        """
        
        logger.info("Running grammar-constrained clinical inference...")
        response = self.llm(
            prompt,
            max_tokens=600,
            temperature=0.1, 
            grammar=self.grammar, 
            stop=["<end_of_turn>"]
        )
        
        return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    MODEL_FILE = "models/gemma-4-E4B-it-Q4_K_M.gguf"
    
    try:
        engine = AxioInferenceEngine(model_path=MODEL_FILE)
        test_note = "Patient is a 62-year-old female with triple-negative breast cancer. No cardiac history."
        test_criteria = "Inclusion: Female with breast cancer. Exclusion: History of heart failure."
        
        result = engine.analyze_note(test_note, test_criteria)
        print("\n" + "="*45)
        print(" GRAMMAR-ENFORCED JSON OUTPUT")
        print("="*45)
        print(result)
        
    except Exception as e:
        logger.error(f"Engine failure: {e}")