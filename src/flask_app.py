from flask import Flask, render_template, request, jsonify
import json
import logging
import PyPDF2 # ADDED FOR PDF SUPPORT
from rag_engine import ClinicalRetriever
from inference_engine import AxioInferenceEngine

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Booting Air-Gapped AI Engines (This takes ~15 seconds)...")
retriever = ClinicalRetriever(data_path="data/mtsamples.csv")
engine = AxioInferenceEngine(model_path="models/gemma-4-E4B-it-Q4_K_M.gguf")
print("Engines Ready. Server starting...")

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint 1: Search the CSV Database
@app.route('/api/match', methods=['POST'])
def run_match():
    data = request.json
    criteria = data.get('criteria', '')
    if not criteria: return jsonify({"error": "No criteria provided"}), 400

    matched_patients = retriever.search_patients(criteria, top_k=3)
    results = []
    for patient in matched_patients:
        raw_json = engine.analyze_note(patient['clinical_note'], criteria)
        try:
            analysis = json.loads(raw_json)
        except json.JSONDecodeError:
            analysis = {"is_match": False, "failed_criteria": ["Model JSON Error"]}
            
        results.append({
            "patient_id": patient['patient_id'],
            "specialty": patient['specialty'],
            "case_name": patient['case_name'],
            "score": patient['search_score'],
            "analysis": analysis
        })
    return jsonify(results)

# Endpoint 2: Process a Custom PDF
@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    criteria = request.form.get('criteria', '')
    
    try:
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = "".join([page.extract_text() for page in pdf_reader.pages])
        
        # Run Gemma 4 directly on the PDF text
        raw_json = engine.analyze_note(extracted_text, criteria)
        analysis = json.loads(raw_json)
        
        # Return as a single result card
        return jsonify([{
            "patient_id": "NEW PDF: " + file.filename,
            "specialty": "External Upload",
            "case_name": "Direct Ingestion",
            "score": "N/A",
            "analysis": analysis
        }])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)