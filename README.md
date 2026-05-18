# 🧬 AxioMatch Engine

**Air-Gapped EMR-To-Trial Matcher powered by Gemma 4**

AxioMatch is a privacy-first, 100% offline Clinical Trial Patient Matching engine. It utilizes a local RAG (Retrieval-Augmented Generation) pipeline and the open-weights **Gemma-4-E4B** model to instantly evaluate unstructured Electronic Medical Records (EMRs) against complex clinical trial Inclusion/Exclusion criteria—without ever sending protected health information (PHI) over the internet.

---

## 🚀 The Problem & The Solution
**The Problem:** Finding eligible patients for clinical trials requires clinical coordinators to manually read hundreds of pages of unstructured medical notes. Modern cloud-based LLMs (like OpenAI) cannot be used because hospitals are strictly bound by HIPAA privacy laws and cannot upload patient data to external servers.

**The Solution:** AxioMatch brings the AI to the data. By leveraging `llama.cpp` and hardware-level quantization, AxioMatch runs a powerful Gemma 4 reasoning engine entirely on local hardware. 

## ✨ Key Features
* 🔒 **100% Air-Gapped:** Zero external API calls. Fully HIPAA-compliant by design.
* ⚡ **Zero-Latency RAG Pre-Filtering:** Uses Scikit-Learn TF-IDF mathematical vector spaces to instantly filter thousands of EMRs down to the highest-probability candidates before LLM execution.
* 📄 **Multimodal PDF Ingestion:** Instantly extracts and evaluates text from newly scanned, external EMR PDFs.
* 🧠 **Deterministic Output (GBNF):** Uses strict Grammar-Based Generation (GBNF) to force the LLM to output its medical reasoning as structured JSON, eliminating hallucinations.
* 📥 **Enterprise Audit Trails:** Generates clean, Excel-ready CSV cohort reports for Lead Investigator review.

---

## 🛠️ System Architecture

1. **The Database Engine (`rag_engine.py`):** Converts 5,000+ unstructured medical notes into vector representations to mathematically isolate top candidates.
2. **The Inference Engine (`inference_engine.py`):** Loads the quantized Gemma 4 model into local RAM. Evaluates the text against trial criteria and applies GBNF grammar constraints to output strict `MATCH` or `EXCLUDED` JSON objects with exact text evidence.
3. **The Web Dashboard (`flask_app.py` & `index.html`):** A sleek, single-page Flask application using Tailwind CSS that manages state, routes AI reasoning, and builds the UI execution matrix.

---

## 💻 Tech Stack
* **AI Model:** Gemma-4-E4B-it (Quantized Q4_K_M via Unsloth)
* **Backend:** Python, Flask, `llama-cpp-python`
* **Vector Search:** `scikit-learn`, `numpy`, `pandas`
* **Document Parsing:** `PyPDF2`
* **Frontend:** HTML5, JavaScript, Tailwind CSS

---

## ⚙️ Installation & Setup

To run this application locally, you need a standard machine (Intel i5/Ryzen 5 or better) with at least 8GB of RAM. No dedicated GPU is required.

### 1. Clone the Repository
```bash
git clone https://github.com/meghabothe996/AxioMatch-Engine.git
cd AxioMatch-Engine

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Download the AI Model

Due to file size limits, the Gemma model is not included in this repository.

1. Download the **Gemma-4-E4B-it-Q4_K_M.gguf** model from Hugging Face (Unsloth).
2. Place the `.gguf` file inside the `models/` directory.

### 4. Run the Engine

```bash
python src/flask_app.py

```

*Open your browser and navigate to `http://127.0.0.1:5000`*

---

## 🏥 Usage Guide

The dashboard is split into two distinct workflows:

**Option A: Hospital Database Scan**

1. Define your Inclusion/Exclusion criteria in the left panel.
2. Click **Run Cohort Match**.
3. The RAG engine will instantly search the `mtsamples.csv` dataset, hand the top matches to Gemma 4, and generate an AI Validation card for each patient.
4. Click **Export Master Cohort (CSV)** to download the results.

**Option B: New Patient (PDF)**

1. Toggle the workflow mode to **New Patient (PDF)**.
2. Upload a scanned medical record (`.pdf`).
3. Click **Extract & Verify PDF**.
4. The system will bypass the database, read the document directly, and use Gemma 4 to verify eligibility.

---

## 📂 Repository Structure

```text
AxioMatch-Engine/
│
├── data/
│   └── mtsamples.csv           # Anonymized EMR dataset (5,000+ records)
│
├── models/
│   └── [PLACE_GGUF_MODEL_HERE] # Gemma 4 weights go here
│
├── src/
│   ├── templates/
│   │   └── index.html          # Tailwind SPA Dashboard
│   ├── data_loader.py          # Data sanitation and ingestion
│   ├── flask_app.py            # API routing and Web Server
│   ├── inference_engine.py     # Llama.cpp and GBNF handling
│   └── rag_engine.py           # TF-IDF Vectorization logic
│
├── requirements.txt            # Project dependencies
└── README.md

```
---

## 📚 Acknowledgements & Citations
* **AI Model:** [Gemma-4-E4B-it-GGUF by Unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF). We utilized the `Q4_K_M` quantization for optimal performance on local CPU/RAM constraints.
* **Dataset:** [Medical Transcriptions (MTSamples) via Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions). This public dataset was used to simulate a hospital's unstructured EMR database.

---

*Built for the Kaggle The Gemma4Good Hackathon*
