import os
import torch
from flask import Flask, render_template, request, jsonify
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PyPDF2 import PdfReader
from together import Together
import fitz  # PyMuPDF for PDF text extraction
import os
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import re
import requests
import xml.etree.ElementTree as ET


plagiarism_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Load Fine-Tuned Model and Tokenizer
model_path = "fine_tuned_model"  # Path to fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
# Set model to evaluation mode
model.eval()

def query_together_api(prompt):
    client = Together(api_key="7d6e1d68c2bb19ce8a1b7ebfe90ffde614cf504754a9dd60213f542484937592") 
    response = client.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        stop=["</s>"],
        stream=False  # Change this to False
    )
    return response.choices[0].text.strip()



# Function to generate summary prompt
def generate_summary_prompt(extracted_text):
    return (
        f"You are an intelligent assistant. Summarize the following text extracted from a document:\n\n"
        f"{extracted_text}\n\n"
        f"Provide a concise summary in a few sentences."
    )

def extract_abstract(pdf_path):
    doc = fitz.open(pdf_path)
    abstract_text = ""

    for page in doc:
        text = page.get_text()
        if "abstract" in text.lower():
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().lower() == "abstract":
                    # Abstract is usually right after the "Abstract" heading
                    abstract_text = ""
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() == "" or lines[j].strip().lower() in ["keywords", "introduction"]:
                            break
                        abstract_text += lines[j] + " "
                    return abstract_text.strip()
    return "Abstract not found."

# -------- PDF TEXT EXTRACTION --------
def extract_pdf_text(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# -------- ABSTRACT EXTRACTION --------
def extract_abstract_from_text(text):
    text = text.replace('\r', '\n')  # Normalize newlines
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        if line.strip().lower() == "abstract":
            # Abstract found after the "Abstract" heading
            abstract_text = ""
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip().lower()
                if next_line == "" or next_line in ["keywords", "introduction"]:
                    break
                abstract_text += lines[j] + " "
            return abstract_text.strip()
    
    return None  # Abstract not found

# Plagiarism detection functions

def split_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def search_arxiv(query, max_results=20):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    return response.text

def extract_abstracts(xml_text):
    root = ET.fromstring(xml_text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    abstracts = []
    for entry in root.findall('atom:entry', ns):
        summary = entry.find('atom:summary', ns)
        if summary is not None and summary.text:
            abstracts.append(summary.text.strip().replace('\n', ' '))
    return abstracts

def detect_plagiarism(sentence, abstracts, threshold=0.7):
    if not abstracts:
        return None
    sentence_embedding = plagiarism_model.encode([sentence])
    abstract_embeddings = plagiarism_model.encode(abstracts)
    similarities = cosine_similarity(sentence_embedding, abstract_embeddings)[0]
    max_sim = max(similarities)
    if max_sim >= threshold:
        best_match = abstracts[similarities.argmax()]
        return {
            "sentence": sentence,
            "similarity": round(float(max_sim), 2),
            "matched_abstract": best_match
        }
    return None

def check_plagiarism(abstract_text):
    sentences = split_into_sentences(abstract_text)
    matches = []
    for sentence in sentences:
        xml_data = search_arxiv(sentence, max_results=20)
        abstracts = extract_abstracts(xml_data)
        match = detect_plagiarism(sentence, abstracts)
        if match:
            matches.append(match)
    return matches

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    
def extract_text_from_pdf_no_strip(pdf_path):
    """Extract text from a PDF file"""
    with fitz.open(pdf_path) as doc:
        text = "\n".join([page.get_text() for page in doc]).strip()
    return text

def predict_paper_quality(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    acceptance_score = probabilities[0][1].item() * 100  # Score for 'Accepted' class (1)

    return "Accepted" if predicted_class == 1 else "Rejected", round(acceptance_score, 2)

# Preprocess input and get predictions
def classify_paper(text):
    """Predict acceptance/rejection and confidence score"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU/CPU

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1).squeeze()

    # Extract prediction (0=Rejected, 1=Accepted) and confidence score
    predicted_class = torch.argmax(probs).item()
    confidence_score = probs[1].item() * 100  # Score as percentage (probability of Accepted class)

    return {"prediction": predicted_class, "score": round(confidence_score, 2)}

# Home Page - Upload Form
@app.route("/")
def home():
    return render_template("index.html")

# Handle File Upload and Prediction
@app.route("/predict", methods=["POST"])
def predict():
    """Handles file upload, text extraction, and prediction"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file temporarily
    temp_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(temp_path)

    # Extract text and classify
    paper_text = extract_text_from_pdf(temp_path)

    if not paper_text.strip():
        return jsonify({"error": "Could not extract text from the PDF"}), 400

    # Get classification result
    result = classify_paper(paper_text)
    
    summary_prompt = generate_summary_prompt(paper_text)
    summary = query_together_api(summary_prompt)

    # full_text = extract_pdf_text(temp_path)
    abstract = extract_abstract(temp_path)
    os.remove(temp_path) 
    print(abstract)
    plagiarism_matches = check_plagiarism(abstract)
    print(plagiarism_matches)


    return jsonify({
        "filename": file.filename,
        "prediction": "Accepted" if result["prediction"] == 1 else "Rejected",
        "score": result["score"],
        "summary": summary,
        "plagiarism": plagiarism_matches
    })

if __name__ == "__main__":
    app.run(debug=True)
