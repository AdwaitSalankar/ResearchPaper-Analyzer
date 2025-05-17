import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from app.py import abstract

# Load model locally
model = SentenceTransformer('all-MiniLM-L6-v2')  # Change if you have another preferred model

# Input text (replace this or read from a file)
input_text = abstract

# Sentence splitter using regex
def split_into_sentences(text):
    # Simple sentence boundary detection
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

sentences = split_into_sentences(input_text)

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
    sentence_embedding = model.encode([sentence])
    abstract_embeddings = model.encode(abstracts)
    similarities = cosine_similarity(sentence_embedding, abstract_embeddings)[0]
    max_sim = max(similarities)
    if max_sim >= threshold:
        best_match = abstracts[similarities.argmax()]
        return {
            "sentence": sentence,
            "similarity": max_sim,
            "matched_abstract": best_match
        }
    return None

# Run check
matches = []
for sentence in sentences:
    xml_data = search_arxiv(sentence, max_results=20)
    abstracts = extract_abstracts(xml_data)
    match = detect_plagiarism(sentence, abstracts)
    if match:
        matches.append(match)

# Output results
if matches:
    print("\n⚠️ Plagiarized Sentences Detected:")
    for m in matches:
        print(f"\nSentence: {m['sentence']}")
        print(f"Similarity: {m['similarity']:.2f}")
        print(f"Matched Abstract: {m['matched_abstract']}\n")
else:
    print("✅ No plagiarism detected.")
