from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import requests
import os
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# --- Constants ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=13rKiqpIlViPzH1U7oRhJk8py5ch-YbFe"
MODEL_FILE = "fake_news_model.keras"
VECTORIZER_FILE = "vectorizer.pkl"  # Make sure this is deployed alongside or fetched too

# --- Utilities ---

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"🔽 Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"🟢 {filename} already exists.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_url(url):
    r = requests.get(url, timeout=5)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.title.string if soup.title else ''
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    body = ' '.join(paragraphs)
    return clean_text(title + " " + body)

# --- Download and load model ---
download_file(MODEL_URL, MODEL_FILE)
model = tf.keras.models.load_model(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# --- API Route ---
@app.route("/check", methods=["POST"])
def check_fake_news():
    data = request.json
    url = data.get("url")

    try:
        article_text = scrape_url(url)
        vector = vectorizer.transform([article_text])
        prediction = model.predict(vector)[0][0]
        result = "Fake" if prediction > 0.5 else "Real"
        return jsonify({
            "result": result,
            "confidence": float(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
