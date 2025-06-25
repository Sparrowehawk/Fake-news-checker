from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load vectorizer and model
vectorizer = joblib.load('./vectorizer.pkl')
model = tf.keras.models.load_model('./fake_news_model.keras')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/check', methods=['POST'])
def check_fake_news():
    data = request.json
    url = data.get('url')
    
    # Scrape
    response = requests.get(url, timeout=5)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string if soup.title else ''
    body = ' '.join([p.get_text() for p in soup.find_all('p')])

    # Preprocess
    text = clean_text(title + " " + body)
    vector = vectorizer.transform([text])
    
    # Predict
    pred = model.predict(vector)[0][0]
    result = "Fake" if pred > 0.5 else "Real"
    
    return jsonify({'result': result, 'confidence': float(pred)})
