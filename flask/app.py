import os
import re
from flask import Flask, request, render_template
from PIL import Image
import pytesseract
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


tfidf = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')
best_svc_model = joblib.load('best_svc_model.joblib')

with open('clinical-stopwords.txt', 'r') as f:
    clinical_stopwords = set(f.read().splitlines())

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word not in clinical_stopwords])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    image = request.files['image']
    img = Image.open(image)

    extracted_text = pytesseract.image_to_string(img)

    cleaned_text = re.sub(r"Hospital Name:.*", "", extracted_text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text).strip()
    cleaned_text = cleaned_text.replace('|', 'I')
    cleaned_text = re.sub(r"[^\x00-\x7F]+", '', cleaned_text)

    if "Symptoms:" in cleaned_text:
        description = cleaned_text.split("Symptoms:")[-1].strip()
    else:
        description = cleaned_text.strip()
    description = description.replace('\n', ' ').strip()

    description_tfidf = tfidf.transform([preprocess_text(description)])
    predicted_label_encoded = best_svc_model.predict(description_tfidf.toarray())
    predicted_disease = label_encoder.inverse_transform(predicted_label_encoded)

    return render_template('index.html', prediction=predicted_disease[0])

if __name__ == "__main__":
    app.run(debug=True)
