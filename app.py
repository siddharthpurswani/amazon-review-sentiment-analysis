import joblib
from pydantic import BaseModel
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from fastapi import Depends
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix



app = FastAPI()

templates = Jinja2Templates(directory="templates")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs, emails, HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
negations = {'not', 'no', 'nor', "didn't", "wasn't", "isn't", "aren't"}
stop_words = stop_words - negations
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2 ]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

vectorizer = joblib.load('tfidf_vectorizer.joblib')


model = joblib.load("logistic_regression_model.joblib")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class Review(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, review_text: str = Form(...)):  
    cleaned_txt = clean_text(review_text)
    preprocessed = preprocess(cleaned_txt)
    vectorized = vectorizer.transform([preprocessed])

    pred = model.predict(vectorized)[0]
    sentiment = "positive" if pred == 1 else "negative"
    return templates.TemplateResponse(
        "result.html", 
        {"request": request, "review": review_text, "sentiment": sentiment}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)