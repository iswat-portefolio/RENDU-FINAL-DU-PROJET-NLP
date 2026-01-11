# ======================
# app.py - Version fonctionnelle
# ======================

import os

import streamlit as st
import time
import numpy as np
import tensorflow as tf
import spacy
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

# --- Configuration de la page ---
st.set_page_config(page_title="NLP BBC Demo", layout="wide")
st.title("🤖 Analyseur d'Articles BBC News")

# ======================
# --- Chargement des ressources avec cache ---
# ======================
@st.cache_resource
def load_all():
    # SpaCy pour NLP
    nlp = spacy.load("en_core_web_sm")

    # Résumé BART
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # BERT
    # BERT
    tokenizer_bert = AutoTokenizer.from_pretrained("modele_bert")
    model_bert = TFAutoModelForSequenceClassification.from_pretrained("modele_bert")

    # LSTM + tokenizer Keras
    # Charger le modèle LSTM
    model_lstm = tf.keras.models.load_model("model_lstm.keras", compile=False)

    with open("tokenizer_keras.pickle", "rb") as handle:
        tokenizer_lstm = pickle.load(handle)
    max_len = 200  # longueur utilisée lors de l'entraînement

    return nlp, summarizer, model_bert, tokenizer_bert, model_lstm, tokenizer_lstm, max_len

# Chargement
nlp, summarizer, model_bert, tokenizer_bert, model_lstm, tokenizer_lstm, max_len = load_all()

# ======================
# --- Upload fichier TXT ---
# ======================
uploaded_file = st.file_uploader("Upload d'un fichier TXT", type="txt")
if uploaded_file:
    texte = uploaded_file.read().decode("utf-8")

    col1, col2, col3 = st.columns(3)
    modele_choisi = st.selectbox("Choisir le modèle de classification", ["LSTM", "BERT"])

    # --- Bouton CLASSIFIER ---
    if col1.button("🎯 Classifier"):
        start = time.time()

        categories = ["Business", "Entertainment", "Tech"]

        if modele_choisi == "BERT":
            inputs = tokenizer_bert(texte, return_tensors="tf", truncation=True, padding=True, max_length=128)
            outputs = model_bert(inputs)
            probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
            label = categories[np.argmax(probs)]

        elif modele_choisi == "LSTM":
            seq = tokenizer_lstm.texts_to_sequences([texte])
            X_input = pad_sequences(seq, maxlen=max_len)
            probs = model_lstm.predict(X_input)[0]
            label = categories[np.argmax(probs)]

        st.info(f"✅ Modèle utilisé : {modele_choisi}")
        st.write(f"Classe prédite : **{label}**")
        st.write("Probabilités :", {cat: round(float(p), 3) for cat, p in zip(categories, probs)})
        st.caption(f"Temps d'inférence : {round(time.time()-start, 3)}s")

    # --- Bouton EXTRACTION D'ENTITÉS ---
    if col2.button("🔍 Extraire infos"):
        start = time.time()
        doc = nlp(texte)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON","ORG","GPE"]]
        st.write("**Entités détectées :**", entities)
        st.caption(f"Temps : {round(time.time()-start, 3)}s")

    # --- Bouton RESUMÉ ABSTRACTIF ---
    if col3.button("📝 Résumer"):
        start = time.time()
        with st.spinner("Le modèle BART génère le résumé..."):
            summary = summarizer(texte[:1024], max_length=130, min_length=30)[0]['summary_text']
            st.success(summary)
        st.caption(f"Temps : {round(time.time()-start, 3)}s")

    st.divider()
    st.subheader("Texte Original")
    st.write(texte)
