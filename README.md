# NLP BBC News Project

## Structure
- `preprocessing.ipynb` : Nettoyage des textes + préparation Word2Vec + tokenizer
- `train_models.ipynb` : Entraînement LSTM, BERT, DistilBERT
- `app.py` : Interface Streamlit
- `tokenizer_keras.pickle` : Tokenizer LSTM
- `model_lstm.keras` : Modèle LSTM
- `modele_bert/` et `modele_distilbert/` : Modèles BERT sauvegardés

## Installation
```bash
pip install -r requirements.txt

## lancer l'application
streamlit run app.py

## notes
Python 3.10+ recommandé

TensorFlow 2.13 et Keras 3

Supprimer les anciens fichiers model_lstm*.h5 ou .keras avant de recharger