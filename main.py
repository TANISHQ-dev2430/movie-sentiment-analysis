import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model 

#load the imdb word index 
words_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in words_index.items()}

#load model 
model = load_model('imdb_rnn_model.h5')

#helper function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#preprocess input text
def preprocess_text(text): 
    words = text.lower().split()
    encoded_review = [words_index.get(word, 0) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500) ##2d input 
    return padded_review

#predict sentiment
def predict_sentiment(text):
    preprocessed = preprocess_text(text)
    prediction = model.predict(preprocessed)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment,prediction[0][0]


## Streamlit app 

import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

#user input
user_input = st.text_area("Enter a movie review:")


if st.button("Predict Sentiment"):
    preprocessed_input = preprocess_text(user_input)

    #make prediction 
    prediction = model.predict(preprocessed_input)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

    #display result 
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")

else: 
    st.write("Please enter a movie review and click the button to predict its sentiment.")
