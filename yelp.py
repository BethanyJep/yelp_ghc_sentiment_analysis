import streamlit as st
import pandas as pd
import pickle

st.title("Detect sentiments from reviews")
    
user_review = st.text_area("Let us rate your review:enter your review for sentiment analysis")

with open('yelp_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predicted_sentiment(text):
    prediction = model.predict([text])
    return prediction

# When the button is clicked
if st.button("Rate my review"):
    # Perform sentiment analysis using the loaded model
    prediction = predicted_sentiment(user_review)
    
    # Determine the sentiment category (Adjust as per your model's output)
    if prediction == 5:
        result = 'Positive 😊'
    elif prediction == 3:
        result = 'Neutral 😐'
    else:
        result = 'Negative 😞'

    # Display the result
    st.write(f"Predicted Sentiment: {result}")