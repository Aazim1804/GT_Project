import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("📰 Fake News Detection App")
st.write("Enter any news article or headline below to check if it's **Real** or **Fake**.")

user_input = st.text_area("Paste News Text Here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if prediction == 1:
            st.success("This news is **REAL**.")
        else:
            st.error("This news is **FAKE**.")
