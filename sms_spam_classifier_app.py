import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl' , 'rb'))

st.title("Email/SMS SPAM CLASSIFIER")

input_sms = st.text_input("Enter the message")


# 1 Preprocess
# 2. Vectorize
# 3. Predict
# 4. Display


def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenization - after this text is converted into list
    y = []
    for i in text:
        if i.isalnum():  # checking for alphanumerics
            y.append(i)

    text = y[:]  # copying the string y to text
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Predict"):

    # PREPROCESSING
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display

    if result == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")