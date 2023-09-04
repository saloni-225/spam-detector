import streamlit as st
import pickle
import string 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


def transform_text(text):
    
    # convert into lower case.
    text=text.lower()
    
    # tokenize into words.
    text=nltk.word_tokenize(text)
    y=[]
    
    # remove the special character.
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Span Classifier")

input_sms=st.text_input("Enter the message")

if st.button('Predict'):


    # 1.Preprocess
    transformed_sms=transform_text(input_sms)
    # 2.Vectorize
    vector_input=tfidf.transform([transformed_sms])
    # 3.predict
    result=model.predict(vector_input)[0]
    # 4.Result
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")