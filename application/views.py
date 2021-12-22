from django.shortcuts import render
from django.http import HttpResponse, response
from application import forms

# Create your views here.
def index(request):
    review_form=forms.InsertReviewForm()
    context={
       "form":review_form
    }
    response=render(request,'index.html',context)
    return response

def check_review(request):
    if request.method=="POST":
        review=request.POST.get('review')
        # context={
        #  "form":review
        #  }
        print("Check 1")
        print(review)
        result=predict(review)
        context={
         "res":result
         }

        response=render(request, 'output.html',context)
        return response
        # print("result => ", result)
        

       
        #  response=render(request,'index.html',context)
        #  return response

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import spacy
import pickle
import re

vectorizer = pickle.load(open("application/vectorizer.pkl", "rb"))
sk_nblearn = pickle.load(open("application/sk_nblearn.pkl", "rb"))
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# function to clean text data
def clean_desc(desc):
    clean_1 = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    clean_2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    desc = [clean_1.sub("", line.lower()) for line in desc]
    desc = [clean_2.sub(" ", line) for line in desc]
    #print(desc)
    return desc

# tokenization using spaCy
def tokenization(x):
    desc_tokens = []
    for i in x:
        i = nlp(i)
        temp = []
        for j in i:
            temp.append(j.text)
        desc_tokens.append(temp)
    print('asjhskjddkjd')    
    print(desc_tokens)
    return desc_tokens
  
# function to remove stopwords
def remove_stopwords(desc):
    s = []
    for r in desc:
        s_2 = []
        for token in r:
            if nlp.vocab[token].is_stop == True:
                continue
            else:
                s_2.append(token)
        s.append(" ".join(s_2))    
        
    return s

def process(review):
    print("original input: ", review)
    df_inp = pd.DataFrame({ 'text': [review] })
    review = clean_desc(df_inp['text'])
    review = tokenization(review)
    review = remove_stopwords(review)

    X_rev_idf = vectorizer.transform(review)
    print("! -- ", X_rev_idf.shape[0] == len(review))
    return X_rev_idf

def predict(review):
    X_rev_idf = process(review)
    print("Processed:")
    prediction = sk_nblearn.predict(X_rev_idf)
    probablity = sk_nblearn.predict_proba(X_rev_idf)
    print('='*30)
    print("results: ")
    print(prediction, probablity, np.max(probablity))
    return {'result':prediction, 'prob':np.max(probablity)}




        


