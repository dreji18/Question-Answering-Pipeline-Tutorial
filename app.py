# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 05:56:28 2023

@author: dreji18
"""

# loading the packages
from rake_nltk import Rake
import wikipedia
from rank_bm25 import BM25Okapi
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# keyword extraction function
def keyword_extractor(query):
    """
    Rake has some features:
        1. convert automatically to lower case
        2. extract important key phrases
        3. it will extract combine words also (eg. Deep Learning, Capital City)
    """
    r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(query)
    keywords = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
    return keywords

# data collection using wikepedia
def data_collection(search_words):
    """wikipedia"""
    search_query = ' '.join(search_words)
    wiki_pages = wikipedia.search(search_query, results = 5)
    
    information_list = []
    pages_list = []
    for i in wiki_pages:
        try:
            info = wikipedia.summary(i)
            if any(word in info.lower() for word in search_words):
                information_list.append(info)
                pages_list.append(i)
        except:
            pass
    
    original_info = information_list
    information_list = [item[:1000] for item in information_list] # limiting the word len to 512
    
    return information_list, pages_list, original_info

# document ranking function
def document_ranking(documents, query, n):
    """BM25"""
    try:
        tokenized_corpus = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        datastore = bm25.get_top_n(tokenized_query, documents, n)
    except:
        pass
    return datastore

def qna(context, question):
    """DistilBert"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad', return_dict=False)
    encoding = tokenizer.encode_plus(question, context)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer_tokens_to_string

@app.get("/predict")
def answergen(search_string: str):
    try:
        keyword_list = keyword_extractor(search_string)
        information, pages, original_data = data_collection(keyword_list)
        datastore = document_ranking(information, search_string, 3)
        
        answers_list = []    
        for i in range(len(datastore)):
            result = qna(datastore[i], search_string)
            answers_list.append(result)
            
        return {"answer 1": answers_list[0],
                "answer 2": answers_list[1],
                "answer 3": answers_list[2]}
    except:
        return {"sorry couldn't process the request"}

#uvicorn app:app --port 8000 --reload

#%%
