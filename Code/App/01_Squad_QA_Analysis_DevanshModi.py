# -*- coding: utf-8 -*-
"""01_Squad_QA_Analysis_DevanshModi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rl7YakhwVcstRlGiVhEEB4F4YzBL1TzR
"""

import requests
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

"""## Initial Analysis of Squad Data Format"""

#importing the dataset into the notebook
!wget 'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json'
with open("train-v1.1.json") as f:
  squad = json.load(f)

squad['data'][0].keys()

"""As we can see there are two keys in the dataset, title and paragraph

Each title has many paragraph inside it, and for each paragraph there are many questions and corresponding answers.

So there are 35 rows with title and paragraph keys in this dataset

## Preprocesing
"""

import json

from google.colab import drive
drive.mount('/content/drive')

# This file has been saved in my Google Drive
with open('/content/drive/MyDrive/merged_file.json', 'rb') as f:
  val_data= json.load(f)

print('Size of the complete dataset ' + str(len(val_data['data'])))

import numpy as np
import pandas as pd

def squad_to_df(path):
    file = json.loads(open(path).read())

    json_data = pd.json_normalize(file, 'data')
    qas = pd.json_normalize(file, ['data','paragraphs','qas'])
    context = pd.json_normalize(file,['data','paragraphs'])
    
    #print(r['context'].values)

    contexts = np.repeat(context['context'].values, context.qas.str.len())
    qas['context'] = contexts

    data = qas[['id','question','context','answers']].set_index('id').reset_index()
    data['context_id'] = data['context'].factorize()[0]
    
    return data

data = squad_to_df('/content/drive/MyDrive/merged_file.json')
data.head()

"""## Get Unique Documents"""

data[['context']].drop_duplicates().reset_index(drop=True)

vectorizer_cnf = {
    'lowercase': True,
    'stop_words': 'english',
    'analyzer': 'word',
    'binary': True,
}

nn_cnf = {
    'n_neighbors': 4,
    'metric': 'cosine' #'euclidean'
}

embedding = TfidfVectorizer(**vectorizer_cnf)
nearest_neighbor = NearestNeighbors(**nn_cnf)

import sklearn
sorted(sklearn.neighbors.VALID_METRICS['brute'])

X = embedding.fit_transform(data['context'])
nearest_neighbor.fit(X, data['context_id'])

text = 'What are the most popular things in Houston?'

vector = embedding.transform([text])
vector = embedding.inverse_transform(vector)

vector

value = nearest_neighbor.kneighbors(embedding.transform([text]), return_distance=False)
selected = data.iloc[value[0][0]]['context']
selected