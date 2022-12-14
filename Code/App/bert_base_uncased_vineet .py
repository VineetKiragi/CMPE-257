# -*- coding: utf-8 -*-
"""Bert_Base_Uncased_Vineet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mQdXTqhxXwKYWzZh8lpsw8mhje9SY0Nv
"""

# installing transformers
!pip install transformers

# importing required libraries
import requests
import json
import torch
import os
from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
from transformers import AdamW
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Load the training dataset and take a look at it
with open('train.json', 'rb') as f:
  train= json.load(f)

# Load the training dataset and take a look at it
with open('valid.json', 'rb') as f:
  valid= json.load(f)

# Each 'data' dict has two keys (title and paragraphs)
train['data'][0].keys()

train['data'][0]

"""As we can see each row contains a key "qas" which contains all the information about the annotated answers and another key "context" which contains the data which is used for annotation. """

len(train['data'])

len(valid['data'])



#Converting the Squad data into df for visualization.
p = []
q=[]
ans = []
astart = []
aend= []
for i in train["data"]:
    for j in i["paragraphs"][0]["qas"]:
        #print(j)
        if(len(j["answers"])!=0):
          astart.append(j["answers"][0]["answer_start"])

          p.append(i["paragraphs"][0]["context"])
          q.append(j["question"])
          
          ans.append(j["answers"][0]["text"])
          aend.append(j["answers"][0]["answer_end"])
        
df = pd.DataFrame()
df["context"] = p
df["questions"] = q
df["answer"] = ans
df["answer_start"] = astart
df["answer_end"] = aend

df

# loading the data in triplets of context, questions and answers
def read_data(path):  
  
  with open(path, 'rb') as f:
    squad = json.load(f)

  contexts = []
  questions = []
  answers = []

  for group in squad['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)
  return contexts, questions, answers

#calling the read_data function
train_contexts, train_questions, train_answers = read_data('train.json')
valid_contexts, valid_questions, valid_answers = read_data('valid.json')

train_answers

print(f'There are {len(train_questions)} train questions')
print(f'There are {len(valid_questions)} dev questions')

def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

"""# Visualization"""

#Importing the required libraries for visualization
from itertools import chain
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
#from sklearn.model_selection import train_test_split

allwords = " ".join(df["answer"])



wordcloud = WordCloud(width = 400, height = 400, 
                    background_color ='white', 
                    stopwords = set(STOPWORDS), 
                    min_font_size = 10).generate(allwords)

plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.figure(figsize = (7, 7), facecolor = 'white', edgecolor='blue') 
plt.imshow(wordcloud) 

plt.show()

"""As we can see there are few words like Museums, county and park are repeated many times. We can make use of these words while annotating the data and ask questions on these."""

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df["questions"])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
vec_df = pd.DataFrame(denselist, columns=feature_names)

vec_df

!pip install nltk

!nltk.download('omw-1.4')

from nltk import everygrams
from nltk.corpus import stopwords
import re
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer as snowball_stemmer


def clean_text_function2(text):
    t = text #contractions.fix(text)
    t= t.lower().split()#lower case
    t = [(re.sub(r'[^a-z ]', '', ch)) for ch in t]#remove everything other than a-z
    #t=[word for word in t if word not in stopword]#removing stop words
    t= [wn.lemmatize(word) for word in t]
    t=[snowball_stemmer.stem(word) for word in t]
    t=(' ').join(t)
    t=list(everygrams(t.split(), 2, 3))
    return t

# clean_text_function2(df['context'][0])
a=df["context"][0]
a

s = a.split()

words = []
for i in df["context"].unique():
  s=i.split()
  words = [*words, *s]

# STOPWORDS
import nltk

df_bigram = pd.DataFrame()
df_bigram2 = pd.Series(nltk.ngrams(words, 2)).value_counts()
bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())
bigrams = list(bigrams_series.index)
print(len(bigrams))
clean = []
flag = 0
for gram in bigrams:
  for stop in STOPWORDS:
    if stop in gram:
      # print(stop,gram)
      bigrams.remove(gram)
      flag = 1
      break
    flag = 0
  if flag == 0:
    clean.append(gram)
  

# clean = [gram for gram in bigrams if not any(stop in bigrams for stop in STOPWORDS)]
print(clean[:10])
print(len(clean))

bigram_series_2 = bigrams_series.loc[clean]

bigram_series_2[:20].sort_values().plot.barh(color='blue', width=0.8, figsize=(12,8))

"""The bigrams as we can see shows the two words which appear next to each other. This we can use while annotating the answers for questions.

Tokenizing the data
"""

# getting the model and its tokenizer (currently training on few rows as it is very time consuming)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

train_encodings.keys()

"""Visualing the output of tokenizer, input ids are the token indices with padding of 0s, token_type_ids are different integers for different sequences and attention mask states which positions to give attention to while training"""

train_encodings["input_ids"][0]

train_encodings["token_type_ids"][0]

train_encodings["attention_mask"][0]

# printing the number of training data
no_of_encodings = len(train_encodings['input_ids'])
print(f'We have {no_of_encodings} context-question pairs')

# adding the answers in the training set for fine tuning
def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

# creating the dataset in the format it is required for fine tuning BERT
class SQuAD_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# loading the BERT model which we will fine tune
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# checking the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

# Fine tuning it per batch
N_EPOCHS = 5
optim = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

for epoch in range(N_EPOCHS):
  loop = tqdm(train_loader, leave=True)
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    loss = outputs[0]
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix(loss=loss.item())

# checking the performance
model.eval()

acc = []

for batch in tqdm(valid_loader):
  with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_true = batch['start_positions'].to(device)
    end_true = batch['end_positions'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)

    start_pred = torch.argmax(outputs['start_logits'], dim=1)
    end_pred = torch.argmax(outputs['end_logits'], dim=1)

    acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
    acc.append(((end_pred == end_true).sum()/len(end_pred)).item())

acc = sum(acc)/len(acc)

acc