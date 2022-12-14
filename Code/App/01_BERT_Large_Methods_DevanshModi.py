# -*- coding: utf-8 -*-
"""01_Methods_DevanshModi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1clEgg_JXZ_uH6GmPjw1FGBUvvA7v-9Zw
"""

!nvidia-smi

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

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

from google.colab import drive
drive.mount('/content/drive')

!pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab,ocr]

# For Colab/linux based machines:
!wget https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz
!tar -xvf xpdf-tools-linux-4.04.tar.gz && sudo cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

!pip install pytorch-transformers

"""# Performing Question Answering with a Fine-Tuned BERT"""

# We have training data built using Haystack Data Annotator [https://annotate.deepset.ai/index.html]
# This file has been saved in my Google Drive
with open('/content/drive/MyDrive/train-v1.json', 'rb') as f:
  merged= json.load(f)

print('Size of training dataset ' + str(len(merged['data'])))

# This file has been saved in my Google Drive
with open('/content/drive/MyDrive/dev-v1.json', 'rb') as f:
  val_data= json.load(f)

print('Size of validation dataset ' + str(len(val_data['data'])))

from haystack.nodes import FARMReader
from haystack.utils import fetch_archive_from_http

"""### epochs = 1"""

reader = FARMReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad", use_gpu=True)
data_dir = "/content/drive/MyDrive"
reader.train(data_dir=data_dir, train_filename="merged_file.json", use_gpu=True, n_epochs=1, save_dir="devansh_model")

data_dir = "/content/drive/MyDrive"
results = reader.eval_on_file(data_dir, "/content/drive/MyDrive/dev-v1.json", device="cuda")
results

"""### epochs = 10"""

reader = FARMReader(model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad", use_gpu=True)
data_dir = "/content/drive/MyDrive"
reader.train(data_dir=data_dir, train_filename="merged_file.json", use_gpu=True, n_epochs=10, save_dir="devansh_model")

"""### Run again with training and validation split"""

data_dir = "/content/drive/MyDrive"
results = reader.eval_on_file(data_dir, "/content/drive/MyDrive/dev-v1.json", device="cuda")
results

"""We are able to get a 93% accuracy for our context.

Test on a Single Question
"""

import torch

from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

"""## Example Question

I will now try an example question and answer.
"""

question = "What is San Francisco known for?"

answer_text = "San Francisco, officially the City and County of San Francisco, is the commercial, financial, and cultural center of Northern California. The city proper is the fourth most populous in California and 17th most populous in the United States, with 815,201 residents as of 2021."

"""## Find out tokens in this example input"""

input_ids = tokenizer.encode(question, answer_text)

print('The input has a total of {:} tokens.'.format(len(input_ids)))

tokens = tokenizer.convert_ids_to_tokens(input_ids)

out = []
for token, id in zip(tokens, input_ids):
    out.append('{:}'.format(token))

out

# run the model with the inputs
outputs = model(torch.tensor([input_ids]),
                             return_dict=True) 

start_scores = outputs.start_logits
end_scores = outputs.end_logits

"""Now we can highlight the answer just by looking at the most probable start and end words. """

answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

answer = ' '.join(tokens[answer_start:answer_end+1])

print('Answer: "' + answer + '"')

"""Answer is correct. This shows to be very powerful.

# Conclusion
"""

start_scores = start_scores.detach().numpy().flatten()
end_scores = end_scores.detach().numpy().flatten()

"""Score distribution for each word in the context"""

import pandas as pd

scores = []
for (i, token_label) in enumerate(out):

    scores.append({'token_label': token_label, 
                   'score': start_scores[i],
                   'Type': 'start'})
    
    scores.append({'token_label': token_label, 
                   'score': end_scores[i],
                   'Type': 'end'})
    
df = pd.DataFrame(scores)

g = sns.catplot(x="token_label", y="score", hue="Type", data=df,
                kind="bar", height=4, aspect=2)

g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")
g.ax.grid(True)