---
title: NLP Based Question-Answering model on SQuAD Dataset.
date: "November 2022"

author: Rahul Raghava Peela, San José State University
author: Devansh Modi, San José State University
author: Vineet Kiragi, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract
Our project is focused on applying Natural Language Processing techniques in the domain of Question-Answering problem. This is a major search and machine learning problem that has several usecases in the real world. We investigate and explore multiple techniques to find the right machine learning models. Our study includes linear modeling, basic BERT, BERT Large, and BERT Distrilled models.

# Tasks
- Devansh: work on BERT model and help on linear model.
- Rahul: work on BERT model and help more with report.
- Vineet: work on Linear model.

# Introduction
Reading Comprehension (RC), or the ability to read text and then answer questions about it, is a challenging task for machines, requiring both understanding of natural language and knowledge about the world. Question Answering models are able to retrieve the answer to a question from a given text. This is useful for searching for an answer in a document. Depending on the model used, the answer can be directly extracted from text or generated from scratch. Question Answering (QA) models are often used to automate the response to frequently asked questions by using a knowledge base (e.g. documents) as context. As such, they are useful for smart virtual assistants, employed in customer support or for enterprise FAQ bots

# Methods (All)

## Vineet

Implemented the basic bert base uncased model for the Dataset we created by importing a model from Hugging face. Have written in detail about why choose this model in paper. To run the imported models from hugging face the data needs to be in a particular format. So parsed the data which is in json format to create 3 lists of contexts, questions and answers, which is the required format. Then created embeddings of the data using the recommended tokenizer from haystack. Using model.train command tried to fine tune the model to fit our Dataset. And checking the accuracy using a validation dataset. All this is implemented in a single Jupiter notebook in a single git commit with linkLinks to an external site.

Implemented a linear model using Euclidean distance and cosine similarity. To achieve this, first I created the embeddings of the data using Facebooks Infersent embeddings, which uses sematic embeddings of the data. The creation of embeddings is implemented in a single notebook in a single git commit with link Links to an external site.. using Euclidean Furthermore using the sentence embeddings created a unsupervised learning model is developed using Euclidean distance and cosine similarity which will figure out the sentence nearest to question in the context which in turn might contain similarity since all the question are from context. The unsupervised model is implemented in a single notebook in a single git commit with link.

## Applying BERT Large Uncased Whole Word Masking with Squad Benchmarking (Devansh)

This technique uses an English language pre-trained model by employing masked language modeling (MLM) scheme. 
This model is not case-sensitive; and, it does not distinguish between `english` and `English`.

This BERT model was trained using a novel method called Whole Word Masking, unlike conventional BERT models. In this instance, a word's tokens are all simultaneously masked. Overall masking rate is unchanged.

Each masked WordPiece token is predicted separately; the training is the same.

After collecting the training data of cities in the US, we need to fine-tune our model to work well for this dataset. The format is maintained to be SQuAD-like.

This model has the following configuration:
- 24-layer
- 1024 hidden dimension
- 16 attention heads
- 336M parameters.

### Impact of Transfer Learning

In order to leverage the benefits of transfer learning, we must train the BERT pre-trained model again with our training dataset. This will update the weights to make the model's predictions relevant.

In this method, I explore the following:
- What it means for BERT to achieve "human-level performance on question-answering"? 
- Is using BERT a powerful search technique for question-answering?

### Data Augmentation

Data augmentation techniques can help bring the most out of BERT data model.

## Rahul

### Applying Distilbert base uncased distilled squad for the dataset.

* Distilbert is a small, fast, cheap, and light transformer model trained by disitilled BERT base.
* It has 40% fewer parameters then Bert based uncased which is an older version of the model.
* It is the best suited and latest available Question Answer problem set application with SQuAD data.

In this method:
- The Distilbert is a pre-trained model in hugging face. It is especially trained for the SQuAD Dataset and Coco Dataset.
- So, we have used the pretrained model to fit into our custom dataset. Here, a framework known as Haystack had to be imported inorder to use the libray FARMReader which is especially dedicated to import the model from hugging face.
- After importing the model, the model is fit to the training dataset that we prepared and tried to fine tune the model by tweaking some hyperparameters such as number of epochs etc.
- While training the model it could be observed that at some point in the training, the training is getting to the least. So, in resemblence with the pocket algorithm the model is storing the most optimised parameters where it achieved the least training error and that particular weights of the model are being stored.
- Now, with this model we tested it on the test file that we created. We have taken a ratio of 80:20 to split the dataset into train and test respectively.
- Which yielded an F1 score of 0.5418, Exact Match of 0.2247191, top_n_accuracy of 0.80898.
- We have given an context that is not related to california but from India. Astonishingly, the model could predict well enough. Although, the model could not intutively eliminate the case that it cannot answer some questions from the given questions and is still trying to answer. This, brings us to the case that the data is not sufficient for the model to learn that it cannot answer all the questions using the context. We might need some more data to make the model know its limitations.

# Comparisons (Mainly linear vs BERT - Vineet/Devansh / Rahul for BERT comparison)

# Example Analysis (Devansh)

# Conclusions (Devansh)


# References
