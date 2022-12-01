---
title: Semester Project [rename]
date: "November 2022"
author: Carlos Rojas, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

# Tasks
- Devansh: work on BERT model and help on linear model.
- Rahul: work on BERT model and help more with report.
- Vineet: work on Linear model.

Pizza [@pizza2000identification] is an understudied yet widely utilized implement for delivering in-vivo *Solanum lycopersicum* based liquid mediums in a variety of next-generation mastications studies. Here we describe a de novo approach for large scale *T. aestivum* assemblies based on protein folding that drastically reduces the generation time of the mutation rate.

# Introduction (Vineet)

# Methods (All)

## Vineet

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

## Rahul

### Applying Distilbert base uncased distilled squad for the dataset.

* Distilbert is a small, fast, cheap, and light transformer model trained by disitilled BERT base.
* It has 40% fewer parameters then Bert based uncased which is an older version of the model.
* It is the best suited and latest available Question Answer problem set application with SQuAD data.

In this method:
- The Distilbert is a pre-trained model in hugging face. It is especially trained for the SQuAD Dataset and Coco Dataset.
- So, we have used the pretrained model to fit into our custom dataset. 
- Here, a framework known as Haystack had to be imported inorder to use the libray FARMReader which is especially dedicated to import the model from hugging face.
- After importing the model, the model is fit to the training dataset that we prepared and tried to fine tune the model by tweaking some hyperparameters such as number of epochs etc.
- While training the model it could be observed that at some point in the training, the training is getting to the least.
- So, in resemblence with the pocket algorithm the model is storing the most optimised parameters where it achieved the least training error and that particular weights of the model are being stored.
- Now, with this model we tested it on the test file that we created. We have taken a ratio of 80:20 to split the dataset into train and test respectively.
- Which yielded an F1 score of 0.5418, Exact Match of 0.2247191, top_n_accuracy of 0.80898.
- We have given an context that is not related to california but from India. Astonishingly, the model could predict well enough. Although, the model could not intutively eliminate the case that it cannot answer some questions from the given questions and is still trying to answer.
- This, brings us to the case that the data is not sufficient for the model to learn that it cannot answer all the questions using the context. We might need some more data to make the model know its limitations.

# Comparisons (Mainly linear vs BERT - Vineet/Devansh / Rahul for BERT comparison)

# Example Analysis (Devansh)

# Conclusions (Devansh)


# References
