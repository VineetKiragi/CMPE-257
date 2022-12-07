---
title: NLP Based Question-Answering model on SQuAD Dataset.
date: "November 2022"

author: Rahul Raghava Peela, Devansh Modi, Vineet Kiragi, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract
Our project presents a question-answering model based on natural language processing (NLP) techniques, which was trained and evaluated on the Stanford Question Answering Dataset (SQuAD). This is a major search and machine learning problem that has several usecases in the real world. We investigate and explore multiple techniques to find the right machine learning models. The model uses a combination of word embedding to generate vector representations of words, sentences, and contexts, which are then used to generate answers to questions. Our study includes linear modeling, basic BERT, BERT Large, and BERT Distrilled models. The results of our experiments show that the model achieves decent performances on the SQuAD dataset, demonstrating its effectiveness in answering a wide range of questions from the given contexts.

# Introduction
Natural language processing (NLP) is a field of computer science and artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. One popular application of NLP is in the development of question-answering systems, which can automatically generate answers to user-provided questions based on a given dataset.The SQuAD dataset is a widely-used benchmark for evaluating the performance of NLP-based question-answering systems. In this paper, we evaluated the performace of few question-answering model that is trained on the SQuAD dataset and demonstrate its effectiveness in generating accurate and relevant answers to a wide range of questions.<img width="759" alt="Screen Shot 2022-12-06 at 5 20 21 PM" src="https://user-images.githubusercontent.com/37727735/206065501-0c1d9886-d364-468b-ba50-c64f1e328717.png">


Overall, our results demonstrate the potential of NLP-based question-answering systems to provide valuable assistance to users seeking information from a given body of text. This work has important implications for a variety of applications, including search engines, virtual assistants, and educational platforms.

## Why Question Answering is important in Natural Language Processing?

Natural Language Processing powers some of the most customer-oriented, lifestyle assisting, and real-world oriented tools and technologies. Question-Answering has significance in many fields such as business intelligence, chatbots, and conversational analytics. 

## Design of Question Answering Workflows

First step of designing a question answering workflow involves building the required dataset.

<p align="center">
    <image src="assets/qa-workflow.png">
    <b>Fig: Question Answering Workflow (Ref: FastForwardLabs) </b>
</p>

Most systems require a Document Reader, which can return an answer for the requested question. These readers in turn are trained using NLP techniques. Each model that is trained for the purpose of question-answering system requires data to be structured in a friendly manner. One of the most common structures is one introduced by Stanford.

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

*In this paper, our team has compiled our own dataset by choosing a context of the Top 150 Most Populated Cities in the United States. We have created this dataset in SQuAD format.*

## Relevance of BERT

BERT is a transformers model that has been pretrained on a significant body of English data in an self-supervised manner. This indicates that it was trained exclusively on the raw texts, with no human labeling of any kind (thus, it can use a lot of material that is readily available to the public), and an automatic procedure to generate inputs and labels from those texts.

BERT has originally been released in base and large variations, for cased and uncased input text. The uncased models also strips out an accent markers.

**In this paper, we have explored multiple BERT models and Linear models to compare their eventual performance for our dataset and context.**

## Dataset Preparation

In order to build the dataset for our modeling methods, we used a tool called `Haystack by Deepset`. This tool allowed us to annotate custom context with questions and answers in the SQuAD format.

<p align="center">
    <image src="assets/haystack.png">
    <b>Fig: Sample Annotation of San Bernardino in Haystack.</b>
</p>

For each city in our training set, we have stored the context, question, and text as seen in the figure above.

# Methods (All)

  ### 1. Linear Model
  The goal of implementing the Linear model is not to reach the state of the art accuracy, but to learn different NLP concepts, implement them and explore more solutions. but to check if we could fit a Linear model with decent enough results. However, my goal is not to reach the state of the art accuracy but to learn different NLP concepts, implement them and explore more solutions. Starting with basic models to know the baseline has been the approach here. The model will focus on **Facebook sentence embeddings** and how it can be used in building QA systems.
  ### Infersent, Facebook Sentence Embedding
  The basic idea behind all these embeddings is to use vectors of various dimensions to represent entities numerically, which makes it easier for computers to understand them for various downstream tasks. Traditionally, the average of vectors of all the words in a sentence is called the bag of words approach. Each sentence is tokenized to words, vectors for these words can be found using glove embeddings and then take the average of all these vectors. This technique has performed decently, but this is not a very accurate approach as it does not take care of the order of words. But [Infersent from Facebook](https://github.com/facebookresearch/InferSent) is a sentence embeddings method that provides semantic sentence representations. It is trained on natural language inference data and generalizes well to many different tasks.
  
  Creating Embeddings from Infersent and finding similarity
  - Create a vocabulary from the training data and use this vocabulary to train infersent model.
  - Once the model is trained, provide sentence as input to the encoder function which will return a 4096-dimensional vector irrespective of the number of words in the sentence.
  - Break the paragraph/context into multiple sentences using the package for processing text data [Textblob](https://textblob.readthedocs.io/en/dev/) Textblob. 
  - Get the vector representation of each sentence and question using Infersent mode. An example is shown below.
  <img width="1091" alt="Screen Shot 2022-12-06 at 3 52 49 PM" src="https://user-images.githubusercontent.com/37727735/206051149-7b3d601c-2dc2-422c-be39-0e83b296ba3c.png">
  - Create features like distance, based on cosine similarity and Euclidean distance for each sentence-question pair
  
  ### Unsupervised model 
  Unsupervised Learning is not using the target variable. Here, the model returns the sentence from the paragraph which has the minimum distance from the given question. First the euclidean distance was used to detect the sentence having minimum distance from the question. The accuracy of this model came around 33% but, when switched to cosine similarity the accuracy improved slightly from 33% to 38%. This makes sense because euclidean distance does not care for alignment or angle between the vectors whereas cosine takes care of that. Direction is important in case of vectorial representations.
  
  ### 2. Bert-Base-Uncased
  The BERT base uncased model is a state-of-the-art natural language processing model developed by Google. It is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, which uses a deep neural network trained on a large amount of unstructured text data to produce highly accurate results when processing and generating human-like language. The "uncased" version of the model means that it is not case-sensitive, meaning that it does not distinguish between uppercase and lowercase letters when processing text. Bert Base Uncased Model is a pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this [paper](https://arxiv.org/pdf/1810.04805.pdf).
  
  The Transformer architecture makes it possible to parallelize ML training extremely efficiently. Massive parallelization thus makes it feasible to train BERT on large amounts of data in a relatively short period of time. Transformers work by leveraging attention, a powerful deep-learning algorithm, first seen in computer vision models. Transformers create differential weights signaling which words in a sentence are the most critical to further process. A transformer does this by successively processing an input through a stack of transformer layers, usually called the encoder. If necessary, another stack of transformer layers - the decoder - can be used to predict a target output. BERT however, doesn’t use a decoder.
 
<p float="left">
  <img src="https://user-images.githubusercontent.com/37727735/206064906-c3012e84-b957-4f8b-a584-da44a0c5a87a.png" width="210" />
  <img src="https://user-images.githubusercontent.com/37727735/206065527-9c3728e3-ada6-44fb-b6f5-253bb5edfe52.png" width="759", height="210" /> 
</p>
Implemented the basic bert base uncased model for the Dataset we created by importing a model from Hugging face. Have written in detail about why choose this model in paper. To run the imported models from hugging face the data needs to be in a particular format. So parsed the data which is in json format to create 3 lists of contexts, questions and answers, which is the required format. Then created embeddings of the data using the recommended tokenizer from haystack. Using model.train command tried to fine tune the model to fit our Dataset. And checking the accuracy using a validation dataset. All this is implemented in a single Jupiter notebook in a single git commit with linkLinks to an external site.

Implemented a linear model using Euclidean distance and cosine similarity. To achieve this, first I created the embeddings of the data using Facebooks Infersent embeddings, which uses sematic embeddings of the data. The creation of embeddings is implemented in a single notebook in a single git commit with link Links to an external site.. using Euclidean Furthermore using the sentence embeddings created a unsupervised learning model is developed using Euclidean distance and cosine similarity which will figure out the sentence nearest to question in the context which in turn might contain similarity since all the question are from context. The unsupervised model is implemented in a single notebook in a single git commit with link.

## Applying BERT Large Uncased Whole Word Masking with Squad Benchmarking

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

<p align="center">
    <image src="assets/dataframe.png">
    <b>Fig: Sample Dataframe for Question and Answers from Top 150 Cities in USA.</b>
</p>

### Applying Distilbert base uncased distilled squad for the dataset.

DistilBert base uncased distilled squad is a state-of-the-art natural language processing (NLP) model developed by Hugging Face. It is a smaller, faster, and more efficient version of the popular BERT model, which has been fine-tuned on the SQuAD dataset for question-answering tasks. This configuration uses the `uncased` version of the BERT model, which means that it is not sensitive to the casing of the input text.
  
* Distilbert is a small, fast, cheap, and light transformer model trained by disitilled BERT base.
* It has 40% fewer parameters then Bert based uncased which is an older version of the model.
* It is the best suited and latest available Question Answer problem set application with SQuAD data.
  
  
  ![Screenshot (34)](https://user-images.githubusercontent.com/117317413/206058662-3129907b-46b7-4f04-866f-2f70d823377f.png)

  
 This model has the following configuration:
- 12-layers
- 768 hidden units
- 12 attention heads 
- 66M parameters.
  
Embeddings in DistiBert:

Word embeddings are generated using a technique called word encoding. This involves representing each word in the input text as a fixed-length vector of numbers, which captures the semantic meaning of the word in a numerical form. These vectors are then used as input to the model, allowing it to process the input text and generate a response

### Implementation Details:

The Distilbert is a pre-trained model in hugging face. It is especially trained for the SQuAD Dataset and Coco Dataset. So, we have used the pretrained model to fit into our custom dataset. Here, a framework known as Haystack had to be imported inorder to use the libray FARMReader which is especially dedicated to import the model from hugging face. .
  
After importing the model, the model is fit to the training dataset that we prepared and tried to fine tune the model by tweaking some hyperparameters such as number of epochs etc. While training the model it could be observed that at some point in the training, the training is getting to the least. So, in resemblence with the pocket algorithm the model is storing the most optimised parameters where it achieved the least training error and that particular weights of the model are being stored. Now, with this model we tested it on the test file that we created. We have taken a ratio of 80:20 to split the dataset into train and test respectively. Which yielded an F1 score of 0.5418, Exact Match of 0.2247191, top_n_accuracy of 0.80898. We have given an context that is not related to california but from India. Astonishingly, the model could predict well enough. 

# Comparisons (Mainly linear vs BERT - Vineet/Devansh / Rahul for BERT comparison)

# Example Analysis (Devansh)

# Conclusions (Devansh)


# References
