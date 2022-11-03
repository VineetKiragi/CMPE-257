# NLP Question-Answering model on a SQUAD dataset.
## Team Members
- Devansh
- Vineet - VineetKiragi
- Rahul - RahulRaghava1

# CMPE-257
Creating this repo to maintain and update the code for Machine Learning group Project

# Project Proposal

## Objective
### The goal of our project is to apply machine learning techniques to contextual information generation domain of answering questions automatically.

## Dataset
We are using a dataset in SQUAD format. SQUAD datasets are a perfect fit to create a Question-Answer model. There are many SQUAD dataset readily available in on the internet, upon which we can implement our Question-Answering model. But we choose to take a pre existing dataset from [here](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) and make a new dataset of own by adding few more contexts and question using the Haystack Annotation tool [link](https://www.deepset.ai/annotation-tool-for-labeling-datasets).

Steps followed to create the Dataset

1. Use the same exisitng dataset from the above [link](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset).
2. Add our own contexts and Questions and Answers using the above mentioned annotation tool.
3. Export the dataset in SQUAD format which will be json file.
## Problem Description
Question answering is a critical NLP problem and a long-standing artificial intelligence milestone. QA systems allow a user to express a question in natural language and get an immediate and brief response

We would be building a question answering model that would answer the questions related to a particular paragraph that the model has been trained on. Putting it simply, it is a reading comprehension model. Here a SQuAD(Stanford Question Answering Dataset) format dataset will be used, which contains nested dictionaries. These nested dictionaries have the title, paragraph as the keys. These would internally contain the context i.e, the description of the context. It also has questions, boolean to tell whether the questions could be answered or not, answer for the particular question etc. 

The goal would be to train and fit the model on the train dataset fro Question Answer pairs on a particular context. So that it can generate answers to unseen questions on the same context.

## Potential Methods

We are looking to import a model from hugging face library  which is custom built for SQUAD dataset types. And fit that model to our particular dataset and produce some decent accuracy figures. After researching a bit on the potential model which fit best on the Question-Answering problems with SQUAD dataset, we found Distilbert-base-Uncased-distilled-squad model fits best for such Datatypes.

DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling BERT base. This model was proposed in the blog[link](https://medium.com/huggingface/distilbert-8cf3380435b5).

The advantages of this model or the reason to choose this model over others are

-It has 40% fewer parameters than Bert-base-uncased which is an older version of the model.
-Runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.
-It is cheaper, smaller, faster, and uses less processing power(GPU) than the older versions of the model.
-It is the best-suited and latest available Question Answer problem set application with SQUAD data.

The model would learn from the context with the already defined question and answers. Later, it would be tested to answer questions that are new to the model. 

