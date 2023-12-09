# Multi-Class Text Classification with BERT Model

## Business Overview

In this project, we leverage the power of BERT (Bidirectional Encoder Representations from Transformers), a pre-trained model for Natural Language Processing (NLP), to perform multi-class text classification. BERT, developed by Google, is renowned for its state-of-the-art performance in various NLP tasks.

---

## Aim

The primary objective is to conduct multi-class text classification on a dataset comprising over two million customer complaints related to consumer financial products. This involves associating each complaint with a specific product category. To enhance text representation, we use pre-trained word vectors from the GloVe dataset (glove.6B).

---

## Data Description

The dataset includes columns for the text of the complaint and the associated product category. To handle the text, we utilize pre-trained word vectors from GloVe. The dataset will be preprocessed to handle null values, duplicate labels, and convert the text into a suitable format for model training.

---

## Tech Stack

- Language: `Python`
- Libraries: `pandas`, `torch`, `nltk`, `numpy`, `pickle`, `re`, `tqdm`, `sklearn`, `transformers`

---

## Approach

### 1. Installation and Imports

Using the `pip` command, install the necessary packages. Import required libraries for the project.

### 2. Configuration

Define configuration file paths to manage data and model-related parameters.

### 3. Process Text Data

- Read the CSV file and handle null values.
- Address duplicate labels.
- Encode the label column and save the encoder and encoded labels.

### 4. Data Preprocessing

- Convert text to lowercase.
- Remove punctuation, digits, consecutive instances of 'x', and extra spaces.
- Tokenize the text and save the tokens.

### 5. Model

- Create the BERT model.
- Define a function for the PyTorch dataset.
- Create functions to train and test the model.

### 6. Train the BERT Model

- Load necessary files.
- Split data into train, test, and validation sets.
- Create PyTorch datasets.
- Create data loaders.
- Create the model object.
- Define loss function and optimizer.
- Move the model to GPU if available.
- Train the model.
- Test the model.

### 7. Make Predictions on New Text

---

## Modular Code Overview

1. **Input**: Contains data required for analysis, including:
   - `complaints.csv`

2. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `data.py`
   - `utils.py`

   These Python files contain functions used in the `Engine.py` file.

3. **Output**: Contains files required for model training, including:
   - `bert_pre_trained.pth`
   - `label_encoder.pkl`
   - `labels.pkl`
   - `tokens.pkl`

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, training the model and saving it in the output folder.

---
