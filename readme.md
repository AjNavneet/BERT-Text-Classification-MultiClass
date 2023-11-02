# Multi-Class Text Classification with BERT Model

### Business Overview

In this project, we take a different approach by utilizing a pre-trained model known as BERT (Bidirectional Encoder Representations from Transformers) for text classification. BERT is an open-source machine learning framework for Natural Language Processing (NLP) developed by Google, known for its state-of-the-art performance in a wide range of NLP tasks.

---

### Aim

The main objective of this project is to perform multi-class text classification on a dataset using the pre-trained BERT model.

---

### Data Description

The dataset comprises more than two million customer complaints about consumer financial products. It includes columns for the actual text of the complaint and the product category associated with each complaint. Pre-trained word vectors from the GloVe dataset (glove.6B) are used to enhance text representation.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `torch`, `nltk`, `numpy`, `pickle`, `re`, `tqdm`, `sklearn`, `transformers`

---

## Approach

1. Install the necessary packages using the `pip` command.
2. Import the required libraries.
3. Define configuration file paths.
4. Process Text data:
   - Read the CSV file and remove null values.
   - Handle duplicate labels.
   - Encode the label column and save the encoder and encoded labels.
5. Data Preprocessing:
   - Convert text to lowercase.
   - Remove punctuation.
   - Eliminate digits.
   - Remove consecutive instances of 'x'.
   - Remove additional spaces.
   - Tokenize the text.
   - Save the tokens.
6. Model:
   - Create the BERT model.
   - Define a function for the PyTorch dataset.
   - Create functions to train and test the model.
7. Train the BERT model:
   - Load the necessary files.
   - Split data into train, test, and validation sets.
   - Create PyTorch datasets.
   - Create data loaders.
   - Create the model object.
   - Define the loss function and optimizer.
   - Move the model to GPU if available.
   - Train the model.
   - Test the model.
8. Make predictions on new text.

---

## Modular Code Overview

1. **Input**: Contains data required for analysis, including:
   - `complaints.csv`

2. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `data.py`
   - `utils.py`

   These Python files contain helpful functions used in the `Engine.py` file.

3. **Output**: Contains files required for model training, including:
   - `bert_pre_trained.pth`
   - `label_encoder.pkl`
   - `labels.pkl`
   - `tokens.pkl`

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, which trains the model and saves it in the output folder.

---

## Key Concepts Explored

1. Understanding the business problem.
2. Introduction to pre-trained models.
3. Understanding how BERT works.
4. Data preparation for the BERT model.
5. Removing spaces and digits.
6. Punctuation removal.
7. BERT tokenization.
8. Architecture of the BERT model.
9. Creating a data loader for the BERT model.
10. Building the BERT model.
11. Training the pre-trained BERT model using GPU or CPU.
12. Making predictions on new text data.


---
