# Multi-Class Text Classification with BERT Model

## Business Overview

This project leverages the **BERT (Bidirectional Encoder Representations from Transformers)** model, a state-of-the-art pre-trained Natural Language Processing (NLP) model developed by Google, to perform multi-class text classification. BERT's ability to handle contextual embeddings makes it highly effective for a wide range of NLP tasks, including this multi-class classification project.

---

## Aim

The primary objective of this project is to classify over two million customer complaints related to consumer financial products into their respective product categories. To enhance text representation, we use pre-trained word vectors from the **GloVe (Global Vectors for Word Representation)** dataset (`glove.6B`).

---

## Data Description

The dataset includes:

- **Complaint Text**: Textual data representing customer complaints.
- **Product Category**: The associated category for each complaint.

### Preprocessing Steps:
- Handle null values and duplicate labels.
- Encode the product categories into numeric labels.
- Prepare the text data for input into the BERT model by tokenizing and formatting.

---

## Tech Stack

- **Programming Language**: [Python](https://www.python.org/)
- **Libraries and Tools**:
  - [`pandas`](https://pandas.pydata.org/) for data manipulation.
  - [`torch`](https://pytorch.org/) for deep learning.
  - [`nltk`](https://www.nltk.org/) for text preprocessing.
  - [`numpy`](https://numpy.org/) for numerical operations.
  - [`transformers`](https://huggingface.co/transformers/) for BERT implementation.
  - [`sklearn`](https://scikit-learn.org/) for model evaluation.
  - [`pickle`](https://docs.python.org/3/library/pickle.html) for saving and loading models and encoders.
  - [`tqdm`](https://tqdm.github.io/) for progress bars.

---

## Approach

### 1. Installation and Imports
- Use `pip` to install the required libraries.
- Import necessary modules for data preprocessing, modeling, and evaluation.

### 2. Configuration
- Set up a `config.py` file to define paths for data, models, and parameters.

### 3. Data Preprocessing
- **Read and Clean Data**:
  - Load the dataset from a CSV file.
  - Handle null values and duplicate entries.
  - Encode labels and save the encoder for later use.

- **Text Cleaning**:
  - Convert text to lowercase.
  - Remove punctuation, digits, and unnecessary spaces.
  - Tokenize the text and save the processed tokens.

### 4. Model Development
- Define a custom BERT-based classification model using the Hugging Face Transformers library.
- Create PyTorch datasets and data loaders for efficient training and validation.
- Define training and evaluation functions.

### 5. Training the Model
- Split the dataset into training, validation, and test sets.
- Train the BERT model using:
  - **Cross-Entropy Loss** for multi-class classification.
  - **Adam Optimizer** with weight decay for optimization.
- Save the trained model and encoders.

### 6. Making Predictions
- Load the trained model and encoder.
- Preprocess new input text and make predictions.
- Decode numeric labels back into product categories.

---

## Project Structure

```plaintext
.
├── data/                                  # Input data (e.g., complaints.csv).
├── src/                                   # Source code folder.
│   ├── data.py                            # Data preprocessing functions.
│   ├── model.py                           # BERT model implementation.
│   ├── utils.py                           # Utility functions (e.g., encoding, decoding).
├── output/                                # Contains model files and encoders.
│   ├── bert_pre_trained.pth               # Trained BERT model.
│   ├── label_encoder.pkl                  # Saved label encoder.
│   ├── labels.pkl                         # Encoded labels.
│   ├── tokens.pkl                         # Tokenized text data.
├── config.py                              # Project configuration file.
├── Engine.py                              # Main script to run the project.
├── requirements.txt                       # List of dependencies.
└── README.md                              # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

To execute the pipeline, run the `Engine.py` script:

```bash
python Engine.py
```

### 4. Make Predictions

Use the trained model to classify new text data. Preprocess the input text, pass it through the model, and decode the predicted labels.

---

## Results

- **Classification Accuracy**:
  - Achieved state-of-the-art performance using BERT on multi-class text classification.
- **Efficiency**:
  - Efficient preprocessing pipeline and GPU-accelerated training.
- **Deployment-Ready**:
  - Trained model saved for easy reuse in production environments.

---

## Features of This Project?

- **State-of-the-Art Model**: Utilizes Google's BERT model for superior text classification.
- **Scalable and Modular**: Well-structured codebase for easy adaptation to other datasets.
- **Real-World Application**: Addresses a practical NLP use case with millions of customer complaints.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Abhinav Navneet
- **Email**: mailme.AbhinavN@gmail.com
- **GitHub**: [AjNavneet](https://github.com/AjNavneet)

---

## Acknowledgments

Special thanks to:

- [Hugging Face Transformers](https://huggingface.co/transformers/) for providing pre-trained BERT models.
- [GloVe Dataset](https://nlp.stanford.edu/projects/glove/) for pre-trained word embeddings.
- The Python open-source community for their excellent tools and resources.

---

