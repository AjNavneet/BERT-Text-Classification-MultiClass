# Learning rate for training
lr = 1e-3

# Sequence length (often used for sequences in NLP)
seq_len = 20

# Dropout rate for regularization
dropout = 0.5

# Number of training epochs
num_epochs = 2

# Column name for labels in the dataset
label_col = "Product"

# Path to the file containing tokenized data
tokens_path = "Output/tokens.pkl"

# Path to the file containing label data
labels_path = "Output/labels.pkl"

# Path to the input data file (complaints dataset)
data_path = "Input/complaints.csv"

# Path to save the pre-trained model
model_path = "Output/bert_pre_trained.pth"

# Column name for text data in the dataset
text_col_name = "Consumer complaint narrative"

# Path to the label encoder file
label_encoder_path = "Output/label_encoder.pkl"

# Mapping of product categories to their corresponding labels
product_map = {
    'Vehicle loan or lease': 'vehicle_loan',
    'Credit reporting, credit repair services, or other personal consumer reports': 'credit_report',
    'Credit card or prepaid card': 'card',
    'Money transfer, virtual currency, or money service': 'money_transfer',
    'virtual currency': 'money_transfer',
    'Mortgage': 'mortgage',
    'Payday loan, title loan, or personal loan': 'loan',
    'Debt collection': 'debt_collection',
    'Checking or savings account': 'savings_account',
    'Credit card': 'card',
    'Bank account or service': 'savings_account',
    'Credit reporting': 'credit_report',
    'Prepaid card': 'card',
    'Payday loan': 'loan',
    'Other financial service': 'others',
    'Virtual currency': 'money_transfer',
    'Student loan': 'loan',
    'Consumer Loan': 'loan',
    'Money transfers': 'money_transfer'
}
