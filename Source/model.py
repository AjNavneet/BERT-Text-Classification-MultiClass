import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel
from sklearn.metrics import accuracy_score

class BertClassifier(nn.Module):

    def __init__(self, dropout, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)  # 768 is the BERT hidden size
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # Perform a forward pass through the model
        _, bert_output = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   return_dict=False)
        dropout_output = self.activation(self.dropout(bert_output))
        final_output = self.linear(dropout_output)
        return final_output

def train(train_loader, valid_loader, model, criterion, optimizer,
          device, num_epochs, model_path):
    """
    Function to train the model
    :param train_loader: Data loader for the train dataset
    :param valid_loader: Data loader for the validation dataset
    :param model: Model object
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param device: CUDA or CPU
    :param num_epochs: Number of epochs
    :param model_path: Path to save the model
    """
    best_loss = 1e8
    for i in range(num_epochs):
        print(f"Epoch {i+1} of {num_epochs}")
        valid_loss, train_loss = [], []
        model.train()  # Set the model in training mode
        # Train loop
        for batch_labels, batch_data in tqdm(train_loader):
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = torch.squeeze(input_ids, 1)
            # Forward pass
            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Gradient update step
            optimizer.step()
        model.eval()  # Set the model in evaluation mode
        # Validation loop
        for batch_labels, batch_data in tqdm(valid_loader):
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = torch.squeeze(input_ids, 1)
            # Forward pass
            batch_output = model(input_ids, attention_mask)
            batch_output = torch.squeeze(batch_output)
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            valid_loss.append(loss.item())
        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        print(f"Train Loss: {t_loss}, Validation Loss: {v_loss}")
        if v_loss < best_loss:
            best_loss = v_loss
            # Save the model if validation loss improves
            torch.save(model.state_dict(), model_path)
        print(f"Best Validation Loss: {best_loss}")

def test(test_loader, model, criterion, device):
    """
    Function to test the model
    :param test_loader: Data loader for the test dataset
    :param model: Model object
    :param criterion: Loss function
    :param device: CUDA or CPU
    """
    model.eval()  # Set the model in evaluation mode
    test_loss = []
    test_accu = []
    for batch_labels, batch_data in tqdm(test_loader):
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        # Move data to GPU if available
        batch_labels = batch_labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        input_ids = torch.squeeze(input_ids, 1)
        # Forward pass
        batch_output = model(input_ids, attention_mask)
        batch_output = torch.squeeze(batch_output)
        # Calculate loss
        loss = criterion(batch_output, batch_labels)
        test_loss.append(loss.item())
        batch_preds = torch.argmax(batch_output, axis=1)
        # Move predictions to CPU
        if torch.cuda.is_available():
            batch_labels = batch_labels.cpu()
            batch_preds = batch_preds.cpu()
        # Compute accuracy
        test_accu.append(accuracy_score(batch_labels.detach().numpy(),
                                        batch_preds.detach().numpy()))
    test_loss = np.mean(test_loss)
    test_accu = np.mean(test_accu)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc})
