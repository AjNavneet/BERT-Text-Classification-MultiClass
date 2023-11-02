import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        """
        Initialize a TextDataset.

        Args:
            tokens (list): List of tokenized text data.
            labels (list): List of corresponding labels.

        This class is designed to work as a PyTorch Dataset, which means it can be used with PyTorch's DataLoader for efficient data loading during training and evaluation.
        """
        self.tokens = tokens  # List of tokenized text data
        self.labels = labels  # List of corresponding labels

    def __len__(self):
        """
        Get the total number of data points in the dataset.

        Returns:
            int: Number of data points in the dataset.
        """
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Get a specific data point (a pair of text data and its label) from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the label and tokenized text data for the specified data point.
        """
        return self.labels[idx], self.tokens[idx]
