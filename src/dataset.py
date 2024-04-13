from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from src.tokenizer import SimpleTokenizer

class AmazonReviewsDataset(Dataset):
    def __init__(self, filepath, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.reviews = []
            self.targets = []
            for line in lines:
                label, review = line.split(' ', 1)
                self.targets.append(1 if label == '__label__2' else 0)
                self.reviews.append(review.strip())
        
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        token_ids = self.tokenizer.tokenize_and_pad(review)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return token_ids, target

