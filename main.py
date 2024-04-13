import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from src.transformer import TransformerClassifier
from src.finetuner import LoRALayer, RoSALayer
from src.dataset import AmazonReviewsDataset
from src.train import train_model
from src.tokenizer import SimpleTokenizer
import pickle

def build_vocab_from_dataset(dataset, num_words=10000):
    word_freq = Counter()
    for review in dataset.reviews:
        words = review.lower().split()
        word_freq.update(words)
    most_common_words = [word for word, freq in word_freq.most_common(num_words)]
    return most_common_words

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model on Amazon Reviews with optional adaptation layers.")
    
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--embed_size", type=int, default=512, help="Embedding size.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--forward_expansion", type=int, default=4, help="Forward expansion size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--vocab_size", type=int, default=10002, help="Vocabulary size.")
    parser.add_argument("--adaptation_layer", type=str, default=None, choices=[None, "LoRA", "RoSA"], help="Adaptation layer to use.")
    parser.add_argument("--rank", type=int, default=4, help="Rank for LoRA/RoSA.")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity for RoSA.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = AmazonReviewsDataset(filepath='/content/drive/MyDrive/project/finetune/data/sample/train_sample.ft1.txt', tokenizer=None)
    vocab = build_vocab_from_dataset(train_dataset, num_words=10000)
    tokenizer = SimpleTokenizer(vocab=vocab, max_length=args.max_length)
    train_dataset = AmazonReviewsDataset(filepath='/data/sample/train_sample.ft1.txt', tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = AmazonReviewsDataset(filepath='data/sample/test_sample.ft1.txt', tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    with open('/data/simple_tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)
    model = TransformerClassifier(
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        forward_expansion=args.forward_expansion,
        num_layers=args.num_layers,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        device=device,
        adaptation_layer=args.adaptation_layer,
        rank=args.rank,
        sparsity=args.sparsity
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_model(model, train_loader, val_loader, args.epochs, optimizer, loss_fn, device)

if __name__ == "__main__":
    main()
