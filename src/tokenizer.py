class SimpleTokenizer:
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        special_tokens = ["<PAD>", "<UNK>"]
        updated_vocab = special_tokens + sorted(vocab)
        self.token2idx = {token: idx for idx, token in enumerate(updated_vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(updated_vocab)}
        self.vocab_size = len(updated_vocab)

    def encode(self, text):
        tokens = text.lower().split()
        token_ids = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        token_ids = token_ids[:self.max_length]
        return token_ids

    def pad(self, token_ids):
        padded = token_ids + [self.token2idx["<PAD>"]] * (self.max_length - len(token_ids))
        return padded

    def tokenize_and_pad(self, text):
        token_ids = self.encode(text)
        padded_token_ids = self.pad(token_ids)
        return padded_token_ids

