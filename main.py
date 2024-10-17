import torch
import torch.nn as nn
import numpy as np

DIMENSIONS = 10000  # HDC typically uses high dimensions
SPARSITY = 0.05     # Fraction of active bits in the sparse vector

def generate_hdc_vector(dimensions=DIMENSIONS, sparsity=SPARSITY):
    """Generates a high-dimensional vector with given sparsity."""
    vector = np.zeros(dimensions)
    num_active_bits = int(sparsity * dimensions)
    active_indices = np.random.choice(dimensions, num_active_bits, replace=False)
    vector[active_indices] = 1
    return vector

class HDCEncoder(nn.Module):
    """HDC-based Encoder that replaces the traditional embedding layer."""
    def __init__(self, vocab_size):
        super(HDCEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.hdc_vocab = np.array([generate_hdc_vector() for _ in range(vocab_size)])

    def forward(self, tokens):
        """Encode tokens into high-dimensional vectors."""
        hdc_vectors = [self.hdc_vocab[token.item()] for token in tokens]
        hdc_vectors = torch.tensor(hdc_vectors, dtype=torch.float32)
        return hdc_vectors

class HDCDecoder(nn.Module):
    """HDC-based Decoder that replaces the traditional output layer."""
    def __init__(self, hdc_vocab):
        super(HDCDecoder, self).__init__()
        self.hdc_vocab = hdc_vocab

    def forward(self, encoded_vector):
        """Decode high-dimensional vectors to token indices."""
        similarities = np.dot(encoded_vector, self.hdc_vocab.T)  # Use dot product for similarity
        decoded_token = torch.argmax(torch.tensor(similarities))
        return decoded_token
