"""
Data utilities for character-level language modeling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os

class CharDataset(Dataset):
    """
    Character-level dataset for language modeling
    """
    def __init__(self, text, seq_len, char_to_idx):
        self.text = text
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.text) - self.seq_len
        
    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_len + 1]
        input_seq = torch.tensor([self.char_to_idx[c] for c in chunk[:-1]], dtype=torch.long)
        target_seq = torch.tensor([self.char_to_idx[c] for c in chunk[1:]], dtype=torch.long)
        return input_seq, target_seq

def create_vocabulary(text):
    """
    Create character-to-index and index-to-character mappings
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return chars, vocab_size, char_to_idx, idx_to_char

def save_text_to_file(text, filepath):
    """
    Save text to file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def load_text_from_file(filepath):
    """
    Load text from file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_dataloader(text, seq_len, batch_size, char_to_idx, shuffle=True):
    """
    Create DataLoader for training
    """
    dataset = CharDataset(text, seq_len, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def create_causal_mask(seq_len, device):
    """
    Create causal mask for decoder (lower triangular matrix)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def print_data_stats(text, vocab_size, seq_len):
    """
    Print dataset statistics
    """
    print(f"Dataset Statistics:")
    print(f"- Text length: {len(text):,} characters")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of sequences: {len(text) - seq_len:,}")
    print(f"- Sample characters: {sorted(list(set(text)))[:20]}")