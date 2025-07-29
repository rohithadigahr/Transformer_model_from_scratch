"""
Transformer Model Implementation from Scratch
Contains all core components: Attention, Multi-Head Attention, 
Positional Encoding, Feed Forward, Layer Norm, and Encoder Stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, n_heads, seq_len, d_k)
        batch_size, n_heads, seq_len, d_k = Q.size()
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(attn_output)
        
        return output

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    Consists of multi-head attention followed by feed-forward network
    Both with residual connections and layer normalization
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Stack of N Encoder Layers
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Token embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class TransformerLanguageModel(nn.Module):
    """
    Complete Transformer model for language modeling
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, n_layers, 
                                        d_ff, max_seq_len, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Pass through encoder
        encoded = self.encoder(x, mask)
        
        # Project to vocabulary size for next token prediction
        output = self.output_projection(encoded)
        
        return output
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)