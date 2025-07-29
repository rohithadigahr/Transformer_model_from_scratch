"""
Training script for Transformer Language Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformer_model import TransformerLanguageModel
from data_utils import create_vocabulary, create_dataloader, create_causal_mask, print_data_stats, save_text_to_file

def train_model(config):
    """
    Main training function
    """
    # Create directories
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Save sample text to file
    text_path = os.path.join(config.data_dir, "sample_text.txt")
    save_text_to_file(config.sample_text, text_path)
    
    # Create character vocabulary
    chars, vocab_size, char_to_idx, idx_to_char = create_vocabulary(config.sample_text)
    
    # Print dataset statistics
    print("="*60)
    print("TRANSFORMER LANGUAGE MODEL - TRAINING")
    print("="*60)
    print_data_stats(config.sample_text, vocab_size, config.seq_len)
    
    # Create dataloader
    dataloader = create_dataloader(
        config.sample_text, 
        config.seq_len, 
        config.batch_size, 
        char_to_idx, 
        shuffle=True
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training tracking
    train_losses = []
    train_perplexities = []
    
    print(f"\nStarting training for {config.num_epochs} epochs...")
    print("-" * 60)
    
    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}', leave=False)
        
        for batch_idx, (input_seq, target_seq) in enumerate(pbar):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Create causal mask for autoregressive generation
            mask = create_causal_mask(config.seq_len, device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_seq, mask)
            
            # Calculate loss
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        train_losses.append(avg_loss)
        train_perplexities.append(perplexity)
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.num_epochs - 1:
            print(f'Epoch [{epoch+1:3d}/{config.num_epochs}] | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Perplexity: {perplexity:.4f}')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Save training logs
    log_path = os.path.join(config.log_dir, "training.log")
    with open(log_path, 'w') as f:
        f.write("Epoch,Loss,Perplexity\n")
        for i, (loss, ppl) in enumerate(zip(train_losses, train_perplexities)):
            f.write(f"{i+1},{loss:.6f},{ppl:.6f}\n")
    
    print(f"Training logs saved to: {log_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, train_perplexities, config.output_dir)
    
    return model, char_to_idx, idx_to_char, vocab_size

def plot_training_curves(losses, perplexities, output_dir):
    """
    Plot and save training curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True)
    
    # Perplexity curve
    ax2.plot(perplexities)
    ax2.set_title('Training Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")

def save_model_checkpoint(model, char_to_idx, idx_to_char, config, filepath):
    """
    Save model checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'config': config,
        'vocab_size': len(char_to_idx)
    }
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to: {filepath}")

def load_model_checkpoint(filepath, device):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    config = checkpoint['config']
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    vocab_size = checkpoint['vocab_size']
    
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, char_to_idx, idx_to_char