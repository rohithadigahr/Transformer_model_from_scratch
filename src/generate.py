"""
Text generation using trained Transformer model
"""

import torch
import torch.nn.functional as F
import os
from data_utils import create_causal_mask, save_text_to_file

def generate_text(model, char_to_idx, idx_to_char, seed_text, length, temperature, seq_len, device):
    """
    Generate text using the trained model
    
    Args:
        model: Trained Transformer model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Starting text for generation
        length: Number of characters to generate
        temperature: Sampling temperature (higher = more random)
        seq_len: Maximum sequence length
        device: Device to run on
    
    Returns:
        generated_text: Complete generated text including seed
    """
    model.eval()
    
    with torch.no_grad():
        # Convert seed text to indices
        input_seq = torch.tensor([char_to_idx[c] for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
        generated_text = seed_text
        
        # Generate characters one by one
        for _ in range(length):
            # Get current sequence (last seq_len characters)
            current_seq = input_seq[:, -seq_len:] if input_seq.size(1) > seq_len else input_seq
            
            # Create causal mask
            current_len = current_seq.size(1)
            mask = create_causal_mask(current_len, device)
            
            # Forward pass
            output = model(current_seq, mask)
            
            # Get probabilities for next character
            next_char_logits = output[0, -1, :]
            next_char_probs = F.softmax(next_char_logits / temperature, dim=0)
            
            # Sample next character
            next_char_idx = torch.multinomial(next_char_probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated_text += next_char
            
            # Update input sequence
            next_char_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            input_seq = torch.cat([input_seq, next_char_tensor], dim=1)
    
    return generated_text

def generate_with_top_k(model, char_to_idx, idx_to_char, seed_text, length, temperature, top_k, seq_len, device):
    """
    Generate text with top-k sampling
    """
    model.eval()
    
    with torch.no_grad():
        input_seq = torch.tensor([char_to_idx[c] for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
        generated_text = seed_text
        
        for _ in range(length):
            current_seq = input_seq[:, -seq_len:] if input_seq.size(1) > seq_len else input_seq
            current_len = current_seq.size(1)
            mask = create_causal_mask(current_len, device)
            
            output = model(current_seq, mask)
            next_char_logits = output[0, -1, :]
            
            # Apply top-k filtering
            if top_k > 0:
                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(next_char_logits, top_k)
                
                # Create a new tensor with -inf for non-top-k elements
                filtered_logits = torch.full_like(next_char_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_values
                
                next_char_probs = F.softmax(filtered_logits / temperature, dim=0)
            else:
                next_char_probs = F.softmax(next_char_logits / temperature, dim=0)
            
            next_char_idx = torch.multinomial(next_char_probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated_text += next_char
            
            next_char_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            input_seq = torch.cat([input_seq, next_char_tensor], dim=1)
    
    return generated_text

def generate_multiple_samples(model, char_to_idx, idx_to_char, config, device, num_samples=3):
    """
    Generate multiple text samples with different settings
    """
    print("\n" + "="*60)
    print("TEXT GENERATION SAMPLES")
    print("="*60)
    
    samples = []
    
    # Sample 1: Low temperature (more focused)
    print(f"\nSample 1 - Low Temperature (0.5):")
    print("-" * 40)
    sample1 = generate_text(
        model, char_to_idx, idx_to_char, 
        config.seed_text, config.generation_length, 
        temperature=0.5, seq_len=config.seq_len, device=device
    )
    print(sample1)
    samples.append(("Low Temperature (0.5)", sample1))
    
    # Sample 2: Medium temperature (balanced)
    print(f"\nSample 2 - Medium Temperature ({config.temperature}):")
    print("-" * 40)
    sample2 = generate_text(
        model, char_to_idx, idx_to_char, 
        config.seed_text, config.generation_length, 
        temperature=config.temperature, seq_len=config.seq_len, device=device
    )
    print(sample2)
    samples.append((f"Medium Temperature ({config.temperature})", sample2))
    
    # Sample 3: High temperature (more random)
    print(f"\nSample 3 - High Temperature (1.2):")
    print("-" * 40)
    sample3 = generate_text(
        model, char_to_idx, idx_to_char, 
        config.seed_text, config.generation_length, 
        temperature=1.2, seq_len=config.seq_len, device=device
    )
    print(sample3)
    samples.append(("High Temperature (1.2)", sample3))
    
    # Sample 4: Top-k sampling
    print(f"\nSample 4 - Top-K Sampling (k=10):")
    print("-" * 40)
    sample4 = generate_with_top_k(
        model, char_to_idx, idx_to_char, 
        config.seed_text, config.generation_length, 
        temperature=config.temperature, top_k=10, 
        seq_len=config.seq_len, device=device
    )
    print(sample4)
    samples.append(("Top-K Sampling (k=10)", sample4))
    
    return samples

def save_generation_samples(samples, config):
    """
    Save generated samples to file
    """
    output_path = os.path.join(config.output_dir, "generated_samples.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TRANSFORMER MODEL - GENERATED TEXT SAMPLES\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (method, text) in enumerate(samples, 1):
            f.write(f"Sample {i} - {method}:\n")
            f.write("-" * 40 + "\n")
            f.write(text + "\n\n")
    
    print(f"\nGenerated samples saved to: {output_path}")

def interactive_generation(model, char_to_idx, idx_to_char, config, device):
    """
    Interactive text generation
    """
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter seed text (or 'quit' to exit)")
    print("Parameters can be adjusted by typing: temp=0.8 length=100")
    
    current_temp = config.temperature
    current_length = config.generation_length
    
    while True:
        user_input = input(f"\nSeed text (temp={current_temp}, len={current_length}): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Check for parameter adjustments
        if '=' in user_input:
            try:
                if user_input.startswith('temp='):
                    current_temp = float(user_input.split('=')[1])
                    print(f"Temperature set to {current_temp}")
                    continue
                elif user_input.startswith('length='):
                    current_length = int(user_input.split('=')[1])
                    print(f"Generation length set to {current_length}")
                    continue
            except:
                print("Invalid parameter format. Use: temp=0.8 or length=100")
                continue
        
        if len(user_input) == 0:
            user_input = config.seed_text
        
        # Check if all characters in seed text are in vocabulary
        if not all(c in char_to_idx for c in user_input):
            print("Warning: Some characters in seed text are not in vocabulary!")
            # Filter out unknown characters
            user_input = ''.join(c for c in user_input if c in char_to_idx)
            if len(user_input) == 0:
                print("No valid characters found. Using default seed.")
                user_input = config.seed_text
        
        print("\nGenerating...")
        try:
            generated = generate_text(
                model, char_to_idx, idx_to_char, 
                user_input, current_length, current_temp, 
                config.seq_len, device
            )
            print(f"\nGenerated text:\n{generated}")
        except Exception as e:
            print(f"Error during generation: {e}")

def evaluate_model_quality(model, char_to_idx, idx_to_char, config, device, test_prompts=None):
    """
    Evaluate model quality with different prompts
    """
    if test_prompts is None:
        test_prompts = [
            "to be or not",
            "the king of",
            "in the night",
            "love is",
            "death comes"
        ]
    
    print("\n" + "="*60)
    print("MODEL QUALITY EVALUATION")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i} - Prompt: '{prompt}'")
        print("-" * 40)
        
        try:
            generated = generate_text(
                model, char_to_idx, idx_to_char, 
                prompt, 100, config.temperature, 
                config.seq_len, device
            )
            print(generated)
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
    
    return test_prompts