"""
Main script to run the Transformer Language Model training and generation
"""

import torch
import os
import sys

# Add src directory to path
sys.path.append('src')

from config import Config
from train import train_model, save_model_checkpoint
from generate import generate_multiple_samples, save_generation_samples, interactive_generation, evaluate_model_quality

def main():
    """
    Main function to orchestrate training and generation
    """
    print("🚀 TRANSFORMER FROM SCRATCH")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check if CUDA is available and print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    try:
        # Stage 1: Training
        print("\n🔥 Stage 1: Training the Transformer Model")
        print("-" * 60)
        
        model, char_to_idx, idx_to_char, vocab_size = train_model(config)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(config.output_dir, "transformer_checkpoint.pth")
        save_model_checkpoint(model, char_to_idx, idx_to_char, config, checkpoint_path)
        
        # Stage 2: Text Generation
        print("\n✨ Stage 2: Generating Text Samples")
        print("-" * 60)
        
        # Generate multiple samples
        samples = generate_multiple_samples(model, char_to_idx, idx_to_char, config, device)
        
        # Save samples to file
        save_generation_samples(samples, config)
        
        # Stage 3: Model Evaluation
        print("\n📊 Stage 3: Model Quality Evaluation")
        print("-" * 60)
        
        test_prompts = evaluate_model_quality(model, char_to_idx, idx_to_char, config, device)
        
        # Stage 4: Interactive Generation (Optional)
        print("\n🎮 Stage 4: Interactive Generation (Optional)")
        print("-" * 60)
        print("Would you like to try interactive text generation? (y/n): ", end="")
        
        try:
            user_choice = input().strip().lower()
            if user_choice == 'y' or user_choice == 'yes':
                interactive_generation(model, char_to_idx, idx_to_char, config, device)
        except KeyboardInterrupt:
            print("\nInteractive generation skipped.")
        
        # Final Summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING AND GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📁 Files created:")
        print(f"   • Training data: {os.path.join(config.data_dir, 'sample_text.txt')}")
        print(f"   • Model checkpoint: {checkpoint_path}")
        print(f"   • Training logs: {os.path.join(config.log_dir, 'training.log')}")
        print(f"   • Training curves: {os.path.join(config.output_dir, 'training_curves.png')}")
        print(f"   • Generated samples: {os.path.join(config.output_dir, 'generated_samples.txt')}")
        
        print(f"\n📊 Model Statistics:")
        print(f"   • Parameters: {model.count_parameters():,}")
        print(f"   • Vocabulary size: {vocab_size}")
        print(f"   • Architecture: {config.n_layers} layers, {config.n_heads} heads, {config.d_model}d model")
        
        print(f"\n🎯 Next Steps:")
        print(f"   • Check generated samples in: {config.output_dir}")
        print(f"   • View training curves: {os.path.join(config.output_dir, 'training_curves.png')}")
        print(f"   • Experiment with different hyperparameters in config.py")
        print(f"   • Try longer training or different datasets")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Partial results may be available in the output directory.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check your setup and try again.")
        raise
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def quick_demo():
    """
    Quick demo version with reduced parameters for fast testing
    """
    print("🚀 TRANSFORMER QUICK DEMO")
    print("=" * 40)
    
    # Create a demo config with smaller parameters
    config = Config()
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 3
    config.num_epochs = 20
    config.generation_length = 100
    
    print("Running quick demo with reduced parameters...")
    print(f"Model: {config.d_model}d, {config.n_heads} heads, {config.n_layers} layers")
    print(f"Training: {config.num_epochs} epochs")
    
    # Run the main process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model, char_to_idx, idx_to_char, vocab_size = train_model(config)
        samples = generate_multiple_samples(model, char_to_idx, idx_to_char, config, device, num_samples=2)
        save_generation_samples(samples, config)
        
        print("\n🎉 Quick demo complete! Check the outputs folder for results.")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        quick_demo()
    else:
        main()