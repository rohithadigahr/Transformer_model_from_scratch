"""
Dynamic Configuration with multiple text options
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from text_collections import TextCollections
except ImportError:
    # Fallback if text_collections is not available
    class TextCollections:
        @staticmethod
        def get_mixed_collection():
            return "fallback text content" * 100

class Config:
    # Model Architecture
    d_model = 256           # Model dimension
    n_heads = 8             # Number of attention heads
    n_layers = 6            # Number of encoder layers
    d_ff = 1024             # Feed-forward dimension
    max_seq_len = 5000      # Maximum sequence length for positional encoding
    dropout = 0.1           # Dropout rate
    
    # Training Parameters
    seq_len = 64            # Input sequence length
    batch_size = 32         # Batch size
    learning_rate = 0.0001  # Learning rate
    num_epochs = 5        # Number of training epochs
    
    # Generation Parameters
    generation_length = 200 # Number of characters to generate
    temperature = 0.8       # Temperature for sampling (higher = more random)
    seed_text = "once upon a time"  # Seed text for generation
    
    # File Paths
    data_dir = "data/"
    output_dir = "outputs/"
    log_dir = "logs/"
    
    # TEXT SELECTION OPTIONS
    # Choose your training text by uncommenting one of these options:
    
    # Option 1: Mixed genres (RECOMMENDED for diverse output)
    sample_text = TextCollections.get_mixed_collection(['classic', 'fantasy', 'scifi', 'mystery'])
    
    # Option 2: Single genre focus
    # sample_text = TextCollections.get_fantasy_adventure() * 5      # Fantasy stories
    # sample_text = TextCollections.get_science_fiction() * 5       # Sci-fi content
    # sample_text = TextCollections.get_mystery_detective() * 5     # Mystery stories
    # sample_text = TextCollections.get_classic_literature() * 5    # Classic literature
    # sample_text = TextCollections.get_poetry_collection() * 8     # Poetry and verse
    # sample_text = TextCollections.get_historical_fiction() * 5    # Historical settings
    # sample_text = TextCollections.get_conversation_data() * 6     # Dialogue focused
    
    # Option 3: Custom combinations
    # sample_text = (TextCollections.get_fantasy_adventure() * 3 + 
    #               TextCollections.get_science_fiction() * 3 +
    #               TextCollections.get_poetry_collection() * 2)
    
    # Option 4: Your own text (replace with your content)
    # sample_text = """
    # Your custom text here...
    # This can be anything you want to train on.
    # Make sure it's at least 10,000+ characters.
    # """ * 10
    
    @classmethod
    def get_text_stats(cls):
        """Get statistics about the selected text"""
        text = cls.sample_text
        return {
            'total_chars': len(text),
            'unique_chars': len(set(text)),
            'estimated_sequences': len(text) - cls.seq_len,
            'sample_preview': text[:200] + "..." if len(text) > 200 else text
        }
    
    @classmethod
    def print_text_info(cls):
        """Print information about the selected training text"""
        stats = cls.get_text_stats()
        print(f"Training Text Statistics:")
        print(f"  Total characters: {stats['total_chars']:,}")
        print(f"  Unique characters: {stats['unique_chars']}")
        print(f"  Training sequences: {stats['estimated_sequences']:,}")
        print(f"  Preview: {stats['sample_preview']}")
    
    @classmethod
    def set_genre(cls, genre):
        """Dynamically set the text genre"""
        genre_map = {
            'mixed': lambda: TextCollections.get_mixed_collection(),
            'fantasy': lambda: TextCollections.get_fantasy_adventure() * 5,
            'scifi': lambda: TextCollections.get_science_fiction() * 5,
            'mystery': lambda: TextCollections.get_mystery_detective() * 5,
            'classic': lambda: TextCollections.get_classic_literature() * 5,
            'poetry': lambda: TextCollections.get_poetry_collection() * 8,
            'historical': lambda: TextCollections.get_historical_fiction() * 5,
            'conversation': lambda: TextCollections.get_conversation_data() * 6
        }
        
        if genre in genre_map:
            cls.sample_text = genre_map[genre]()
            print(f"✅ Text genre set to: {genre}")
            cls.print_text_info()
        else:
            print(f"❌ Unknown genre: {genre}")
            print(f"Available genres: {list(genre_map.keys())}")

# Quick demo configuration for faster testing
class QuickConfig(Config):
    d_model = 128
    n_heads = 4
    n_layers = 3
    num_epochs = 20
    generation_length = 100
    sample_text = TextCollections.get_fantasy_adventure() * 2  # Smaller dataset for demo