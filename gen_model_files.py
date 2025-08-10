import mmap 
import struct
import os
import argparse

from scripts.config import Config
from scripts.tokenizer import Tokenizer
from scripts.weights import Weights

MODEL_FILE_PATH = 'scripts/input/stories260K.bin'
TOKENIZER_FILE_PATH = 'scripts/input/tok512.bin'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model files from checkpoints and tokenizer data.")
    parser.add_argument("--checkpoint", 
                        default=MODEL_FILE_PATH, 
                        help="Path to the model checkpoint file. Default is 'stories260K.bin'.")
    parser.add_argument("--tokenizer", 
                        default=TOKENIZER_FILE_PATH, 
                        help="Path to the tokenizer file. Default is 'tok512.bin'.")
    args = parser.parse_args()

    config = Config()
<<<<<<< HEAD
    config.read_checkpoint(args.checkpoint, "src/config.bin")

    tokenizer = Tokenizer()
    tokenizer.build_tokenizer(args.tokenizer, config.vocab_size)
    tokenizer.save_tokenizer("src/tokenizer.bin")
    tokenizer.free_tokenizer()

    weights = Weights()
    weights.read_weights(args.checkpoint, "src/weights.psp")
=======
    config.read_checkpoint(args.checkpoint, "config.bin")

    tokenizer = Tokenizer()
    tokenizer.build_tokenizer(args.tokenizer, config.vocab_size)
    tokenizer.save_tokenizer("tokenizer.bin")
    tokenizer.free_tokenizer()

    weights = Weights()
    weights.read_weights(args.checkpoint, "weights.psp")
>>>>>>> test

    print(f"Tokenizer saved to tokenizer.bin")
    print(f"Config saved to config.bin")
    print(f"Weights saved as PSP image to weights.psp")