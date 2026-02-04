"""
Prepare meta.pkl for arithmetic dataset from input_arithmetic.txt
Creates a character-level vocabulary (stoi/itos) and saves meta.pkl
in the same folder as this script.
"""
import os
import pickle

def build_meta_from_input(input_path: str, output_dir: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    # create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"✓ Saved meta.pkl to {meta_path}")
    print(f"vocab_size: {vocab_size}")


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    input_dir = os.path.join(script_dir, '..\data\\arithmetic')
    input_path = os.path.join(input_dir, 'input_arithmetic.txt')
    build_meta_from_input(input_path=input_path, output_dir=script_dir)
