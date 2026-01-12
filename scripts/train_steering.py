#!/usr/bin/env python3
"""
Train the final steering vector on all training data at the optimal layer.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.data_utils import load_and_prepare_data


def main():
    parser = argparse.ArgumentParser(
        description="Train final steering vector at optimal layer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Model name or path (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=config.TRAIN_DATA_PATH,
        help="Path to training data"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=config.DEFAULT_BEST_LAYER,
        help=f"Layer to apply steering (default: {config.DEFAULT_BEST_LAYER})"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layers to apply steering (overrides --layer)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for steering vector (default: steering_vectors/steering_vec_layerN.pt)"
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "pca", "logistic_regression"],
        help="Aggregation method for steering vector"
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Limit training examples (for quick testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING FINAL STEERING VECTOR")
    print("=" * 80)
    
    # Determine layers
    if args.layers:
        layers = args.layers
    else:
        layers = [args.layer]
    
    # Set output path
    if args.output is None:
        os.makedirs(config.STEERING_VECTOR_DIR, exist_ok=True)
        layer_str = "_".join(map(str, layers))
        args.output = os.path.join(
            config.STEERING_VECTOR_DIR,
            f"steering_vec_layer{layer_str}.pt"
        )
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Layers: {layers}")
    print(f"  Aggregator: {args.aggregator}")
    print(f"  Output: {args.output}")
    
    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=config.LOAD_IN_8BIT,
        device_map="auto"
    )
    print("✓ Model loaded")
    
    # Load data
    print(f"\nLoading training data from {args.train_data}")
    train_dataset, _ = load_and_prepare_data(
        args.train_data,
        config.TEST_DATA_PATH,
        limit_train=args.limit_train
    )
    print(f"✓ Loaded {len(train_dataset)} training examples")
    
    # Train steering vector
    print(f"\nTraining steering vector on layer(s) {layers}...")
    steering_vec = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        layers=layers,
        show_progress=True
    )
    print("✓ Steering vector trained")
    
    # Save steering vector
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(steering_vec, args.output)
    
    # Save metadata
    metadata = {
        "model": args.model,
        "layers": layers,
        "aggregator": args.aggregator,
        "train_size": len(train_dataset),
        "vector_path": args.output
    }
    
    metadata_path = args.output.replace(".pt", "_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved steering vector to {args.output}")
    print(f"✓ Saved metadata to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate.py --steering {args.output}")
    print(f"  2. Demo: python scripts/demo.py --steering {args.output}")


if __name__ == "__main__":
    main()