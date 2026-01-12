#!/usr/bin/env python3
"""
Find the optimal layer for steering vector application using validation-based selection.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.model_selection import train_test_split
from steering_vectors import train_steering_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.data_utils import load_and_prepare_data
from src.evaluation import evaluate_model, evaluate_steering_effect
from src.visualization import plot_layer_response, save_results_summary


def find_best_layer(
    model,
    tokenizer,
    train_data,
    val_data,
    layer_range=None,
    baseline_score=None,
    batch_size=config.BATCH_SIZE,
    verbose=True
):
    """Sweep through layers to find the one with best steering effect.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        train_data: Training data for fitting steering vectors
        val_data: Validation data for evaluating each layer
        layer_range: Range of layers to test (default: 8-20 for 7B models)
        baseline_score: Baseline evaluation score (compute if not provided)
        batch_size: Batch size for evaluation
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_layer, layer_scores_dict, baseline_score)
    """
    if layer_range is None:
        layer_range = range(config.LAYER_RANGE_START, config.LAYER_RANGE_END)
    
    # Compute baseline if not provided
    if baseline_score is None:
        if verbose:
            print("Computing baseline score (no steering)...")
        baseline_score = evaluate_model(
            model, tokenizer, val_data,
            batch_size=batch_size,
            show_progress=verbose
        )
        if verbose:
            print(f"Baseline: {baseline_score:.3f}\n")
    
    layer_scores = {}
    
    for layer in layer_range:
        if verbose:
            print(f"Testing layer {layer}...")
        
        # Train steering vector for this layer
        steering_vec = train_steering_vector(
            model,
            tokenizer,
            train_data,
            layers=[layer],
            show_progress=False
        )
        
        # Evaluate steering effect on validation set
        effect = evaluate_steering_effect(
            model,
            tokenizer,
            steering_vec,
            val_data,
            baseline_score,
            multiplier=-1.0,
            batch_size=batch_size,
            show_progress=verbose
        )
        
        layer_scores[layer] = effect
        if verbose:
            print(f"  Layer {layer}: Effect = {effect:.3f}\n")
    
    # Find best layer
    best_layer = max(layer_scores, key=layer_scores.get)
    
    return best_layer, layer_scores, baseline_score


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal layer for steering vector"
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
        "--test-data",
        type=str,
        default=config.TEST_DATA_PATH,
        help="Path to test data (used for baseline)"
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=config.LAYER_RANGE_START,
        help="First layer to test"
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=config.LAYER_RANGE_END,
        help="Last layer to test (exclusive)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=config.TRAIN_VAL_SPLIT,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.RESULTS_DIR,
        help="Directory to save results"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't display plot (still saves it)"
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Limit training examples (for quick testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FINDING BEST LAYER FOR STEERING")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=config.LOAD_IN_8BIT,
        device_map="auto"
    )
    print("✓ Model loaded")
    
    # Load and split data
    print(f"\nLoading data from {args.train_data}")
    train_dataset, test_dataset = load_and_prepare_data(
        args.train_data,
        args.test_data,
        limit_train=args.limit_train
    )
    
    train_data, val_data = train_test_split(
        train_dataset,
        test_size=args.val_split,
        random_state=config.RANDOM_SEED
    )
    
    print(f"✓ Data loaded:")
    print(f"  Training: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    
    # Compute baseline on test set
    print("\nComputing baseline on test set...")
    baseline_score = evaluate_model(
        model, tokenizer, test_dataset,
        batch_size=args.batch_size,
        show_progress=True
    )
    print(f"✓ Baseline score: {baseline_score:.3f}")
    
    # Find best layer
    print(f"\nSearching layers {args.layer_start} to {args.layer_end-1}...")
    best_layer, layer_scores, _ = find_best_layer(
        model,
        tokenizer,
        train_data,
        val_data,
        layer_range=range(args.layer_start, args.layer_end),
        baseline_score=None,  # Will compute on validation
        batch_size=args.batch_size,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print(f"🎯 BEST LAYER: {best_layer}")
    print(f"   Steering Effect: {layer_scores[best_layer]:.3f}")
    print("=" * 80)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "best_layer": best_layer,
        "best_effect": layer_scores[best_layer],
        "baseline_score": baseline_score,
        "layer_scores": {str(k): v for k, v in layer_scores.items()},
        "config": {
            "model": args.model,
            "layer_range": f"{args.layer_start}-{args.layer_end}",
            "train_size": len(train_data),
            "val_size": len(val_data)
        }
    }
    
    results_path = os.path.join(args.output_dir, "layer_selection_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Plot results
    plot_path = os.path.join(args.output_dir, "plots", "layer_response_curve.png")
    plot_layer_response(
        layer_scores,
        save_path=plot_path,
        show=not args.no_plot
    )
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, "layer_selection_summary.txt")
    save_results_summary(results, summary_path)
    
    print(f"\n✓ All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()