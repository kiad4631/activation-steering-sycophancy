#!/usr/bin/env python3
"""
Comprehensive evaluation of trained steering vector.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.data_utils import load_and_prepare_data
from src.evaluation import evaluate_model, evaluate_multiple_multipliers
from src.visualization import plot_multiplier_effects, plot_comparison_bars, save_results_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained steering vector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"Model name or path (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--steering",
        type=str,
        required=True,
        help="Path to trained steering vector (.pt file)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=config.TEST_DATA_PATH,
        help="Path to test data"
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=config.DEFAULT_MULTIPLIERS,
        help="Multiplier values to test"
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
        help="Don't display plots"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVALUATING STEERING VECTOR")
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
    
    # Load steering vector
    print(f"\nLoading steering vector from {args.steering}")
    steering_vec = torch.load(args.steering)
    print("✓ Steering vector loaded")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}")
    _, test_dataset = load_and_prepare_data(
        config.TRAIN_DATA_PATH,
        args.test_data
    )
    print(f"✓ Loaded {len(test_dataset)} test examples")
    
    # Evaluate baseline
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION (No Steering)")
    print("=" * 80)
    baseline_score = evaluate_model(
        model, tokenizer, test_dataset,
        batch_size=args.batch_size,
        show_progress=True
    )
    print(f"\nBaseline Score: {baseline_score:.3f}")
    print("(Higher = more non-sycophantic/honest)")
    
    # Evaluate with different multipliers
    print("\n" + "=" * 80)
    print("STEERING EVALUATION")
    print("=" * 80)
    multiplier_scores = evaluate_multiple_multipliers(
        model,
        tokenizer,
        steering_vec,
        test_dataset,
        multipliers=args.multipliers,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Baseline (no steering): {baseline_score:.3f}")
    for mult, score in sorted(multiplier_scores.items()):
        delta = score - baseline_score
        direction = "↑" if delta > 0 else "↓"
        print(f"Multiplier {mult:+.1f}: {score:.3f} ({direction} {abs(delta):.3f})")
    
    # Find best multiplier for reducing sycophancy
    best_mult_reduce = max(multiplier_scores, key=multiplier_scores.get)
    print(f"\n🎯 Best for REDUCING sycophancy: {best_mult_reduce:+.1f} (score: {multiplier_scores[best_mult_reduce]:.3f})")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "baseline_score": baseline_score,
        "multiplier_scores": {str(k): v for k, v in multiplier_scores.items()},
        "best_multiplier_reduce": best_mult_reduce,
        "best_score_reduce": multiplier_scores[best_mult_reduce],
        "config": {
            "model": args.model,
            "steering_vector": args.steering,
            "test_size": len(test_dataset),
            "batch_size": args.batch_size
        }
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Multiplier effects plot
    mult_plot_path = os.path.join(args.output_dir, "plots", "multiplier_effects.png")
    plot_multiplier_effects(
        multiplier_scores,
        baseline_score=baseline_score,
        save_path=mult_plot_path,
        show=not args.no_plot
    )
    
    # Comparison bar plot
    comparison_data = {
        "Baseline": baseline_score,
        f"Steered ({best_mult_reduce:+.1f})": multiplier_scores[best_mult_reduce]
    }
    comp_plot_path = os.path.join(args.output_dir, "plots", "baseline_vs_steered.png")
    plot_comparison_bars(
        comparison_data,
        title="Baseline vs Best Steering",
        ylabel="Non-Sycophantic Score",
        save_path=comp_plot_path,
        show=not args.no_plot
    )
    
    # Save text summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    save_results_summary(results, summary_path)
    
    print(f"✓ All results saved to {args.output_dir}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()