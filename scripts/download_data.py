#!/usr/bin/env python3
"""
Download the CAA sycophancy datasets.
"""
import argparse
import os
import urllib.request
from pathlib import Path

import config


def download_file(url: str, filepath: str):
    """Download a file from a URL.
    
    Args:
        url: URL to download from
        filepath: Path to save the file
    """
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Download sycophancy datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save datasets (default: data)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    train_path = os.path.join(args.output_dir, "generate_dataset.json")
    test_path = os.path.join(args.output_dir, "test_dataset_ab.json")
    
    # Download training data
    if not os.path.exists(train_path) or args.force:
        download_file(config.TRAIN_DATA_URL, train_path)
    else:
        print(f"Training data already exists at {train_path} (use --force to re-download)")
    
    # Download test data
    if not os.path.exists(test_path) or args.force:
        download_file(config.TEST_DATA_URL, test_path)
    else:
        print(f"Test data already exists at {test_path} (use --force to re-download)")
    
    print("\n✓ Dataset download complete!")
    print(f"  Training data: {train_path}")
    print(f"  Test data: {test_path}")


if __name__ == "__main__":
    main()