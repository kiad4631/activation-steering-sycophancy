"""
Data loading and preprocessing utilities.
"""
import json
import random
from typing import Dict, List, Tuple

import config


def load_json_data(filepath: str) -> List[Dict[str, str]]:
    """Load JSON data from a file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def make_pos_neg_pair(mwe_data: Dict[str, str]) -> Tuple[str, str]:
    """Creates a (positive, negative) pair for getting contrastive activations.
    
    Args:
        mwe_data: Dictionary containing question and answer fields
        
    Returns:
        Tuple of (non-sycophantic prompt, sycophantic prompt)
    """
    # Positive = Non-sycophantic (honest answer)
    pos = config.PROMPT_TEMPLATE.format(
        question=mwe_data["question"], 
        answer=mwe_data["answer_not_matching_behavior"]
    )
    # Negative = Sycophantic (agrees with user bias)
    neg = config.PROMPT_TEMPLATE.format(
        question=mwe_data["question"], 
        answer=mwe_data["answer_matching_behavior"]
    )
    return pos, neg


def make_dataset(list_mwe_data: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Creates a list of (positive, negative) pairs for training/evaluation.
    
    Args:
        list_mwe_data: List of dictionaries with questions and answers
        
    Returns:
        List of (non-sycophantic, sycophantic) prompt pairs
    """
    return [make_pos_neg_pair(mwe_data) for mwe_data in list_mwe_data]


def load_and_prepare_data(
    train_path: str = config.TRAIN_DATA_PATH,
    test_path: str = config.TEST_DATA_PATH,
    shuffle: bool = True,
    seed: int = config.RANDOM_SEED,
    limit_train: int = None,
    limit_test: int = None
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load and prepare training and test datasets.
    
    Args:
        train_path: Path to training data JSON
        test_path: Path to test data JSON
        shuffle: Whether to shuffle the data
        seed: Random seed for shuffling
        limit_train: Limit number of training examples (for quick testing)
        limit_test: Limit number of test examples
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load raw data
    train_data = load_json_data(train_path)
    test_data = load_json_data(test_path)
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(train_data)
        random.shuffle(test_data)
    
    # Apply limits
    if limit_train:
        train_data = train_data[:limit_train]
    if limit_test:
        test_data = test_data[:limit_test]
    
    # Convert to prompt pairs
    train_dataset = make_dataset(train_data)
    test_dataset = make_dataset(test_data)
    
    return train_dataset, test_dataset


def print_example(dataset: List[Tuple[str, str]], index: int = 0):
    """Print an example from the dataset for inspection.
    
    Args:
        dataset: List of (positive, negative) prompt pairs
        index: Index of example to print
    """
    pos, neg = dataset[index]
    print("=" * 80)
    print("EXAMPLE FROM DATASET")
    print("=" * 80)
    print("\n### NON-SYCOPHANTIC (Positive) ###")
    print(pos)
    print("\n### SYCOPHANTIC (Negative) ###")
    print(neg)
    print("=" * 80)