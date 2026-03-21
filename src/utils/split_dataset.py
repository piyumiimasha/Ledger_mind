"""
Split Q/A pairs into training and test sets.

This script:
1. Loads Q/A pairs from uber_report_qa_pairs.jsonl
2. Shuffles them randomly
3. Splits into 80% train / 20% test
4. Saves to train.jsonl and golden_test_set.jsonl
"""

import json
import random
from pathlib import Path
from datetime import datetime


def split_qa_dataset(
    input_file: Path,
    train_file: Path,
    test_file: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    Split Q/A pairs into training and test sets.

    Args:
        input_file: Path to input JSONL file with Q/A pairs
        train_file: Path to output training JSONL file
        test_file: Path to output test JSONL file
        train_ratio: Fraction of data for training (default: 0.8)
        random_seed: Random seed for reproducibility (default: 42)
    """
    # Load all Q/A pairs
    print(f"Loading Q/A pairs from: {input_file}")
    qa_pairs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    print(f"✅ Loaded {len(qa_pairs)} Q/A pairs")

    # Shuffle randomly with seed for reproducibility
    random.seed(random_seed)
    random.shuffle(qa_pairs)
    print(f"✅ Shuffled data (seed={random_seed})")

    # Split into train/test
    split_idx = int(len(qa_pairs) * train_ratio)
    train_pairs = qa_pairs[:split_idx]
    test_pairs = qa_pairs[split_idx:]

    print(f"\nSplit Summary:")
    print(f"  Total pairs: {len(qa_pairs)}")
    print(f"  Training: {len(train_pairs)} ({len(train_pairs)/len(qa_pairs)*100:.1f}%)")
    print(f"  Test: {len(test_pairs)} ({len(test_pairs)/len(qa_pairs)*100:.1f}%)")

    # Save training set
    print(f"\nSaving training set to: {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"✅ Saved {len(train_pairs)} training pairs")

    # Save test set
    print(f"\nSaving test set to: {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"✅ Saved {len(test_pairs)} test pairs")

    # Save split metadata
    metadata = {
        "source_file": str(input_file),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "total_pairs": len(qa_pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "split_at": datetime.now().isoformat()
    }

    metadata_file = train_file.parent / "split_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Saved split metadata to: {metadata_file}")

    # Display samples
    print(f"\n{'='*80}")
    print("SAMPLE FROM TRAINING SET")
    print(f"{'='*80}")
    for i, pair in enumerate(train_pairs[:3], 1):
        print(f"\n{i}. Q: {pair['question']}")
        print(f"   A: {pair['answer']}")
        print(f"   ID: {pair['qa_pair_id']}")

    print(f"\n{'='*80}")
    print("SAMPLE FROM TEST SET")
    print(f"{'='*80}")
    for i, pair in enumerate(test_pairs[:3], 1):
        print(f"\n{i}. Q: {pair['question']}")
        print(f"   A: {pair['answer']}")
        print(f"   ID: {pair['qa_pair_id']}")

    print(f"\n{'='*80}")
    print("✅ Dataset split complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Define paths
    data_dir = Path("data/qa_pairs")
    input_file = data_dir / "uber_report_qa_pairs.jsonl"
    train_file = data_dir / "train.jsonl"
    test_file = data_dir / "golden_test_set.jsonl"

    # Check if input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"   Please run notebooks/02_generate_qa_pairs.ipynb first to generate Q/A pairs.")
        exit(1)

    # Split the dataset
    split_qa_dataset(
        input_file=input_file,
        train_file=train_file,
        test_file=test_file,
        train_ratio=0.8,
        random_seed=42
    )
