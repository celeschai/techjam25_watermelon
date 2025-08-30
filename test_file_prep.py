#!/usr/bin/env python3
"""
Script to split a CSV file into non-overlapping test and validation sets.
Takes a CSV file, shuffles the data, and outputs two separate CSV files.
"""

import pandas as pd
import argparse
import random
import os
from pathlib import Path


def split_csv(input_file, test_size, val_size, output_dir=".", seed=None):
    """
    Split a CSV file into test and validation sets.
    
    Args:
        input_file (str): Path to input CSV file
        test_size (int): Number of rows for test set
        val_size (int): Number of rows for validation set
        output_dir (str): Directory to save output files
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Paths to test and validation CSV files
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        pd.util.hash_pandas_object = lambda x: hash(tuple(x))
    
    # Read the CSV file
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}")
    
    # Check if requested sizes are valid
    if test_size + val_size > total_rows:
        raise ValueError(f"Requested sizes ({test_size} + {val_size} = {test_size + val_size}) "
                        f"exceed total rows ({total_rows})")
    
    # Shuffle the dataframe
    print("Shuffling data...")
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split into test and validation sets
    test_df = df_shuffled.iloc[:test_size]
    val_df = df_shuffled.iloc[test_size:test_size + val_size]
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    input_stem = Path(input_file).stem
    test_file = os.path.join(output_dir, f"{input_stem}_test.csv")
    val_file = os.path.join(output_dir, f"{input_stem}_validation.csv")
    
    # Save the splits
    print(f"Saving test set ({len(test_df)} rows) to: {test_file}")
    test_df.to_csv(test_file, index=False)
    
    print(f"Saving validation set ({len(val_df)} rows) to: {val_file}")
    val_df.to_csv(val_file, index=False)
    
    # Print summary
    print(f"\nSplit complete!")
    print(f"Test set: {len(test_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Remaining rows: {total_rows - test_size - val_size}")
    
    return test_file, val_file


def main():
    """Main function to handle command line arguments and execute the split."""
    parser = argparse.ArgumentParser(
        description="Split a CSV file into non-overlapping test and validation sets"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "test_size", 
        type=int, 
        help="Number of rows for test set"
    )
    parser.add_argument(
        "val_size", 
        type=int, 
        help="Number of rows for validation set"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        default=".", 
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--seed", 
        "-s", 
        type=int, 
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        test_file, val_file = split_csv(
            input_file=args.input_file,
            test_size=args.test_size,
            val_size=args.val_size,
            output_dir=args.output_dir,
            seed=args.seed
        )
        print(f"\nFiles created successfully!")
        print(f"Test set: {test_file}")
        print(f"Validation set: {val_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
