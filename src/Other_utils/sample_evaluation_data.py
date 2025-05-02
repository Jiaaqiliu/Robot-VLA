#!/usr/bin/env python3
import json
import random
import os
from pathlib import Path

def sample_json_data(input_path, output_path, sample_size=1000, seed=42):
    """
    Randomly sample entries from a JSON file and save them to a new location.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path to save the sampled JSON file
        sample_size (int): Number of entries to sample
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read the input JSON file
        print(f"Reading data from {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a list
        if not isinstance(data, list):
            raise ValueError("Input JSON must contain a list of entries")
        
        # Get total number of entries
        total_entries = len(data)
        print(f"Total entries in input file: {total_entries}")
        
        # Adjust sample size if it's larger than the total entries
        if sample_size > total_entries:
            print(f"Warning: Sample size {sample_size} is larger than total entries {total_entries}")
            print(f"Adjusting sample size to {total_entries}")
            sample_size = total_entries
        
        # Randomly sample entries
        sampled_data = random.sample(data, sample_size)
        print(f"Sampled {len(sampled_data)} entries")
        
        # Save sampled data to output file
        print(f"Saving sampled data to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
        
        print("Done!")
        
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Define paths
    task_name = "future_prediction_task"
    input_path = f"/home/jqliu/Myprojects/Robot-VLA-R1/{task_name}_video_update.json"
    output_path = f"/home/jqliu/Myprojects/Video-R1/src/r1-v/Evaluation/eval_{task_name}.json"
    
    # Sample the data
    sample_json_data(input_path, output_path) 