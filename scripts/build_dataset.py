import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.preprocess import process_market_data

def build_dataset():
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "nasdaq100_data.pkl"
    output_path = project_root / "data" / "market_dataset.pt"
    
    print(f"Building dataset from {raw_data_path}...")
    
    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        sys.exit(1)
        
    # Process data
    processed_data = process_market_data(str(raw_data_path))
    
    # Save to .pt file
    print(f"Saving processed dataset to {output_path}...")
    torch.save(processed_data, output_path)
    print("Dataset build complete.")

if __name__ == "__main__":
    build_dataset()
