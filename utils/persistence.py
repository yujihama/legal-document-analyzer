import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_processing_results(data: Dict[str, Any], filename: str) -> None:
    """Save processing results to a JSON file"""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")

def load_processing_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load processing results from a JSON file if it exists"""
    filepath = Path("data") / filename
    if not filepath.exists():
        return None
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results from {filepath}: {e}")
        return None

def convert_numpy_arrays(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists back to numpy arrays where needed"""
    if isinstance(data, dict):
        return {k: convert_numpy_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_arrays(item) for item in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    return data
