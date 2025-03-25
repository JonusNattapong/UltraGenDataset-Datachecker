import os
import json
from typing import Dict, List, Any
from PIL import Image
import numpy as np
from tqdm import tqdm
import config

class DatasetChecker:
    def __init__(self):
        self.config = config.CHECKER_CONFIG
        
    def check_image(self, image_path: str) -> Dict[str, Any]:
        """Check individual image quality"""
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in self.config["allowed_formats"]:
                    return {
                        "status": "error",
                        "message": f"Invalid format: {img.format}",
                        "path": image_path
                    }
                
                # Check size
                width, height = img.size
                min_w, min_h = self.config["min_image_size"]
                max_w, max_h = self.config["max_image_size"]
                
                if width < min_w or height < min_h or width > max_w or height > max_h:
                    return {
                        "status": "error",
                        "message": f"Invalid size: {width}x{height}",
                        "path": image_path
                    }
                
                # Check image quality
                img_array = np.array(img)
                if img_array.size == 0:
                    return {
                        "status": "error",
                        "message": "Empty image",
                        "path": image_path
                    }
                
                # Check for corrupted images
                if np.isnan(img_array).any() or np.isinf(img_array).any():
                    return {
                        "status": "error",
                        "message": "Corrupted image data",
                        "path": image_path
                    }
                
                return {
                    "status": "success",
                    "path": image_path,
                    "size": img.size,
                    "format": img.format
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "path": image_path
            }
    
    def check_text(self, text: str) -> Dict[str, Any]:
        """Check individual text quality"""
        try:
            # Check length
            if len(text) < self.config["text_min_length"]:
                return {
                    "status": "error",
                    "message": f"Text too short: {len(text)} chars"
                }
            
            if len(text) > self.config["text_max_length"]:
                return {
                    "status": "error",
                    "message": f"Text too long: {len(text)} chars"
                }
            
            # Check for empty or whitespace-only text
            if not text.strip():
                return {
                    "status": "error",
                    "message": "Empty or whitespace-only text"
                }
            
            return {
                "status": "success",
                "length": len(text)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def check_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Check entire dataset quality"""
        results = {
            "total_samples": 0,
            "valid_samples": 0,
            "errors": [],
            "details": []
        }
        
        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            results["total_samples"] = len(dataset)
            
            # Check each sample
            for sample in tqdm(dataset, desc="Checking dataset"):
                sample_result = {
                    "id": sample.get("id", "unknown"),
                    "status": "success",
                    "errors": []
                }
                
                # Check text if present
                if "text" in sample:
                    text_check = self.check_text(sample["text"])
                    if text_check["status"] == "error":
                        sample_result["status"] = "error"
                        sample_result["errors"].append(text_check["message"])
                
                # Check image if present
                if "image_path" in sample:
                    image_check = self.check_image(sample["image_path"])
                    if image_check["status"] == "error":
                        sample_result["status"] = "error"
                        sample_result["errors"].append(image_check["message"])
                
                results["details"].append(sample_result)
                
                if sample_result["status"] == "success":
                    results["valid_samples"] += 1
                else:
                    results["errors"].extend(sample_result["errors"])
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check dataset: {str(e)}"
            }
    
    def save_check_results(self, results: Dict[str, Any], output_path: str):
        """Save check results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2) 