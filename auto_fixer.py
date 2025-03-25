import os
import json
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import config
import logging

logger = logging.getLogger(__name__)

class DatasetAutoFixer:
    def __init__(self):
        self.config = config.CHECKER_CONFIG
        
    def fix_image(self, image_path: str) -> Optional[str]:
        """Fix image quality issues"""
        try:
            with Image.open(image_path) as img:
                # Resize if needed
                width, height = img.size
                min_w, min_h = self.config["min_image_size"]
                max_w, max_h = self.config["max_image_size"]
                
                if width < min_w or height < min_h or width > max_w or height > max_h:
                    # Calculate new size maintaining aspect ratio
                    ratio = min(min_w/width, min_h/height)
                    new_size = (int(width*ratio), int(height*ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert format if needed
                if img.format not in self.config["allowed_formats"]:
                    img = img.convert('RGB')
                
                # Fix corrupted data
                img_array = np.array(img)
                if np.isnan(img_array).any() or np.isinf(img_array).any():
                    # Replace NaN and Inf with 0
                    img_array = np.nan_to_num(img_array, nan=0, posinf=0, neginf=0)
                    img = Image.fromarray(img_array.astype(np.uint8))
                
                # Save fixed image
                fixed_path = image_path.replace('.', '_fixed.')
                img.save(fixed_path, format=config.IMAGE_FORMAT, quality=config.IMAGE_QUALITY)
                
                return fixed_path
                
        except Exception as e:
            logger.error(f"Error fixing image {image_path}: {str(e)}")
            return None
    
    def fix_text(self, text: str) -> Optional[str]:
        """Fix text quality issues"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Truncate if too long
            if len(text) > self.config["text_max_length"]:
                text = text[:self.config["text_max_length"]]
            
            # Skip if too short
            if len(text) < self.config["text_min_length"]:
                return None
            
            return text
            
        except Exception as e:
            logger.error(f"Error fixing text: {str(e)}")
            return None
    
    def fix_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Fix entire dataset issues"""
        results = {
            "total_samples": 0,
            "fixed_samples": 0,
            "skipped_samples": 0,
            "errors": [],
            "details": []
        }
        
        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            results["total_samples"] = len(dataset)
            fixed_dataset = []
            
            # Fix each sample
            for sample in tqdm(dataset, desc="Fixing dataset"):
                sample_result = {
                    "id": sample.get("id", "unknown"),
                    "status": "success",
                    "changes": []
                }
                
                fixed_sample = sample.copy()
                
                # Fix text if present
                if "text" in sample:
                    fixed_text = self.fix_text(sample["text"])
                    if fixed_text is not None:
                        fixed_sample["text"] = fixed_text
                    else:
                        sample_result["status"] = "skipped"
                        sample_result["changes"].append("Text too short to fix")
                
                # Fix image if present
                if "image_path" in sample:
                    fixed_path = self.fix_image(sample["image_path"])
                    if fixed_path is not None:
                        fixed_sample["image_path"] = fixed_path
                        sample_result["changes"].append("Image fixed")
                    else:
                        sample_result["status"] = "skipped"
                        sample_result["changes"].append("Failed to fix image")
                
                if sample_result["status"] == "success":
                    fixed_dataset.append(fixed_sample)
                    results["fixed_samples"] += 1
                else:
                    results["skipped_samples"] += 1
                
                results["details"].append(sample_result)
            
            # Save fixed dataset
            output_path = dataset_path.replace('.json', '_fixed.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fixed_dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Fixed dataset saved to {output_path}")
            return results
            
        except Exception as e:
            error_msg = f"Failed to fix dataset: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def save_fix_results(self, results: Dict[str, Any], output_path: str):
        """Save fix results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2) 