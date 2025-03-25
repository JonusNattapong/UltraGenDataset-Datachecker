import os
import json
from typing import Dict, List, Any
from PIL import Image
import numpy as np
from tqdm import tqdm
import config
import logging

logger = logging.getLogger(__name__)

class ProblemAnalyzer:
    def __init__(self):
        self.config = config.CHECKER_CONFIG
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image problems"""
        problems = []
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in self.config["allowed_formats"]:
                    problems.append({
                        "type": "format",
                        "message": f"Invalid format: {img.format}",
                        "current": img.format,
                        "allowed": self.config["allowed_formats"]
                    })
                
                # Check size
                width, height = img.size
                min_w, min_h = self.config["min_image_size"]
                max_w, max_h = self.config["max_image_size"]
                
                if width < min_w or height < min_h:
                    problems.append({
                        "type": "size",
                        "message": "Image too small",
                        "current": (width, height),
                        "minimum": (min_w, min_h)
                    })
                elif width > max_w or height > max_h:
                    problems.append({
                        "type": "size",
                        "message": "Image too large",
                        "current": (width, height),
                        "maximum": (max_w, max_h)
                    })
                
                # Check quality
                img_array = np.array(img)
                if img_array.size == 0:
                    problems.append({
                        "type": "quality",
                        "message": "Empty image",
                        "details": "Image has no data"
                    })
                
                if np.isnan(img_array).any() or np.isinf(img_array).any():
                    problems.append({
                        "type": "quality",
                        "message": "Corrupted image data",
                        "details": "Contains NaN or Inf values"
                    })
                
                # Check file size
                file_size = os.path.getsize(image_path)
                if file_size > self.config["max_file_size"]:
                    problems.append({
                        "type": "file_size",
                        "message": "File too large",
                        "current": file_size,
                        "maximum": self.config["max_file_size"]
                    })
                
        except Exception as e:
            problems.append({
                "type": "error",
                "message": f"Failed to analyze image: {str(e)}"
            })
        
        return {
            "path": image_path,
            "problems": problems,
            "total_problems": len(problems)
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text problems"""
        problems = []
        
        # Check length
        if len(text) < self.config["text_min_length"]:
            problems.append({
                "type": "length",
                "message": "Text too short",
                "current": len(text),
                "minimum": self.config["text_min_length"]
            })
        
        if len(text) > self.config["text_max_length"]:
            problems.append({
                "type": "length",
                "message": "Text too long",
                "current": len(text),
                "maximum": self.config["text_max_length"]
            })
        
        # Check content
        if not text.strip():
            problems.append({
                "type": "content",
                "message": "Empty or whitespace-only text",
                "details": "Text contains only whitespace characters"
            })
        
        # Check for common issues
        if text.count(' ') > len(text) * 0.5:  # More than 50% spaces
            problems.append({
                "type": "content",
                "message": "Too many spaces",
                "details": "Text contains excessive whitespace"
            })
        
        return {
            "text": text,
            "problems": problems,
            "total_problems": len(problems)
        }
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Analyze entire dataset problems"""
        results = {
            "total_samples": 0,
            "samples_with_problems": 0,
            "total_problems": 0,
            "problem_types": {},
            "samples": []
        }
        
        try:
            # Load dataset
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            results["total_samples"] = len(dataset)
            
            # Analyze each sample
            for sample in tqdm(dataset, desc="Analyzing dataset"):
                sample_analysis = {
                    "id": sample.get("id", "unknown"),
                    "problems": [],
                    "total_problems": 0
                }
                
                # Analyze text if present
                if "text" in sample:
                    text_analysis = self.analyze_text(sample["text"])
                    if text_analysis["problems"]:
                        sample_analysis["problems"].extend(text_analysis["problems"])
                        sample_analysis["total_problems"] += text_analysis["total_problems"]
                
                # Analyze image if present
                if "image_path" in sample:
                    image_analysis = self.analyze_image(sample["image_path"])
                    if image_analysis["problems"]:
                        sample_analysis["problems"].extend(image_analysis["problems"])
                        sample_analysis["total_problems"] += image_analysis["total_problems"]
                
                if sample_analysis["problems"]:
                    results["samples_with_problems"] += 1
                    results["total_problems"] += sample_analysis["total_problems"]
                    
                    # Count problem types
                    for problem in sample_analysis["problems"]:
                        problem_type = problem["type"]
                        results["problem_types"][problem_type] = results["problem_types"].get(problem_type, 0) + 1
                
                results["samples"].append(sample_analysis)
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to analyze dataset: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate human-readable report"""
        report = []
        
        # Summary
        report.append("=== Dataset Analysis Report ===\n")
        report.append(f"Total samples: {analysis_results['total_samples']}")
        report.append(f"Samples with problems: {analysis_results['samples_with_problems']}")
        report.append(f"Total problems found: {analysis_results['total_problems']}\n")
        
        # Problem types summary
        report.append("Problem Types Summary:")
        for problem_type, count in analysis_results["problem_types"].items():
            report.append(f"- {problem_type}: {count} occurrences")
        report.append("")
        
        # Detailed sample analysis
        report.append("Detailed Sample Analysis:")
        for sample in analysis_results["samples"]:
            if sample["problems"]:
                report.append(f"\nSample ID: {sample['id']}")
                report.append(f"Total problems: {sample['total_problems']}")
                for problem in sample["problems"]:
                    report.append(f"- {problem['type']}: {problem['message']}")
                    if "details" in problem:
                        report.append(f"  Details: {problem['details']}")
                    if "current" in problem and "minimum" in problem:
                        report.append(f"  Current: {problem['current']}, Minimum: {problem['minimum']}")
                    elif "current" in problem and "maximum" in problem:
                        report.append(f"  Current: {problem['current']}, Maximum: {problem['maximum']}")
        
        return "\n".join(report)
    
    def save_analysis_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save human-readable report
        report_path = output_path.replace('.json', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(results)) 