import os
import json
from typing import List, Dict, Any
from mistralai.client import MistralClient
from tqdm import tqdm
import config

class DatasetGenerator:
    def __init__(self):
        self.client = MistralClient(api_key=config.MISTRAL_API_KEY)
        self.model = config.MISTRAL_MODEL
        
        # Create output directories
        os.makedirs(config.TEXT_OUTPUT_DIR, exist_ok=True)
        
    def generate_text(self, prompt: str, num_samples: int = 1) -> List[str]:
        """Generate text samples using Mistral AI"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        responses = []
        for _ in tqdm(range(num_samples), desc="Generating text"):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error generating text: {e}")
                continue
                
        return responses
    
    def generate_dataset(self, prompt_template: str, num_samples: int = config.NUM_SAMPLES) -> List[Dict[str, Any]]:
        """Generate a complete dataset with text samples"""
        dataset = []
        
        for i in tqdm(range(0, num_samples, config.BATCH_SIZE)):
            batch_size = min(config.BATCH_SIZE, num_samples - i)
            texts = self.generate_text(prompt_template, batch_size)
            
            for text in texts:
                sample = {
                    "id": f"sample_{i}",
                    "text": text,
                    "metadata": {
                        "model": self.model,
                        "prompt": prompt_template
                    }
                }
                dataset.append(sample)
                
                # Save individual sample
                self._save_sample(sample)
                
        return dataset
    
    def _save_sample(self, sample: Dict[str, Any]):
        """Save individual sample to file"""
        filename = f"{sample['id']}.json"
        filepath = os.path.join(config.TEXT_OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
            
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "dataset.json"):
        """Save complete dataset to file"""
        filepath = os.path.join(config.TEXT_OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2) 