import os
import json
import base64
import requests
import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import config
import logging

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, use_local_model: bool = False):
        self.use_local_model = use_local_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if use_local_model:
            logger.info("Initializing local model...")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                config.LOCAL_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipeline.to(self.device)
            logger.info("Local model initialized successfully")
        else:
            logger.info("Using API-based image generation")
            if not config.STABILITY_API_KEY:
                raise ValueError("STABILITY_API_KEY is required for API-based generation")
        
        # Create output directory
        os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)
    
    def _generate_with_api(self, prompt: str, size: Tuple[int, int]) -> Optional[Image.Image]:
        """Generate image using Stability AI API"""
        try:
            url = f"https://{config.STABILITY_API_HOST}/v1/generation/{config.STABILITY_MODEL}/text-to-image"
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {config.STABILITY_API_KEY}"
            }
            
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": config.GUIDANCE_SCALE,
                "height": size[1],
                "width": size[0],
                "samples": 1,
                "steps": config.NUM_INFERENCE_STEPS
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=config.REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.text}")
                return None
            
            data = response.json()
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            
            return Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            logger.error(f"Error generating image with API: {str(e)}")
            return None
    
    def generate_image(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        size: Tuple[int, int] = config.IMAGE_SIZE,
        num_inference_steps: int = config.NUM_INFERENCE_STEPS
    ) -> Optional[Image.Image]:
        """Generate a single image from text prompt"""
        try:
            if self.use_local_model:
                logger.info("Generating image with local model")
                image = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    width=size[0],
                    height=size[1]
                ).images[0]
            else:
                logger.info("Generating image with API")
                image = self._generate_with_api(prompt, size)
            
            if image is None:
                return None
            
            if output_path:
                image.save(output_path, format=config.IMAGE_FORMAT, quality=config.IMAGE_QUALITY)
                logger.info(f"Image saved to {output_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_dataset(
        self,
        prompts: list,
        output_prefix: str = "image",
        size: Tuple[int, int] = config.IMAGE_SIZE
    ) -> list:
        """Generate multiple images from a list of prompts"""
        generated_images = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            try:
                output_path = os.path.join(
                    config.IMAGE_OUTPUT_DIR,
                    f"{output_prefix}_{i}.{config.IMAGE_FORMAT.lower()}"
                )
                
                image = self.generate_image(
                    prompt=prompt,
                    output_path=output_path,
                    size=size
                )
                
                if image is not None:
                    generated_images.append({
                        "id": f"image_{i}",
                        "prompt": prompt,
                        "path": output_path,
                        "size": size,
                        "format": config.IMAGE_FORMAT
                    })
                    logger.info(f"Successfully generated image {i}")
                else:
                    logger.warning(f"Failed to generate image {i}")
                
            except Exception as e:
                logger.error(f"Error generating image {i}: {str(e)}")
                continue
        
        return generated_images
    
    def save_metadata(self, metadata: list, filename: str = "image_metadata.json"):
        """Save image generation metadata to file"""
        try:
            filepath = os.path.join(config.IMAGE_OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Metadata saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}") 