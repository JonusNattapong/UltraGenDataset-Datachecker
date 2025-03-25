import os
import argparse
import logging
import pandas as pd
from generator import DatasetGenerator
from image_generator import ImageGenerator
from dataset_checker import DatasetChecker
from auto_fixer import DatasetAutoFixer
from problem_analyzer import ProblemAnalyzer
import config

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

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load dataset from JSON file into pandas DataFrame"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = pd.read_json(f)
    return data

def save_dataset(data: pd.DataFrame, output_path: str):
    """Save DataFrame to JSON file"""
    data.to_json(output_path, orient='records', force_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate dataset with text and images")
    parser.add_argument("--text-only", action="store_true", help="Generate only text dataset")
    parser.add_argument("--image-only", action="store_true", help="Generate only image dataset")
    parser.add_argument("--use-local-model", action="store_true", help="Use local model for image generation")
    parser.add_argument("--num-samples", type=int, default=config.NUM_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt template for generation")
    parser.add_argument("--check-only", action="store_true", help="Only check existing dataset")
    parser.add_argument("--dataset-path", type=str, help="Path to existing dataset for checking")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix dataset issues")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset problems")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECK_RESULTS_DIR, exist_ok=True)
    
    # Initialize components
    fixer = DatasetAutoFixer()
    analyzer = ProblemAnalyzer()
    
    if args.check_only:
        if not args.dataset_path:
            logger.error("Dataset path is required for checking")
            return
            
        logger.info(f"Checking dataset at {args.dataset_path}")
        
        # Load dataset into DataFrame
        data = load_dataset(args.dataset_path)
        
        # Initialize dataset checker
        checker = DatasetChecker(data, name=os.path.basename(args.dataset_path))
        
        # Run comprehensive quality check
        report = checker.run_quality_check(
            check_missing=True,
            check_outliers=True,
            check_duplicates=True,
            check_format=True,
            check_balance=True,
            check_distribution=True
        )
        
        # Save check results
        output_path = os.path.join(
            config.CHECK_RESULTS_DIR,
            f"check_results_{os.path.basename(args.dataset_path)}"
        )
        report.to_json(output_path)
        report.to_html(output_path.replace('.json', '.html'))
        
        logger.info(f"Dataset check completed. Overall quality score: {report.overall_score:.2f}")
        logger.info(f"Check results saved to {output_path}")
        
        if report.overall_score < 0.8:
            logger.warning("Dataset quality needs improvement")
            
            # Analyze problems if requested
            if args.analyze:
                logger.info("Analyzing dataset problems...")
                analysis_results = analyzer.analyze_dataset(args.dataset_path)
                analyzer.save_analysis_results(
                    analysis_results,
                    os.path.join(config.CHECK_RESULTS_DIR, f"analysis_{os.path.basename(args.dataset_path)}")
                )
                logger.info("Problem analysis completed")
            
            # Auto-fix if requested
            if args.auto_fix:
                logger.info("Starting automatic fix...")
                fix_results = fixer.fix_dataset(args.dataset_path)
                
                # Save fix results
                fix_output_path = os.path.join(
                    config.CHECK_RESULTS_DIR,
                    f"fix_results_{os.path.basename(args.dataset_path)}"
                )
                fixer.save_fix_results(fix_results, fix_output_path)
                
                logger.info(f"Auto-fix completed. Fixed: {fix_results['fixed_samples']}, Skipped: {fix_results['skipped_samples']}")
                
                # Check fixed dataset
                fixed_dataset_path = args.dataset_path.replace('.json', '_fixed.json')
                if os.path.exists(fixed_dataset_path):
                    logger.info("Checking fixed dataset...")
                    fixed_data = load_dataset(fixed_dataset_path)
                    fixed_checker = DatasetChecker(fixed_data, name=f"{os.path.basename(args.dataset_path)}_fixed")
                    fixed_report = fixed_checker.run_quality_check()
                    
                    # Save fixed dataset check results
                    fixed_check_path = os.path.join(
                        config.CHECK_RESULTS_DIR,
                        f"fixed_check_results_{os.path.basename(args.dataset_path)}"
                    )
                    fixed_report.to_json(fixed_check_path)
                    fixed_report.to_html(fixed_check_path.replace('.json', '.html'))
                    
                    logger.info(f"Fixed dataset check completed. Overall quality score: {fixed_report.overall_score:.2f}")
                    
                    # Analyze fixed dataset if requested
                    if args.analyze:
                        logger.info("Analyzing fixed dataset problems...")
                        fixed_analysis_results = analyzer.analyze_dataset(fixed_dataset_path)
                        analyzer.save_analysis_results(
                            fixed_analysis_results,
                            os.path.join(config.CHECK_RESULTS_DIR, f"fixed_analysis_{os.path.basename(args.dataset_path)}")
                        )
                        logger.info("Fixed dataset problem analysis completed")
        return
    
    # Generate text dataset
    if not args.image_only:
        logger.info("Generating text dataset...")
        text_generator = DatasetGenerator()
        text_dataset = text_generator.generate_dataset(
            prompt_template=args.prompt,
            num_samples=args.num_samples
        )
        text_generator.save_dataset(text_dataset)
        logger.info(f"Generated {len(text_dataset)} text samples")
        
        # Check text dataset
        text_data = load_dataset(os.path.join(config.TEXT_OUTPUT_DIR, "dataset.json"))
        text_checker = DatasetChecker(text_data, name="text_dataset")
        text_report = text_checker.run_quality_check()
        text_report.to_json(os.path.join(config.CHECK_RESULTS_DIR, "text_check_results.json"))
        text_report.to_html(os.path.join(config.CHECK_RESULTS_DIR, "text_check_results.html"))
        
        # Analyze problems if requested
        if args.analyze:
            logger.info("Analyzing text dataset problems...")
            text_analysis_results = analyzer.analyze_dataset(os.path.join(config.TEXT_OUTPUT_DIR, "dataset.json"))
            analyzer.save_analysis_results(
                text_analysis_results,
                os.path.join(config.CHECK_RESULTS_DIR, "text_analysis.json")
            )
            logger.info("Text dataset problem analysis completed")
        
        # Auto-fix if requested
        if args.auto_fix and text_report.overall_score < 0.8:
            logger.info("Starting automatic fix for text dataset...")
            text_fix_results = fixer.fix_dataset(os.path.join(config.TEXT_OUTPUT_DIR, "dataset.json"))
            fixer.save_fix_results(
                text_fix_results,
                os.path.join(config.CHECK_RESULTS_DIR, "text_fix_results.json")
            )
    
    # Generate image dataset
    if not args.text_only:
        logger.info("Generating image dataset...")
        image_generator = ImageGenerator(use_local_model=args.use_local_model)
        
        # Use text prompts from text dataset if available
        if not args.image_only and 'text_dataset' in locals():
            prompts = [sample["text"] for sample in text_dataset]
        else:
            # Generate prompts using text generator
            text_generator = DatasetGenerator()
            prompts = text_generator.generate_text(args.prompt, args.num_samples)
        
        image_dataset = image_generator.generate_dataset(
            prompts=prompts,
            output_prefix="image"
        )
        image_generator.save_metadata(image_dataset)
        logger.info(f"Generated {len(image_dataset)} images")
        
        # Check image dataset
        image_data = load_dataset(os.path.join(config.IMAGE_OUTPUT_DIR, "image_metadata.json"))
        image_checker = DatasetChecker(image_data, name="image_dataset")
        image_report = image_checker.run_quality_check()
        image_report.to_json(os.path.join(config.CHECK_RESULTS_DIR, "image_check_results.json"))
        image_report.to_html(os.path.join(config.CHECK_RESULTS_DIR, "image_check_results.html"))
        
        # Analyze problems if requested
        if args.analyze:
            logger.info("Analyzing image dataset problems...")
            image_analysis_results = analyzer.analyze_dataset(os.path.join(config.IMAGE_OUTPUT_DIR, "image_metadata.json"))
            analyzer.save_analysis_results(
                image_analysis_results,
                os.path.join(config.CHECK_RESULTS_DIR, "image_analysis.json")
            )
            logger.info("Image dataset problem analysis completed")
        
        # Auto-fix if requested
        if args.auto_fix and image_report.overall_score < 0.8:
            logger.info("Starting automatic fix for image dataset...")
            image_fix_results = fixer.fix_dataset(os.path.join(config.IMAGE_OUTPUT_DIR, "image_metadata.json"))
            fixer.save_fix_results(
                image_fix_results,
                os.path.join(config.CHECK_RESULTS_DIR, "image_fix_results.json")
            )
    
    logger.info("Dataset generation completed successfully")

if __name__ == "__main__":
    main() 