#!/usr/bin/env python3
"""
Consolidated Property Similarity Analysis Pipeline

This script consolidates the functionality from run_similarity_analysis.py, 
similarity_analysis.py, and property_comparison_tester.py into a single,
streamlined testing pipeline.
"""

import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision import transforms
import requests
from io import BytesIO
from tqdm import tqdm
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class PropertySimilarityPipeline:
    """Consolidated pipeline for property similarity analysis."""
    
    def __init__(self, config_path: str, payload_name: str = "default"):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.payload_name = payload_name
        self._setup_directories()
        self._setup_logging()  # Move this BEFORE _load_model()
        self.model = self._load_model()  # Now logger exists
        self.transform = self._setup_transforms()
        
        # State management
        self.state_file = os.path.join(self.config['output']['save_dir'], f'{payload_name}_analysis_state.json')
        self.results = []
        self.completed_properties = set()
        self._load_state()

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = {
            'model': ['checkpoint_path', 'device'],
            'data': ['image_size', 'normalize_mean', 'normalize_std'],
            'output': ['save_dir', 'visualization_dir', 'report_dir']
        }
        
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing required field '{field}' in section '{section}'")
        
        return config

    def _load_model(self) -> torch.nn.Module:
        """Load the trained DINOv2 model with weight source verification."""
        checkpoint_path = self.config['model']['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.config['model']['device'])
        print(f"✅ Checkpoint loaded from: {checkpoint_path}")
        
        # Verify weight source before proceeding
        self._verify_weight_source(checkpoint)
        
        # Load DINOv2 model architecture
        from models.model_builder import DINOv2Retrieval
        
        model = DINOv2Retrieval(
            model_name="vit_base_patch14_dinov2",
            pretrained=True,
            embedding_dim=768,  # DINOv2 base model uses 768
            dropout=0.1,
            freeze_backbone=True
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model weights loaded from 'model_state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model weights loaded directly from checkpoint")
        
        model.eval()
        model.to(self.config['model']['device'])
        
        # Final validation
        self._validate_model_ready(model)
        
        return model

    def _verify_weight_source(self, checkpoint):
        """Verify if weights are from custom training or pre-trained source."""
        print("\n" + "="*50)
        print("🔍 DINOv2 MODEL WEIGHT SOURCE VERIFICATION")
        print("="*50)
        
        # Show model file name
        model_filename = os.path.basename(self.config['model']['checkpoint_path'])
        print(f"📁 Model file: {model_filename}")
        
        # Check if this is our custom DINOv2 model
        if model_filename == "DINOv2_custom.pth":
            print("✅ CUSTOM DINOv2 MODEL DETECTED")
            print("   - This is your trained DINOv2 model")
            print("   - Optimized for property comparison")
        else:
            print("⚠️  UNKNOWN MODEL FILE")
            print(f"   - Expected: DINOv2_custom.pth")
            print(f"   - Found: {model_filename}")
        
        # Check training metadata presence
        training_indicators = ['epoch', 'optimizer_state_dict', 'loss', 'train_loss', 'val_loss']
        training_metadata = {key: checkpoint.get(key) for key in training_indicators if key in checkpoint}
        
        if training_metadata:
            print("✅ CUSTOM TRAINED MODEL DETECTED")

        
        print("="*50)

    def _validate_model_ready(self, model):
        """Final validation that model is ready for inference."""
        print("\n🔧 FINAL MODEL VALIDATION")
        print("-" * 30)
        
        # Check 1: Model mode
        mode_status = "EVALUATION" if not model.training else "TRAINING"
        print(f"✅ Model mode: {mode_status}")
        
        # Check 2: Device placement
        device = next(model.parameters()).device
        print(f"✅ Model device: {device}")
        
        
        print("✅ Model is ready for processing!")
        print("-" * 30)

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms."""
        return transforms.Compose([
            transforms.Resize(self.config['data']['image_size']),
            transforms.ToTensor(),  # This converts PIL Image to tensor
            transforms.Normalize(
                mean=self.config['data']['normalize_mean'],
                std=self.config['data']['normalize_std']
            )
        ])

    def _setup_directories(self):
        """Setup output directories."""
        dirs = [
            self.config['output']['save_dir'],
            self.config['output']['visualization_dir'],
            self.config['output']['report_dir'],
            os.path.join(self.config['output']['save_dir'], 'logs')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration to capture all logs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.config['output']['save_dir'], 'logs', f'analysis_{self.payload_name}_{timestamp}.log')
        
        # Clear any existing handlers to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure root logger to capture all logs
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True  # Override existing configuration
        )
        
        # Set specific loggers to INFO level
        logging.getLogger('models.model_builder').setLevel(logging.INFO)
        logging.getLogger('__main__').setLevel(logging.INFO)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - saving to: {log_file}")

    def _load_state(self):
        """Load previous state if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.results = state.get('results', [])
                    self.completed_properties = set(state.get('completed_properties', []))
                    self.logger.info(f"Loaded state: {len(self.results)} results, {len(self.completed_properties)} completed")
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            state = {
                'results': self.results,
                'completed_properties': list(self.completed_properties),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save state: {e}")

    def force_restart(self):
        """Force restart by clearing state."""
        self.results = []
        self.completed_properties = set()
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        self.logger.info("Forced restart: cleared all state")

    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to load image from {url}: {e}")
            raise

    def _preprocess_image(self, image_url: str) -> torch.Tensor:
        """Preprocess image for model input."""
        image = self.load_image_from_url(image_url)
        tensor = self.transform(image)
        
        # Ensure it's a tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = transforms.ToTensor()(tensor)
        
        return tensor.unsqueeze(0).to(self.config['model']['device'])

    def compute_similarity(self, img1_url: str, img2_url: str) -> float:
        """Compute similarity between two images using DINOv2."""
        try:
            # Preprocess images
            img1_tensor = self._preprocess_image(img1_url)
            img2_tensor = self._preprocess_image(img2_url)
            
            # Get embeddings from DINOv2 model
            with torch.no_grad():
                emb1 = self.model(img1_tensor)  # Shape: [1, 768]
                emb2 = self.model(img2_tensor)  # Shape: [1, 768]

            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
            
            # Convert from [-1, 1] to [0, 10] scale
            score = float(similarity.item() * 5 + 5)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0

    def process_property_images(self, image_urls: List[str]) -> List[str]:
        """Process and validate property images."""
        return [url for url in image_urls if url and url.strip()]

    def compare_properties(self, subject_images: List[str], comp_images: List[str]) -> float:
        """Compare two properties using their images."""
        if not subject_images or not comp_images:
            return 0.0
        
        similarities = []
        
        # Compare each subject image with each comp image
        for subject_img in subject_images:
            for comp_img in comp_images:
                sim = self.compute_similarity(subject_img, comp_img)
                similarities.append(sim)
        
        # Return average similarity
        return float(np.mean(similarities)) if similarities else 0.0

    def generate_visualization(self, subject_data: Dict, comp_data: Dict, similarity_score: float) -> str:
        """Generate visualization comparing subject and comp properties."""
        try:
            # Create figure
            fig, axes = plt.subplots(2, 8, figsize=(16, 8))
            fig.suptitle(f'Property Comparison - Score: {similarity_score:.2f}/10', fontsize=14)
            
            # Subject images (top row)
            subject_images = subject_data.get('photos', [])[:8]
            for i, ax in enumerate(axes[0]):
                if i < len(subject_images):
                    try:
                        img = self.load_image_from_url(subject_images[i]['url'])
                        ax.imshow(img)
                        ax.set_title(f'Subject {i+1}', fontsize=10)
                    except:
                        ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                ax.axis('off')
            
            # Comp images (bottom row)
            comp_images = comp_data.get('photos', [])[:8]
            for i, ax in enumerate(axes[1]):
                if i < len(comp_images):
                    try:
                        img = self.load_image_from_url(comp_images[i]['url'])
                        ax.imshow(img)
                        ax.set_title(f'Comp {i+1}', fontsize=10)
                    except:
                        ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                ax.axis('off')
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comparison_{comp_data.get('uid', 'unknown')}_{timestamp}.png"
            filepath = os.path.join(self.config['output']['visualization_dir'], filename)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            return ""

    def _get_classification(self, score: float) -> str:
        """Get classification based on similarity score."""
        if score >= 7.0:
            return "SIMILAR"
        elif score >= 5.0:
            return "MODERATELY SIMILAR"
        else:
            return "DISSIMILAR"

    def process_json_payload(self, json_data: Dict) -> List[Dict]:
        """Process JSON payload and analyze properties."""
        self.logger.info(f"Starting analysis of {len(json_data['comps'])} comparable properties")
        
        # Process subject property
        subject_images = self.process_property_images(
            [p['url'] for p in json_data['subject_property']['photos']]
        )
        self.logger.info(f"Processed {len(subject_images)} subject images")

        # Filter out already completed properties
        remaining_comps = [
            comp for comp in json_data['comps'] 
            if comp['uid'] not in self.completed_properties
        ]
        
        if not remaining_comps:
            self.logger.info("All properties already processed!")
            return self.results
        
        self.logger.info(f"Processing {len(remaining_comps)} remaining properties...")
        
        # Process each comparable property
        for comp_idx, comp in enumerate(tqdm(remaining_comps, desc="Processing Properties")):
            self.logger.info(f"Processing property {comp_idx + 1}/{len(remaining_comps)}: {comp['address']}")
            
            try:
                comp_images = self.process_property_images(
                    [p['url'] for p in comp['photos']]
                )
                
                # Compute similarity
                similarity_score = self.compare_properties(subject_images, comp_images)
                
                # Generate visualization
                vis_path = self.generate_visualization(
                    json_data['subject_property'], 
                    comp, 
                    similarity_score
                )
                
                # Store results
                result = {
                    "analysis_id": f"subject_vs_{comp['uid']}",
                    "similarity_score": similarity_score,
                    "classification": self._get_classification(similarity_score),
                    "subject_images": {
                        "count": len(json_data['subject_property']['photos']),
                        "address": json_data['subject_property']['address']
                    },
                    "comp_images": {
                        "count": len(comp['photos']),
                        "address": comp['address']
                    },
                    "visualization_path": vis_path,
                    "processed_at": datetime.now().isoformat()
                }
                
                self.results.append(result)
                self.completed_properties.add(comp['uid'])
                
                # Save state
                self._save_state()
                
                self.logger.info(f"Property {comp_idx + 1} completed - Score: {similarity_score:.2f}/10")
                
            except Exception as e:
                self.logger.error(f"Error processing property {comp['uid']}: {e}")
                continue

        return self.results

    def generate_analysis_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive analysis report."""
        if not results:
            return {}
        
        scores = [r['similarity_score'] for r in results]
        classifications = [r['classification'] for r in results]
        
        # Statistical analysis
        analysis = {
            "summary": {
                "total_comparisons": len(results),
                "average_similarity": float(np.mean(scores)),
                "median_similarity": float(np.median(scores)),
                "std_similarity": float(np.std(scores)),
                "min_similarity": float(np.min(scores)),
                "max_similarity": float(np.max(scores))
            },
            "classifications": {
                "similar": len([c for c in classifications if c == "SIMILAR"]),
                "moderately_similar": len([c for c in classifications if c == "MODERATELY SIMILAR"]),
                "dissimilar": len([c for c in classifications if c == "DISSIMILAR"])
            },
            "top_matches": sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:5],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis

    def save_results(self, results: List[Dict]):
        """Save analysis results."""
        if not results:
            return
        
        # Generate analysis report
        analysis = self.generate_analysis_report(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            self.config['output']['report_dir'], 
            f'analysis_report_{self.payload_name}_{timestamp}.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump({
                "analysis": analysis,
                "detailed_results": results
            }, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")

    def analyze_json_payload(self, json_data: Dict) -> Dict:
        """Main analysis function."""
        self.logger.info("Starting similarity analysis...")
        
        # Validate JSON structure
        if 'subject_property' not in json_data:
            raise ValueError("JSON missing 'subject_property' key")
        if 'comps' not in json_data:
            raise ValueError("JSON missing 'comps' key")
        
        # Process the data
        results = self.process_json_payload(json_data)
        
        # Generate analysis
        analysis = self.generate_analysis_report(results)
        
        # Save results
        self.save_results(results)
        
        return analysis

def get_json_files_from_path(directory_path: str) -> List[str]:
    """Get all JSON files from the specified directory."""
    try:
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        json_files = []
        for file in os.listdir(directory_path):
            if file.endswith('.json'):
                json_files.append(os.path.join(directory_path, file))
        
        return json_files
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return []

def select_json_files(json_files: List[str]) -> List[str]:
    """Allow user to select which JSON files to process."""
    if not json_files:
        print("❌ No JSON files found")
        return []
    
    print(f"\n📁 Found {len(json_files)} JSON files:")
    for i, file in enumerate(json_files, 1):
        file_name = os.path.basename(file)
        file_size = os.path.getsize(file) / 1024  # Size in KB
        print(f"   {i}. {file_name} ({file_size:.1f} KB)")
    
    print("\n🔧 Processing Options:")
    print("   1. Process ALL files")
    print("   2. Select specific files")
    
    while True:
        try:
            choice = input("\nSelect option (1-2): ").strip()
            if choice == '1':
                print(f"✅ Selected: ALL {len(json_files)} files")
                return json_files
            elif choice == '2':
                return select_specific_files(json_files)
            else:
                print("❌ Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return []

def select_specific_files(json_files: List[str]) -> List[str]:
    """Allow user to select specific files to process."""
    print(f"\n📋 Select files to process (enter numbers separated by commas):")
    print("   Example: 1,3,5 or 1-3,5 or 1,3-5")
    
    while True:
        try:
            selection = input(f"Enter selection (1-{len(json_files)}): ").strip()
            if not selection:
                print("❌ No selection made")
                continue
            
            selected_indices = parse_selection(selection, len(json_files))
            if selected_indices:
                selected_files = [json_files[i-1] for i in selected_indices]
                print(f"✅ Selected {len(selected_files)} files:")
                for file in selected_files:
                    print(f"   • {os.path.basename(file)}")
                return selected_files
            else:
                print("❌ Invalid selection. Please try again.")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return []

def parse_selection(selection: str, max_num: int) -> List[int]:
    """Parse user selection string into list of indices."""
    try:
        indices = set()
        parts = selection.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range like "1-3"
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                if 1 <= start <= max_num and 1 <= end <= max_num and start <= end:
                    indices.update(range(start, end + 1))
                else:
                    return []
            else:
                # Handle single number
                num = int(part)
                if 1 <= num <= max_num:
                    indices.add(num)
                else:
                    return []
        
        return sorted(list(indices))
    except ValueError:
        return []

def process_single_file(file_path: str, config_path: str) -> bool:
    """Process a single JSON file and return success status."""
    try:
        print(f"\n{'='*60}")
        print(f"🔄 Processing: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # Load JSON data
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Validate JSON structure
        if 'subject_property' not in json_data:
            print("❌ Invalid JSON: missing 'subject_property'")
            return False
        if 'comps' not in json_data:
            print("❌ Invalid JSON: missing 'comps'")
            return False
        
        print(f"✅ JSON validated: {len(json_data['comps'])} comparable properties")
        
        # Initialize pipeline
        payload_name = os.path.splitext(os.path.basename(file_path))[0]
        pipeline = PropertySimilarityPipeline(config_path, payload_name=payload_name)
        
        # Run analysis
        analysis = pipeline.analyze_json_payload(json_data)
        
        # Print summary
        print(f"\n📈 Analysis Summary for {os.path.basename(file_path)}:")
        print(f"   Total Comparisons: {analysis['summary']['total_comparisons']}")
        print(f"   Average Similarity: {analysis['summary']['average_similarity']:.2f}/10")
        print(f"   Similar Properties: {analysis['classifications']['similar']}")
        print(f"   Moderately Similar: {analysis['classifications']['moderately_similar']}")
        print(f"   Dissimilar: {analysis['classifications']['dissimilar']}")
        
        if analysis['top_matches']:
            print(f"\n🏆 Top 3 Matches:")
            for i, match in enumerate(analysis['top_matches'][:3], 1):
                print(f"   {i}. {match['comp_images']['address']} - Score: {match['similarity_score']:.2f}")
        
        print(f"✅ {os.path.basename(file_path)} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_analysis():
    """Enhanced main CLI function with custom path selection and batch processing."""
    print("🚀 Property Similarity Analysis Pipeline")
    print("=" * 50)
    
    # Configuration
    config_path = "test_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    # Step 1: Get JSON files directory
    print("\n📂 JSON Files Directory Selection:")
    print("   1. Use current directory")
    print("   2. Enter custom directory path")
    print("   3. Use fake_payloads directory")
    
    while True:
        try:
            dir_choice = input("\nSelect directory option (1-3): ").strip()
            if dir_choice == '1':
                json_dir = os.getcwd()
                print(f"✅ Using current directory: {json_dir}")
                break
            elif dir_choice == '2':
                json_dir = input("📂 Enter directory path: ").strip()
                if not json_dir:
                    print("❌ No directory path provided")
                    continue
                json_dir = os.path.abspath(json_dir)
                if not os.path.exists(json_dir):
                    print(f"❌ Directory not found: {json_dir}")
                    continue
                print(f"✅ Using custom directory: {json_dir}")
                break
            elif dir_choice == '3':
                json_dir = os.path.join(os.getcwd(), "fake_payloads")
                if not os.path.exists(json_dir):
                    print(f"❌ fake_payloads directory not found: {json_dir}")
                    continue
                print(f"✅ Using fake_payloads directory: {json_dir}")
                break
            else:
                print("❌ Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return
    
    # Step 2: Get JSON files from selected directory
    json_files = get_json_files_from_path(json_dir)
    
    if not json_files:
        print("❌ No JSON files found in the selected directory")
        return
    
    # Step 3: Let user select which files to process
    selected_files = select_json_files(json_files)
    
    if not selected_files:
        print("❌ No files selected for processing")
        return
    
    # Step 4: Process selected files
    print(f"\n🚀 Starting batch processing of {len(selected_files)} files...")
    
    successful_files = 0
    failed_files = 0
    
    for i, file_path in enumerate(selected_files, 1):
        print(f"\n📊 Processing file {i}/{len(selected_files)}")
        
        success = process_single_file(file_path, config_path)
        
        if success:
            successful_files += 1
        else:
            failed_files += 1
        
        # Add a small delay between files to prevent overwhelming the system
        if i < len(selected_files):
            time.sleep(1)
    
    # Step 5: Final summary
    print(f"\n{'='*60}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_files} files")
    print(f"❌ Failed: {failed_files} files")
    print(f"📁 Total files: {len(selected_files)}")
    
    if successful_files > 0:
        print(f"\n📁 Results saved in:")
        print(f"   • Reports: test_results/reports/")
        print(f"   • Visualizations: test_results/visualizations/")
        print(f"   • Logs: test_results/logs/")
    
    print("\n🎉 Batch processing completed!")

if __name__ == "__main__":
    run_analysis() 