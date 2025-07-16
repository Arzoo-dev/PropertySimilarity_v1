# visual_comparison_following_train_validation_split.py
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import json
import sys
import time
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from models.model_builder import DINOv2Retrieval
from dataset.train_dataset import TripletDataset
from utils.common import (
    TeeOutput, 
    load_config, 
    get_all_batch_files, 
    setup_analysis_logging
)
from utils.dataset_utils import (
    create_validation_dataset_same_as_train,
    extract_features_from_validation_dataset
)
from tqdm import tqdm



def plot_comparison_from_validation_set(pretrained_features, custom_features, batch_name, save_path="tsne_output/"):
    """Plot side-by-side t-SNE comparison using validation set"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data
    def prepare_data(features_dict):
        all_features = []
        all_labels = []
        for category in ['anchor', 'positive', 'negative']:
            for feature in features_dict[category]:
                    all_features.append(feature)
                    all_labels.append(category)
        return np.array(all_features), all_labels
    
    pre_features, pre_labels = prepare_data(pretrained_features)
    custom_features_arr, custom_labels = prepare_data(custom_features)
    
    print(f"Pre-trained features shape: {pre_features.shape}")
    print(f"Custom features shape: {custom_features_arr.shape}")
    
    # Apply PCA first if needed
    if pre_features.shape[1] > 50:
        print("Applying PCA dimensionality reduction...")
        pca = PCA(n_components=50, random_state=42)
        pre_features = pca.fit_transform(pre_features)
        custom_features_arr = pca.fit_transform(custom_features_arr)
    
    # Run t-SNE
    print("Running t-SNE for pre-trained model...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(pre_features)//4), random_state=42, verbose=1)
    pre_tsne = tsne.fit_transform(pre_features)
    
    print("Running t-SNE for custom model...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(custom_features_arr)//4), random_state=42, verbose=1)
    custom_tsne = tsne.fit_transform(custom_features_arr)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = {'anchor': 'blue', 'positive': 'green', 'negative': 'red'}
    
    # Pre-trained
    for category in ['anchor', 'positive', 'negative']:
        mask = [label == category for label in pre_labels]
        if any(mask):
            ax1.scatter(pre_tsne[mask, 0], pre_tsne[mask, 1], 
                       c=colors[category], label=category, alpha=0.7, s=30)
    ax1.set_title('Pre-trained DINOv2 (Validation Set)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Custom
    for category in ['anchor', 'positive', 'negative']:
        mask = [label == category for label in custom_labels]
        if any(mask):
            ax2.scatter(custom_tsne[mask, 0], custom_tsne[mask, 1], 
                       c=colors[category], label=category, alpha=0.7, s=30)
    ax2.set_title(f'Custom Trained Model (Validation Set)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Validation Set t-SNE Comparison - {batch_name}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(save_path, f"validation_tsne_comparison_{batch_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved comparison to {output_file}")
    
    # Save results data for analysis
    results_data = {
        'batch_name': batch_name,
        'pretrained_stats': {
            'total_features': len(pre_features),
            'anchor_count': len([l for l in pre_labels if l == 'anchor']),
            'positive_count': len([l for l in pre_labels if l == 'positive']),
            'negative_count': len([l for l in pre_labels if l == 'negative']),
        },
        'custom_stats': {
            'total_features': len(custom_features_arr),
            'anchor_count': len([l for l in custom_labels if l == 'anchor']),
            'positive_count': len([l for l in custom_labels if l == 'positive']),
            'negative_count': len([l for l in custom_labels if l == 'negative']),
        }
    }
    
    results_file = os.path.join(save_path, f"validation_analysis_{batch_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved analysis data to {results_file}")
    
    return output_file

def process_single_batch(batch_file_path, config, device):
    """Process a single batch file and return results"""
    batch_name = os.path.splitext(os.path.basename(batch_file_path))[0]
    
    # Setup logging for this specific batch
    tee_output, log_path = setup_analysis_logging("tsne_output", "tsne_comparison", batch_name)
    
    try:
        print(f"Processing batch: {batch_name}")
        print(f"Batch file: {batch_file_path}")
        print(f"Device: {device}")
        print(f"Config file: config/training_config.yaml")
        print("")
        
        # Create validation dataset following train.py exactly
        print("="*60)
        print("CREATING VALIDATION DATASET (SAME AS TRAIN.PY)")
        print("="*60)
        train_dataset, val_dataset = create_validation_dataset_same_as_train(config, batch_file_path)
        if val_dataset is None:
            print("❌ Failed to create validation dataset. Skipping this batch.")
            return None
        
        # Load models
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        
        # Load pre-trained model
        print("Loading pre-trained DINOv2...")
        pretrained_model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0)
        pretrained_model.to(device).eval()
        print("✓ Pre-trained model loaded")
        
        # Load custom model
        print("Loading custom model...")
        custom_model = DINOv2Retrieval(
            model_name='vit_base_patch14_dinov2',
            pretrained=True,
            embedding_dim=768,
            dropout=0.1,
            freeze_backbone=False
        )
        
        # Load checkpoint
        checkpoint = torch.load("./testing/checkpoint_epoch_50_train_batch_8.pth", map_location=device)
        if 'model_state_dict' in checkpoint:
            custom_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            custom_model.load_state_dict(checkpoint)
        
        custom_model.to(device).eval()
        print("✓ Custom model loaded")
        
        # Extract features from the validation set
        print("\n" + "="*60)
        print("EXTRACTING FEATURES FROM VALIDATION SET")
        print("="*60)
        
        print("Extracting features with pre-trained model...")
        pretrained_features = extract_features_from_validation_dataset(pretrained_model, val_dataset, device)
        
        print("Extracting features with custom model...")
        custom_features = extract_features_from_validation_dataset(custom_model, val_dataset, device)
        
        # Generate comparison
        print("\n" + "="*60)
        print("GENERATING t-SNE COMPARISON")
        print("="*60)
        print("Generating t-SNE comparison from validation set...")
        output_file = plot_comparison_from_validation_set(pretrained_features, custom_features, batch_name)
        
        print("\n" + "="*80)
        print(f"✅ BATCH {batch_name.upper()} COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"✓ Used exact same validation set creation process as train.py")
        print(f"✓ Batch: {batch_name}")
        print(f"✓ Validation triplets: {len(val_dataset)}")
        print(f"✓ Train ratio: {config['data']['train_ratio']}")
        print(f"✓ Random seed: {config['data']['random_seed']}")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Log file saved to: {log_path}")
        print("")
        print(f"Batch {batch_name} analysis completed successfully!")
        print("="*80)
        
        # Clean up models to free memory
        del pretrained_model, custom_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'batch_name': batch_name,
            'output_file': output_file,
            'log_path': log_path,
            'validation_triplets': len(val_dataset),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED IN BATCH {batch_name}: {str(e)}")
        print("="*80)
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        print("="*80)
        
        return {
            'batch_name': batch_name,
            'output_file': None,
            'log_path': log_path,
            'validation_triplets': 0,
            'status': 'error',
            'error': str(e)
        }
        
    finally:
        # Ensure logging is properly closed for this batch
        print(f"\nClosing log file for batch {batch_name}: {os.path.basename(log_path)}")
        
        # Restore original stdout and close log file
        sys.stdout = tee_output.terminal
        tee_output.close()

def main():
    # Create tsne_output directory at the very beginning
    tsne_output_dir = "tsne_output"
    os.makedirs(tsne_output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print initial info to terminal (not logged yet)
    print("="*80)
    print("AUTOMATED t-SNE COMPARISON FOR ALL BATCHES")
    print("="*80)
    print(f"Using device: {device}")
    print(f"Output directory: {tsne_output_dir}/")
    print("")
    
    # Load configuration
    config = load_config()
    print(f"✓ Loaded configuration from config/training_config.yaml")
    
    # Get all batch files automatically
    print("="*60)
    print("DISCOVERING BATCH FILES")
    print("="*60)
    batch_files = get_all_batch_files()
    if not batch_files:
        print("❌ No batch files found. Exiting.")
        return
    
    print(f"Found {len(batch_files)} batch files:")
    for i, batch_file in enumerate(batch_files, 1):
        batch_name = os.path.splitext(os.path.basename(batch_file))[0]
        print(f"  {i}. {batch_name}")
    print("")
    
    # Ask for confirmation
    user_input = input(f"Process all {len(batch_files)} batches automatically? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Operation cancelled by user.")
        return
    
    print(f"\n🚀 Starting automated processing of {len(batch_files)} batches...")
    print("="*80)
    
    # Process all batches
    all_results = []
    successful_batches = 0
    failed_batches = 0
    
    for i, batch_file in enumerate(batch_files, 1):
        batch_name = os.path.splitext(os.path.basename(batch_file))[0]
        
        print(f"\n{'='*80}")
        print(f"PROCESSING BATCH {i}/{len(batch_files)}: {batch_name.upper()}")
        print(f"{'='*80}")
        
        # Process this batch
        result = process_single_batch(batch_file, config, device)
        all_results.append(result)
        
        if result and result['status'] == 'success':
            successful_batches += 1
            print(f"✅ Batch {batch_name} completed successfully")
        else:
            failed_batches += 1
            print(f"❌ Batch {batch_name} failed")
        
        # Print progress
        print(f"\nProgress: {i}/{len(batch_files)} batches processed")
        print(f"Successful: {successful_batches}, Failed: {failed_batches}")
        
        # Brief pause between batches
        if i < len(batch_files):
            print("Preparing for next batch...\n")
            time.sleep(2)
    
    # Create final summary
    print("\n" + "="*80)
    print("🎉 ALL BATCHES PROCESSING COMPLETE!")
    print("="*80)
    
    # Generate summary report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tsne_output_dir = "tsne_output"
    os.makedirs(tsne_output_dir, exist_ok=True)
    summary_file = os.path.join(tsne_output_dir, f"tsne_comparison_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("t-SNE COMPARISON SUMMARY REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total batches processed: {len(batch_files)}\n")
        f.write(f"Successful: {successful_batches}\n")
        f.write(f"Failed: {failed_batches}\n")
        f.write(f"Success rate: {(successful_batches/len(batch_files)*100):.1f}%\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for result in all_results:
            f.write(f"Batch: {result['batch_name']}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"Validation triplets: {result['validation_triplets']}\n")
                f.write(f"Output file: {result['output_file']}\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write(f"Log file: {result['log_path']}\n")
            f.write("-" * 40 + "\n")
    
    # Print summary to terminal
    print(f"📊 FINAL SUMMARY:")
    print(f"   Total batches: {len(batch_files)}")
    print(f"   Successful: {successful_batches}")
    print(f"   Failed: {failed_batches}")
    print(f"   Success rate: {(successful_batches/len(batch_files)*100):.1f}%")
    print(f"   Summary report: {summary_file}")
    print("")
    
    if successful_batches > 0:
        print(f"✅ Successfully processed {successful_batches} batches!")
        print("📁 All results saved to 'tsne_output/' directory")
        print("📄 Individual logs and summary saved in 'tsne_output/' directory")
    
    if failed_batches > 0:
        print(f"⚠️  {failed_batches} batches failed - check individual log files for details")
    
    print("\n" + "="*80)
    print("Automated t-SNE comparison complete!")
    print(f"📁 All outputs available in: {tsne_output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
