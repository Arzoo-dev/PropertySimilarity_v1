import torch
import os
import json
import sys
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_paths():
    """Add project paths to system path for importing"""
    # Get the current directory
    current_dir = Path(__file__).resolve().parent
    
    # Add parent directory to path to import siamese_network
    parent_dir = current_dir.parent
    
    if str(parent_dir) not in sys.path:
        sys.path.append(str(parent_dir))
        logger.info(f"Added {parent_dir} to system path")

def load_model_config(config_path):
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def convert_model_to_torchscript(model_path, output_path, backbone="resnet50"):
    """
    Convert PyTorch model to TorchScript format for deployment
    
    Args:
        model_path: Path to the source model file
        output_path: Directory to save the TorchScript model
        backbone: Backbone to use (resnet50 is used in the API)
    """
    try:
        # Setup paths to import SiameseNetwork
        setup_paths()
        
        # Import here to avoid import errors
        from siamese_network import SiameseNetwork, SiameseEmbedding
        
        # Check if we're loading a full model or just the embedding part or state dict
        logger.info(f"Loading model from {model_path}")
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Determine what type of model data we've loaded
        if isinstance(model_data, dict) and not hasattr(model_data, 'eval'):
            logger.info("Loaded state dictionary instead of full model")
            
            # Create a new model instance with the specified backbone
            embedding_model = SiameseEmbedding(embedding_dim=256, backbone=backbone)
            
            # Try to load state dict - handle both full model and embedding-only state dicts
            try:
                # Check if this looks like a full model state dict with embedding_model
                if 'embedding_model.state_dict' in str(model_data):
                    # Extract embedding_model state dict from full model state dict
                    embedding_model_state_dict = {}
                    prefix = 'embedding_model.'
                    
                    # Extract embedding model keys
                    for key in model_data.keys():
                        if key.startswith(prefix):
                            new_key = key[len(prefix):]  # Remove the prefix
                            embedding_model_state_dict[new_key] = model_data[key]
                    
                    if embedding_model_state_dict:
                        embedding_model.load_state_dict(embedding_model_state_dict)
                        logger.info("Loaded embedding model from full model state dict")
                    else:
                        # Try direct loading as a fallback
                        embedding_model.load_state_dict(model_data)
                        logger.info("Loaded state dict directly into embedding model")
                else:
                    # Try to load directly
                    embedding_model.load_state_dict(model_data)
                    logger.info("Loaded state dict directly into embedding model")
                
            except Exception as load_e:
                logger.error(f"Error loading state dict: {str(load_e)}")
                raise
                
        elif hasattr(model_data, 'embedding_model'):
            # Full model loaded
            logger.info("Loaded complete SiameseNetwork model")
            embedding_model = model_data.embedding_model
        else:
            # Direct embedding model
            logger.info("Using loaded model directly as embedding model")
            embedding_model = model_data
        
        # Verify backbone type if possible
        if hasattr(embedding_model, 'backbone_type') and embedding_model.backbone_type != backbone:
            logger.warning(f"Loaded model uses {embedding_model.backbone_type} backbone but converting for {backbone}")
            logger.warning("This may cause compatibility issues with the API")
        
        # Set model to evaluation mode
        embedding_model.eval()
        
        # Create a sample input for tracing
        sample_input = torch.randn(1, 3, 224, 224)
        
        # Convert to TorchScript via tracing
        logger.info("Converting model to TorchScript...")
        traced_model = torch.jit.trace(embedding_model, sample_input)
        
        # Save the model
        os.makedirs(output_path, exist_ok=True)
        torchscript_path = os.path.join(output_path, "siamese_embedding_model.pt")
        traced_model.save(torchscript_path)
        logger.info(f"TorchScript model saved to {torchscript_path}")
        
        return torchscript_path
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        raise

def create_model_archive(model_dir, handler_path, output_dir):
    """
    Create a model archive (.mar) file using torch-model-archiver
    
    Args:
        model_dir: Directory containing the TorchScript model
        handler_path: Path to the custom handler script
        output_dir: Directory to save the model archive
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the model file path
        model_file = os.path.join(model_dir, "siamese_embedding_model.pt")
        
        # Build the torch-model-archiver command
        cmd = f"""
        torch-model-archiver -f \\
            --model-name model \\
            --version 1.0 \\
            --serialized-file {model_file} \\
            --handler {handler_path} \\
            --export-path {output_dir}
        """
        
        logger.info(f"Running command: {cmd}")
        os.system(cmd)
        
        logger.info(f"Model archive created at {output_dir}/model.mar")
    except Exception as e:
        logger.error(f"Error creating model archive: {str(e)}")
        raise

def get_model_path():
    """Ask user for model path interactively"""
    # Define default paths
    base_dir = Path(__file__).resolve().parent.parent
    default_model_path = base_dir / "final_model" / "siamese_embedding_model.pt"
    
    # Prompt user
    print("\n=== Model Export for Vertex AI ===")
    print(f"Default model path: {default_model_path}")
    
    user_input = input("Enter model path (or press Enter to use default): ").strip()
    
    # Use user input if provided, otherwise use default
    if user_input:
        model_path = Path(user_input)
        config_path = model_path.parent / "model_config.json"
    else:
        model_path = default_model_path
        config_path = base_dir / "final_model" / "model_config.json"
    
    # Verify path exists
    if not model_path.exists():
        print(f"WARNING: Model file not found at {model_path}")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower()
        if proceed != 'y':
            sys.exit("Export cancelled.")
    
    return str(model_path), str(config_path)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to Vertex AI deployable format")
    parser.add_argument("--model-path", type=str, help="Path to the model file")
    parser.add_argument("--config-path", type=str, help="Path to the model config file")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone to use (resnet50 or efficientnet)")
    parser.add_argument("--no-interactive", action="store_true", help="Run in non-interactive mode")
    
    return parser.parse_args()

def main():
    """Main function to convert model and create model archive"""
    # Parse command line arguments
    args = parse_args()
    
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    vertex_dir = base_dir / "VertexDeployment"
    torchscript_dir = vertex_dir / "torchscript_model"
    handler_path = vertex_dir / "model_handler.py"
    archive_dir = vertex_dir / "model_artifacts"
    
    # Get model path - either from args, interactive prompt, or default
    if args.model_path and not args.no_interactive:
        model_path = args.model_path
        config_path = args.config_path or str(Path(model_path).parent / "model_config.json")
    elif args.no_interactive and args.model_path:
        model_path = args.model_path
        config_path = args.config_path or str(Path(model_path).parent / "model_config.json")
    else:
        model_path, config_path = get_model_path()
    
    # Load model config
    try:
        config = load_model_config(config_path)
        logger.info(f"Loaded model config: {config}")
    except Exception as e:
        logger.warning(f"Could not load model config: {str(e)}")
        config = {}
    
    # Use ResNet50 backbone to match the API implementation
    backbone = args.backbone or "resnet50"
    if "backbone" in config:
        config_backbone = config.get("backbone")
        if config_backbone != backbone:
            logger.warning(f"Config specifies {config_backbone} but using {backbone} for conversion.")
    
    # Convert model to TorchScript
    torchscript_path = convert_model_to_torchscript(model_path, torchscript_dir, backbone=backbone)
    
    # Create model archive
    create_model_archive(torchscript_dir, handler_path, archive_dir)
    
    logger.info("Model export completed successfully")
    print(f"\nModel exported successfully!")
    print(f"TorchScript model: {torchscript_dir}/siamese_embedding_model.pt")
    print(f"Model Archive (.mar): {archive_dir}/model.mar")

if __name__ == "__main__":
    main() 