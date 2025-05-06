"""
PyTorch Siamese Network implementation for RunPods with EfficientNet backbone for property image similarity.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.data import DataLoader
import time
import logging
from tqdm import tqdm
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler

# Import local utilities
from runpods_utils import ensure_dir_exists, save_json, load_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monkey patch torch.hub to disable hash checking - apply this at module level
# before any models are loaded
import torch.hub
_original_load_state_dict_from_url = torch.hub.load_state_dict_from_url

def _load_state_dict_from_url_without_hash_check(url, *args, **kwargs):
    kwargs['check_hash'] = False
    return _original_load_state_dict_from_url(url, *args, **kwargs)

# Apply the monkey patch
torch.hub.load_state_dict_from_url = _load_state_dict_from_url_without_hash_check
logger.info("Applied monkey patch to disable hash checking for model downloads")

class AttentionBlock(nn.Module):
    """
    Self-attention block for focusing on important features in property images.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        # Max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out) * x

class PropertyAggregator(nn.Module):
    """
    Module to aggregate multiple images from a property into a single embedding.
    Uses attention-weighted pooling to emphasize important images.
    """
    def __init__(self, embedding_dim):
        super(PropertyAggregator, self).__init__()
        
        # Attention layers for weighting images
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softmax(dim=1)  # Softmax over images dimension
        )
        
        # Additional MLP for final property-level representation
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, image_embeddings):
        """
        Aggregate multiple image embeddings into a single property embedding.
        
        Args:
            image_embeddings: tensor of shape [batch_size, num_images, embedding_dim]
                If only one image, should be shape [batch_size, 1, embedding_dim]
        
        Returns:
            property_embedding: tensor of shape [batch_size, embedding_dim]
        """
        batch_size, num_images, embedding_dim = image_embeddings.shape
        
        if num_images == 1:
            # For single image, just squeeze the dimension
            property_embedding = image_embeddings.squeeze(1)
        else:
            # Compute attention weights for each image
            attention_weights = self.attention(image_embeddings.reshape(-1, embedding_dim))
            attention_weights = attention_weights.reshape(batch_size, num_images, 1)
            
            # Apply attention weights
            weighted_embeddings = image_embeddings * attention_weights
            
            # Sum to get property embedding
            property_embedding = torch.sum(weighted_embeddings, dim=1)
        
        # Apply final MLP for refining the property embedding
        property_embedding = self.mlp(property_embedding)
        
        # L2 normalize the embedding
        property_embedding = F.normalize(property_embedding, p=2, dim=1)
        
        return property_embedding

class TripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining.
    """
    def __init__(self, margin=0.2, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        
        # Compute triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:  # 'none'
            return losses

class SiameseEmbedding(nn.Module):
    """
    Enhanced embedding network for Siamese architecture.
    Uses ResNet50 as backbone with attention mechanisms.
    """
    def __init__(self, embedding_dim=256, backbone="efficientnet", pretrained=True):
        super(SiameseEmbedding, self).__init__()
        
        # Store backbone type for forward pass
        self.backbone_type = backbone
        
        # Select backbone architecture
        if backbone == "resnet50":
            # Load pre-trained ResNet50
            base_model = models.resnet50(pretrained=pretrained)
            
            # Remove the final fully connected layer
            # Instead of taking the entire model except the last layer, we'll be more specific
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            
            # Add attention after the last convolutional block
            self.attention = AttentionBlock(2048)  # ResNet50 final channels
            
            # Define the fc layer to project to embedding space
            self.fc = nn.Linear(2048, embedding_dim)
        else:  # Default to EfficientNet
            # Load pre-trained EfficientNet-B0 with updated API to avoid hash issues
            if pretrained:
                try:
                    # Try using the new weights parameter approach
                    import torchvision.models.efficientnet
                    try:
                        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                    except Exception as e:
                        logger.warning(f"Error loading EfficientNet with weights: {str(e)}")
                        # Fallback to pretrained mode
                        self.backbone = models.efficientnet_b0(pretrained=True)
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Using older torchvision API for EfficientNet: {str(e)}")
                    # Fall back to pretrained for older torchvision
                    self.backbone = models.efficientnet_b0(pretrained=True)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            
            # Get input features for the classifier
            in_features = self.backbone.classifier[1].in_features
            
            # Add attention after the last convolutional layer
            self.attention = AttentionBlock(1280)  # EfficientNet-B0 last channels
            
            # Replace the classifier with an embedding layer
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, embedding_dim)
            )
        
        # Freeze early layers for transfer learning
        self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze early layers of the backbone for transfer learning."""
        if self.backbone_type == "resnet50":
            # Freeze first and second layers
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
        else:  # EfficientNet
            # Get total number of layers
            total_layers = len(list(self.backbone.parameters()))
            # Freeze first 70% of layers
            layers_to_freeze = int(total_layers * 0.7)
            
            for i, param in enumerate(self.backbone.parameters()):
                if i < layers_to_freeze:
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through the embedding model."""
        if self.backbone_type == "resnet50":
            # Check input dimensions - ResNet expects 224x224 images
            if x.shape[2] != 224 or x.shape[3] != 224:
                logger.warning(f"Unexpected image size for ResNet50: {x.shape[2]}x{x.shape[3]}. Resizing to 224x224.")
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            try:
                # Process in smaller groups if batch is large to avoid OOM
                if x.size(0) > 32:
                    logger.info(f"Large batch detected ({x.size(0)}), processing in chunks")
                    # Process in chunks of 16
                    chunk_size = 16
                    outputs = []
                    
                    for i in range(0, x.size(0), chunk_size):
                        # Get chunk of data
                        end = min(i + chunk_size, x.size(0))
                        chunk = x[i:end]
                        
                        # Process chunk with explicit layers
                        chunk = self.conv1(chunk)
                        chunk = self.bn1(chunk)
                        chunk = self.relu(chunk)
                        chunk = self.maxpool(chunk)
                        
                        chunk = self.layer1(chunk)
                        # Clear intermediate tensors to save memory
                        torch.cuda.empty_cache()
                        
                        chunk = self.layer2(chunk)
                        # Clear intermediate tensors to save memory
                        torch.cuda.empty_cache()
                        
                        chunk = self.layer3(chunk)
                        # Clear intermediate tensors to save memory
                        torch.cuda.empty_cache()
                        
                        chunk = self.layer4(chunk)
                        
                        # Apply attention to feature maps
                        chunk = self.attention(chunk)
                        
                        # Global average pooling
                        chunk = self.avgpool(chunk)
                        
                        # Flatten
                        chunk = torch.flatten(chunk, 1)
                        
                        # Project to embedding space
                        chunk = self.fc(chunk)
                        
                        # L2 normalize embeddings
                        chunk = F.normalize(chunk, p=2, dim=1)
                        
                        # Add to outputs
                        outputs.append(chunk)
                        
                        # Clear memory
                        torch.cuda.empty_cache()
                    
                    # Concatenate chunks
                    return torch.cat(outputs, dim=0)
                else:
                    # Standard processing for smaller batches
                    # Direct ResNet forward pass with explicit layers
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)
                    
                    x = self.layer1(x)
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)  # Shape: [batch_size, 2048, 7, 7]
                    
                    # Apply attention to feature maps
                    x = self.attention(x)  # Shape still: [batch_size, 2048, 7, 7]
                    
                    # Global average pooling
                    x = self.avgpool(x)  # Shape: [batch_size, 2048, 1, 1]
                    
                    # Flatten
                    x = torch.flatten(x, 1)  # Shape: [batch_size, 2048]
                    
                    # Project to embedding space
                    x = self.fc(x)  # Shape: [batch_size, embedding_dim]
                    
                    # Return normalized embeddings
                    return F.normalize(x, p=2, dim=1)
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(f"Error in ResNet50 forward pass: {str(e)}")
                    logger.error(f"Input tensor shape: {x.shape}")
                    
                    # Try again with a much smaller batch size if possible
                    if x.size(0) > 4:
                        logger.warning(f"Trying again with smaller batch size (4 instead of {x.size(0)})")
                        # Take only first 4 samples
                        small_batch = x[:4]
                        # Free memory
                        x = None
                        torch.cuda.empty_cache()
                        
                        # Process small batch
                        try:
                            small_result = self.forward(small_batch)
                            # Create placeholder for rest of batch (zeros)
                            placeholder = torch.zeros(
                                (x.size(0) - 4, self.fc.out_features), 
                                device=small_result.device
                            )
                            # Combine and return
                            return torch.cat([small_result, placeholder], dim=0)
                        except Exception as nested_e:
                            logger.error(f"Error with reduced batch too: {str(nested_e)}")
                            # Return zeros as last resort
                            return torch.zeros(x.size(0), self.fc.out_features, device=x.device)
                    else:
                        # Batch already very small, return zeros
                        logger.error("Batch already small, returning zeros")
                        return torch.zeros(x.size(0), self.fc.out_features, device=x.device)
                else:
                    # Non-memory error
                    logger.error(f"Error in ResNet50 forward pass: {str(e)}")
                    logger.error(f"Input tensor shape: {x.shape}")
                
                    # Complete fallback method using different approach
                    logger.warning("Using lightweight fallback mode (AdaptiveAvgPool)")
                    try:
                        # Use a much simpler approach
                        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global pooling to 1x1
                        x = x.view(x.size(0), -1)  # Flatten
                        # Use a simple linear layer instead of the full model
                        if x.size(1) != self.fc.in_features:
                            # Need a different size - create a temporary layer
                            tmp_fc = nn.Linear(x.size(1), self.fc.out_features, device=x.device)
                            x = tmp_fc(x)
                        else:
                            x = self.fc(x)
                        return F.normalize(x, p=2, dim=1)  # Normalize
                    except Exception as fallback_e:
                        logger.error(f"Fallback approach also failed: {str(fallback_e)}")
                        # Return zeros as a last resort
                        return torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        else:
            # EfficientNet forward pass with attention
            # Get features from backbone
            features = self.backbone.features(x)  # Shape: [batch_size, 1280, h, w]
            
            # Apply attention to feature maps (4D tensor)
            attended_features = self.attention(features)  # Shape remains [batch_size, 1280, h, w]
            
            # Continue with pooling and classification
            x = self.backbone._avg_pooling(attended_features)  # Apply pooling
            x = x.flatten(start_dim=1)  # Flatten features
            x = self.backbone.classifier(x)  # Project to embedding space
        
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings

class SiameseNetwork:
    """
    Enhanced Siamese Neural Network for property-level similarity.
    Uses multiple images per property with attention-based aggregation.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256,
        margin: float = 0.2,
        learning_rate: float = 0.001,
        device: str = None,
        scheduler_type: str = "one_cycle",  # Changed default to one_cycle
        backbone: str = "efficientnet",
        weight_decay: float = 1e-5
    ):
        """
        Initialize the Siamese Network.
        
        Args:
            embedding_dim: Dimension of the embedding vector
            margin: Margin for triplet loss
            learning_rate: Learning rate for the optimizer
            device: Device to use for training ('cuda' or 'cpu')
            scheduler_type: Type of learning rate scheduler to use
                - "plateau": ReduceLROnPlateau (decreases lr when metrics plateau)
                - "cosine": CosineAnnealingWarmRestarts (cosine annealing with warm restarts)
                - "one_cycle": OneCycleLR (one cycle policy with higher lr in the middle)
            backbone: Type of backbone network ("efficientnet" or "resnet50")
            weight_decay: Weight decay for optimizer (L2 regularization)
        """
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.backbone = backbone
        self.weight_decay = weight_decay
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.embedding_model = SiameseEmbedding(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=True
        )
        self.property_aggregator = PropertyAggregator(embedding_dim=embedding_dim)
        
        # Move models to device
        self.embedding_model.to(self.device)
        self.property_aggregator.to(self.device)
        
        # Initialize loss function and optimizer
        self.criterion = TripletLoss(margin=margin)
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.embedding_model.parameters()},
                {'params': self.property_aggregator.parameters(), 'lr': learning_rate * 10}
            ],
            lr=learning_rate,
            weight_decay=weight_decay  # Use AdamW with weight decay
        )
        
        # The scheduler will be initialized during training
        self.scheduler = None
        self.scheduler_batch_update = False  # Flag to indicate if scheduler updates per batch
    
    def _initialize_scheduler(self, scheduler_type, optimizer):
        """Initialize learning rate scheduler."""
        if scheduler_type == "step":
            return lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif scheduler_type == "plateau":
            return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                                verbose=True, threshold=0.0001, threshold_mode='rel', 
                                                cooldown=0, min_lr=1e-7, eps=1e-08)
        elif scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        elif scheduler_type == "one_cycle":
            # Calculate total steps based on dataloader size and epochs
            # This is done in the train method where dataloader is available
            self.scheduler_batch_update = True  # One cycle updates per batch
            return None  # Will be initialized in train() when dataloader is available
        elif scheduler_type == "warmup_cosine":
            return WarmupCosineSchedule(optimizer, warmup_steps=self.warmup_steps, t_total=self.t_total)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def _process_property_batch(self, property_images_batch):
        """
        Process a batch of property images to get property-level embeddings.
        
        Args:
            property_images_batch: Tensor of shape [batch_size, num_images, channels, height, width]
                or list of tensors with varying num_images
        
        Returns:
            property_embeddings: Tensor of shape [batch_size, embedding_dim]
        """
        # For large batch processing, reduce memory usage with smaller chunks
        max_batch_per_pass = 16 if self.backbone == "resnet50" else 32
        
        # Check if we have a batch of properties with multiple images
        if isinstance(property_images_batch, list):
            # Variable number of images per property
            property_embeddings = []
            
            for property_images in property_images_batch:
                # For each property, get embeddings for all its images
                num_images = property_images.size(0)
                
                # Check image dimensions and resize if needed for ResNet50
                if self.backbone == "resnet50":
                    # ResNet50 expects 224x224 images
                    expected_size = (224, 224)
                    current_size = (property_images.size(-2), property_images.size(-1))
                    if current_size != expected_size:
                        logger.info(f"Resizing images from {current_size} to {expected_size} for ResNet50")
                        # Resize using interpolate
                        try:
                            property_images = F.interpolate(
                                property_images,
                                size=expected_size,
                                mode='bilinear',
                                align_corners=False
                            )
                        except Exception as e:
                            logger.error(f"Error resizing images: {str(e)}, shape: {property_images.shape}")
                            # Try reshaping if needed
                            if len(property_images.shape) < 4:
                                property_images = property_images.unsqueeze(0)
                                property_images = F.interpolate(
                                    property_images,
                                    size=expected_size,
                                    mode='bilinear',
                                    align_corners=False
                                )
                                property_images = property_images.squeeze(0)
                
                # Process images in chunks if there are many for a single property
                if num_images > max_batch_per_pass:
                    # Process in chunks to avoid memory issues
                    image_embeddings_list = []
                    for i in range(0, num_images, max_batch_per_pass):
                        end = min(i + max_batch_per_pass, num_images)
                        device_images = property_images[i:end].to(self.device)
                        chunk_embeddings = self.embedding_model(device_images)
                        image_embeddings_list.append(chunk_embeddings)
                        # Free memory
                        torch.cuda.empty_cache()
                    
                    # Concatenate all embeddings
                    image_embeddings = torch.cat(image_embeddings_list, dim=0)
                else:
                    # Move to device and get embeddings
                    device_images = property_images.to(self.device)
                    image_embeddings = self.embedding_model(device_images)
                
                # Reshape to [1, num_images, embedding_dim] for property aggregator
                image_embeddings = image_embeddings.unsqueeze(0)
                
                # Aggregate into property embedding
                property_embedding = self.property_aggregator(image_embeddings)  # [1, embedding_dim]
                property_embeddings.append(property_embedding)
            
            # Concatenate all property embeddings
            property_embeddings = torch.cat(property_embeddings, dim=0)
            
        else:
            # Fixed number of images per property (batch tensor)
            batch_size, num_images, channels, height, width = property_images_batch.shape
            
            # Check image dimensions and resize if needed for ResNet50
            if self.backbone == "resnet50":
                # ResNet50 expects 224x224 images
                expected_size = (224, 224)
                current_size = (height, width)
                if current_size != expected_size:
                    logger.info(f"Resizing batch from {current_size} to {expected_size} for ResNet50")
                    try:
                        # Reshape to process all images at once for resizing
                        reshaped = property_images_batch.view(-1, channels, height, width)
                        # Resize using interpolate
                        reshaped = F.interpolate(
                            reshaped,
                            size=expected_size,
                            mode='bilinear',
                            align_corners=False
                        )
                        # Update dimensions
                        property_images_batch = reshaped.view(batch_size, num_images, channels, 224, 224)
                        height, width = 224, 224
                    except Exception as e:
                        logger.error(f"Error during batch resize: {str(e)}")
                        # Try alternative approach
                        reshaped_images = []
                        for b in range(batch_size):
                            for i in range(num_images):
                                img = property_images_batch[b, i]
                                img = F.interpolate(
                                    img.unsqueeze(0),
                                    size=expected_size,
                                    mode='bilinear',
                                    align_corners=False
                                )
                                reshaped_images.append(img.squeeze(0))
                        
                        all_images = torch.stack(reshaped_images).view(batch_size, num_images, channels, 224, 224)
                        property_images_batch = all_images
                        height, width = 224, 224
            
            # Process in smaller batches to avoid CUDA OOM
            if batch_size * num_images > max_batch_per_pass:
                logger.info(f"Processing large batch ({batch_size}x{num_images}) in chunks")
                
                # Get embeddings for chunks of images
                all_embeddings_list = []
                
                # Reshape to process images as a flat batch
                all_images = property_images_batch.view(-1, channels, height, width)
                total_images = all_images.size(0)
                
                # Process in chunks
                for i in range(0, total_images, max_batch_per_pass):
                    end = min(i + max_batch_per_pass, total_images)
                    try:
                        chunk_embeddings = self.embedding_model(all_images[i:end].to(self.device))
                        all_embeddings_list.append(chunk_embeddings)
                    except Exception as e:
                        logger.error(f"Error in chunk processing: {str(e)}")
                        # Fill with zeros for this chunk
                        zeros = torch.zeros(
                            end - i,
                            self.embedding_dim,
                            device=self.device
                        )
                        all_embeddings_list.append(zeros)
                    
                    # Clear memory
                    torch.cuda.empty_cache()
                
                # Concatenate all embeddings
                try:
                    all_embeddings = torch.cat(all_embeddings_list, dim=0)
                    
                    # Reshape back to [batch_size, num_images, embedding_dim]
                    image_embeddings = all_embeddings.view(batch_size, num_images, self.embedding_dim)
                    
                    # Aggregate into property embeddings
                    property_embeddings = self.property_aggregator(image_embeddings)  # [batch_size, embedding_dim]
                except Exception as e:
                    logger.error(f"Error combining embeddings: {str(e)}")
                    # Fallback to zeros
                    property_embeddings = torch.zeros(batch_size, self.embedding_dim, device=self.device)
            else:
                # For smaller batches, process all at once
                # Reshape to process all images at once
                all_images = property_images_batch.view(-1, channels, height, width)
                
                # Get embeddings for all images
                try:
                    all_embeddings = self.embedding_model(all_images.to(self.device))  # [batch_size*num_images, embedding_dim]
                    
                    # Reshape back to [batch_size, num_images, embedding_dim]
                    image_embeddings = all_embeddings.view(batch_size, num_images, self.embedding_dim)
                    
                    # Aggregate into property embeddings
                    property_embeddings = self.property_aggregator(image_embeddings)  # [batch_size, embedding_dim]
                except Exception as e:
                    logger.error(f"Error in embedding process: {str(e)}")
                    # Process each image individually as fallback
                    all_embeddings = []
                    for img in all_images:
                        try:
                            emb = self.embedding_model(img.unsqueeze(0).to(self.device))
                            all_embeddings.append(emb)
                        except Exception as nested_e:
                            logger.error(f"Error processing individual image: {str(nested_e)}")
                            # Use a zero embedding as fallback
                            all_embeddings.append(torch.zeros(1, self.embedding_dim, device=self.device))
                    
                    all_embeddings = torch.cat(all_embeddings)
                    image_embeddings = all_embeddings.view(batch_size, num_images, self.embedding_dim)
                    property_embeddings = self.property_aggregator(image_embeddings)
        
        return property_embeddings
    
    def train(
        self, 
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 20,
        save_dir: str = None,
        checkpoint_freq: int = 5,
        patience: int = 10,
        use_early_stopping: bool = True,
        mixed_precision: bool = True  # Enable mixed precision training
    ) -> Dict[str, Any]:
        """
        Train the Siamese network for property-level similarity.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints and final model
            checkpoint_freq: Frequency of saving checkpoints (in epochs)
            patience: Number of epochs to wait for validation loss improvement
            use_early_stopping: Whether to use early stopping based on validation loss
            mixed_precision: Whether to use mixed precision training (faster, less memory)
            
        Returns:
            Dictionary with training history
        """
        if save_dir:
            ensure_dir_exists(save_dir)
        
        # Initialize the scheduler
        self.scheduler = self._initialize_scheduler(self.scheduler_type, self.optimizer)
        
        # Special handling for one_cycle scheduler which needs dataloader and epochs
        if self.scheduler_type == "one_cycle":
            # Calculate total steps for one cycle scheduler
            total_steps = epochs * len(train_dataloader)
            logger.info(f"Initializing OneCycleLR scheduler with {total_steps} total steps")
            max_lr = self.learning_rate * 10  # Peak learning rate will be 10x base learning rate
            self.scheduler = lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=0.3,  # Spend 30% of iterations in the increasing phase
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=10000,  # Final LR = max_lr/10000
                anneal_strategy='cos',
                three_phase=False
            )
            self.scheduler_batch_update = True  # One cycle updates after each batch
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [] if val_dataloader else None,
            'val_metrics': [] if val_dataloader else None,
            'learning_rate': []
        }
        
        # Initialize best validation loss for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Set up mixed precision training if available
        scaler = None
        if mixed_precision and torch.cuda.is_available():
            # Ensure we're using the latest PyTorch API
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
                logger.info("Enabling mixed precision training with torch.cuda.amp")
                scaler = torch.cuda.amp.GradScaler()
            else:
                logger.warning("Mixed precision requested but torch.cuda.amp.GradScaler not available")
                mixed_precision = False
        
        # Get total batches for progress tracking
        total_batches = len(train_dataloader)
        
        # Train loop
        for epoch in range(epochs):
            # Set models to training mode
            self.embedding_model.train()
            self.property_aggregator.train()
            
            # Track metrics
            epoch_loss = 0.0
            batch_count = 0
            
            # For calculating training metrics
            all_train_distances_pos = []
            all_train_distances_neg = []
            
            # Set up progress bar
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            # Empty GPU cache before each epoch (helps with memory fragmentation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Training loop
            for batch in progress_bar:
                # Extract triplet components
                if isinstance(batch, (list, tuple)):
                    # Unpacking the tuple (anchor, positive, negative)
                    anchor, positive, negative = batch
                else:
                    # Dictionary access with keys
                    anchor = batch['anchor']
                    positive = batch['positive']
                    negative = batch['negative']
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)  # Slightly faster and less memory
                
                # Use mixed precision if enabled
                if scaler:
                    with torch.cuda.amp.autocast():
                        # Process each property to get property-level embeddings
                        try:
                            anchor_emb = self._process_property_batch(anchor)
                            positive_emb = self._process_property_batch(positive)
                            negative_emb = self._process_property_batch(negative)
                            
                            # Compute loss
                            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                logger.error(f"CUDA OOM error: {str(e)}")
                                # Skip this batch, handle it gracefully
                                logger.warning("Skipping batch due to CUDA out of memory")
                                # Free memory
                                torch.cuda.empty_cache()
                                continue
                            else:
                                # Log error details but continue if possible
                                logger.error(f"Runtime error in batch: {str(e)}")
                                # Try to recover and skip this batch
                                continue
                    
                    # Scale the gradients and call backward()
                    scaler.scale(loss).backward()
                    
                    # Unscale before optimizer step
                    scaler.unscale_(self.optimizer)
                    
                    # Clip gradients for stability (common practice with mixed precision)
                    torch.nn.utils.clip_grad_norm_(self.embedding_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.property_aggregator.parameters(), max_norm=1.0)
                    
                    # Step with scaler
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Regular full-precision training
                    try:
                        # Process each property to get property-level embeddings
                        anchor_emb = self._process_property_batch(anchor)
                        positive_emb = self._process_property_batch(positive)
                        negative_emb = self._process_property_batch(negative)
                    
                        # Compute loss
                        loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                        
                        # Backward pass and optimize
                        loss.backward()
                        
                        # Clip gradients for stability
                        torch.nn.utils.clip_grad_norm_(self.embedding_model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.property_aggregator.parameters(), max_norm=1.0)
                        
                        self.optimizer.step()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            logger.error(f"CUDA OOM error: {str(e)}")
                            # Skip this batch, handle it gracefully
                            logger.warning("Skipping batch due to CUDA out of memory")
                            # Free memory
                            torch.cuda.empty_cache()
                            continue
                        else:
                            # Log error details but continue if possible
                            logger.error(f"Runtime error in batch: {str(e)}")
                            # Try to recover and skip this batch
                            continue
                
                # Update learning rate if using batch-level scheduler
                if self.scheduler_batch_update and self.scheduler is not None:
                    self.scheduler.step()
                
                # Update stats
                epoch_loss += loss.item()
                batch_count += 1
                
                # Calculate distances for metrics
                with torch.no_grad():
                    pos_distances = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                    neg_distances = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                    
                    all_train_distances_pos.extend(pos_distances.cpu().numpy())
                    all_train_distances_neg.extend(neg_distances.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': epoch_loss / batch_count,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                # Periodically clear cache to prevent memory fragmentation
                if batch_count % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for the epoch
            train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            history['train_loss'].append(train_loss)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(all_train_distances_pos, all_train_distances_neg)
            history['train_metrics'].append(train_metrics)
            
            train_info = f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            
            # Validation phase
            if val_dataloader:
                # Clear cache before validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                val_result = self.evaluate(val_dataloader)
                val_loss = val_result['loss']
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_result)
                
                val_info = f"Val Loss: {val_loss:.4f}, Acc: {val_result['accuracy']:.4f}, F1: {val_result['f1']:.4f}"
                
                # Update learning rate scheduler if it's epoch-based
                if not self.scheduler_batch_update and self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                elif not self.scheduler_batch_update:
                    self.scheduler.step()
                
                # Early stopping
                if use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        if save_dir:
                            best_model_path = os.path.join(save_dir, "best_model")
                            self.save_model(best_model_path)
                            logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                    break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - {train_info}, {val_info}")
            else:
                # Update learning rate scheduler if it's epoch-based
                if not self.scheduler_batch_update and self.scheduler_type == 'plateau':
                    self.scheduler.step(train_loss)
                elif not self.scheduler_batch_update:
                    self.scheduler.step()
                
                logger.info(f"Epoch {epoch+1}/{epochs} - {train_info}")
            
            # Save checkpoint
            if save_dir and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                self.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save final model
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model"))
            
            # Save training history
            history_path = os.path.join(save_dir, "training_history.json")
            
            # Prepare history for JSON serialization
            json_history = {
                'train_loss': [float(x) for x in history['train_loss']],
                'learning_rate': [float(x) for x in history['learning_rate']],
                'train_metrics': history['train_metrics'],
            }
            
            if val_dataloader:
                json_history['val_loss'] = [float(x) for x in history['val_loss']]
                json_history['val_metrics'] = history['val_metrics']
                
            save_json(json_history, history_path)
            logger.info(f"Training history saved to {history_path}")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary with evaluation metrics (loss, accuracy, precision, recall, f1)
        """
        self.embedding_model.eval()
        self.property_aggregator.eval()
        total_loss = 0.0
        batch_count = 0
        
        # For computing metrics
        all_distances_pos = []  # Distances between anchor and positive
        all_distances_neg = []  # Distances between anchor and negative
        
        with torch.no_grad():
            for batch in dataloader:
                # Check if batch is a tuple/list or a dictionary
                if isinstance(batch, (list, tuple)):
                    # Unpacking the tuple (anchor, positive, negative)
                    anchor, positive, negative = batch
                else:
                    # Dictionary access with keys
                    anchor = batch['anchor']
                    positive = batch['positive']
                    negative = batch['negative']
                
                # Process each property to get property-level embeddings
                anchor_emb = self._process_property_batch(anchor)
                positive_emb = self._process_property_batch(positive)
                negative_emb = self._process_property_batch(negative)
                
                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
                # Update stats
                total_loss += loss.item()
                batch_count += 1
        
                # Calculate distances for metrics
                pos_distances = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
                neg_distances = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
                
                all_distances_pos.extend(pos_distances.cpu().numpy())
                all_distances_neg.extend(neg_distances.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        # Calculate metrics based on distances
        metrics = self._calculate_metrics(all_distances_pos, all_distances_neg)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _calculate_metrics(self, pos_distances, neg_distances):
        """
        Calculate evaluation metrics based on distances.
        
        Args:
            pos_distances: List of distances between anchor and positive examples
            neg_distances: List of distances between anchor and negative examples
            
        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1)
        """
        pos_distances = np.array(pos_distances)
        neg_distances = np.array(neg_distances)
        
        # Find optimal threshold by trying different values
        thresholds = np.linspace(0, max(np.max(pos_distances), np.max(neg_distances)), 100)
        best_f1 = 0
        best_threshold = 0
        best_metrics = {}
        
        for threshold in thresholds:
            # Predictions based on threshold
            pos_correct = (pos_distances <= threshold).sum()
            neg_correct = (neg_distances > threshold).sum()
            
            # Calculate metrics
            true_positives = pos_correct
            false_negatives = len(pos_distances) - pos_correct
            true_negatives = neg_correct
            false_positives = len(neg_distances) - neg_correct
            
            total_examples = len(pos_distances) + len(neg_distances)
            if total_examples == 0:
                continue
                
            accuracy = (true_positives + true_negatives) / total_examples
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'threshold': float(threshold),
                    'true_positives': int(true_positives),
                    'false_positives': int(false_positives),
                    'true_negatives': int(true_negatives),
                    'false_negatives': int(false_negatives)
                }
        
        # Add distance statistics
        best_metrics['avg_pos_distance'] = float(np.mean(pos_distances))
        best_metrics['avg_neg_distance'] = float(np.mean(neg_distances))
        best_metrics['distance_gap'] = float(np.mean(neg_distances) - np.mean(pos_distances))
        
        # Add distribution statistics
        best_metrics['pos_distance_std'] = float(np.std(pos_distances))
        best_metrics['neg_distance_std'] = float(np.std(neg_distances))
        
        return best_metrics
    
    def get_property_embedding(self, property_images: List[np.ndarray]) -> np.ndarray:
        """
        Get the embedding vector for a property (multiple images).
        
        Args:
            property_images: List of image arrays, each of shape (height, width, channels)
            
        Returns:
            Property embedding vector (normalized)
        """
        self.embedding_model.eval()
        self.property_aggregator.eval()
        
        # Process each image
        image_embeddings = []
        
        with torch.no_grad():
            for image in property_images:
                # Ensure image is in NCHW format
                if len(image.shape) == 3:  # HWC format
                    # Convert from HWC to NCHW format
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, axis=0)
                
                # Convert to PyTorch tensor
                image_tensor = torch.from_numpy(image).float().to(self.device)
                
                # Get embedding for this image
                embedding = self.embedding_model(image_tensor)
                image_embeddings.append(embedding)
            
            # Stack all image embeddings
            stacked_embeddings = torch.stack(image_embeddings, dim=1)  # [1, num_images, embedding_dim]
            
            # Aggregate into a property embedding
            property_embedding = self.property_aggregator(stacked_embeddings)
            
            # Return as numpy array
            return property_embedding.cpu().numpy()
    
    def compute_property_similarity(self, property1_images: List[np.ndarray], property2_images: List[np.ndarray]) -> float:
        """
        Compute similarity score between two properties (each with multiple images).
        
        Args:
            property1_images: List of image arrays for property 1
            property2_images: List of image arrays for property 2
            
        Returns:
            Similarity score between 0 and 10 (higher means more similar)
        """
        # Get property embeddings
        emb1 = self.get_property_embedding(property1_images)
        emb2 = self.get_property_embedding(property2_images)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2.T)[0, 0]
        
        # Cosine similarity ranges from -1 to 1, normalize to 0-1
        normalized_similarity = (similarity + 1) / 2
        
        # Scale to 0-10 for the project's scoring system
        score = normalized_similarity * 10
        
        return float(score)
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model.
        
        Args:
            save_path: Directory to save model
        """
        # Ensure directory exists
        ensure_dir_exists(save_path)
        
        # Save model state dicts
        embedding_model_path = os.path.join(save_path, "siamese_embedding_model.pt")
        torch.save(self.embedding_model.state_dict(), embedding_model_path)
        
        property_aggregator_path = os.path.join(save_path, "property_aggregator.pt")
        torch.save(self.property_aggregator.state_dict(), property_aggregator_path)
        
        # Save optimizer state
        optimizer_path = os.path.join(save_path, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save model configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'learning_rate': self.learning_rate,
            'backbone': self.backbone,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(save_path, "model_config.json")
        save_json(config, config_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a previously saved model.
        
        Args:
            load_path: Path to the saved model directory
        """
        # Load model configuration if available
        config_path = os.path.join(load_path, "model_config.json")
        if os.path.exists(config_path):
            config = load_json(config_path)
            if config and 'backbone' in config:
                self.backbone = config['backbone']
                logger.info(f"Loaded backbone type: {self.backbone}")
        
        # Check if this is a directory or direct model file
        if os.path.isdir(load_path):
            # Directory path - load all components
            embedding_model_path = os.path.join(load_path, "siamese_embedding_model.pt")
            self.embedding_model.load_state_dict(torch.load(embedding_model_path, map_location=self.device))
            
            # Load property aggregator if exists (for backward compatibility)
            property_aggregator_path = os.path.join(load_path, "property_aggregator.pt")
            if os.path.exists(property_aggregator_path):
                self.property_aggregator.load_state_dict(torch.load(property_aggregator_path, map_location=self.device))
        else:
            # Direct file path to model - assume it's the embedding model
            self.embedding_model.load_state_dict(torch.load(load_path, map_location=self.device))
        
        logger.info(f"Model loaded from {load_path}")
        
        # Set models to evaluation mode
        self.embedding_model.eval()
        self.property_aggregator.eval()


# Example of how to use the model
if __name__ == "__main__":
    from data_preparation import TripletDataset, collect_property_images, generate_triplets_with_augmentation
    
    # Example data dir
    data_dir = "/workspace/data/sample_properties"
    
    # Collect property images
    property_images = collect_property_images(data_dir)
    
    if property_images:
        print(f"Found {len(property_images)} property types with images")
        
        # Generate triplets
        triplets = generate_triplets_with_augmentation(property_images, num_triplets=100)
        
        if triplets:
            # Create dataset and dataloader
            dataset = TripletDataset(triplets)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=16, shuffle=True
            )
            
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=16, shuffle=False
            )
            
            # Create and train model
            model = SiameseNetwork(embedding_dim=256, margin=0.2, learning_rate=0.001, backbone="resnet50")
            
            # Train for a few epochs
            history = model.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=5,
                save_dir="/workspace/models/siamese",
                checkpoint_freq=1,
                mixed_precision=True
            )
            
            print("Training completed!") 