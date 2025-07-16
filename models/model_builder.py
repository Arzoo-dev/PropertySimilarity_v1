import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
import logging

logger = logging.getLogger(__name__)

class DINOv2Retrieval(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        embedding_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        logger.info(f"Initializing DINOv2Retrieval with {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.freeze_backbone = freeze_backbone
        
        # Freeze backbone if specified
        if freeze_backbone:
            logger.info("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Always create a projection head to ensure trainable parameters
        backbone_dim = int(getattr(self.backbone, 'embed_dim', 768))
        logger.info(f"Creating projection head from {backbone_dim} to {embedding_dim} dimensions")
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Linear(backbone_dim // 2, embedding_dim)
        )
        
        # Ensure projection head parameters are trainable
        for param in self.projection.parameters():
            param.requires_grad = True
        logger.info("Projection head parameters are set to trainable")

    def forward(self, x):
        # Forward pass through backbone
        if self.freeze_backbone:
            with torch.no_grad():  # Only use no_grad when backbone is frozen
                if hasattr(self.backbone, 'forward_features') and callable(self.backbone.forward_features):
                    backbone_features = self.backbone.forward_features(x)
                else:
                    backbone_features = self.backbone(x)
                cls_token = backbone_features[:, 0]  # Get CLS token
        else:
            if hasattr(self.backbone, 'forward_features') and callable(self.backbone.forward_features):
                backbone_features = self.backbone.forward_features(x)
            else:
                backbone_features = self.backbone(x)
            cls_token = backbone_features[:, 0]  # Get CLS token
        
        # Apply dropout
        dropped = self.dropout(cls_token)
        
        # Project if needed
        embeddings = self.projection(dropped)
        
        # L2 normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return normalized_embeddings


