"""
Command-line argument parsing module for Siamese network training on RunPods.

GPU Memory Usage Guidelines:
----------------------------
ResNet50 backbone:
- RTX A5000 (24GB): Use batch_size=16 with mixed_precision=True
- V100 (16GB): Use batch_size=8 with mixed_precision=True
- RTX 3080/3090 (10-24GB): Use batch_size=12 with mixed_precision=True

EfficientNet backbone:
- RTX A5000 (24GB): Use batch_size=64 with mixed_precision=True
- V100 (16GB): Use batch_size=32 with mixed_precision=True
- RTX 3080/3090 (10-24GB): Use batch_size=32 with mixed_precision=True

For even larger batch sizes, the code will automatically process in chunks
to avoid CUDA out-of-memory errors.
"""

import argparse
import logging
import os
import torch 