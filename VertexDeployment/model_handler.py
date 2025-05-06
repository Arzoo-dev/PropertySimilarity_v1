import json
import logging
import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class SiameseModelHandler(BaseHandler):
    """
    Custom handler for the Siamese neural network model deployment on Vertex AI.
    """
    
    def __init__(self):
        super(SiameseModelHandler, self).__init__()
        self.transform = None
        self.device = None
        self.model = None
        
    def initialize(self, context):
        """Initialize model and preprocessing transforms."""
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        model_dir = properties.get("model_dir")
        model_file = os.path.join(model_dir, "siamese_embedding_model.pt")
        self.model = torch.jit.load(model_file, map_location=self.device)
        self.model.eval()
        
        # Initialize preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("Model loaded successfully")
        self.initialized = True
        
    def preprocess(self, data):
        """
        Preprocess the input images.
        """
        images = []
        
        for row in data:
            # Vertex AI sends data as JSON
            image_data = row.get("data") or row.get("body")
            
            if isinstance(image_data, str):
                # If input is a JSON string
                image_data = json.loads(image_data)
            
            # Check if we're getting a batch of images or a single image
            if "instances" in image_data:
                # Handle batch format from Vertex AI
                instances = image_data["instances"]
                for instance in instances:
                    # Convert base64 to image
                    if "b64" in instance:
                        image_bytes = io.BytesIO(instance["b64"])
                    else:
                        # Handle URL or other formats as needed
                        continue
                        
                    # Process image
                    img = Image.open(image_bytes).convert('RGB')
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
            else:
                # Handle direct image input
                img_bytes = io.BytesIO(image_data)
                img = Image.open(img_bytes).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
        
        # Stack images into a batch
        if images:
            return torch.stack(images).to(self.device)
        else:
            raise ValueError("No valid images found in the request")

    def inference(self, data):
        """
        Generate embeddings for the input images.
        """
        with torch.no_grad():
            embeddings = self.model(data)
            # Normalize embeddings for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def postprocess(self, inference_output):
        """
        Convert embeddings to the expected response format.
        """
        # Convert embeddings to list format for JSON serialization
        embeddings = inference_output.cpu().numpy().tolist()
        
        # Format for Vertex AI response
        response = {
            "predictions": embeddings
        }
        
        return [response]
    
    def handle(self, data, context):
        """
        Handle the complete inference process.
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output) 