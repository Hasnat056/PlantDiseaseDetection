from celery import shared_task
from django.shortcuts import get_object_or_404

from .models import Uploads, UserImages
from django.contrib.auth.models import User
from django.conf import settings
import os
from PIL import Image
import torch

# Import your model and preprocessing
from .pytorch import ResNet9, transform

# Your class names and model setup (same as your view)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)_Powdery_mildew",
    "Cherry_(including_sour)_healthy",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)Common_rust",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Corn_(maize)_healthy",
    "Grape___Black_rot",
    "Grape__Esca(Black_Measles)",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,bell__Bacterial_spot",
    "Pepper,bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Load model once (global for the worker process)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
MODEL_PATH = os.path.join(settings.BASE_DIR, 'PredictionModel', 'plant_disease_model.pth')

model = ResNet9(in_channels=3, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


@shared_task
def process_plant_disease(image_path, user_id=None):
    """
    Process plant disease image and save to database if successful

    Args:
        image_path: Path to uploaded image file
        user_id: ID of authenticated user (None for anonymous)

    Returns:
        dict with processing results
    """
    try:
        # Get full path to image
        full_path = os.path.join(settings.MEDIA_ROOT, image_path)

        # Load and preprocess image
        image = Image.open(full_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_index = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_index].item()

        pred_class = CLASS_NAMES[pred_index]

        # Save to database only if processing succeeded
        upload_record = Uploads.objects.create(
            image=image_path,
            result=pred_class,
            confidence_score=round(confidence * 100, 2)
        )

        # Link to user if authenticated
        if user_id:
            user = get_object_or_404(User, id=user_id)
            UserImages.objects.create(
                image=upload_record,
                user=user
            )

        # Clean up temp file if using temp storage
        if 'temp/' in image_path:
            try:
                os.remove(full_path)
            except:
                pass

        return {
            'success': True,
            'upload_id': upload_record.id,
            'prediction': pred_class,
            'confidence': round(confidence * 100, 2)
        }

    except Exception as e:
        # Clean up file on failure
        try:
            full_path = os.path.join(settings.MEDIA_ROOT, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
        except:
            pass

        return {
            'success': False,
            'error': str(e)
        }