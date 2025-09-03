import torch
from torchvision import transforms
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import os
from PIL import Image


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

# Load the model once (global, so it doesn’t reload every request)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38  # total number of classes
MODEL_PATH = os.path.join(settings.BASE_DIR, 'PredictionModel', 'plant_disease_model.pth')  # adjust path

from .pytorch import ResNet9
model = ResNet9(in_channels=3, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # set to evaluation mode


# Transformations for incoming image
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # as per your training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # typical ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

def predict_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save uploaded image
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(f'temp/{uploaded_file.name}', uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Open and transform image
        image = Image.open(full_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_index = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_index].item()

        pred_class = CLASS_NAMES[pred_index]

        # Send to template
        context = {
            'img_url': default_storage.url(file_path),
            'pred_class': pred_index,
            'pred_name': pred_class,
            'confidence': f"{confidence*100:.2f}%"  # convert to %
        }
        return render(request, 'result.html', context)

    return render(request, 'upload.html')
