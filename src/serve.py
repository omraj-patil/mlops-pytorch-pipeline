import io
import os

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

image_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def load_model():
    global model

    checkpoint_path = os.environ.get(
        "MODEL_PATH",
        "checkpoints/classifier_v1.pt",
    )

    model = get_model("resnet18", num_classes=10)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy"
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()

    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = image_transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    return {
        "probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item(), 6)
            for i in range(len(CLASS_NAMES))
        }
    }