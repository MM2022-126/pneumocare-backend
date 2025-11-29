import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# -------------------------------------------------------------
# 1. GOOGLE DRIVE DOWNLOAD SUPPORT
# -------------------------------------------------------------

DRIVE_FILE_ID = "1eFl_sVCjV9H9fEXtooWiA8Vu6lMZkexO"
MODEL_PATH = "resnet50_final.pth"


def download_file_from_google_drive(file_id, destination):
    """Download a large file from Google Drive"""
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Google adds a "confirm download" token for large files
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Save model to file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("✔ Model downloaded successfully!")


# -------------------------------------------------------------
# 2. DOWNLOAD THE MODEL IF NOT PRESENT
# -------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("⚠ Model not found locally. Downloading from Google Drive...")
    download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
else:
    print("✔ Model found locally.")


# -------------------------------------------------------------
# 3. LOAD THE MODEL
# -------------------------------------------------------------
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaModel().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -------------------------------------------------------------
# 4. PREDICTION ENDPOINT
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"]).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    classes = ["Normal", "Pneumonia"]
    result = classes[predicted.item()]

    return jsonify({"prediction": result})


# -------------------------------------------------------------
# 5. RUN SERVER
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
