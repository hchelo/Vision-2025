import torch
from torchvision import models, transforms
from PIL import Image
import os
import time

# Definir el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definir el modelo VGG16 y reemplazar la última capa
model = models.vgg16(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 5)

# Cargar los pesos guardados
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para medir el tiempo
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Tiempo transcurrido: %f segundos.\n" % tempTimeInterval)

def tic():
    toc(False)

# Función para predecir una sola imagen
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Añadir una dimensión para el batch
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)

    return predicted.item()

# Prueba de predicciones
test_images = [
    ('BD_New_DKC/testing_dkc/0/izq0117.png', 0),
    ('BD_New_DKC/testing_dkc/0/izq0147.png', 0),
    ('BD_New_DKC/testing_dkc/1/semizq 0117.png', 1),
    ('BD_New_DKC/testing_dkc/1/semizq 0136.png', 1),
    ('BD_New_DKC/testing_dkc/2/adelante0081.png', 2),
    ('BD_New_DKC/testing_dkc/2/adelante0099.png', 2),
    ('BD_New_DKC/testing_dkc/3/semder0072.png', 3),
    ('BD_New_DKC/testing_dkc/3/semder0081.png', 3),
    ('BD_New_DKC/testing_dkc/4/der0187.png', 4),
    ('BD_New_DKC/testing_dkc/4/der0194.png', 4)
]

for image_path, expected in test_images:
    print(f"Probando imagen: {image_path} (Esperado: {expected})")
    tic()
    result = predict_image(image_path, model, device)
    print(f"Resultado: {result}")
    toc()
