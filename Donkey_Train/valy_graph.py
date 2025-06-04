import os
import time
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image, ImageDraw

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

# Ruta a la carpeta de imágenes
folder_path = 'BD_New_DKC/testing_dkc'

# Obtener una lista de todas las subcarpetas (categorías) en la carpeta
subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

# Dimensiones del rectángulo
rect_width = 8
rect_height = 50

# Ángulo de rotación
rotation_angle = 45

# Iterar sobre todas las subcarpetas
for subfolder in subfolders:
    # Obtener la lista de imágenes en la subcarpeta
    image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Iterar sobre las imágenes en pasos de 10
    for i in range(0, len(image_files), 10):
        # Obtener el nombre del archivo de imagen
        image_file = image_files[i]

        # Cargar la imagen
        img = Image.open(image_file)
        
        # Crear una nueva imagen para el rectángulo
        rect_img = Image.new('RGBA', (rect_width, rect_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect_img)

        # Dibujar el rectángulo azul en la nueva imagen
        draw.rectangle([0, 0, rect_width, rect_height], fill="cyan")

        # Rotar la imagen del rectángulo
        rect_img = rect_img.rotate(rotation_angle, expand=True)

        # Obtener las dimensiones de la imagen original
        img_width, img_height = img.size

        # Calcular la posición para pegar el rectángulo en la parte inferior central
        paste_x = (img_width - rect_img.width) // 2
        paste_y = img_height - rect_img.height

        # Pegar el rectángulo rotado en la imagen original
        img.paste(rect_img, (paste_x, paste_y), rect_img)

        # Mostrar la imagen
        plt.imshow(img)
        plt.title(f"Imagen: {os.path.basename(image_file)}")
        plt.axis('off')  # Ocultar ejes
        plt.draw()
        # Esperar 3 segundos antes de mostrar la siguiente imagen
        plt.pause(0.5)
        time.sleep(0.5)
        # Cerrar la imagen actual
        plt.close()
