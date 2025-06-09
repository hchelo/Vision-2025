import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from PIL import Image, ImageDraw

# Definir el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelo VGG16 con pesos preentrenados
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 5)

# Cargar pesos entrenados por el usuario
model.load_state_dict(torch.load('best_modelito.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Cronómetro para medir tiempo
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti

TicToc = TicTocGenerator()

def toc(print_time=True):
    delta = next(TicToc)
    if print_time:
        print("Tiempo transcurrido: %.4f segundos\n" % delta)

def tic():
    toc(False)

# Función para predecir clase de una imagen
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return predicted.item()

# Carpeta de imágenes
folder_path = 'BD_New_DKC/testing_dkc'
subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
              if os.path.isdir(os.path.join(folder_path, d))]

# Dimensiones del rectángulo
rect_width = 8
rect_height = 50

# Iterar por carpeta
for subfolder in subfolders:
    image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()  # Opcional: ordena las imágenes

    for i in range(0, len(image_files), 1):
        image_file = image_files[i]

        # Predicción
        result = predict_image(image_file, model, device)
        print(f"Imagen: {os.path.basename(image_file)}, Predicción: {result}")

        # Definir ángulo según clase
        angle_map = {0: 55, 1: 15, 2: 0, 3: -15, 4: -55}
        rotation_angle = angle_map.get(result, 0)

        # Abrir imagen original
        img = Image.open(image_file)

        # Crear rectángulo
        rect_img = Image.new('RGBA', (rect_width, rect_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rect_img)
        draw.rectangle([0, 0, rect_width, rect_height], fill="cyan")
        rect_img = rect_img.rotate(rotation_angle, expand=True)

        # Pegar rectángulo rotado
        img_width, img_height = img.size
        paste_x = (img_width - rect_img.width) // 2
        paste_y = img_height - rect_img.height
        img.paste(rect_img, (paste_x, paste_y), rect_img)

        # Mostrar imagen
        plt.imshow(img)
        plt.title(f"Imagen: {os.path.basename(image_file)}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
