import time
import math
import types
import torch
import orion
import orion.models as models
import psutil
import os
import resource
from PIL import Image
from torchvision import transforms
from orion.core.utils import mae

# Set seed for reproducibility
torch.manual_seed(42)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize the Orion scheme
scheme = orion.init_scheme("../configs/resnet.yml")

# Load the model structure
net = models.ResNet20()

# Monkey patch the forward method to print the layer before Linear
def new_forward(self, x):
    out = self.act(self.bn1(self.conv1(x)))
    out = self.pool(out)
    for layer in self.layers:
        out = layer(out)
    out = self.avgpool(out)
    out = self.flatten(out)
    
    # Ensure full printing
    torch.set_printoptions(threshold=float('inf'), linewidth=2000)

    output_str = ""
    mode = "Cleartext"

    if hasattr(out, 'decrypt'):
        mode = "FHE (Decrypted)"
        try:
            dec = out.decrypt()
            decoded = dec.decode()
            output_str = str(decoded)
        except Exception as e:
            output_str = f"Error decrypting: {e}"
    else:
        mode = "Cleartext"
        output_str = str(out)
    
    print(f"\n[DEBUG] Layer before Linear ({mode}):")
    print(output_str)

    # Save to file if image name is available
    if hasattr(self, 'current_img_name'):
        filename = f"embedding_{self.current_img_name}.txt"
        with open(filename, "a") as f:
            f.write(f"--- {mode} ---\n")
            f.write(output_str + "\n\n")
        
    return self.linear(out)

net.forward = types.MethodType(new_forward, net)

# Load the pre-trained weights
model_path = "resnet_cifar.pth"
# Note: The user specified that this file might not be present locally but will be on the target machine.
# We add a check but proceed if possible or fail gracefully.

if os.path.exists(model_path):
    print(f"Chargement du modèle depuis {model_path}...")
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
else:
    print(f"Attention: Le fichier {model_path} n'a pas été trouvé localement.")
    print("Le script continuera, mais l'inférence échouera si le fichier est manquant lors de l'exécution réelle.")
    # We can't really continue without weights if we want meaningful results, but we'll let the error happen at load time if it crashes.
    # Actually, let's try to load it and catch the error to give a nice message.
    try:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError:
        print("Erreur critique : Impossible de charger les poids du modèle.")
        exit(1)

net.eval()

# Define transforms (Resize to 32x32 and normalize like CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2470, 0.2435, 0.2616)
    ),
])

# Load images
img_dir = "../img_test"
if not os.path.exists(img_dir):
    print(f"Error: Directory '{img_dir}' not found.")
    exit(1)

image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

if not image_files:
    print(f"No images found in {img_dir}")
    exit(1)

print(f"Found {len(image_files)} images: {image_files}\n")

# Prepare samples
samples = []
for img_file in image_files:
    img_path = os.path.join(img_dir, img_file)
    try:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension
        samples.append((img_file, input_tensor))
    except Exception as e:
        print(f"Error loading image {img_file}: {e}")

if not samples:
    print("No valid samples loaded.")
    exit(1)

# Cleartext inference
print("Inférence en clair...\n")
clear_outputs = []
for img_name, inp in samples:
    net.current_img_name = img_name
    # Remove existing file to start fresh
    filename = f"embedding_{img_name}.txt"
    if os.path.exists(filename):
        os.remove(filename)

    start_time = time.time()
    with torch.no_grad():
        out = net(inp)
    clear_outputs.append(out)
    
    class_idx = out.argmax(1).item()
    class_name = classes[class_idx]
    print(f"Image: {img_name}")
    print(f"Classe prédite (Clair): {class_name}")
    print(f"Temps: {time.time() - start_time:.4f}s\n")


# Prepare for FHE inference
print("Préparation de l'inférence en FHE...\n")
# Use the first sample for calibration
orion.fit(net, samples[0][1])
input_level = orion.compile(net)

# Switch to FHE mode
net.he()

print(f"Lancement de {len(samples)} inférences chiffrées...\n")

for i, (img_name, inp) in enumerate(samples):
    print(f"--- Inférence {i+1}/{len(samples)}: {img_name} ---")
    
    net.current_img_name = img_name
    out_clear = clear_outputs[i]
    
    # Encrypt
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    # Run FHE inference
    print("Starting FHE inference", flush=True)
    start = time.time()
    
    out_ctxt = net(vec_ctxt)
    
    end = time.time()
    
    # Decrypt and decode
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode()
    print(f"FHE inference completed in {end - start:.2f} seconds.")

    # Compare
    dist = mae(out_clear, out_fhe)
    print(f"MAE: {dist:.4f}")
    print(f"Precision: {-math.log2(dist):.4f}")
    
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Max Memory (Process): {max_rss_mb:.2f} MB")
    
    # Classes
    class_clear = classes[out_clear.argmax(1).item()]
    class_fhe = classes[out_fhe.argmax(1).item()]
    
    print(f"Classe inférée (Clair): {class_clear}")
    print(f"Classe inférée (FHE): {class_fhe}")
    print("-" * 30 + "\n")
