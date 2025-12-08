import time
import math
import torch
import torchvision.transforms as transforms
from PIL import Image
import orion
import sys
sys.path.insert(0, "../")
import orion.models as models
from orion.core.utils import (
    get_cifar_datasets,
    mae, 
    train_on_cifar
)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
net = models.ResNet20()

# Load image from img_test folder
img_path = "../img_test/luis.png"  # Remplacer par le nom de l'image
image = Image.open(img_path).convert('RGB')

# Preprocess image to match CIFAR-10 format
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
inp = transform(image).unsqueeze(0)  # Add batch dimension

# Run cleartext inference
net.eval()
out_clear = net(inp)

# Prepare for FHE inference
orion.fit(net, inp)
input_level = orion.compile(net)

# Encode and encrypt
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
net.he()

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# Decrypt and decode
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()

# Compare results
print()
print(out_clear)
print(out_fhe)

dist = mae(out_clear, out_fhe)
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")