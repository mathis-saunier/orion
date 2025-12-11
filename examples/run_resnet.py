import time
import math
import torch
import orion
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
trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
net = models.ResNet20()

print("Début de l'entraînement du modèle...\n")
time_start = time.time()
# Train model (optional)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Utilisation de l'appareil :", device, "\n")
train_on_cifar(net, data_dir="../data", epochs=1, device=device)

print(f"Entraînement terminé en {time.time() - time_start:.2f} secondes.\n")

# Get a test batch to pass through our network
print("Inférence en clair...\n")
start_time = time.time()
inp, _ = next(iter(testloader))

# Run cleartext inference
net.eval()
out_clear = net(inp)
print(f"Inférence en clair terminée en {time.time() - start_time:.2f} secondes.\n")

# Prepare for FHE inference. 
# Some polynomial activation functions require knowing the range of possible 
# input values. We'll estimate these ranges using training set statistics, 
# adjusted to be wider by a tolerance factor (= margin).
print("Inférence en FHE...\n")
start_time = time.time()
orion.fit(net, inp)
input_level = orion.compile(net)

# Encode and encrypt the input vector 
vec_ptxt = orion.encode(inp, input_level)
vec_ctxt = orion.encrypt(vec_ptxt)
net.he()  # Switch to FHE mode

# Run FHE inference
print("\nStarting FHE inference", flush=True)
start = time.time()
out_ctxt = net(vec_ctxt)
end = time.time()

# Get the FHE results and decrypt + decode.
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode()
print(f"\nFHE inference completed in {time.time() - start_time:.2f} seconds.\n")

# Compare the cleartext and FHE results.
print()
print("Cleartext output:\n")
print(out_clear)
print("\nFHE output:\n")
print(out_fhe)

dist = mae(out_clear, out_fhe)
print(f"\nMAE: {dist:.4f}")
print(f"Precision: {-math.log2(dist):.4f}")
print(f"Runtime: {end-start:.4f} secs.\n")

# Save the model and scheme
torch.save(net.state_dict(), "resnet_cifar.pth")
scheme.save(f"resnet_scheme{time.time()}.pkl")