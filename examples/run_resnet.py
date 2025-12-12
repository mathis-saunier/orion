import time
import math
import torch
import orion
import orion.models as models
import psutil
import os
import resource
from orion.core.utils import (
    get_cifar_datasets,
    mae, 
    train_on_cifar,
    test_epoch
)

# Set seed for reproducibility
torch.manual_seed(42)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
net = models.ResNet20()

print("Début de l'entraînement du modèle...\n")
time_start = time.time()
# Train model (optional)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Utilisation de l'appareil :", device, "\n")
train_on_cifar(net, data_dir="../data", epochs=300, device=device)

print(f"Entraînement terminé en {time.time() - time_start:.2f} secondes.\n")

# Get a test batch to pass through our network
print("Inférence en clair...\n")
start_time = time.time()
criterion = torch.nn.CrossEntropyLoss()
net.to(device)
acc = test_epoch(net, testloader, criterion, torch.device(device))
net.to("cpu")
print(f"Précision sur le jeu de test: {acc:.2f}%")
print(f"Inférence en clair terminée en {time.time() - start_time:.2f} secondes.\n")

# Prepare 5 samples
print("Sélection de 5 échantillons pour l'inférence FHE...\n")
samples = []
test_iter = iter(testloader)
for _ in range(5):
    samples.append(next(test_iter))

# Cleartext inference on samples
net.eval()
clear_outputs = []
for inp, label in samples:
    with torch.no_grad():
        out = net(inp)
    clear_outputs.append(out)

# Prepare for FHE inference. 
# Some polynomial activation functions require knowing the range of possible 
# input values. We'll estimate these ranges using training set statistics, 
# adjusted to be wider by a tolerance factor (= margin).
print("Préparation de l'inférence en FHE...\n")
orion.fit(net, samples[0][0])
input_level = orion.compile(net)

# Encode and encrypt the input vector 
net.he()  # Switch to FHE mode

print("Lancement de 5 inférences chiffrées...\n")

process = psutil.Process(os.getpid())

for i, (inp, label) in enumerate(samples):
    print(f"--- Inférence {i+1}/5 ---")
    
    out_clear = clear_outputs[i]
    
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    # Run FHE inference
    print("Starting FHE inference", flush=True)
    start = time.time()
    
    out_ctxt = net(vec_ctxt)
    
    end = time.time()
    
    # Get the FHE results and decrypt + decode.
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode()
    print(f"FHE inference completed in {end - start:.2f} seconds.")

    # Compare the cleartext and FHE results.
    print("\nCleartext output:")
    print(out_clear)
    print("FHE output:")
    print(out_fhe)

    dist = mae(out_clear, out_fhe)
    print(f"\nMAE: {dist:.4f}")
    print(f"Precision: {-math.log2(dist):.4f}")
    print(f"Runtime: {end-start:.4f} secs.")
    
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Max Memory (Process): {max_rss_mb:.2f} MB")
    
    # Classes
    class_clear = classes[out_clear.argmax(1).item()]
    class_fhe = classes[out_fhe.argmax(1).item()]
    class_true = classes[label.item()]
    
    print(f"Classe réelle: {class_true}")
    print(f"Classe inférée (Clair): {class_clear}")
    print(f"Classe inférée (FHE): {class_fhe}")
    print("-" * 30 + "\n")

# Save the model and scheme
torch.save(net.state_dict(), "resnet_cifar.pth")
scheme.save(f"resnet_scheme{time.time()}.pkl")
print("Modèle et schéma sauvegardés avec succès.\n")