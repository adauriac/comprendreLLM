#!/usr/bin/env python
import torch,sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import scan_cache_dir

def human_size(size):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"

def choose_model():
    # Analyse le cache Hugging Face
    cache = scan_cache_dir()
    # Ne conserver que les dépôts de modèles
    repos = [
        repo for repo in cache.repos
        if repo.repo_type == "model"
    ]
    repos.sort(key=lambda r: r.repo_id.lower())
    if not repos:
        print("Aucun modèle trouvé.")
        return None
    print("\nModèles disponibles :\n")
    for i, repo in enumerate(repos, 1):
        print(
            f"{i:2d} - {repo.repo_id:<55} "
            f"{human_size(repo.size_on_disk)}"
        )
    print("\n 0 - Quitter")
    while True:
        try:
            choice = int(input("\nVotre choix : "))
            if choice == 0:
                return None
            if 1 <= choice <= len(repos):
                selected = repos[choice - 1]
                print(f"\nModèle sélectionné : {selected.repo_id}")
                return selected.repo_id
            print("Choix invalide.")
        except ValueError:
            print("Veuillez saisir un numéro.")
# FIN def choose_model()

######################################################################################
#                                EN AVANT SIMONE                                     #
######################################################################################
# ----------------------------------------
# 1. Chargement du modèle
# ----------------------------------------
MODEL_PATH = choose_model()
if MODEL_PATH==None:
    sys.exit(1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32
)
model.eval()

# ----------------------------------------
# 2. Question et paramètres
# ----------------------------------------

question = "Qu'est-ce qu'un Transformer ?"
messages = [
    {"role": "user", "content": question}
]
# prompt = (
#     "<|im_start|>user\n"
#     + question
#     + "<|im_end|>\n"
#     + "<|im_start|>assistant\n"
# )
prompt = "The Transformer architecture is a neural network architecture that"
inputs = tokenizer(prompt, return_tensors="pt")
parametres = {
    "max_new_tokens": 100,
    "do_sample": False,
    "num_beams": 1,
}

# ----------------------------------------
# 3. Fonction de génération
# ----------------------------------------
def interroger():
    with torch.no_grad():
        output = model.generate(
            **inputs,
            **parametres
        )
    # Ne conserver que les tokens générés
    nouveaux_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        nouveaux_tokens,
        skip_special_tokens=True
    )

# ----------------------------------------
# 4. Première expérience
# ----------------------------------------
print("\n===== MODELE ORIGINAL =====")
reponse_1 = interroger()
print(f"{reponse_1=}")

# ----------------------------------------
# 5. Modification d'un poids
# ----------------------------------------
W = model.model.layers[0].self_attn.q_proj.weight
i, j = 0, 0
ancienne_valeur = W[i, j].item()
delta = 1.0
with torch.no_grad():
    W[i, j] += delta
torch.manual_seed(51154)
with torch.no_grad():
    W.normal_(0.0, 1.0)
print("\n===== MODIFICATION =====")
print("Matrice : q_proj.weight")
print("Couche  : 0")
print("Position :", (i, j))
print("Ancienne valeur :", ancienne_valeur)
print("Nouvelle valeur :", W[i, j].item())

# ----------------------------------------
# 6. Deuxième expérience
# ----------------------------------------

print("\n===== MODELE MODIFIE =====")

reponse_2 = interroger()

print(reponse_2)

# ----------------------------------------
# 7. Comparaison
# ----------------------------------------

print("\n===== COMPARAISON =====")

print("\nRéponse originale :")
print(reponse_1)

print("\nRéponse modifiée :")
print(reponse_2)
