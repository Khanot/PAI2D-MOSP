# Benchmark des critères d'arrêt `stop`, `stop2`, `stop3`, `stop4`

import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from graph_commente1 import *

NB_LANDMARKS = 40

# Charger le graphe
G2Classes = load_from_json("GrapheParis2Classes.json")

# Récupération des landmarks 
df = read_landmarks_file("distances_landmarks.csv")
df = df.set_index('vertex_name')
columns = [f"0:L{i}" for i in range(NB_LANDMARKS)] + [f"1:L{i}" for i in range(NB_LANDMARKS)]
df = df[columns]
for c in columns:
    df[c] = df[c].astype("float16") # réduction de la taille des types pour potentiellement accélerer les calculs ?

# Fonctions heuristiques
heuris_functions = {
    "oiseau": None,
    "lm": df
}

# Tous les sommets disponibles dans le graphe
vertices = list(G2Classes.adj[0].keys())

# Nombre de tests aléatoires
NB_TESTS = 10
SEUIL = 5
DIST_MAX = 3000  # mètres
DIST_MIN = 2000

# Pour stocker les résultats
results = defaultdict(list)

for i in range(NB_TESTS):

    # Tirer deux sommets distincts à moins de 2 km
    while True:
        source = random.choice(vertices)
        dest = random.choice(vertices)
        s = [float(x) for x in source.name.split(",")]
        d = [float(x) for x in dest.name.split(",")]
        #print(s, d)

        if source == dest or 2.265 > s[1] or s[1] > 2.41 or 2.265 > d[1] or d[1] > 2.41 : # éviter les bois
            continue

        distance = G2Classes.distance_a_vol_d_oiseau(source, dest)

        if DIST_MIN <= distance <= DIST_MAX:
            break

    print(
        f"Test {i+1}/{NB_TESTS} : "
        f"{source.name} -> {dest.name} | dist = {distance:.1f} m"
    )

    for nom, func in heuris_functions.items():
        # try:
        start = time.perf_counter()

        _ = G2Classes.DijkstraMultiObjBidirectionnelSeuil(
            source,
            dest,
            condition_darret=stop,
            seuil=SEUIL,
            heuris=func,
            nb_lm=NB_LANDMARKS
        )
        
        elapsed = time.perf_counter() - start
        results[nom].append((distance, elapsed)) #, nodes))


        print(f"    {nom:<6} : {elapsed:.6f} s")

        # except Exception as e:
        #     print(f"    {nom:<6} : ERREUR -> {e}")

# --- Affichage brut des points ---
plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    distances = [d for d, _ in data] # ajout n si noeuds
    times = [t for _, t in data]

    plt.scatter(distances, times, alpha=0.5, label=nom)

plt.xlabel("Distance à vol d'oiseau entre source et destination (m)")
plt.ylabel("Temps d'exécution (s)")
plt.title("Temps d'exécution selon la distance (distance <= 3 km)")
plt.legend()
plt.grid(True)
plt.show()

# --- Courbe moyenne par tranches de distance ---
BIN_SIZE = 250  # mètres

plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    bins = defaultdict(list)

    for dist, t in data: #ajout n si noeuds
        k = BIN_SIZE * int(dist // BIN_SIZE)
        bins[k].append(t)

    xs = sorted(bins.keys())
    ys = [sum(bins[x]) / len(bins[x]) for x in xs]

    plt.plot(xs, ys, marker="o", label=nom)

plt.xlabel(f"Distance (tranches de {BIN_SIZE} m)")
plt.ylabel("Temps moyen d'exécution (s)")
plt.title("Temps moyen d'exécution selon la distance (<= 3 km)")
plt.legend()
plt.grid(True)
plt.show()

'''
# --- Affichage brut des points ---
plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    distances = [d for d, _,_ in data]
    nodes = [n for _,_,n in data]

    plt.scatter(distances, nodes, alpha=0.5, label=nom)

plt.xlabel("Distance à vol d'oiseau entre source et destination (m)")
plt.ylabel("Nombre de labels développés")
plt.title("Nombre de labels développés selon la distance (distance <= 3 km)")
plt.legend()
plt.grid(True)
plt.show()

# --- Courbe moyenne par tranches de distance ---
BIN_SIZE = 250  # mètres

plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    bins = defaultdict(list)

    for dist, t, node in data:
        k = BIN_SIZE * int(dist // BIN_SIZE)
        bins[k].append(node)

    xs = sorted(bins.keys())
    ys = [sum(bins[x]) / len(bins[x]) for x in xs]

    plt.plot(xs, ys, marker="o", label=nom)

plt.xlabel(f"Distance (tranches de {BIN_SIZE} m)")
plt.ylabel("Nombre moyen de labels développés")
plt.title("Nombre moyen de labels développés selon la distance (<= 3 km)")
plt.legend()
plt.grid(True)
plt.show()
'''
