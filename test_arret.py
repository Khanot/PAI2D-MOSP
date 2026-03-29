# Benchmark des critères d'arrêt `stop`, `stop2`, `stop3`, `stop4`

import random
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from graph_commente1 import *

# Charger le graphe
G2Classes = load_from_json("GrapheParis2Classes.json")

# Associer les fonctions à leur nom
stop_functions = {
    "stop": stop,
    "stop2": stop2,
    "stop3": stop3,
    "stop4": stop4,
}

# Tous les sommets disponibles dans le graphe
vertices = list(G2Classes.adj[0].keys())

# Nombre de tests aléatoires
NB_TESTS = 200
SEUIL = 10
DIST_MAX = 2000  # mètres

# Pour stocker les résultats
results = defaultdict(list)

for i in range(NB_TESTS):

    # Tirer deux sommets distincts à moins de 2 km
    while True:
        source = random.choice(vertices)
        dest = random.choice(vertices)

        if source == dest:
            continue

        distance = G2Classes.distance_a_vol_d_oiseau(source, dest)

        if distance <= DIST_MAX:
            break

    print(
        f"Test {i+1}/{NB_TESTS} : "
        f"{source.name} -> {dest.name} | dist = {distance:.1f} m"
    )

    for nom, func in stop_functions.items():
        try:
            start = time.perf_counter()

            G2Classes.DijkstraMultiObjBidirectionnelSeuil(
                source,
                dest,
                condition_darret=func,
                seuil=SEUIL
            )

            elapsed = time.perf_counter() - start
            results[nom].append((distance, elapsed))

            print(f"    {nom:<6} : {elapsed:.6f} s")

        except Exception as e:
            print(f"    {nom:<6} : ERREUR -> {e}")

# --- Affichage brut des points ---
plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    distances = [d for d, _ in data]
    times = [t for _, t in data]

    plt.scatter(distances, times, alpha=0.5, label=nom)

plt.xlabel("Distance à vol d'oiseau entre source et destination (m)")
plt.ylabel("Temps d'exécution (s)")
plt.title("Temps d'exécution selon la distance (distance <= 2 km)")
plt.legend()
plt.grid(True)
plt.savefig("graphiques/temps_execution_tous.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Courbe moyenne par tranches de distance ---
BIN_SIZE = 250  # mètres

plt.figure(figsize=(10, 6))

for nom, data in results.items():
    if not data:
        continue

    bins = defaultdict(list)

    for dist, t in data:
        k = BIN_SIZE * int(dist // BIN_SIZE)
        bins[k].append(t)

    xs = sorted(bins.keys())
    ys = [sum(bins[x]) / len(bins[x]) for x in xs]

    plt.plot(xs, ys, marker="o", label=nom)

plt.xlabel(f"Distance (tranches de {BIN_SIZE} m)")
plt.ylabel("Temps moyen d'exécution (s)")
plt.title("Temps moyen d'exécution selon la distance (<= 2 km)")
plt.legend()
plt.grid(True)
plt.savefig("graphiques/temps_moyen_tous.png", dpi=300, bbox_inches="tight")
plt.show()


# --- Même graphiques mais sans stop4 ---
results_sans_stop4 = {
    nom: data
    for nom, data in results.items()
    if nom != "stop4"
}

# Nuage de points
plt.figure(figsize=(10, 6))

for nom, data in results_sans_stop4.items():
    distances = [d for d, _ in data]
    times = [t for _, t in data]

    plt.scatter(distances, times, alpha=0.5, label=nom)

plt.xlabel("Distance à vol d'oiseau entre source et destination (m)")
plt.ylabel("Temps d'exécution (s)")
plt.title("Temps d'exécution selon la distance (sans stop4)")
plt.legend()
plt.grid(True)
plt.savefig("graphiques/temps_execution_sans_stop4.png", dpi=300, bbox_inches="tight")
plt.show()

# Courbe moyenne par tranches
plt.figure(figsize=(10, 6))

for nom, data in results_sans_stop4.items():
    bins = defaultdict(list)

    for dist, t in data:
        k = BIN_SIZE * int(dist // BIN_SIZE)
        bins[k].append(t)

    xs = sorted(bins.keys())
    ys = [sum(bins[x]) / len(bins[x]) for x in xs]

    plt.plot(xs, ys, marker="o", label=nom)

plt.xlabel(f"Distance (tranches de {BIN_SIZE} m)")
plt.ylabel("Temps moyen d'exécution (s)")
plt.title("Temps moyen d'exécution selon la distance (sans stop4)")
plt.legend()
plt.grid(True)
plt.savefig("graphiques/temps_moyen_sans_stop4.png", dpi=300, bbox_inches="tight")
plt.show()
