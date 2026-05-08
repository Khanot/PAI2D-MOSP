# Benchmark des critères d'arrêt `stop`, `stop2`, `stop3`, `stop4`

import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import graph_poids_reduits as gpr
import graph_commente1 as gc

NB_LANDMARKS = 40

# Charger le graphe
GC = gc.load_from_json("GrapheParisCHG.json")

# Récupération des landmarks 
df = gc.read_landmarks_file("distances_landmarks_CHG.csv")
df = df.set_index('vertex_name')
columns = [f"0:L{i}" for i in range(NB_LANDMARKS)] + [f"1:L{i}" for i in range(NB_LANDMARKS)]
df = df[columns]
for c in columns:
    df[c] = df[c].astype("float16") # réduction de la taille des types pour potentiellement accélerer les calculs ?

GPR = gpr.load_from_json("GrapheParisCHG.json")

# Fonctions
type_graphes = {
    "SANS POIDS" : GC, 
    "AVEC POIDS" : GPR
}

# Tous les sommets disponibles dans le graphe
vertices = list(GC.adj[0].keys())

# Nombre de tests aléatoires
NB_TESTS = 20
SEUIL = 10
DIST_MAX = 1000  # mètres
DIST_MIN = 0000

# Pour stocker les résultats
results = defaultdict(list)

for i in range(NB_TESTS):

    # Tirer deux sommets distincts
    while True:
        source = random.choice(vertices)
        dest = random.choice(vertices)
        s = [float(x) for x in source.name.split(",")]
        d = [float(x) for x in dest.name.split(",")]

        if source == dest or 2.265 > s[1] or s[1] > 2.41 or 2.265 > d[1] or d[1] > 2.41 : # éviter les bois
            continue

        if source == dest:
            continue

        distance = GC.distance_a_vol_d_oiseau(source, dest)

        if DIST_MIN <= distance <= DIST_MAX:
            break

    print(
        f"Test {i+1}/{NB_TESTS} : "
        f"{source.name} -> {dest.name} | dist = {distance:.1f} m"
    )

    source2 = GPR.search_vertex(source.name)
    dest2 = GPR.search_vertex(dest.name)
    #"Normal": ([1, 1], [0, 0]),
    #"Elagage" : ([0.8, 0.8], [0.2, 0.2]),
    #"Gros élagage" : ([0.6, 0.6], [0.4, 0.4])
    GPR.omega = [0.8]*2
    GPR.gamma = [0.2]*2
    GPR.weight_vertices = GPR._compute_weight_vertices()

    for nom, poids in type_graphes.items():

        start = time.perf_counter()

        if nom == "SANS POIDS":

            _ = GC.AStarMultiObjBidirectionnelSeuil(
                source,
                dest,
                condition_darret=gc.stop3,
                seuil=SEUIL,
                heuris=df,
                nb_lm=NB_LANDMARKS
            )

        if nom == "AVEC POIDS":
            

            _ = GPR.AStarMultiObjBidirectionnelSeuil(
                source2,
                dest2,
                condition_darret=gpr.stop3,
                seuil=SEUIL,
                heuris=df,
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
plt.title(f"Temps d'exécution selon la distance ({DIST_MIN/1000} <= distance <= {DIST_MAX/1000} km)")
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
plt.title(f"Temps moyen d'exécution selon la distance ({DIST_MIN/1000} <= distance <= {DIST_MAX/1000} km)")
plt.legend()
plt.grid(True)
plt.show()
