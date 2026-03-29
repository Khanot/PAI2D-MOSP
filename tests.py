from graph_commente1 import *
from openStreetMap import *
'''
PETIT GRAPHE ALEATOIRE
np.random.seed(0)
G = generate_random_graph("test", 8, 0.6, 3)
G.affiche_dico_adj()
origin, dest = random.sample(vertices, 2) 
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")
print("RES avec dist_max =", G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10)) #seuil 10% supplémentaire
print("RES sans =", G.DijkstraMultiObjBidirectionnel(origin, dest))
'''

'''
GRAPHE DE PARIS 11 CLASSES
G = load_from_json("PAI2D-MOSP/GrapheParis.json")

origin, dest = Vertex("48.87596,2.28708"), Vertex("48.87593,2.28707")
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")

G.affiche_dico_adj()
origin = Vertex("48.86241,2.37252")
dest = Vertex("48.83828,2.34562")

print(G.distance_a_vol_d_oiseau(origin, dest))
affiche_results(G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10))
print("END")
'''

'''
GRAPHE DE PARIS 2 CLASSES
'''
G2Classes = load_from_json("PAI2D-MOSP/graphes/GrapheParis2Classes.json")

origin, dest = Vertex("48.87596,2.28708"), Vertex("48.87593,2.28707")
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")

G2Classes.affiche_dico_adj()

origin = Vertex("48.86241,2.37252")
dest = Vertex("48.83828,2.34562")

print(G2Classes.distance_a_vol_d_oiseau(origin, dest))
affiche_results(G2Classes.DijkstraMultiObjBidirectionnelSeuil(origin, dest, stop3, 10))
print("END")