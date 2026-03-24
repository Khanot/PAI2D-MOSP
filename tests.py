from graph_commente1 import *
'''
np.random.seed(0)
G = generate_random_graph("test", 8, 0.6, 3)
G.affiche_dico_adj()
origin, dest = random.sample(vertices, 2) 
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")
print("RES avec dist_max =", G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10)) #seuil 10% supplémentaire
print("RES sans =", G.DijkstraMultiObjBidirectionnel(origin, dest))
'''

G = load_from_json("PAI2D-MOSP/GrapheParis.json")

origin, dest = Vertex("48.87596,2.28708"), Vertex("48.87593,2.28707")
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")

G.affiche_dico_adj()
origin = Vertex("48.85445,2.37223")
dest = Vertex("48.85439,2.4063")

print(G.distance_a_vol_d_oiseau(origin, dest))
affiche_results(G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10))
print("END")
