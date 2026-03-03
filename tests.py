from graph_commente import *
'''
print((4, 2) < (1,3))
np.random.seed(0)

G = Graph.load_from_json("mon_graphe.json")
G.affiche_dico_adj()

vertices = list(G.vertices)
origin, dest = [Vertex('V4'), Vertex('V5')]
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")
print("RES avec dist_max =", G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10)) #seuil 10% supplémentaire
print("RES sans =", G.DijkstraMultiObjBidirectionnel(origin, dest))
'''
G = generate_random_graph("test", 8, 0.6, 3)
G.affiche_dico_adj()

vertices = list(G.vertices)
origin, dest = random.sample(vertices, 2) 
print(f"ORIGINE = {origin.name}, ARRIVEE = {dest.name}")
print("RES avec dist_max =", G.DijkstraMultiObjBidirectionnelSeuil(origin, dest, 10)) #seuil 10% supplémentaire
print("RES sans =", G.DijkstraMultiObjBidirectionnel(origin, dest))
