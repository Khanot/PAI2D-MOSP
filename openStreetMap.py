#Telecharger depuis https://download.geofabrik.de/europe/france/ (ile-de-france-260216.osm.pbf)
#pip intall osmnx
from graph_commente1 import *
import osmnx as ox



def edge_to_class(data):
    """
    Convertit un edge OSMnx en classe qualitative A-Z.
    
    """

    highway = data.get("highway", "")
    surface = data.get("surface", "")
    cycleway = data.get("cycleway", "")
    maxspeed = data.get("maxspeed", 30)

    # normaliser maxspeed
    try:
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        maxspeed = int(maxspeed)
    except:
        maxspeed = 30

    # --- scoring sécurité vélo ---
    score = 0

    # pistes cyclables
    if highway in ["cycleway", "path"]:
        score += 0
    elif highway in ["residential", "living_street"]:
        score += 2
    elif highway in ["secondary", "tertiary"]:
        score += 4
    elif highway in ["primary", "trunk"]:
        score += 7
    else:
        score += 5

    # présence aménagement vélo
    if cycleway in ["track"]:
        score -= 2
    elif cycleway in ["lane"]:
        score -= 1

    # surface mauvaise
    if surface in ["gravel", "dirt", "ground"]:
        score += 2

    # vitesse élevée
    if maxspeed > 50:
        score += 2

    # clamp score 0–11
    score = max(0, min(score, 11))

    # convertir en lettre A-Z
    classe = chr(ord('A') + score)
    return classe



def convert_osmnx_to_custom_graph(G_osm, name="OSM_graph", nbClasses=11):
    
    custom = Graph(name, nbClasses)

    node_name = {}

    # --- sommets ---
    for node, data in G_osm.nodes(data=True):
        lat = round(data["y"], 5)
        lon = round(data["x"], 5)
        name_v = f"{lat},{lon}"
        node_name[node] = name_v
        custom.add_vertex(name_v)


    # --- ajouter edges ---
    for u, v, k, data in G_osm.edges(keys=True, data=True):

        dist = data.get("length", 1.0)
        classe = edge_to_class(data)

        try:
            custom.add_edge(node_name[u], node_name[v], round(float(dist),3), classe)
        except:
            pass

    return custom

"""
G_osm = ox.graph_from_place("Paris, France", network_type="bike")

G_custom = convert_osmnx_to_custom_graph(G_osm)


G_custom.save_to_json("testGraphe.json")
"""

G=load_from_json("GrapheParis.json")
G.affiche_dico_adj()
V1=G.search_vertex("48.85445,2.37223")
V2=next(iter(G.adj[0][V1])).vertices[1]
V3=next(iter(G.adj[0][V2])).vertices[1]
print(G.distance_a_vol_d_oiseau(V1,V2))
print(G.DijkstraMultiObjBidirectionnelSeuil(V1,V2,100))

print("END")
