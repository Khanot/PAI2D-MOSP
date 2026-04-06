from graph_commente1 import *

# Charger le graphe
G2Classes = load_from_json("GrapheParis2Classes.json")

# Tous les sommets disponibles dans le graphe
vertices = list(G2Classes.adj[0].keys())

# Landmarks 
nb = 10
clignancourt = "48.90119,2.34002" #nord
porte_maillot = "48.8785,2.27961" #nord ouest
auteuil = "48.84996,2.26568" #ouest
porte_sevres = "48.8362,2.27818" #sud ouest
porte_italie = "48.81884,2.35977" #sud est
porte_vincennes = "48.84713,2.41077" #est
porte_bagnolet = "48.86422,2.40884" #est 
la_villette = "48.88965,2.38832" #nord est
bastille = "48.8533,2.37012" #est centre
hotel_de_ville = "48.85752,2.35118" #centre

# Ajouts supplémentaires 
# ouest
nb += 10
porte_clichy = "48.89436,2.31325"
porte_asnieres = "48.89175,2.30165"
porte_champerret = "48.88575,2.28895"
porte_dauphine = "48.87125,2.27419"
porte_la_muette = "48.86303,2.26915"
porte_passy = "48.85735,2.26466"
porte_auteuil = "48.84799,2.25733"
porte_molitor = "48.84536,2.25736"
porte_saint_cloud = "48.83804,2.25773"
quai_issy = "48.8371,2.27448" 

# est 
nb += 8
porte_pantin = "48.88963,2.39574"
porte_pre_saint_gervais = "48.8811,2.40163"
porte_lilas = "48.87725,2.40679"
porte_montreuil = "48.85397,2.41296"
porte_saint_mande = "48.84427,2.41044"
porte_doree = "48.8356,2.4072"
porte_charenton = "48.83202,2.39883"
porte_bercy = "48.83685,2.37864"

# nord 
nb += 3
porte_la_chapelle = "48.89847,2.35975"
porte_aubervilliers = "48.89872,2.37016" #pas vrmt
porte_saint_ouen = "48.89773,2.32901"

# sud 
nb += 3
porte_orleans = "48.82327,2.32546"
porte_vanves = "48.82775,2.3055"
quai_ivry = "48.8262,2.3873"

# points centraux 
nb += 6
gare_nord = "48.88089,2.36026"
republique = "48.86744,2.36322"
chatelet = "48.85772,2.34778"
montparnasse = "48.84238,2.32095"
etoile = "48.87442,2.29269"
place_italie = "48.8314,2.35693"

landmarks = [clignancourt, porte_maillot, auteuil, porte_sevres, porte_italie, porte_vincennes, porte_bagnolet, la_villette, bastille, hotel_de_ville, 
             porte_clichy,
porte_asnieres,
porte_champerret,
porte_dauphine,
porte_la_muette,
porte_passy,
porte_auteuil,
porte_molitor,
porte_saint_cloud,
quai_issy,
porte_pantin,
porte_pre_saint_gervais,
porte_lilas,
porte_montreuil,
porte_saint_mande,
porte_doree,
porte_charenton,
porte_bercy,
porte_la_chapelle,
porte_aubervilliers,
porte_saint_ouen,
porte_orleans,
porte_vanves,
quai_ivry,
gare_nord,
republique,
chatelet,
montparnasse,
etoile,
place_italie]

G2Classes.landmarks_distance_computing(landmarks)
