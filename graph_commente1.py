import numpy as np, math, heapq, json
from typing import Tuple, List, Dict

class Vertex:

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.label_list=[[],[]] # liste des listes forward et backward des labels (Dijkstra MO bi-directionnel)

    def __eq__(self, vertexPrime):
        '''
        Retourne True si le sommet self est le sommet vertexPrime (comparaison des noms),
        False sinon.

        :param vertexPrime:
        '''
        if not isinstance(vertexPrime, Vertex):
            return False
        return self.name == vertexPrime.name  
    
    def __hash__(self):
        return hash(self.name)
    
    def coordonnees(self): 
        '''
        Retourne le tuple de coordonnées associées au sommet. 
        (MOGPL : positions représentées par des sommets)
        '''
        l = self.label
        j = 0
        x = -1
        while True:
            if l[j] == "-":
                if x == -1:
                    x = int(l[:j])
                    i = j+1
                else:
                    return (x, int(l[i:j]))
            j += 1

    def addLabel(self, label, direction: int) -> None: 
        '''
        Ajoute un label à la liste des labels du sommet et supprime les anciens labels qui sont dominés.
        (Dijkstra MO bi-directionnel)
        
        :param label: label (noeud courant, vecteur de coûts, label précédent)
        :param direction: forward (0) ou backward (1)
        '''
        vector = label.vector
        old_vectors = np.array([old_label.vector for old_label in self.label_list[direction]])
        if old_vectors.size != 0:
            vecteurs_a_jeter = dominates_in_list(vector, old_vectors)
            self.label_list[direction] = [label] + [old_label for old_label, a_jeter in zip(self.label_list[direction], vecteurs_a_jeter) if not a_jeter]
        else:
            self.label_list[direction] = [label]

class Edge:

    def __init__(self, v1: Vertex, v2: Vertex, dist: float, classe: str) -> None:
        self.vertices: Tuple[Vertex, Vertex] = (v1, v2)
        self.weight = (dist, classe)  # exemple : (5,"B") -> la classe doit être une lettre majuscule autorisée (en fonction de nbClasses)


class Label:

    def __init__(self, vertex: Vertex, cost_vector: List[float], previous_label, code: int):
        self.vertex = vertex
        self.vector = cost_vector
        self.prev_label = previous_label
        self.code = code

    def labelToString(self):
        '''
        Transforme un label en chaîne de caractères.
        
        :param label: label (noeud courant, vecteur de coûts, label précédent)
        '''
        res : str = "(" + self.vertex.name + "," + str(self.vector) + ", "
        if self.prev_label == None:
            return res + "None)"
        return res + self.labelToString(self.prev_label) + ")" 

    def dominated_by_list(self, labelListe: List) -> bool:
        """
        Retourne True si le vecteur coûts du label est dominé par au moins un autre d'un label de labelListe,
        False sinon.

        :param label: label (noeud courant, vecteur de coûts, label précédent)
        :param labelListe: liste de labels
        """
        vectors = np.array([label.vector for label in labelListe])
        if vectors.size != 0:
            return np.any(dominated_in_list(self.vector, vectors))
        return False
    
    def succ_label(self, new_vertex: Vertex, edge: Edge, nbClasses: int, code: int):
        '''
        Crée un nouveau label qui succède à self (vecteur de coût mis à jour avec edge).
        
        :param label: label (noeud courant, vecteur de coûts, label précédent)
        :param edge: arc
        :param nbClasses: dimension du vecteur de coûts
        '''
        vector = self.vector
        classe = ord(edge.weight[1]) - 65 # ord('A')
        dist = edge.weight[0]
        new_vector = np.array(vector.copy()).astype(float)
        new_vector[np.arange(nbClasses) <= classe] += dist
        return Label(new_vertex, list(new_vector), self, code)
    
    def combine(self, labelListe, direction: int, dist_max: float = math.inf, seuil = math.inf, chemin_opt = dict(), poids_arete: float = math.inf,verbose=False) -> List: # A VECTORISER
        '''
        Retourne une liste des chemins combinés entre un label et une liste de labels.
        Un chemin : (label depuis source, label depuis destination, vecteur de coûts total)
        ou etiquette = label ou les deux procédures se rejoignent

        :param label: label 
        :param labelListe: liste de labels dans la direction opposée
        :param direction: direction 0 avant ou 1 arrière
        :param dist_max: distance maximale à ne pas dépasser pour un chemin (distance totale pour le premier critère)
        '''
        vecteurs_cout_finaux = []
        vec = self.vector
        nb_dim = len(vec)

        # Déterminer les vecteurs de coûts totaux
        for label in labelListe:
            vec_suivant = label.vector
            if vec[0] + vec_suivant[0] <= dist_max :
                if self.vertex.name in chemin_opt:
                    if verbose:
                        print(f"\t\t\t {self.vertex.name} combine : sous_dist_max = {(1 + seuil/100)*chemin_opt[self.vertex.name]}")
                    # if (1 + seuil/100)*chemin_opt[self.vertex.name] >= poids_arete + vec_suivant[0]: 
                vecteurs_cout_finaux.append([vec[j] + vec_suivant[j] for j in range(nb_dim)])

        if direction == 0: # forward
            return [(self, labelListe[i], vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]
        # backward
        return [(labelListe[i], self, vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]


class Graph:

    def __init__(self, name: str, nbClasses: int) -> None:
        self.name: str = name
        self.vertices: set[Vertex] = set()
        self.edges: set[Edge] = set()
        self.adj: List[Dict[Vertex, set[Edge]], Dict[Vertex, set[Edge]]] = [dict(), dict()] # liste de successeurs, liste de prédécesseurs (donnés par les arcs)
        self.nbClasses = nbClasses # niveaux de sécurité d'un tronçon (lettres majuscules)
        self._index: dict[str, Vertex] = {}  
    def copie(self): 
        '''
        Renvoie une copie du graphe.
        '''
        g = Graph(self.name, self.nbClasses) 

        # Copie des sommets
        for v in self.vertices: 
            g.add_vertex(v.name) 

        # Copie des arcs
        for e in self.edges: 
            g.add_edge(e.vertices[0].name, e.vertices[1].name, e.weight[0], e.weight[1]) 

        return g

    def nbVertices(self) -> int:
        '''
        Renvoie le nombre de sommets du graphe.
        '''
        return len(self.vertices)

    def nbEdges(self) -> int:
        '''
        Renvoie le nombre d'arcs du graphe.
        '''
        return len(self.edges)

    def add_vertex(self, name: str) -> Vertex :
        '''
        Ajoute un sommet à un graphe s'il n'y est pas déjà
        et le retourne.

        :param name: nom du sommet à ajouter
        '''
        v = Vertex(name)
        if v not in self.vertices:
            self.vertices.add(v)
            self.adj[0][v] = set()
            self.adj[1][v] = set()
            self._index[name] = v
        return self._index[name]

    def search_vertex(self, name : str) -> Vertex | None:
        '''
        Renvoie le sommet dans le graphe courant de nom "name" s'il existe, sinon renvoie None
        '''
        return self._index.get(name)
    
    def distance_a_vol_d_oiseau(self,v1 : Vertex,  v2 : Vertex) -> float:
        """
        Renvoie la distance euclidienne entre deux sommets
        dont les noms sont des coordonnées "lat,lon".
        """

        lat1, lon1 = map(float, v1.name.split(","))
        lat2, lon2 = map(float, v2.name.split(","))

        R = 6371000  # rayon de la Terre en mètres

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def add_edge(self, namev1: str, namev2: str, dist: float, classe: str) -> None:
        '''
        Ajoute une arête au graphe.
        
        :param namev1: nom d'un vecteur qui *existe* dans le graphe
        :param namev2: nom d'un vecteur qui *existe* dans le graphe
        :param dist: poids de l'arc
        :param classe: classe de l'arc
        '''
        # Récupération des vertex dans le graphe
        """
        vertex1 = next(v for v in self.vertices if v.name == namev1)
        vertex2 = next(v for v in self.vertices if v.name == namev2)
        """
        vertex1 = self._index.get(namev1)
        vertex2 = self._index.get(namev2)
        # Pas de boucle autorisée
        if vertex1 == vertex2:
            return

        # Création de l'arc
        e = Edge(vertex1, vertex2, dist, classe)

        # Ajout dans les listes d'adjacence
        self.edges.add(e)
        self.adj[0][vertex1].add(e)
        self.adj[1][vertex2].add(e)

    def delete_vertex(self, name: str) -> None: 
        '''
        Supprime un sommet du graphe s'il y est présent.

        :param name: nom du sommet à supprimer
        '''
        vertex = self._index.get(name)
        if vertex is None:
            return

        self.edges = {e for e in self.edges if vertex not in e.vertices}

        self.adj[0] = {v: {e for e in edges if e in self.edges}
                    for v, edges in self.adj[0].items() if v != vertex}
        self.adj[1] = {v: {e for e in edges if e in self.edges}
                    for v, edges in self.adj[1].items() if v != vertex}

        self.vertices.remove(vertex)
        del self._index[name]

    def delete_vertices(self, name_list: List[str]) -> None: 
        '''
        Supprime l'ensemble des sommets du graphe dont les noms sont présents dans name_list.

        :param name_list: liste de noms de sommets (qui existent ou non dans le graphe)
        '''
        for nom in name_list:
            self.delete_vertex(nom)

    def degres(self, sens: int) -> List:
        '''
        Renvoie un tableau contenant les tuples (sommet s, degré(s)) pour les sommets du graphe.

        :param sens: 1 si degrés sortants, 0 si degrés entrants
        '''
        return [(v, len(neighbors)) for (v, neighbors) in self.adj[sens].items()]


    def max_degre(self, sens: int) -> str:
        '''
        Renvoie le nom du (premier) sommet de degré maximal du graphe.

        :param sens: 1 si degrés sortants, 0 si degrés entrants
        '''
        deg = self.degres(sens)
        m = max(d[1] for d in deg)
        return next(x[0].name for x in deg if x[1] == m)

    def affiche_dico_adj(self) -> None:
        '''
        Affiche le tableau des dictionnaires d'adjacence du graphe.
        '''
        for v in self.adj[0]:
            affiche = str(v.name) + " -> ["
            for e in self.adj[0][v]:
                affiche += e.vertices[1].name + "(" + str(e.weight[0]) + "," + e.weight[1] + "), "
            if self.adj[0][v]:
                print(affiche[:-2] + "]")
            else:
                print(affiche + "]")

    def affiche_etats_avec_labels(self) -> None:
        '''
        Affiche les différents sommets et leurs labels associés (forward et backward).
        '''
        for v in self.vertices:
            res = str(v.name) + " -> FORWARD ["
            for l in v.label_list[0]:
                res += l.labelToString() + ", "
            res += "] BACKWARD ["
            for l in v.label_list[1]:
                res += l.labelToString() + ", "
            print(res + "]")


    def getNeighbors(self, vertex : Vertex, dir : int) -> List[Edge]:
        '''
        Renvoie la liste des arcs de vertex.
        
        :param vertex: sommet courant
        :param dir: direction de parcours (0: successeurs, 1: predecesseurs)
        '''
        return [e for e in self.adj[dir][vertex]]
    

    def DijkstraMultiObjBidirectionnel(self, source: Vertex, dest: Vertex,condition_darret=None, dist_max: float = math.inf, chemins_opt: List = [dict(), dict()], seuil: float = math.inf, verbose = False) -> List:
        '''
        Applique l'algorithme de Dijkstra multi-objectif bi-directionnel
        pour récupérer l'ensemble des chemins Pareto-optimaux 
        allant du sommet source au sommet dest.
        
        :param source: sommet source
        :param dest: sommet destination
        :param dist_max: distance maximale a ne pas depasser (distance totale)
        :param chemins_opt: chemin optimal pour l'objectif de la distance (dans les deux sens)
        :param seuil: seuil pour borner les longueurs des sous-chemins passant par des sommets de chemin_opt
        '''
        T = [[],[]] # tas des labels temporaires (pour les deux directions)
        Lres = [] # liste des chemins Pareto-optimaux

        # Ajout du label de source à T[0]
        code = 0 # compteur du nombre de labels créés
        sourceLabel = Label(source, [0 for _ in range(self.nbClasses)], None, code) 
        code += 1 
        source.addLabel(sourceLabel, 0)
        heapq.heappush(T[0], (sourceLabel.vector, sourceLabel.code, sourceLabel))

        # Ajout du label de destination à T[1]
        destLabel = Label(dest, [0 for _ in range(self.nbClasses)], None, code)
        code += 1
        dest.addLabel(destLabel, 1)
        heapq.heappush(T[1], (destLabel.vector, destLabel.code, destLabel))
        d: int = 1 # direction
        if verbose:
            print("chemin optimal =", chemins_opt)
        while not condition_darret(T, Lres, self, dest): # stop est la condition d'arrêt implantée plus tard
            d = 1-d # changement de direction
            if verbose:
                afficher_T(T,d)
            # Récupération d'un label dans T[d]
            _, _, label = heapq.heappop(T[d])

            # Récupération du sommet courant et de ses arcs (entrants ou sortants en fonction de la direction)
            owner: Vertex = label.vertex
            if verbose:
                print("Sommet courant =", owner.name, " direction =", d, "code =", label.code)
            neighbors: List[Edge] = self.getNeighbors(owner, d)

            # Parcours des voisins
            e: Edge
            for e in neighbors:
                voisin = e.vertices[1-d] # récupération du voisin
                newLabel = label.succ_label(voisin, e, self.nbClasses, code)
                if verbose:
                    print(f"\t voisin = {newLabel.vertex.name}, {newLabel.vector[0]}, code = {newLabel.code}")
                code += 1

                # Si la distance parcourue + distance minimale possible > dist_max, on n'exploite pas le label
                if d==0:
                    obj=dest
                else:
                    obj=source
                dist_restante=self.distance_a_vol_d_oiseau(voisin,obj)
                if newLabel.vector[0] + dist_restante> dist_max*(100+seuil)/100: 
                    if verbose:
                        print("\t\tdistance totale trop grande !")
                    continue
                """
                On abandonne cette idée pour l'instant
                # Si on est sur un sommet du chemin optimal et que la distance parcourue > (1 + seuil/100)*sous-distance optimale, on n'exploite pas le label
                if newLabel.vertex.name in chemins_opt[d]:
                    print(f"\t\tdist opt = {chemins_opt[d][newLabel.vertex.name]}, a ne pas depasser = {(1 + seuil/100)*chemins_opt[d][newLabel.vertex.name]}")
                    if newLabel.vector[0] > (1 + seuil/100)*chemins_opt[d][newLabel.vertex.name]:
                        print("\t\tdistance du sous-chemin trop grande !")
                        continue   
                """  
                
                # Si le nouveau label n'est pas dominé par ceux dans la liste de voisins, on l'ajoute
                if not newLabel.dominated_by_list(voisin.label_list[d]): 
                    voisin.addLabel(newLabel, d)
                    heapq.heappush(T[d], (newLabel.vector, newLabel.code, newLabel))

                    # Si la liste des labels dans l'autre direction n'est pas vide, combiner les chemins
                    if voisin.label_list[1-d] != []:
                        # print("\tbizarre", voisin.name, voisin.label_list[1-d])
                        for c in newLabel.combine(voisin.label_list[1-d], d, dist_max, seuil, chemins_opt[1-d], e.weight[0]):
                            addResults(c, Lres)
                if verbose:
                    afficher_lres(Lres)
            # print("lres", Lres)
            # print("apres", T[d])
            if verbose:
                print("---")

        return Lres

    def DijkstraMultiObjBidirectionnelSeuil(self, source: Vertex, dest: Vertex,condition_darret, seuil: float) -> List: 
        '''
        Applique l'algorithme de Dijkstra multi-objectif bi-directionnel
        pour récupérer l'ensemble des chemins Pareto-optimaux 
        allant du sommet source au sommet dest
        avec la longueur d'un chemin qui ne dépasse pas 100 + seuil % du chemin optimal (mono-objectif).
        
        :param source: sommet source
        :param dest: sommet destination
        :param seuil: pourcentages supplémentaires du chemin optimal 
        '''
        # Appliquer Dijkstra en version mono-objectif (distance totale) pour récupérer le chemin de longueur minimale
        copie_graphe: Graph = self.copie()
        # print("AVANT TOUT")
        # print("Graphe :")
        # print("\tSommets :")
        # for v in self.vertices:
        #     print("nom :", v.name, "| label_list :", v.label_list, "| id :", id(v))
        # print("\tAretes")
        # for e in self.edges:
        #     print("sommet 1 : ", e.vertices[0].name, id(e.vertices[0]), "| sommet 2 : ", e.vertices[1].name,  id(e.vertices[1]), "| poids :", e.weight)

        # print("Copie du graphe :")
        # print("\tSommets :")
        # for v in copie_graphe.vertices:
        #     print("nom :", v.name, "| label_list :", v.label_list, "| id :", id(v))
        # print("\tAretes")
        # for e in copie_graphe.edges:
        #     print("sommet 1 : ", e.vertices[0].name, id(e.vertices[0]), "| sommet 2 : ", e.vertices[1].name,  id(e.vertices[1]), "| poids :", e.weight)

        oriA = copie_graphe._index[source.name]
        destA = copie_graphe._index[dest.name]

        for e in copie_graphe.edges: 
            e.weight = (e.weight[0], 'A')
        print("------------ APPEL MONO ---------------")
        liste_distance = copie_graphe.DijkstraMultiObjBidirectionnel(oriA, destA,condition_darret)
        
        # print("APRES MONO")
        # print("Graphe :")
        # print("\tSommets :")
        # for v in self.vertices:
        #     print("nom :", v.name, "| label_list :", v.label_list, "| id :", id(v))
        # print("\tAretes")
        # for e in self.edges:
        #     print("sommet 1 : ", e.vertices[0].name, id(e.vertices[0]), "| sommet 2 : ", e.vertices[1].name,  id(e.vertices[1]), "| poids :", e.weight)

        # print("Copie du graphe :")
        # print("\tSommets :")
        # for v in copie_graphe.vertices:
        #     print("nom :", v.name, "| label_list :", v.label_list, "| id :", id(v))
        # print("\tAretes")
        # for e in copie_graphe.edges:
        #     print("sommet 1 : ", e.vertices[0].name, id(e.vertices[0]), "| sommet 2 : ", e.vertices[1].name,  id(e.vertices[1]), "| poids :", e.weight)

        if not liste_distance: 
            return []
        
        distance = liste_distance[0][1][0] 
        chemins_opt = [liste_distance[0][2],liste_distance[0][3]]

        # Appliquer Dijkstra MO avec la distance à ne pas dépasser 
        distance_max: float = (1 + seuil/100) * distance 
        print("------------ APPEL BI ---------------")
        return self.DijkstraMultiObjBidirectionnel(source, dest,condition_darret, distance_max, chemins_opt, seuil, verbose=False)

    def save_to_json(self, filename: str):
        '''
        Enregistre un graphe sous format json.

        :param filename: nom du fichier dans lequel sera enregistré le graphe
        '''
        data = {
            "name": self.name,
            "nbClasses": self.nbClasses,
            "vertices": [v.name for v in self.vertices],
            "edges": [
                {
                    "from": e.vertices[0].name,
                    "to": e.vertices[1].name,
                    "dist": e.weight[0],
                    "classe": e.weight[1]
                }
                for e in self.edges
            ]
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

def load_from_json(filename: str):
    '''
    Génère un graphe issu d'un format json.

    :param filename: nom du fichier dans lequel est enregistré le graphe
    '''
    with open(filename, "r") as f:
        data = json.load(f)

    G = Graph(data["name"], data["nbClasses"])

    for v in data["vertices"]:
        G.add_vertex(v)

    for e in data["edges"]:
        G.add_edge(e["from"], e["to"], e["dist"], e["classe"])

    return G

### PARETO DOMINANCE ###

def dominates(v1: List[float], v2: List[float]) -> bool:
    '''
    Retourne True si le vecteur v1 domine v2,
    False sinon.

    :param v1: vecteur de coût 
    :param v2: vecteur de coût
    '''
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.all(v1 <= v2) and np.any(v1 < v2)

def dominates_in_list(v, liste_v):
    ''' 
    Renvoie un array de booleens : pour chaque vecteur de liste_v,
    True si v le domine, False sinon. 

    :param v: vecteur de coût 
    :param liste_v: liste de vecteurs de coût
    '''
    return np.all(liste_v >= v, axis=1) & np.any(liste_v > v, axis=1)


def dominated_in_list(v, liste_v):
    ''' 
    Renvoie un array de booleens : pour chaque vecteur de liste_v,
    True si v est domine par lui, False sinon. 

    :param v: vecteur de coût 
    :param liste_v: liste de vecteurs de coût
    '''
    return np.all(liste_v <= v, axis=1) & np.any(liste_v < v, axis=1)

### CONDITION D'ARRET DANS DIJKSTRA MO BD ###

def stop(T, Lres,graph,dest):
    '''
    Retourne True si Tmin est dominé par au moins un chemin de Lres,
    False sinon.
    (Dijkstra MO BD : boucle a arreter si True)
    
    :param T: liste des labels temporaires
    :param Lres: liste des chemins Pareto-optimaux
    '''
    TF = T[0] # liste des etiquettes temporaires (forward)
    TB = T[1] # liste des etiquettes temporaires (backward)

    # Il n'y a plus de labels dans l'un des deux tas
    if not TF or not TB:
        return True

    # Forward : construire le vecteur de coûts minimum pour chaque objectif à partir des vecteurs de TF
    TminF = np.min(np.array([vecteur[0] for vecteur in TF]), axis = 0)

    # Backward : construire le vecteur de cout minimum pour chaque objectif a partir des vecteurs de TB
    TminB = np.min(np.array([vecteur[0] for vecteur in TB]), axis = 0)

    Tmin = list(TminF + TminB)
    labTmin = Label(None, Tmin, None, -1)
    Lres_labels = np.array([Label(None, vect, None, -1) for (_, vect,_ , _) in Lres])
    return labTmin.dominated_by_list(Lres_labels) 

def stop2(T, Lres,graph,dest):
    TF, TB = T[0], T[1]

    if not TF or not TB:
        return True

    if not Lres:
        return False

    # Borne inférieure uniquement sur TB
    TminB = np.min(np.array([vecteur[0] for vecteur in TB]), axis=0)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect, _, _) in Lres]

    # Pour chaque label forward, vérifier si lF + TminB est dominé
    for (vect, _, _) in TF:
        combined = Label(None, list(np.array(vect) + TminB), None, -1)
        if not combined.dominated_by_list(Lres_labels):
            return False  # ce label peut encore produire un chemin non dominé

    return True

def stop3(T, Lres, graph, dest):
    TF, TB = T[0], T[1]

    if not TF or not TB:
        return True

    if not Lres:
        return False

    TminB = np.min(np.array([vecteur[0] for vecteur in TB]), axis=0)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect, _, _) in Lres]

    for (vect, _, lbl) in TF:
        # Crow-fly pour le 1er objectif (distance)
        dist_restante = graph.distance_a_vol_d_oiseau(lbl.vertex, dest)

        # Crow-fly pour le 1er objectif, TminB pour les objectifs de sécurité
        combined_vec = [vect[0] + dist_restante] + [v + b for v, b in zip(vect[1:], TminB[1:])]
        combined = Label(None, combined_vec, None, -1)

        if not combined.dominated_by_list(Lres_labels):
            return False

    return True


def stop4(T, Lres, graph, dest):
    """
    Arrête la recherche si toutes les combinaisons possibles
    (label forward + borne vol d'oiseau + label backward)
    sont déjà dominées par un chemin de Lres.

    :param T: [TF, TB]
    :param Lres: liste des chemins Pareto-optimaux trouvés
    :param graph: graphe avec distance_a_vol_d_oiseau(u, dest)
    :param dest: sommet destination
    """
    TF, TB = T[0], T[1]

    # Plus rien à explorer d'un côté
    if not TF or not TB:
        return True

    # Aucun résultat trouvé => on ne peut pas pruner
    if not Lres:
        return False

    Lres_labels = [
        Label(None, vect, None, -1)
        for (_, vect, _, _) in Lres
    ]

    # Pour chaque combinaison forward/backward
    for (vectF, _, lblF) in TF:

        # Borne vol d'oiseau pour le premier objectif
        crow = graph.distance_a_vol_d_oiseau(lblF.vertex, dest)

        for (vectB, _, _) in TB:

            # On remplace le premier objectif backward
            # par la borne vol d'oiseau si elle est plus optimiste
            first_obj = vectF[0] + min(crow, vectB[0])

            combined_vec = [first_obj] + [
                f + b for f, b in zip(vectF[1:], vectB[1:])
            ]

            combined = Label(None, combined_vec, None, -1)

            # S'il existe UNE combinaison non dominée,
            # il faut continuer la recherche
            if not combined.dominated_by_list(Lres_labels):
                return False

    # Toutes les combinaisons sont dominées
    return True


### FONCTIONS LIEES AUX CHEMINS DANS DIJKSTRA MO BD ###

def reconstruireChemin(chemin):
    '''
    Retourne un chemin reconstruit, i.e.
    ([liste des sommets du chemin de source à destination], vecteur de coûts total, dictionnaire du détail du chemin avant, idem mais arrière)
    
    :param chemin: (label depuis source, label depuis dest, vecteur de coûts total)
    '''
    depuis_ori, depuis_dest, vect = chemin
    
    # Chemin jusqu'à origine
    chemin_ori = []
    sommet = depuis_ori.vertex
    label_prec = depuis_ori.prev_label
    sommet_union = sommet.name
    distance_ori = [(depuis_ori.vertex.name, depuis_ori.vector[0])]
    while label_prec != None:
        sommet = label_prec.vertex
        chemin_ori = [sommet.name] + chemin_ori
        distance_ori.append((sommet.name,label_prec.vector[0]))
        label_prec = label_prec.prev_label
    distance_ori.reverse()

    # Chemin jusqu'à destination
    chemin_dest = []
    label_prec = depuis_dest.prev_label
    distance_dest = [(depuis_dest.vertex.name, depuis_dest.vector[0])]
    while label_prec != None:
        sommet = label_prec.vertex
        chemin_dest = chemin_dest + [sommet.name]
        distance_dest.append((sommet.name,label_prec.vector[0]))
        label_prec = label_prec.prev_label

    chemin = chemin_ori + [sommet_union] + chemin_dest
    return (chemin_ori, sommet_union, chemin_dest, vect, distance_ori, distance_dest)


def addResults(path, liste_res) -> None: # A VECTORISER
    """
    Reconstruit le chemin path et l'ajoute à liste_res s'il n'est pas dominé par un chemin de liste_res.

    :param path: chemin à ajouter (depuis_ori, depuis_dest, vecteur cout)
    :param liste_res: liste des chemins (liste_sommets, vecteur cout) déjà découverts
    """
    chemin_ori, sommet_union, chemin_dest, vec, distance_ori, distance_dest = reconstruireChemin(path)
    liste_sommets = chemin_ori + [sommet_union] + chemin_dest
    a_retirer = []

    # Pour tout chemin r dans Lres
    for r in liste_res:
        # chemin, vecTemp, liste avant, liste arriere
        liste_sommetsTemp, vecTemp,_,_ = r

        # Chemin déjà dans liste_res
        if liste_sommetsTemp == liste_sommets:
            return

        # Si le chemin est dominé par le nouveau chemin path, on retire r
        if dominates(vec, vecTemp):
            a_retirer.append(r)

        # Si le nouveau chemin est dominé, on ne change rien a Lres
        if dominates(vecTemp, vec):
            return
    
    for ar in a_retirer:
        liste_res.remove(ar)

    # Conserver les distances cumulees pour les sous-chemins (seuil par etape) 
    distance_avant = [0]
    for i in range(1, len(distance_dest)):
        distance_avant.append(distance_dest[i-1][1]-distance_dest[i][1])
    distance_avant_cum =  np.cumsum(distance_avant)
    distance_avant_fin = [(distance_dest[i][0], int(distance_avant_cum[i]) + distance_ori[-1][1]) for i in range(len(distance_avant_cum))]
    chemin_avant = distance_ori + distance_avant_fin[1:]
    dico_avant = {t[0]:t[1] for t in chemin_avant}

    distance_arriere = [0]
    for i in range(len(distance_ori)-2,-1,-1):
        distance_arriere.append(distance_ori[i+1][1]-distance_ori[i][1])
    distance_arriere_cum =  list(np.cumsum(distance_arriere))
    distance_arriere_cum.reverse()
    distance_arriere_fin = [(distance_ori[i][0], int(distance_arriere_cum[i]) + distance_dest[0][1]) for i in range(len(distance_arriere_cum))]
    chemin_arriere = distance_arriere_fin[:-1] + distance_dest
    dico_arriere = {t[0]:t[1] for t in chemin_arriere}

    liste_res.append((liste_sommets, vec, dico_avant, dico_arriere))  

### AFFICHER LES RESULTATS DE LRES DANS DIJKSTRA MO BD ###

def affiche_results(lres: List) -> None:
    ''' 
    Affiche les chemins contenus dans lres de la forme suivante : 
    Chemin i = V1 -(2)-> V4 -(5)-> V7 | secu <= A : 7, secu <= B : 5, secu <= C : 2

    :param lres: liste des resultats telle que renvoye par DijkstraMultiObjBidirectionnel
    '''
    j = 1
    
    for chemin, vect, chemin_avant, _ in lres:
        res = f"Chemin {j} : "
        for i in range(len(chemin)-1):
            sommet = chemin[i]
            sommet_suiv = chemin[i+1]
            res += sommet + " -(" + str(chemin_avant[sommet_suiv]-chemin_avant[sommet]) + ")-> "
        res += chemin[-1] + " | "
        for k in range(len(vect)):
            res += f"secu <= {chr(ord('A') + k)} : " + str(vect[k]) + ", "
        j += 1
        print(res[:-2])


### GENERATION DE GRAPHES ALEATOIRES ###
    
def generate_random_graph(name: str, nbVertex: int, probaEdge: float, nbClasses: int):
    '''
    Génère un graphe aléatoire.
    
    :param name: nom du graphe
    :param nbVertex: nombre de sommets
    :param probaEdge: probabilité d'ajouter un arc pour chaque paire de sommets (distance entre 1 et 50)
    :param nbClasses: nombre de niveaux de sécurité / dimensions du vecteur de coûts
    '''
    G = Graph(name, nbClasses)

    # Création de nbVertex sommets
    for i in range(nbVertex):
        G.add_vertex(f"V{i}")

    # Création d'arcs
    ascii_A = ord('A')
    for i in range(nbVertex):
        for j in range(nbVertex):
            if i != j and np.random.random() < probaEdge:
                G.add_edge(f"V{i}", f"V{j}", np.random.randint(1, 50), chr(ascii_A + np.random.randint(nbClasses)))

    return G

def afficher_T(T,d):
    res = f"T[{d}] =\n" 
    for t in T[d]:
        res += f"\t{t[2].vertex.name} : {[int(x) for x in t[0]]}, code = {t[2].code}\n"
    
    print(res)

def afficher_lres(lres):
    res = "\t\tLres =\n"
    for liste_sommets, vec, _, _ in lres:
        res += f"\t\t\tchemin : {liste_sommets}, cout = {vec}\n"
    print(res)


# G = generate_random_graph("test", 10, 0.7, 3)
# for v in G.vertices:
#     if v.name == "V1":
#         source = v 
#     if v.name == "V2":
#         dest = v
"""
G = Graph("eh", 2)

V1=G.add_vertex("48.87596,2.28708")
V2=G.add_vertex("48.87593,2.28707")
V3=G.add_vertex("48.8759,2.28696")
V4=G.add_vertex("48.83602,2.42751")
G.add_edge("48.87596,2.28708", "48.87593,2.28707", G.distance_a_vol_d_oiseau(V1,V2), "A")
G.add_edge("48.87596,2.28708", "48.8759,2.28696",G.distance_a_vol_d_oiseau(V1,V3), "B")
G.add_edge("48.87593,2.28707", "48.8759,2.28696", G.distance_a_vol_d_oiseau(V2,V3), "A")
G.add_edge("48.8759,2.28696", "48.83602,2.42751", G.distance_a_vol_d_oiseau(V3,V4), "A")

#48° 50′ 46″ nord, 2° 21′ 21″ est Place Jussieu
#48.857362, 2.306119 Rue Cler
affiche_results(G.DijkstraMultiObjBidirectionnelSeuil(V1,V4, 10))
"""