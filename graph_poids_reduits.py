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

    def addLabel(self, label, direction: int, weight_vertices=None) -> None:
        vector = label.vector
        old_vectors = np.array([l.vector for l in self.label_list[direction]])
        if old_vectors.size != 0:
            if weight_vertices is None:
                dominated = dominates_in_list(vector, old_vectors)
            else:
                dominated = dominates_in_list_interval(vector, old_vectors, weight_vertices)
            self.label_list[direction] = [label] + [
                l for l, d in zip(self.label_list[direction], dominated) if not d
            ]
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

    
    def succ_label(self, new_vertex: Vertex, edge: Edge, nbClasses: int, code: int):
        vector = self.vector
        classe = ord(edge.weight[1]) - 65
        dist = edge.weight[0]
        new_vector = list(vector)
        # On plafonne à nbClasses-1 si la classe de l'arc dépasse le nombre de critères
        classe = min(classe, nbClasses - 1)
        new_vector[classe] += dist
        return Label(new_vertex, new_vector, self, code)
    
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
            if total_dist(vec) + total_dist(vec_suivant) <= dist_max:
                if self.vertex.name in chemin_opt:
                    if verbose:
                        print(f"\t\t\t {self.vertex.name} combine : sous_dist_max = {(1 + seuil/100)*chemin_opt[self.vertex.name]}")
                    # if (1 + seuil/100)*chemin_opt[self.vertex.name] >= poids_arete + vec_suivant[0]: 
                vecteurs_cout_finaux.append([vec[j] + vec_suivant[j] for j in range(nb_dim)])

        if direction == 0: # forward
            return [(self, labelListe[i], vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]
        # backward
        return [(labelListe[i], self, vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux))]
    def dominated_by_list(self, labelListe: List, weight_vertices=None) -> bool:
        vectors = np.array([label.vector for label in labelListe])
        if vectors.size == 0:
            return False
        if weight_vertices is None:
            # Mode classique
            return np.any(dominated_in_list(self.vector, vectors))
        return np.any(dominated_in_list_interval(self.vector, vectors, weight_vertices))


class Graph:

    def __init__(self, name: str, nbClasses: int) -> None:
        self.name: str = name
        self.vertices: set[Vertex] = set()
        self.edges: set[Edge] = set()
        self.adj: List[Dict[Vertex, set[Edge]], Dict[Vertex, set[Edge]]] = [dict(), dict()] # liste de successeurs, liste de prédécesseurs (donnés par les arcs)
        self.nbClasses = nbClasses # niveaux de sécurité d'un tronçon (lettres majuscules)
        self._index: dict[str, Vertex] = {}  
        self.omega = [1.01 for _ in range (nbClasses-1)]  # liste de n-1 valeurs
        self.gamma = [100 for _ in range (nbClasses-1)]  # liste de n-1 valeurs
        self.weight_vertices = self._compute_weight_vertices() if self.omega else None

    def _compute_weight_vertices(self) -> List[List[float]]:
        """
        Calcule les 2^(n-1) points extrêmes du polyèdre des poids
        via Proposition 3 (Ahn & Park 2014).
        
        Pour chaque combinaison de bornes (omega_j ou gamma_j),
        on résout : w_j = r_j * w_{j+1}, sum(w) = 1.
        """
        n = self.nbClasses
        vertices = []

        for bits in range(2 ** (n - 1)):
            # Choix des rapports r_j = omega_j (bit=0) ou gamma_j (bit=1)
            r = [self.gamma[j] if (bits >> j) & 1 else self.omega[j]
                for j in range(n - 1)]

            # w_j = (r_j * r_{j+1} * ... * r_{n-2}) * w_{n-1}
            # On calcule les produits cumulatifs depuis la droite
            w = [1.0] * n
            for j in range(n - 2, -1, -1):
                w[j] = r[j] * w[j + 1]

            # Normalisation : sum(w) = 1
            total = sum(w)
            w = [wi / total for wi in w]
            vertices.append(w)

        return vertices

    def reset_labels(self) -> None:
        """
        Réinitialise les listes de labels forward et backward
        de tous les sommets du graphe.
        """
        for vertex in self.vertices:
            vertex.label_list = [[], []]

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
    

    def DijkstraMultiObjBidirectionnel(self, source: Vertex, dest: Vertex,
                                    condition_darret=None, dist_max: float = math.inf,
                                    chemins_opt: List = [dict(), dict()],
                                    seuil: float = math.inf, verbose=False) -> List:

        T = [[], []]
        Lres = []
        wv = self.weight_vertices  # None si mode classique

        def heap_key(vector):
            if wv is None:
                return vector
            return float(np.min(np.array(wv) @ np.array(vector)))

        # Label source (forward)
        code = 0
        sourceLabel = Label(source, [0] * self.nbClasses, None, code)
        code += 1
        source.addLabel(sourceLabel, 0, wv)
        heapq.heappush(T[0], (heap_key(sourceLabel.vector), sourceLabel.code, sourceLabel))

        # Label destination (backward)
        destLabel = Label(dest, [0] * self.nbClasses, None, code)
        code += 1
        dest.addLabel(destLabel, 1, wv)
        heapq.heappush(T[1], (heap_key(destLabel.vector), destLabel.code, destLabel))

        d = 1
        if verbose:
            print("chemin optimal =", chemins_opt)

        while not condition_darret(T, Lres, self, dest):
            d = 1 - d
            if verbose:
                afficher_T(T, d)

            _, _, label = heapq.heappop(T[d])

            owner: Vertex = label.vertex
            if verbose:
                print("Sommet courant =", owner.name, " direction =", d, "code =", label.code)

            neighbors: List[Edge] = self.getNeighbors(owner, d)

            e: Edge
            for e in neighbors:
                voisin = e.vertices[1 - d]
                newLabel = label.succ_label(voisin, e, self.nbClasses, code)
                if verbose:
                    print(f"\t voisin = {newLabel.vertex.name}, {newLabel.vector[0]}, code = {newLabel.code}")
                code += 1

                # Élagage par distance à vol d'oiseau
                obj = dest if d == 0 else source
                dist_restante = self.distance_a_vol_d_oiseau(voisin, obj)
                if total_dist(newLabel.vector) + dist_restante > dist_max * (100 + seuil) / 100:
                    if verbose:
                        print("\t\tdistance totale trop grande !")
                    continue

                # Test de dominance (mode classique ou interval selon wv)
                if not newLabel.dominated_by_list(voisin.label_list[d], wv):
                    voisin.addLabel(newLabel, d, wv)
                    heapq.heappush(T[d], (heap_key(newLabel.vector), newLabel.code, newLabel))

                    # Combinaison avec labels dans la direction opposée
                    if voisin.label_list[1 - d]:
                        for c in newLabel.combine(voisin.label_list[1 - d], d, dist_max,
                                                seuil, chemins_opt[1 - d], e.weight[0]):
                            addResults(c, Lres, wv)  # wv passé à addResults

                if verbose:
                    afficher_lres(Lres)

            if verbose:
                print("---")

        self.reset_labels()
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

        oriA = copie_graphe._index[source.name]
        destA = copie_graphe._index[dest.name]

        for e in copie_graphe.edges: 
            e.weight = (e.weight[0], 'A')
        print("------------ APPEL MONO ---------------")
        mono = copie_graphe.DijkstraMultiObjBidirectionnel(oriA, destA,condition_darret)

        if not mono: 
            return []
        
        distance = total_dist(mono[0][1]) 
        chemins_opt = [mono[0][2],mono[0][3]]

        # Appliquer Dijkstra MO avec la distance à ne pas dépasser 
        distance_max: float = (1 + seuil/100) * distance 
        print("------------ APPEL BI ---------------")
        return mono[0], self.DijkstraMultiObjBidirectionnel(source, dest,condition_darret, distance_max, chemins_opt, seuil, verbose=False)

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

def total_dist(vector) -> float:
    """Remplace l'ancien vect[0] partout où on voulait la distance totale."""
    return sum(vector)

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

###version liste

def dominates_interval(cA: List[float], cB: List[float],
                        weight_vertices: List[List[float]]) -> bool:
    """
    Pairwise dominance (Section 3, Ahn & Park) :
    cA domine cB ssi w·cA <= w·cB pour tout point extrême w,
    avec inégalité stricte pour au moins un.
    """
    cA, cB = np.array(cA), np.array(cB)
    W = np.array(weight_vertices)          # shape (2^(n-1), n)
    diff = W @ (cA - cB)                   # w·cA - w·cB pour chaque sommet
    return bool(np.all(diff <= 0) and np.any(diff < 0))


def dominates_in_list_interval(v: List[float], liste_v: np.ndarray,
                                weight_vertices: List[List[float]]) -> np.ndarray:
    """
    Pour chaque vecteur de liste_v, True si v le domine (au sens interval).
    Retourne un array de booléens de taille len(liste_v).
    """
    W = np.array(weight_vertices)          # (K, n)
    sv = W @ np.array(v)                   # (K,)
    SL = W @ liste_v.T                     # (K, m)
    diff = sv[:, None] - SL                # w·v - w·c  pour chaque (sommet, chemin)
    return np.all(diff <= 0, axis=0) & np.any(diff < 0, axis=0)


def dominated_in_list_interval(v: List[float], liste_v: np.ndarray,
                                weight_vertices: List[List[float]]) -> np.ndarray:
    """
    Pour chaque vecteur de liste_v, True si v est dominé par lui.
    """
    W = np.array(weight_vertices)
    sv = W @ np.array(v)
    SL = W @ liste_v.T
    diff = SL - sv[:, None]                # w·c - w·v
    return np.all(diff <= 0, axis=0) & np.any(diff < 0, axis=0)



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

    wv = graph.weight_vertices
    TminB = np.min(np.array([lbl.vector for _, _, lbl in TB]), axis=0)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect, _, _) in Lres]

    for (_, _, lbl) in TF:
        vect = np.array(lbl.vector)
        dist_restante = graph.distance_a_vol_d_oiseau(lbl.vertex, dest)

        # Borne inf : on ajoute dist_restante uniformément sur toutes les classes
        # (on ne sait pas sur quelle classe tomberont les arcs restants)
        combined_vec = list(vect + TminB)
        # Correction sur la distance totale : au moins dist_restante à parcourir
        total_combined = total_dist(combined_vec)
        total_forward  = total_dist(vect)
        if total_forward + dist_restante > total_combined:
            # On répartit le surplus sur la classe la moins sûre (borne pessimiste)
            combined_vec[-1] += (total_forward + dist_restante - total_combined)

        combined = Label(None, combined_vec, None, -1)
        if not combined.dominated_by_list(Lres_labels, wv):
            return False
    return True


def stop4(T, Lres, graph, dest):
    TF, TB = T[0], T[1]
    if not TF or not TB:
        return True
    if not Lres:
        return False

    wv = graph.weight_vertices
    Lres_labels = [Label(None, vect, None, -1) for (_, vect, _, _) in Lres]

    for (_, _, lblF) in TF:
        vectF = np.array(lblF.vector)
        crow  = graph.distance_a_vol_d_oiseau(lblF.vertex, dest)

        for (_, _, lblB) in TB:
            vectB = np.array(lblB.vector)
            combined_vec = list(vectF + vectB)

            # S'assurer que la distance totale >= crow (borne vol d'oiseau)
            total_combined = total_dist(combined_vec)
            total_forward  = total_dist(vectF)
            if total_forward + crow > total_combined:
                combined_vec[-1] += (total_forward + crow - total_combined)

            combined = Label(None, combined_vec, None, -1)
            if not combined.dominated_by_list(Lres_labels, wv):
                return False
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


def addResults(path, liste_res, weight_vertices=None) -> None:
    chemin_ori, sommet_union, chemin_dest, vec, distance_ori, distance_dest = reconstruireChemin(path)
    liste_sommets = chemin_ori + [sommet_union] + chemin_dest
    a_retirer = []

    for r in liste_res:
        liste_sommetsTemp, vecTemp, _, _ = r

        if liste_sommetsTemp == liste_sommets:
            return

        # Dominance de vec sur vecTemp
        if weight_vertices is None:
            dom_new_over_old = dominates(vec, vecTemp)
            dom_old_over_new = dominates(vecTemp, vec)
        else:
            dom_new_over_old = dominates_interval(vec, vecTemp, weight_vertices)
            dom_old_over_new = dominates_interval(vecTemp, vec, weight_vertices)

        if dom_new_over_old:
            a_retirer.append(r)
        if dom_old_over_new:
            return

    for ar in a_retirer:
        liste_res.remove(ar)

    # [reste inchangé : calcul des distances cumulées]
    distance_avant = [0]
    for i in range(1, len(distance_dest)):
        distance_avant.append(distance_dest[i-1][1] - distance_dest[i][1])
    distance_avant_cum = np.cumsum(distance_avant)
    distance_avant_fin = [(distance_dest[i][0], int(distance_avant_cum[i]) + distance_ori[-1][1])
                          for i in range(len(distance_avant_cum))]
    chemin_avant = distance_ori + distance_avant_fin[1:]
    dico_avant = {t[0]: t[1] for t in chemin_avant}

    distance_arriere = [0]
    for i in range(len(distance_ori) - 2, -1, -1):
        distance_arriere.append(distance_ori[i+1][1] - distance_ori[i][1])
    distance_arriere_cum = list(np.cumsum(distance_arriere))
    distance_arriere_cum.reverse()
    distance_arriere_fin = [(distance_ori[i][0], int(distance_arriere_cum[i]) + distance_dest[0][1])
                            for i in range(len(distance_arriere_cum))]
    chemin_arriere = distance_arriere_fin[:-1] + distance_dest
    dico_arriere = {t[0]: t[1] for t in chemin_arriere}

    liste_res.append((liste_sommets, vec, dico_avant, dico_arriere))

### AFFICHER LES RESULTATS DE LRES DANS DIJKSTRA MO BD ###

def affiche_results(lres: List) -> None:
    ''' 
    Affiche les chemins contenus dans lres de la forme suivante : 
    Chemin i = V1 -(2)-> V4 -(5)-> V7 | secu <= A : 7, secu <= B : 5, secu <= C : 2

    :param lres: liste des resultats telle que renvoye par DijkstraMultiObjBidirectionnel
    '''
    j = 1
    
    
    for j, (chemin, vect, chemin_avant, _) in enumerate(lres, 1):
        res = f"Chemin {j} : "
        for i in range(len(chemin) - 1):
            s, s_suiv = chemin[i], chemin[i+1]
            res += s + " -(" + str(chemin_avant[s_suiv] - chemin_avant[s]) + ")-> "
        res += chemin[-1] + " | "
        res += f"distance totale : {total_dist(vect)}, "
        for k in range(len(vect)):
            res += f"km classe {chr(ord('A') + k)} : {vect[k]}, "
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
