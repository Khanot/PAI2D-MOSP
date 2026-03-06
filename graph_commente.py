import numpy as np, random, math, heapq, json
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
            j+=1

    def addLabel(self, label, direction: int) -> None: 
        '''
        Ajoute un label à la liste des labels du sommet.
        (Dijkstra MO bi-directionnel)
        
        :param label: label (noeud courant, vecteur de coûts, label précédent)
        :param direction: forward (0) ou backward (1)
        '''
        vector = label.vector
        new_list = []

        # Construction de la nouvelle liste de labels pour le sommet.
        for old_label in self.label_list[direction]:
            vectorTemp = old_label.vector

            # Si le nouveau label est dominé, on garde notre liste initiale.
            if dominates(vectorTemp, vector):
                return
            
            # On ne garde pas les anciens labels désormais dominés par le nouveau label.
            if not dominates(vector, vectorTemp):
                new_list.append(old_label)

        # Si le nouveau label n'est pas dominé, on met à jour notre liste (sans les labels dominés par le nouveau label)
        new_list.append(label)
        self.label_list[direction] = new_list

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
        for label in labelListe:
            if dominates(label.vector, self.vector):
                return True
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
        
        return Label(new_vertex, [dist + vector[i] if i <= classe else vector[i] for i in range(nbClasses)], self, code)
    
    def combine(self, labelListe, direction: int, dist_max: float = math.inf) -> List:
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
            vecteurs_cout_finaux.append([vec[j] + vec_suivant[j] for j in range(nb_dim)])

        if direction == 0: # forward
            return [(self, labelListe[i], vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux)) if vecteurs_cout_finaux[i][0] <= dist_max]
        # backward
        return [(labelListe[i], self, vecteurs_cout_finaux[i]) for i in range(len(vecteurs_cout_finaux)) if vecteurs_cout_finaux[i][0] <= dist_max]


class Graph:

    def __init__(self, name: str, nbClasses: int) -> None:
        self.name: str = name
        self.vertices: set[Vertex] = set()
        self.edges: set[Edge] = set()
        self.adj: List[Dict[Vertex, set[Edge]], Dict[Vertex, set[Edge]]] = [dict(), dict()] # liste de successeurs, liste de prédécesseurs (donnés par les arcs)
        self.nbClasses = nbClasses # niveaux de sécurité d'un tronçon (lettres majuscules)

    def copy(self): 
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

    def add_vertex(self, name: str) -> Vertex:
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
        return v 
    

    def add_edge(self, namev1: str, namev2: str, dist: float, classe: str) -> None:
        '''
        Ajoute une arête au graphe.
        
        :param namev1: nom d'un vecteur qui *existe* dans le graphe
        :param namev2: nom d'un vecteur qui *existe* dans le graphe
        :param dist: poids de l'arc
        :param classe: classe de l'arc
        '''
        # Récupération des vecteurs dans le graphe
        vertex1 = next(v for v in self.vertices if v.name == namev1)
        vertex2 = next(v for v in self.vertices if v.name == namev2)

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
        # Retrouver le sommet à supprimer
        vertex = Vertex(name)
        if vertex not in self.vertices:
            return #le sommet n'est déjà pas dans le graphe

        # Mettre à jour l'ensemble des arcs
        self.edges = {e for e in self.edges if vertex not in e.vertices} 

        # Construction des nouveaux dictionnaires d'adjacence
        new_adj = [{sommet:set() for sommet in self.adj[0]}, {sommet:set() for sommet in self.adj[1]}]

        for sommet, ens in self.adj[0]:
            for e in ens:
                if e not in self.edges:
                    new_adj[0][sommet].remove(e)

        for sommet, ens in self.adj[1]:
            for e in ens:
                if e not in self.edges:
                    new_adj[1][sommet].remove(e)

        # Retirer le sommet de l'ensemble des sommets
        self.vertices.remove(vertex)

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
        return [(v, len(neighbors)) for (v, neighbors) in self.adj[sens]]


    def max_degre(self, sens: int) -> str:
        '''
        Renvoie le nom du (premier) sommet de degré maximal du graphe.

        :param sens: 1 si degrés sortants, 0 si degrés entrants
        '''
        deg = self.degres(sens)
        return [x for x in deg if x[1] == max([lenNeighbors[1] for lenNeighbors in deg])][0][0].name


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
    

    def DijkstraMultiObjBidirectionnel(self, source: Vertex, dest: Vertex, dist_max: float = math.inf) -> List:
        '''
        Applique l'algorithme de Dijkstra multi-objectif bi-directionnel
        pour récupérer l'ensemble des chemins Pareto-optimaux 
        allant du sommet source au sommet dest.
        
        :param source: sommet source
        :param dest: sommet destination
        '''
        T = [[],[]] # tas des labels temporaires (pour les deux directions)
        Lres = [] # liste des chemins Pareto-optimaux

        # Ajout du label de source à T[0]
        code = 0 # compteur du nombre de labels créés
        sourceLabel = Label(source, [0 for _ in range(self.nbClasses)], None, code) 
        code +=1 
        source.addLabel(sourceLabel, 0)
        heapq.heappush(T[0], (sourceLabel.vector, sourceLabel.code, sourceLabel))

        # Ajout du label de destination à T[1]
        destLabel = Label(dest, [0 for _ in range(self.nbClasses)], None, code)
        code += 1
        dest.addLabel(destLabel, 1)
        heapq.heappush(T[1], (destLabel.vector, destLabel.code, destLabel))

        d: int = 1 # direction
        while not (stop(T, Lres)): # stop est la condition d'arrêt implantée plus tard
            d = 1-d # changement de direction

            # Récupération d'un label dans T[d]
            _, _, label = heapq.heappop(T[d])

            # Récupération du sommet courant et de ses arcs (entrants ou sortants en fonction de la direction)
            owner: Vertex = label.vertex
            neighbors: List[Edge] = self.getNeighbors(owner, d)

            # Parcours des voisins
            e: Edge
            for e in neighbors:
                voisin = e.vertices[1-d] # récupération du voisin
                newLabel = label.succ_label(voisin, e, self.nbClasses, code)
                code += 1

                # Si la distance parcourue > dist_max, on n'exploite pas le label
                if newLabel.vector[0] > dist_max: 
                    continue
                
                # Si le nouveau label n'est pas dominé par ceux dans la liste de voisins, on l'ajoute
                if not newLabel.dominated_by_list(voisin.label_list[d]): 
                    voisin.addLabel(newLabel, d)
                    heapq.heappush(T[d], (newLabel.vector, newLabel.code, newLabel))

                    # Si la liste des labels dans l'autre direction n'est pas vide, combiner les chemins
                    if voisin.label_list[1-d] != []:
                        for c in newLabel.combine(voisin.label_list[1-d], d, dist_max):
                            addResults(c, Lres)

        return Lres

    def DijkstraMultiObjBidirectionnelSeuil(self, source: Vertex, dest: Vertex, seuil: float) -> List: 
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
        copie_graphe = self.copy()

        for v in copie_graphe.vertices: # récupérer les copies des sommets source et destination
            if v.name == source.name:
                oriA = v 
            if v.name == dest.name:
                destA = v

        for e in copie_graphe.edges: 
            e.weight = (e.weight[0], 'A')

        liste_distance = copie_graphe.DijkstraMultiObjBidirectionnel(oriA, destA)

        if not liste_distance: 
            return []
        distance = liste_distance[0][1][0] 

        # Appliquer Dijkstra MO avec la distance à ne pas dépasser 
        distance_max: float = (1 + seuil/100) * distance 

        return self.DijkstraMultiObjBidirectionnel(source, dest, distance_max)

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


### CONDITION D'ARRET DANS DIJKSTRA MO BD ###

def stop(T, Lres):
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
    
    TminF = TF[0][2].vector.copy() # vecteur de coûts du premier élément
    nb_dim: int = len(TminF) # dimension du vecteur de coûts i.e. nombre de catégories
    for i in range(1, len(TF)):
        for j in range(nb_dim):
            if TminF[j] > TF[i][2].vector[j]:
                TminF[j] = TF[i][2].vector[j]

    # Backward : construire le vecteur de cout minimum pour chaque objectif a partir des vecteurs de TB
    TminB = TB[0][2].vector.copy()
    for i in range(1, len(TB)): 
        for j in range(nb_dim):
            if TminB[j] > TB[i][2].vector[j]:
                TminB[j] = TB[i][2].vector[j]

    Tmin = [TminB[i] + TminF[i] for i in range(nb_dim)]
    labTmin = Label(None, Tmin, None, -1)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]
    return labTmin.dominated_by_list(Lres_labels) 

### FONCTIONS LIEES AUX CHEMINS DANS DIJKSTRA MO BD ###

def reconstruireChemin(chemin):
    '''
    Retourne un chemin reconstruit, i.e.
    ([liste des sommets du chemin de source à destination], vecteur de coûts total)
    
    :param chemin: (label depuis source, label depuis dest, vecteur de coûts total)
    '''
    depuis_ori, depuis_dest, vect = chemin
    
    # Chemin jusqu'à origine
    chemin_ori = []
    sommet = depuis_ori.vertex
    label_prec = depuis_ori.prev_label
    sommet_union = sommet.name
    while label_prec != None:
        sommet = label_prec.vertex
        label_prec = label_prec.prev_label
        chemin_ori = [sommet.name] + chemin_ori

    # Chemin jusqu'à destination
    chemin_dest = []
    label_prec = depuis_dest.prev_label
    while label_prec != None:
        sommet = label_prec.vertex
        label_prec = label_prec.prev_label
        chemin_dest = chemin_dest + [sommet.name]
        
    return (chemin_ori + [sommet_union] + chemin_dest, vect)


def addResults(path, liste_res) -> None:
    """
    Reconstruit le chemin path et l'ajoute à liste_res s'il n'est pas dominé par un chemin de liste_res.

    :param path: chemin à ajouter (depuis_ori, depuis_dest, vecteur cout)
    :param liste_res: liste des chemins (liste_sommets, vecteur cout) déjà découverts
    """
    liste_sommets, vec = reconstruireChemin(path)
    a_retirer = []

    # Pour tout chemin r dans Lres
    for r in liste_res:
        liste_sommetsTemp, vecTemp = r

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

    liste_res.append((liste_sommets, vec))  


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
