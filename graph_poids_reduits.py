import numpy as np, math, heapq, json, pandas, time, csv
from typing import Tuple, List, Dict

class Vertex:

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.label_list: List[List] = [[],[]] # liste des listes forward et backward des labels (A* MO bi-directionnel)

    def __eq__(self, vertexPrime) -> bool:
        '''
        Retourne True si le sommet self est le sommet vertexPrime (comparaison des noms),
        False sinon.

        :param vertexPrime:
        '''
        if not isinstance(vertexPrime, Vertex):
            return False
        return self.name == vertexPrime.name  
    
    def __hash__(self) -> int:
        ''' 
        Retourne une valeur entière associé au nom du sommet.
        '''
        return hash(self.name)

    
    def coordonnees(self) -> Tuple[int, int]: 
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

    def coordonnees2(self) -> Tuple[float, float]:
        '''
        Retourne le tuple de coordonnées associées au sommet. 
        (PAI2D : nom d'un sommet (latitude, longitude))
        '''
        lat, lon = self.name.split(",")
        return float(lat), float(lon)

    def addLabel(self, label, direction: int, weight_vertices: List[List[float]] = None) -> None:
        '''
        Ajoute un label à la liste des labels du sommet et supprime les anciens labels qui sont dominés.
        # (A* MO bi-directionnel)
        
        :param label: label (noeud courant, vecteur de coût, label précédent, code)
        :param direction: direction (forward : 0, backward : 1)
        :param weight_vertices: liste des sommets extrêmes du polyèdre
        '''
        vector: Vertex = label.vector
        old_vectors = np.array([l.vector for l in self.label_list[direction]]) 

        # Vérifier si la liste des vecteurs du sommet est vide ou non
        if old_vectors.size != 0:

            if weight_vertices is None: 
                dominated = dominates_in_list(vector, old_vectors)
            else:
                dominated = dominates_in_list_interval(vector, old_vectors, weight_vertices)

            # Ajouter seulement les labels non dominés par le nouveau label
            self.label_list[direction] = [label] + [
                l for l, d in zip(self.label_list[direction], dominated) if not d
            ]

        else:
            self.label_list[direction] = [label]


class Edge:

    def __init__(self, v1: Vertex, v2: Vertex, dist: float, classe: str) -> None:
        self.vertices: Tuple[Vertex, Vertex] = (v1, v2)
        self.weight: Tuple[float, str] = (dist, classe)  # exemple : (5,"B") -> la classe doit être une lettre majuscule autorisée (en fonction de nbClasses)


class Label:

    def __init__(self, vertex: Vertex, cost_vector: List[float], previous_label, code: int):
        self.vertex: Vertex = vertex
        self.vector: List[float] = cost_vector
        self.prev_label = previous_label
        self.code: int = code

    def labelToString(self) -> str:
        '''
        Retourne le label transformé en chaîne de caractères.
        
        :param label: label (noeud courant, vecteur de coût, label précédent)
        '''
        res : str = "(" + self.vertex.name + "," + str(self.vector) + ", "
        if self.prev_label == None:
            return res + "None)"
        return res + self.labelToString(self.prev_label) + ")" 

    def succ_label(self, new_vertex: Vertex, edge: Edge, nbClasses: int, code: int):
        '''
        Crée un nouveau label qui succède à self (vecteur de coût mis à jour avec edge).
        
        :param label: label (noeud courant, vecteur de coût, label précédent, code)
        :param edge: arc
        :param nbClasses: dimension du vecteur de coût
        :param code: code du label
        '''
        vector: Vertex = self.vector
        classe: int = ord(edge.weight[1]) - 65 # 65 = ord('A') | 0 = 'A', 1 = 'B', etc.
        dist: float = edge.weight[0]
        new_vector = list(vector)
        # On plafonne à nbClasses-1 si la classe de l'arc dépasse le nombre de critères
        classe = min(classe, nbClasses - 1)
        new_vector[classe] += dist # pas de distance de niveau de sécurité cumulé, distance réelle pour chaque niveau
        return Label(new_vertex, new_vector, self, code)
    
    def combine(self, labelListe, direction: int, dist_max: float = math.inf) -> List: 
        '''
        Retourne une liste des chemins combinés entre un label et une liste de labels.
        Un chemin : (label depuis source, label depuis destination, vecteur de coût total)
        ou etiquette = label ou les deux procédures se rejoignent

        :param labelListe: liste de labels dans la direction opposée
        :param direction: direction (forward : 0, backward : 1)
        :param dist_max: distance maximale à ne pas dépasser pour un chemin (distance totale pour le premier critère)
        '''
        res = []
        vec = self.vector
        nb_dim = len(vec)

        # Déterminer les vecteurs de coût total
        for label in labelListe:
            vec_suivant = label.vector
            if total_dist(vec) + total_dist(vec_suivant) <= dist_max:
                if direction == 0:
                    res.append((self, label, [vec[j] + vec_suivant[j] for j in range(nb_dim)]))
                else:
                    res.append((label, self, [vec[j] + vec_suivant[j] for j in range(nb_dim)]))

        return res
    
    def dominated_by_list(self, labelListe: List, weight_vertices = None) -> bool:
        '''
        Retourne True si le vecteur de coût du label est dominé par au moins un autre d'un label de labelListe,
        False sinon.

        :param label: label (noeud courant, vecteur de coût, label précédent, code)
        :param labelListe: liste de labels
        :param weight_vertices: liste des sommets extrêmes du polyèdre
        '''
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
        self._index: dict[str, Vertex] = {}  # dictionnaire nom : sommet
        self.omega: List[float] = [1 for _ in range(nbClasses - 1)] 
        self.gamma: List[float] = [0  for _ in range(nbClasses - 1)] 
        self.weight_vertices = self._compute_weight_vertices() if self.omega else None # sommets extrêmes du polyèdre (resteindre le problème)

    def _compute_weight_vertices(self) -> List[List[float]]:
        '''
        Calcule les 2^(n-1) points extrêmes du polyèdre des poids
        via Proposition 3 (Ahn & Park 2014).
        
        Pour chaque combinaison de bornes (omega_j ou gamma_j),
        on résout : w_j = r_j * w_{j+1}, sum(w) = 1.
        '''
        n: int = self.nbClasses
        vertices: List[List[float]] = []

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
            total = 1 #sum(w)
            w = [wi / total for wi in w]
            vertices.append(w)

        return vertices
    
    # Copie du graphe

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
    
    # Fonctions sur la taille du graphe

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
    
    # Fonctions d'ajout de sommets ou d'arcs

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
    
    def add_edge(self, namev1: str, namev2: str, dist: float, classe: str) -> None:
        '''
        Ajoute une arête au graphe.
        
        :param namev1: nom d'un vecteur qui *existe* dans le graphe
        :param namev2: nom d'un vecteur qui *existe* dans le graphe
        :param dist: poids de l'arc
        :param classe: classe de l'arc
        '''
        # Récupération des vertex dans le graphe
        vertex1 = self._index.get(namev1) # next(v for v in self.vertices if v.name == namev1)
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

    # Recherche d'un sommet 

    def search_vertex(self, name : str) -> Vertex | None:
        '''
        Renvoie le sommet dans le graphe courant de nom "name" s'il existe, sinon renvoie None

        :param name: nom du sommet à trouver
        '''
        return self._index.get(name)

    # Fonctions de suppression de sommets ou de listes de labels

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
    
    def reset_labels(self) -> None:
        '''
        Réinitialise les listes de labels forward et backward
        de tous les sommets du graphe.
        '''
        for vertex in self.vertices:
            vertex.label_list = [[], []]
        
    # Récupération des voisins d'un sommet du graphe

    def getNeighbors(self, vertex: Vertex, dir: int) -> List[Edge]:
        '''
        Renvoie la liste des arcs de vertex.
        
        :param vertex: sommet courant
        :param dir: direction de parcours (0: successeurs, 1: predecesseurs)
        '''
        return [e for e in self.adj[dir][vertex]]

    # Fonctions sur le degré des sommets du graphe 

    def degres(self, dir: int) -> List:
        '''
        Renvoie un tableau contenant les tuples (sommet s, degré(s)) pour les sommets du graphe.

        :param dir: direction de parcours (0: successeurs, 1: predecesseurs)
        '''
        return [(v, len(neighbors)) for (v, neighbors) in self.adj[dir].items()]

    def max_degre(self, dir: int) -> str:
        '''
        Renvoie le nom du (premier) sommet de degré maximal du graphe.

        :param dir: direction de parcours (0: successeurs, 1: predecesseurs)
        '''
        deg = self.degres(dir)
        m = max(d[1] for d in deg)
        return next(x[0].name for x in deg if x[1] == m)
    
    # Distance entre deux sommets qui représentent des coordonnées
    
    def distance_a_vol_d_oiseau(self, v1 : Vertex,  v2 : Vertex) -> float:
        '''
        Renvoie la distance euclidienne entre deux sommets
        dont les noms sont des coordonnées "lat, lon".

        :param v1: sommet 1
        :param v2: sommet 2
        '''

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
    
    # Fonctions d'affichage du graphe

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

    # Sauvegarde d'un graphe 

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

    # Algorithmes 

    def landmarks_distance_computing(self, landmark_names: list[str], nom_fichier: str = "") -> None:
        ''' 
        Produit un fichier csv "distances_landmarks_{nom_fichier}.csv" contenant les distances à tous les landmarks.
        
        :param landmark_names: liste des noms des landmarks 
        :param nom_fichier: suffixe du nom du fichier
        '''
        # Copie du graphe pour tout passer en mono-objectif
        copie_graphe: Graph = self.copie()
        for e in copie_graphe.edges: 
            e.weight = (e.weight[0], 'A')

        # Création des dictionnaires contenant les distances des landmarks aux sommets (1) et inversement (0)
        dist = [dict(),dict()]
        for vert in copie_graphe.vertices:
            dist[1][vert] = []
            dist[0][vert] = []

        # Recherche des sommets associés aux noms des landmarks dans le graphe
        landmarks = [copie_graphe._index[w] for w in landmark_names]
        
        # Calcul des distances des landmarks jusqu'aux sommets
        print("Landmark -> points")
        for lm in landmarks:
            start = time.time()
            dico = copie_graphe.Dijkstra(lm)       
            for p,v in dico.items():
                dist[1][p].append(v)         
            print(f"Fin {lm.name} en {time.time()-start:.2f} s")
        
         # Calcul des distances des sommets jusqu'aux landmarks
        print("Point -> landmarks")
        for lm in landmarks:
            start = time.time()
            dico = copie_graphe.Dijkstra(lm, d=1)       
            for p,v in dico.items():
                dist[0][p].append(v)         
            print(f"Fin {lm.name} en {time.time()-start:.2f} s")
        
        # Ecriture des distances dans un fichier csv 
        n: int = len(landmark_names)
        with open(f"distances_landmarks_{nom_fichier}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["vertex_name"] + [f"0:L{i}" for i in range(n)] + [f"1:L{i}" for i in range(n)])
            dist0 = dist[0]
            dist1 = dist[1]
            for v in dist0:
                writer.writerow([v.name] + dist0[v] + dist1[v])

    def Dijkstra(self, source: Vertex, d: int = 0):
        ''' 
        Applique l'algorithme de Dijkstra.
        Calcul des distances entre un sommet source et tous les sommets.
        
        :param source: sommet source à partir duquel on calcule les distances
        :param d: direction (forward : 0, backward : 1)
        '''
        # Initialisation du dictionnaire des distances
        distances = {v: float("inf") for v in self.vertices}
        distances[source] = 0 

        # Création d'une file de priorité
        code = 0 # sert à choisir un sommet quand plusieurs ont la même distance à l'origine
        file = [(0, code, source)] # file
        heapq.heapify(file)

        # Création d'un ensemble de sommets déjà visités
        visited = set()

        while file:  
            distance, _, sommet = heapq.heappop(file)

            if sommet in visited:
                continue 

            visited.add(sommet)

            for e in self.getNeighbors(sommet, d):
                voisin = e.vertices[1-d]
                tentative_distance = distance + e.weight[0]
                if tentative_distance < distances[voisin]:
                    distances[voisin] = tentative_distance
                    code += 1
                    heapq.heappush(file, (tentative_distance, code, voisin))
        
        return distances

    def AStarMultiObjBidirectionnel(self, source: Vertex, dest: Vertex, heuris, nb_lm: int, 
                                    condition_darret, dist_max: float = math.inf,
                                    seuil: float = math.inf, verbose: bool = False) -> List:
        
        '''
        Applique l'algorithme d'A* multi-objectif bi-directionnel
        pour récupérer l'ensemble des chemins Pareto-optimaux 
        allant du sommet source au sommet dest.
        Utilisation des heuristiques : distance à vol d'oiseau + distance avec landmarks.
        
        :param source: sommet source
        :param dest: sommet destination
        :param heuris: distances entre tous les points et les landmarks utilisées pour réduire l'exploration
        :param nb_lm: nombre de landmarks
        :param condition_darret: fonction de condition d'arret, renvoie True (s'il faut arrêter) ou False
        :param dist_max: distance maximale à ne pas depasser (distance totale)
        :param seuil: seuil pour borner les longueurs des sous-chemins passant par des sommets de chemin_opt
        :param verbose: affiche des commentaires (True) ou rien (False)
        '''

        T: List[List[Label]] = [[], []]  # tas des labels temporaires (pour les deux directions)
        Lres: List = []  # liste des chemins Pareto-optimaux
        wv = self.weight_vertices  # None si mode classique
        if verbose:
            print("Sommets extrêmes du polyèdre des préférences de l'utilisateur.ice :", wv)

        def heap_key(vector): 
            '''
            Retourne la valeur (ou le vecteur) associé à un vecteur pour le tri du tas. 

            :param vector: vecteur à ajouter dans le tas
            '''
            if wv is None:
                return vector
            return float(np.min(np.array(wv) @ np.array(vector)))

        # Ajout du label de source à T[0]
        code: int = 0 # compteur du nombre de labels créés
        sourceLabel: Label = Label(source, [0] * self.nbClasses, None, code)
        code += 1
        source.addLabel(sourceLabel, 0, wv)
        heapq.heappush(T[0], (heap_key(sourceLabel.vector), sourceLabel.code, sourceLabel))

        # Ajout du label de destination à T[1]
        destLabel: Label = Label(dest, [0] * self.nbClasses, None, code)
        code += 1
        dest.addLabel(destLabel, 1, wv)
        heapq.heappush(T[1], (heap_key(destLabel.vector), destLabel.code, destLabel))

        # Recuperation des heuristiques pour la source et la destination
        ds, dt = calcul_st(heuris, source, dest)    
        d_s_L = ds[:nb_lm]   # distances de s vers landmarks 
        d_L_s = ds[nb_lm:]   # distances des landmarks vers s 
        d_t_L = dt[:nb_lm]   # distances de t vers landmarks 
        d_L_t = dt[nb_lm:]   # distances des landmarks vers t 
    

        d: int = 1 # direction (0 : forward, 1 : backward)
        nbElagages: int = 0
        nbLabelsExplores: int = 0

        while not condition_darret(T, Lres, self, dest):
            d = 1 - d
            if verbose:
                afficher_T(T, d)

            nbLabelsExplores += 1

            # Récupération d'un label dans T[d]
            _, _, label = heapq.heappop(T[d])

            # Récupération du sommet courant et de ses arcs (entrants ou sortants en fonction de la direction)
            owner: Vertex = label.vertex
            if verbose:
                print("Sommet courant :", owner.name, " direction :", d, "code :", label.code)
            neighbors: List[Edge] = self.getNeighbors(owner, d)

            # Parcours des voisins
            e: Edge
            for e in neighbors:
                voisin: Vertex = e.vertices[1 - d]
                newLabel = label.succ_label(voisin, e, self.nbClasses, code)
                if verbose:
                    print(f"\t voisin = {newLabel.vertex.name}, {newLabel.vector}, code = {newLabel.code}")
                code += 1

                # Élagage par distance à vol d'oiseau et/ou distance avec landmarks
                obj = dest if d == 0 else source
                distance_oiseau = self.distance_a_vol_d_oiseau(voisin, obj)
                if dist_max < 2000:
                    distance_landmarks = 0
                else:
                    ligne_sommet = find_row(heuris, voisin.name)
                    distance_landmarks = calcul_sommet_landmarks(ligne_sommet, nb_lm, d_s_L, d_L_s, d_t_L, d_L_t, d)
                
                dist_restante = max(distance_oiseau, distance_landmarks)

                if total_dist(newLabel.vector) + dist_restante > dist_max * (100 + seuil) / 100: 
                    nbElagages += 1
                    continue

                # Test de dominance (mode classique ou avec intervalles selon wv)
                if not newLabel.dominated_by_list(voisin.label_list[d], wv):
                    voisin.addLabel(newLabel, d, wv)
                    heapq.heappush(T[d], (heap_key(newLabel.vector), newLabel.code, newLabel))

                    # Combinaison avec labels dans la direction opposée
                    if voisin.label_list[1 - d]:
                        
                        for c in newLabel.combine(voisin.label_list[1 - d], d, dist_max):
                            addResults(c, Lres, wv)  # wv passé à addResults
            
                if verbose:
                    affiche_results(Lres)

            if verbose:
                print("---")

        # Réinitialisation des listes de labels des sommets
        self.reset_labels()

        if verbose:
            print("Nombre d'élagages :", nbElagages)
            print("Nombre de labels explorés :", nbLabelsExplores)
            print("Nombre de solutions :", len(Lres[0]))
        return Lres

    def AStarMultiObjBidirectionnelSeuil(self, source: Vertex, dest: Vertex, heuris, nb_lm: int, condition_darret, seuil: float, verbose: bool = False) -> List: 
        '''
        Applique l'algorithme d'A* multi-objectif bi-directionnel
        pour récupérer l'ensemble des chemins Pareto-optimaux 
        allant du sommet source au sommet dest
        avec la longueur d'un chemin qui ne dépasse pas 100 + seuil % du chemin optimal (mono-objectif).
        
        :param source: sommet source
        :param dest: sommet destination
        :param heuris: distances entre tous les points et les landmarks utilisées pour réduire l'exploration
        :param nb_lm: nombre de landmarks
        :param condition_darret: fonction de condition d'arret, renvoie True (s'il faut arreter) ou False
        :param seuil: pourcentages supplémentaires du chemin optimal 
        :param verbose: affiche des commentaires (True) ou rien (False)
        '''
        # Appliquer A* en version mono-objectif (distance totale) pour récupérer le chemin de longueur minimale
        copie_graphe: Graph = self.copie()

        oriA = copie_graphe._index[source.name]
        destA = copie_graphe._index[dest.name]

        for e in copie_graphe.edges: 
            e.weight = (e.weight[0], 'A')

        print("------------ APPEL MONO ---------------")
        mono = copie_graphe.AStarMultiObjBidirectionnel(oriA, destA, heuris, nb_lm, condition_darret)

        if not mono: 
            return []
        
        distance = total_dist(mono[0][1])
        if verbose:
            print("Chemin optimal :", mono[0][0])

        # Appliquer A* MO avec la distance à ne pas dépasser 
        distance_max: float = (1 + seuil/100) * distance 
        print("------------ APPEL BI ---------------")
        return mono[0], self.AStarMultiObjBidirectionnel(source, dest, heuris, nb_lm, condition_darret, distance_max, seuil, verbose)


def load_from_json(filename: str) -> Graph:
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

def total_dist(vector) -> float:
    '''
    Retourne la distance totale d'un vecteur de coût.
    (-> remplace l'ancien vect[0])

    :param vector: vecteur de coût
    '''
    return sum(vector)


### PARETO DOMINANCE ### -> obsolète

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


### PARETO DOMINANCE DANS LE POLYEDRE ###

def dominates_interval(cA: List[float], cB: List[float],
                        weight_vertices: List[List[float]]) -> bool:
    '''
    Pairwise dominance (Section 3, Ahn & Park) :
    cA domine cB ssi w·cA <= w·cB pour tout point extrême w,
    avec inégalité stricte pour au moins un.
    '''
    cA, cB = np.array(cA), np.array(cB)
    W = np.array(weight_vertices)          # shape (2^(n-1), n)
    diff = W @ (cA - cB)                   # w·cA - w·cB pour chaque sommet
    return bool(np.all(diff <= 0) and np.any(diff < 0))


def dominates_in_list_interval(v: List[float], liste_v: np.ndarray,
                                weight_vertices: List[List[float]]) -> np.ndarray:
    '''
    Pour chaque vecteur de liste_v, True si v le domine (au sens interval).
    Retourne un array de booléens de taille len(liste_v).
    '''
    W = np.array(weight_vertices)          # (K, n)
    sv = W @ np.array(v)                   # (K,)
    SL = W @ liste_v.T                     # (K, m)
    diff = sv[:, None] - SL                # w·v - w·c  pour chaque (sommet, chemin)
    return np.all(diff <= 0, axis=0) & np.any(diff < 0, axis=0)


def dominated_in_list_interval(v: List[float], liste_v: np.ndarray,
                                weight_vertices: List[List[float]]) -> np.ndarray:
    '''
    Pour chaque vecteur de liste_v, True si v est dominé par lui.
    '''
    W = np.array(weight_vertices)
    sv = W @ np.array(v)
    SL = W @ liste_v.T
    diff = SL - sv[:, None]                # w·c - w·v
    return np.all(diff <= 0, axis=0) & np.any(diff < 0, axis=0)


### CONDITION D'ARRET DANS A* MO BD ###

def stop(T: List[List[Label]], Lres: List[Tuple[List[str], List[float]]], graph = None, dest = None) -> bool:
    '''
    Arrête la recherche si Tmin (vecteur idéal) 
    est déjà dominé par au moins un chemin de Lres.
    (A* MO BD : boucle à arrêter si True)
    
    :param T: liste des labels temporaires
    :param Lres: liste des chemins Pareto-optimaux
    '''
    # Listes des etiquettes temporaires
    TF, TB = T[0], T[1]
    if not TF or not TB:
        return True
    if not Lres:
        return False

    TminF = np.min(np.array([lbl.vector for _, _, lbl in TF]), axis=0)  # ← lbl.vector
    TminB = np.min(np.array([lbl.vector for _, _, lbl in TB]), axis=0)  # ← lbl.vector
    Tmin = list(TminF + TminB)
    labTmin = Label(None, Tmin, None, -1)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]
    return labTmin.dominated_by_list(Lres_labels)


def stop2(T: List[List[Label]], Lres: List[Tuple[List[str], List[float]]], graph = None, dest = None) -> bool:
    ''' 
    Arrête la recherche si toutes les combinaisons possibles
    (TminB + label foward)
    sont déjà dominées par un chemin de Lres.
    TminB est le vecteur idéal de la recherche arrière.
    (A* MO BD : boucle à arrêter si True)
    
    :param T: liste des labels temporaires
    :param Lres: liste des chemins Pareto-optimaux
    '''
    TF, TB = T[0], T[1]
    if not TF or not TB:
        return True
    if not Lres:
        return False

    TminB = np.min(np.array([lbl.vector for _, _, lbl in TB]), axis=0)  # ← lbl.vector
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]

    for (_, _, lbl) in TF:  # ← lbl.vector
        combined = Label(None, list(np.array(lbl.vector) + TminB), None, -1)
        if not combined.dominated_by_list(Lres_labels):
            return False
    return True

def stop3(T: List[List[Label]], Lres: List[Tuple[List[str], List[float]]], graph, dest) -> bool: 
    ''' 
    Arrête la recherche si toutes les combinaisons possibles
    (TminB + label forward) - en prenant en compte la distance à vol d'oiseau - 
    sont déjà dominées par un chemin de Lres.
    TminB est le vecteur idéal de la recherche arrière.
    (A* MO BD : boucle à arrêter si True)

    :param T: liste des labels temporaires
    :param Lres: liste des chemins Pareto-optimaux
    :param graph: graphe avec distance_a_vol_d_oiseau(u, dest)
    :param dest: sommet destination
    '''
    TF, TB = T[0], T[1]
    if not TF or not TB:
        return True
    if not Lres:
        return False

    wv = graph.weight_vertices
    TminB = np.min(np.array([lbl.vector for _, _, lbl in TB]), axis=0)
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]

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


def stop4(T: List[List[Label]], Lres: List[Tuple[List[str], List[float]]], graph, dest) -> bool:
    '''
    Arrête la recherche si toutes les combinaisons possibles
    (label forward + borne vol d'oiseau + label backward)
    sont déjà dominées par un chemin de Lres.
    (A* MO BD : boucle à arrêter si True)

    :param T: liste des labels temporaires
    :param Lres: liste des chemins Pareto-optimaux
    :param graph: graphe avec distance_a_vol_d_oiseau(u, dest)
    :param dest: sommet destination
    '''
    TF, TB = T[0], T[1]
    if not TF or not TB:
        return True
    if not Lres:
        return False

    wv = graph.weight_vertices 
    Lres_labels = [Label(None, vect, None, -1) for (_, vect) in Lres]

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


### FONCTIONS LIEES AUX CHEMINS DANS A* MO BD ###

def reconstruireChemin(chemin: Tuple[Label, Label, List[float]]) -> Tuple[List[str], List[float]]:
    '''
    Retourne un chemin reconstruit, i.e.
    (liste des noms des sommets du chemin de source à destination, vecteur de coût total)
    
    :param chemin: (label du chemin depuis source, label du chemin depuis dest, vecteur de coût total)
    '''
    depuis_ori, depuis_dest, vect = chemin

    chemin_ori = []
    sommet = depuis_ori.vertex
    label_prec = depuis_ori.prev_label
    sommet_union = sommet.name
    while label_prec is not None:
        sommet = label_prec.vertex
        chemin_ori = [sommet.name] + chemin_ori
        label_prec = label_prec.prev_label

    chemin_dest = []
    label_prec = depuis_dest.prev_label
    while label_prec is not None:
        sommet = label_prec.vertex
        chemin_dest = chemin_dest + [sommet.name]
        label_prec = label_prec.prev_label

    liste_sommets = chemin_ori + [sommet_union] + chemin_dest

    return (liste_sommets, vect)


def inclusion_avec_ratio(list1: List, list2: List) -> int:
    ''' 
    Retourne le ratio du nombre d'éléments sur le nombre d'éléments 
    de la plus petite des deux listes.

    :param list1: liste 1
    :param list2: liste 2
    '''
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2) / min(len(set1), len(set2))


def addResults(chemin: Tuple[Label, Label, List[float]], liste_res: List[Tuple[List[str], List[float]]], weight_vertices = None) -> None:
    """
    Reconstruit un chemin et l'ajoute à liste_res s'il n'est pas dominé par un chemin de liste_res.

    :param chemin: chemin à ajouter (label du chemin depuis la source, label du chemin depuis la destination, vecteur de coût)
    :param liste_res: liste des chemins (liste_sommets, vecteur de coût) déjà découverts
    """
    liste_sommets, vec = reconstruireChemin(chemin)
    a_retirer = []

    for r in liste_res:
        liste_sommetsTemp, vecTemp = r
        
        if inclusion_avec_ratio(liste_sommets, liste_sommetsTemp) >= 0.9 or np.all(np.isclose(vecTemp, vec)) == True: 
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

    liste_res.append((liste_sommets, vec))


### FONCTIONS LIEES AUX LANDMARKS DANS A* MO BD ###

def read_landmarks_file(csvname: str):
    '''
    Retourne le dataframe d'un fichier csv.
    :param csvname: nom du fichier csv
    '''
    fichier_csv = pandas.read_csv(csvname)
    return pandas.DataFrame(fichier_csv)


def find_row(df, vname: str):
    ''' 
    Retourne la ligne associée (version numpy) à un sommet dans un dataframe.
    :param df: dataframe des distances 
    :vname: nom du sommet
    '''
    return df.loc[vname].values


def calcul_st(df, source: Vertex, dest: Vertex):
    ''' 
    Retourne des numpy arrays des distances (avant et arriere) des sommets source et destination aux landmarks.
    :param df: dataframe des distances
    :param source: sommet de départ
    :param dest: sommet d'arrivée
    :param nb_lm: nombre de landmarks 
    '''
    rs = df.loc[source.name].values
    rt = df.loc[dest.name].values
    return rs, rt

def calcul_sommet_landmarks(v, nb_lm: int, d_s_L, d_L_s, d_t_L, d_L_t, direction: int) -> float:
    ''' 
    Retourne la borne inférieure avec les landmarks pour le sommet v.

    Pour tout landmark L, 
    Forward : 
        d(v->t) >= d(L->t) - d(L->v)
        d(v->t) >= d(v->L) - d(t->L)
    Backward : 
        d(v->s)inv >= d(L->s)inv - d(L->v)inv, soit d(s->v) >= d(s->L) - d(v->L)
        d(v->s)inv >= d(v->L)inv - d(s->L)inv, soit d(s->v) >= d(L->v) - d(L->s)
    
    On prend le max de toutes ces distances. 

    :param v: array des distances entre un sommet donné et les landmarks (forward, backward)
    :param nb_lm: nombre de landmarks s
    :param direction: direction (forward : 0, backward : 1)
    '''
    d_v_L = v[:nb_lm]    # distances de v vers landmarks 
    d_L_v = v[nb_lm:]    # distances des landmarks vers v 

    diff1 = d_L_t - d_L_v if direction == 0 else d_s_L - d_v_L 
    diff2 = d_v_L - d_t_L if direction == 0 else d_L_v - d_L_s  
    
    return np.nanmax(np.concatenate([diff1,diff2]))

### AFFICHER LES RESULTATS DANS A* MO BD ###

def affiche_results(lres: List[Tuple[List[str], List[float]]]) -> None:
    ''' 
    Affiche les chemins contenus dans lres de la forme suivante : 
    Chemin i = V1 -(2)-> V4 -(5)-> V7 | secu <= A : 7, secu <= B : 5, secu <= C : 2

    :param lres: liste des résultats telle que renvoyée par AStarMultiObjBidirectionnel
    '''
    j: int = 1
        
    for j, (chemin, vect) in enumerate(lres, 1):
        res = f"Chemin {j} : "
        for i in range(len(chemin) - 1):
            res += chemin[i] + " -> "
        res += chemin[-1] + " | "
        res += f"distance totale : {total_dist(vect)}, "
        for k in range(len(vect)):
            res += f"km classe {chr(ord('A') + k)} : {vect[k]}, "
        print(res[:-2])

def afficher_T(T: List[List[Label]], d: int) -> None:
    ''' 
    Affiche la liste des labels temporaires d'une direction donnée.

    :param T: liste des labels temporaires
    :param d: direction (0 : forward, 1 : backward)
    '''
    res = f"T[{d}] =\n"
    for t in T[d]:
        res += f"\t{t[2].vertex.name} : {[int(x) for x in t[2].vector]}, code = {t[2].code}\n"
    print(res)

### GENERATION DE GRAPHES ALEATOIRES ###
    
def generate_random_graph(name: str, nbVertex: int, probaEdge: float, nbClasses: int) -> Graph:
    '''
    Génère un graphe aléatoire.
    
    :param name: nom du graphe
    :param nbVertex: nombre de sommets
    :param probaEdge: probabilité d'ajouter un arc pour chaque paire de sommets (distance entre 1 et 50)
    :param nbClasses: nombre de niveaux de sécurité / dimensions du vecteur de coût
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
