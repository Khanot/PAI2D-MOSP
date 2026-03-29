from flask import Flask, jsonify, render_template, request
from graph_commente1 import *
import json
import math

app = Flask(__name__)

# Chargement du graphe au démarrage
G = load_from_json("graphes/GrapheParis2Classes.json")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/vertices")
def get_vertices():
    vertices = []
    for v in G.vertices:
        lat, lon = map(float, v.name.split(","))
        vertices.append({"lat": lat, "lon": lon, "name": v.name})
    return jsonify(vertices)

@app.route("/itineraire", methods=["POST"])
def itineraire():
    data = request.json
    source = G.search_vertex(data["source"])
    dest = G.search_vertex(data["dest"])

    if source is None or dest is None:
        return jsonify({"error": "Sommet introuvable"}), 404

    # Réinitialiser les labels (important entre deux appels)
    for v in G.vertices:
        v.label_list = [[], []]

    resultats = G.DijkstraMultiObjBidirectionnelSeuil(source, dest, stop3, seuil=data.get("seuil", 20))

    if not resultats:
        return jsonify({"error": "Aucun chemin trouvé"}), 404

    # Retourner les chemins Pareto-optimaux
    chemins = []
    for (path, vect, _, _) in resultats:
        coords = []
        for v in path:
            name = v.name if hasattr(v, "name") else v  # gère les deux cas
            lat, lon = map(float, name.split(","))
            coords.append({"lat": lat, "lon": lon})
            chemins.append({"coords": coords, "vecteur": vect, "distance_km": round(vect[0] / 1000, 2)})

    return jsonify(chemins)

@app.route("/nearest")
def nearest():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    best = None
    best_dist = math.inf
    for v in G.vertices:
        vlat, vlon = map(float, v.name.split(","))
        d = (vlat - lat)**2 + (vlon - lon)**2  # distance euclidienne suffit ici
        if d < best_dist:
            best_dist = d
            best = v

    if best is None:
        return jsonify({"error": "Aucun sommet trouvé"}), 404

    blat, blon = map(float, best.name.split(","))
    return jsonify({"name": best.name, "lat": blat, "lon": blon})

if __name__ == "__main__":
    app.run(debug=True)