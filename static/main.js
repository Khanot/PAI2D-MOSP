const map = L.map("map").setView([48.8566, 2.3522], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap"
}).addTo(map);

let source = null, dest = null;
let srcMarker = null, dstMarker = null;
let routeLayers = [];
let monoLayer = [];
let allMarkers = L.layerGroup().addTo(map);
const renderer = L.canvas({ padding: 0.5 });

// Chargement de tous les sommets
fetch("/vertices")
    .then(r => r.json())
    .then(vertices => {
        document.getElementById("status").textContent = `${vertices.length} sommets chargés.`;
        vertices.forEach(v => {
            const circle = L.circleMarker([v.lat, v.lon], {
                renderer,          // ← ajout
                radius: 3, color: "#2196F3", fillOpacity: 0.5, weight: 1
            }).addTo(allMarkers);
            circle.on("click", () => selectVertex(v, circle));
        });
    });

function selectVertex(v, circle) {
    if (!source) {
        source = v;
        srcMarker = L.marker([v.lat, v.lon], {
            icon: L.divIcon({ className: "", html: "🟢", iconSize: [20, 20] })
        }).addTo(map);
        document.getElementById("src-label").textContent = v.name;
        document.getElementById("status").textContent = "Source sélectionnée. Choisissez la destination.";
    } else if (!dest && v.name !== source.name) {
        dest = v;
        dstMarker = L.marker([v.lat, v.lon], {
            icon: L.divIcon({ className: "", html: "🔴", iconSize: [20, 20] })
        }).addTo(map);
        document.getElementById("dst-label").textContent = v.name;
        document.getElementById("status").textContent = "Destination sélectionnée. Lancez le calcul.";
    }
}

function calculer() {
    if (!source || !dest) {
        document.getElementById("status").textContent = "Sélectionnez source et destination.";
        return;
    }
    document.getElementById("status").textContent = "Calcul en cours...";
    clearRoutes();
    if (monoLayer.length > 0) { monoLayer.forEach(l => map.removeLayer(l)); monoLayer = []; }
    const seuil = parseInt(document.getElementById("seuil").value);
    fetch("/itineraire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source.name, dest: dest.name, seuil })
    })
    .then(r => r.json())
    .then(data => {
    if (data.error) {
        document.getElementById("status").textContent = "Erreur : " + data.error;
        return;
    }

    const chemins = data.chemins;
    window.cheminsList = chemins.sort((a, b) => a.distance_km - b.distance_km);

    // Afficher le chemin mono en violet
    if (data.mono && data.mono.coords.length > 0) {
        const monoCoords = data.mono.coords.map(c => [c.lat, c.lon]);
        const monoLine = L.polyline(monoCoords, { color: "purple", weight: 4, opacity: 0.9 })
            .bindTooltip(" Plus court chemin 1 cat", { sticky: true })
            .addTo(map);
        monoLayer.push(monoLine);
    }

    // Construire la liste
    const container = document.getElementById("chemins-list");
    container.innerHTML = `<b>${chemins.length} chemin(s) trouvé(s) :</b>`;
    chemins.forEach((chemin, i) => {
        const item = document.createElement("div");
        item.className = "chemin-item";
        item.id = `chemin-${i}`;
        item.innerHTML = `
            <div class="distance">Chemin ${i + 1} — ${chemin.distance_km} km</div>
            <div class="vecteur">[${chemin.vecteur.map(v => v.toFixed(1)).join(", ")}]</div>
        `;
        item.onclick = () => afficherChemin(i);
        container.appendChild(item);
    });

    document.getElementById("status").textContent = "Sélectionnez un chemin.";
    document.getElementById("distance-info").textContent = "";
});
}

function clearRoutes() {
    routeLayers.forEach(l => map.removeLayer(l));
    routeLayers = [];
}

function resetSelection() {
    source = null; dest = null;
    if (srcMarker) map.removeLayer(srcMarker);
    if (dstMarker) map.removeLayer(dstMarker);
    srcMarker = null; dstMarker = null;
    document.getElementById("src-label").textContent = "non sélectionnée";
    document.getElementById("dst-label").textContent = "non sélectionnée";
    document.getElementById("status").textContent = "Cliquez sur un sommet pour le sélectionner.";
    document.getElementById("distance-info").textContent = "";
    document.getElementById("search-src").value = "";
    document.getElementById("search-dst").value = "";
    document.getElementById("chemins-list").innerHTML = "";
    clearRoutes();
    if (monoLayer) { monoLayer.forEach(l => map.removeLayer(l)); monoLayer = []; }
    document.getElementById("chemins-list").innerHTML = "";
    window.cheminsList = [];
}

function setupSearch(inputId, suggestionsId, role) {
    const input = document.getElementById(inputId);
    const suggestions = document.getElementById(suggestionsId);
    let timer = null;

    input.addEventListener("input", () => {
        clearTimeout(timer);
        const query = input.value.trim();
        if (query.length < 3) { suggestions.style.display = "none"; return; }

        timer = setTimeout(() => {
            fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query + " Paris")}&format=json&limit=5`)
                .then(r => r.json())
                .then(results => {
                    suggestions.innerHTML = "";
                    if (!results.length) { suggestions.style.display = "none"; return; }
                    results.forEach(r => {
                        const item = document.createElement("div");
                        item.className = "suggestion-item";
                        item.textContent = r.display_name;
                        item.onclick = () => selectFromGeocoding(r.lat, r.lon, r.display_name, role, suggestions, input);
                        suggestions.appendChild(item);
                    });
                    suggestions.style.display = "block";
                });
        }, 400); // debounce 400ms
    });

    // Ferme les suggestions si on clique ailleurs
    document.addEventListener("click", e => {
        if (!input.contains(e.target)) suggestions.style.display = "none";
    });
}

function selectFromGeocoding(lat, lon, label, role, suggestions, input) {
    suggestions.style.display = "none";
    input.value = label.split(",")[0];

    // Trouve le sommet le plus proche
    fetch(`/nearest?lat=${lat}&lon=${lon}`)
        .then(r => r.json())
        .then(v => {
            if (v.error) { alert("Aucun sommet proche trouvé"); return; }
            if (role === "source") {
                source = v;
                if (srcMarker) map.removeLayer(srcMarker);
                srcMarker = L.marker([v.lat, v.lon], {
                    icon: L.divIcon({ className: "", html: "🟢", iconSize: [20, 20] })
                }).addTo(map);
                document.getElementById("src-label").textContent = label.split(",")[0];
                map.setView([v.lat, v.lon], 15);
            } else {
                dest = v;
                if (dstMarker) map.removeLayer(dstMarker);
                dstMarker = L.marker([v.lat, v.lon], {
                    icon: L.divIcon({ className: "", html: "🔴", iconSize: [20, 20] })
                }).addTo(map);
                document.getElementById("dst-label").textContent = label.split(",")[0];
                map.setView([v.lat, v.lon], 15);
            }
            document.getElementById("status").textContent = 
                source && dest ? "Prêt à calculer !" : "Sélectionnez l'autre point.";
        });
}

function afficherChemin(index) {
    clearRoutes();

    // Désélectionner tous les items
    document.querySelectorAll(".chemin-item").forEach(el => el.classList.remove("selected"));
    document.getElementById(`chemin-${index}`).classList.add("selected");

    const chemin = window.cheminsList[index];
    const coords = chemin.coords.map(c => [c.lat, c.lon]);
    const line = L.polyline(coords, { color: "blue", weight: 5, opacity: 0.9 })
        .bindTooltip(`⭐ ${chemin.distance_km} km`, { sticky: true })
        .addTo(map);
    routeLayers.push(line);

    map.fitBounds(line.getBounds());
    document.getElementById("distance-info").textContent =
        `Chemin sélectionné : ${chemin.distance_km} km`;
}

// Initialise les deux champs
setupSearch("search-src", "suggestions-src", "source");
setupSearch("search-dst", "suggestions-dst", "dest");