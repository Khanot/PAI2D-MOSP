const map = L.map("map").setView([48.8566, 2.3522], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap"
}).addTo(map);

let source = null, dest = null;
let srcMarker = null, dstMarker = null;
let routeLayers = [];
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

    const seuil = parseInt(document.getElementById("seuil").value);
    fetch("/itineraire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: source.name, dest: dest.name, seuil })
    })
    .then(r => r.json())
    .then(chemins => {
        if (chemins.error) {
            document.getElementById("status").textContent = "Erreur : " + chemins.error;
            return;
        }
        console.log(chemins[0]);
        chemins.forEach((chemin, i) => {
            const color = i === 0 ? "blue" : "orange";
            const coords = chemin.coords.map(c => [c.lat, c.lon]);
            const line = L.polyline(coords, { color, weight: 4, opacity: 0.8 })
                .bindTooltip(`${chemin.distance_km} km`, { sticky: true })
                .addTo(map);
            routeLayers.push(line);
        });
        map.fitBounds(routeLayers[0].getBounds());
        document.getElementById("status").textContent =
            `${chemins.length} chemin(s) Pareto-opt trouvé(s).`;
            
    // Affiche la distance du chemin proposé 
    map.fitBounds(routeLayers[0].getBounds());
    document.getElementById("status").textContent =
        `${chemins.length} chemin(s) Pareto-opt trouvé(s).`;
    document.getElementById("distance-info").textContent =
        `Chemin proposé : ${chemins[0].distance_km} km`;
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
    clearRoutes();
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

// Initialise les deux champs
setupSearch("search-src", "suggestions-src", "source");
setupSearch("search-dst", "suggestions-dst", "dest");