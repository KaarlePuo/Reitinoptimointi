# ============================================
# app.py
# Streamlit UI – Kauppareitin optimointi
# ============================================
# Tämä sovellus:
# - lukee tuotteet ja myymälän grid-mallin CSV:stä
# - huomioi, että hyllyjen läpi ei voi kävellä
# - vertaa baseline-reittiä ja nearest neighbor -optimointia
# - visualisoi reitin käytävillä
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque

# --------------------------------------------
# Streamlit asetukset (TÄMÄ SAA OLLA VAIN KERRAN)
# --------------------------------------------
st.set_page_config(
    page_title="Reitinoptimointi",
    layout="wide"
)

# --------------------------------------------
# Aputoiminnot: data
# --------------------------------------------
@st.cache_data
def load_data():
    """
    Lataa projektin CSV-datat data/raw -kansiosta
    """
    project_root = Path(__file__).resolve().parent
    data_raw = project_root / "data" / "raw"

    products = pd.read_csv(data_raw / "products.csv")
    points = pd.read_csv(data_raw / "store_points.csv")
    grid = pd.read_csv(data_raw / "store_grid.csv")

    return products, points, grid


def build_walkable_set(grid_df):
    """
    Muodostaa joukon (x, y) koordinaateista,
    joissa kävely on sallittu
    """
    return {
        (int(row.x), int(row.y))
        for row in grid_df.itertuples(index=False)
        if int(row.walkable) == 1
    }


def neighbors(cell):
    """
    Palauttaa 4-suuntaiset naapurit (ei diagonaaleja)
    """
    x, y = cell
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1)
    ]


# --------------------------------------------
# Lyhin etäisyys gridissä (BFS)
# --------------------------------------------
def shortest_path_length(start, goal, walkable):
    """
    BFS-lyhin etäisyys kahden pisteen välillä
    """
    start = tuple(start)
    goal = tuple(goal)

    queue = deque([start])
    visited = {start: 0}

    while queue:
        current = queue.popleft()
        if current == goal:
            return visited[current]

        for nb in neighbors(current):
            if nb in walkable and nb not in visited:
                visited[nb] = visited[current] + 1
                queue.append(nb)

    raise ValueError("Reittiä ei löytynyt")


def shortest_path(start, goal, walkable):
    """
    Palauttaa koko reitin solmuina (kävelyreitti)
    """
    start = tuple(start)
    goal = tuple(goal)

    queue = deque([start])
    prev = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = prev[current]
            return path[::-1]

        for nb in neighbors(current):
            if nb in walkable and nb not in prev:
                prev[nb] = current
                queue.append(nb)

    raise ValueError("Reittiä ei löytynyt")


# --------------------------------------------
# Reitin laskenta
# --------------------------------------------
def route_distance(route, walkable):
    """
    Laskee kokonaismatkan gridissä
    """
    total = 0
    for i in range(len(route) - 1):
        a = route[i][1:]
        b = route[i + 1][1:]
        total += shortest_path_length(a, b, walkable)
    return total


def nearest_neighbor(start, stops, end, walkable):
    """
    Nearest Neighbor -heuristiikka
    """
    route = [start]
    remaining = stops.copy()
    current = start

    while remaining:
        next_stop = min(
            remaining,
            key=lambda s: shortest_path_length(current[1:], s[1:], walkable)
        )
        route.append(next_stop)
        remaining.remove(next_stop)
        current = next_stop

    route.append(end)
    return route


def build_full_path(route, walkable):
    """
    Yhdistää yksittäiset reittiosat yhdeksi poluksi
    """
    full = []
    for i in range(len(route) - 1):
        segment = shortest_path(route[i][1:], route[i + 1][1:], walkable)
        if i > 0:
            segment = segment[1:]
        full.extend(segment)
    return full


# --------------------------------------------
# Visualisointi
# --------------------------------------------
def plot_route(grid, route, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    walk = grid[grid["walkable"] == 1]
    walls = grid[grid["walkable"] == 0]

    ax.scatter(walk["x"], walk["y"], s=10, label="Käytävä")
    ax.scatter(walls["x"], walls["y"], s=25, color="black", label="Hylly")

    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, linewidth=2)

    for i, (name, x, y) in enumerate(route):
        ax.scatter(x, y, marker="X", s=120)
        ax.text(x + 0.1, y + 0.1, f"{i}:{name}")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    return fig


# ============================================
# STREAMLIT UI
# ============================================
st.title("🛒 Reitinoptimointi")
st.write(
    "Valitse ostoslista ja optimoi reitti. "
    "Etäisyys lasketaan käytävillä (ei hyllyjen läpi)."
)

products, points, grid = load_data()
walkable = build_walkable_set(grid)

# Entrance & checkout
entrance_xy = points.loc[points["point"] == "entrance", ["x", "y"]].iloc[0]
checkout_xy = points.loc[points["point"] == "checkout", ["x", "y"]].iloc[0]

entrance = ("entrance", int(entrance_xy.x), int(entrance_xy.y))
checkout = ("checkout", int(checkout_xy.x), int(checkout_xy.y))

# Sidebar
st.sidebar.header("Ostoslista")
selected = st.sidebar.multiselect(
    "Valitse tuotteet",
    options=sorted(products["product"].unique()),
    default=["Maito", "Leipä", "Pasta", "Kahvi"]
)

method = st.sidebar.selectbox(
    "Optimointimenetelmä",
    ["Baseline", "Nearest Neighbor"]
)

if st.sidebar.button("Optimoi reitti"):
    items = products[products["product"].isin(selected)]
    stops = [
        (row.product, int(row.x), int(row.y))
        for row in items.itertuples()
    ]

    baseline = [entrance] + stops + [checkout]
    baseline_dist = route_distance(baseline, walkable)

    if method == "Baseline":
        best_route = baseline
    else:
        best_route = nearest_neighbor(entrance, stops, checkout, walkable)

    best_dist = route_distance(best_route, walkable)
    full_path = build_full_path(best_route, walkable)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Baseline", baseline_dist)
        st.metric("Valittu reitti", best_dist)
        st.metric("Säästö", baseline_dist - best_dist)
        st.write([name for name, _, _ in best_route])

    with col2:
        fig = plot_route(
            grid,
            best_route,
            full_path,
            f"{method} reitti (matka = {best_dist})"
        )
        st.pyplot(fig)