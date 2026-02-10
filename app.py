# ============================================
# app.py
# Streamlit UI – Kauppareitin optimointi
# ============================================
# - Etäisyys lasketaan käytävillä (ei hyllyjen läpi)
# - Näyttää Baseline vs Nearest Neighbor vs NN+2-opt
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import deque

# --------------------------------------------
# Streamlit config (vain kerran ja ennen muuta)
# --------------------------------------------
st.set_page_config(page_title="Reitinoptimointi", layout="wide")

# --------------------------------------------
# Data
# --------------------------------------------
@st.cache_data
def load_data():
    """Lataa projektin CSV-datat data/raw -kansiosta."""
    project_root = Path(__file__).resolve().parent
    data_raw = project_root / "data" / "raw"

    products = pd.read_csv(data_raw / "products.csv")
    points = pd.read_csv(data_raw / "store_points.csv")
    grid = pd.read_csv(data_raw / "store_grid.csv")
    return products, points, grid


def build_walkable_set(grid_df):
    """Palauttaa setin (x,y) soluista joissa walkable=1."""
    return {
        (int(r.x), int(r.y))
        for r in grid_df.itertuples(index=False)
        if int(r.walkable) == 1
    }


def neighbors(cell):
    """4-suuntaiset naapurit (ei diagonaaleja)."""
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


# --------------------------------------------
# BFS shortest path (grid)
# --------------------------------------------
def shortest_path_length(start, goal, walkable):
    """BFS-lyhin etäisyys kahden solun välillä askelina."""
    start = tuple(start)
    goal = tuple(goal)

    q = deque([start])
    dist = {start: 0}

    while q:
        cur = q.popleft()
        if cur == goal:
            return dist[cur]

        for nb in neighbors(cur):
            if nb in walkable and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)

    raise ValueError("Reittiä ei löytynyt (BFS).")


def shortest_path(start, goal, walkable):
    """Palauttaa koko polun solmuina (x,y)."""
    start = tuple(start)
    goal = tuple(goal)

    q = deque([start])
    prev = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]

        for nb in neighbors(cur):
            if nb in walkable and nb not in prev:
                prev[nb] = cur
                q.append(nb)

    raise ValueError("Reittiä ei löytynyt (BFS path).")


def build_full_path(route, walkable):
    """Yhdistää pysähdyspisteet yhdeksi käytäväpoluksi."""
    full = []
    for i in range(len(route) - 1):
        seg = shortest_path(route[i][1:], route[i + 1][1:], walkable)
        if i > 0:
            seg = seg[1:]  # vältetään tuplapiste liitoksessa
        full.extend(seg)
    return full


# --------------------------------------------
# Route distance + heuristics
# --------------------------------------------
def route_distance_cached(route, walkable, cache):
    """Kokonaismatka (askelia) välimuistilla BFS-etäisyyksille."""
    total = 0
    for i in range(len(route) - 1):
        a = tuple(route[i][1:])
        b = tuple(route[i + 1][1:])
        key = (a, b)
        if key not in cache:
            cache[key] = shortest_path_length(a, b, walkable)
        total += cache[key]
    return total


def nearest_neighbor(start, stops, end, walkable):
    """Nearest Neighbor -heuristiikka BFS-etäisyyksillä."""
    route = [start]
    remaining = stops.copy()
    cur = start

    while remaining:
        nxt = min(remaining, key=lambda s: shortest_path_length(cur[1:], s[1:], walkable))
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    route.append(end)
    return route


def two_opt(route, walkable, max_passes=10):
    """
    2-opt parantaa reittiä kääntämällä välejä (local search).
    Entrance ja checkout pidetään paikallaan.
    """
    cache = {}
    best = route[:]
    best_dist = route_distance_cached(best, walkable, cache)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                cand_dist = route_distance_cached(cand, walkable, cache)
                if cand_dist < best_dist:
                    best, best_dist = cand, cand_dist
                    improved = True
        if not improved:
            break

    return best, best_dist


# --------------------------------------------
# Plot
# --------------------------------------------
def plot_route(grid, route, path, title):
    """Piirtää käveltävät ruudut, hyllyt, polun ja pysähdykset."""
    fig, ax = plt.subplots(figsize=(7, 5))

    walk = grid[grid["walkable"] == 1]
    walls = grid[grid["walkable"] == 0]

    ax.scatter(walk["x"], walk["y"], s=10, label="Käytävä")
    ax.scatter(walls["x"], walls["y"], s=25, color="black", label="Hylly")

    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, linewidth=2)

    for i, (name, x, y) in enumerate(route):
        ax.scatter(x, y, marker="X", s=120)
        ax.text(x + 0.1, y + 0.1, f"{i}:{name}", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    return fig


# ============================================
# UI
# ============================================
st.title("🛒 Reitinoptimointi")
st.write("Valitse tuotteet ja optimoi reitti.")

products, points, grid = load_data()
walkable = build_walkable_set(grid)

# Entrance & checkout
entrance_xy = points.loc[points["point"] == "entrance", ["x", "y"]].iloc[0]
checkout_xy = points.loc[points["point"] == "checkout", ["x", "y"]].iloc[0]
entrance = ("entrance", int(entrance_xy.x), int(entrance_xy.y))
checkout = ("checkout", int(checkout_xy.x), int(checkout_xy.y))

# Sidebar: tuotteet
st.sidebar.header("Ostoslista")
selected = st.sidebar.multiselect(
    "Valitse tuotteet (järjestys säilyy)",
    options=sorted(products["product"].unique()),
    default=["Maito", "Leipä", "Pasta", "Kahvi"],
)

show_baseline = st.sidebar.checkbox("Näytä baseline kartta", value=False)

if st.sidebar.button("Optimoi reitti"):
    if not selected:
        st.warning("Valitse ainakin yksi tuote.")
        st.stop()

    # Säilytä käyttäjän valitsema järjestys
    selected_df = pd.DataFrame({"product": selected})
    items = selected_df.merge(products, on="product", how="left")

    if items[["x", "y"]].isna().any().any():
        missing = items[items["x"].isna() | items["y"].isna()]["product"].tolist()
        st.error(f"Puuttuvat koordinaatit tuotteille: {missing}")
        st.stop()

    stops = [(r.product, int(r.x), int(r.y)) for r in items.itertuples(index=False)]

    # Baseline
    baseline_route = [entrance] + stops + [checkout]
    baseline_path = build_full_path(baseline_route, walkable)
    baseline_dist = route_distance_cached(baseline_route, walkable, cache={})

    # NN
    nn_route = nearest_neighbor(entrance, stops, checkout, walkable)
    nn_path = build_full_path(nn_route, walkable)
    nn_dist = route_distance_cached(nn_route, walkable, cache={})

    # NN + 2-opt
    opt_route, opt_dist = two_opt(nn_route, walkable, max_passes=10)
    opt_path = build_full_path(opt_route, walkable)

    # Metrics
    s_nn = baseline_dist - nn_dist
    s_opt = baseline_dist - opt_dist
    s_nn_pct = (s_nn / baseline_dist) * 100 if baseline_dist else 0
    s_opt_pct = (s_opt / baseline_dist) * 100 if baseline_dist else 0

    st.subheader("📏 Mittarit")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline", baseline_dist)
    m2.metric("Nearest Neighbor", f"{nn_dist}  (−{s_nn_pct:.1f}%)")
    m3.metric("NN + 2-opt", f"{opt_dist}  (−{s_opt_pct:.1f}%)")

    st.subheader("🗺️ Reittivertailu")

left, right = st.columns(2)

with left:
    st.subheader("Nearest Neighbor")
    st.pyplot(plot_route(grid, nn_route, nn_path, f"NN (distance = {nn_dist})"))

with right:
    st.subheader("NN + 2-opt")
    st.pyplot(plot_route(grid, opt_route, opt_path, f"2-opt (distance = {opt_dist})"))

# Halutessa baseline omaan expanderiin (ei sotke vertailua)
if show_baseline:
    with st.expander("Baseline (vertailu)"):
        st.pyplot(plot_route(grid, baseline_route, baseline_path, f"Baseline (distance = {baseline_dist})"))

else:
    st.info("Valitse tuotteet vasemmalta ja paina **Optimoi reitti**.")