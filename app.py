# ============================================
# app.py — WOW++ Streamlit demo
# Kauppareitin optimointi käytävillä (BFS)
# Baseline vs Nearest Neighbor vs NN + 2-opt
# + Overlay, animaatio, KPI:t, export, pitch
# ============================================

import time
from pathlib import Path
from collections import deque

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
        nxt = min(
            remaining,
            key=lambda s: shortest_path_length(cur[1:], s[1:], walkable)
        )
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
# Helpers
# --------------------------------------------
def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def to_route_df(route):
    return pd.DataFrame(
        [{"order": i, "name": name, "x": x, "y": y} for i, (name, x, y) in enumerate(route)]
    )


# --------------------------------------------
# Plot
# --------------------------------------------
def plot_base_grid(ax, grid):
    """Piirtää käytävät + hyllyt samaan akseliin."""
    walk = grid[grid["walkable"] == 1]
    walls = grid[grid["walkable"] == 0]

    ax.scatter(walk["x"], walk["y"], s=10, label="Käytävä")
    ax.scatter(walls["x"], walls["y"], s=25, color="black", label="Hylly")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)


def plot_path(ax, path, label, color=None, lw=2.5, alpha=1.0):
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, linewidth=lw, label=label, color=color, alpha=alpha)


def plot_points(ax, route, show_labels=True):
    for i, (name, x, y) in enumerate(route):
        ax.scatter(x, y, marker="X", s=120)
        if show_labels:
            ax.text(x + 0.12, y + 0.12, f"{i}:{name}", fontsize=9)


def fig_overlay(grid, nn_route, nn_path, opt_route, opt_path, title, show_labels=True):
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    plot_base_grid(ax, grid)
    plot_path(ax, nn_path, "NN", color="tab:blue", lw=2.8, alpha=0.8)
    plot_path(ax, opt_path, "2-opt", color="tab:green", lw=2.8, alpha=0.8)
    plot_points(ax, opt_route, show_labels=show_labels)
    ax.set_title(title)
    ax.legend()
    return fig


def fig_single(grid, route, path, title, label, color=None, show_labels=True):
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    plot_base_grid(ax, grid)
    plot_path(ax, path, label=label, color=color, lw=2.8)
    plot_points(ax, route, show_labels=show_labels)
    ax.set_title(title)
    ax.legend()
    return fig


def fig_animated(grid, route, path, title, label, color=None, show_labels=True, upto=0):
    """Piirtää polun vain osittain (animaatio sliderilla)."""
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    plot_base_grid(ax, grid)

    upto = max(1, min(upto, len(path)))
    partial = path[:upto]
    plot_path(ax, partial, label=label, color=color, lw=3.2)

    plot_points(ax, route, show_labels=show_labels)
    ax.set_title(title)
    ax.legend()
    return fig


# ============================================
# UI
# ============================================
st.title("🛒 Reitinoptimointi — WOW++ demo")
st.caption("Grid-käytävät + BFS-etäisyys | Baseline vs NN vs NN+2-opt | Overlay + animaatio + export")

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
    "Valitse tuotteet (järjestys säilyy)",
    options=sorted(products["product"].unique()),
    default=["Maito", "Leipä", "Pasta", "Kahvi"],
)

st.sidebar.header("Mallin asetukset")
max_passes = st.sidebar.slider("2-opt passit", min_value=1, max_value=60, value=15, step=1)
step_meters = st.sidebar.slider("Askel → metri", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
walk_speed = st.sidebar.slider("Kävelynopeus (m/s)", min_value=0.6, max_value=2.0, value=1.3, step=0.1)

st.sidebar.header("Näyttö")
show_baseline = st.sidebar.checkbox("Näytä baseline (expander)", value=False)
show_labels = st.sidebar.checkbox("Näytä pisteiden nimet kartalla", value=True)
show_overlay = st.sidebar.checkbox("Näytä overlay-kartta (NN + 2-opt)", value=True)

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = None
if "autoplay" not in st.session_state:
    st.session_state["autoplay"] = False

# Action
if st.sidebar.button("Optimoi reitti"):
    if not selected:
        st.warning("Valitse ainakin yksi tuote.")
        st.session_state["results"] = None
    else:
        df_sel = pd.DataFrame({"product": selected})
        items = df_sel.merge(products, on="product", how="left")

        if items[["x", "y"]].isna().any().any():
            missing = items[items["x"].isna() | items["y"].isna()]["product"].tolist()
            st.error(f"Puuttuvat koordinaatit tuotteille: {missing}")
            st.session_state["results"] = None
        else:
            stops = [(r.product, int(r.x), int(r.y)) for r in items.itertuples(index=False)]
            dist_cache = {}

            # Baseline
            baseline_route = [entrance] + stops + [checkout]
            baseline_path = build_full_path(baseline_route, walkable)
            baseline_steps = route_distance_cached(baseline_route, walkable, cache=dist_cache)

            # NN
            nn_route = nearest_neighbor(entrance, stops, checkout, walkable)
            nn_path = build_full_path(nn_route, walkable)
            nn_steps = route_distance_cached(nn_route, walkable, cache=dist_cache)

            # NN + 2-opt
            opt_route, opt_steps = two_opt(nn_route, walkable, max_passes=max_passes)
            opt_path = build_full_path(opt_route, walkable)

            # KPI calc
            baseline_m = baseline_steps * step_meters
            nn_m = nn_steps * step_meters
            opt_m = opt_steps * step_meters

            baseline_t = baseline_m / walk_speed
            nn_t = nn_m / walk_speed
            opt_t = opt_m / walk_speed

            s_nn_m = baseline_m - nn_m
            s_opt_m = baseline_m - opt_m
            s_nn_pct = (s_nn_m / baseline_m) * 100 if baseline_m else 0
            s_opt_pct = (s_opt_m / baseline_m) * 100 if baseline_m else 0

            st.session_state["results"] = {
                "baseline": {
                    "route": baseline_route,
                    "path": baseline_path,
                    "steps": baseline_steps,
                    "meters": baseline_m,
                    "time_s": baseline_t,
                },
                "nn": {
                    "route": nn_route,
                    "path": nn_path,
                    "steps": nn_steps,
                    "meters": nn_m,
                    "time_s": nn_t,
                },
                "opt": {
                    "route": opt_route,
                    "path": opt_path,
                    "steps": opt_steps,
                    "meters": opt_m,
                    "time_s": opt_t,
                },
                "savings": {
                    "nn_m": s_nn_m,
                    "opt_m": s_opt_m,
                    "nn_pct": s_nn_pct,
                    "opt_pct": s_opt_pct,
                },
                "settings": {
                    "max_passes": max_passes,
                    "step_meters": step_meters,
                    "walk_speed": walk_speed,
                }
            }
            st.session_state["autoplay"] = False

# Render
res = st.session_state["results"]
if res is None:
    st.info("Valitse tuotteet vasemmalta ja paina **Optimoi reitti**.")
    st.stop()

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline", f'{res["baseline"]["meters"]:.0f} m', format_seconds(res["baseline"]["time_s"]))
k2.metric("Nearest Neighbor", f'{res["nn"]["meters"]:.0f} m', f'-{res["savings"]["nn_m"]:.0f} m ({res["savings"]["nn_pct"]:.1f}%)')
k3.metric("NN + 2-opt", f'{res["opt"]["meters"]:.0f} m', f'-{res["savings"]["opt_m"]:.0f} m ({res["savings"]["opt_pct"]:.1f}%)')
k4.metric("Arvioitu ajansäästö", format_seconds(res["baseline"]["time_s"] - res["opt"]["time_s"]))

tabs = st.tabs(["✨ Wow-näkymä", "🗺️ Kartat", "🎬 Animaatio", "🧾 Reitit", "⚙️ Data"])

with tabs[0]:
    st.subheader("✨ Wow-näkymä (overlay + pitch + vertailu)")
    left, right = st.columns([1.15, 0.85])

    with left:
        if show_overlay:
            fig = fig_overlay(
                grid,
                res["nn"]["route"], res["nn"]["path"],
                res["opt"]["route"], res["opt"]["path"],
                title="Overlay: NN (sininen) vs 2-opt (vihreä)",
                show_labels=show_labels,
            )
            st.pyplot(fig)
        else:
            st.info("Laita sivupalkista päälle: **Näytä overlay-kartta**")

    with right:
        st.markdown("### 📌 Pitch (copy/paste)")
        pitch = (
            f"- Optimoin ostosreitin kaupassa käytävillä (grid + BFS).\n"
            f"- Vertailu: Baseline vs Nearest Neighbor vs NN+2-opt.\n"
            f"- Tulos: säästö {res['savings']['opt_m']:.0f} m ({res['savings']['opt_pct']:.1f}%), "
            f"arvioitu ajansäästö {format_seconds(res['baseline']['time_s'] - res['opt']['time_s'])}.\n"
            f"- Parametrit: 2-opt passit={res['settings']['max_passes']}, askel→m={res['settings']['step_meters']}, kävely={res['settings']['walk_speed']} m/s."
        )
        st.code(pitch, language="text")

        st.markdown("### 🧠 Mitä algoritmit tekee?")
        st.write(
            "- **Baseline**: käy tuotteet siinä järjestyksessä kuin valitsit.\n"
            "- **Nearest Neighbor**: valitsee aina seuraavaksi lähimmän tuotteen (ahne).\n"
            "- **2-opt**: yrittää parantaa reittiä vaihtamalla/reversoimalla välejä (local search)."
        )

with tabs[1]:
    st.subheader("🗺️ Kartat")
    colA, colB = st.columns(2)

    with colA:
        st.pyplot(fig_single(
            grid, res["nn"]["route"], res["nn"]["path"],
            title=f"Nearest Neighbor ({res['nn']['steps']} steps)",
            label="NN", color="tab:blue", show_labels=show_labels
        ))

    with colB:
        st.pyplot(fig_single(
            grid, res["opt"]["route"], res["opt"]["path"],
            title=f"NN + 2-opt ({res['opt']['steps']} steps)",
            label="2-opt", color="tab:green", show_labels=show_labels
        ))

    if show_baseline:
        with st.expander("Näytä Baseline"):
            st.pyplot(fig_single(
                grid, res["baseline"]["route"], res["baseline"]["path"],
                title=f"Baseline ({res['baseline']['steps']} steps)",
                label="Baseline", color="tab:gray", show_labels=show_labels
            ))

with tabs[2]:
    st.subheader("🎬 Reittianimaatio (2-opt)")
    path = res["opt"]["path"]
    n = len(path)

    # Autoplay controls
    a1, a2, a3 = st.columns([0.22, 0.22, 0.56])
    with a1:
        if st.button("▶️ Play"):
            st.session_state["autoplay"] = True
    with a2:
        if st.button("⏸ Pause"):
            st.session_state["autoplay"] = False
    with a3:
        speed_ms = st.slider("Nopeus (ms / frame)", min_value=30, max_value=400, value=120, step=10)

    # Frame slider
    frame = st.slider("Frame", min_value=1, max_value=n, value=min(40, n), step=1)

    # Autoplay loop: rerun by toggling session_state
    if st.session_state["autoplay"]:
        frame = min(frame + 1, n)
        # Hack: keep slider value in session_state
        st.session_state["_frame_override"] = frame
        time.sleep(speed_ms / 1000.0)
        st.rerun()

    # Apply override once (so slider can move)
    if "_frame_override" in st.session_state:
        frame = st.session_state.pop("_frame_override")

    st.pyplot(fig_animated(
        grid, res["opt"]["route"], res["opt"]["path"],
        title=f"2-opt reitti (frame {frame}/{n})",
        label="2-opt", color="tab:green", show_labels=show_labels, upto=frame
    ))

with tabs[3]:
    st.subheader("🧾 Reitit ja export")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Optimoitu reitti (2-opt)")
        df_opt = to_route_df(res["opt"]["route"])
        st.dataframe(df_opt, use_container_width=True)

        st.download_button(
            "⬇️ Lataa optimoitu reitti CSV:nä",
            data=df_opt.to_csv(index=False).encode("utf-8"),
            file_name="optimized_route.csv",
            mime="text/csv",
        )

    with r2:
        st.markdown("#### Nearest Neighbor reitti")
        df_nn = to_route_df(res["nn"]["route"])
        st.dataframe(df_nn, use_container_width=True)

        st.download_button(
            "⬇️ Lataa NN reitti CSV:nä",
            data=df_nn.to_csv(index=False).encode("utf-8"),
            file_name="nn_route.csv",
            mime="text/csv",
        )

with tabs[4]:
    st.subheader("⚙️ Data")
    st.json(res["settings"])

    with st.expander("Näytä products.csv"):
        st.dataframe(products, use_container_width=True)

    with st.expander("Näytä store_points.csv"):
        st.dataframe(points, use_container_width=True)

    with st.expander("Näytä store_grid.csv (head 200)"):
        st.dataframe(grid.head(200), use_container_width=True)