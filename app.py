import json
import math
import hashlib
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_plotly_events import plotly_events

# IMPORTANT: must be the first Streamlit command
st.set_page_config(page_title="IntellectMap", layout="wide")

# -----------------------------
# HERO (default)
# -----------------------------
if "show_app" not in st.session_state:
    st.session_state["show_app"] = False

def _img_to_data_uri(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    b = base64.b64encode(p.read_bytes()).decode("utf-8")
    ext = p.suffix.lower().replace(".", "")
    mime = "png" if ext == "png" else ("jpeg" if ext in ["jpg", "jpeg"] else ext)
    return f"data:image/{mime};base64,{b}"

if not st.session_state["show_app"]:
    bg_uri = _img_to_data_uri("img/Hero.png")  # <-- save your map image here

    bg_css = ""
    if bg_uri:
        bg_css = f"""
        .hero-bg {{
            position:absolute;
            inset:0;
            background-image: url("{bg_uri}");
            background-size: cover;
            background-position: center;
            filter: blur(10px);
            opacity: .22;
            transform: scale(1.08);
            z-index: 0;
        }}
        """
    else:
        # fallback: no image
        bg_css = """
        .hero-bg{
            position:absolute;
            inset:0;
            background: radial-gradient(900px 520px at 50% 10%, rgba(60,120,255,.10), transparent 60%),
                        radial-gradient(900px 520px at 50% 80%, rgba(255,80,200,.08), transparent 60%);
            z-index:0;
        }
        """

    st.markdown(
        f"""
<style>
section[data-testid="stSidebar"] {{ display:none; }}
.block-container {{ padding-top: 2.2rem; }}

.hero-wrap{{
    position: relative;
    min-height: 88vh;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    overflow:hidden;
}}

{bg_css}

.hero-content{{
    position: relative;
    z-index: 2;
    max-width: 980px;
    padding: 10px 20px;
}}

.hero-title{{
    font-size: 78px;
    font-weight: 900;
    letter-spacing: -1.6px;
    line-height: 1.05;
    color: #2d2f36;
    margin: 0;
}}

.hero-tagline{{
    font-size: 22px;
    color: #3d4048;
    margin: 18px 0 0 0;
}}

.hero-sub{{
    font-size: 16px;
    color: #6b6f7a;
    margin: 10px 0 0 0;
}}

.hero-card{{
    display:inline-block;
    margin-top: 26px;
    padding: 18px 22px;
    border: 1px solid rgba(0,0,0,.06);
    background: rgba(255,255,255,.65);
    backdrop-filter: blur(10px);
    border-radius: 18px;
}}

.hero-hint{{
    font-size: 13px;
    color: #7b7f8a;
    margin-top: 10px;
}}

.stButton > button {{
    width: 220px;
    height: 48px;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,.10);
    background: white;
    color: #2d2f36;
    font-weight: 700;
    transition: all .18s ease;
    box-shadow: 0 10px 25px rgba(0,0,0,.08);
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(0,0,0,.12);
}}
.stButton > button:active {{
    transform: translateY(0px) scale(.99);
}}
</style>

<div class="hero-wrap">
  <div class="hero-bg"></div>

  <div class="hero-content">
    <h1 class="hero-title">IntellectMap</h1>
    <p class="hero-tagline">Find people, clubs, and events by interest — instantly.</p>
    <p class="hero-sub">Explore profiles, get recommended matches, and save the best connections.</p>
    <p class="hero-main">Start exploring your territory map</p>
    <p class="hero-hint">Click the Explore button to enter.</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([4, 2, 4])
    with c2:
        if st.button("Explore", use_container_width=True):
            st.session_state["show_app"] = True
            st.rerun()

    st.stop()

# -----------------------------
# APP VIEW (sidebar + Home)
# -----------------------------
with st.sidebar:
    if st.button("← Home", use_container_width=True):
        st.session_state["show_app"] = False
        st.rerun()

# ----------------------------
# APP CSS (map entrance animation)
# ----------------------------
st.markdown(
    """
<style>
:root{ --ease-apple: cubic-bezier(.16, 1, .3, 1); }
.globe-wrap{
  transform-origin: 50% 100%;
  animation: globeIn 900ms var(--ease-apple) both;
  will-change: transform, opacity, filter;
}
@keyframes globeIn{
  0%   { opacity: 0; transform: translate3d(0, 44px, 0) scale(.94); filter: blur(10px); }
  60%  { opacity: 1; filter: blur(2px); }
  100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); filter: blur(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_nodes() -> pd.DataFrame:
    with open("data/nodes.json", "r", encoding="utf-8") as f:
        nodes = json.load(f)
    df = pd.DataFrame(nodes)

    if "tags" not in df.columns:
        df["tags"] = [[] for _ in range(len(df))]
    if "description" not in df.columns:
        df["description"] = ""
    if "domain" not in df.columns:
        df["domain"] = "General"
    if "university" not in df.columns:
        df["university"] = "Unknown University"
    if "type" not in df.columns:
        df["type"] = "person"
    if "title" not in df.columns:
        df["title"] = "Untitled"

    return df

# ----------------------------
# Map helpers
# ----------------------------
UNIVERSITY_CENTERS = {
    "Saint Louis University": (38.6365, -90.2332),
    "Washington University in St. Louis": (38.6488, -90.3108),
    "Harvard University": (42.3770, -71.1167),
    "MIT": (42.3601, -71.0942),
    "Stanford University": (37.4275, -122.1697),
    "UC Berkeley": (37.8719, -122.2585),
    "NYU": (40.7295, -73.9965),
    "Columbia University": (40.8075, -73.9626),
    "University of Chicago": (41.7886, -87.5987),

    "University of Toronto": (43.6629, -79.3957),
    "University of British Columbia": (49.2606, -123.2460),

    "University of Oxford": (51.7548, -1.2544),
    "University of Cambridge": (52.2043, 0.1149),
    "Imperial College London": (51.4988, -0.1749),
    "UCL": (51.5246, -0.1340),

    "ETH Zurich": (47.3763, 8.5476),
    "TU Munich": (48.2620, 11.6670),
    "Sorbonne University": (48.8462, 2.3449),

    "IIT Delhi": (28.5450, 77.1926),
    "IIT Bombay": (19.1334, 72.9133),
    "National University of Singapore": (1.2966, 103.7764),
    "Nanyang Technological University": (1.3483, 103.6831),
    "Tsinghua University": (40.0030, 116.3269),
    "University of Tokyo": (35.7126, 139.7610),

    "University of Melbourne": (-37.7963, 144.9614),
    "University of Sydney": (-33.8898, 151.1872),

    "KAUST": (22.3041, 39.1046),
    "University of Cape Town": (-33.9570, 18.4607),
}
DEFAULT_CENTER = (20.0, 0.0)

def _stable_int(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16)

def meters_to_latlon(lat0: float, lon0: float, dx_m: float, dy_m: float):
    dlat = dy_m / 111_111.0
    dlon = dx_m / (111_111.0 * max(0.2, math.cos(math.radians(lat0))))
    return lat0 + dlat, lon0 + dlon

def add_geo_positions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        if df["lat"].notna().all() and df["lon"].notna().all():
            return df

    centers = [UNIVERSITY_CENTERS.get(str(uni), DEFAULT_CENTER) for uni in df["university"].astype(str).tolist()]
    df["_center_lat"] = [c[0] for c in centers]
    df["_center_lon"] = [c[1] for c in centers]

    domains = sorted(df["domain"].astype(str).unique().tolist())
    domain_to_angle = {d: (i / max(1, len(domains))) * 2 * math.pi for i, d in enumerate(domains)}
    domain_to_radius_m = {d: 900 + 160 * (i % 6) for i, d in enumerate(domains)}

    lats, lons = [], []
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        domain = str(r.get("domain", "General"))
        typ = str(r.get("type", "person"))
        lat0, lon0 = float(r["_center_lat"]), float(r["_center_lon"])

        ang = domain_to_angle.get(domain, 0.0)
        ring_r = domain_to_radius_m.get(domain, 900.0)
        tx = ring_r * math.cos(ang)
        ty = ring_r * math.sin(ang)

        seed = _stable_int(f"{title}|{domain}|{typ}|{r.get('university','')}")
        rng = np.random.default_rng(seed)
        jx = float(rng.normal(0, 280))
        jy = float(rng.normal(0, 280))

        lat, lon = meters_to_latlon(lat0, lon0, dx_m=(tx + jx), dy_m=(ty + jy))
        lats.append(lat)
        lons.append(lon)

    df["lat"] = lats
    df["lon"] = lons
    return df.drop(columns=["_center_lat", "_center_lon"], errors="ignore")

# ----------------------------
# Text similarity
# ----------------------------
def build_text(row: pd.Series) -> str:
    tags = row.get("tags", [])
    tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)
    return f'{row.get("title","")} {row.get("type","")} {row.get("university","")} {row.get("domain","")} {tags_str} {row.get("description","")}'

def compute_vectors(df: pd.DataFrame):
    texts = df.apply(build_text, axis=1).tolist()
    vec = TfidfVectorizer(stop_words="english", max_features=2500)
    X = vec.fit_transform(texts)
    return vec, X

def similarity_rank(df: pd.DataFrame, X, user_vec, top_n: int = 12) -> pd.DataFrame:
    sims = cosine_similarity(user_vec, X).flatten()
    out = df.copy()
    out["score"] = sims
    return out.sort_values("score", ascending=False).head(top_n)

def intro_message(user_name, user_uni, user_interests, target_title, overlap):
    overlap_str = ", ".join(overlap[:4]) if overlap else "your work"
    interests_str = ", ".join(user_interests[:3]) if user_interests else "your area"
    return (
        f"Hi {target_title},\n\n"
        f"I'm {user_name} from {user_uni}. I noticed we overlap on {overlap_str}. "
        f"I'm looking to connect with people in {interests_str} and would love to learn what you're working on.\n\n"
        f"Would you be open to a quick chat?\n\n"
        f"Thanks,\n{user_name}"
    )

# ----------------------------
# Load + state
# ----------------------------
df = add_geo_positions(load_nodes())

if "saved" not in st.session_state:
    st.session_state["saved"] = {}

if "selected_node_id" not in st.session_state:
    st.session_state["selected_node_id"] = None

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Your Profile")
    user_name = st.text_input("Name", value="Smit")

    unis = sorted(df["university"].astype(str).unique().tolist())
    default_idx = unis.index("Saint Louis University") if "Saint Louis University" in unis else 0
    user_uni = st.selectbox("University", unis, index=default_idx)

    domains = sorted(df["domain"].astype(str).unique().tolist())
    user_domains = st.multiselect("Interest domains", domains, default=[])

    node_types = st.multiselect("Show types", ["person", "club", "event"], default=[])

    st.divider()
    global_view = st.toggle("Global View (show all universities)", value=True)
    st.caption("Tip: Turn OFF Global View for a campus territory look.")

# ----------------------------
# No filters → empty globe + message
# ----------------------------
no_filters = (not user_domains) and (not node_types)
if no_filters:
    empty = pd.DataFrame({"lat": [], "lon": []})
    fig_empty = px.scatter_geo(empty, lat="lat", lon="lon", projection="orthographic")
    fig_empty.update_layout(
        geo=dict(
            projection_type="orthographic",
            showland=True, landcolor="rgb(235,235,235)",
            showocean=True, oceancolor="rgb(180,215,235)",
            showcountries=True, countrycolor="rgb(140,140,140)",
            showcoastlines=True, coastlinecolor="rgb(120,120,120)",
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep",
    )
    st.markdown('<div class="globe-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_empty, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; padding:12px 0 6px;">
          <h3 style="margin-bottom:6px;">Select filters to begin</h3>
          <p style="color:#666; margin:0;">Choose interest domains or types from the sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------
# Apply filters
# ----------------------------
f = df.copy()
if node_types:
    f = f[f["type"].astype(str).isin(node_types)]
if user_domains:
    f = f[f["domain"].astype(str).isin(user_domains)]
if not global_view:
    f = f[f["university"].astype(str) == str(user_uni)]

if len(f) < 5:
    st.warning("Not enough nodes with current filters. Select more domains or types, or enable Global View.")
    st.stop()

f["lon"] = ((f["lon"].astype(float) + 180) % 360) - 180

vec, X = compute_vectors(f)
user_text = f"{user_name} {user_uni} " + " ".join(user_domains)
user_vec = vec.transform([user_text])

tab1, tab2, tab3, tab4 = st.tabs(["Map", "Recommendations", "Story", "Saved"])

# ----------------------------
# Map tab
# ----------------------------
with tab1:
    st.subheader("Territory map")
    st.caption("Click a dot to inspect it (Global View controls globe vs campus)")

    fm = f.reset_index(drop=True).copy()
    fm["_node_id"] = fm.apply(
        lambda r: hashlib.sha256(
            f"{r.get('title','')}|{r.get('type','')}|{r.get('domain','')}|{r.get('university','')}".encode("utf-8")
        ).hexdigest()[:16],
        axis=1,
    )

    titles = fm["title"].astype(str).tolist()

    size_map = {"person": 10, "club": 14, "event": 18}
    fm["size"] = fm["type"].astype(str).map(size_map).fillna(12)

    # --- FORCE numeric + remove invalid coords ---
    fm["lat"] = pd.to_numeric(fm["lat"], errors="coerce")
    fm["lon"] = pd.to_numeric(fm["lon"], errors="coerce")
    fm = fm.dropna(subset=["lat", "lon"])

    if fm.empty:
        st.warning("No valid coordinates to display.")
        st.stop()

    st.caption(f"Nodes on map: {len(fm)} | NaN coords removed: {f.shape[0] - fm.shape[0]}")

    # Normalize longitude
    fm["lon"] = ((fm["lon"] + 180) % 360) - 180

    # --- Center globe on actual data ---
    center_lat = float(fm["lat"].mean())
    center_lon = float(fm["lon"].mean())

    if global_view:
        fig = px.scatter_geo(
            fm,
            lat="lat",
            lon="lon",
            color="domain",
            size="size",
            symbol="type",
            hover_name="title",
            projection="orthographic",
        )

        fig.update_traces(
            customdata=fm[["_node_id", "university", "type", "domain"]].to_numpy(),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "%{customdata[1]}<br>"
                "%{customdata[2]} • %{customdata[3]}"
                "<extra></extra>"
            ),
        )

        fig.update_layout(
            geo=dict(
                projection_type="orthographic",
                projection_rotation=dict(lat=center_lat, lon=center_lon),  # ⭐ CRITICAL FIX
                showland=True,
                landcolor="rgb(235,235,235)",
                showocean=True,
                oceancolor="rgb(180,215,235)",
                showcountries=True,
                countrycolor="rgb(140,140,140)",
                showcoastlines=True,
                coastlinecolor="rgb(120,120,120)",
                bgcolor="white",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            uirevision="keep",
            legend_title_text="Domains",
        )
    else:
        latlon = UNIVERSITY_CENTERS.get(str(user_uni), (float(fm["lat"].mean()), float(fm["lon"].mean())))
        center = {"lat": float(latlon[0]), "lon": float(latlon[1])}

        fig = px.scatter_mapbox(
            fm, lat="lat", lon="lon",
            color="domain", size="size",
            hover_name="title", zoom=11.5, center=center,
        )
        fig.update_traces(
            customdata=fm[["_node_id", "university", "type", "domain"]].to_numpy(),
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "%{customdata[1]}<br>"
                "%{customdata[2]} • %{customdata[3]}"
                "<extra></extra>"
            ),
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            uirevision="keep",
            legend_title_text="Domains",
            mapbox=dict(bounds=dict(west=-180, east=180, south=-85, north=85)),
        )

    fig.update_layout(dragmode="pan")

    st.markdown('<div class="globe-wrap">', unsafe_allow_html=True)
    clicked = plotly_events(
        fig, click_event=True, hover_event=False, select_event=False,
        override_height=560, key="map",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["selected_node_id"] is None:
        st.session_state["selected_node_id"] = fm["_node_id"].iloc[0]

    if "last_click_sig" not in st.session_state:
        st.session_state["last_click_sig"] = None

    if clicked:
        ev = clicked[0]
        sig = (ev.get("curveNumber"), ev.get("pointIndex"))
        if sig != st.session_state["last_click_sig"]:
            st.session_state["last_click_sig"] = sig
            cn = ev.get("curveNumber", 0)
            pi = ev.get("pointIndex", None)

            new_id = None
            try:
                if pi is not None:
                    new_id = fig.data[cn].customdata[pi][0]
            except Exception:
                new_id = None

            if new_id and str(new_id) != str(st.session_state["selected_node_id"]):
                st.session_state["selected_node_id"] = str(new_id)
                st.rerun()

    if st.session_state["selected_node_id"] not in set(fm["_node_id"].tolist()):
        st.session_state["selected_node_id"] = fm["_node_id"].iloc[0]

    node = fm[fm["_node_id"] == st.session_state["selected_node_id"]].iloc[0]

    colA, colB = st.columns([1, 3])
    with colA:
        node_id = str(node["_node_id"])
        already_saved = node_id in st.session_state["saved"]
        if st.button("Saved" if already_saved else "Save", key=f"save_{node_id}", disabled=already_saved):
            st.session_state["saved"][node_id] = {
                "node_id": node_id,
                "title": str(node.get("title", "")),
                "type": str(node.get("type", "")),
                "domain": str(node.get("domain", "")),
                "university": str(node.get("university", "")),
                "description": str(node.get("description", "")),
            }
            st.success("Saved to your list.")
    with colB:
        st.caption(f"Saved: {len(st.session_state['saved'])}")

    sel_index = titles.index(str(node["title"])) if str(node["title"]) in titles else 0
    selected_title = st.selectbox("Select a node to inspect", titles, index=sel_index, key="selected_title")

    if selected_title != str(node["title"]):
        cand = fm[fm["title"].astype(str) == str(selected_title)]
        if not cand.empty:
            st.session_state["selected_node_id"] = cand["_node_id"].iloc[0]
            st.rerun()

    node = fm[fm["_node_id"] == st.session_state["selected_node_id"]].iloc[0]
    st.markdown(f"### {node['title']}")
    st.write(f"Type: {node['type']} | Domain: {node['domain']} | University: {node['university']}")
    st.write(node.get("description", ""))

    node_tags = set(node["tags"]) if isinstance(node.get("tags", []), list) else set()
    overlap = list(node_tags.intersection(set(user_domains)))
    if node.get("domain") in user_domains and node.get("domain") not in overlap:
        overlap.append(node.get("domain"))

    msg = intro_message(user_name, user_uni, user_domains, node["title"], overlap)
    st.text_area("Generated intro message", value=msg, height=180)

# ----------------------------
# Recommendations tab
# ----------------------------
with tab2:
    st.subheader("Top matches for you")
    rec = similarity_rank(f, X, user_vec, top_n=12)

    show = rec[["title", "type", "domain", "university", "score"]].copy()
    show["score"] = show["score"].round(3)
    st.dataframe(show, use_container_width=True)

    pick = st.selectbox("Pick a recommended node", rec["title"].astype(str).tolist(), key="rec_pick")
    node2 = rec[rec["title"].astype(str) == str(pick)].iloc[0]
    st.markdown(f"### {node2['title']}")
    st.write(f"Type: {node2['type']} | Domain: {node2['domain']} | University: {node2['university']}")
    st.write(node2.get("description", ""))

# ----------------------------
# Story tab
# ----------------------------
with tab3:
    st.subheader("Problem → Solution → Why now")
    st.write(
        "Students don’t know where to find the right people on campus. Existing platforms mostly show only your connections.\n\n"
        "IntellectMap clusters people, clubs, and events by interest and visualizes them like territories on a world/campus map.\n"
        "So a first-year can network like a final-year instantly."
    )
    st.write("Intellect MAP: (DEMO) Mock data + matching + territory map.")
    st.write("Intellect MAP: (Future)Real universities + real roles + real people.")

# ----------------------------
# Saved tab
# ----------------------------
with tab4:
    st.subheader("Saved list")
    st.caption("Your saved people, clubs, and events (session-only)")

    saved_dict = st.session_state.get("saved", {})
    if not saved_dict:
        st.info("No saved nodes yet. Go to the Map tab and click Save on a node.")
    else:
        saved_df = pd.DataFrame(list(saved_dict.values()))
        show_cols = [c for c in ["title", "type", "domain", "university", "description"] if c in saved_df.columns]
        st.dataframe(saved_df[show_cols], use_container_width=True)

        csv_bytes = saved_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download saved list (CSV)",
            data=csv_bytes,
            file_name="intellectmap_saved.csv",
            mime="text/csv",
        )

        st.divider()
        st.write("Remove items")
        for row in saved_df.itertuples(index=False):
            rid = getattr(row, "node_id", None)
            title = getattr(row, "title", "Untitled")
            meta = f"{getattr(row, 'type', '')} · {getattr(row, 'domain', '')} · {getattr(row, 'university', '')}"
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"**{title}** — {meta}")
            with c2:
                if st.button("Remove", key=f"rm_{rid}"):
                    st.session_state["saved"].pop(rid, None)
                    st.rerun()

st.caption("IntellectMap: mock data, similarity matching, and territory-style map visualization.")
