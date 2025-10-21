import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import requests

st.set_page_config(page_title="Seed of Tomorrow", page_icon="🌱", layout="wide")

# ----------------------------
# Constants & data
# ----------------------------
DEFAULT_CENTER = [24.182, 120.610]  # Taichung Xitun approx
CLASSES = ["Deadwood", "Pest Damage", "Soil Issue", "Healthy"]

# Taiwan cities/districts you’ll demo (add more if needed)
DISTRICTS = {
    "Taichung City": {
        "Xitun District": [24.182, 120.610],
        "North District": [24.167, 120.683],
        "West District":  [24.145, 120.662],
        "Dali District":  [24.094, 120.676],
        "Wuri District":  [24.074, 120.606],
    },
    "Taipei City": {
        "Zhongzheng District": [25.0327, 121.5199],
        "Da’an District":      [25.0260, 121.5436],
        "Wanhua District":     [25.0354, 121.4970],
    },
}

# Fungi / pH actions and approximate CO2 benefit
SOLUTIONS = {
    "Deadwood":    ("Inoculate Pleurotus ostreatus", 25),
    "Pest Damage": ("Apply Trichoderma harzianum",   15),
    "Soil Issue":  ("Adjust pH (lime/sulfur)",        10),
    "Healthy":     ("Monitoring only",                 0),
}

# ----------------------------
# Tiny deterministic “AI” (mock). Replace later with real model.
# ----------------------------
def predict_from_bytes(image_bytes: bytes):
    """Return (label, confidence 0-1) based on a stable hash of the image."""
    if not image_bytes:
        return "Healthy", 0.55
    h = hashlib.sha256(image_bytes).hexdigest()
    idx = int(h[:2], 16) % len(CLASSES)
    conf = 0.6 + (int(h[2:4], 16) / 255.0) * 0.39
    return CLASSES[idx], round(conf, 2)

# ----------------------------
# OSM base tree fetch (fallback)
# ----------------------------
def fetch_osm_trees(center, km=1.2):
    """
    Fetch OSM points tagged natural=tree within ~km radius of center.
    Returns list[(lat, lon)].
    """
    lat, lon = center
    d = km / 111.0  # rough degree buffer for small areas
    bbox = (lat - d, lon - d, lat + d, lon + d)
    query = f"""
    [out:json][timeout:25];
    node["natural"="tree"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    out body;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={'data': query})
    r.raise_for_status()
    data = r.json()
    return [(el['lat'], el['lon']) for el in data.get('elements', []) if el['type']=='node']

# ----------------------------
# Session init
# ----------------------------
if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns="""
        ts reporter city district label tree_id status confidence diameter_cm soil_pH action co2_saved_kg lat lon notify
    """.split())

# ----------------------------
# Sidebar metrics + export
# ----------------------------
st.sidebar.title("🌱 Seed of Tomorrow")
st.sidebar.caption("Team HengJie · AI + Fungi Net-Zero System")
st.sidebar.markdown("---")
total_cases = len(st.session_state.db)
total_co2 = float(st.session_state.db["co2_saved_kg"].sum()) if total_cases else 0.0
st.sidebar.metric("Total cases", total_cases)
st.sidebar.metric("Total CO₂ saved", f"{total_co2:.1f} kg")
st.sidebar.markdown("---")
if total_cases:
    st.sidebar.download_button(
        "⬇️ Export logs.csv",
        st.session_state.db.to_csv(index=False).encode("utf-8"),
        file_name="seed_of_tomorrow_logs.csv",
        mime="text/csv"
    )

# ----------------------------
# Tabs
# ----------------------------
tab_map, tab_report, tab_dashboard, tab_gov = st.tabs([
    "🗺️ Map", "📷 Report Tree", "📊 Dashboard", "🏛️ Government View"
])

# ============================
# Tab 1: Map
# ============================
with tab_map:
    st.subheader("Interactive Map")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        city_view = st.selectbox("City / 城市", list(DISTRICTS.keys()), index=0)
    with colB:
        district_view = st.selectbox("District / 行政區", list(DISTRICTS[city_view].keys()), index=0)
    with colC:
        st.write("")

    center = DISTRICTS[city_view][district_view]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # Base tree layer options
    st.write("**Base tree layers**")
    col1, col2 = st.columns(2)
    with col1:
        show_xitun = st.checkbox("Show base trees — Xitun demo CSV", value=(city_view=="Taichung City" and district_view=="Xitun District"))
    with col2:
        show_osm = st.checkbox("Show base trees — OSM natural=tree", value=False)

    # B) OSM fallback (works anywhere)
    if show_osm:
        try:
            pts = fetch_osm_trees(center)
            for lat0, lon0 in pts:
                folium.CircleMarker([lat0, lon0], radius=3, color="#1a9850",
                                    fill=True, fill_color="#1a9850").add_to(m)
            st.caption(f"🟢 Loaded {len(pts)} trees from OSM (natural=tree)")
        except Exception as e:
            st.warning(f"OSM fetch failed: {e}")

    # Plot logged markers (citizen reports)
    df_map = st.session_state.db.copy()
    if len(df_map):
        df_map = df_map[(df_map["city"] == city_view) & (df_map["district"] == district_view)]
        for _, r in df_map.iterrows():
            c = ("blue" if r["status"] == "Deadwood" and (5.8 <= float(r["soil_pH"]) <= 7.5)
                 else "orange" if r["status"] == "Pest Damage"
                 else "yellow" if not (5.8 <= float(r["soil_pH"]) <= 7.5)
                 else "green")
            folium.CircleMarker(
                [float(r["lat"]), float(r["lon"])], radius=7, color=c, fill=True, fill_color=c,
                popup=f"{r['tree_id']} · {r['status']} → {r['action']} ({r['co2_saved_kg']} kg)"
            ).add_to(m)

    st_folium(m, width=1000, height=540)

# ============================
# Tab 2: Report Tree (Citizen)
# ============================
with tab_report:
    st.subheader("Citizen Reporting · 公民回報")

    loc_mode = st.radio(
        "📍 Choose location mode",
        ["City → District", "Search address / lane", "Use my location (GPS)", "Pick on map"],
        horizontal=True
    )

    geolocator = Nominatim(user_agent="seed_of_tomorrow_demo")
    picked_latlon, picked_label = None, None
    picked_city, picked_district = "", ""

    if loc_mode == "City → District":
        picked_city = st.selectbox("City / 城市", list(DISTRICTS.keys()), index=0, key="rep_city")
        picked_district = st.selectbox("District / 行政區", list(DISTRICTS[picked_city].keys()), index=0, key="rep_dist")
        picked_latlon = DISTRICTS[picked_city][picked_district]
        picked_label = f"{picked_city}, {picked_district}"

    elif loc_mode == "Search address / lane":
        picked_city = st.selectbox("City / 城市", list(DISTRICTS.keys()), index=0, key="rep_city2")
        addr = st.text_input("地址 / 巷 / 路 (e.g., '台中市西區河南路三段 100 巷')")
        if st.button("🔎 Geocode"):
            if addr.strip():
                q = addr if picked_city in addr else f"{picked_city} {addr}"
                loc = geolocator.geocode(q, timeout=10)
                if loc:
                    picked_latlon = [loc.latitude, loc.longitude]
                    picked_label = loc.address
                    st.success(f"Found: {picked_label}")
                else:
                    st.error("找不到地址，請換個關鍵字或加上門牌/區名")
        
    else:  # Pick on map
        st.caption("在地圖上點一下以放置樹木位置")
        tmp_map = folium.Map(location=DEFAULT_CENTER, zoom_start=13, tiles="cartodbpositron")
        out = st_folium(tmp_map, width=900, height=420)
        if out and out.get("last_clicked"):
            picked_latlon = [out["last_clicked"]["lat"], out["last_clicked"]["lng"]]
            picked_label  = f"Map pin: {picked_latlon[0]:.5f}, {picked_latlon[1]:.5f}"
            st.success(f"選擇：{picked_label}")

    st.markdown("---")

    if picked_latlon:
        col1, col2 = st.columns([1,1])

        with col1:
            reporter = st.text_input("Your name (optional)", value="Citizen")
            lat = st.number_input("Latitude", value=float(picked_latlon[0]), format="%.6f")
            lon = st.number_input("Longitude", value=float(picked_latlon[1]), format="%.6f")
            tree_id = st.text_input("Tree ID (optional)", value=f"T-{np.random.randint(1000,9999)}")
            diameter = st.slider("Diameter (cm)", 5, 120, 32)
            soil_pH = st.slider("Soil pH", 3.0, 9.5, 6.2, 0.1)

            photo = st.file_uploader("Upload tree photo", type=["jpg","png"])
            status, conf = ("Healthy", 0.55)
            if photo:
                st.image(photo, caption="Uploaded tree", use_column_width=True)
                status, conf = predict_from_bytes(photo.getvalue())

            st.write(f"**AI Prediction**: {status} (confidence {conf})")
            action, co2_saved = SOLUTIONS[status]

            pH_ok = (5.8 <= soil_pH <= 7.5)
            if not pH_ok and status != "Healthy":
                st.warning("⚠️ pH not ideal — correct soil first, then apply fungi.")
            if co2_saved:
                st.metric("CO₂ saved (if treated)", f"{co2_saved} kg")
            st.info(f"Recommended action: **{action}**")

            near_walkway = st.checkbox("Near sidewalk (<2 m)?", value=True)
            cluster_reports = st.number_input("Reports in 200 m (7 days)", 0, 20, 0)
            notify = (
                (status == "Deadwood" and diameter > 40 and near_walkway) or
                (status == "Pest Damage" and cluster_reports >= 3) or
                (soil_pH < 4.5 or soil_pH > 8.5)
            )
            if notify:
                st.error("建議通報主管機關 / Notify Government")

            if st.button("➕ Log this tree"):
                row = dict(
                    ts=datetime.utcnow().isoformat(timespec="seconds"),
                    reporter=reporter,
                    city=picked_city,
                    district=picked_district,
                    label=picked_label or "",
                    tree_id=tree_id, status=status, confidence=conf,
                    diameter_cm=diameter, soil_pH=soil_pH,
                    action=action, co2_saved_kg=co2_saved,
                    lat=lat, lon=lon, notify=notify
                )
                st.session_state.db = pd.concat([st.session_state.db, pd.DataFrame([row])], ignore_index=True)
                st.success("Logged! Check the map & dashboard tabs.")

        with col2:
            m2 = folium.Map(location=[lat, lon], zoom_start=15, tiles="cartodbpositron")
            color = ("blue" if status == "Deadwood" and pH_ok
                     else "orange" if status == "Pest Damage"
                     else "yellow" if not pH_ok
                     else "green")
            folium.CircleMarker([lat, lon], radius=12, color=color, fill=True, fill_color=color,
                                popup=f"{status} @ {picked_label or ''}").add_to(m2)
            st_folium(m2, width=900, height=420)

    else:
        st.info("先選擇地點（任一模式），地圖將自動移動到該位置。")

# ============================
# Tab 3: Dashboard
# ============================
with tab_dashboard:
    st.subheader("Reports & Metrics")
    df = st.session_state.db.copy()
    if len(df) == 0:
        st.info("No data yet — submit a report in the 'Report Tree' tab.")
    else:
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            f_city = st.selectbox("Filter City", ["(All)"] + sorted(df["city"].dropna().unique().tolist()))
        with colB:
            if f_city != "(All)":
                f_district = st.selectbox(
                    "Filter District",
                    ["(All)"] + sorted(df[df["city"] == f_city]["district"].dropna().unique().tolist())
                )
            else:
                f_district = "(All)"
        with colC:
            f_status = st.selectbox("Filter Status", ["(All)"] + CLASSES)

        if f_city != "(All)":
            df = df[df["city"] == f_city]
        if f_district != "(All)":
            df = df[df["district"] == f_district]
        if f_status != "(All)":
            df = df[df["status"] == f_status]

        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cases", len(df))
        with c2:
            st.metric("CO₂ saved (sum)", f"{float(df['co2_saved_kg'].sum()):.1f} kg")
        with c3:
            ph_ok_rate = (df["soil_pH"].astype(float).between(5.8, 7.5)).mean()*100 if len(df) else 0
            st.metric("pH in ideal range", f"{ph_ok_rate:.0f}%")

# ============================
# Tab 4: Government View
# ============================
with tab_gov:
    st.subheader("Summary for Municipal Action")
    df = st.session_state.db.copy()
    if len(df) == 0:
        st.info("No data yet — submit a report in the 'Report Tree' tab.")
    else:
        grp = df.groupby(["city", "district", "status"]).agg(
            cases=("status", "count"),
            co2=("co2_saved_kg", "sum")
        ).reset_index().sort_values(["city", "district", "cases"], ascending=[True, True, False])
        st.write("Hotspots by district & status")
        st.dataframe(grp, use_container_width=True, hide_index=True)

        st.write("⚠️ Priority cases (auto-notify rules)")
        q = df[df["notify"] == True]
        st.dataframe(
            q[["ts", "city", "district", "tree_id", "status", "diameter_cm", "soil_pH", "lat", "lon"]],
            use_container_width=True, hide_index=True
        )

st.caption("Prototype: dosages & CO₂ are illustrative. Field work should follow local forestry guidelines.")
