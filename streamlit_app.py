import streamlit as st
import streamlit as st, pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(page_title="Seed of Tomorrow", layout="wide")

# ---------- HEADER ----------
st.title("🌱 Seed of Tomorrow — AI + Fungi Net-Zero System")
st.caption("Team HengJie · Prototype simulator for competition demo")

# ---------- INPUTS ----------
left, right = st.columns([1,1], gap="large")

with left:
    st.subheader("1) Case Intake")
    status = st.selectbox("Predicted status (simulate AI):", ["deadwood","pest","healthy"], index=0)
    D = st.slider("Trunk diameter (cm)", 5, 120, 32)
    pH = st.slider("Soil pH", 3.0, 9.5, 6.2, 0.1)

    near_walkway = st.checkbox("Near sidewalk (<2 m)?", value=True)
    cluster_reports = st.number_input("Reports in 200 m (last 7 days)", 0, 20, 0)
    lat = st.number_input("Latitude", value=24.150000, format="%.6f")
    lon = st.number_input("Longitude", value=120.670000, format="%.6f")

# ---------- RULES & CALCS ----------
def diameter_class(d):
    return "S" if d<20 else ("M" if d<=40 else "L")

def materials_calc(status, D):
    cls = diameter_class(D)
    if status == "deadwood":
        factor = {"S":0.04,"M":0.06,"L":0.08}[cls]
        substrate_kg = round(factor * D, 2)
        spore_L = round(substrate_kg * 0.5, 2)
        return "inoculate_fungi", substrate_kg, spore_L, 0.0, 25.0
    if status == "pest":
        factor = {"S":0.10,"M":0.15,"L":0.20}[cls]
        spray_L = round(factor * D, 2)
        return "apply_biocontrol_fungi", 0.0, 0.0, spray_L, 0.0
    return "monitor", 0.0, 0.0, 0.0, 0.0

def amendment_calc(pH, D):
    lime_kg = round(max(0.0, 0.02 * D * (5.8 - pH)), 2) if pH < 5.8 else 0.0
    sulfur_kg = round(max(0.0, 0.004 * D * (pH - 7.5)), 2) if pH > 7.5 else 0.0
    return lime_kg, sulfur_kg

def should_notify(status, D, near_walkway, cluster_reports, pH):
    if status=="deadwood" and D>40 and near_walkway: return True
    if status=="pest" and cluster_reports>=3: return True
    if pH<4.5 or pH>8.5: return True
    return False

# pH gate & amendments
lime_kg, sulfur_kg = amendment_calc(pH, D)
pH_ok = (5.8 <= pH <= 7.5)

# action + materials
action, substrate_kg, spore_L, spray_L, co2_saved = materials_calc(status, D)

# color states
# Red = deadwood untreated, Orange = pest untreated, Yellow = pH correction pending,
# Blue = treated today, Green = recovered/monitor
if not pH_ok:
    state_color = "yellow"
elif status == "deadwood":
    state_color = "blue"
elif status == "pest":
    state_color = "orange"
elif status == "healthy":
    state_color = "green"
else:
    state_color = "green"

notify = should_notify(status, D, near_walkway, cluster_reports, pH)

# ---------- UI OUTPUTS ----------
with left:
    st.subheader("2) Decision Support")
    if not pH_ok:
        st.warning("⚠️ pH not ideal — correct soil before applying fungi.")
        if lime_kg: st.write(f"• Suggest **lime**: {lime_kg} kg (prototype estimate)")
        if sulfur_kg: st.write(f"• Suggest **elemental sulfur**: {sulfur_kg} kg (prototype estimate)")
    else:
        st.success(f"✅ pH suitable. Recommended action: **{action}**")
        if substrate_kg: st.write(f"• Substrate: **{substrate_kg} kg**")
        if spore_L:     st.write(f"• Spore solution: **{spore_L} L**")
        if spray_L:     st.write(f"• Biocontrol spray: **{spray_L} L**")
        if co2_saved:   st.metric("CO₂ saved (this case)", f"{co2_saved:.1f} kg")

    if notify:
        st.error("建議通報主管機關 / Notify Government recommended")

# lightweight in-memory “DB”
if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame(columns="""
        ts tree_id status diameter_cm soil_pH action substrate_kg spore_L spray_L
        co2_saved_kg lat lon notify state_color
    """.split())

if st.button("➕ Log this case"):
    row = dict(
        ts=datetime.utcnow().isoformat(timespec="seconds"),
        tree_id=f"T-{len(st.session_state.db)+1:03d}",
        status=status, diameter_cm=D, soil_pH=pH, action=action,
        substrate_kg=substrate_kg, spore_L=spore_L, spray_L=spray_L,
        co2_saved_kg=co2_saved, lat=lat, lon=lon,
        notify=notify, state_color=state_color
    )
    st.session_state.db = pd.concat([st.session_state.db, pd.DataFrame([row])], ignore_index=True)
    st.success("Logged! See dashboard & map below.")

with right:
    st.subheader("3) Dashboard")
    db = st.session_state.db
    st.metric("Total cases", len(db))
    st.metric("Total CO₂ saved", f"{float(db['co2_saved_kg'].sum()):.1f} kg" if len(db) else "0.0 kg")
    if len(db):
        st.dataframe(db[["tree_id","status","action","soil_pH","co2_saved_kg","state_color","notify"]], use_container_width=True)

st.markdown("---")
st.subheader("4) Map Simulation")

center = [24.1500, 120.6700]
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

# legend colors
color_map = {"deadwood":"red","pest":"orange","healthy":"green","yellow":"yellow","blue":"blue","green":"green"}
# plot current point (big)
folium.CircleMarker(center, radius=12, color=state_color, fill=True, fill_color=state_color,
                    popup=f"status:{status}, pH:{pH}").add_to(m)

# plot logged points (small)
if len(st.session_state.db):
    for _, r in st.session_state.db.iterrows():
        c = color_map.get(r["state_color"], "green")
        folium.CircleMarker([float(r["lat"]), float(r["lon"])], radius=6, color=c,
                            fill=True, fill_color=c,
                            popup=f"{r['tree_id']}: {r['status']} → {r['action']} ({r['co2_saved_kg']} kg)").add_to(m)
st_folium(m, width=900, height=450)

st.caption("Legend — Red: Deadwood (untreated) · Orange: Pest (untreated) · Yellow: pH correction pending · Blue: Treated · Green: Recovered/Monitoring")
st.markdown("> Prototype simulation. Amendment and dosage are demo estimates; field application follows local guidelines.")
