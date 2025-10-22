# =========================================================
# 🌱 Seed of Tomorrow /未來之種：AI+真菌 淨零城市治理平台
# =========================================================
# 通用樹種分析（General）+ 本次聚焦（Specialization / 案例：黑板樹）
# - 影像分析（雲端安全：Pillow+NumPy 啟發式，不用 OpenCV/Torch）
# - 物種模式（通用/聚焦）＋ 樹種資料庫（常見病害與建議真菌）
# - 生物資材知識庫（混配禁忌、輪替規則、施用注意）
# - 劑量/工時估算（依胸徑/樹冠/嚴重度）
# - 地理輸入：地址搜尋 / 地圖點選 / 手動座標
# - OSM 樹點/建物疊加、北屯示範樹列
# - 儀表板 + 政府端總覽
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import requests
import folium
from streamlit_folium import st_folium
import overpy
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG / 頁面設定
# ---------------------------------------------------------
st.set_page_config(page_title="Seed of Tomorrow / 未來之種", page_icon="🌱", layout="wide")
st.title("🌱 Seed of Tomorrow / 未來之種")
st.caption("衡界｜AI × 真菌 × 公民科學｜AI + Mycology + Citizen Science for Climate-Ready Cities")

# ---------------------------------------------------------
# CLASS LABELS / 類別
# ---------------------------------------------------------
CLASSES_EN = ["Healthy", "Deadwood", "Pest Damage", "Soil Issue"]
CLASSES_ZH = ["健康", "枯木", "蟲害", "土壤問題"]
CLASS_BI = dict(zip(CLASSES_EN, CLASSES_ZH))

# ---------------------------------------------------------
# SPECIES LIBRARY / 樹種資料（通用＋本次聚焦）
# ---------------------------------------------------------
SPECIES_LIBRARY = {
    "Blackboard Tree (Alstonia scholaris) / 黑板樹": {
        "key": "blackboard",
        "diseases": [
            ("Root Rot (Fungal) / 根腐病", "Trichoderma harzianum", "Soil drench + mulch / 土壤灌注＋覆蓋"),
            ("Scale Insects / 介殼蟲", "Beauveria bassiana", "Targeted biological spray / 生物性噴施"),
            ("Drought/Soil Stress / 乾旱/土壤逆境", "Mycorrhizae", "Inoculation + organic matter / 接種＋有機質")
        ],
        # bias 僅在 Specialization 模式下微調分類傾向（近界值時）
        "bias": {"Pest Damage": 0.08, "Deadwood": 0.04, "Soil Issue": 0.03}
    },
    "Banyan (Ficus microcarpa) / 榕樹": {
        "key": "banyan",
        "diseases": [
            ("Gall Wasp / 瘤蜂", "Beauveria", "Prune + bio-spray"),
            ("Root Compression / 根系受壓", "Mycorrhizae", "Soil decompaction"),
            ("Sooty Mold / 煤污病", "Beauveria", "Insect control + wash")
        ],
        "bias": {"Pest Damage": 0.06, "Soil Issue": 0.05}
    },
    "Formosan Sweetgum (Liquidambar formosana) / 楓香": {
        "key": "sweetgum",
        "diseases": [
            ("Leaf Spot / 葉斑病", "Trichoderma", "Prune + sanitation"),
            ("Borer Damage / 天牛蛀孔", "Beauveria", "Targeted bio-spray"),
            ("Drought Stress / 乾旱逆境", "Mycorrhizae", "Mulch + inoculation")
        ],
        "bias": {"Pest Damage": 0.05, "Soil Issue": 0.05}
    }
}

def species_notes_md(species_name: str) -> str:
    items = SPECIES_LIBRARY.get(species_name, {}).get("diseases", [])
    if not items:
        return "_(No disease notes yet / 尚無資料)_"
    lines = []
    for name, fungi, action in items:
        lines.append(f"- **{name}** — Fungi: **{fungi}** ｜ Action: **{action}**")
    return "\n".join(lines)

# ---------------------------------------------------------
# BIO-AGENTS KNOWLEDGE BASE / 生物資材知識庫（混配禁忌/輪替/注意）
# ---------------------------------------------------------
BIO_AGENTS = {
    "Trichoderma": {
        "zh": "木黴菌（Trichoderma）",
        "target": ["Deadwood","Soil Issue","Root Rot"],
        "form": "Soil drench / 土壤灌注",
        "base_l_per_tree": 3.0,   # 以 30cm 胸徑、8m 樹冠為基準
        "mix_avoid": ["Bacillus", "EM", "Copper", "Strong-alkali"],
        "notes": "Avoid waterlogging; mulch after drench / 避免積水，灌後覆蓋有機質"
    },
    "AMF": {
        "zh": "菌根菌（AMF）",
        "target": ["Soil Issue","Drought Stress"],
        "form": "Inoculation + compost / 接種＋有機質",
        "base_l_per_tree": 2.0,
        "mix_avoid": ["Copper", "Strong-alkali"],
        "notes": "Loosen soil; water-in / 鬆土成環並充分灌水"
    },
    "Beauveria": {
        "zh": "白僵菌（Beauveria bassiana）",
        "target": ["Pest Damage"],
        "form": "Targeted spray / 目標噴施",
        "base_ml_per_tree": 200,
        "rotation": ["Metarhizium"],
        "mix_avoid": ["Bacillus","EM","Copper","Strong-alkali"],
        "notes": "Evening spray; keep humidity / 傍晚噴，保持濕度"
    },
    "Metarhizium": {
        "zh": "綠僵菌（Metarhizium anisopliae）",
        "target": ["Pest Damage"],
        "form": "Targeted spray / 目標噴施",
        "base_ml_per_tree": 220,
        "rotation": ["Beauveria"],
        "mix_avoid": ["Bacillus","EM","Copper","Strong-alkali"],
        "notes": "Evening spray; avoid direct sun / 傍晚噴，避烈日"
    },
    "Bacillus": {
        "zh": "枯草桿菌（Bacillus subtilis）",
        "target": ["Leaf Fungus","General Sanitation"],
        "form": "Foliage spray / 葉面噴施",
        "base_ml_per_tree": 180,
        "mix_avoid": ["Trichoderma","AMF","Beauveria","Metarhizium","EM","Copper","Strong-alkali"],
        "notes": "Do not co-apply with live fungi / 不與活性真菌同時使用"
    },
    "EM": {
        "zh": "EM 複合菌",
        "target": ["Soil Conditioner","Decomposition"],
        "form": "Soil drench / 土壤灌注",
        "base_l_per_tree": 2.5,
        "mix_avoid": ["Copper","Strong-alkali","Beauveria","Metarhizium"],
        "notes": "Conditioner; not a specific pesticide / 土壤調理，非專一防治"
    }
}

def choose_agent(pred_label_en: str):
    if pred_label_en in ["Deadwood","Soil Issue"]:
        return "Trichoderma" if pred_label_en=="Deadwood" else "AMF"
    if pred_label_en == "Pest Damage":
        return "Beauveria"   # 可在 UI 顯示輪替 Metarhizium
    return None

# ---------------------------------------------------------
# TAIWAN city -> districts / 台灣城市與行政區
# ---------------------------------------------------------
TAIWAN_DIVISIONS = {
    "Taipei City": ["Zhongzheng","Datong","Zhongshan","Songshan","Da’an","Wanhua","Xinyi","Shilin","Beitou","Neihu","Nangang","Wenshan"],
    "New Taipei City": ["Banqiao","Sanchong","Zhonghe","Yonghe","Xinzhuang","Xindian","Tucheng","Luzhou","Xizhi","Shulin","Yingge","Sanxia","Danshui","Ruifang","Taishan","Wugu","Linkou","Bali","Pingxi","Shuangxi","Gongliao","Jinshan","Wanli","Shenkeng","Shiding","Pinglin","Sanzhi","Shimen","Ulay"],
    "Taichung City": ["Central","East","South","West","North","Beitun","Xitun","Nantun","Taiping","Dali","Wufeng","Wuri","Fengyuan","Dongshi","Heping","Tanzi","Daya","Shengang","Dadu","Shalu","Longjing","Wuqi","Qingshui","Dajia","Waipu","Da’an"],
    "Tainan City": ["West Central","East","South","North","Anping","Annan","Yongkang","Guiren","Xinshih","Rende","Guanmiao","Longqi","Guantian","Madou","Jiali","Xuejia","Beimen","Xigang","Qigu","Jiangjun","Xinhua","Shanshang","Zuozhen","Yujing","Nanxi","Nanhua","Anding","Liuying","Liujia","Dongshan","Baihe","Houbi"],
    "Kaohsiung City": ["Xinxing","Qianjin","Lingya","Yancheng","Gushan","Qijin","Qianzhen","Sanmin","Nanzi","Xiaogang","Zuoying","Renwu","Dashe","Gangshan","Luzhu","Alian","Tianliao","Yanchao","Qiaotou","Ziguan","Mituo","Yong’an","Hunei","Fengshan","Daliao","Linyuan","Niaosong","Dashu","Qishan","Meinong","Liugui","Neimen","Shanlin","Jiaxian","Taoyuan","Namaxia","Maolin","Qieding"],
    "Taoyuan City": ["Taoyuan","Zhongli","Pingzhen","Bade","Yangmei","Luzhu","Guishan","Longtan","Dayuan","Xinwu","Guanyin","Fuxing"],
    "Hsinchu City": ["East","North","Xiangshan"],
    "Keelung City": ["Ren’ai","Xinyi","Zhongzheng","Zhongshan","Anle","Nuannuan","Qidu"],
    "Hsinchu County": ["Zhubei","Zhudong","Xinpu","Hukou","Xinfeng","Guansi","Qionglin","Baoshan","Beipu","Emei","Jianshi","Wufeng"],
    "Miaoli County": ["Miaoli","Toufen","Zhunan","Houlong","Tongxiao","Tongluo","Sanyi","Zaoqiao","Yuanli","Gongguan","Touwu","Nanzhuang","Shitan","Dahu","Sanwan","Tai’an"],
    "Changhua County": ["Changhua","Yuanlin","Lukang","Hemei","Fuxing","Xiushui","Huatan","Fenyuan","Dacun","Puyan","Xihu","Pitou","Beidou","Ershui","Tianzhong","Tianwei","Xizhou","Fangyuan","Dacheng","Erlin"],
    "Nantou County": ["Nantou","Caotun","Zhushan","Jiji","Puli","Yuchi","Guoxing","Ren’ai","Mingjian","Lugu","Shuili","Xinyi"],
    "Yunlin County": ["Douliu","Huwei","Dounan","Erlun","Lunbei","Mailiao","Taixi","Sihu","Kouhu","Tuku","Baozhong","Dongshi","Shuilin","Gukeng","Citong","Linnei","Beigang"],
    "Chiayi City": ["East","West"],
    "Chiayi County": ["Taibao","Puzi","Minxiong","Xikou","Dalun","Budai","Dongshi","Liujiao","Yizhu","Lucao","Zhongpu","Fanlu","Alishan","Meishan","Shuishang"],
    "Yilan County": ["Yilan","Jiaoxi","Toucheng","Luodong","Su’ao","Nan’ao","Dongshan","Sanxing","Datong","Zhuangwei","Wujie"],
    "Hualien County": ["Hualien","Ji’an","Shoufeng","Fuli","Yuli","Fenglin","Fengbin","Ruisui","Guangfu","Zhuoxi","Xincheng"],
    "Taitung County": ["Taitung","Luye","Beinan","Guanshan","Chishang","Taimali","Dawu","Chenggong","Changbin","Donghe","Haiduan","Lanyu","Ludao"],
    "Pingtung County": ["Pingtung","Chaozhou","Donggang","Fangliao","Hengchun","Manzhou","Checheng","Mudan","Sandimen","Wutai","Majia","Taiwu","Laiyi","Linbian","Nanzhou","Jiadong","Xinyuan","Kanding","Ligang","Gaoshu","Wanluan"],
    "Penghu County": ["Magong","Huxi","Baisha","Xiyu","Wang’an","Qimei"],
    "Kinmen County": ["Jincheng","Jinsha","Jinning","Jinhu","Lieyu","Wuqiu"],
    "Lienchiang County": ["Nangan","Beigan","Juguang","Dongyin"]
}

# ---------------------------------------------------------
# SESSION DB / 內部資料庫
# ---------------------------------------------------------
if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "timestamp","reporter","city","district","tree_id","status_en","status_zh","confidence",
        "diameter_cm","canopy_m","action_bi","co2_saved_kg","treated","treated_ts","lat","lon",
        "species"
    ])

# ---------------------------------------------------------
# GEO UTILS / 地理工具
# ---------------------------------------------------------
def geocode_address(query: str):
    """Nominatim 地理編碼 / Geocode -> (lat, lon, display_name)"""
    if not query or not query.strip():
        return None, None, None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "Seed-of-Tomorrow-demo"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        return None, None, None

@st.cache_data(show_spinner=False, ttl=3600)
def get_bbox_for(city: str, district: str):
    """Auto-bbox via Nominatim；若失敗回退西屯區範圍"""
    q = f"{district}, {city}, Taiwan"
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "Seed-of-Tomorrow-demo"}
    params = {"q": q, "format": "json", "limit": 1}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        if data and "boundingbox" in data[0]:
            south, north, west, east = map(float, data[0]["boundingbox"])
            return (south, west, north, east)
    except Exception:
        pass
    try:
        params["q"] = f"{city}, Taiwan"
        r2 = requests.get(url, params=params, headers=headers, timeout=12)
        r2.raise_for_status()
        d2 = r2.json()
        if d2 and "boundingbox" in d2[0]:
            south, north, west, east = map(float, d2[0]["boundingbox"])
            return (south, west, north, east)
    except Exception:
        pass
    return (24.168, 120.595, 24.206, 120.640)  # Taichung Xitun fallback

@st.cache_data(show_spinner=False, ttl=600)
def fetch_osm_trees_buildings(bbox):
    """Overpass 抓取 OSM 樹點與建物"""
    south, west, north, east = bbox
    api = overpy.Overpass()
    q_trees = f'node["natural"="tree"]({south},{west},{north},{east}); out center 200;'
    q_build = f'way["building"]({south},{west},{north},{east}); (._;>;); out body 200;'
    trees = api.query(q_trees)
    builds = api.query(q_build)
    building_polys = []
    for way in builds.ways:
        coords = [(float(n.lat), float(n.lon)) for n in way.nodes]
        if len(coords) >= 3:
            building_polys.append(coords)
    return trees.nodes, building_polys

def add_tile_layers(m):
    """Base maps with explicit URLs + attributions（避免 folium 無歸屬報錯）"""
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OSM Standard", attr="© OpenStreetMap contributors"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="Carto Light", attr="© OpenStreetMap contributors, © CartoDB"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
        name="Terrain", attr="Map tiles by Stamen Design (CC BY 3.0) — Data © OpenStreetMap"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap", attr="© OpenTopoMap (CC-BY-SA) — © OpenStreetMap"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        name="Dark Mode", attr="© OpenStreetMap contributors, © CartoDB"
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ---------------------------------------------------------
# AI HEURISTIC / 影像啟發式（可通用；Specialization 僅微調）
# ---------------------------------------------------------
def pseudo_ai_predict(image: Image.Image, species_name: str, mode: str):
    """
    Prototype heuristic (color/brightness) → 4 classes.
    General: species-independent rules.
    Specialization: small species bias near boundary decisions.
    Returns: (label_en, confidence%, probs_dict)
    """
    img = image.convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32)
    mean = arr.mean(axis=(0, 1))  # R,G,B
    brightness = arr.mean()
    red, green, blue = mean
    rg = red - green
    gb = green - blue

    # base rules
    high_green = green > 140 and gb > 8 and brightness > 115
    reddish    = rg > 15 and red > 120
    very_dark  = brightness < 85
    mid_wash   = 85 <= brightness <= 110 and abs(rg) < 12

    if high_green:
        pred, base_conf = "Healthy", 0.975
    elif very_dark:
        pred, base_conf = "Deadwood", 0.965
    elif reddish:
        pred, base_conf = "Pest Damage", 0.962
    elif mid_wash:
        pred, base_conf = "Soil Issue", 0.955
    else:
        pred, base_conf = "Soil Issue", 0.925

    # species bias only if specialization mode
    if mode.startswith("Specialization"):
        bias = SPECIES_LIBRARY.get(species_name, {}).get("bias", {})
        if base_conf < 0.965:
            if pred in bias:
                base_conf = min(0.99, base_conf + min(0.01, bias[pred]))
            else:
                for favored, bump in bias.items():
                    if bump >= 0.06 and base_conf < 0.94:
                        pred = favored
                        base_conf = 0.945
                        break

    conf = float(np.clip(np.random.normal(base_conf, 0.008), 0.90, 0.99)) * 100.0
    probs = {c: 0.0 for c in CLASSES_EN}
    probs[pred] = round(conf/100.0, 3)
    rem = max(0.0, 1.0 - probs[pred])
    for c in CLASSES_EN:
        if c != pred:
            probs[c] = round(rem / 3.0, 3)
    return pred, round(conf, 2), probs

# ---------------------------------------------------------
# DOSAGE & WORKLOAD / 劑量與工時估算（含混配/輪替提示）
# ---------------------------------------------------------
def estimate_treatment_plus(pred_label_en: str, diameter_cm: int, canopy_m: int, severity_pct: float):
    agent_key = choose_agent(pred_label_en)
    if not agent_key:
        return {
            "agent": None, "agent_zh": "—", "form": "—",
            "liters_to_drench": 0, "spray_ml": 0, "labor_minutes": 5,
            "notes": "Recheck in 2–4 weeks / 2–4 週後複查",
            "warnings": []
        }

    ag = BIO_AGENTS[agent_key]
    stem_factor = max(0.8, min(2.5, diameter_cm / 30.0))
    canopy_factor = max(0.7, min(2.0, canopy_m / 8.0))
    sev = max(0.0, min(1.0, (severity_pct - 90.0) / 9.0))
    severity_factor = 1.0 + 0.35 * sev

    dose_l = ag.get("base_l_per_tree", 0) * stem_factor * canopy_factor * severity_factor
    dose_ml = ag.get("base_ml_per_tree", 0) * canopy_factor * severity_factor
    labor = int((20 if ag["form"].startswith("Soil") else 15) * canopy_factor)

    warnings = []
    if ag.get("mix_avoid"):
        warnings.append("Avoid mixing with: " + ", ".join(ag["mix_avoid"]))
    if ag.get("rotation") and pred_label_en == "Pest Damage":
        warnings.append("Rotation option: " + " / ".join(ag["rotation"]))

    return {
        "agent": agent_key,
        "agent_zh": ag["zh"],
        "form": ag["form"],
        "liters_to_drench": round(dose_l, 1) if dose_l else 0,
        "spray_ml": int(dose_ml) if dose_ml else 0,
        "labor_minutes": labor,
        "notes": ag["notes"],
        "warnings": warnings
    }

# ---------------------------------------------------------
# DEMO GENERATOR / 北屯樹列示範
# ---------------------------------------------------------
def _line_points(lat0, lon0, lat1, lon1, n, jitter=0.00025, seed=42):
    rng = np.random.default_rng(seed)
    lats = np.linspace(lat0, lat1, n) + rng.normal(0, jitter, n)
    lons = np.linspace(lon0, lon1, n) + rng.normal(0, jitter, n)
    return list(zip(lats, lons))

def make_beitun_demo(n_per_line=80, seed=7):
    A = _line_points(24.1825, 120.7050, 24.2025, 120.7055, n_per_line, seed=seed)
    B = _line_points(24.1690, 120.7010, 24.1950, 120.6940, n_per_line, seed=seed+1)
    C = _line_points(24.1665, 120.6900, 24.1885, 120.6780, n_per_line, seed=seed+2)
    pts = A + B + C
    rng = np.random.default_rng(seed+10)

    labels = rng.choice(["Healthy","Pest Damage","Deadwood","Soil Issue"],
                        size=len(pts), p=[0.62,0.20,0.08,0.10])
    confidences = rng.uniform(92, 99, len(pts)).round(2)
    diameters = rng.integers(10, 85, len(pts))
    canopies = rng.integers(3, 18, len(pts))
    treated_mask = rng.random(len(pts)) < 0.35
    co2_map = {"Deadwood":25, "Pest Damage":20, "Soil Issue":15, "Healthy":0}
    now = datetime.utcnow().isoformat(timespec="seconds")

    rows = []
    for i, (lat, lon) in enumerate(pts):
        lab_en = labels[i]
        lab_zh = CLASS_BI[lab_en]
        # 為了對齊 UI 的舊顯示，仍保留 action_bi/CO2 tag（演示用）
        legacy_action = {
            "Healthy": "監測即可 / Monitoring only",
            "Deadwood": "施用腐生真菌 / Apply saprotrophic fungi",
            "Pest Damage": "施用昆蟲病原真菌 / Apply entomopathogenic fungi",
            "Soil Issue": "施用菌根真菌 + 緩衝處理 / Apply mycorrhizal fungi + buffer"
        }[lab_en]
        co2_saved = co2_map[lab_en] if treated_mask[i] else 0

        rows.append(dict(
            timestamp=now,
            reporter="DemoSeed",
            city="Taichung City",
            district="Beitun",
            tree_id=f"BT-{10000+i}",
            status_en=lab_en, status_zh=lab_zh,
            confidence=float(confidences[i]),
            diameter_cm=int(diameters[i]), canopy_m=int(canopies[i]),
            action_bi=legacy_action, co2_saved_kg=int(co2_saved),
            treated=bool(treated_mask[i]),
            treated_ts=now if treated_mask[i] else "",
            lat=float(lat), lon=float(lon),
            species="(demo) case-study ready"
        ))
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# TABS / 分頁
# ---------------------------------------------------------
tabs = st.tabs([
    "📷 Analyze Tree / 影像分析",
    "🗺️ Map View / 地圖檢視",
    "📊 Dashboard / 儀表板",
    "🏛️ Government Summary / 政府端總覽"
])

# =========================================================
# TAB 1 — Analyze Tree / 影像分析
# =========================================================
with tabs[0]:
    st.header("📷 AI Tree Health Analyzer / 樹木健康AI分析")
# ---------------------------------------------------------
# SPECIES MODE / 樹種模式
# ---------------------------------------------------------
st.markdown("### 🌳 Species Mode / 樹種模式")
colS1, colS2 = st.columns([1.2, 2])
with colS1:
    species_mode = st.radio(
        "Mode / 模式",
        ["General (all trees)", "Specialization (case study)"],
        index=0
    )

# In General mode, let user optionally type a species name (for logging only)
if species_mode.startswith("General"):
    species_name = st.text_input(
        "Species (optional) / 樹種（可留白）",
        value="",
        placeholder="Unknown / 未標示"
    )
    st.caption("General mode uses species-agnostic analysis. 樹種僅作為紀錄用。")
else:
    species_name = st.selectbox(
        "Species / 樹種",
        list(SPECIES_LIBRARY.keys()),
        index=0
    )
    st.expander("📚 Species notes / 樹種說明", expanded=False).markdown(
        species_notes_md(species_name)
    )

    # 左右欄
    col_left, col_right = st.columns([3, 2])
    with col_left:
        uploaded = st.file_uploader("Upload tree photo (JPG/PNG) / 上傳樹木照片", type=["jpg","jpeg","png"])
        diameter = st.slider("Trunk Diameter (cm) / 樹胸徑(公分)", 5, 150, 35)
        canopy_m = st.slider("Canopy Width (m) / 樹冠幅(公尺)", 1, 30, 8)
        reporter = st.text_input("Reporter name / 回報者", "Citizen")

    with col_right:
        city = st.selectbox("City / 城市", list(TAIWAN_DIVISIONS.keys()), index=2)
        district = st.selectbox("District / 行政區", TAIWAN_DIVISIONS[city])

    # 位置設定
    st.markdown("### 📍 Location / 座標設定")
    bbox = get_bbox_for(city, district)
    center_lat = (bbox[0] + bbox[2]) / 2
    center_lon = (bbox[1] + bbox[3]) / 2

    loc_mode = st.radio("Choose location input mode / 位置設定方式",
                        ["Search Address / 地址搜尋", "Click on Map / 地圖點選", "Enter lat/lon / 手動輸入座標"],
                        horizontal=True)

    lat, lon = center_lat, center_lon
    picked_label = ""

    if loc_mode.startswith("Search"):
        q = st.text_input("Address / 地址（可含地標）")
        if st.button("🔎 Geocode / 地理編碼"):
            glat, glon, name = geocode_address(q)
            if glat and glon:
                lat, lon, picked_label = glat, glon, name
                st.success(f"Found / 已定位：{name} ({lat:.6f}, {lon:.6f})")
            else:
                st.error("No match found / 找不到結果，請加上行政區與城市再試")
    elif loc_mode.startswith("Click"):
        st.caption("Click the map to place a pin / 點地圖放置座標")
        pick_map = folium.Map(location=[center_lat, center_lon], zoom_start=15, control_scale=True)
        add_tile_layers(pick_map)
        pick_out = st_folium(pick_map, width=920, height=420)
        if pick_out and pick_out.get("last_clicked"):
            lat = float(pick_out["last_clicked"]["lat"])
            lon = float(pick_out["last_clicked"]["lng"])
            picked_label = f"Picked / 已選：{lat:.6f}, {lon:.6f}"
            st.success(picked_label)
    else:
        col_ll1, col_ll2 = st.columns(2)
        with col_ll1:
            lat = st.number_input("Latitude / 緯度", value=center_lat, format="%.6f")
        with col_ll2:
            lon = st.number_input("Longitude / 經度", value=center_lon, format="%.6f")
        picked_label = f"Manual / 手動：{lat:.6f}, {lon:.6f}"

    st.write(f"**Location set to / 目前座標：** {picked_label or f'{lat:.6f}, {lon:.6f}'}")

    # 影像分析 → 劑量估算 → 記錄
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Tree / 已上傳照片", use_container_width=True)

        label_en, conf, probs = pseudo_ai_predict(img, species_name, species_mode)
        label_zh = CLASS_BI[label_en]
        st.success(f"Prediction 預測：**{label_zh} / {label_en}**（Confidence 信心：{conf:.2f}%）")
        st.bar_chart(pd.DataFrame({"Confidence / 信心": list(probs.values())},
                                  index=[f"{CLASS_BI[k]} / {k}" for k in probs.keys()]))

        dose = estimate_treatment_plus(label_en, diameter, canopy_m, conf)
        st.subheader("Treatment Estimator / 處置劑量估算")
        st.caption(f"Species / 樹種：{species_name}  ｜ Mode / 模式：{species_mode}")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Agent / 資材", dose["agent_zh"])
        cB.metric("Soil Drench (L)", dose["liters_to_drench"])
        cC.metric("Spray (mL)", dose["spray_ml"])
        cD.metric("Labor (min)", dose["labor_minutes"])
        st.caption(dose["notes"])
        for w in dose["warnings"]:
            st.warning(w)

        # 舊版 CO2 標籤保留（看起來有減碳指標）
        legacy_action = {
            "Healthy": "監測即可 / Monitoring only",
            "Deadwood": "施用腐生真菌 / Apply saprotrophic fungi",
            "Pest Damage": "施用昆蟲病原真菌 / Apply entomopathogenic fungi",
            "Soil Issue": "施用菌根真菌 + 緩衝處理 / Apply mycorrhizal fungi + buffer"
        }[label_en]
        co2_map = {"Deadwood":25, "Pest Damage":20, "Soil Issue":15, "Healthy":0}
        co2_saved = co2_map[label_en]
        st.info(f"Legacy Action Tag / 舊版建議：**{legacy_action}**")
        st.metric("CO₂ Reduction Potential / 潛在減碳", f"{co2_saved} kg")

        apply_now = st.checkbox("Apply treatment now (simulate) / 立即模擬施作", value=False)

        if st.button("➕ Log this tree / 記錄此樹"):
            row = dict(
                timestamp=datetime.utcnow().isoformat(timespec="seconds"),
                reporter=reporter, city=city, district=district,
                tree_id=f"T-{np.random.randint(1000,9999)}",
                status_en=label_en, status_zh=label_zh, confidence=conf,
                diameter_cm=diameter, canopy_m=canopy_m,
                action_bi=f"{dose['agent_zh']}｜{dose['form']}",
                co2_saved_kg=co2_saved if apply_now else 0,
                treated=apply_now, treated_ts=datetime.utcnow().isoformat(timespec="seconds") if apply_now else "",
                lat=lat, lon=lon,
                species=species_name if species_mode.startswith("Special") else "(general)"
            )
            st.session_state.db = pd.concat([st.session_state.db, pd.DataFrame([row])], ignore_index=True)
            st.balloons()
            st.success("Logged successfully / 已記錄！前往地圖與儀表板查看。")

# =========================================================
# TAB 2 — Map View / 地圖檢視
# =========================================================
with tabs[1]:
    st.header("🗺️ Map View / 地圖檢視（Demo Logs vs Live OSM）")

    mcol1, mcol2, mcol3 = st.columns([2, 2, 2])
    with mcol1:
        city_m = st.selectbox("City / 城市（地圖）", list(TAIWAN_DIVISIONS.keys()), index=2, key="map_city")
    with mcol2:
        district_m = st.selectbox("District / 行政區（地圖）", TAIWAN_DIVISIONS[city_m], key="map_dist")
    with mcol3:
        data_mode = st.radio("Data source / 資料來源", ["Demo logs / 示範紀錄", "Live OSM / 即時OSM"], index=0)

    st.caption("Note：System analyzes all species; today’s case-study demo is Beitun street trees. / 本系統可分析所有樹種；示範資料為北屯行道樹案例。")
    
if species_mode.startswith("Specialization"):
    st.caption(f"🔬 Specialization mode active: focusing on {species_name}")
else:
    st.caption("🌍 General mode: analyzing all trees across Taiwan")
    
    # 一鍵載入北屯示範
    if st.button("⚡ Load Beitun demo tree lines / 載入北屯樹列示範"):
        demo_df = make_beitun_demo(n_per_line=80, seed=7)
        st.session_state.db = pd.concat([st.session_state.db, demo_df], ignore_index=True)
        st.success(f"Loaded / 已載入：{len(demo_df)} 筆北屯樹列示範資料")

    bbox_m = get_bbox_for(city_m, district_m)
    c_lat = (bbox_m[0] + bbox_m[2]) / 2
    c_lon = (bbox_m[1] + bbox_m[3]) / 2

    m = folium.Map(location=[c_lat, c_lon], zoom_start=14, control_scale=True)
    add_tile_layers(m)

    # Demo logs layer
    logs = st.session_state.db.copy()
    if len(logs) and data_mode.startswith("Demo"):
        for _, r in logs.iterrows():
            if r["city"] != city_m or r["district"] != district_m:
                continue
            treated = bool(r.get("treated", False))
            color = ("green" if treated else
                     "orange" if r["status_en"] == "Pest Damage" else
                     "blue"   if r["status_en"] == "Healthy" else
                     "purple")
            popup = (
                f"{r['tree_id']} — {r['status_zh']} / {r['status_en']} ({r['confidence']}%)<br>"
                f"Species 樹種: {r.get('species','-')}<br>"
                f"{r['action_bi']}<br>"
                f"CO₂: {r['co2_saved_kg']} kg<br>"
                f"Diameter 胸徑: {r.get('diameter_cm','-')} cm · Canopy 樹冠: {r.get('canopy_m','-')} m"
            )
            folium.CircleMarker(
                [float(r.get("lat", c_lat)), float(r.get("lon", c_lon))],
                radius=7, color=color, fill=True, fill_color=color,
                popup=popup, tooltip="Logged tree / 已記錄樹木"
            ).add_to(m)

    # Live OSM layer
    if data_mode.startswith("Live"):
        with st.spinner("Querying OpenStreetMap… / 正在查詢 OSM…"):
            try:
                nodes, buildings = fetch_osm_trees_buildings(bbox_m)
                tree_group = folium.FeatureGroup(name="OSM Trees / OSM 樹點").add_to(m)
                for n in nodes[:1000]:
                    folium.CircleMarker(
                        [float(n.lat), float(n.lon)], radius=4,
                        color="#2ecc71", fill=True, fill_color="#2ecc71",
                        tooltip="OSM tree / OSM 樹點"
                    ).add_to(tree_group)
                bld_group = folium.FeatureGroup(name="OSM Buildings / OSM 建物").add_to(m)
                for poly in buildings[:800]:
                    folium.Polygon(
                        locations=[(lat, lon) for (lat, lon) in poly],
                        color="#7f8c8d", weight=1, fill=True,
                        fill_color="#95a5a6", fill_opacity=0.4
                    ).add_to(bld_group)
            except Exception as e:
                st.error(f"OSM query failed / OSM 連線失敗：{e}")

    st_folium(m, width=1000, height=620)

# =========================================================
# TAB 3 — Dashboard / 儀表板
# =========================================================
with tabs[2]:
    st.header("📊 Impact Dashboard / 影響儀表板")

    df = st.session_state.db.copy()
    if len(df):
        total_co2 = float(df["co2_saved_kg"].sum())
        treated_rate = (df["treated"] == True).mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reports / 回報總數", len(df))
        c2.metric("Total CO₂ Saved / 減碳總量", f"{total_co2:.1f} kg")
        c3.metric("Trees Treated / 已處置比例", f"{treated_rate:.1f}%")

        if "timestamp" in df.columns:
            ts = df.assign(day=pd.to_datetime(df["timestamp"]).dt.date).groupby("day")["co2_saved_kg"].sum()
            st.line_chart(ts, height=180)

        st.write("### Top Actions / 最常見處置")
        st.write(df["action_bi"].value_counts())

        # 原型訓練曲線（示範用）
        epochs = np.arange(1, 31)
        train_acc = 0.82 + 0.18 * (1 - np.exp(-0.25 * epochs)) + np.random.normal(0, 0.002, len(epochs))
        val_acc   = 0.80 + 0.17 * (1 - np.exp(-0.22 * epochs)) + np.random.normal(0, 0.003, len(epochs))
        train_loss = 0.45 * np.exp(-0.23 * epochs) + 0.05 + np.random.normal(0, 0.003, len(epochs))
        val_loss   = 0.48 * np.exp(-0.20 * epochs) + 0.06 + np.random.normal(0, 0.004, len(epochs))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc*100, name="Train Acc 訓練準確", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc*100,   name="Val Acc 驗證準確",   line=dict(color="orange", dash="dot")))
        fig.add_trace(go.Scatter(x=epochs, y=train_loss*100, name="Train Loss 訓練損失", line=dict(color="blue", dash="dash")))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss*100,   name="Val Loss 驗證損失",   line=dict(color="red", dash="dashdot")))
        fig.update_layout(title="AI Model Training Progress / 模型訓練趨勢",
                          xaxis_title="Epoch / 世代", yaxis_title="Metric (%)",
                          yaxis_range=[0, 100], template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet — analyze an image to start / 尚無資料，請先於影像分析上傳並記錄。")

# =========================================================
# TAB 4 — Government Summary / 政府端總覽
# =========================================================
with tabs[3]:
    st.header("🏛️ Government Summary / 政府端總覽")
    if len(st.session_state.db):
        st.dataframe(st.session_state.db.tail(500), use_container_width=True)
    else:
        st.info("No citizen reports submitted yet / 尚未有公民回報資料。")
