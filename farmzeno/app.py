# app.py  —  FarmZeno Advisory Application
import os, re, requests, pdfplumber
import numpy as np
from datetime import datetime
from io import BytesIO

from models.disease_model import load_disease_model, predict_disease
from models.crop_model    import load_crop_model, predict_crop, model_loaded

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import cm, mm

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv


app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER       = "uploads"
IMD_CACHE_FOLDER    = "imd_cache"
USER_REPORTS_FOLDER = "user_reports"
app.config["UPLOAD_FOLDER"]       = UPLOAD_FOLDER
app.config["IMD_CACHE_FOLDER"]    = IMD_CACHE_FOLDER
app.config["USER_REPORTS_FOLDER"] = USER_REPORTS_FOLDER
app.config["SECRET_KEY"]          = "farmzeno-secret-key-2025"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///farmzeno.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMD_CACHE_FOLDER, exist_ok=True)
os.makedirs(USER_REPORTS_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access FarmZeno."

# ── Database Models ──────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    reports       = db.relationship("SavedReport", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, pw): self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)

class SavedReport(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    report_type = db.Column(db.String(20), nullable=False)   # "advisory" | "disease"
    title       = db.Column(db.String(200), nullable=False)
    filename    = db.Column(db.String(200), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

with app.app_context():
    db.create_all()

# ── Load models at startup ───────────────────────────────────────────────────
load_disease_model()
load_crop_model()


# ════════════════════════════════════════════════════════════════════════════
#  WEATHER
# ════════════════════════════════════════════════════════════════════════════

def get_current_weather(city: str) -> dict:
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=20,
    )
    data = r.json()
    if data.get("cod") != 200:
        raise ValueError(f"OpenWeather error: {data.get('message')}")
    return {
        "city":        city,
        "temp":        data["main"]["temp"],
        "humidity":    data["main"]["humidity"],
        "pressure":    data["main"]["pressure"],
        "wind_speed":  data.get("wind", {}).get("speed", 0),
        "wind_deg":    data.get("wind", {}).get("deg"),
        "clouds":      data.get("clouds", {}).get("all"),
        "rain_1h":     data.get("rain", {}).get("1h", 0),
        "description": data["weather"][0]["description"],
    }


def get_24h_forecast_summary(city: str) -> dict:
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=20,
    )
    data = r.json()
    if data.get("cod") != "200":
        raise ValueError(f"Forecast error: {data.get('message')}")
    items = data["list"][:8]
    return {
        "temp_min_24h":   round(min(x["main"]["temp"] for x in items), 2),
        "temp_max_24h":   round(max(x["main"]["temp"] for x in items), 2),
        "rain_total_24h": round(sum(x.get("rain", {}).get("3h", 0) for x in items), 2),
        "wind_max_24h":   round(max(x.get("wind", {}).get("speed", 0) for x in items), 2),
    }


def get_weather_summary(city: str) -> dict:
    try:
        return {"success": True, "city": city,
                "weather": get_current_weather(city),
                "forecast": get_24h_forecast_summary(city)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  GEOCODING
# ════════════════════════════════════════════════════════════════════════════

def geocode_city(city: str) -> dict:
    data = requests.get(
        f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}",
        timeout=20,
    ).json()
    if not data:
        raise ValueError(f"City not found: {city}")
    obj = data[0]
    return {"city": obj.get("name", city), "state": obj.get("state"), "country": obj.get("country"),
            "lat": obj["lat"], "lon": obj["lon"]}


def reverse_geocode_district(lat: float, lon: float):
    data = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1},
        headers={"User-Agent": "FarmZenoBot/1.0"},
        timeout=20,
    ).json()
    addr = data.get("address", {})
    district = (addr.get("state_district") or addr.get("county") or
                addr.get("district") or addr.get("region"))
    return district.replace(" District", "").strip() if district else None



# ════════════════════════════════════════════════════════════════════════════
#  IMD BULLETIN
# ════════════════════════════════════════════════════════════════════════════

def _safe(s): return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)


def download_imd_district_bulletin(state, district, save_dir="imd_cache"):
    os.makedirs(save_dir, exist_ok=True)
    r = requests.get("https://imdagrimet.gov.in/Services/DistrictBulletin.php",
                     params={"state": state, "district": district, "language": "English"}, timeout=30)
    if "pdf" in (r.headers.get("Content-Type") or "").lower():
        path = os.path.join(save_dir, f"{_safe(state)}_{_safe(district)}_{datetime.now().strftime('%Y-%m-%d')}.pdf")
        open(path, "wb").write(r.content)
        return path, r.url, "DISTRICT"
    return None, r.url, None


def download_imd_state_bulletin(state, save_dir="imd_cache"):
    os.makedirs(save_dir, exist_ok=True)
    r = requests.get("https://imdagrimet.gov.in/Services/StateBulletin.php",
                     params={"state": state, "language": "English"}, timeout=30)
    if "pdf" not in (r.headers.get("Content-Type") or "").lower():
        raise ValueError("State bulletin did not return PDF.")
    path = os.path.join(save_dir, f"{_safe(state)}_STATE_{datetime.now().strftime('%Y-%m-%d')}.pdf")
    open(path, "wb").write(r.content)
    return path, r.url, "STATE"


def extract_pdf_text(pdf_path, max_pages=10):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(len(pdf.pages), max_pages)):
            txt = pdf.pages[i].extract_text() or ""
            texts.append(re.sub(r"\s+\n", "\n", txt))
    return "\n\n".join(texts)


# ════════════════════════════════════════════════════════════════════════════
#  RAG
# ════════════════════════════════════════════════════════════════════════════

def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180,
                                              separators=["\n\n", "\n", ".", " ", ""])
    docs = [Document(page_content=c) for c in splitter.split_text(text)]
    return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))


def retrieve_docs(vectorstore, weather, forecast, k=5):
    query = (f"advisory irrigation wind spraying rainfall humidity disease pest "
             f"temp={weather['temp']} hum={weather['humidity']} wind={weather['wind_speed']} "
             f"rain={weather['rain_1h']} rain24={forecast['rain_total_24h']}")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k}).invoke(query)


# ════════════════════════════════════════════════════════════════════════════
#  ADVISORY PROCESSING
# ════════════════════════════════════════════════════════════════════════════

def _clean(text):
    if not text: return ""
    text = re.sub(r"[▪■◾◆●○□▯▭▬▮➤→↓↑←✓✗☑☐★☆]+", "", text)
    text = re.sub(r"A\.\d+(\.\d+)?|[A-Z]\.\d+|\s*@\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-zA-Z0-9\s.,:()\/-]", "", text)
    return re.sub(r"\s+([.,])", r"\1", text).strip()


def _complete_sentence(t):
    if not t: return ""
    t = t.strip()
    for bad in ["to the", "like", "by", "with", "such as", "per liter of", "per litre of","and","or","in"]:
        if t.lower().endswith(bad): return ""
    return t if t.endswith(".") else t + "."


def _merge_points(points):
    cleaned, buf = [], ""
    for p in points:
        p = _clean(p)
        if not p: continue
        if p.startswith(")") or any(p.startswith(x) for x in ["Wilt)", "Rot)", "Mildew)", "Blight)"]):
            buf = (buf + " " + p).strip(); continue
        if buf: cleaned.append(buf)
        buf = p
        if buf.endswith(".") or len(buf) > 140:
            cleaned.append(buf); buf = ""
    if buf: cleaned.append(buf)
    seen, final = set(), []
    for x in cleaned:
        x = _complete_sentence(x)
        if x and x not in seen and len(x) > 25:
            seen.add(x); final.append(x)
    return final


RULES = {
    "Weather warnings":                   ["warning","alert","thunderstorm","hailstorm","heavy rain","strong wind","fog","cold wave","heat wave","nil"],
    "Crop pest/disease alert":            ["pest","disease","mite","thrips","aphid","bollworm","rust","blight","wilt","rot","fungal","incidence","mildew","worm"],
    "Irrigation guidance":                ["irrigation","irrigate","watering","drainage","waterlogging","reduce irrigation","light irrigation"],
    "Spraying / fertilizer guidance":     ["spray","spraying","pesticide","insecticide","fungicide","herbicide","fertilizer","dose"],
    "Harvest / storage / field operations":["sowing","harvesting","threshing","weeding","interculture","plough","storage","nursery","transplant"],
}
ACTION_WORDS = ["spray","apply","control","monitor","avoid","drench","irrigation","ensure","minimize","prevent","trap"]


def categorize_advisory(docs, max_per_cat=5):
    cats = {k: [] for k in RULES}
    head_pats = [r"^A\.\d+",r"^WEATHER WARNINGS",r"^Likely impacts",r"^Realized Rainfall",
                 r"^Bulletin Number",r"^India Meteorological",r"^Government of India"]

    def is_heading(l):
        if len(l.strip()) < 18: return True
        for p in head_pats:
            if re.search(p, l, re.IGNORECASE): return True
        return l.strip().endswith(":") and not any(w in l.lower() for w in ACTION_WORDS)

    def actionable(l): return any(w in l.lower() for w in ACTION_WORDS)

    raw = []
    for doc in docs:
        for p in re.split(r"(?<=[.!?])\s+|\n|•", doc.page_content):
            p = _clean(p)
            if p: raw.append(p)

    merged = []
    for line in raw:
        if merged and len(merged[-1]) < 70 and not merged[-1].endswith("."):
            merged[-1] = (merged[-1] + " " + line).strip()
        else:
            merged.append(line)

    for line in merged:
        if is_heading(line) or not actionable(line): continue
        low = line.lower()
        for cat, keys in RULES.items():
            if any(k in low for k in keys):
                if line not in cats[cat]: cats[cat].append(line)
                break

    for c in cats:
        cats[c] = _merge_points(cats[c])[:max_per_cat]
    return cats


def add_weather_advice(weather, forecast, cats):
    temp = weather["temp"]; hum = weather["humidity"]
    wind = weather["wind_speed"]; rain24 = forecast["rain_total_24h"]

    if rain24 == 0:
        cats["Weather warnings"].insert(0, "No rainfall expected in next 24 hours. Plan field activities accordingly.")
    if wind >= 6:
        cats["Weather warnings"].insert(0, f"High wind speed of {wind} m/s expected. Avoid spraying during peak wind hours.")
    if hum < 45 and rain24 == 0:
        cats["Irrigation guidance"].insert(0, "Low humidity and no rainfall expected. Increase irrigation frequency and use mulching.")
    if rain24 >= 5:
        cats["Irrigation guidance"].insert(0, "Rainfall expected in 24 hours. Reduce irrigation and clear drainage channels.")
    if wind >= 6:
        cats["Spraying / fertilizer guidance"].insert(0, "Avoid spraying during windy hours. Prefer early morning applications.")
    if rain24 >= 3:
        cats["Spraying / fertilizer guidance"].insert(0, "Rainfall expected. Avoid spraying before rain; apply after it clears.")
    if rain24 == 0:
        cats["Harvest / storage / field operations"].insert(0, "No rain expected. Good window for weeding, interculture and harvesting.")
    if hum >= 75:
        cats["Crop pest/disease alert"].insert(0, f"High humidity ({hum}%). Monitor for fungal diseases like leaf spots and powdery mildew.")

    for k in cats:
        cats[k] = _merge_points(cats[k])[:6]
    return cats


def filter_by_crop(cats, crop=None):
    if not crop: return cats
    crop = crop.lower()
    return {s: [p for p in pts if crop in p.lower()] for s, pts in cats.items()}


def rank_advisories(cats, weather, forecast, top_n=5):
    hum = weather["humidity"]; wind = weather["wind_speed"]; rain24 = forecast["rain_total_24h"]
    cat_scores = {"Weather warnings":5,"Crop pest/disease alert":4,"Spraying / fertilizer guidance":3,
                  "Irrigation guidance":3,"Harvest / storage / field operations":2}
    ranked = []
    for cat, pts in cats.items():
        for p in pts:
            low = p.lower()
            s = cat_scores.get(cat, 1)
            if wind >= 6 and "spray" in low: s += 4
            if rain24 >= 5 and ("drainage" in low or "waterlogging" in low): s += 4
            if hum < 45 and "irrigation" in low: s += 3
            if hum >= 75 and ("fungal" in low or "mildew" in low): s += 3
            ranked.append((s, cat, p))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[:top_n]


def extract_bulletin_meta(text):
    meta = {}
    m = re.search(r"Bulletin\s*Number\s*[:\-]?\s*([0-9/]+)", text, re.IGNORECASE)
    if m: meta["bulletin_number"] = m.group(1)
    d = re.search(r"FROM\s+(\d{2}-\d{2}-\d{4})\s+to\s+(\d{2}-\d{2}-\d{4})", text, re.IGNORECASE)
    if d: meta["forecast_date_from"] = d.group(1); meta["forecast_date_to"] = d.group(2)
    return meta


# ════════════════════════════════════════════════════════════════════════════
#  CROP RECOMMENDATION (ML-based)
# ════════════════════════════════════════════════════════════════════════════

# Default pH fallback (neutral loamy soil — reasonable general assumption)
DEFAULT_PH = 6.5

CROP_EMOJIS = {
    "rice":"","wheat":"","maize":"🌽","corn":"🌽","millet":"","sorghum":"",
    "cotton":"","sugarcane":"🎋","jute":"","coffee":"☕","tea":"🍵",
    "rubber":"🌳","coconut":"🥥","papaya":"🍈","mango":"🥭","banana":"🍌",
    "grapes":"🍇","apple":"🍎","muskmelon":"🍈","watermelon":"🍉","pomegranate":"🍎",
    "lentil":"🫘","blackgram":"🫘","mungbean":"🫘","mothbeans":"🫘","pigeonpeas":"🫘",
    "chickpea":"🫘","kidneybeans":"🫘","tomato":"🍅","potato":"🥔","onion":"🧅",
}

def get_crop_emoji(crop_name):
    c = crop_name.lower().replace(" ", "")
    for key, emoji in CROP_EMOJIS.items():
        if key in c: return emoji
    return "🌱"


def get_ml_crop_recommendations(weather: dict, forecast: dict, ph: float = DEFAULT_PH) -> dict:
    """
    Use the trained ML model to recommend crops.
    Features: Temperature, Humidity, pH, Rainfall (24h forecast as proxy)
    """
    temp     = weather["temp"]
    humidity = weather["humidity"]
    rain24   = forecast["rain_total_24h"]
    wind     = weather["wind_speed"]

    result = {
        "ml_available": False,
        "recommended_crop": None,
        "top5_crops": [],
        "management_tips": [],
        "input_features": {
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall_24h_mm": rain24,
        }
    }

    if model_loaded():
        try:
            crop_name, top5 = predict_crop(temp, humidity, ph, rain24)
            result["ml_available"]     = True
            result["recommended_crop"] = crop_name
            result["top5_crops"]       = top5
        except Exception as e:
            print(f"[CropRec] ML prediction error: {e}")

    # Always add weather-based management tips
    tips = []
    if wind >= 8:
        tips += [" Strong winds expected — provide windbreaks for delicate crops",
                 "🌽 Staking recommended for tall crops like maize",
                 "🚫 Avoid pesticide spraying in windy conditions"]
    if rain24 >= 10:
        tips += ["️ Heavy rainfall expected — ensure proper field drainage",
                 "🚜 Postpone fertilizer application until after rains"]
    elif rain24 >= 3:
        tips += ["🌦️ Light rain expected — good window for transplanting",
                 " Reduce irrigation to avoid waterlogging"]
    elif rain24 == 0:
        tips += [" Dry conditions — plan your irrigation schedule",
                 "🌱 Rainfed crops may need supplemental irrigation"]
    if humidity >= 75:
        tips.append("🦠 High humidity — monitor crops for fungal diseases regularly")
    if humidity < 40:
        tips += ["💦 Low humidity — drip irrigation recommended",
                 "🌱 Use mulching to retain soil moisture"]
    if temp > 35:
        tips += ["☀️ High temperature — provide shade nets for sensitive crops",
                 " Increase irrigation frequency during peak heat hours"]
    elif temp < 15:
        tips.append(" Cool conditions — use mulch to maintain soil warmth")
    if ph < 5.5:
        tips.append(" Acidic soil detected — consider liming before sowing")
    elif ph > 8.0:
        tips.append(" Alkaline soil detected — gypsum application may improve yield")

    result["management_tips"] = list(dict.fromkeys(tips))
    return result


# ════════════════════════════════════════════════════════════════════════════
#  FULL ADVISORY
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  FULL ADVISORY
# ════════════════════════════════════════════════════════════════════════════

def smart_farm_advisory(city: str, crop: str = None):
    weather  = get_current_weather(city)
    forecast = get_24h_forecast_summary(city)
    loc      = geocode_city(city)
    state    = loc["state"]
    district = reverse_geocode_district(loc["lat"], loc["lon"])

    pdf_path, pdf_url, level = None, None, None
    if state and district:
        pdf_path, pdf_url, level = download_imd_district_bulletin(state, district, IMD_CACHE_FOLDER)
    if not pdf_path:
        pdf_path, pdf_url, level = download_imd_state_bulletin(state, IMD_CACHE_FOLDER)

    text  = extract_pdf_text(pdf_path)
    meta  = extract_bulletin_meta(text)
    vs    = build_vectorstore(text)
    docs  = retrieve_docs(vs, weather, forecast)
    cats  = categorize_advisory(docs)
    cats  = add_weather_advice(weather, forecast, cats)
    cats  = filter_by_crop(cats, crop)
    top   = rank_advisories(cats, weather, forecast)

    return {
        "city": city, "crop": crop, "weather": weather, "forecast_24h": forecast,
        "state": state, "district": district, "bulletin_level": level,
        "bulletin_url": pdf_url, "bulletin_meta": meta,
        "top_advisories": top, "categorized_advisory": cats,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ════════════════════════════════════════════════════════════════════════════

# ── Shared PDF helpers ───────────────────────────────────────────────────────

G_GREEN  = colors.HexColor("#3a7a1a")
G_ORANGE = colors.HexColor("#e8621a")
G_PALE   = colors.HexColor("#eef8f2")
G_BORDER = colors.HexColor("#c2e8d0")
G_DARK   = colors.HexColor("#1a3a0a")
G_MID    = colors.HexColor("#4caf78")
O_PALE   = colors.HexColor("#fff3ec")
O_BORDER = colors.HexColor("#f5c9a8")
GREY_LT  = colors.HexColor("#f7f7f7")
GREY_MID = colors.HexColor("#e0e0e0")
GREY_TXT = colors.HexColor("#555555")
RED_TXT  = colors.HexColor("#c0392b")
WHITE    = colors.white
BLACK    = colors.HexColor("#1a1a1a")

def _brand_table(page_w):
    """FarmZeno header row — Farm green, Zeno orange, with leaf icon simulation."""
    brand = Table([[
        Paragraph('<font name="Helvetica-Bold" size="30" color="#3a7a1a">Farm</font>'
                  '<font name="Helvetica-Bold" size="30" color="#e8621a">Zeno</font>',
                  ParagraphStyle("bt", alignment=TA_CENTER, fontSize=30,
                                 fontName="Helvetica-Bold", leading=36))
    ]], colWidths=[page_w])
    brand.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"), ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    return brand

def _green_rule(page_w, thickness=3):
    return HRFlowable(width=page_w, thickness=thickness, color=G_GREEN, spaceAfter=6, spaceBefore=2)

def _orange_rule(page_w, thickness=1.5):
    return HRFlowable(width=page_w, thickness=thickness, color=G_ORANGE, spaceAfter=4, spaceBefore=2)

def _section_header(title, page_w, bg=None):
    bg = bg or G_GREEN
    t = Table([[Paragraph(title, ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=9,
               textColor=WHITE, leading=13))]],
              colWidths=[page_w])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), bg),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),10),
    ]))
    return t

def _tip_box(text, page_w, icon="ℹ", bg=None):
    bg = bg or G_PALE
    t = Table([[Paragraph(f'<font color="#3a7a1a"><b>{icon}</b></font>  {text}',
                ParagraphStyle("tb", fontSize=8.5, leading=13, textColor=BLACK))]],
              colWidths=[page_w])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), bg),
        ("BOX",(0,0),(-1,-1),0.5, G_BORDER),
        ("TOPPADDING",(0,0),(-1,-1),6), ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),10), ("RIGHTPADDING",(0,0),(-1,-1),10),
        ("ROUNDEDCORNERS",[4]),
    ]))
    return t

# Disease → plain text advice mapping
DISEASE_ADVICE = {
    "apple___black_rot":       ("Black Rot (Fungus)", "Remove mummified fruits, prune infected wood. Spray Captan 50WP at 2g/L or Mancozeb 75WP at 2.5g/L every 10–14 days during wet weather.", "Avoid overhead irrigation. Destroy fallen leaves."),
    "apple___apple_scab":      ("Apple Scab (Fungus)", "Apply Myclobutanil 1g/L or Carbendazim 1g/L at bud-break and repeat every 7–10 days.", "Remove and burn fallen leaves. Ensure good air circulation."),
    "apple___cedar_rust":      ("Cedar Apple Rust (Fungus)", "Spray Triadimefon 1ml/L at pink bud stage, repeat every 7–10 days.", "Remove nearby juniper/cedar trees if possible."),
    "apple___healthy":         ("Healthy Plant ✓", "No treatment needed. Maintain regular fertilisation and irrigation schedule.", "Continue monitoring weekly for early signs of disease."),
    "tomato___early_blight":   ("Early Blight (Fungus)", "Spray Mancozeb 2.5g/L or Chlorothalonil 2g/L every 7 days. Remove lower infected leaves.", "Avoid wetting foliage. Rotate crops next season."),
    "tomato___late_blight":    ("Late Blight (Oomycete)", "Apply Metalaxyl + Mancozeb 2.5g/L immediately. Repeat every 5–7 days in wet weather.", "Remove and destroy infected plants. Do not compost them."),
    "tomato___leaf_mold":      ("Leaf Mould (Fungus)", "Spray Chlorothalonil 2g/L or Copper oxychloride 3g/L every 10 days.", "Improve ventilation in greenhouse. Reduce humidity."),
    "tomato___septoria_leaf_spot": ("Septoria Leaf Spot (Fungus)", "Apply Mancozeb 2.5g/L or Chlorothalonil 2g/L at first sign. Repeat weekly.", "Remove lower leaves touching soil. Avoid overhead watering."),
    "tomato___spider_mites":   ("Spider Mites (Pest)", "Spray Abamectin 0.5ml/L or Dicofol 2ml/L. Repeat in 7 days.", "Increase humidity around plants. Avoid water stress."),
    "tomato___target_spot":    ("Target Spot (Fungus)", "Apply Azoxystrobin 1ml/L or Tebuconazole 1ml/L every 14 days.", "Remove infected leaves. Improve air circulation."),
    "tomato___mosaic_virus":   ("Tomato Mosaic Virus", "No chemical cure. Remove and destroy infected plants immediately.", "Control aphid vectors with Imidacloprid 0.5ml/L. Disinfect tools."),
    "tomato___yellow_leaf_curl_virus": ("Yellow Leaf Curl Virus", "Remove infected plants. Control whitefly with Thiamethoxam 0.3g/L.", "Use reflective mulch to repel whiteflies. Use virus-resistant varieties."),
    "tomato___healthy":        ("Healthy Plant ✓", "No treatment needed. Maintain balanced NPK fertilisation.", "Continue weekly scouting for early pest/disease signs."),
    "potato___early_blight":   ("Early Blight (Fungus)", "Spray Mancozeb 2.5g/L or Chlorothalonil 2g/L every 10 days.", "Ensure adequate potassium nutrition. Avoid water stress."),
    "potato___late_blight":    ("Late Blight (Oomycete)", "Apply Metalaxyl 2g/L + Mancozeb 2.5g/L immediately. Repeat every 5–7 days.", "Destroy infected foliage. Avoid storing infected tubers."),
    "potato___healthy":        ("Healthy Plant ✓", "No treatment needed. Hill up soil around stems.", "Monitor for Colorado beetle and late blight weekly."),
    "corn___common_rust":      ("Common Rust (Fungus)", "Apply Propiconazole 1ml/L or Tebuconazole 1ml/L at first sign.", "Plant resistant hybrids next season."),
    "corn___northern_leaf_blight": ("Northern Leaf Blight (Fungus)", "Spray Azoxystrobin 1ml/L at tasselling stage. Repeat in 14 days.", "Rotate with non-host crops. Till infected residue."),
    "corn___gray_leaf_spot":   ("Gray Leaf Spot (Fungus)", "Apply Strobilurin fungicide at V6 stage. Repeat every 14 days.", "Improve air movement. Rotate crops."),
    "corn___healthy":          ("Healthy Plant ✓", "No treatment needed. Maintain soil nitrogen levels.", "Scout for fall armyworm and corn borer weekly."),
    "grape___black_rot":       ("Black Rot (Fungus)", "Spray Myclobutanil 1g/L or Captan 2g/L from bud break, every 10 days.", "Remove mummified berries. Prune for air circulation."),
    "grape___leaf_blight":     ("Leaf Blight (Fungus)", "Apply Copper oxychloride 3g/L or Mancozeb 2.5g/L.", "Improve drainage. Remove infected leaves."),
    "grape___healthy":         ("Healthy Plant ✓", "No treatment needed. Maintain pruning and tying schedule.", "Monitor for powdery mildew and downy mildew."),
    "pepper,_bell___bacterial_spot": ("Bacterial Spot", "Apply Copper hydroxide 3g/L every 5–7 days. Remove infected leaves.", "Use certified disease-free seed. Avoid overhead irrigation."),
    "pepper,_bell___healthy":  ("Healthy Plant ✓", "No treatment needed. Ensure consistent watering.", "Scout for aphids and thrips weekly."),
    "strawberry___leaf_scorch": ("Leaf Scorch (Fungus)", "Spray Myclobutanil 1g/L or Captan 2g/L every 10 days.", "Remove old leaves. Avoid over-fertilising with nitrogen."),
    "strawberry___healthy":    ("Healthy Plant ✓", "No treatment needed. Maintain adequate potassium.", "Monitor for grey mould during flowering."),
    "cherry___powdery_mildew": ("Powdery Mildew (Fungus)", "Apply Sulfur-based fungicide 3g/L or Hexaconazole 1ml/L every 7–10 days.", "Avoid excess nitrogen. Ensure good sunlight penetration."),
    "cherry___healthy":        ("Healthy Plant ✓", "No treatment needed. Thin fruits to improve air circulation.", "Monitor for cherry fruit fly."),
    "peach___bacterial_spot":  ("Bacterial Spot", "Spray Copper hydroxide 3g/L during dormancy and at bud break.", "Avoid pruning in wet weather. Choose resistant varieties."),
    "peach___healthy":         ("Healthy Plant ✓", "No treatment needed. Thin fruit clusters.", "Scout for peach leaf curl in spring."),
    "blueberry___healthy":     ("Healthy Plant ✓", "No treatment needed. Maintain soil pH 4.5–5.5.", "Monitor for mummy berry and stem blight."),
    "squash___powdery_mildew": ("Powdery Mildew (Fungus)", "Spray potassium bicarbonate 5g/L or Sulfur 3g/L every 7 days.", "Improve spacing for air circulation. Avoid excess nitrogen."),
    "rice___leaf_blast":       ("Leaf Blast (Fungus)", "Apply Tricyclazole 0.6g/L or Isoprothiolane 1.5ml/L at first sign.", "Avoid excess nitrogen. Drain field periodically."),
    "rice___brown_spot":       ("Brown Spot (Fungus)", "Spray Mancozeb 2.5g/L or Edifenphos 1ml/L.", "Improve potassium nutrition. Use certified seeds."),
    "rice___neck_blast":       ("Neck Blast (Fungus)", "Apply Tricyclazole 0.6g/L at booting stage. Critical timing.", "Avoid late heavy nitrogen. Use resistant varieties."),
}

def _get_disease_advice(disease_name):
    """Look up treatment advice for detected disease."""
    key = disease_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", ",")
    # Try direct match first
    if key in DISEASE_ADVICE:
        return DISEASE_ADVICE[key]
    # Try partial match
    for k, v in DISEASE_ADVICE.items():
        if k in key or key in k:
            return v
    # Generic fallback
    if "healthy" in key:
        return ("Healthy Plant ✓",
                "No disease detected. Continue your regular crop management practices.",
                "Monitor weekly and maintain optimal nutrition and irrigation.")
    return ("Disease Detected",
            "Consult your local agriculture extension officer for specific treatment recommendations for this disease.",
            "Isolate affected plants. Document spread and report to local Krishi Vigyan Kendra (KVK).")


# ── DISEASE PDF ──────────────────────────────────────────────────────────────

def create_disease_pdf(result: dict, image_b64: str = "") -> BytesIO:
    import base64

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=30, bottomMargin=30)

    styles = getSampleStyleSheet()

    title = ParagraphStyle("title", fontSize=18, alignment=TA_CENTER,
                           textColor=G_GREEN, spaceAfter=10)

    normal = ParagraphStyle("normal", fontSize=10, leading=14)

    small = ParagraphStyle("small", fontSize=8, textColor=GREY_TXT)

    disease = result.get("disease", "Unknown")
    conf = round(float(result.get("confidence", 0)), 2)
    top5 = result.get("top5", [])
    weather = result.get("weather", {})

    d_name, d_treat, d_prevent = _get_disease_advice(disease)
    is_healthy = "healthy" in disease.lower()

    story = []

    # HEADER
    story.append(Paragraph(datetime.now().strftime("%d %B %Y  |  %H:%M"), small))
    story.append(_brand_table(A4[0]-80))
    story.append(_green_rule(A4[0]-80))
    story.append(Paragraph("Plant Disease Detection Report", title))
    story.append(Spacer(1, 10))

    # IMAGE
    if image_b64 and "base64," in image_b64:
        try:
            img_bytes = base64.b64decode(image_b64.split("base64,")[1])
            img = RLImage(BytesIO(img_bytes), width=160, height=160)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 10))
        except:
            pass

    # RESULT BOX
    result_table = Table([
        ["Detected Disease", disease],
        ["Confidence", f"{conf}%"],
        ["Type", d_name],
    ], colWidths=[150, 300])

    result_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), GREY_LT),
        ("BOX",(0,0),(-1,-1),0.5,GREY_MID),
        ("INNERGRID",(0,0),(-1,-1),0.3,GREY_MID),
        ("PADDING",(0,0),(-1,-1),6)
    ]))

    story.append(result_table)
    story.append(Spacer(1, 12))

    # CONFIDENCE BAR
    bar_width = 400
    filled = int(bar_width * conf / 100)

    bar = Table([["", ""]], colWidths=[filled, bar_width-filled])
    bar.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), G_GREEN),
        ("BACKGROUND",(1,0),(1,0), GREY_MID)
    ]))

    story.append(Paragraph("Confidence Level", normal))
    story.append(bar)
    story.append(Spacer(1, 12))

    # SEVERITY
    severity = "Low"
    if conf > 80: severity = "High"
    elif conf > 50: severity = "Moderate"

    story.append(Paragraph(f"<b>Severity:</b> {severity}", normal))
    story.append(Spacer(1, 12))

    # WEATHER
    if weather:
        story.append(_section_header("Field Conditions", A4[0]-80))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Temperature: {weather.get('temp','-')}°C<br/>"
            f"Humidity: {weather.get('humidity','-')}%<br/>"
            f"Wind: {weather.get('wind_speed','-')} m/s",
            normal
        ))
        story.append(Spacer(1, 12))

    # TREATMENT
    story.append(_section_header("Treatment & Prevention", A4[0]-80))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Treatment:</b> {d_treat}", normal))
    story.append(Paragraph(f"<b>Prevention:</b> {d_prevent}", normal))
    story.append(Spacer(1, 12))

    # CHECKLIST
    story.append(_section_header("Immediate Actions", A4[0]-80))
    checklist = [
        "Inspect nearby plants",
        "Remove infected parts",
        "Start treatment immediately",
        "Avoid overwatering",
        "Monitor daily"
    ]
    for item in checklist:
        story.append(Paragraph(f"• {item}", normal))
    story.append(Spacer(1, 12))

    # RISK
    if not is_healthy:
        story.append(_section_header("Risk Alert", A4[0]-80))
        story.append(Spacer(1, 6))
        story.append(Paragraph("⚠ Disease can spread rapidly and reduce yield.", normal))
        story.append(Spacer(1, 12))

    # TOP 5
    if top5:
        story.append(_section_header("Top Predictions", A4[0]-80))
        data = [["Rank", "Prediction", "Confidence"]]
        for i,(d,p) in enumerate(top5,1):
            data.append([str(i), d, f"{p}%"])

        t = Table(data, colWidths=[50, 250, 100])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), G_GREEN),
            ("TEXTCOLOR",(0,0),(-1,0), WHITE),
            ("GRID",(0,0),(-1,-1),0.3,GREY_MID)
        ]))
        story.append(t)

    # FOOTER
    story.append(Spacer(1, 15))
    story.append(_orange_rule(A4[0]-80))
    story.append(Paragraph(
        "AI-generated report. Please verify with an agriculture expert.",
        small
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ── ADVISORY PDF ─────────────────────────────────────────────────────────────

def create_advisory_pdf(data: dict) -> BytesIO:
    """
    Single-page professional advisory PDF.
    No emojis — clean table-based layout, fits on one A4 page.
    """
    buf = BytesIO()
    L, R, T, B = 40, 40, 30, 28
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=L, rightMargin=R,
                            topMargin=T, bottomMargin=B)
    PAGE_W = A4[0] - L - R   # ~515 pts

    sty = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, parent=sty["Normal"], **kw)

    date_s  = S("pd",  fontSize=7.5, textColor=GREY_TXT, spaceAfter=1)
    brand_s = S("pb",  alignment=TA_CENTER, fontSize=28, fontName="Helvetica-Bold",
                spaceAfter=10, spaceBefore=6)
    sub_s   = S("ps",  alignment=TA_CENTER, fontSize=10, textColor=GREY_TXT,
                spaceAfter=30, spaceBefore=26)
    disc_s  = S("pdi", alignment=TA_CENTER, fontSize=7.5, fontName="Helvetica-Oblique",
                textColor=RED_TXT, spaceBefore=6, spaceAfter=3)
    cell_v  = S("pcv", fontSize=11, fontName="Helvetica-Bold", alignment=TA_CENTER,
                textColor=G_GREEN, spaceAfter=0, spaceBefore=0, leading=13)
    cell_l  = S("pcl", fontSize=8, alignment=TA_CENTER, textColor=GREY_TXT,
                spaceAfter=0, spaceBefore=0, leading=9)
    cell_v_o = S("pcvo", fontSize=11, fontName="Helvetica-Bold", alignment=TA_CENTER,
                 textColor=G_ORANGE, spaceAfter=0, spaceBefore=0, leading=13)
    sec_hd  = S("psh", fontSize=8, fontName="Helvetica-Bold", textColor=WHITE,
                spaceAfter=0, spaceBefore=0, leading=13)
    adv_cat = S("pac", fontSize=8, fontName="Helvetica-Bold", textColor=G_GREEN,
                spaceAfter=0, spaceBefore=0, leading=11)
    adv_txt = S("pat", fontSize=8, leading=11, textColor=BLACK, spaceAfter=0)
    foot_s  = S("pft", alignment=TA_CENTER, fontSize=8, spaceAfter=0, spaceBefore=3)

    w        = data["weather"]
    f        = data["forecast_24h"]
    city     = data.get("city", "—")
    state    = data.get("state", "—")
    district = data.get("district", "—")
    crop     = data.get("crop") or ""

    story = []

    # ── 1. DATE ───────────────────────────────────────────────────────────────
    story.append(Paragraph(datetime.now().strftime("%d %B %Y  |  %H:%M"), date_s))
    story.append(Spacer(1, 6))

    # ── 2. BRAND HEADER ───────────────────────────────────────────────────────
    story.append(Paragraph(
        '<font color="#3a7a1a">Farm</font><font color="#e8621a">Zeno</font>',
        brand_s))
    story.append(Spacer(1, 10))
    sub_s   = S("ps",  alignment=TA_CENTER, fontSize=14, textColor=BLACK,
            fontName="Helvetica-Bold",
            spaceAfter=8, spaceBefore=2)
    story.append(Paragraph("Smart Farm Advisory Report", sub_s))
    story.append(Spacer(1, 10))

    # ── 3. LOCATION STRIP ─────────────────────────────────────────────────────
    loc_parts = [f"Location: {city}", f"State: {state}", f"District: {district}"]
    if crop:
        loc_parts.append(f"Crop: {crop}")
    loc_str = "  |  ".join(loc_parts)
    loc_t = Table(
        [[Paragraph(loc_str, S("pl", fontSize=8.5, textColor=BLACK,
                               fontName="Helvetica-Bold"))]],
        colWidths=[PAGE_W])
    loc_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), G_PALE),
        ("BOX",           (0,0), (-1,-1), 0.8, G_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(loc_t)
    story.append(Spacer(1, 16))   # was 6 → increase

    # ── 4. WEATHER GRID ───────────────────────────────────────────────────────
    sh_wh = Table(
        [[Paragraph("LIVE WEATHER CONDITIONS", sec_hd)]],
        colWidths=[PAGE_W])
    sh_wh.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), G_GREEN),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(sh_wh)

    CW = PAGE_W / 6

    def wc(val, lbl, orange=False):
        vs = cell_v_o if orange else cell_v
        return [Paragraph(str(val), vs), Paragraph(lbl, cell_l)]

    w_rows = [
        wc(f"{w['temp']}°C",              "Temperature"),
        wc(f"{w['humidity']}%",           "Humidity"),
        wc(f"{w['wind_speed']} m/s",      "Wind Speed"),
        wc(f"{w['rain_1h']} mm",          "Rainfall 1h", True),
        wc(f"{w['pressure']} hPa",        "Pressure"),
        wc(w.get('description','—').title(), "Condition"),
    ]
    wgrid_data = [[
    Paragraph(f"<font size=10><b>{w['temp']}°C</b></font><br/><font size=7 color='#666666'>Temperature</font>", cell_v),

    Paragraph(f"<font size=10><b>{w['humidity']}%</b></font><br/><font size=7 color='#666666'>Humidity</font>", cell_v),

    Paragraph(f"<font size=10><b>{w['wind_speed']} m/s</b></font><br/><font size=7 color='#666666'>Wind Speed</font>", cell_v),

    Paragraph(f"<font size=10><b>{w['rain_1h']} mm</b></font><br/><font size=7 color='#666666'>Rainfall 1h</font>", cell_v_o),

    Paragraph(f"<font size=10><b>{w['pressure']} hPa</b></font><br/><font size=7 color='#666666'>Pressure</font>", cell_v),

    Paragraph(f"<font size=10><b>{w.get('description','—').title()}</b></font><br/><font size=7 color='#666666'>Condition</font>", cell_v),
]]
    wgrid = Table(wgrid_data, colWidths=[CW]*6, rowHeights=[36])

    wgrid.setStyle(TableStyle([
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

    # ✅ MUCH BETTER SPACING
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("RIGHTPADDING", (0,0), (-1,-1), 6),

    # clean grid
    ("GRID", (0,0), (-1,-1), 0.4, GREY_MID),
    ("BOX", (0,0), (-1,-1), 0.6, GREY_MID),
]))
    story.append(wgrid)
    story.append(Spacer(1, 10))

    fc_str = (f"24h Forecast:  Temp {f['temp_min_24h']}°C - {f['temp_max_24h']}°C"
              f"  |  Rain {f['rain_total_24h']} mm"
              f"  |  Max Wind {f['wind_max_24h']} m/s")
    fc_t = Table(
        [[Paragraph(fc_str, S("pfc", fontSize=8, textColor=WHITE))]],
        colWidths=[PAGE_W])
    fc_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), G_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(fc_t)
    story.append(Spacer(1, 16))

    # ── 5. TOP PRIORITY ADVISORIES ────────────────────────────────────────────
    tops = data.get("top_advisories", [])
    if tops:
        sh_top = Table(
            [[Paragraph("TOP PRIORITY ADVISORIES", sec_hd)]],
            colWidths=[PAGE_W])
        sh_top.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), G_ORANGE),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(sh_top)
        adv_rows = []
        for i, adv in enumerate(tops[:5], 1):
            cat  = adv[1]
            text = adv[2]
            if len(text) > 160:
                text = text[:157] + "..."
            bg = G_PALE if i % 2 == 1 else WHITE
            adv_rows.append([
                Paragraph(str(i), S(f"an{i}", fontSize=8, fontName="Helvetica-Bold",
                           textColor=WHITE, alignment=TA_CENTER)),
                Paragraph(cat, adv_cat),
                Paragraph(text, adv_txt),
            ])
        adv_t = Table(adv_rows, colWidths=[18, 135, PAGE_W - 165])
        adv_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),  (0,-1),  G_GREEN),
            ("ROWBACKGROUNDS",(1,0),  (-1,-1), [G_PALE, WHITE]),
            ("LINEBELOW",     (0,0),  (-1,-1), 0.3, GREY_MID),
            ("LINEBEFORE",    (1,0),  (1,-1),  0.5, G_BORDER),
            ("LINEBEFORE",    (2,0),  (2,-1),  0.3, GREY_MID),
            ("BOX",           (0,0),  (-1,-1), 0.4, G_BORDER),
            ("TOPPADDING",    (0,0),  (-1,-1), 4),
            ("BOTTOMPADDING", (0,0),  (-1,-1), 4),
            ("LEFTPADDING",   (0,0),  (-1,-1), 6),
            ("RIGHTPADDING",  (0,0),  (-1,-1), 4),
            ("VALIGN",        (0,0),  (-1,-1), "TOP"),
        ]))
        story.append(adv_t)
        story.append(Spacer(1, 16))

    # ── 6. CATEGORY ADVISORIES ────────────────────────────────────────────────
    cats = data.get("categorized_advisory", {})
    has_any = any(pts for pts in cats.values())
    if has_any:
        sh_cat = Table(
            [[Paragraph("CATEGORY-BASED ADVISORY", sec_hd)]],
            colWidths=[PAGE_W])
        sh_cat.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), G_GREEN),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(sh_cat)

        cat_labels = {
            "Weather warnings":                     "WEATHER",
            "Crop pest/disease alert":              "PEST / DISEASE",
            "Irrigation guidance":                  "IRRIGATION",
            "Spraying / fertilizer guidance":       "SPRAYING / FERTILIZER",
            "Harvest / storage / field operations": "HARVEST / FIELD OPS",
        }
        cat_colors = {
            "Weather warnings":                     G_ORANGE,
            "Crop pest/disease alert":              colors.HexColor("#8B0000"),
            "Irrigation guidance":                  colors.HexColor("#1565C0"),
            "Spraying / fertilizer guidance":       colors.HexColor("#6A1B9A"),
            "Harvest / storage / field operations": G_GREEN,
        }

        cat_rows  = []
        style_cmds = [
            ("TOPPADDING",    (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LINEBELOW",     (0,0), (-1,-1), 0.2, GREY_MID),
        ]
        row_idx = 0
        for cat, pts in cats.items():
            if not pts:
                continue
            label  = cat_labels.get(cat, cat.upper())
            icolor = cat_colors.get(cat, G_GREEN)
            hx     = "#%02x%02x%02x" % (
                int(icolor.red * 255),
                int(icolor.green * 255),
                int(icolor.blue * 255),
            )
            cat_rows.append([
                "",
                Paragraph(label, S(f"chl{row_idx}", fontSize=7.5,
                           fontName="Helvetica-Bold",
                           textColor=icolor)),
            ])
            style_cmds.append(("BACKGROUND", (0,row_idx), (-1,row_idx), GREY_LT))
            style_cmds.append(("LINEABOVE",  (0,row_idx), (-1,row_idx), 0.6, GREY_MID))
            row_idx += 1
            for p in pts[:3]:
                if len(p) > 180:
                    p = p[:177] + "..."
                cat_rows.append([
                    Paragraph("-", S(f"cd{row_idx}", fontSize=7.5,
                               textColor=GREY_TXT, alignment=TA_CENTER)),
                    Paragraph(p, S(f"cp{row_idx}", fontSize=7.8,
                               leading=11, textColor=BLACK)),
                ])
                row_idx += 1

        if cat_rows:
            cat_t = Table(cat_rows, colWidths=[12, PAGE_W - 12])
            cat_t.setStyle(TableStyle(style_cmds))
            story.append(cat_t)
        story.append(Spacer(1, 20))

    # ── 7. FOOTER STRIP ───────────────────────────────────────────────────────
    foot_t = Table([[
        Paragraph("Kisan Call Centre: 1800-180-1551 (Toll Free)",
                  S("pf1", fontSize=7.5, textColor=WHITE)),
        Paragraph("IMD Agrimet: imdagrimet.gov.in",
                  S("pf2", fontSize=7.5, textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("Consult your local KVK for site-specific advice.",
                  S("pf3", fontSize=7.5, textColor=WHITE, alignment=TA_RIGHT)),
    ]], colWidths=[PAGE_W / 3] * 3)
    foot_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), G_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(foot_t)

    # ── 8. DISCLAIMER + BRAND ─────────────────────────────────────────────────
    story.append(HRFlowable(width=PAGE_W, thickness=1, color=G_ORANGE,
                            spaceAfter=3, spaceBefore=4))
    story.append(Paragraph(
        "<i>DISCLAIMER: Verify all chemical names, dosages and timing with your local "
        "agriculture officer or Krishi Vigyan Kendra before application. FarmZeno advisories "
        "are derived from IMD Agrimet bulletins and live weather data and are for informational "
        "guidance only.</i>",
        disc_s))
    story.append(Paragraph(
        f'<font color="#3a7a1a"><b>Farm</b></font><font color="#e8621a"><b>Zeno</b></font>'
        f'  <font color="#aaaaaa" size="7">— IMD Agrimet + OpenWeatherMap  |  '
        f'{datetime.now().strftime("%d %b %Y")}</font>',
        foot_s))

    doc.build(story)
    buf.seek(0)
    return buf



# ════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def home():         return render_template("home.html")

@app.route("/advisory")
@login_required
def advisory():     return render_template("advisory.html")

@app.route("/disease")
@login_required
def disease():      return render_template("disease.html")

@app.route("/crop")
@login_required
def crop():         return render_template("crop.html")

# ── AUTH ROUTES ──────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get("next")
            return redirect(next_page or url_for("home"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        name     = request.form.get("name","").strip()
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        confirm  = request.form.get("confirm","")
        if not name or not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
        else:
            user = User(name=name, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user, remember=True)
            return redirect(url_for("home"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/my-reports")
@login_required
def my_reports():
    reports = SavedReport.query.filter_by(user_id=current_user.id).order_by(SavedReport.created_at.desc()).all()
    return render_template("my_reports.html", reports=reports)

@app.route("/my-reports/download/<int:report_id>")
@login_required
def download_saved_report(report_id):
    report = SavedReport.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    fpath  = os.path.join(USER_REPORTS_FOLDER, report.filename)
    if not os.path.exists(fpath):
        flash("Report file not found.", "error")
        return redirect(url_for("my_reports"))
    return send_file(fpath, as_attachment=True, download_name=report.filename)

@app.route("/my-reports/delete/<int:report_id>", methods=["POST"])
@login_required
def delete_saved_report(report_id):
    report = SavedReport.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    fpath  = os.path.join(USER_REPORTS_FOLDER, report.filename)
    if os.path.exists(fpath):
        os.remove(fpath)
    db.session.delete(report)
    db.session.commit()
    return jsonify({"success": True})

@app.route("/api/save-report", methods=["POST"])
@login_required
def api_save_report():
    """Save a generated PDF to the user's report library."""
    data = request.get_json() or {}
    rtype  = data.get("type","")          # "advisory" or "disease"
    title  = data.get("title","Report")
    pdf_b64 = data.get("pdf_b64","")
    if not pdf_b64 or rtype not in ("advisory","disease"):
        return jsonify({"success":False,"error":"Invalid data"}), 400
    import base64
    pdf_bytes = base64.b64decode(pdf_b64)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_uid = str(current_user.id)
    fname    = f"{rtype}_{safe_uid}_{ts}.pdf"
    fpath    = os.path.join(USER_REPORTS_FOLDER, fname)
    with open(fpath, "wb") as f:
        f.write(pdf_bytes)
    rec = SavedReport(user_id=current_user.id, report_type=rtype,
                      title=title[:200], filename=fname)
    db.session.add(rec)
    db.session.commit()
    return jsonify({"success":True,"report_id":rec.id})


@app.route("/api/weather-summary")
@login_required
def api_weather_summary():
    city = request.args.get("city","").strip()
    if not city: return jsonify({"success":False,"error":"City required"}), 400
    return jsonify(get_weather_summary(city))


@app.route("/api/advisory", methods=["POST"])
def api_advisory():
    try:
        data = request.get_json() or {}
        print("[DEBUG] Request received:", data)

        city = data.get("city","").strip()
        crop = data.get("crop","").strip() or None

        if not city:
            return jsonify({"error":"City required"}), 400

        print("[DEBUG] Calling advisory...")

        result = smart_farm_advisory(city, crop)

        print("[DEBUG] Advisory generated")

        result["top_advisories"] = [list(a) for a in result["top_advisories"]]

        return jsonify(result)

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    if "file" not in request.files: return jsonify({"error":"No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error":"No file selected"}), 400
    fname  = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    fpath  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(fpath)
    try:
        label, conf, top5 = predict_disease(fpath)
        return jsonify({"disease": label, "confidence": conf, "top5": top5})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(fpath): os.remove(fpath)


@app.route("/api/crop-recommend", methods=["POST"])
@login_required
def api_crop_recommend():
    data = request.get_json() or {}
    city = data.get("city","").strip()
    ph   = float(data.get("ph", DEFAULT_PH))
    if not city: return jsonify({"error":"City required"}), 400
    try:
        weather  = get_current_weather(city)
        forecast = get_24h_forecast_summary(city)
        rec      = get_ml_crop_recommendations(weather, forecast, ph)
        return jsonify({"city": city, "weather": weather, "forecast": forecast, "recommendations": rec})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-report")
@login_required
def api_download_report():
    city = request.args.get("city","").strip()
    crop = request.args.get("crop","").strip() or None
    if not city: return jsonify({"error":"City required"}), 400
    try:
        result = smart_farm_advisory(city, crop)
        result["top_advisories"] = [list(a) for a in result["top_advisories"]]
        buf  = create_advisory_pdf(result)
        name = f"FarmZeno_Advisory_{city}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return send_file(buf, as_attachment=True, download_name=name, mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/disease-report", methods=["POST"])
@login_required
def api_disease_report():
    """Generate a PDF report for plant disease detection result with FarmZeno logo."""
    import base64 as b64mod
    data       = request.get_json() or {}
    result     = data.get("result", {})
    image_b64  = data.get("image_b64", "")

    if not result:
        return jsonify({"error": "No result data provided"}), 400

    try:
        buf = create_disease_pdf(result, image_b64)
        name = f"FarmZeno_Disease_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, as_attachment=True, download_name=name, mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_disease_pdf(result: dict, image_b64: str = "") -> BytesIO:
    import base64 as b64mod
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=30, bottomMargin=30)

    styles = getSampleStyleSheet()
    PAGE_W = 495

    normal = ParagraphStyle("normal", fontSize=10, leading=14)
    title = ParagraphStyle("title", alignment=TA_CENTER, fontSize=16,
                           fontName="Helvetica-Bold", textColor=colors.black)

    story = []

    # HEADER
    story.append(Paragraph(datetime.now().strftime("%d %B %Y | %H:%M"), styles["Normal"]))
    story.append(_brand_table(PAGE_W))
    story.append(Paragraph("Plant Disease Detection Report", title))
    story.append(Spacer(1, 15))

    # IMAGE
    if image_b64 and "base64," in image_b64:
        if image_b64 and "base64," in image_b64: 
            try: 
                img = RLImage(BytesIO(b64mod.b64decode(image_b64.split("base64,")[1])), width=6*cm, height=6*cm) 
                img.hAlign = "CENTER" 
                story.append(img) 
                story.append(Spacer(1, 15)) 
            except: 
                pass

    # DATA
    disease = result.get("disease", "Unknown")
    conf = round(float(result.get("confidence", 0)), 2)
    top5 = result.get("top5", [])
    weather = result.get("weather", {})

    d_name, d_treat, d_prevent = _get_disease_advice(disease)
    is_healthy = "healthy" in disease.lower()

    # RESULT BOX
    table = Table([
        ["Disease", disease],
        ["Confidence", f"{conf}%"],
        ["Type", d_name]
    ], colWidths=[150, 300])

    table.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),1,colors.green),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.grey),
        ("PADDING",(0,0),(-1,-1),6)
    ]))

    story.append(table)
    story.append(Spacer(1, 12))

    # CONFIDENCE BAR
    filled = int(PAGE_W * conf / 100)
    bar = Table([["",""]], colWidths=[filled, PAGE_W-filled])
    bar.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), colors.HexColor("#E8621A")),
        ("BACKGROUND",(1,0),(1,0), colors.lightgrey)
    ]))
    color = ParagraphStyle(
    name="color_style",          # ✅ REQUIRED
    parent=normal,
    textColor=colors.HexColor("#E8621A"),   # ✅ comma added
)

    story.append(Paragraph("<b>Confidence Level</b>", color))
    story.append(bar)
    story.append(Spacer(1, 10))


    # WEATHER
    if weather:
        story.append(Paragraph("<b>Field Conditions</b>", normal))
        story.append(Paragraph(
            f"Temp: {weather.get('temp','-')}°C | "
            f"Humidity: {weather.get('humidity','-')}% | "
            f"Wind: {weather.get('wind_speed','-')} m/s",
            normal))
        story.append(Spacer(1, 10))

    # TREATMENT
    story.append(Paragraph("<b>Treatment</b>", normal))
    story.append(Paragraph(d_treat, normal))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Prevention</b>", normal))
    story.append(Paragraph(d_prevent, normal))
    story.append(Spacer(1, 10))

    # CHECKLIST
    story.append(Paragraph("<b>Immediate Actions</b>", normal))
    checklist = [
        "Inspect nearby plants",
        "Remove infected leaves",
        "Start treatment immediately",
        "Avoid overwatering",
        "Monitor daily"
    ]
    for c in checklist:
        story.append(Paragraph(f"• {c}", normal))

    story.append(Spacer(1, 10))

    # RISK
    if not is_healthy:
        story.append(Paragraph("<b>Risk:</b> Disease can spread and reduce yield.", normal))

    story.append(Spacer(1, 10))

    # TOP 5
    if top5:
        data = [["Rank","Prediction","Confidence"]]
        for i,(d,p) in enumerate(top5,1):
            data.append([str(i), d, f"{p}%"])

        t = Table(data)
        t.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.green),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white)
        ]))
        center = ParagraphStyle(
        "center",
        parent=normal,
        alignment=TA_CENTER,
        textColor = colors.HexColor("#E8621A")
        )
        story.append(Paragraph("<b>Top Predictions</b>", center))
        story.append(t)

    # FOOTER
    story.append(Spacer(1, 15))
    story.append(_orange_rule(PAGE_W))

    # DISCLAIMER (RED)
    center_red = ParagraphStyle(
    "center_red",
    parent=normal,
    alignment=TA_CENTER,
    textColor=colors.red
)
    story.append(Paragraph(
    "<b>Disclaimer:</b> This report is AI-generated and should be verified by an agricultural expert.",
    center_red
))

    doc.build(story)
    buf.seek(0)
    return buf


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
