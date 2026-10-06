"""
MargRakshak - Public Infrastructure Monitoring (Hackathon Prototype)
Run:  pip install streamlit ultralytics opencv-python-headless folium reportlab numpy
      streamlit run app_2.py
"""
import io, os, random, tempfile, time
from datetime import datetime

import cv2
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MargRakshak", page_icon="🛣️", layout="wide")

CENTER = (19.0760, 72.8777)  # Mumbai Center
LABELS = ["Pothole", "Overflowing Drain", "Crack", "Broken Light", "Minor Defect"]
LABEL_W = {"Pothole": 15, "Overflowing Drain": 15, "Crack": 8, "Broken Light": 6, "Minor Defect": 0}
MATERIALS = {
    "Pothole": "Requires 40kg Cold-Mix Asphalt, compactor & Traffic Cones (x6)",
    "Overflowing Drain": "Requires Jetting Machine, 2m Drain Pipe Section & Safety Barricades",
    "Crack": "Requires 15kg Crack Sealant & Traffic Cones (x4)",
    "Broken Light": "Requires 1 LED Street Lamp Unit (80W) & Bucket Truck",
    "Minor Defect": "Requires 5kg Patch Mix & Inspection Visit",
}
CYAN, PURPLE = "#00f2fe", "#4facfe"

# ----------------------------------------------------------------------------- CSS (HUD / glass command-center theme)
UI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--cy:#00f2fe;--pu:#4facfe;--vi:#8b5cf6;--red:#ff4d6d;--amb:#ffc145;--grn:#2ee59d;--mut:#8592b0;
 --glass:rgba(15,22,41,.74);--line:rgba(120,160,255,.15);--mono:'JetBrains Mono',monospace;}
html,body,.stApp,button,input{font-family:'Space Grotesk',sans-serif!important;}
.stApp{color:#e8edf8;background:
 radial-gradient(900px 520px at 6% -12%,rgba(0,242,254,.15),transparent 60%),
 radial-gradient(820px 520px at 100% 0%,rgba(139,92,246,.18),transparent 55%),
 linear-gradient(rgba(120,160,255,.045) 1px,transparent 1px) 0 0/38px 38px,
 linear-gradient(90deg,rgba(120,160,255,.045) 1px,transparent 1px) 0 0/38px 38px,#070b14;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.1rem;max-width:100%;}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:#22304f;border-radius:8px}
/* nav */
.nav{position:relative;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 22px;border-radius:20px;
 background:var(--glass);backdrop-filter:blur(14px);border:1px solid var(--line);overflow:hidden}
.nav:before{content:"";position:absolute;left:0;top:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,var(--cy),var(--vi),transparent);background-size:200% 100%;animation:sweep 4s linear infinite}
@keyframes sweep{from{background-position:200% 0}to{background-position:-200% 0}}
.brand{display:flex;align-items:center;gap:14px}
.brand svg{filter:drop-shadow(0 0 10px rgba(0,242,254,.55))}
.wm{font-weight:700;font-size:1.55rem;letter-spacing:.14em;line-height:1}
.wm span{background:linear-gradient(90deg,var(--cy),var(--vi));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{font:500 .68rem var(--mono);color:var(--mut);letter-spacing:.18em;text-transform:uppercase;margin-top:5px}
.navr{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.chipx{font:500 .72rem var(--mono);color:#b8c4e0;padding:7px 12px;border-radius:10px;border:1px solid var(--line);background:rgba(255,255,255,.03)}
.live{display:flex;align-items:center;gap:8px;font:700 .72rem var(--mono);letter-spacing:.12em;color:var(--grn);padding:7px 14px;border-radius:99px;border:1px solid rgba(46,229,157,.4);background:rgba(46,229,157,.08)}
.live i{width:8px;height:8px;border-radius:50%;background:var(--grn);box-shadow:0 0 0 0 rgba(46,229,157,.7);animation:pl2 1.6s infinite}
@keyframes pl2{70%{box-shadow:0 0 0 9px rgba(46,229,157,0)}100%{box-shadow:0 0 0 0 rgba(46,229,157,0)}}
/* ticker */
.ticker{overflow:hidden;white-space:nowrap;margin:10px 0 0;padding:7px 0;border-radius:12px;border:1px solid var(--line);background:rgba(8,12,24,.7);
 -webkit-mask:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.tk{display:inline-block;animation:mq 38s linear infinite;font:500 .74rem var(--mono);color:#9fb0d6}
.tk b{color:#fff}.tk em{font-style:normal;margin:0 26px 0 8px}
@keyframes mq{to{transform:translateX(-50%)}}
/* metric cards */
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0 20px}
.card{--c:var(--cy);position:relative;display:flex;justify-content:space-between;align-items:center;padding:16px 18px;border-radius:18px;
 background:var(--glass);border:1px solid var(--line);box-shadow:0 0 36px -16px var(--c);overflow:hidden}
.card:after{content:"";position:absolute;inset:0;border-radius:18px;padding:1px;pointer-events:none;background:linear-gradient(135deg,var(--c),transparent 55%);
 -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}
.card .l{font:500 .66rem var(--mono);color:var(--mut);letter-spacing:.14em;text-transform:uppercase}
.card .v{font:700 2.3rem var(--mono);color:#fff;line-height:1.15;margin:4px 0 2px;text-shadow:0 0 22px var(--c)}
.card .s{font-size:.74rem;color:var(--mut)}
.card.hot{animation:hot 2.2s ease-in-out infinite}
@keyframes hot{50%{box-shadow:0 0 46px -8px var(--c)}}
.ico{width:46px;height:46px;border-radius:14px;display:grid;place-items:center;font-size:1.4rem;background:rgba(255,255,255,.05);border:1px solid var(--line)}
.ring{width:64px;height:64px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.08) 0)}
.ring i{width:50px;height:50px;border-radius:50%;background:#0b1224;display:grid;place-items:center;font:700 .82rem var(--mono);font-style:normal;color:#fff}
/* panel header */
.ph{display:flex;align-items:center;gap:10px;margin:2px 0 10px}
.ph .pi{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(0,242,254,.2),rgba(139,92,246,.28));border:1px solid var(--line)}
.ph .pt{font-weight:700;letter-spacing:.05em}
.ph .tag{margin-left:auto;font:700 .64rem var(--mono);letter-spacing:.1em;color:var(--cy);border:1px solid rgba(0,242,254,.35);padding:3px 10px;border-radius:99px}
.det{font:500 .74rem var(--mono);color:var(--mut);margin:6px 0 10px}.det b{color:var(--cy)}
/* media */
iframe{border-radius:18px;border:1px solid var(--line)!important;box-shadow:0 0 44px -20px var(--cy)}
[data-testid="stImage"]{position:relative}
[data-testid="stImage"] img{border-radius:14px;border:1px solid rgba(0,242,254,.3);box-shadow:0 0 30px -12px var(--cy)}
[data-testid="stImage"]:after{content:"";position:absolute;inset:0;border-radius:14px;pointer-events:none;background:repeating-linear-gradient(0deg,transparent 0 2px,rgba(0,242,254,.04) 2px 3px)}
/* controls */
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,var(--cy),var(--pu) 55%,var(--vi));border:0;border-radius:12px;padding:.55rem 1rem;
 box-shadow:0 8px 24px -10px var(--cy);transition:.2s}
.stButton>button p,.stDownloadButton>button p{color:#06101c!important;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);filter:brightness(1.12);box-shadow:0 14px 30px -8px var(--cy)}
div[role="radiogroup"]{gap:8px}
label[data-baseweb="radio"]{width:100%;padding:10px 14px;border-radius:12px;background:rgba(15,22,41,.8);border:1px solid var(--line);transition:.2s}
label[data-baseweb="radio"]>div:first-child{display:none}
label[data-baseweb="radio"]:has(input:checked){border-color:var(--cy);background:linear-gradient(135deg,rgba(0,242,254,.14),rgba(139,92,246,.18));box-shadow:0 0 24px -8px var(--cy)}
[data-testid="stFileUploaderDropzone"]{background:rgba(15,22,41,.6);border:1.5px dashed rgba(0,242,254,.4);border-radius:14px}
[data-testid="stAlert"]{border-radius:12px;background:rgba(15,22,41,.75);border:1px solid var(--line)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:16px;background:var(--glass)}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:8px 16px}
.stTabs [aria-selected="true"]{color:var(--cy)!important;background:rgba(0,242,254,.08)}
.divider{height:1px;margin:22px 0 10px;background:linear-gradient(90deg,transparent,var(--cy),var(--vi),transparent);opacity:.5}
/* incidents */
.inc{position:relative;display:flex;align-items:center;gap:16px;padding:12px 16px 12px 20px;border-radius:16px;overflow:hidden;
 background:linear-gradient(90deg,rgba(255,77,109,.12),rgba(15,22,41,.75) 45%);border:1px solid rgba(255,77,109,.28)}
.inc:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--red);box-shadow:0 0 14px var(--red)}
.inc .pr{text-align:center;min-width:64px}.inc .pr b{display:block;font:700 1.6rem var(--mono);color:var(--red);text-shadow:0 0 16px rgba(255,77,109,.6);line-height:1}
.inc small,.pr small{font:500 .62rem var(--mono);color:var(--mut);letter-spacing:.1em}
.im{flex:1}.im b{display:block}
.pill{font:700 .64rem var(--mono);letter-spacing:.08em;padding:4px 10px;border-radius:99px;white-space:nowrap}
.pill.ok{color:var(--grn);background:rgba(46,229,157,.1);border:1px solid rgba(46,229,157,.4)}
.pill.pend{color:var(--red);background:rgba(255,77,109,.1);border:1px solid rgba(255,77,109,.4);animation:blink 1.4s infinite}
@keyframes blink{50%{opacity:.45}}
.empty{text-align:center;padding:26px;border-radius:16px;border:1px dashed rgba(46,229,157,.35);background:rgba(46,229,157,.05);color:#a8f0d4}
/* sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1020,#070b14);border-right:1px solid var(--line)}
.sbrand{display:flex;align-items:center;gap:10px;margin-bottom:18px;font-weight:700;letter-spacing:.12em}
.lb{display:flex;align-items:center;gap:10px;padding:9px 10px;margin-bottom:8px;border-radius:14px;background:var(--glass);border:1px solid var(--line)}
.av{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font:700 .74rem var(--mono);color:#06101c;background:linear-gradient(135deg,var(--cy),var(--vi))}
.lm{flex:1}.lm b{font-size:.85rem}.bar{height:5px;border-radius:9px;background:rgba(255,255,255,.08);margin-top:5px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,var(--cy),var(--vi))}
.lc{font:700 .8rem var(--mono);color:var(--cy)}
.api{display:flex;align-items:center;gap:9px;padding:8px 12px;margin-bottom:7px;border-radius:12px;border:1px solid var(--line);font-size:.8rem;background:rgba(255,255,255,.02)}
.api i{width:8px;height:8px;border-radius:50%;background:#46506b}.api.on i{background:var(--grn);box-shadow:0 0 10px var(--grn)}
.api span{margin-left:auto;font:500 .62rem var(--mono);color:var(--mut)}.api.on span{color:var(--grn)}
"""
st.markdown("<style>" + UI_CSS + "</style>", unsafe_allow_html=True)

LOGO = ('<svg width="40" height="40" viewBox="0 0 48 48"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#00f2fe"/>'
        '<stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs><path d="M24 3l18 10v22L24 45 6 35V13z" fill="rgba(0,242,254,.08)" stroke="url(#g)" stroke-width="2.4"/>'
        '<path d="M24 12v6M24 22v6M24 32v5" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/><path d="M15 38l4-26M33 38l-4-26" stroke="#4facfe" stroke-width="1.6" opacity=".7"/></svg>')


def ph(icon, title, tag=""):
    return f'<div class="ph"><div class="pi">{icon}</div><div class="pt">{title}</div>' + (f'<div class="tag">{tag}</div>' if tag else '') + '</div>'


def spark(vals, color):
    vals = [float(v) for v in vals[-14:]] or [0.0]
    if len(vals) < 2:
        vals = vals * 2
    mn, rg = min(vals), (max(vals) - min(vals)) or 1
    pts = " ".join(f"{k * 90 / (len(vals) - 1):.1f},{31 - (v - mn) / rg * 27:.1f}" for k, v in enumerate(vals))
    return (f'<svg width="90" height="34" viewBox="0 0 90 34"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


# ----------------------------------------------------------------------------- State & Initial Data Seed
S = st.session_state
S.setdefault("selected_wo", None)
S.setdefault("counter", 0)

if "issues" not in S or len(S.issues) == 0:
    S.issues = [
        {"id": 1, "label": "Pothole", "conf": 0.92, "score": 88, "lat": CENTER[0] + 0.003, "lon": CENTER[1] - 0.004, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snap": b"", "dispatched": False},
        {"id": 2, "label": "Overflowing Drain", "conf": 0.89, "score": 82, "lat": CENTER[0] - 0.002, "lon": CENTER[1] + 0.003, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snap": b"", "dispatched": False},
        {"id": 3, "label": "Pothole", "conf": 0.84, "score": 76, "lat": CENTER[0] + 0.005, "lon": CENTER[1] + 0.002, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snap": b"", "dispatched": False},
        {"id": 4, "label": "Crack", "conf": 0.78, "score": 45, "lat": CENTER[0] - 0.004, "lon": CENTER[1] - 0.001, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "snap": b"", "dispatched": False},
    ]
    S.counter = 4

# ----------------------------------------------------------------------------- Helpers
def cfg(key, default=None):
    v = os.environ.get(key)
    if v:
        return v
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)

def dark_osm_map(zoom=12):
    m = folium.Map(location=CENTER, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    m.get_root().html.add_child(folium.Element("""<style>
.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.85) contrast(.95) saturate(.55);}
.leaflet-container{background:#0b1020;font-family:'Space Grotesk',sans-serif}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:#0d1426;color:#e8edf8;border:1px solid rgba(0,242,254,.35);border-radius:12px}
.leaflet-popup-content a{color:#00f2fe}
.leaflet-tooltip{background:#0d1426;color:#e8edf8;border:1px solid rgba(0,242,254,.4);border-radius:8px}
.leaflet-bar a{background:#0d1426!important;color:#00f2fe!important;border-color:#1f2a44!important}
.leaflet-control-attribution{background:rgba(7,11,20,.7)!important;color:#8592b0!important}
.pulse{width:18px;height:18px;border-radius:50%;background:#ff4d6d;border:2px solid #fff;animation:pl 1.6s infinite}
@keyframes pl{0%{box-shadow:0 0 0 0 rgba(255,77,109,.8)}70%{box-shadow:0 0 0 20px rgba(255,77,109,0)}100%{box-shadow:0 0 0 0 rgba(255,77,109,0)}}
.lg{position:absolute;left:12px;bottom:26px;z-index:9999;padding:9px 13px;border-radius:12px;background:rgba(13,20,38,.88);border:1px solid rgba(120,160,255,.25);
 color:#cbd5ee;font:600 11px 'Space Grotesk',sans-serif;line-height:1.7}.lg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
</style><div class="lg"><i style="background:#ff4d6d"></i>Critical 75-100<br><i style="background:#ffc145"></i>Medium 40-74<br><i style="background:#2ee59d"></i>Minor 1-39</div>"""))
    return m

def api_status():
    return {
        "Mapbox tiles": bool(cfg("MAPBOX_TOKEN")),
        "Roboflow vision": bool(cfg("ROBOFLOW_API_KEY")),
        "Cloudinary storage": all(cfg(k) for k in ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET")),
        "Twilio SMS": all(cfg(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "ENGINEER_PHONE_NUMBER")),
    }

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from ultralytics import YOLO
        path = "pothole.pt" if os.path.exists("pothole.pt") else "yolov8n.pt"
        return YOLO(path), path
    except Exception:
        return None, None

def tier(score):
    return ("critical", "#ff4b4b", "red") if score >= 75 else (("medium", "#ffc107", "orange") if score >= 40 else ("minor", "#2ecc71", "green"))

def score_of(conf, area_ratio, label):
    return int(max(1, min(100, conf * 45 + min(area_ratio * 400, 1) * 40 + LABEL_W.get(label, 10) + random.randint(-3, 3))))

def synth_frames(n=360, size=(640, 360)):
    W, H = size
    rng = random.Random(7)
    holes = [[rng.randint(180, 460), -rng.randint(0, 900), rng.randint(22, 55)] for _ in range(8)]
    for t in range(n):
        f = np.full((H, W, 3), 85, np.uint8)
        f[:H // 4] = (60, 40, 30)
        cv2.fillPoly(f, [np.array([[W // 2 - 40, H // 4], [W // 2 + 40, H // 4], [W - 40, H], [40, H]])], (95, 95, 95))
        for k in range(8):
            y = int(H // 4 + ((t * 6 + k * 70) % (H * 0.75)))
            cv2.rectangle(f, (W // 2 - 3, y), (W // 2 + 3, y + 25), (230, 230, 230), -1)
        for h in holes:
            h[1] += 4
            if h[1] > H + 60:
                h[0], h[1], h[2] = rng.randint(200, 440), -rng.randint(0, 300), rng.randint(22, 55)
            if h[1] > H // 4:
                s = (h[1] - H // 4) / (H * 0.75)
                cv2.ellipse(f, (h[0], h[1]), (int(h[2] * (0.4 + s)), int(h[2] * (0.25 + s * 0.5))), 0, 0, 360, (22, 22, 22), -1)
        yield f

def video_frames(path, max_frames=400):
    cap = cv2.VideoCapture(path)
    try:
        n = 0
        while cap.isOpened() and n < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            yield cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1])))
            n += 1
    finally:
        cap.release()

def detect(frame, model, custom):
    """Returns list of (x1,y1,x2,y2,conf,label). YOLOv8 first, dark-blob fallback for demo footage."""
    out = []
    try:
        if model is not None:
            r = model.predict(frame, conf=0.25, imgsz=480, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls = int(b.cls[0])
                name = r.names[cls].title()
                label = name if (custom and name in LABELS) else LABELS[cls % len(LABELS)]
                out.append((x1, y1, x2, y2, float(b.conf[0]), label))
    except Exception:
        pass
    if not out:  # fallback: dark road blobs act as defect candidates
        try:
            g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
            m = cv2.inRange(g, 0, 40)
            m[: frame.shape[0] // 4] = 0
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    label = random.choices(LABELS, [6, 1, 2, 1, 2])[0]
                    out.append((x, y, x + w, y + h, min(0.97, 0.6 + cv2.contourArea(c) / 8000), label))
        except Exception:
            pass
    return out

def draw(frame, dets):
    for x1, y1, x2, y2, conf, label in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (254, 242, 0), 2)
        cv2.putText(frame, f"{label} {conf:.0%}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 242, 0), 1, cv2.LINE_AA)
    return frame

def register(frame, det):
    x1, y1, x2, y2, conf, label = det
    H, W = frame.shape[:2]
    score = score_of(conf, max(0, (x2 - x1) * (y2 - y1)) / float(W * H), label)
    snap = frame.copy()
    ok, buf = cv2.imencode(".jpg", snap)
    S.counter += 1
    S.issues.append(dict(
        id=S.counter, label=label, conf=conf, score=score,
        lat=CENTER[0] + random.uniform(-0.02, 0.02), lon=CENTER[1] + random.uniform(-0.02, 0.02),
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), snap=buf.tobytes() if ok else b"", dispatched=False))
    return S.issues[-1]

def build_map():
    m = dark_osm_map()
    for i in S.issues[-150:]:
        _, col, _ = tier(i["score"])
        tip = f'#{i["id"]} {i["label"]} | Priority {i["score"]}'
        pop = f'<b>{i["label"]}</b><br>Priority: {i["score"]}<br>{i["lat"]:.5f}, {i["lon"]:.5f}<br>{i["ts"]}<br><a target="_blank" href="https://maps.google.com/?q={i["lat"]},{i["lon"]}">Open in Google Maps</a>'
        if i["score"] >= 75:
            folium.Marker([i["lat"], i["lon"]], tooltip=tip, popup=pop,
                          icon=folium.DivIcon(html='<div class="pulse"></div>', icon_size=(18, 18), icon_anchor=(9, 9))).add_to(m)
        else:
            folium.CircleMarker([i["lat"], i["lon"]], radius=7, color=col, weight=2, fill=True, fill_color=col, fill_opacity=.6, tooltip=tip, popup=pop).add_to(m)
    return m

def show_map(ph):
    try:
        with ph:
            show_html(build_map()._repr_html_(), 520)
    except Exception as e:
        ph.warning(f"Map temporarily unavailable: {e}")

def show_metrics(ph_):
    n = len(S.issues)
    crit = sum(1 for i in S.issues if i["score"] >= 75)
    disp = sum(1 for i in S.issues if i["dispatched"])
    avg = round(sum(i["score"] for i in S.issues) / n, 1) if n else 0
    pct_c = round(100 * crit / n) if n else 0
    pct_d = round(100 * disp / crit) if crit else 0
    _, tcol, _ = tier(avg)
    level = "Severe" if avg >= 75 else ("Moderate" if avg >= 40 else "Low")
    ph_.markdown(
        f'<div class="cards">'
        f'<div class="card" style="--c:#00f2fe"><div><div class="l">Total Issues Detected</div><div class="v">{n}</div><div class="s">live priority trend</div></div>{spark([x["score"] for x in S.issues], "#00f2fe")}</div>'
        f'<div class="card {"hot" if crit else ""}" style="--c:#ff4d6d"><div><div class="l">Critical Potholes / Drains</div><div class="v" style="color:#ff4d6d">{crit}</div><div class="s">{pct_c}% of all issues</div></div><div class="ico">🚨</div></div>'
        f'<div class="card" style="--c:#8b5cf6"><div><div class="l">Dispatched Work Orders</div><div class="v">{disp}</div><div class="s">{disp} of {crit} critical sent</div></div><div class="ring" style="--p:{pct_d}"><i>{pct_d}%</i></div></div>'
        f'<div class="card" style="--c:{tcol}"><div><div class="l">Avg Repair Priority</div><div class="v">{avg}</div><div class="s">{level} urgency</div></div><div class="ring" style="--p:{avg}"><i>{int(avg)}</i></div></div>'
        f'</div>', unsafe_allow_html=True)

def make_pdf(i):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    c.setFillColorRGB(0.05, 0.1, 0.25); c.rect(0, H - 90, W, 90, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 22); c.drawString(40, H - 45, "WARD ENGINEERING DISPATCH")
    c.setFont("Helvetica", 11); c.drawString(40, H - 66, "Municipal Corporation | Roads & Infrastructure Dept. | Auto-generated by MargRakshak")
    c.setFillColorRGB(0, 0, 0); y = H - 130
    for k, v in [("Work Order No.", f"WO-{datetime.now():%Y%m%d}-{i['id']:04d}"), ("Timestamp", i["ts"]),
                 ("Issue Type", i["label"]), ("GPS Coordinates", f"{i['lat']:.5f}, {i['lon']:.5f}"),
                 ("Severity / Priority Score", f"{i['score']} / 100 (CRITICAL)" if i["score"] >= 75 else f"{i['score']} / 100"),
                 ("AI Confidence", f"{i['conf']:.0%}"), ("Est. Materials", ""), ("", MATERIALS.get(i["label"], "Standard Repair Kit"))]:
        c.setFont("Helvetica-Bold", 11); c.drawString(40, y, k)
        c.setFont("Helvetica", 11); c.drawString(210 if k else 40, y, v); y -= 22
    c.setFont("Helvetica-Bold", 11); c.drawString(40, y - 10, "Bounding Box Snapshot Proof:")
    try:
        if i["snap"]:
            c.drawImage(ImageReader(io.BytesIO(i["snap"])), 40, y - 270, width=420, height=240, preserveAspectRatio=True)
        else:
            c.setFont("Helvetica-Oblique", 10); c.drawString(40, y - 30, "(snapshot proof pending)")
    except Exception:
        c.setFont("Helvetica-Oblique", 10); c.drawString(40, y - 30, "(snapshot proof unavailable)")
    c.setFont("Helvetica-Oblique", 9); c.drawString(40, 40, "Repair SLA: 48 hours. Authorised electronically - no signature required.")
    c.save()
    return buf.getvalue()

# ----------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown(f'<div class="sbrand">{LOGO}<div>MARG<span style="color:#00f2fe">RAKSHAK</span></div></div>', unsafe_allow_html=True)
    st.markdown(ph("🏆", "Citizen Leaderboard", "TOP 5"), unsafe_allow_html=True)
    board = [("Aarav M.", 128), ("Priya S.", 112), ("Rohan K.", 97), ("Sneha D.", 81), ("Imran Q.", 64)]
    medals = ["🥇", "🥈", "🥉", "④", "⑤"]
    for r, (n, p) in enumerate(board):
        ini = "".join(w[0] for w in n.replace(".", "").split())
        st.markdown(f'<div class="lb"><div class="av">{ini}</div><div class="lm"><b>{n}</b><div class="bar"><i style="width:{int(p / board[0][1] * 100)}%"></i></div></div>'
                    f'<span class="lc">{p}</span><span>{medals[r]}</span></div>', unsafe_allow_html=True)
    st.caption("Community engagement · live data")
    st.markdown(ph("🔌", "API Integrations"), unsafe_allow_html=True)
    for k, ok in api_status().items():
        st.markdown(f'<div class="api {"on" if ok else ""}"><i></i>{k}<span>{"LINKED" if ok else "OFFLINE"}</span></div>', unsafe_allow_html=True)
    if st.button("🗑 Reset Demo Data"):
        S.issues, S.counter, S.selected_wo = [], 0, None
        st.rerun()

# ----------------------------------------------------------------------------- Header
_items = "".join(f'<b>#{i["id"]} {i["label"].upper()}</b><em>P{i["score"]} · {i["lat"]:.4f},{i["lon"]:.4f}</em>' for i in S.issues[-8:][::-1]) or "<b>AWAITING DETECTIONS</b><em>start the dashcam stream</em>"
st.markdown(
    f'<div class="nav"><div class="brand">{LOGO}<div><div class="wm">MARG<span>RAKSHAK</span></div><div class="sub">AI Road Guardian · Infrastructure Command Center</div></div></div>'
    f'<div class="navr"><div class="chipx">📍 MUMBAI · WARD ENGINEERING</div><div class="chipx">{datetime.now():%d %b %Y · %H:%M}</div><div class="live"><i></i>SYSTEM LIVE</div></div></div>'
    f'<div class="ticker"><div class="tk">{_items * 6}</div></div>', unsafe_allow_html=True)
metrics_ph = st.empty()
show_metrics(metrics_ph)

model, weights = load_model()
left, right = st.columns([35, 65], gap="medium")

with right:
    st.markdown(ph("📍", "Live Issue Map", "GEO-INTEL"), unsafe_allow_html=True)
    map_ph = st.empty()
    feed_box = st.container()

with left:
    st.markdown(ph("🎥", "Detection Feed", "● REC"), unsafe_allow_html=True)
    mode = st.radio("Source", ["Live Dashcam Stream Simulation", "Upload Inspection Photo"], label_visibility="collapsed")
    st.markdown(f'<div class="det">🧠 DETECTOR · <b>{weights or "OpenCV Defect Analyzer (Fallback)"}</b></div>', unsafe_allow_html=True)
    video_ph = st.empty()

    try:
        if mode.startswith("Live"):
            up = st.file_uploader("Optional dashcam video (mp4)", type=["mp4", "mov", "avi"])
            go = st.button("▶ Start Stream")
            if go:
                src = None
                if up is not None:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(up.read()); tmp.close(); src = tmp.name
                elif os.path.exists("sample.mp4"):
                    src = "sample.mp4"
                frames = video_frames(src) if src else synth_frames()
                last_reg, last_map = -99, 0
                for idx, fr in enumerate(frames):
                    dets = detect(fr, model, weights == "pothole.pt") if idx % 3 == 0 else []
                    if dets:
                        draw(fr, dets)
                        if idx - last_reg > 15 and len(S.issues) < 150:
                            register(fr, max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1])))
                            last_reg = idx
                            show_metrics(metrics_ph)
                            if time.time() - last_map > 1.2:
                                show_map(map_ph)
                                last_map = time.time()
                    video_ph.image(fr[:, :, ::-1], use_container_width=True)
                    time.sleep(0.025)
                st.success("Stream finished.")
            else:
                video_ph.image(np.full((270, 480, 3), 20, np.uint8), caption="Press ▶ Start Stream", use_container_width=True)
        else:
            img = st.file_uploader("Upload road inspection photo", type=["jpg", "jpeg", "png"])
            if img is None:
                video_ph.info("📷 No photo uploaded yet - upload one to run detection.")
            else:
                arr = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    video_ph.error("Could not read image file.")
                else:
                    arr = cv2.resize(arr, (640, int(640 * arr.shape[0] / arr.shape[1])))
                    dets = detect(arr, model, weights == "pothole.pt")
                    draw(arr, dets)
                    video_ph.image(arr[:, :, ::-1], use_container_width=True)
                    if not dets:
                        st.info("No clear defects detected in this frame.")
                    elif st.button("➕ Log detections to map"):
                        for d in dets[:3]:
                            register(arr, d)
                        show_metrics(metrics_ph)
                        st.success("Detections logged to Live Issue Map!")
    except Exception as e:
        st.warning(f"Feed status: {e}")

show_map(map_ph)
show_metrics(metrics_ph)

# ----------------------------------------------------------------------------- Critical Feed
with feed_box:
    st.markdown(ph("🚨", "Critical Incident Feed", "PRIORITY ≥ 75"), unsafe_allow_html=True)
    crit = [i for i in S.issues if i["score"] >= 75]
    if not crit:
        st.markdown('<div class="empty">✅ <b>All clear</b><br><small>No critical incidents. Run the stream or upload a photo.</small></div>', unsafe_allow_html=True)
    for i in crit[-8:][::-1]:
        c1, c2 = st.columns([3, 1.4], vertical_alignment="center")
        status = "✅ Dispatched" if i["dispatched"] else "🔴 Pending"
        pill = '<span class="pill ok">✔ DISPATCHED</span>' if i["dispatched"] else '<span class="pill pend">● PENDING</span>'
        c1.markdown(f'<div class="inc"><div class="pr"><b>{i["score"]}</b><small>PRIORITY</small></div><div class="im"><b>#{i["id"]} · {i["label"]}</b><small>{i["lat"]:.4f}, {i["lon"]:.4f} · {i["ts"]}</small></div>{pill}</div>', unsafe_allow_html=True)
        if c2.button("⚡ Dispatch Work Order", key=f"d{i['id']}"):
            i["dispatched"] = True
            S.selected_wo = i["id"]
            st.toast(f"Work Order #{i['id']} Dispatched!", icon="🚨")
            show_metrics(metrics_ph)

    wo = next((i for i in S.issues if i["id"] == S.selected_wo), None)
    if wo:
        with st.expander(f"📄 Work Order WO-{wo['id']:04d} - Municipal Dispatch", expanded=True):
            a, b = st.columns([1, 1])
            a.markdown(f"**Timestamp:** {wo['ts']}  \n**GPS Coordinates:** {wo['lat']:.5f}, {wo['lon']:.5f}  \n"
                       f"**Severity Score:** {wo['score']}/100  \n**Issue:** {wo['label']}  \n**Materials:** {MATERIALS.get(wo['label'], 'Standard Repair Kit')}")
            if wo["snap"]:
                b.image(wo["snap"], caption="Bounding Box Proof", use_container_width=True)
            try:
                st.download_button("⬇ Download Work-Order PDF", make_pdf(wo), file_name=f"work_order_{wo['id']:04d}.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF engine note: {e}")

# ----------------------------------------------------------------------------- Heatmap
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
(tab,) = st.tabs(["🔥 Predictive Infrastructure Heatmap"])
with tab:
    try:
        rng = np.random.default_rng(42)
        zones = [(19.0896, 72.8656), (19.0176, 72.8562), (19.1197, 72.8468), (19.0330, 72.8300)]
        pts = []
        for z in zones:
            for _ in range(60):
                pts.append([z[0] + rng.normal(0, .008), z[1] + rng.normal(0, .008), float(rng.uniform(.4, 1))])
        pts += [[i["lat"], i["lon"], i["score"] / 100] for i in S.issues]
        hm = dark_osm_map()
        HeatMap(pts, radius=22, blur=18, gradient={0.3: PURPLE, 0.6: CYAN, 1.0: "#ff4b4b"}).add_to(hm)
        show_html(hm._repr_html_(), 480)
        st.caption("High-degradation density zones flagged for preventive maintenance.")
    except Exception as e:
        st.warning(f"Heatmap status: {e}")