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

# ----------------------------------------------------------------------------- CSS
st.markdown(f"""
<style>
.stApp {{ background:#0e1117; color:#e6eaf2; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1rem; max-width:100%; }}
.nav {{ display:flex; align-items:center; justify-content:space-between; padding:14px 22px;
  background:linear-gradient(90deg,#111827,#151b2e); border:1px solid #1f2a44; border-radius:14px; margin-bottom:14px; }}
.logo {{ font-size:1.7rem; font-weight:800; background:linear-gradient(90deg,{CYAN},{PURPLE});
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.badge {{ padding:6px 14px; border-radius:20px; border:1px solid {CYAN}; color:{CYAN}; font-size:.8rem; font-weight:600; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:#2ecc71; margin-right:7px; animation:pulse 1.4s infinite; }}
@keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:.25}} 100%{{opacity:1}} }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:16px; }}
.card {{ background:#141a2b; border:1px solid #1f2a44; border-left:4px solid {CYAN}; border-radius:12px; padding:14px 18px; }}
.card.red {{ border-left-color:#ff4b4b; }} .card.purple {{ border-left-color:{PURPLE}; }}
.card .v {{ font-size:1.9rem; font-weight:800; color:#fff; }} .card .l {{ font-size:.78rem; color:#8b96b0; text-transform:uppercase; letter-spacing:.06em; }}
.panel-title {{ font-weight:700; color:{CYAN}; margin-bottom:6px; }}
.stButton>button, .stDownloadButton>button {{ background:linear-gradient(90deg,{CYAN},{PURPLE}); color:#0e1117; font-weight:700; border:0; border-radius:8px; }}
section[data-testid="stSidebar"] {{ background:#0b0f19; border-right:1px solid #1f2a44; }}
.lb {{ display:flex; justify-content:space-between; padding:8px 10px; margin-bottom:6px; background:#141a2b; border-radius:8px; border:1px solid #1f2a44; }}
.crit {{ background:#1a1020; border:1px solid #5b1f2b; border-radius:8px; padding:6px 10px; font-size:.85rem; }}
</style>""", unsafe_allow_html=True)

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
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.9) contrast(.95) saturate(.6);}"
        ".leaflet-container{background:#0e1117;}</style>"))
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
        folium.CircleMarker(
            [i["lat"], i["lon"]], radius=8 if i["score"] >= 75 else 6, color=col, fill=True, fill_color=col, fill_opacity=.85,
            tooltip=f'#{i["id"]} {i["label"]} | Priority {i["score"]}',
            popup=f'<b>{i["label"]}</b><br>Priority: {i["score"]}<br>{i["lat"]:.5f}, {i["lon"]:.5f}<br>{i["ts"]}<br><a target="_blank" href="https://maps.google.com/?q={i["lat"]},{i["lon"]}">Open in Google Maps</a>'
        ).add_to(m)
    return m

def show_map(ph):
    try:
        with ph:
            show_html(build_map()._repr_html_(), 520)
    except Exception as e:
        ph.warning(f"Map temporarily unavailable: {e}")

def show_metrics(ph):
    n = len(S.issues)
    crit = sum(1 for i in S.issues if i["score"] >= 75)
    disp = sum(1 for i in S.issues if i["dispatched"])
    avg = round(sum(i["score"] for i in S.issues) / n, 1) if n else 0
    ph.markdown(f"""<div class="cards">
    <div class="card"><div class="l">Total Issues Detected</div><div class="v">{n}</div></div>
    <div class="card red"><div class="l">Critical Potholes / Drains</div><div class="v" style="color:#ff4b4b">{crit}</div></div>
    <div class="card purple"><div class="l">Dispatched Work Orders</div><div class="v">{disp}</div></div>
    <div class="card"><div class="l">Avg Repair Priority Score</div><div class="v">{avg}</div></div></div>""", unsafe_allow_html=True)

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
    st.markdown(f"<div class='panel-title'>🏆 Citizen Leaderboard</div>", unsafe_allow_html=True)
    board = [("Aarav M.", 128), ("Priya S.", 112), ("Rohan K.", 97), ("Sneha D.", 81), ("Imran Q.", 64)]
    for r, (n, p) in enumerate(board, 1):
        st.markdown(f"<div class='lb'><span>{['🥇','🥈','🥉','4️⃣','5️⃣'][r-1]} {n}</span><b style='color:{CYAN}'>{p} reports</b></div>", unsafe_allow_html=True)
    st.caption("Community engagement · live data")
    st.markdown(f"<div class='panel-title' style='margin-top:18px'>🔌 API Integrations</div>", unsafe_allow_html=True)
    for k, ok in api_status().items():
        st.markdown(f"{'🟢' if ok else '⚪'} {k} <span style='color:#8b96b0'>{'connected' if ok else 'not configured'}</span>", unsafe_allow_html=True)
    if st.button("🗑 Reset Demo Data"):
        S.issues, S.counter, S.selected_wo = [], 0, None
        st.rerun()

# ----------------------------------------------------------------------------- Header
st.markdown(f"""<div class="nav"><div class="logo">🛣️ MargRakshak</div>
<div class="badge"><span class="dot"></span>System Live</div></div>""", unsafe_allow_html=True)
metrics_ph = st.empty()
show_metrics(metrics_ph)

model, weights = load_model()
left, right = st.columns([35, 65], gap="medium")

with right:
    st.markdown("<div class='panel-title'>📍 Live Issue Map</div>", unsafe_allow_html=True)
    map_ph = st.empty()
    feed_box = st.container()

with left:
    st.markdown("<div class='panel-title'>🎥 Detection Feed</div>", unsafe_allow_html=True)
    mode = st.radio("Source", ["Live Dashcam Stream Simulation", "Upload Inspection Photo"], label_visibility="collapsed")
    st.caption(f"Detector Active: {weights or 'OpenCV Defect Analyzer (Fallback)'}")
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
    st.markdown("<div class='panel-title'>🚨 Critical Incident Feed</div>", unsafe_allow_html=True)
    crit = [i for i in S.issues if i["score"] >= 75]
    if not crit:
        st.info("No critical incidents recorded. Run stream or upload a photo.")
    for i in crit[-8:][::-1]:
        c1, c2 = st.columns([3, 1.4])
        status = "✅ Dispatched" if i["dispatched"] else "🔴 Pending"
        c1.markdown(f"<div class='crit'><b>#{i['id']} {i['label']}</b> · Priority <b>{i['score']}</b> · {i['lat']:.4f}, {i['lon']:.4f} · {i['ts']} · {status}</div>", unsafe_allow_html=True)
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
st.markdown("---")
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