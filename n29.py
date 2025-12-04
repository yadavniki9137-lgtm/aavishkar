# =====================================================================
# FINAL COMPLETE APP.PY — Verdura Lens
# YOLO Detection + Emoji Recommendations + Awareness Hub + Admin Panel
# =====================================================================

import streamlit as st
import sqlite3
from datetime import datetime, timezone
import pandas as pd
import hashlib
import os
from PIL import Image
import io
import json
import numpy as np
import cv2

# -----------------------------------------------------------
# LIGHT TEAL UI (same UI but polished)
# -----------------------------------------------------------
LIGHT_TEAL_UI = """
<style>
:root{
    --bg1:#d9fffa;
    --bg2:#c4f7f0;
    --card:rgba(255,255,255,0.55);
    --glass-border:rgba(0,120,120,0.25);
    --accent:#009688;
    --accent2:#00b3a4;
    --text-dark:#002d2d;
}
.stApp{
    background: linear-gradient(135deg, var(--bg1), var(--bg2));
    font-family: 'Inter', sans-serif;
}
.block{
    backdrop-filter: blur(16px) saturate(180%);
    background: var(--card);
    border: 1px solid var(--glass-border);
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 22px;
}
h1,h2,h3,h4,h5,h6,p,span,div{
    color: var(--text-dark) !important;
}
input, textarea, select{
    background: rgba(255,255,255,0.95) !important;
    color: var(--text-dark) !important;
    border-radius: 12px !important;
    border:1px solid var(--accent) !important;
}
div[data-baseweb="popover"] * {
    color: var(--text-dark) !important;
    background: white !important;
}
.footer{
    text-align:center;
    margin-top:25px;
    color: var(--text-dark);
}
</style>
"""
st.markdown(LIGHT_TEAL_UI, unsafe_allow_html=True)

# -----------------------------------------------------------
# DATABASE + FOLDERS
# -----------------------------------------------------------
DB_PATH = "garbage_reports.db"
IMAGES_DIR = "uploaded_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            created_at TEXT,
            role TEXT DEFAULT "user"
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            garbage_type TEXT,
            sub_type TEXT,
            condition TEXT,
            quantity INTEGER,
            weight_range TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            image_path TEXT,
            chosen_reuse TEXT,
            recommended_reuse TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS presets(
            name TEXT PRIMARY KEY,
            recommendation TEXT,
            reuse_options TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# -----------------------------------------------------------
# AUTH FUNCTIONS
# -----------------------------------------------------------
def hash_password(x):
    return hashlib.sha256(x.encode()).hexdigest()

def register_user(u, p):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES(?,?,?,?)",
                  (u, hash_password(p),
                   datetime.now(timezone.utc).isoformat(), "user"))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def verify_user(u, p):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username=?", (u,))
    row = c.fetchone()
    conn.close()
    return row and row[0] == hash_password(p)

# -----------------------------------------------------------
# SESSION
# -----------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "Report Garbage"

# -----------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------
if not st.session_state.logged_in:
    st.title("Verdura Lens — Login")

    with st.container():
        st.markdown('<div class="block">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if verify_user(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error("Invalid login")

        with tab2:
            ru = st.text_input("Create Username")
            rp = st.text_input("Create Password", type="password")
            if st.button("Create Account"):
                if register_user(ru, rp):
                    st.success("Account created! You can login now.")
                else:
                    st.error("Username already exists")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------
# LOAD YOLO (OPTIONAL)
# -----------------------------------------------------------
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# -----------------------------------------------------------
# CATEGORY MAPPING (Neutral Names)
# -----------------------------------------------------------
EXTENDED_MAPPING = {
    "bottle": "Container",
    "cup": "Container",
    "fork": "Utensil",
    "spoon": "Utensil",
    "knife": "Utensil",
    "book": "Paper",
    "box": "Paper",
    "banana": "Organic",
    "apple": "Organic",
    "orange": "Organic",
    "can": "Metal",
    "wine glass": "Container",
    "vase": "Container",
    "laptop": "Electronic",
    "cell phone": "Electronic",
    "keyboard": "Electronic",
    "tv": "Electronic",
    "chair": "General",
    "sofa": "General",
    "bench": "General",
    "person": "Unknown"
}

# -----------------------------------------------------------
# YOLO DETECTION
# -----------------------------------------------------------
def detect_yolo_and_draw(bytes_data, save_path):
    try:
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    except:
        return "General", 0, [], None

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if not YOLO_AVAILABLE:
        return "General", 0, [], np.array(img)

    results = YOLO_MODEL(img_cv, conf=0.35)
    r = results[0]

    mapping_scores = {}
    detections = []

    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = YOLO_MODEL.names[cls].lower()

        detections.append((label, conf))
        mapped = EXTENDED_MAPPING.get(label, "General")
        mapping_scores[mapped] = mapping_scores.get(mapped, 0) + conf

    if mapping_scores:
        final_type = max(mapping_scores, key=mapping_scores.get)
        final_conf = float(mapping_scores[final_type])
    else:
        final_type = "General"
        final_conf = 0.0

    return final_type, final_conf, detections, np.array(img)
# -----------------------------------------------------------
# RECOMMENDATION ENGINE (emoji-rich, neutral wording)
# -----------------------------------------------------------
def generate_recommendation(gtype, conf, condition, contamination, qty, prefs):
    """
    gtype: generic category name (e.g. Container, Utensil, Organic, Paper, Metal, Electronic, General)
    Returns an emoji-rich recommendation string suitable for both glass/plastic containers.
    """
    t = (gtype or "Unknown").title()
    out = []

    # Header / summary
    out.append(f"🔎 Detected Category: {t}")
    out.append(f"📈 Confidence: {conf:.2f}")
    out.append(f"🧾 Condition: {condition}")
    out.append(f"🧪 Contamination: {contamination}")
    out.append("")

    # Universal safety
    out.append("🛡️ Universal Safety & Handling")
    out.append("• 🧤 Wear gloves when handling items.")
    out.append("• 🚫 Keep children and pets away while handling or storing waste.")
    out.append("• 🧼 Wash hands thoroughly after handling.")
    out.append("• 🏷️ Label bags or boxes that contain sharp/broken items.")
    out.append("• 🚮 Avoid leaving items in corridors or public walkways.")
    out.append("")

    # Category-specific friendly suggestions (suitable wording for glass/plastic)
    if "container" in t.lower() or "utensil" in t.lower():
        out.append("📦 Containers & Utensils — Practical Ideas")
        out.append("• 💧 Rinse and let items dry before reusing or storing.")
        out.append("• 🌱 Reuse as planters, seed starters, or storage jars.")
        out.append("• 🧵 Upcycle: decorate jars for stationery, or use jars as craft organisers.")
        out.append("• 🪣 Use sturdy containers for scoops, watering cans or small tool storage.")
        out.append("• 🛍️ Donate usable items (kitchens, community centers) instead of discarding.")
        out.append("• ⚠️ Wrap broken or sharp pieces before disposal to prevent injury.")
        out.append("")

    elif "organic" in t.lower():
        out.append("🌿 Organic / Food Waste")
        out.append("• 🗑️ Keep wet waste in a covered bin to reduce odour and pests.")
        out.append("• 🌱 Use peels and vegetable scraps for compost or regrowth (e.g., spring onions).")
        out.append("• ❄️ Freeze scraps if you plan to compost later or collect them for community composting.")
        out.append("• 🐾 Avoid leaving food waste outside overnight (attracts animals).")
        out.append("")

    elif "paper" in t.lower():
        out.append("📄 Paper & Cardboard")
        out.append("• 📦 Flatten boxes before storing or handing over to recyclers.")
        out.append("• ✏️ Reuse clean paper for notes or craft projects.")
        out.append("• 💨 Keep paper dry to maintain recyclability.")
        out.append("")

    elif "metal" in t.lower():
        out.append("🔩 Metal Items")
        out.append("• 💧 Rinse cans or metal containers to avoid smell/bugs.")
        out.append("• 🔧 Collect small metal pieces separately to avoid injury.")
        out.append("• 📦 Hand over larger metal items during scrap collection drives.")
        out.append("")

    elif "electronic" in t.lower():
        out.append("💻 Electronic Items")
        out.append("• 🔌 Remove memory cards, SIMs and personal data before donating or discarding.")
        out.append("• 🧴 Keep batteries separate and store safely until disposal.")
        out.append("• ⚠️ Do not break, burn or throw electronics in general waste.")
        out.append("• ♻️ Use authorised e-waste collection points for handover.")
        out.append("")

    elif "general" in t.lower() or "unknown" in t.lower():
        out.append("🗑️ General / Mixed Items")
        out.append("• 🔄 Separate recyclables and non-recyclables where possible.")
        out.append("• 🛍️ Tie bags tightly to prevent spillage.")
        out.append("• 📢 Report overflowing bins or large items to community staff.")
        out.append("")

    else:
        out.append("🔔 Quick Tips")
        out.append("• 🔍 Inspect item: if reusable, consider donating or repurposing.")
        out.append("• 🧹 Keep area tidy and avoid leaving loose items in public spaces.")
        out.append("")

    # Reuse preferences suggestions (from user multiselect)
    if prefs:
        out.append("✨ Reuse Preference Ideas")
        for p in prefs:
            p = p.lower()
            if "recycle" in p:
                out.append("• ♻️ Collect and hand over to recyclers during scheduled drives.")
            elif "compost" in p:
                out.append("• 🌿 Add suitable organic scraps to your compost bin.")
            elif "donate" in p:
                out.append("• 🤝 Check local community groups or donation centres for usable items.")
            elif "art" in p or "craft" in p:
                out.append("• 🎨 Use items in craft projects or as DIY organisers.")
            else:
                out.append(f"• ✅ {p.title()} — consider this option based on item condition.")
        out.append("")

    # Quick contact / community note
    out.append("📞 Community Tips")
    out.append("• 🗓️ Keep an eye out for local collection days for special items.")
    out.append("• 🧑‍🤝‍🧑 Encourage neighbours to separate waste — small actions add up!")
    out.append("")

    return "\n".join(out)

# -----------------------------------------------------------
# SAVE REPORT
# -----------------------------------------------------------
def save_report(username, data):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO reports ({','.join(data.keys())}) VALUES ({','.join(['?']*len(data))})",
        tuple(data.values())
    )
    conn.commit()
    conn.close()

# -----------------------------------------------------------
# HELP CENTER + CHAT SUPPORT (NEW)
# -----------------------------------------------------------

# Predefined Q&A the user can browse or search
HELP_FAQ = {
    "How do I report garbage?": """
1. Go to **Report Garbage** page.  
2. Upload an image or use the **camera**.  
3. Enter location, condition, contamination, quantity.  
4. Click **Analyze Image**.  
5. AI shows category + recommendations.  
6. Click save → report is stored in database.
""",
    "Where can I see all my reports?": """
All reports are visible in **Analytics**, **Leaderboard**, and **Locations** pages.
Admins can see full details under **Admin → Report Management**.
""",
    "How does AI detect garbage?": """
Verdura Lens uses **YOLOv8 model** to detect items in images.  
Detected labels are mapped to categories like Container, Paper, Organic, etc.
""",
    "Camera not working — what should I do?": """
Try:  
• Use Chrome browser  
• Disable/enable camera in browser settings  
• Check Windows Privacy → Camera ON  
• Restart Streamlit with: `streamlit run app.py`
""",
    "My image shows 'no detection' — why?": """
Reasons:  
• Image is too dark  
• Object is too small  
• Background clutter  
• Camera blur  
Try uploading a clearer, closer image with good lighting.
""",
    "What is Awareness Hub?": """
Awareness Hub contains videos and websites to learn about recycling, composting, and environmental care.
""",
    "How can I download reports?": """
Go to **Admin → Exports** (admin only).  
Download full reports, users or presets as CSV.
""",
    "How do I add recommendations?": """
Admins can go to **Admin → Preset Management** and add custom emoji-rich recommendation templates.
""",
    "Where can I see map locations?": """
Go to **Locations** page – it shows a map of all latitude/longitude values from reports.
""",
    "How do I change text color / UI?": """
UI can be changed inside the CSS block at the top of the code (LIGHT_TEAL_UI variable).
"""
}

# -----------------------------------------------------------
# SEARCH / CHAT-LIKE FUNCTION
# -----------------------------------------------------------

def get_help_answer(user_question):
    """Matches a question to FAQ keys (simple fuzzy logic)."""
    uq = user_question.lower()

    # Exact match shortcut
    if uq in [q.lower() for q in HELP_FAQ.keys()]:
        for k, v in HELP_FAQ.items():
            if k.lower() == uq:
                return v

    # Fuzzy match
    for key in HELP_FAQ:
        if any(word in key.lower() for word in uq.split()):
            return HELP_FAQ[key]

    return "❓ Sorry, I don't have an answer for that yet. Try selecting a question from the dropdown."


# -----------------------------------------------------------
# SIDEBAR NAVIGATION (includes Awareness Hub)
# -----------------------------------------------------------
st.session_state.page = st.sidebar.selectbox(
    "Navigate",
    [
        "Report Garbage",
        "Recommendations",
        "Awareness Hub",
        "Help / Chat Support",   # NEW PAGE
        "Analytics",
        "Leaderboard",
        "Locations",
        "Admin"
    ]
)


if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -----------------------------------------------------------
# REPORT GARBAGE PAGE
# -----------------------------------------------------------
if st.session_state.page == "Report Garbage":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("Upload or Capture Image — Verdura Lens")

    condition = st.selectbox("Condition", ["Clean", "Dirty", "Broken"])
    contamination = st.selectbox("Contamination", ["None", "Low", "Medium", "High"])
    qty = st.number_input("Quantity", min_value=1, value=1)

    location = st.text_input("Location / Landmark")
    latitude = st.text_input("Latitude")
    longitude = st.text_input("Longitude")

    prefs = st.multiselect("Reuse Preference", ["Recycle", "Compost", "Donate", "Art", "Craft"])

    img1 = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    img2 = st.camera_input("Or Capture Using Camera")

    img_data = img1 if img1 else img2

    if st.button("Analyze Image"):
        if not img_data:
            st.error("Please upload or capture an image")
        else:
            bytes_data = img_data.getvalue()
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            save_path = f"{IMAGES_DIR}/{ts}.jpg"
            open(save_path,"wb").write(bytes_data)

            gtype, conf, dets, img = detect_yolo_and_draw(bytes_data, save_path)

            st.subheader("🔍 AI Detection Result")
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.info("No preview available.")

            rec_text = generate_recommendation(gtype, conf, condition, contamination, qty, prefs)

            st.subheader("📝 Recommendations")
            st.markdown(f"```\n{rec_text}\n```")

            try:
                lat = float(latitude) if latitude else None
                lon = float(longitude) if longitude else None
            except:
                lat = lon = None

            save_report(st.session_state.username, {
                "username": st.session_state.username,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "garbage_type": gtype,
                "sub_type": "",
                "condition": condition,
                "quantity": qty,
                "weight_range": "",
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "image_path": save_path,
                "chosen_reuse": ",".join(prefs),
                "recommended_reuse": rec_text
            })

            st.success("✅ Report saved!")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# PRESET RECOMMENDATIONS PAGE
# -----------------------------------------------------------
if st.session_state.page == "Recommendations":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("Preset Recommendations — Verdura Lens")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM presets", conn)
    except:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        st.info("No presets available. Admin can add emoji-rich presets in the Admin panel.")
    else:
        for _, row in df.iterrows():
            with st.expander(row["name"]):
                st.markdown(row["recommendation"])
                try:
                    opts = json.loads(row["reuse_options"])
                except:
                    opts = []
                if opts:
                    st.write("🧩 Reuse Options:", opts)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# AWARENESS HUB (VIDEOS + SITES) — NEW
# -----------------------------------------------------------
if st.session_state.page == "Awareness Hub":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("Awareness Hub — Learn & Share 🌍")

    st.markdown("Discover quick videos and trusted sites — open one and spark a cleanup in your community! 💡")

    # --- YouTube picks (embed where possible)
    st.subheader("🎬 Quick Videos (YouTube)")
    st.markdown("Short and motivating picks — watch and share with neighbours:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1) 'How to Reduce Household Waste (5 mins)'**")
        st.video("https://www.youtube.com/watch?v=Qk6x0G6QJkY")  # example link (replace if desired)
        st.markdown("[Open on YouTube ▶](https://www.youtube.com/results?search_query=how+to+reduce+household+waste)")

    with col2:
        st.markdown("**2) 'Composting at Home — Beginner's Guide'**")
        st.video("https://www.youtube.com/watch?v=3q6gyV6o0U4")  # example
        st.markdown("[Open on YouTube ▶](https://www.youtube.com/results?search_query=home+composting+beginner)")

    st.markdown("---")
    st.subheader("🌐 Helpful Websites & Resources")
    st.markdown("Click any link to learn more — short reads and interactive tools.")

    st.write("• 🌱 **UN Environment** — global guides and campaigns:", unsafe_allow_html=True)
    st.markdown("[UN Environment Programme — Waste Management](https://www.unep.org/resources)")

    st.write("• 🏛️ **EPA / National** — local rules & recycling tips:", unsafe_allow_html=True)
    st.markdown("[Search local recycling guidelines](https://www.google.com/search?q=recycling+guidelines+near+me)")

    st.write("• ♻️ **Practical Guides & DIY** — crafts, upcycling, composting ideas:", unsafe_allow_html=True)
    st.markdown("[Search DIY upcycling ideas](https://www.google.com/search?q=upcycling+ideas)")

    st.markdown("---")
    st.subheader("📣 Community Challenge (Fun!)")
    st.markdown("Share a 30-second before/after photo of a small cleanup and tag your neighbours — little contests help build habit! 🏆")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # HELP CENTER / CHAT SUPPORT PAGE (NEW)
    # -----------------------------------------------------------
    if st.session_state.page == "Help / Chat Support":

        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.header("💬 Help & Chat Support — Verdura Lens Assistant")

        st.write("Get instant answers to common questions or ask your own!")

        # Section 1: Predefined FAQ
        st.subheader("📘 Common Questions")
        faq_choice = st.selectbox("Select a question", ["(Select a question)"] + list(HELP_FAQ.keys()))

        if faq_choice != "(Select a question)":
            st.markdown(f"### 🟩 Answer")
            st.markdown(HELP_FAQ[faq_choice])

        st.markdown("---")

        # Section 2: Chat-style Question Box
        st.subheader("💭 Ask Your Own Question")

        user_q = st.text_input("Type your question here...")
        if st.button("Get Answer"):
            if user_q.strip() == "":
                st.warning("Please enter a question.")
            else:
                answer = get_help_answer(user_q)
                st.markdown("### 🟩 Answer")
                st.markdown(answer)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# HELP / CHAT SUPPORT PAGE (NEW)
# -----------------------------------------------------------
if st.session_state.page == "Help / Chat Support":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("💬 Help & Chat Support — Verdura Assistant")

    st.write("Find answers instantly! Select a question or type your own.")

    # ---------- FAQ DROPDOWN ----------
    st.subheader("📘 Frequently Asked Questions")
    faq_option = st.selectbox("Select a question", ["(Select a question)"] + list(HELP_FAQ.keys()))

    if faq_option != "(Select a question)":
        st.markdown("### 🟦 Answer")
        st.markdown(HELP_FAQ[faq_option])

    st.markdown("---")

    # ---------- CHAT-LIKE QUESTION BOX ----------
    st.subheader("💭 Ask a Question")

    user_ask = st.text_input("Type your question here...", placeholder="Example: How do I report garbage?")

    if st.button("Get Answer"):
        if user_ask.strip() == "":
            st.warning("Please type something.")
        else:
            response = get_help_answer(user_ask)
            st.markdown("### 🟩 Answer")
            st.markdown(response)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# ANALYTICS PAGE
# -----------------------------------------------------------
if st.session_state.page == "Analytics":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("📊 Analytics — Your Waste Reports")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM reports WHERE username=?", conn, params=(st.session_state.username,))
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        st.info("No reports available yet.")
    else:
        st.subheader("Category Distribution")
        st.bar_chart(df["garbage_type"].value_counts())

        st.subheader("Reports Over Time")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.line_chart(df.set_index("timestamp")["quantity"])

        st.subheader("All Your Reports")
        st.dataframe(df)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# LEADERBOARD PAGE
# -----------------------------------------------------------
if st.session_state.page == "Leaderboard":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("🏆 Leaderboard — Community Impact")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT username, COUNT(*) AS reports FROM reports GROUP BY username ORDER BY reports DESC",
            conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        st.info("No reports found.")
    else:
        # Rank users
        df["Rank"] = df["reports"].rank(method="dense", ascending=False).astype(int)
        st.dataframe(df)

        # Highlight current user
        my_row = df[df["username"] == st.session_state.username]
        if not my_row.empty:
            st.success(f"🥇 Your Rank: {int(my_row['Rank'].values[0])}")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOCATIONS PAGE (MAP)
# -----------------------------------------------------------
if st.session_state.page == "Locations":

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.header("🗺️ Waste Locations Map")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT latitude, longitude, garbage_type FROM reports WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
            conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        st.info("No location data available.")
    else:
        st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))

        st.subheader("All Locations")
        st.dataframe(df)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# -------------------- PART 3 (Admin + Footer) --------------------
# -----------------------------------------------------------

# NOTE: This block expects DB, helper functions, and pages from Part 1+2 to already be defined.

# -----------------------------------------------------------
# -------------------- PART 3 (Admin + Footer) --------------------
# -----------------------------------------------------------
# ---------- START ADMIN BLOCK ----------
# -------------------- PART 3 (Admin + Footer) --------------------
# NOTE: Replace your existing Part 3 with this block.

# -----------------------------------------------------------
# -------------------- ADMIN PANEL (Clean B2 Layout) --------
# -----------------------------------------------------------

if st.session_state.page == "Admin":

    # Auto-assign admin role to whoever logs in
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET role='admin' WHERE username=?", (st.session_state.username,))
    conn.commit()
    conn.close()

    st.title("🛠️ Admin Dashboard — Verdura Lens")
    st.caption("You are logged in as an admin. All admin tools are unlocked.")

    # --------------------------------------------
    # Load Fresh Data
    # --------------------------------------------
    conn = sqlite3.connect(DB_PATH)
    try:
        df_users = pd.read_sql_query("SELECT username, role, created_at FROM users", conn)
    except: df_users = pd.DataFrame()

    try:
        df_reports = pd.read_sql_query("SELECT * FROM reports ORDER BY timestamp DESC", conn)
    except: df_reports = pd.DataFrame()

    try:
        df_presets = pd.read_sql_query("SELECT * FROM presets", conn)
    except: df_presets = pd.DataFrame()

    conn.close()

    # --------------------------------------------
    # Admin Navigation
    # --------------------------------------------
    admin_nav = st.radio(
        "Select a section",
        ["Dashboard", "Users", "Reports", "Presets", "Exports", "Tools"],
        horizontal=True
    )

    # --------------------------------------------
    # DASHBOARD
    # --------------------------------------------
    if admin_nav == "Dashboard":
        st.subheader("📊 Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Users", len(df_users))
        col2.metric("Reports", len(df_reports))
        col3.metric(
            "Categories",
            df_reports["garbage_type"].nunique() if not df_reports.empty else 0
        )

        st.markdown("---")

        if not df_reports.empty:
            c1, c2 = st.columns(2)

            with c1:
                st.write("Category Breakdown")
                st.bar_chart(df_reports["garbage_type"].value_counts())

            with c2:
                st.write("Reports Over Time")
                temp = df_reports.copy()
                temp["timestamp"] = pd.to_datetime(temp["timestamp"], format="ISO8601")

                st.line_chart(temp.set_index("timestamp")["quantity"])

            st.markdown("### Latest Reports")
            st.dataframe(df_reports.head(10))

        else:
            st.info("No reports found.")

    # --------------------------------------------
    # USERS
    # --------------------------------------------
    if admin_nav == "Users":
        st.subheader("👤 User Management")

        if df_users.empty:
            st.info("No users registered yet.")
        else:
            st.dataframe(df_users)

        st.markdown("### Modify User Role")
        user_list = df_users["username"].tolist() if not df_users.empty else []
        sel_user = st.selectbox("Select a user", user_list)

        c1, c2, c3 = st.columns(3)

        if c1.button("Promote to Admin"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE users SET role='admin' WHERE username=?", (sel_user,))
            conn.commit(); conn.close()
            st.success(f"{sel_user} promoted to admin.")
            st.rerun()

        if c2.button("Demote to User"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("UPDATE users SET role='user' WHERE username=?", (sel_user,))
            conn.commit(); conn.close()
            st.warning(f"{sel_user} demoted to user.")
            st.rerun()

        if c3.button("Delete User"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM reports WHERE username=?", (sel_user,))
            conn.execute("DELETE FROM users WHERE username=?", (sel_user,))
            conn.commit(); conn.close()
            st.error(f"{sel_user} deleted.")
            st.rerun()

    # --------------------------------------------
    # REPORT MANAGEMENT
    # --------------------------------------------
    if admin_nav == "Reports":
        st.subheader("🗂️ Report Management")

        if df_reports.empty:
            st.info("No reports available.")
        else:
            colf1, colf2 = st.columns(2)

            ftype = colf1.selectbox("Filter by Category", ["All"] + sorted(df_reports["garbage_type"].unique()))
            fuser = colf2.selectbox(
                "Filter by User",
                ["All"] + sorted([u for u in df_reports["username"].unique() if isinstance(u, str)])
            )

            filtered = df_reports.copy()
            if ftype != "All": filtered = filtered[filtered["garbage_type"] == ftype]
            if fuser != "All": filtered = filtered[filtered["username"] == fuser]

            st.write(f"Showing {len(filtered)} reports")
            st.dataframe(filtered)

            st.markdown("### Delete a Report")
            rid = st.number_input("Report ID", min_value=1)
            if st.button("Delete Report"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM reports WHERE id=?", (int(rid),))
                conn.commit(); conn.close()
                st.warning(f"Report {rid} deleted.")
                st.rerun()

    # --------------------------------------------
    # PRESETS
    # --------------------------------------------
    if admin_nav == "Presets":
        st.subheader("✨ Preset Recommendations")

        if df_presets.empty:
            st.info("No presets found.")
        else:
            st.dataframe(df_presets)

        st.markdown("### Add / Edit Preset")

        pname = st.text_input("Preset Name")
        preco = st.text_area("Recommendation Text")
        popts = st.text_input("Reuse Options (comma separated)")

        if st.button("Save Preset"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO presets VALUES (?, ?, ?)",
                (pname, preco, json.dumps([x.strip() for x in popts.split(",") if x.strip()]))
            )
            conn.commit(); conn.close()
            st.success("Preset saved.")
            st.rerun()

    # --------------------------------------------
    # EXPORTS
    # --------------------------------------------
    if admin_nav == "Exports":
        st.subheader("⬇️ Export Data")

        col1, col2, col3 = st.columns(3)

        if col1.button("Export Reports CSV"):
            csv = df_reports.to_csv(index=False).encode("utf-8")
            st.download_button("Download", csv, file_name="reports.csv")

        if col2.button("Export Users CSV"):
            csv = df_users.to_csv(index=False).encode("utf-8")
            st.download_button("Download", csv, file_name="users.csv")

        if col3.button("Export Presets CSV"):
            csv = df_presets.to_csv(index=False).encode("utf-8")
            st.download_button("Download", csv, file_name="presets.csv")

        st.markdown("---")

        st.subheader("Database Backup")
        with open(DB_PATH, "rb") as f:
            st.download_button("Download .db file", f, file_name="backup.db")

        st.markdown("---")

        st.subheader("Download All Images (ZIP)")
        import zipfile
        zip_path = "images.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for imgfile in os.listdir(IMAGES_DIR):
                z.write(f"{IMAGES_DIR}/{imgfile}", imgfile)

        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", f, file_name="images.zip")

    # --------------------------------------------
    # TOOLS
    # --------------------------------------------
    if admin_nav == "Tools":
        st.subheader("🔧 Maintenance Tools")

        if st.button("Find Missing Images"):
            missing = []
            for p in df_reports["image_path"].dropna().unique():
                if not os.path.exists(p):
                    missing.append(p)

            st.write("Missing Images:", len(missing))
            if missing:
                st.dataframe(missing)

        if st.button("Remove Orphaned Images"):
            db_list = df_reports["image_path"].dropna().tolist()
            removed = 0
            for img in os.listdir(IMAGES_DIR):
                full = f"{IMAGES_DIR}/{img}"
                if full not in db_list:
                    os.remove(full)
                    removed += 1
            st.warning(f"Removed {removed} orphan images.")

        st.markdown("---")

        if st.checkbox("Delete ALL Reports (Reset)"):
            if st.button("Confirm Delete ALL"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM reports")
                conn.commit(); conn.close()
                st.error("All reports deleted.")
                st.rerun()


# ---------- END ADMIN BLOCK ----------


# -----------------------------------------------------------
# Footer (global)
# -----------------------------------------------------------
st.write("---")
st.markdown(
    "<div class='footer'>Verdura Lens — Helping Communities Keep Their Surroundings Clean • Built with ❤️</div>",
    unsafe_allow_html=True
)

# Small admin note for first-time setup
if st.session_state.username == "admin":
    st.info("Tip: If 'presets' table is empty, go to Admin → Preset Management and add helpful emoji-rich recommendations users will love.")
