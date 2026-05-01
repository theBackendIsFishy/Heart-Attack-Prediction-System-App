import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck — Heart Attack Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "Heart_Attack_Prediction.pkl")
    return joblib.load(path)

model = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1100px; width: 100% !important; padding-left: 3rem !important; padding-right: 3rem !important; }

/* ── NAV ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0 20px 0;
    border-bottom: 1px solid #e4ddd4;
    margin-bottom: 36px;
}
.nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #c0392b;
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-dot {
    width: 30px; height: 30px;
    background: #c0392b;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 15px;
}

/* ── HERO ── */
.hero-tag {
    display: inline-block;
    background: #fdf0ee;
    color: #c0392b;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    border: 1px solid #f5c6c0;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    line-height: 1.1;
    color: #1a1a1a;
    letter-spacing: -1px;
    margin-bottom: 16px;
}
.hero-title em { color: #c0392b; font-style: italic; }
.hero-sub {
    font-size: 17px;
    color: #666;
    line-height: 1.7;
    margin-bottom: 32px;
}

/* ── CARDS ── */
.step-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 48px;
}
.step-card {
    background: #fff;
    border: 1px solid #e8e2db;
    border-radius: 16px;
    padding: 24px 20px;
}
.step-num {
    width: 34px; height: 34px;
    background: #fdf0ee;
    color: #c0392b;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 500;
    margin-bottom: 14px;
}
.step-card h3 { font-size: 14px; font-weight: 500; margin-bottom: 7px; color: #1a1a1a; }
.step-card p { font-size: 12px; color: #888; line-height: 1.6; }

/* ── GLOSSARY ── */
.glo-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 40px;
}
.glo-card {
    background: #fff;
    border: 1px solid #e8e2db;
    border-radius: 12px;
    padding: 16px 18px;
}
.glo-name { font-size: 12px; font-weight: 500; color: #c0392b; margin-bottom: 4px; }
.glo-desc { font-size: 11px; color: #777; line-height: 1.55; }

/* ── DISCLAIMER ── */
.disclaimer {
    background: #fffbf0;
    border: 1px solid #f5e4a8;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 13px;
    color: #7a5c00;
    line-height: 1.6;
    margin-bottom: 36px;
}

/* ── SECTION HEADERS ── */
.sec-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #aaa;
    text-align: center;
    margin-bottom: 10px;
}
.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 32px;
    color: #1a1a1a;
}

/* ── FORM SECTIONS ── */
.form-section {
    background: #fff;
    border: 1px solid #e8e2db;
    border-radius: 16px;
    padding: 24px 24px 8px 24px;
    margin-bottom: 18px;
}
.form-sec-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 16px;
    border-bottom: 1px solid #f0ece7;
    padding-bottom: 10px;
}

/* ── RESULT ── */
.result-high {
    background: #fdecea;
    border: 1px solid #f5c6c0;
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    margin-bottom: 24px;
}
.result-low {
    background: #eaf6f0;
    border: 1px solid #b7e4cc;
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    margin-bottom: 24px;
}
.result-icon { font-size: 52px; margin-bottom: 14px; }
.result-title-high {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: #c0392b;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}
.result-title-low {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: #1e8449;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}
.result-sub { font-size: 15px; color: #555; line-height: 1.6; }

.prob-wrap {
    background: #fff;
    border: 1px solid #e8e2db;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #888;
    margin-bottom: 10px;
}
.prob-label-row span:last-child { font-weight: 500; color: #1a1a1a; }
.prob-bar-bg {
    background: #f0ebe4;
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 6px;
}
.prob-bar-high {
    height: 100%;
    border-radius: 100px;
    background: #c0392b;
}
.prob-bar-low {
    height: 100%;
    border-radius: 100px;
    background: #1e8449;
}

.summary-card {
    background: #fff;
    border: 1px solid #e8e2db;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.summary-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 14px;
}
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f5f1ed;
    font-size: 13px;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #888; }
.summary-val { font-weight: 500; color: #1a1a1a; }

.note-box {
    background: #fffbf0;
    border: 1px solid #f5e4a8;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 12px;
    color: #7a5c00;
    line-height: 1.7;
    margin-bottom: 28px;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #444 !important;
}
div[data-testid="stButton"] button {
    border-radius: 100px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #c0392b !important;
    border: none !important;
    padding: 10px 28px !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #a93226 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "result" not in st.session_state:
    st.session_state.result = None


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_landing():
    # Nav
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">
        <span class="nav-logo-dot">♥</span> CardioCheck
      </div>
      <span style="font-size:13px;color:#aaa;">Heart Attack Risk Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 40px;">
      <div class="hero-tag">ML-Powered Heart Health Screening</div>
      <div class="hero-title">Know your heart<br/><em>before it speaks</em></div>
      <div class="hero-sub">
        Enter basic clinical values and our Random Forest model will<br/>
        estimate your risk of a heart attack in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Assessment →", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # How it works
    st.markdown('<p class="sec-label">Process</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Three simple steps</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-grid">
      <div class="step-card">
        <div class="step-num">1</div>
        <h3>Enter your data</h3>
        <p>Fill in 13 clinical measurements from a recent blood test or checkup. Each field has a plain-language tooltip.</p>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <h3>Model runs instantly</h3>
        <p>A Random Forest classifier trained on the UCI Heart Disease dataset evaluates your inputs in real time.</p>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <h3>See your risk score</h3>
        <p>Get a clear High or Low risk result with a probability percentage and a full summary of your inputs.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Glossary
    st.markdown('<p class="sec-label">Field Guide</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">What each field means</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glo-grid">
      <div class="glo-card"><div class="glo-name">Age</div><div class="glo-desc">Your age in years. Cardiac risk generally increases with age, especially after 45 for men and 55 for women.</div></div>
      <div class="glo-card"><div class="glo-name">Sex</div><div class="glo-desc">Biological sex. Males statistically show higher rates of early-onset heart disease.</div></div>
      <div class="glo-card"><div class="glo-name">Chest Pain Type (cp)</div><div class="glo-desc">0 = Typical angina · 1 = Atypical angina · 2 = Non-anginal pain · 3 = Asymptomatic (no pain)</div></div>
      <div class="glo-card"><div class="glo-name">Resting Blood Pressure</div><div class="glo-desc">Systolic BP at rest in mmHg. Normal ≈ 120. Above 140 is considered high blood pressure.</div></div>
      <div class="glo-card"><div class="glo-name">Cholesterol</div><div class="glo-desc">Serum cholesterol in mg/dl. Below 200 = desirable · 200–239 = borderline · 240+ = high.</div></div>
      <div class="glo-card"><div class="glo-name">Fasting Blood Sugar</div><div class="glo-desc">Whether fasting blood sugar exceeds 120 mg/dl. Elevated levels may indicate diabetes.</div></div>
      <div class="glo-card"><div class="glo-name">Resting ECG (restecg)</div><div class="glo-desc">0 = Normal · 1 = ST-T wave abnormality (possible ischemia) · 2 = Left ventricular hypertrophy</div></div>
      <div class="glo-card"><div class="glo-name">Max Heart Rate (thalach)</div><div class="glo-desc">Highest heart rate reached during an exercise stress test. Lower max rate can indicate problems.</div></div>
      <div class="glo-card"><div class="glo-name">Exercise Induced Angina</div><div class="glo-desc">Chest pain triggered by physical activity. 1 = Yes. A strong indicator of coronary artery disease.</div></div>
      <div class="glo-card"><div class="glo-name">Oldpeak (ST Depression)</div><div class="glo-desc">ST segment drop during exercise vs rest on ECG. Range 0–6.2. Higher = more cardiac stress.</div></div>
      <div class="glo-card"><div class="glo-name">ST Slope</div><div class="glo-desc">Shape of the ST segment at peak exercise. 0 = Upsloping (better) · 1 = Flat · 2 = Downsloping (worse)</div></div>
      <div class="glo-card"><div class="glo-name">Major Vessels (ca)</div><div class="glo-desc">Number of major coronary arteries (0–3) visible via fluoroscopy. More visible = generally better flow.</div></div>
      <div class="glo-card"><div class="glo-name">Thalassemia (thal)</div><div class="glo-desc">Nuclear stress test result. 0 = Normal · 1 = Fixed defect · 2 = Reversible defect · 3 = Unknown</div></div>
      <div class="glo-card"><div class="glo-name">Heart Attack Risk (target)</div><div class="glo-desc">The prediction output. 1 = Higher risk of heart attack · 0 = Lower risk. This is what the model predicts.</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>This tool is educational, not medical advice.</strong>
      Results are generated by a machine learning model trained on public data and should never
      replace a diagnosis from a qualified cardiologist or physician.
      If you have concerns about your heart health, please consult a doctor immediately.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("I understand — begin assessment →", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# FORM PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_form():
    # Nav
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">
        <span class="nav-logo-dot">♥</span> CardioCheck
      </div>
      <span style="font-size:13px;color:#aaa;">Patient Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to home"):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown("""
    <div style="margin: 24px 0 8px;">
      <div style="font-family:'DM Serif Display',serif; font-size:34px; letter-spacing:-0.5px; color:#1a1a1a;">Patient Assessment</div>
      <div style="font-size:14px; color:#888; margin-top:6px;">Fill in all 13 fields below. Values should come from a recent medical checkup or blood test.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Section 1: Demographics ──
    st.markdown('<div class="form-section"><div class="form-sec-title">Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=None,
                              placeholder="e.g. 54", help="Your age in full years.")
    with col2:
        sex = st.selectbox("Sex", options=["", "Male", "Female"],
                           help="Biological sex. Males have statistically higher early cardiac risk.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Symptoms ──
    st.markdown('<div class="form-section"><div class="form-sec-title">Symptoms</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cp = st.selectbox("Chest Pain Type",
                          options=["", "0 – Typical angina", "1 – Atypical angina",
                                   "2 – Non-anginal pain", "3 – Asymptomatic"],
                          help="0=Classic squeezing chest pain. 1=Atypical. 2=Unrelated to heart. 3=No pain at all.")
    with col2:
        exang = st.selectbox("Exercise Induced Angina",
                             options=["", "Yes (1)", "No (0)"],
                             help="Do you experience chest pain or tightness during physical activity?")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Vitals ──
    st.markdown('<div class="form-section"><div class="form-sec-title">Vitals &amp; Blood Work</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220,
                                   value=None, placeholder="e.g. 120",
                                   help="Your systolic BP at rest. Normal ≈ 120 mmHg. Above 140 is high.")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                           options=["", "Yes (1) — above 120 mg/dl", "No (0) — 120 or below"],
                           help="Measured after 8+ hours without food. High levels may indicate diabetes.")
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600,
                               value=None, placeholder="e.g. 200",
                               help="Total serum cholesterol. Below 200 = good. 200–239 = borderline. 240+ = high.")
        thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220,
                                  value=None, placeholder="e.g. 150",
                                  help="Peak heart rate during exercise stress test. Typical range: 60–202 bpm.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4: ECG ──
    st.markdown('<div class="form-section"><div class="form-sec-title">ECG &amp; Stress Test Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        restecg = st.selectbox("Resting ECG Result",
                               options=["", "0 – Normal", "1 – ST-T wave abnormality",
                                        "2 – Left ventricular hypertrophy"],
                               help="0=Normal electrical activity. 1=Possible ischemia. 2=Enlarged heart muscle.")
        slope = st.selectbox("Slope of Peak Exercise ST",
                             options=["", "0 – Upsloping", "1 – Flat", "2 – Downsloping"],
                             help="Direction of the ST segment at peak exercise. Upsloping = better prognosis.")
        ca = st.selectbox("Major Vessels (Fluoroscopy, 0–3)",
                          options=["", "0", "1", "2", "3"],
                          help="Number of major coronary arteries visible via fluoroscopy. More = generally better.")
    with col2:
        oldpeak = st.slider("Oldpeak — ST Depression", min_value=0.0, max_value=6.2,
                            value=0.0, step=0.1,
                            help="ST segment drop during exercise vs rest. 0 = none. Higher = more cardiac stress.")
        thal = st.selectbox("Thalassemia (thal)",
                            options=["", "0 – Normal", "1 – Fixed defect",
                                     "2 – Reversible defect", "3 – Unknown"],
                            help="Nuclear stress test result. Fixed defect = permanent damage. Reversible = only during stress.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Validate & Predict ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("Predict My Risk →", type="primary", use_container_width=True)

    if predict_btn:
        # Collect & validate
        errors = []
        vals = {}

        if not age:                 errors.append("Age")
        else:                       vals["age"] = float(age)

        if sex == "":               errors.append("Sex")
        else:                       vals["sex"] = 1.0 if sex == "Male" else 0.0

        if cp == "":                errors.append("Chest Pain Type")
        else:                       vals["cp"] = float(cp[0])

        if not trestbps:            errors.append("Resting Blood Pressure")
        else:                       vals["trestbps"] = float(trestbps)

        if not chol:                errors.append("Cholesterol")
        else:                       vals["chol"] = float(chol)

        if fbs == "":               errors.append("Fasting Blood Sugar")
        else:                       vals["fbs"] = 1.0 if fbs.startswith("Yes") else 0.0

        if restecg == "":           errors.append("Resting ECG")
        else:                       vals["restecg"] = float(restecg[0])

        if not thalach:             errors.append("Max Heart Rate")
        else:                       vals["thalach"] = float(thalach)

        if exang == "":             errors.append("Exercise Induced Angina")
        else:                       vals["exang"] = 1.0 if exang.startswith("Yes") else 0.0

        vals["oldpeak"] = float(oldpeak)

        if slope == "":             errors.append("ST Slope")
        else:                       vals["slope"] = float(slope[0])

        if ca == "":                errors.append("Major Vessels")
        else:                       vals["ca"] = float(ca)

        if thal == "":              errors.append("Thalassemia")
        else:                       vals["thal"] = float(thal[0])

        if errors:
            st.error(f"Please fill in all fields before predicting. Missing: {', '.join(errors)}")
        else:
            # Run model
            features = np.array([[
                vals["age"], vals["sex"], vals["cp"], vals["trestbps"],
                vals["chol"], vals["fbs"], vals["restecg"], vals["thalach"],
                vals["exang"], vals["oldpeak"], vals["slope"], vals["ca"], vals["thal"]
            ]])
            prediction = int(model.predict(features)[0])
            probability = model.predict_proba(features)[0]

            # Store display-friendly labels
            display = {
                "Age": f"{int(vals['age'])} years",
                "Sex": "Male" if vals["sex"] == 1 else "Female",
                "Chest Pain Type": ["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"][int(vals["cp"])],
                "Resting Blood Pressure": f"{int(vals['trestbps'])} mmHg",
                "Cholesterol": f"{int(vals['chol'])} mg/dl",
                "Fasting Blood Sugar >120": "Yes" if vals["fbs"] == 1 else "No",
                "Resting ECG": ["Normal","ST-T abnormality","LV hypertrophy"][int(vals["restecg"])],
                "Max Heart Rate": f"{int(vals['thalach'])} bpm",
                "Exercise Induced Angina": "Yes" if vals["exang"] == 1 else "No",
                "Oldpeak (ST Depression)": f"{vals['oldpeak']:.1f}",
                "ST Slope": ["Upsloping","Flat","Downsloping"][int(vals["slope"])],
                "Major Vessels": str(int(vals["ca"])),
                "Thalassemia": ["Normal","Fixed defect","Reversible defect","Unknown"][int(vals["thal"])],
            }

            st.session_state.result = {
                "prediction": prediction,
                "prob_high": round(float(probability[1]) * 100, 1),
                "prob_low": round(float(probability[0]) * 100, 1),
                "display": display,
            }
            st.session_state.page = "result"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RESULT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_result():
    r = st.session_state.result
    is_high = r["prediction"] == 1
    pct = r["prob_high"]

    # Nav
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">
        <span class="nav-logo-dot">♥</span> CardioCheck
      </div>
      <span style="font-size:13px;color:#aaa;">Your Result</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Edit"):
            st.session_state.page = "form"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # Result card
    if is_high:
        st.markdown(f"""
        <div class="result-high">
          <div class="result-icon">⚠️</div>
          <div class="result-title-high">Higher Risk Indicated</div>
          <div class="result-sub">The model suggests elevated cardiovascular risk.<br/>Please consult a doctor as soon as possible.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
          <div class="result-icon">✅</div>
          <div class="result-title-low">Lower Risk Indicated</div>
          <div class="result-sub">The model suggests lower cardiovascular risk.<br/>Keep up a heart-healthy lifestyle!</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    bar_color = "#c0392b" if is_high else "#1e8449"
    st.markdown(f"""
    <div class="prob-wrap">
      <div class="prob-label-row">
        <span>Estimated probability of high cardiac risk</span>
        <span>{pct}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="{'prob-bar-high' if is_high else 'prob-bar-low'}" style="width:{pct}%"></div>
      </div>
      <div style="font-size:11px;color:#bbb;text-align:right;">based on Random Forest model prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # Input summary
    rows_html = ""
    for k, v in r["display"].items():
        rows_html += f'<div class="summary-row"><span class="summary-key">{k}</span><span class="summary-val">{v}</span></div>'

    st.markdown(f"""
    <div class="summary-card">
      <div class="summary-title">Your inputs summary</div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)

    # Note
    st.markdown("""
    <div class="note-box">
      ⚠️ This is a machine learning estimate, not a clinical diagnosis.
      The model was trained on the UCI Heart Disease dataset with approximately 85% test accuracy.
      Always consult a qualified cardiologist or physician before making any health decisions.
    </div>
    """, unsafe_allow_html=True)

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
    with col2:
        if st.button("Reassess", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    page_landing()
elif st.session_state.page == "form":
    page_form()
elif st.session_state.page == "result":
    page_result()