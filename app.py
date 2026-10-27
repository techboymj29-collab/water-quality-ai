import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 💧 WATER QUALITY AI
# ==========================================

st.set_page_config(
    page_title="Water Quality AI",
    page_icon="💧",
    layout="wide"
)

# ==========================================
# 🎨 DESIGN AND ANIMATIONS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #ffffff;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.water-header {
    background: linear-gradient(120deg, #075985, #0891b2, #10b981);
    padding: 30px 15px;
    border-radius: 18px;
    text-align: center;
    color: white;
    animation: appear 1s ease-out;
}

.water-header h1 {
    color: white;
    margin-bottom: 8px;
}

.water-drop {
    font-size: 48px;
    display: inline-block;
    animation: floatWater 2s ease-in-out infinite;
}

@keyframes floatWater {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    animation: appear 0.7s ease-out;
    margin-top: 20px;
}

.result-icon {
    font-size: 48px;
}

.probability-track {
    background: #e5e7eb;
    height: 18px;
    border-radius: 20px;
    overflow: hidden;
}

.probability-fill {
    height: 18px;
    border-radius: 20px;
    animation: fillBar 1.4s ease-out;
}

@keyframes fillBar {
    from { width: 0%; }
}

div.stButton > button {
    border-radius: 9px;
    font-weight: bold;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 12px rgba(8, 145, 178, 0.2);
}

.section-title {
    color: #075985;
    font-size: 23px;
    font-weight: bold;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌊 HEADER
# ==========================================

st.markdown("""
<div class="water-header">
    <div class="water-drop">💧</div>
    <h1>WATER QUALITY AI</h1>
    <p>Intelligent Water Potability Classification</p>
    <small>MACHINE LEARNING • RANDOM FOREST</small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background:#ecfeff;
    border-left:5px solid #0891b2;
    padding:15px;
    border-radius:10px;
    margin-top:18px;
">
<b style="color:#155e75;">🧪 How to use</b><br>
Enter the nine water-quality measurements and click
<b>ANALYSE WATER</b> to obtain a prediction.
</div>
""", unsafe_allow_html=True)

# ==========================================
# 📊 DATASET AND MODEL
# ==========================================

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

st.sidebar.title("💧 AquaSense AI")
st.sidebar.write("Water Quality Classification")
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload water_potability.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "👋 Welcome! Upload the water_potability.csv dataset "
        "using the sidebar to activate the prediction model."
    )

    st.markdown("""
    ### 🌊 About this project

    This machine-learning project classifies water samples
    into two categories:

    - 🟢 **Potable:** Classified as potable by the model.
    - 🔴 **Not potable:** Classified as not potable by the model.

    **Model:** Random Forest Classifier  
    **Input:** Nine water-quality measurements  
    **Target:** Potability (0 or 1)
    """)

    st.stop()

# ==========================================
# 🧠 TRAIN THE MODEL
# ==========================================

@st.cache_resource
def train_model(file_bytes):
    import io

    data = pd.read_csv(io.BytesIO(file_bytes))

    required = FEATURES + ["Potability"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Dataset is missing these columns: "
            + ", ".join(missing)
        )

    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=["Potability"])

    data = data[
        data["Potability"].isin([0, 1])
    ]

    if data.empty:
        raise ValueError("No valid Potability labels were found.")

    X = data[FEATURES]
    y = data["Potability"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "The dataset must contain both Potability classes: 0 and 1."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="median")

    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return model, imputer, accuracy, matrix, len(data)

try:
    with st.spinner("🤖 Training the water-quality model..."):
        model, imputer, accuracy, matrix, row_count = train_model(
            uploaded_file.getvalue()
        )

except Exception as error:
    st.error(f"Unable to load or train the model: {error}")
    st.stop()

st.sidebar.success("✅ Model ready!")
st.sidebar.write(f"Dataset samples: {row_count}")
st.sidebar.write(f"Test accuracy: {accuracy:.1%}")

# ==========================================
# 🧪 WATER MEASUREMENTS
# ==========================================

st.markdown(
    '<div class="section-title">🧪 Water Measurements</div>',
    unsafe_allow_html=True
)

with st.form("water_form"):

    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input(
            "🧫 pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1
        )

        solids = st.number_input(
            "🧂 Total solids (ppm)",
            min_value=0.0,
            value=250.0,
            step=1.0
        )

        sulfate = st.number_input(
            "⚗️ Sulfate (mg/L)",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        organic_carbon = st.number_input(
            "🌿 Organic carbon",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        turbidity = st.number_input(
            "🌫️ Turbidity (NTU)",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

    with col2:
        hardness = st.number_input(
            "💎 Hardness (mg/L)",
            min_value=0.0,
            value=150.0,
            step=1.0
        )

        chloramines = st.number_input(
            "🧪 Chloramines (mg/L)",
            min_value=0.0,
            value=6.5,
            step=0.1
        )

        conductivity = st.number_input(
            "⚡ Conductivity (µS/cm)",
            min_value=0.0,
            value=300.0,
            step=1.0
        )

        trihalomethanes = st.number_input(
            "🔬 Trihalomethanes (µg/L)",
            min_value=0.0,
            value=50.0,
            step=0.1
        )

    analyse = st.form_submit_button(
        "🔍 ANALYSE WATER",
        use_container_width=True
    )

# ==========================================
# 🔍 PREDICTION
# ==========================================

if analyse:

    sample = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    processed = imputer.transform(sample[FEATURES])

    prediction = int(model.predict(processed)[0])

    probabilities = model.predict_proba(processed)[0]

    classes = list(model.classes_)

    score = float(
        probabilities[classes.index(prediction)]
    ) * 100

    if prediction == 1:
        colour = "#047857"
        background = "#d1fae5"
        icon = "✅"
        title = "POTABLE"
        message = "The model classifies this sample as SAFE."
    else:
        colour = "#b91c1c"
        background = "#fee2e2"
        icon = "⚠️"
        title = "NOT POTABLE"
        message = "The model classifies this sample as UNSAFE."

    st.markdown(f"""
    <div class="result-card" style="
        background:{background};
        border:2px solid {colour};
    ">
        <div class="result-icon">{icon}</div>
        <h1 style="color:{colour};">{title}</h1>
        <p style="color:{colour};font-size:17px;">{message}</p>

        <div style="
            background:white;
            padding:16px;
            border-radius:12px;
            margin-top:15px;
        ">
            <b>Model score for predicted class</b>

            <div class="probability-track" style="margin-top:12px;">
                <div class="probability-fill" style="
                    width:{score:.1f}%;
                    background:{colour};
                "></div>
            </div>

            <h2 style="color:{colour};">{score:.1f}%</h2>

            <small>
                Estimated probability assigned by the model,
                not a verified measure of water safety.
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📋 Measurement Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        sample.T.rename(columns={0: "Entered value"}),
        use_container_width=True
    )

    st.warning(
        "⚠️ Educational prediction only. This model cannot "
        "confirm actual drinking-water safety. Laboratory "
        "testing against applicable drinking-water standards "
        "is required."
    )

# ==========================================
# 📈 MODEL PERFORMANCE
# ==========================================

with st.expander("📊 View Model Performance"):

    st.metric("Test Accuracy", f"{accuracy:.1%}")

    st.write("Confusion matrix (actual rows, predicted columns):")

    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual Not Potable", "Actual Potable"],
            columns=["Predicted Not Potable", "Predicted Potable"]
        ),
        use_container_width=True
    )

    st.caption(
        "Accuracy is measured on a held-out test set. "
        "It does not establish real-world drinking-water safety."
    )

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#64748b;padding:10px;">
    💧 <b>WATER QUALITY AI</b><br>
    Intelligent Water Potability Classification
</div>
""", unsafe_allow_html=True)
