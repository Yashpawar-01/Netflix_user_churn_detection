
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# 1. FILE PATHS
# All files are in the same folder as app.py
# =====================================================


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "netflix_churn_model.pkl"
FEATURES_PATH = BASE_DIR / "netflix_churn_features.pkl"
DATA_PATH = BASE_DIR / "netflix_customer_churn.csv"

st.set_page_config(
    page_title="Netflix Churn Predictor",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# 2. DESIGN
# =====================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #101016, #241019);
}
[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    padding: 25px;
    border: 1px solid #40313a;
    border-radius: 16px;
    background: #19151c;
    margin-bottom: 20px;
}
.hero h1 {
    color: #e50914;
}
.hero p {
    color: #dddddd;
}
div.stButton > button[kind="primary"] {
    background-color: #e50914;
    color: white;
    font-weight: bold;
    border: none;
}
</style>

<div class="hero">
    <h1>🎬 Netflix Customer Churn Predictor</h1>
    <p>Predict whether a customer is likely to leave Netflix using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD SAVED MODEL
# No training and no churn_model session state required
# =====================================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_features(path, modified_time):
    saved = joblib.load(path)

    if isinstance(saved, pd.DataFrame):
        return list(saved.columns)

    return list(saved)


@st.cache_data
def load_dataset(path, modified_time):
    data = pd.read_csv(path)

    # Match the column-name normalization used during training
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return data


# Check that files exist
for file_path in [MODEL_PATH, FEATURES_PATH, DATA_PATH]:
    if not file_path.exists():
        st.error(f"File not found: {file_path.name}")
        st.write("Expected location:", str(file_path))
        st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )

    features = load_features(
        str(FEATURES_PATH),
        FEATURES_PATH.stat().st_mtime
    )

    df = load_dataset(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime
    )

except Exception as error:
    st.error(f"Unable to load the model or data: {error}")
    st.stop()


# =====================================================
# 4. VALIDATE FEATURE COLUMNS
# =====================================================

if not features:
    st.error("The saved feature list is empty.")
    st.stop()

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    st.error(
        "Some model features are missing from the dataset: "
        + ", ".join(missing_features)
    )
    st.write("Dataset columns:", df.columns.tolist())
    st.stop()

st.sidebar.success("Trained model loaded successfully!")
st.sidebar.write(f"Model: {MODEL_PATH.name}")
st.sidebar.write(f"Features: {len(features)}")
st.sidebar.write(f"Dataset: {DATA_PATH.name}")

# =====================================================
# 5. DATASET OVERVIEW
# =====================================================

st.markdown("## Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Input Features", len(features))
col3.metric(
    "Model Size",
    f"{MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MB"
)

with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)


# =====================================================
# 6. CUSTOMER INPUT FORM
# =====================================================

st.markdown("---")
st.markdown("## Enter Customer Details")

st.write(
    "Enter the customer information below and click "
    "**Predict Churn**."
)

customer = {}

with st.form("customer_prediction_form"):

    columns = st.columns(2)

    for index, feature in enumerate(features):

        series = df[feature]
        label = feature.replace("_", " ").title()

        with columns[index % 2]:

            # Numerical features
            if pd.api.types.is_numeric_dtype(series):

                values = pd.to_numeric(
                    series, errors="coerce"
                ).dropna()

                if values.empty:
                    customer[feature] = st.number_input(
                        label,
                        value=0.0,
                        key=f"input_{feature}"
                    )

                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                    is_integer = (
                        pd.api.types.is_integer_dtype(series)
                        or np.all(
                            np.isclose(values, np.round(values))
                        )
                    )

                    if minimum == maximum:

                        if is_integer:
                            customer[feature] = st.number_input(
                                label,
                                value=int(round(default)),
                                step=1,
                                key=f"input_{feature}"
                            )
                        else:
                            customer[feature] = st.number_input(
                                label,
                                value=default,
                                key=f"input_{feature}"
                            )

                    elif is_integer and maximum - minimum <= 10000:

                        customer[feature] = st.number_input(
                            label,
                            min_value=int(np.floor(minimum)),
                            max_value=int(np.ceil(maximum)),
                            value=int(round(default)),
                            step=1,
                            key=f"input_{feature}"
                        )

                    else:

                        customer[feature] = st.number_input(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=float(
                                np.clip(default, minimum, maximum)
                            ),
                            step=0.1,
                            key=f"input_{feature}"
                        )

            # Categorical features
            else:

                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                customer[feature] = st.selectbox(
                    label,
                    options,
                    key=f"input_{feature}"
                )

    submitted = st.form_submit_button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    )


# =====================================================
# 7. PREDICTION
# =====================================================

if submitted:

    try:
        # Maintain exactly the feature order used by the model
        customer_df = pd.DataFrame(
            [customer],
            columns=features
        )

        # Preserve numerical data types from the dataset
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                customer_df[feature] = pd.to_numeric(
                    customer_df[feature],
                    errors="raise"
                )

        # Predict using the saved model
        prediction = model.predict(customer_df)[0]

        st.markdown("---")
        st.markdown("## Prediction Result")

        # Probability of the churn class, if available
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(customer_df)[0]

            classes = getattr(model, "classes_", None)

            # Support sklearn Pipeline models
            if classes is None and hasattr(model, "named_steps"):
                final_model = list(
                    model.named_steps.values()
                )[-1]
                classes = getattr(final_model, "classes_", None)

            if classes is not None:
                class_labels = list(classes)

                if 1 in class_labels:
                    probability = float(
                        probabilities[class_labels.index(1)]
                    )

        left, right = st.columns([1, 2])

        with left:

            if prediction == 1:
                st.error("⚠️ Customer is likely to churn")

            elif prediction == 0:
                st.success("✅ Customer is likely to stay")

            else:
                st.info(f"Predicted class: {prediction}")

        with right:

            if probability is not None:

                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.2%}"
                )

                st.progress(
                    float(np.clip(probability, 0.0, 1.0))
                )

                if probability >= 0.70:
                    st.warning(
                        "Suggested action: Consider a customer "
                        "retention offer or proactive check-in."
                    )

                elif probability >= 0.40:
                    st.info(
                        "Suggested action: Monitor customer "
                        "engagement and recent activity."
                    )

                else:
                    st.info(
                        "Suggested action: Continue normal "
                        "customer engagement."
                    )

            else:
                st.info(
                    "This model does not provide a churn probability."
                )

        st.caption(
            "Predictions are estimates, not guarantees. "
            "Confirm that class 1 represents churn in your training data."
        )

        # Download prediction
        result = customer_df.copy()
        result["predicted_churn"] = prediction

        if probability is not None:
            result["churn_probability"] = probability

        st.download_button(
            label="📥 Download Prediction CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="netflix_customer_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

        st.info(
            "Check that your saved model, feature list, and dataset "
            "are compatible with each other."
        )
