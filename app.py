# ============================================================
# SAFE ROAD AI
# Traffic Accident Risk Prediction using Machine Learning
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SafeRoad AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #dbeafe;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.risk-high {
    padding: 25px;
    border-radius: 15px;
    background: #fee2e2;
    border-left: 7px solid #dc2626;
}

.risk-medium {
    padding: 25px;
    border-radius: 15px;
    background: #fef3c7;
    border-left: 7px solid #d97706;
}

.risk-low {
    padding: 25px;
    border-radius: 15px;
    background: #dcfce7;
    border-left: 7px solid #16a34a;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 20px;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GENERATE DATA
# ============================================================

@st.cache_data
def generate_data(n=5000):

    np.random.seed(42)

    traffic_volume = np.random.randint(100, 5000, n)
    average_speed = np.random.randint(10, 100, n)
    visibility = np.random.randint(100, 10000, n)
    vehicle_density = np.random.uniform(0.1, 1.0, n)

    time_of_day = np.random.choice(
        ["Morning", "Afternoon", "Evening", "Night"],
        n,
        p=[0.25, 0.25, 0.30, 0.20]
    )

    weather = np.random.choice(
        ["Clear", "Rain", "Fog", "Storm"],
        n,
        p=[0.55, 0.25, 0.12, 0.08]
    )

    road_condition = np.random.choice(
        ["Good", "Moderate", "Poor"],
        n,
        p=[0.55, 0.30, 0.15]
    )

    junction = np.random.choice(
        ["No", "Yes"],
        n,
        p=[0.65, 0.35]
    )

    weekend = np.random.choice(
        ["No", "Yes"],
        n,
        p=[0.70, 0.30]
    )

    # -----------------------------
    # Risk Score
    # -----------------------------

    risk_score = np.zeros(n)

    risk_score += np.where(traffic_volume > 3500, 2.0, 0)
    risk_score += np.where(traffic_volume > 4500, 1.5, 0)

    risk_score += np.where(average_speed > 70, 2.0, 0)
    risk_score += np.where(average_speed > 85, 1.5, 0)

    risk_score += np.where(visibility < 2000, 2.0, 0)
    risk_score += np.where(visibility < 800, 1.5, 0)

    risk_score += np.where(vehicle_density > 0.75, 2.0, 0)
    risk_score += np.where(vehicle_density > 0.90, 1.0, 0)

    risk_score += np.where(weather == "Rain", 1.5, 0)
    risk_score += np.where(weather == "Fog", 2.5, 0)
    risk_score += np.where(weather == "Storm", 3.0, 0)

    risk_score += np.where(road_condition == "Moderate", 1.0, 0)
    risk_score += np.where(road_condition == "Poor", 2.5, 0)

    risk_score += np.where(junction == "Yes", 1.5, 0)

    risk_score += np.where(time_of_day == "Evening", 1.0, 0)
    risk_score += np.where(time_of_day == "Night", 1.5, 0)

    risk_score += np.where(weekend == "Yes", 0.3, 0)

    risk_score += np.random.normal(0, 1.2, n)

    risk_level = pd.cut(
        risk_score,
        bins=[-np.inf, 4, 8, np.inf],
        labels=["Low", "Medium", "High"]
    )

    df = pd.DataFrame({
        "Traffic_Volume": traffic_volume,
        "Average_Speed": average_speed,
        "Visibility": visibility,
        "Vehicle_Density": vehicle_density,
        "Time_of_Day": time_of_day,
        "Weather": weather,
        "Road_Condition": road_condition,
        "Junction": junction,
        "Weekend": weekend,
        "Risk_Level": risk_level
    })

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = generate_data()

    X = df.drop("Risk_Level", axis=1)
    y = df["Risk_Level"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"]
    )

    return (
        df,
        model,
        X,
        y_test,
        y_pred,
        accuracy,
        importance,
        cm
    )


# ============================================================
# LOAD MODEL
# ============================================================

(
    df,
    model,
    X,
    y_test,
    y_pred,
    accuracy,
    importance,
    cm
) = train_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 Traffic Safety")

    st.markdown(
        "### Traffic Safety Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Risk Prediction",
            "📊 Model Analytics",
            "🧠 Data Thinking",
            "🚀 Future Scope"
        ]
    )

    st.divider()

    st.markdown(
        """
        **Machine Learning**

        Random Forest Classifier

        **Dataset**

        5,000 simulated traffic records

        **Purpose**

        Early-warning traffic safety
        decision support.
        """
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🚦 Traffic Safety</h1>

<p>
Traffic Accident Risk Prediction using
Data Thinking & Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Traffic Safety Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.1f}%"
        )

    with col3:
        st.metric(
            "ML Algorithm",
            "Random Forest"
        )

    with col4:
        st.metric(
            "Risk Classes",
            "3"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Distribution")

        risk_counts = df["Risk_Level"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            risk_counts.index,
            risk_counts.values
        )

        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Number of Records")
        ax.set_title("Traffic Risk Distribution")

        st.pyplot(fig)

    with col2:

        st.subheader("Top Risk Factors")

        top = importance.head(8).sort_values(
            "Importance"
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            top["Feature"],
            top["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    st.divider()

    st.subheader("🎯 Social Impact")

    st.info(
        """
        SafeRoad AI is designed as an early-warning decision-support
        system. It identifies traffic conditions associated with higher
        accident risk so that preventive measures can be considered.

        Possible interventions include traffic monitoring, road-safety
        awareness, emergency preparedness and infrastructure review.
        """
    )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "🔮 Risk Prediction":

    st.markdown(
        '<div class="section-title">🔮 Predict Traffic Accident Risk</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the current traffic and environmental conditions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        traffic_volume = st.slider(
            "Traffic Volume (vehicles/hour)",
            100,
            5000,
            2500,
            100
        )

        average_speed = st.slider(
            "Average Speed (km/h)",
            10,
            100,
            50
        )

        visibility = st.slider(
            "Visibility (meters)",
            100,
            10000,
            5000,
            100
        )

    with col2:

        vehicle_density = st.slider(
            "Vehicle Density",
            0.1,
            1.0,
            0.5,
            0.05
        )

        time_of_day = st.selectbox(
            "Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Rain",
                "Fog",
                "Storm"
            ]
        )

    with col3:

        road_condition = st.selectbox(
            "Road Condition",
            [
                "Good",
                "Moderate",
                "Poor"
            ]
        )

        junction = st.selectbox(
            "Junction / Intersection",
            [
                "No",
                "Yes"
            ]
        )

        weekend = st.selectbox(
            "Weekend",
            [
                "No",
                "Yes"
            ]
        )

    st.divider()

    predict_button = st.button(
        "🚦 Predict Accident Risk",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        input_data = pd.DataFrame({
            "Traffic_Volume": [traffic_volume],
            "Average_Speed": [average_speed],
            "Visibility": [visibility],
            "Vehicle_Density": [vehicle_density],
            "Time_of_Day": [time_of_day],
            "Weather": [weather],
            "Road_Condition": [road_condition],
            "Junction": [junction],
            "Weekend": [weekend]
        })

        input_encoded = pd.get_dummies(
            input_data,
            drop_first=True
        )

        input_encoded = input_encoded.reindex(
            columns=X.columns,
            fill_value=0
        )

        prediction = model.predict(
            input_encoded
        )[0]

        probabilities = model.predict_proba(
            input_encoded
        )[0]

        probability_dict = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        st.divider()

        # -----------------------------
        # RISK RESULT
        # -----------------------------

        if prediction == "High":

            st.markdown(
                """
                <div class="risk-high">
                <h2>🔴 HIGH RISK</h2>
                <p>
                The current conditions are associated with
                a higher predicted traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif prediction == "Medium":

            st.markdown(
                """
                <div class="risk-medium">
                <h2>🟡 MEDIUM RISK</h2>
                <p>
                The current conditions show a moderate
                predicted traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="risk-low">
                <h2>🟢 LOW RISK</h2>
                <p>
                The current conditions show a lower
                predicted traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Risk Probability")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Low",
                f"{probability_dict.get('Low', 0) * 100:.1f}%"
            )

        with p2:
            st.metric(
                "Medium",
                f"{probability_dict.get('Medium', 0) * 100:.1f}%"
            )

        with p3:
            st.metric(
                "High",
                f"{probability_dict.get('High', 0) * 100:.1f}%"
            )

        st.subheader("Entered Conditions")

        display_data = pd.DataFrame({
            "Parameter": [
                "Traffic Volume",
                "Average Speed",
                "Visibility",
                "Vehicle Density",
                "Time",
                "Weather",
                "Road Condition",
                "Junction",
                "Weekend"
            ],
            "Value": [
                f"{traffic_volume} vehicles/hour",
                f"{average_speed} km/h",
                f"{visibility} meters",
                f"{vehicle_density:.2f}",
                time_of_day,
                weather,
                road_condition,
                junction,
                weekend
            ]
        })

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Prediction is a machine-learning estimate and "
            "should not be interpreted as a guarantee of an accident."
        )


# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.markdown(
        '<div class="section-title">📊 Machine Learning Analytics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Training Records",
            "4,000"
        )

    with col3:
        st.metric(
            "Testing Records",
            "1,000"
        )

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.imshow(cm)

    ax.set_xticks(
        range(3),
        ["Low", "Medium", "High"]
    )

    ax.set_yticks(
        range(3),
        ["Low", "Medium", "High"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title("Traffic Risk Confusion Matrix")

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    st.subheader("Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA THINKING
# ============================================================

elif page == "🧠 Data Thinking":

    st.markdown(
        '<div class="section-title">🧠 Data Thinking Framework</div>',
        unsafe_allow_html=True
    )

    stages = [
        (
            "01",
            "Empathy",
            "Understand drivers, pedestrians, commuters, traffic police and emergency services."
        ),
        (
            "02",
            "Define",
            "Identify traffic and environmental conditions associated with accident risk."
        ),
        (
            "03",
            "Ideate",
            "Design an early-warning system that can support preventive traffic-safety decisions."
        ),
        (
            "04",
            "Data",
            "Collect traffic volume, speed, weather, visibility and road-condition information."
        ),
        (
            "05",
            "Prepare",
            "Clean data, encode categorical variables and prepare the ML features."
        ),
        (
            "06",
            "Model",
            "Train a Random Forest classification model."
        ),
        (
            "07",
            "Test",
            "Evaluate accuracy, precision, recall, F1-score and confusion matrix."
        ),
        (
            "08",
            "Predict",
            "Classify conditions into Low, Medium or High predicted risk."
        ),
        (
            "09",
            "Impact",
            "Use the risk signal to support preventive planning and safety interventions."
        )
    ]

    for number, title, description in stages:

        st.markdown(
            f"""
            <div class="metric-card">

            <h3>{number} — {title}</h3>

            <p>{description}</p>

            </div>

            <br>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FUTURE SCOPE
# ============================================================

elif page == "🚀 Future Scope":

    st.markdown(
        '<div class="section-title">🚀 Future Scope</div>',
        unsafe_allow_html=True
    )

    future = [
        "📡 Real-time traffic sensor integration",
        "🌦️ Live weather API integration",
        "🗺️ GPS and location-based risk mapping",
        "🚑 Emergency-service integration",
        "📱 Mobile safety alerts",
        "🤖 More advanced ML models",
        "📊 Historical accident pattern analysis",
        "🏙️ Smart-city traffic management",
        "🔍 Explainable AI for prediction reasoning",
        "⚖️ Fairness and model monitoring"
    ]

    for item in future:
        st.markdown(
            f"### {item}"
        )

    st.divider()

    st.subheader("⚠️ Responsible AI")

    st.warning(
        """
        This prototype uses synthetic data and demonstrates the
        machine-learning workflow. A real deployment would require
        validated historical traffic and accident data, privacy
        safeguards, model monitoring and human oversight.
        """
    )

    st.subheader("🌍 Social Impact")

    st.success(
        """
        The long-term objective is not simply to predict accidents,
        but to use data-driven signals to support safer roads,
        better planning and faster preventive action.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
    <b>SafeRoad AI</b> • Traffic Accident Risk Prediction
    <br>
    Data Thinking + Machine Learning + Social Impact
    </div>
    """,
    unsafe_allow_html=True
)