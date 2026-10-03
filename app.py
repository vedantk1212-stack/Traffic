# ============================================================
# SAFE ROAD AI
# Traffic Accident Risk Prediction using Regression
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    # --------------------------------------------------------
    # ACCIDENT RISK SCORE
    # --------------------------------------------------------

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

    # Keep score positive
    risk_score = np.clip(risk_score, 0, None)

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
        "Risk_Score": risk_score
    })

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = generate_data()

    X = df.drop("Risk_Score", axis=1)
    y = df["Risk_Score"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    return (
        df,
        model,
        X,
        y_test,
        y_pred,
        mae,
        mse,
        r2,
        importance
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
    mae,
    mse,
    r2,
    importance
) = train_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 Traffic Safety")

    st.markdown("### Traffic Safety Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Risk Prediction",
            "📊 Model Analytics",
            "🚀 Future Scope"
        ]
    )

    st.divider()

    st.markdown(
        """
        **Machine Learning**

        Random Forest Regression

        **Dataset**

        5,000 simulated traffic records

        **Purpose**

        Predict continuous traffic accident risk score.
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
Machine Learning Regression
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
            "R² Score",
            f"{r2:.2f}"
        )

    with col3:
        st.metric(
            "ML Algorithm",
            "Random Forest"
        )

    with col4:
        st.metric(
            "Model Type",
            "Regression"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Score Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            df["Risk_Score"],
            bins=20
        )

        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Number of Records")
        ax.set_title("Traffic Risk Score Distribution")

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
        SafeRoad AI predicts a continuous traffic accident
        risk score from traffic and environmental conditions.

        The prediction can support traffic monitoring,
        safety planning and preventive action.
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
        "🚦 Predict Accident Risk Score",
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

        st.divider()

        # ----------------------------------------------------
        # RISK RESULT
        # ----------------------------------------------------

        st.subheader("Predicted Risk Score")

        st.metric(
            "Accident Risk Score",
            f"{prediction:.2f}"
        )

        # Simple interpretation

        if prediction >= 8:

            st.markdown(
                """
                <div class="risk-high">
                <h2>🔴 HIGH RISK SCORE</h2>
                <p>
                The predicted score indicates relatively
                high traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        elif prediction >= 4:

            st.markdown(
                """
                <div class="risk-medium">
                <h2>🟡 MEDIUM RISK SCORE</h2>
                <p>
                The predicted score indicates moderate
                traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="risk-low">
                <h2>🟢 LOW RISK SCORE</h2>
                <p>
                The predicted score indicates relatively
                lower traffic accident risk.
                </p>
                </div>
                """,
                unsafe_allow_html=True
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
        '<div class="section-title">📊 Regression Model Analytics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.3f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse:.3f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    st.divider()

    st.subheader("Actual vs Predicted Risk Score")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    ax.set_xlabel("Actual Risk Score")
    ax.set_ylabel("Predicted Risk Score")
    ax.set_title("Actual vs Predicted")

    st.pyplot(fig)

    st.divider()

    st.subheader("Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
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
        "🔍 Explainable AI for prediction reasoning"
    ]

    for item in future:

        st.markdown(
            f"### {item}"
        )

    st.divider()

    st.subheader("⚠️ Responsible AI")

    st.warning(
        """
        This prototype uses synthetic data and demonstrates
        a machine-learning regression workflow.

        A real deployment would require validated historical
        traffic and accident data, privacy safeguards,
        model monitoring and human oversight.
        """
    )

    st.subheader("🌍 Social Impact")

    st.success(
        """
        The objective is to predict traffic risk scores
        and use data-driven signals to support safer roads,
        better planning and preventive action.
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
    Regression + Machine Learning + Social Impact
    </div>
    """,
    unsafe_allow_html=True
)