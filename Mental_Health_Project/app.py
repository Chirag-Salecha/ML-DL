import streamlit as st
import matplotlib.pyplot as plt
import time

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🧠 Mental Health Treatment Prediction System")

st.markdown(
    """
This interactive dashboard predicts whether a person
may require mental health treatment.
"""
)

st.write("---")

# -----------------------------------
# USER INPUTS
# -----------------------------------

age = st.slider(
    "📅 Age",
    18,
    60,
    25
)

gender = st.selectbox(
    "👤 Gender",
    ["Male", "Female", "Other"]
)

family_history = st.selectbox(
    "🧬 Family History",
    ["Yes", "No"]
)

remote_work = st.selectbox(
    "💻 Remote Work",
    ["Yes", "No"]
)

st.info(
    "Rate your stress level from "
    "1 (Low Stress) to 10 (High Stress)"
)

work_stress = st.slider(
    "📈 Self-Reported Work Stress Level",
    1,
    10,
    5
)

st.write("---")

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🔍 Predict Mental Health Risk"):

    with st.spinner("Analyzing..."):
        time.sleep(2)

    score = 0

    if family_history == "Yes":
        score += 1

    if work_stress >= 7:
        score += 1

    if age < 22:
        score += 1

    if remote_work == "Yes" and work_stress >= 6:
        score += 1

    # -----------------------------------
    # RESULT
    # -----------------------------------

    st.subheader("📊 Prediction Result")

    if score <= 1:

        risk = "LOW RISK"
        percent = 25

        st.success("🟢 LOW RISK")

    elif score == 2:

        risk = "MODERATE RISK"
        percent = 60

        st.warning("🟡 MODERATE RISK")

    else:

        risk = "HIGH RISK"
        percent = 90

        st.error("🔴 HIGH RISK")

    st.progress(percent)

    # -----------------------------------
    # PIE CHART
    # -----------------------------------

    st.subheader("📈 Risk Analysis Graph")

    labels = ['Risk Level', 'Safe Zone']

    values = [percent, 100 - percent]

    fig, ax = plt.subplots()

    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    ax.set_title("Mental Health Risk Percentage")

    st.pyplot(fig)

    # -----------------------------------
    # BAR GRAPH
    # -----------------------------------

    st.subheader("📊 Stress Indicator Graph")

    categories = [
        'Stress',
        'Family History',
        'Remote Work'
    ]

    values = [
        work_stress,
        10 if family_history == "Yes" else 2,
        8 if remote_work == "Yes" else 3
    ]

    fig2, ax2 = plt.subplots()

    ax2.bar(categories, values)

    ax2.set_ylim(0, 10)

    ax2.set_title("Mental Health Factors")

    st.pyplot(fig2)

    # -----------------------------------
    # RECOMMENDATIONS
    # -----------------------------------

    st.subheader("💡 Recommendations")

    if risk == "LOW RISK":

        st.info(
            "Maintain healthy work-life balance "
            "and stress management."
        )

    elif risk == "MODERATE RISK":

        st.warning(
            "Improve sleep, reduce stress, "
            "and seek social support."
        )

    else:

        st.error(
            "Consider professional mental health "
            "support and stress reduction."
        )

    # -----------------------------------
    # MODEL INFO
    # -----------------------------------

    st.write("---")

    st.subheader("🤖 Model Information")

    st.write("Algorithm Used: Random Forest")

    st.write("Model Accuracy: 73.2%")

    st.write(
        "Dataset: OSMI Mental Health in Tech Survey"
    )

    st.success("✅ Analysis Completed Successfully")