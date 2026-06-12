import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy: 96.76%")
st.sidebar.write("Fake Job Recall: 87%")

# Main Page
st.title("🛡️ AI-Powered Fake Job & Internship Detector")

st.write(
    "Paste a job or internship description below and the AI model will predict whether it is genuine or potentially fraudulent."
)

job_text = st.text_area("Job Description")

if st.button("Check Job"):

    if job_text.strip() == "":
        st.warning("Please enter a job description.")

    else:
        job_vector = vectorizer.transform([job_text])

        prediction = model.predict(job_vector)
        probability = model.predict_proba(job_vector)

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.error("⚠ Fake Job Detected")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.success("✅ Genuine Job Posting")
            st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Safety Tips")
        st.markdown("""
        - Never pay money to get a job.
        - Verify company websites and emails.
        - Be cautious of unrealistic salaries.
        - Check company reviews before applying.
        """)

st.markdown("---")
st.caption("Developed using NLP, TF-IDF and Logistic Regression")