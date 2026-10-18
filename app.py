
import streamlit as st
import pickle

# Load the trained NLP model
with open("student_query_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Student Query Categorization",
    page_icon="🎓",
    layout="centered"
)

# Application heading
st.title("🎓 Student Query Categorization")
st.write(
    "A Natural Language Processing (NLP) application "
    "that categorizes student questions automatically."
)

st.divider()

# Example questions
st.subheader("How can we help you?")
st.write("Enter a question related to your college.")

st.caption(
    "Examples: Exam results, fee payment, attendance, "
    "placements, or technical support."
)

# Input field
query = st.text_area(
    "Enter your student query",
    placeholder="Example: How can I check my exam results?",
    height=100
)

# Predict button
if st.button("Predict Category", type="primary"):
    if query.strip():

        # Predict category
        prediction = model.predict([query])[0]

        # Display result
        st.success("Query categorized successfully!")

        st.subheader("Prediction Result")
        st.write("**Student Query:**", query)
        st.write("**Predicted Category:**", prediction)

        # Category descriptions
        descriptions = {
            "Attendance": "Questions about attendance, absences, and attendance percentage.",
            "Examination": "Questions about exams, results, marks, and exam timetables.",
            "Fees": "Questions about fee payments, due dates, and fee receipts.",
            "Placements": "Questions about campus placements, jobs, and interviews.",
            "Technical Support": "Questions about login problems, portals, and technical issues."
        }

        st.info(
            descriptions.get(
                prediction,
                "This category describes the type of student query."
            )
        )

        # Show model confidence if supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([query])[0]
            confidence = max(probabilities) * 100

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

    else:
        st.warning("Please enter a question first.")

# Footer
st.divider()
st.caption("Student Query Categorization | NLP  Project")


