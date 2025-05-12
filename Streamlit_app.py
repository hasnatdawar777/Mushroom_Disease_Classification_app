# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load trained model and label encoders
# model = joblib.load("best_mushroom_model_have.pkl")
# label_encoders = joblib.load("label_encoders.pkl")  # saved separately

# # Load feature list
# feature_names = [col for col in label_encoders.keys() if col != 'class']

# st.title("🍄 Mushroom Classification App")
# st.write("Enter the characteristics of a mushroom to predict if it is **edible or poisonous**.")

# # User input form
# user_input = {}
# with st.form("mushroom_form"):
#     for feature in feature_names:
#         options = label_encoders[feature].classes_
#         user_input[feature] = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", options)
#     submitted = st.form_submit_button("Predict")

# # Predict when submitted
# if submitted:
#     # Encode user input
#     input_df = pd.DataFrame([user_input])
#     for col in input_df.columns:
#         le = label_encoders[col]
#         input_df[col] = le.transform(input_df[col])

#     # Make prediction
#     prediction = model.predict(input_df)[0]
#     class_le = label_encoders['class']
#     prediction_label = class_le.inverse_transform([prediction])[0]

#     # Output result
#     if prediction_label == 'e':
#         st.success("✅ The mushroom is **Edible**!")
#     else:
#         st.error("⚠️ The mushroom is **Poisonous**!")


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and label encoders
model = joblib.load("best_mushroom_model_have.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # saved separately

# Load feature list
feature_names = [col for col in label_encoders.keys() if col != 'class']

st.title("🍄 Mushroom Classification App")
st.write("Enter the characteristics of a mushroom to predict if it is **edible or poisonous**.")

# Initialize prediction history in session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# User input form
user_input = {}
with st.form("mushroom_form"):
    for feature in feature_names:
        options = label_encoders[feature].classes_
        user_input[feature] = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", options)
    submitted = st.form_submit_button("Predict")

# Predict when submitted
if submitted:
    # Encode user input
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Make prediction
    prediction = model.predict(input_df)[0]
    class_le = label_encoders['class']
    prediction_label = class_le.inverse_transform([prediction])[0]

    # Store prediction in history
    record = user_input.copy()
    record["Prediction"] = "Edible ✅" if prediction_label == 'e' else "Poisonous ⚠️"
    st.session_state.prediction_history.append(record)

    # Output result
    if prediction_label == 'e':
        st.success("✅ The mushroom is **Edible**!")
    else:
        st.error("⚠️ The mushroom is **Poisonous**!")

# Display history table if any predictions were made
if st.session_state.prediction_history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df)
