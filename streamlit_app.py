import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to predict the next value
def predict_next_value(data):
    X = np.array(range(len(data))).reshape(-1, 1)  # Independent variable
    y = np.array(data)  # Dependent variable
    model = LinearRegression()
    model.fit(X, y)
    next_index = np.array([[len(data)]])  # Next index for prediction
    return model.predict(next_index)[0]

# Streamlit app
st.title("Predicting the Next Number")

# Input for the user
input_data = st.text_input("Enter a sequence of numbers (comma separated): ป้อนจำนวนเต็มแล้วตามด้วย ,")
if input_data:
    # Process the input data
    input_numbers = [int(num) for num in input_data.split(',')]
    
    # Create a DataFrame for displaying in a table
    df = pd.DataFrame(input_numbers, columns=["Input Numbers"])
    
    # Predict the next value
    next_value = predict_next_value(input_numbers)
    
    # Show the input data as a table
    st.subheader("Input Data")
    st.dataframe(df)
    
    # Show the prediction result with larger red font
    st.markdown(f"<h1 style='color:red;'>{next_value:.2f}</h1>", unsafe_allow_html=True)

    # Plot the data using Streamlit's built-in chart
    st.line_chart(input_numbers + [next_value])  # Include predicted value
