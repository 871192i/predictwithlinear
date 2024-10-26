import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
input_data = st.text_input("Enter a sequence of numbers (comma separated):")
if input_data:
    # Process the input data
    input_numbers = [int(num) for num in input_data.split(',')]
    
    # Create a DataFrame for displaying in a table
    df = pd.DataFrame(input_numbers, columns=["Input Numbers"])
    
    # Predict the next value
    next_value = predict_next_value(input_numbers)
    
    # Show the input data as a table
    #st.subheader("Input Data")
    #st.dataframe(df)
    
    # Show the prediction result
    st.write(f"Predicted next value: {next_value:.2f}")

    # Plot the data and the predicted next value
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(input_numbers)), input_numbers, marker='o', label='Input Numbers', color='blue')
    plt.axhline(y=next_value, color='red', linestyle='--', label='Predicted Next Value')
    plt.scatter(len(input_numbers), next_value, color='red', s=100)  # Mark predicted point
    plt.title("Number Prediction")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.xticks(range(len(input_numbers) + 1))  # Update x-ticks to include the predicted point
    plt.legend()
    st.pyplot(plt)


