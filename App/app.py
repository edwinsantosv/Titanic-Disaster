import streamlit as st
import Data_Analysis  # Import the analysis page
import Prediction  # Import the prediction page

# Sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page:', ['Data Analysis', 'Prediction'])

# Page rendering
if page == 'Data Analysis':
    Data_Analysis.show_analysis()
elif page == 'Prediction':
    Prediction.show_prediction()
