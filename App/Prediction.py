import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

def show_prediction():
    # Load the data and apply the same cleaning steps
    df = pd.read_csv('train.csv')
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
    df['Age'].fillna(df.Age.median(), inplace=True)
    df['Embarked'].fillna("S", inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Prepare the data for the model
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    st.title("Titanic Survival Prediction")

    # User inputs
    pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3], index=2)
    sex = st.selectbox('Sex', ['male', 'female'], index=0)
    age = st.slider('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].median()))
    sibsp = st.number_input('Number of Siblings/Spouses aboard (SibSp)', min_value=0, max_value=8, value=int(df['SibSp'].median()))
    parch = st.number_input('Number of Parents/Children aboard (Parch)', min_value=0, max_value=6, value=int(df['Parch'].median()))
    fare = st.number_input('Fare', min_value=0.0, max_value=float(df['Fare'].max()), value=float(df['Fare'].median()))
    embarked = st.selectbox('Port of Embarkation (Embarked)', ['S', 'C', 'Q'], index=0)

    # Convert user input to the correct format
    sex = 0 if sex == 'male' else 1
    embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]

    # Create a dataframe for the input features
    input_features = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # Add a predict button
    if st.button('Predict'):
        # Make the prediction
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features)

        # Display the prediction results
        st.subheader('Prediction Result')
        st.write(f"Survival Probability: {prediction_proba[0][1]:.2f}")
        st.write("Prediction:", "Survived" if prediction[0] == 1 else "Did not survive")
