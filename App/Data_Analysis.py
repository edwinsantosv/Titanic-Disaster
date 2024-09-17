import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def show_analysis():
    # Load the data
    st.title('Titanic Data Analysis')
    df = pd.read_csv('train.csv')

    st.subheader('Initial Data')
    st.write(df.head())  # Display the first few rows of raw data

    # Data Cleaning (following the same steps)
    st.subheader('Data Cleaning')
    
    # Step 1: Drop unnecessary columns
    st.write('Dropping unnecessary columns (Cabin, Ticket, Name, PassengerId)...')
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)

    # Step 2: Fill missing values in Age with the median
    st.write('Filling missing values in the Age column with the median...')
    df['Age'].fillna(df.Age.median(), inplace=True)

    # Step 3: Fill missing values in Embarked with the most common value ('S')
    st.write('Filling missing values in the Embarked column with the most common port (S)...')
    df['Embarked'].fillna("S", inplace=True)

    # Step 4: Convert categorical columns to numeric
    st.write('Converting categorical columns to numeric (Sex and Embarked)...')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    st.write('Data after cleaning:')
    st.write(df.head())

    # Step 5: Scaling numerical columns ('Age', 'Fare')
    st.write('Applying StandardScaler to numerical columns (Age, Fare)...')
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    st.write('Data after scaling:')
    st.write(df.head())

    # Visualizing correlations between variables
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Prepare data for the model
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression Model
    st.subheader('Logistic Regression Model Training')
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model evaluation
    st.subheader('Model Evaluation')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Model accuracy: {accuracy:.2f}')

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
