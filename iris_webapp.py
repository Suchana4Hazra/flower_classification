import streamlit as st
import time as t
import pickle
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Load models
lin_model = pickle.load(open('lin_model.pkl', 'rb'))
log_model = pickle.load(open('log_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))

# Function to calculate and display confusion matrix

# Function to classify and display images
def classify(num):
    if num == 0:
        st.image("setosa.jpg", caption='Setosa', use_column_width=True)
        return 'Setosa'
    elif num == 1:
        st.image("versicolor.jpeg", caption='Versicolor', use_column_width=True)
        return 'Versicolor'
    else:
        st.image("virginica.jpeg", caption='Virginica', use_column_width=True)
        return 'Virginica'

def main():
    st.title("Iris Flower Classification")

    # Styling
    st.markdown(
        """
        <style>
           .stApp{
            background-color: rgb(25, 69, 68);
           }
           .st-emotion-cache-6qob1r, .eczjsme3{
            background-color: rgb(25, 44, 68);
           }
           .footer-text{
            
              color: #7a7ad7;
              margin-left: 70%;
              font-style: italic;
           }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Header
    st.markdown("""
    <div style="background-color: hsl(9, 100%, 64%);padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar and Inputs
    activities = ['Linear Regression', 'Logistic Regression', 'SVM']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    
    sl = st.slider('Select Sepal Length', 0.0, 10.0)
    sw = st.slider('Select Sepal Width', 0.0, 10.0)
    pl = st.slider('Select Petal Length', 0.0, 10.0)
    pw = st.slider('Select Petal Width', 0.0, 10.0)
    inputs = [[sl, sw, pl, pw]]
    
    # Classification
    if st.button('Classify'):
      with st.spinner("Just wait...."):
        t.sleep(2)
        st.write("Predicted Class:")
        if option == 'Linear Regression':
            st.success(classify(lin_model.predict(inputs)[0]))
        elif option == 'Logistic Regression':
            st.success(classify(log_model.predict(inputs)[0]))
        else:
            st.success(classify(svc_model.predict(inputs)[0]))


    footer_html = """
        <footer>
            <p class="footer-text">Developed by <b>Suchana Hazra</b></p>
        </footer>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Dictionary to map options to their corresponding image filenames
    dict_img = {
      "See the types of flower": "",
      "Setosa": "setosa.jpg",
      "Virginica": "virginica.jpeg",
      "Versicolor": "versicolor.jpeg"
    }

# Selectbox for image selection
    with st.spinner("Just wait....."):
          t.sleep(2)
    selected_img = st.sidebar.selectbox("", list(dict_img.keys()))

# Display the corresponding image if a valid option is selected
    if selected_img != "See the types of flower":
          st.sidebar.image(dict_img[selected_img], caption=f"{selected_img} Image", use_column_width=True)


st.sidebar.subheader("[Please use the desktop site for better experience]")
st.sidebar.info("The model can classify a flower in one of the three species based on 4 properties -> sepal length, sepal width, petal length, petal width")
st.sidebar.info("You can choose one model(Linear regression, Logistic Regression, SVM classifier to classify flowers)..Different model may provide different predictions.Linear Regression have lower accuracy value than rest two models.")

if __name__ == '__main__':
    main()
