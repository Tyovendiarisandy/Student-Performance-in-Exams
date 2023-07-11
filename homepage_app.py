import pandas as pd
import numpy as np
import streamlit as st
import sklearn 
import imblearn
import pickle 
from PIL import Image

# load dataset
df = pd.read_csv('clean_dataset.csv')

# load model
model = pickle.load(open('xgb_fix_tuned.pkl','rb'))
    
# create title (homepage)
def main():
    load_image = Image.open('./homepage.jpg')
    st.image(load_image)
    st.title('The Student Performance in Exams Prediction')
    st.subheader('Please Input Your Data Below!')

    # choose menu input - Selectbox
    # st.sidebar.subheader('Select Your Input')
    gender = st.selectbox('Select Your Gender!', df['gender'].unique())
    race = st.selectbox('Select Your Race or Ethnicity!', df['race'].unique())
    parental_level_of_education = st.selectbox("Select Your Parent's Level of Education!", df['parental_level_of_education'].unique())
    lunch = st.selectbox('Select Your Lunch Type!', df['lunch'].unique())
    test_preparation_course = st.selectbox('Select Your Status Test Preparation Course!', df['test_preparation_course'].unique())

    # subtitle for symptoms
    st.subheader('Select Your Exams Score!')
    
    # choose menu input - selectbox for symptoms
    math_score = st.slider('What is Your Math Score?', min_value=0, max_value=100)
    reading_score = st.slider('What is Your Reading Score?', min_value=0, max_value=100)
    writing_score = st.slider('What is Your Writing Score?', min_value=0, max_value=100)

    # prediction - button for predict
    if st.button('Predict'):
    # input the data in dataframe
        input_data = pd.DataFrame({
        'gender': [gender],
        'race': [race],
        'parental_level_of_education': [parental_level_of_education],
        'lunch': [lunch],
        'test_preparation_course': [test_preparation_course],
        'math_score': [math_score],
        'reading_score': [reading_score],
        'writing_score': [writing_score]
        })
        
        # do predict with model
        prediction = model.predict(input_data)

        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.markdown('<span style="color:red; background-color:grey">You Didn\'t The SAT Test!</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:green; background-color:grey">You Passed The SAT Test!</span>', unsafe_allow_html=True)
    
    st.write('----')
    st.write('''
    Dashboard Created by [Tyovendi Arisandy](https://www.linkedin.com/in/tyovendiarisandy/)
    ''')

if __name__=='__main__':
    main()
