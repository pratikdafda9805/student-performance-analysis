import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

df=pd.read_csv('final_df.csv',index_col=False)
model = pkl.load(open('final_model.pkl','rb'))

st.header('Student Performance Analysis')
#         'Hours_Studied', 'Attendance', 'Parental_Involvement',
#        'Access_to_Resources', 'Previous_Scores', 'Internet_Access',
#        'Tutoring_Sessions', 'Family_Income', 'Peer_Influence',
#        'Learning_Disabilities', 'Distance_from_Home'

col1, col2, col3 = st.columns(3)

with col1:
    Hours_Studied=st.number_input('Hours_Studied',min_value=0, max_value=40, value=00, step=1)
    Attendance=st.text_input('Attendance')
    Previous_Scores=st.text_input('Previous_Scores')
    Tutoring_Sessions=st.number_input('Tutoring_Sessions',min_value=0, max_value=5, value=00, step=1)
with col2:
    Parental_Involvement=st.selectbox('Parental_Involvement',options= df['Parental_Involvement'].unique())
    Access_to_Resources=st.selectbox('Access_to_Resources',options=df['Access_to_Resources'].unique())
    Internet_Access=st.selectbox('Internet_Access',options= df['Internet_Access'].unique())
    Family_Income=st.selectbox('Family_Income',options=df['Family_Income'].unique())
with col3:
    Peer_Influence=st.selectbox('Peer_Influence',options=df['Peer_Influence'].unique())
    Learning_Disabilities=st.selectbox('Learning_Disabilities',options=df['Learning_Disabilities'].unique())
    Distance_from_Home=st.selectbox('Distance_from_Home',options=df[~df['Distance_from_Home'].isnull()]['Distance_from_Home'].unique())

if st.button('Predict'):

    prediction = model.predict(pd.DataFrame([[Hours_Studied,Attendance,Parental_Involvement,Access_to_Resources,Previous_Scores,Internet_Access,Tutoring_Sessions,Family_Income,Peer_Influence,Learning_Disabilities,Distance_from_Home]],columns=['Hours_Studied','Attendance','Parental_Involvement','Access_to_Resources','Previous_Scores','Internet_Access','Tutoring_Sessions','Family_Income','Peer_Influence','Learning_Disabilities','Distance_from_Home']))[0]
    st.header(np.round(prediction,2))
