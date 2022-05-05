import streamlit as st
import pandas as pd
import datetime
import numpy as np
from google.cloud import firestore
import os
import re
from streamlit.elements import file_uploader
from matplotlib import pyplot as plt

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "maximal-airfoil-345003-94d6109f9175.json"
db = firestore.Client()
#################################################################
################## Page Settings ################################
#################################################################
st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)
################load data from firestore########################
docs = db.collection(u'fbcommentspredict').stream()
items = []
for doc in docs:
    items.append(doc.to_dict())
    df = pd.DataFrame.from_records(items)
df['predictions'] = df['predictions'].apply(lambda x: 'positive' if x == '4' else 'negative')

#################################################################
################## Page Header ##################################
#################################################################
option = st.sidebar.selectbox(
                "Which team do you want to view", 
                ('Saints','Pelicans'))

if option == 'Saints':
    col1, col2 = st.columns((4,1))
    with col1:
        st.header("New Orleans Saints Sentiment Analysis Dashboard")
        st.write("Our application uses Logestic Regression Model to predict sentiment of user comments posted")
        st.markdown('---')
    with col2:
        st.image('Saints.png')

if option == 'Pelicans':
    col1, col2 = st.columns((4,1))
    with col1:
        st.header("New Orleans Pelicans Sentiment Analysis Dashboard")
        st.write("Our application uses Logestic Regression Model to predict sentiment of user comments posted")
        st.markdown('---')
    with col2:
        st.image('Pelicans.png')
    
        
################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Ingest", "Predictions","About"])

################################################################
################## Data cleaning Page ###################################
################################################################
if page_selected == "Home":
    
    ############ Filters ######################################
    ######### Date range slider ################################
    ######### Apply filters ####################################
    if option == 'Saints':
        option2 = st.sidebar.selectbox(
            "Which media platform do you want to view", 
            ('Instagram','Facebook'))
        if option2 == 'Instagram':
            df_filter=df[df['account_id'] == '17841400137310298']
        if option2 == 'Facebook':
            df_filter = df[df['account_id'] == '121195131261394']
    if option == 'Pelicans':
        option2 = st.sidebar.selectbox(
            "Which media platform do you want to view", 
            ('Instagram','Facebook'))
        if option2 == 'Instagram':
            df_filter=df[df['account_id'] == '17841400085890242']
        if option2 == 'Facebook':
            df_filter = df[df['account_id'] == '186563232926']

    if df_filter.shape[0] > 0:
        ######### Main Story Plot ###################################
        col1, col2 = st.columns((2,1))
        with col1: 
            ax = pd.crosstab(df_filter.created_time, df_filter.predictions).plot(
                    kind="bar", 
                    figsize=(6,2), 
                    xlabel = "Date",
                    color={'positive':'skyblue', 'negative': 'orange'})
            st.pyplot(ax.figure)


        ####################################################
        ############## ingest page ########################3
elif page_selected == "Ingest":
    st.subheader("Choose the dataset")
    data_file = st.file_uploader("Choose a CSV")
    if data_file is not None:
        df =pd.read_csv(data_file)
        st.table(df)
    
        ################################################################
elif page_selected == "Predictions":
    st.header("New Orleans Pelicans Dashboard")
    st.subheader("Our Prediction Results")
    
    ########load data ###################
    st.write("Dashboard of Sentiment Analysis")

    ############ Filters ######################################
    ######### Date range slider ################################
    # start, end = st.sidebar.select_slider(
    #                 "Select Date Range", 
    #                 df.created_time.drop_duplicates().sort_values(), 
    #                 value=(df.created_time.min(), df.created_time.max()))
    
    
    ######### Apply filters ####################################
    #df_filter = df.loc[(df.created_time >= start) & (df.created_time <= end), :]
    ###############################################
    ###############################################
   
    # dashboard title
    st.title("Negative | netural | Positive ratio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Negative", "< -0.2")
    col2.metric("Netrual ", "-0.2 <= score <= 0.2")
    col3.metric("Positive", "> 0.2")
    st.bar_chart(df['Sentiment'].value_counts())


    from wordcloud import WordCloud 
    from wordcloud import STOPWORDS
    
    df['message']=df['message'].astype(str)
    # dashboard title
    st.title('wordcloud of all message')
    wc = WordCloud(background_color='white',
                                    stopwords =  set(STOPWORDS),
                                    max_words = 50, 
                                    random_state = 42,)
    wc.generate(' '.join(df['message']))
    st.image(wc.to_image())

        #dashboard of negative /positive
    df_negative = df[df['Sentiment'] == 'Negative']
    df_negative.rename(columns={'Sentiment':'negative_comment'},inplace=True)
    
    ######## Sample Reviews and Sentiment Predictions ###############
    st.subheader("Sample Reviews and Sentiment Predictions")
    
    df_sample = df.head(6)
    if df.shape[0] > 0:
        for index, row in df_sample.iterrows():
            col1, col2 = st.columns((1,5))
            with col1:
                if row['Sentiment'] == "Positive":
                    st.success("Positive") 
                elif row['Sentiment'] == "Neutral":
                    st.success("Netural")
                else: 
                    st.error("Negative")
            with col2:
                st.write(str(row['message']))
    else:
        st.warning("Your selection returned no data. Change your selecton.")



    df_positive = df[df['Sentiment'] == 'Positive']
    df_positive.rename(columns={'Sentiment':'positive_comment'},inplace=True)

        # dashboard title
    st.title('wordcloud of negative_message')
    wc2 = WordCloud(background_color='white',
                                stopwords =  set(STOPWORDS),
                                max_words = 50, 
                                random_state = 42,)
    wc2.generate(' '.join(df_negative['message']))
    st.image(wc2.to_image())


    st.title('wordcloud of positive_message')
    wc3 = WordCloud(background_color='white',
                                stopwords =  set(STOPWORDS),
                                max_words = 50, 
                                random_state = 42,)
    wc3.generate(' '.join(df_positive['message']))
    st.image(wc3.to_image())

        ################################################################
        ############### About My Company and Team ######################
        ################################################################

else:
    col1, col2 = st.columns((1,40))
    with col2:
        st.write("New Orleans Saints & Pelicans Sentiment Analysis ")
    st.subheader('Team')
    ### Member 1
    col1, col2 = st.columns((1,40))
    with col2:
        st.markdown('**Group 2: Morgan Esters, Chen Ge, Conner McCoy, Mingzhe Song, Zoe Xu, Mingzhe Xue, Tianyuan Zhang**')
    col11, col12, col13 = st.columns((1,6,34))
    with col13:
        st.write('Business Analytics Projects Class')
        
