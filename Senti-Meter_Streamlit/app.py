'''
Program          : Certificate Course in Data Science
Offered by       : 'RV College of Engineering (RVCE)' and 'Boston IT Solutions India Pvt. Ltd'
Capstone project : Build a dashboard for customer grievance data from messaging platform”
Tools            : spaCy - Open-source python library for NLP
                   BERT (Bidirectional Encoder Representations from Transformers) - Open source machine learning framework for natural language processing (NLP)
                   Streamlit - Open-source python framework to build userinterface and for production model deployment
Team Members     : Harish Rajanna
                   Rahul Arya
'''

# Importing all necessary libraries
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from io import StringIO


# Initiating Streamlit environment and User interface
st.set_page_config(
    page_title='Senti-Meter',
    page_icon=':triangular_ruler:',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
 )

with open('styles/style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Definitions for sentiment analysis
positiveSmliey = Image.open('assets/positive-smiley-face.png')
negativeSmliey = Image.open('assets/negative-smiley-face.png')
neutralSmliey = Image.open('assets/neutral-smiley-face.png')

#################################################################
# Function run_sentiment_analysis takes NLP model name and input
# text for processing and returns predicted sentiment score.
#################################################################
def run_sentiment_analysis(model, text):
    sentiment_score = []

    sentiment = model(text)

    sentiment_score.append(str(round(sentiment.cats["positive"]*100, 2)) + '%')
    sentiment_score.append(str(round(sentiment.cats["negative"]*100, 2)) + '%')
    sentiment_score.append(str(round(sentiment.cats["neutral"]*100, 2)) + '%')
    
    sentiment_face = sentiment_score.index(max(sentiment_score))

    return sentiment_score, sentiment_face


# Project Title with Similey faces
st.title(":blue[Senti-Meter] :smiley: :neutral_face: :disappointed:")
title_alignment="<style>#senti-meter {text-align: center}</style>"
st.markdown(title_alignment, unsafe_allow_html=True)
st.write("---")

# NLP project topics
st.write(":blue[A Capstone Project for *Certificate Course in Data Science* offered by **RV College of Engineering (RVCE)** and **Boston IT Solutions India Pvt. Ltd**]")
st.write(":blue[*NLP project topic:*] :green[Build a dashboard for customer grievance data from messaging platform.]")
st.write(":blue[*Team members:*] Harish Rajanna, Rahul Arya.")
st.write("---")


# Creating 3 Tabbed interface in the user interface
default_dataset, custom_dataset, custom_query = st.tabs(["Default Dataset", "Custom Dataset", "Custom Query"])


#################################################################################################################
# Tab 1 : To load Dataset used for Model training and demonstrate sentiment analysis
#################################################################################################################
with default_dataset:
    col1, col2 = st.columns([1,9])
    # Read default Dataset (Default_Dataset.csv) using Pandas
    col1.write("**Step 1**")
    col2.write("Reading Default_Dataset.csv using pd.read_csv() command.")
    df = pd.read_csv("Default_Dataset.csv", encoding = "ISO-8859-1")
    st.subheader(":green[**Completed...**]")

    col1, col2 = st.columns([1,9])
    col1.write("**Step 2**")
    col2.write("Displaying dataset sample using df.head() command:")
    st.write(df.head())

    col1, col2 = st.columns([1,9])
    col1.write("**Step 3**")
    col2.write("Displaying dataset shape using df.shape() command:")
    st.subheader(f":green[Total number of observations and features : :blue[**{df.shape}**]]")

    # Integrate Neural netwrok model for sentiment analysis
    # Test the data from the best model
    col1, col2 = st.columns([1,9])
    col1.write("**Step 4**")
    col2.write("Integrating best accurate model trained using spaCy and Bert transformer:")
    nlp_model = pickle.load(open("NLP_Model/spacy_bert_model.sav", "rb"))
    st.subheader(":green[**Completed...**]")

    col1, col2 = st.columns([1,9])
    col1.write("**Step 5**")
    col2.write("Displaying actual sentiment (shown in Dataset) for the loaded dataset:")

    # Plotting the pie chart for above dataframe
    df['Sentiment'].value_counts()
    chartValues = []
    allSentiments = df['Sentiment'].value_counts()
    for i in allSentiments:
        chartValues.append(round(i/df['Sentiment'].count()*100,2))
    plt.pie(chartValues, labels=['Positive', 'Negative', 'Neutral'], radius=1.5 , colors=['#AFD3E2', '#19A7CE', '#146C94'],startangle=90, autopct=' %1.1f%%')
    centre_circle = plt.Circle((0,0),1.0,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    #plt.legend([f'Positive = {chartValues[0]}%', f'Negative = {chartValues[1]}%', f'Neutral = {chartValues[2]}%'], bbox_to_anchor =(0.11, 0.75),  ncol = 1)
    plt.savefig("assets/default_dataset_graph.png")
    plt.close()
    st.image("assets/default_dataset_graph.png", width=100, use_column_width='auto')


    st.write("---")
    col1, col2 = st.columns([1,9])
    col1.write("**Step 6**")
    col2.write("Let user select an observation number for sentiment analysis.")
    min_value = 0
    max_value = df.shape[0]-1
    # Allow user to choose observation using slider function
    obsNum = st.slider(':blue[**Select an observation number for sentiment analysis:**]', min_value, max_value, 123)

    senti_text = df['Text'][obsNum]
    st.subheader(f":blue[*For the selected observation number :red[{obsNum}] :*]")

    # Display sentiment score for above text
    with st.container():
        col1, col2, col3 = st.columns([2.5,1.5,1.5], gap="small")
        
        col1.write(f"**Text :**")
        col1.write(f":blue[*{senti_text}*]")
        col2.write(f"**Actual Sentiment:**")
        col2.write(f":blue[*{df['Sentiment'][obsNum]}*]")

        sentiment_score, sentiment_face_idx = run_sentiment_analysis(nlp_model, senti_text)
        col3.write(f"**Predicted Sentiment :**")
        col3.write(f":green[Positive :  {sentiment_score[0]}]")
        col3.write(f":green[Negative :  {sentiment_score[1]}]")
        col3.write(f":green[Neutral :  {sentiment_score[2]}]")
    
        if sentiment_face_idx == 0:
            sentiment_face = positiveSmliey
        elif sentiment_face_idx == 1:
            sentiment_face = negativeSmliey
        elif sentiment_face_idx == 2:
            sentiment_face = neutralSmliey
        col3.image(sentiment_face, width=50, use_column_width='auto')


#################################################################################################################
# Tab 2 : To let user load a Custom Dataset to demonstrate sentiment analysis
#################################################################################################################
with custom_dataset:
    uploaded_file = None
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    
    disable_button = True

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        #st.write(bytes_data)
        cdf = pd.read_csv(StringIO(bytes_data.decode()), sep=",", encoding = "ISO-8859-1")
        st.write("filename:", uploaded_file.name)
        disable_button = False
    else:
        disable_button = True

    st.write("---")
    
    if st.button("Start Analytics    :point_left:", disabled=disable_button):
        st.subheader(":blue[*Sample data :*]")
        st.write(cdf.head())

        st.subheader(":blue[*Visual sentiment :*]")
        col1, col2 = st.columns(2, gap="small")
        col1.write("*Actual sentiment from loaded dataset :*")
        col2.write("*Predicted sentiment for loaded dataset :*")

        #-------------------------------------------------------------------------------
        # Plotting the pie chart for actual sentiment
        #-------------------------------------------------------------------------------
        chartValues = []
        allSentiments = cdf['Sentiment'].value_counts(sort=False)
        for i in allSentiments:
            chartValues.append(round(i/cdf['Sentiment'].count()*100,2))

        plt.pie(chartValues, labels=['Positive', 'Negative', 'Neutral'], colors=['#9CA777', '#FEE8B0', '#F97B22'], radius=1.5, startangle=90, autopct='%1.1f%%')
        centre_circle = plt.Circle((0,0),1.0,color='black', fc='white',linewidth=0)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        #plt.legend([f'Positive = {chartValues[0]}%', f'Negative = {chartValues[1]}%', f'Neutral = {chartValues[2]}%'], bbox_to_anchor =(0.15, 1.15),  ncol = 1)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig("assets/custom_dataset_actual_graph.png")
        plt.close()

        col1.image("assets/custom_dataset_actual_graph.png", width=400, use_column_width='always')

        #-------------------------------------------------------------------------------
        # Plotting the pie chart for predicted sentiment
        #-------------------------------------------------------------------------------
        predicted_sentiments = []
        predicted_chartValues = []
        for senti_text in cdf['Text']:
            sentiment_score, sentiment_type = run_sentiment_analysis(nlp_model, senti_text)
            predicted_sentiments.append(sentiment_type)

        predicted_sentiments_all = pd.Index(predicted_sentiments).value_counts(sort=False)

        for i in predicted_sentiments_all:
            predicted_chartValues.append(round(i/len(predicted_sentiments)*100,2))

        plt.pie(predicted_chartValues, labels=['Positive', 'Negative', 'Neutral'], radius=1.5 , colors=['#AFD3E2', '#19A7CE', '#146C94'],startangle=90, autopct=' %1.1f%%')
        centre_circle = plt.Circle((0,0),1.0,color='black', fc='white',linewidth=0)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        #plt.legend([f'Positive = {predicted_chartValues[0]}%', f'Negative = {predicted_chartValues[1]}%', f'Neutral = {predicted_chartValues[2]}%'], bbox_to_anchor =(0.15, 1.15),  ncol = 1)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig("assets/custom_dataset_predicted_graph.png")
        plt.close()

        col2.image("assets/custom_dataset_predicted_graph.png", width=400, use_column_width='always')

        st.subheader(":green[**Completed...**]")


        #-------------------------------------------------------------------------------
        # Porcess every entry in the loaded dataset and predict sentiment and display
        #-------------------------------------------------------------------------------
        st.subheader(":blue[*Predictions for every entry in the Dataset :*]")
        count = 0
        for senti_text in cdf['Text']:
            count += 1
            col1, col2, col3, col4 = st.columns([1,3.1,1.5,1.5], gap="small")
            col1.write(f"**Entry {count}:**")
            col2.write(f"**Text :**")
            col2.write(f":blue[*{senti_text}*]")
            col3.write(f"**Actual Sentiment:**")
            col3.write(f":blue[*{cdf['Sentiment'][count-1]}*]")

            sentiment_score, sentiment_face_idx = run_sentiment_analysis(nlp_model, senti_text)
            col4.write(f"**Predicted Sentiment:**")
            col4.write(f":green[Positive : {sentiment_score[0]}]")
            col4.write(f":green[Negative : {sentiment_score[1]}]")
            col4.write(f":green[Neutral : {sentiment_score[2]}]")
        
            if sentiment_face_idx == 0:
                sentiment_face = positiveSmliey
            elif sentiment_face_idx == 1:
                sentiment_face = negativeSmliey
            elif sentiment_face_idx == 2:
                sentiment_face = neutralSmliey
            col4.image(sentiment_face, width=5, use_column_width='always')


#################################################################################################################
# Tab 3 : User interface to accept a Custom query to demonstrate sentiment analysis
#################################################################################################################
with custom_query:
    # Process realtime sentiment analysis using user text promt
    user_text = st.text_area("Please provide your sentence for realtime sentiment analysis:", value="", height=10, max_chars=500, label_visibility="visible")

    # Disable button until some text is available for processing
    disable_button = False if user_text is not None else True

    if st.button("Start Analytics      :point_left:", disabled=disable_button):
        st.write("---")
        # Display sentiment score for above text
        with st.container():
            col1, col2 = st.columns([4,1.5], gap="small")
            
            col1.write(f"**User Text :**")
            col1.write(f":blue[*{user_text}*]")

            sentiment_score, sentiment_face_idx = run_sentiment_analysis(nlp_model, user_text)
            col2.write(f"**Predicted Sentiment :**")
            col2.write(f":green[Positive :  {sentiment_score[0]}]")
            col2.write(f":green[Negative :  {sentiment_score[1]}]")
            col2.write(f":green[Neutral :  {sentiment_score[2]}]")
        
            if sentiment_face_idx == 0:
                sentiment_face = positiveSmliey
            elif sentiment_face_idx == 1:
                sentiment_face = negativeSmliey
            elif sentiment_face_idx == 2:
                sentiment_face = neutralSmliey
            col2.image(sentiment_face, width=50, use_column_width='always')


st.text("")   # For visual spacing
with st.container():
    st.text("")
    st.text("")
    st.write("---")
    st.write("For details of Certificate Course in Data Science offered by **RV College of Engineering (RVCE)** and **Boston IT Solutions India Pvt. Ltd**")
    st.write("**Click Here** :point_right: https://rvce.edu.in/certificate-course-data-science")
    st.write("---")

