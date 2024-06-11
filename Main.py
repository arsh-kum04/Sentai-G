import warnings
import streamlit as st
from datetime import datetime
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from googletrans import Translator
import plotly.express as px
import torch
import base64

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

# Initialize translator
trans1 = Translator()

# Initialize sentiment analysis model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
max_length = model.config.max_position_embeddings  # Set the max sequence length from the model config

# Function for sentiment analysis
def NLP(Data):
    cmnt = Data
    cmnt_words = []
    for word in cmnt.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        cmnt_words.append(word)
    cmnt_proc = " ".join(cmnt_words)
    labels = ['Negative', 'Neutral', 'Positive']
    encoded_cmnt = tokenizer(cmnt_proc, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    input_ids = encoded_cmnt['input_ids']
    attention_mask = encoded_cmnt['attention_mask']

    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]

    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ret = 0
    scr = scores[0]
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        if i >= 1 and scores[i] > scores[i - 1]:
            ret = i
            scr = scores[i]
    if ret == 0:
        return {"Negative": scr}
    elif ret == 1:
        return {"Neutral": scr}
    else:
        return {"Positive": scr}

# Safe translation function
def safe_translate(text, dest='en'):
    try:
        translated = trans1.translate(text, dest=dest).text
    except Exception as e:
        print(f"Error translating comment: {e}")
        translated = text  # Fallback to original text
    return translated

# Function to perform sentiment analysis and filtering
def analyze_and_filter(url, start_time=None, end_time=None, username=None, sentiment=None):
    reddit = praw.Reddit(
        user_agent=True,
        client_id=st.secrets["reddit"]["client_id"],
        client_secret=st.secrets["reddit"]["client_secret"],
        username=st.secrets["reddit"]["username"],
        password=st.secrets["reddit"]["password"]
    )
    post = reddit.submission(url=url)

    post_title = post.title
    post_author = post.author.name if post.author else "Unknown"
    post_date = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
    post_content = post.selftext if post.selftext else "No text content available."
    post_media_url = None

    if hasattr(post, 'is_video') and post.is_video:
        post_media_url = post.media['reddit_video']['fallback_url']
    elif 'media_metadata' in post.__dict__ and post.media_metadata:
        post_media_url = list(post.media_metadata.values())[0]['s']['u']

    pv, nu, nv = 0, 0, 0
    columns = ["Sentiment", "Score", "Original Text", "Translated text", "Author", "Parent", "Time Stamp"]
    DB = []

    post.comments.replace_more(limit=None)
    total_comments = len(post.comments.list())

    progress_bar = st.progress(0)
    progress_text = st.empty()
    status_text = st.empty()

    for i, comment in enumerate(post.comments.list()):
        if comment.body is None:
            continue

        if username and comment.author and comment.author.name != username:
            continue

        Translated = safe_translate(comment.body)

        D = NLP(comment.body)
        comment_sentiment = list(D.keys())[0]
        comment_score = list(D.values())[0]

        if sentiment and comment_sentiment != sentiment:
            continue

        if comment_sentiment == "Positive":
            pv += D["Positive"]
        elif comment_sentiment == "Neutral":
            nu += D["Neutral"]
        else:
            nv += D["Negative"]

        timestamp = datetime.fromtimestamp(comment.created_utc, tz=pytz.utc)
        ist = pytz.timezone('Asia/Kolkata')
        ist_time = timestamp.replace(tzinfo=pytz.utc).astimezone(ist)

        DB.append([comment_sentiment, comment_score, comment.body, Translated, str(comment.author), str(comment.parent()), ist_time.strftime('%Y-%m-%d %H:%M:%S')])

        progress_percentage = (i + 1) / total_comments
        progress_bar.progress(progress_percentage)
        progress_text.text(f'Processing comment {i + 1} of {total_comments}')
        status_text.text(f"Processing comment by {comment.author}")

    DF = pd.DataFrame(DB, columns=columns)

    plt.style.use('bmh')
    plt.xlabel('Sentiment', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    bars = plt.bar(['Positive', 'Neutral', 'Negative'], [pv, nu, nv])
    bars[0].set_color('Green')
    bars[1].set_color('Yellow')
    bars[2].set_color('Red')

    pie_fig = px.pie(DF, names='Sentiment', title='Sentiment Distribution')
    bar_fig = px.bar(DF, x='Sentiment', y='Score', color='Sentiment', title='Sentiment Analysis')

    return DF, pie_fig, bar_fig, post_title, post_author, post_date, post_content, post_media_url

# Streamlit UI
def main():
    st.title("SENTAI-G: Comprehensive Sentiment Analysis Tool")
    st.sidebar.title("Filter Options")

    url = st.sidebar.text_input("Enter Reddit URL")

    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    username = st.sidebar.text_input("Enter Username")
    sentiment = st.sidebar.selectbox("Select Sentiment", ["All", "Positive", "Neutral", "Negative"])

    if st.sidebar.button("Analyze & Filter"):
        if url:
            sentiment = None if sentiment == "All" else sentiment
            df, pie_fig, bar_fig, post_title, post_author, post_date, post_content, post_media_url = analyze_and_filter(url, None, None, username, sentiment)

            st.subheader("Reddit Post Details")
            st.write(f"**Title:** {post_title}")
            st.write(f"**Author:** {post_author}")
            st.write(f"**Date:** {post_date}")
            st.write(f"**Content:** {post_content}")
            if post_media_url:
                st.write(f"**Media URL:** {post_media_url}")

            st.subheader("Sentiment Analysis Results")
            st.plotly_chart(pie_fig)
            st.plotly_chart(bar_fig)
            st.write(df)

            def download_link(df, filename, text):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
                return href

            st.markdown(download_link(df, 'sentiment_analysis_results.csv', 'Download CSV'), unsafe_allow_html=True)
        else:
            st.error("Please enter a valid Reddit URL")

if __name__ == '__main__':
    main()
