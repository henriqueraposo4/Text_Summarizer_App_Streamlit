import streamlit as st
import pandas as pd
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)
from transformers.pipelines import AggregationStrategy
import numpy as np
from langdetect import detect
import re

### Set up App ###

#title 
st.title('Text Summarizer')

#app explanation
st.markdown(
    """
    **Welcome to the Text Summarizer App!**

    This app allows you to summarize your text in two ways:

    1. **Summarization**: Generate a concise summary of your text.
    2. **Keyphrase Extraction**: Extract key phrases from your text.

    Enter your text below, select an action, and see the results!
    """
)

#user input
user_text = st.text_area("Enter your text below:", height=200)

# Function to check if input text is in English and contains words
def check_input_text(text):
    # Check if the input text contains words (not just spaces or punctuation)
    if len(re.findall(r'[a-zA-Z]+', text)) == 0:
        return "Please enter some valid words for analysis."

    # Check if the text is in English using langdetect
    try:
        if detect(text) != 'en':
            return "The input text is not in English. Please provide text in English."
    except:
        return "There was an issue detecting the language. Please ensure the text is clear and in English."

    return None


#select action
action = st.radio("Select an Action:", ("Summarize Text", "Find Key Words"))


### Summarizer ###
#function to summarize text
def summarize_text(text,max_words,min_words):
    #import model from hugging face
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    #set parameters
    #1 word is 1-1.5 tokens, 
    max_length_token =int(max_words * 1.3)
    min_length_token =int(min_words * 1.3)

    #generate output
    output = summarizer(text, max_length=max_length_token, min_length=min_length_token, do_sample = False)

    return output[0]['summary_text']

### Key words Extraction ###
#function to extrack the key words
def extract_key_words(text):

    # Define keyphrase extraction pipeline 
    class KeyphraseExtractionPipeline(TokenClassificationPipeline):
        def __init__(self, model, *args, **kwargs):
            super().__init__(
                model=AutoModelForTokenClassification.from_pretrained(model),
                tokenizer=AutoTokenizer.from_pretrained(model),
                *args,
                **kwargs
            )

        def postprocess(self, all_outputs):
            results = super().postprocess(
                all_outputs=all_outputs,
                aggregation_strategy=AggregationStrategy.FIRST,
            )
            return np.unique([result.get("word").strip() for result in results])

    model_name = "ml6team/keyphrase-extraction-distilbert-inspec"

    #create model
    key_word_extractor = KeyphraseExtractionPipeline(model=model_name)

    #get key words
    key_words = key_word_extractor(user_text)

    return key_words


# Parameters for summarization
if action == "Summarize Text":
    max_words = st.number_input("Enter Maximum Words for Summary:", value=30, step=1)
    min_words = st.number_input("Enter Minimum Words for Summary:", value=10, step=1)

    # Add a warning about tokenization
    st.warning(
        "Please note: The summarized text may not exactly match the specified number of words due to tokenization, but it will approximate the desired reulst"
    )


# Button to trigger analysis
if st.button("Analyze Text"):
    #check if there is an input
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        # Check if the input text contains words and is in English
        input_check_message = check_input_text(user_text)
        if input_check_message:
            st.warning(input_check_message)
        else:
            with st.spinner('The model is running... Please wait'):
                if action == "Summarize Text":
                    summary = summarize_text(user_text, max_words, min_words)  # Pass max_words and min_words
                    st.subheader("Summary")
                    st.success(summary)
                elif action == "Find Key Words":
                    key_words = extract_key_words(user_text)
                    st.subheader("Key Words")
                    st.success(", ".join(key_words))

#to run app
#python3 -m streamlit run streamlit_app.py