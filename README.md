# Text Summarizer App
This is an app developed in Streamlit that can summarize a text and find key words.

Access the app here: https://text-summarizer-by-henrique.streamlit.app/

### Set up
1. This was written in Python 3.11.11
2. Run ```pip install -r requirements.txt``` to install all libraries
3. Load app localy by running ```python3 -m streamlit run streamlit_app.py```

### Using the App
1. Add the English text you want to summarize
2. Select the action to be performed
3. If you want to summarize the text, then choose the maximum and minium number of words
4. Click "Analyze Text"

### About Summarize Text
BART, a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder, is used to summarize the text. The model was obtained through hugging face. As it was not developed or fine-tuned specifically for this app, quality may vary. For consistency, the randomness aspect of the model was also disabled, which may cause the sentences to be cut-off depending on how the maximum number of words is set.

For more information, see here: https://huggingface.co/facebook/bart-large-cnn

### About Find Key Words
The key words are found by using a fine-tuned version of the DistilBERT model. This model is a transformer, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model. This model was also obtained through hugging face. 

For more information, see here: https://huggingface.co/ml6team/keyphrase-extraction-distilbert-inspec

