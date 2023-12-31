import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
import re
import swifter
import joblib
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

wordnet_lemmatizer = WordNetLemmatizer()

# Load the trained pipeline and label models
pipeline = joblib.load('pipeline.pkl')
label_models = {}
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'misogyny']:
    label_models[label] = joblib.load(f'{label}_model.pkl')

# Load the test dataset
df_test = pd.read_csv("C:\\Users\\lunac\\OneDrive\\Documents\\Schoolprojecten\\IPASS\\test.csv", encoding='latin1')

# Define a function to clean the text
def clean(text):
    """
    Cleans the text by removing non-alphabetic characters.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Apply the clean function to the 'comment_text' column and create a new column 'Cleaned comments'
df_test['Cleaned comments'] = df_test['comment_text'].apply(clean)

# POS tagger dictionary for mapping tags to WordNet POS tags
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

# Function for tokenization, stop word removal, and POS tagging
def token_stop_pos(text):
    """
    Tokenizes the text, removes stopwords, and performs POS tagging.

    Args:
        text (str): Input text to be processed.

    Returns:
        list: List of tuples containing word and POS tag pairs.
    """
    # Tokenize the text and apply POS tagging
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            # Remove stop words and map POS tags to WordNet POS tags
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

# Apply tokenization, stop word removal, and POS tagging in parallel using swifter
df_test['POS tagged'] = df_test['Cleaned comments'].swifter.apply(token_stop_pos)

# Function for lemmatization
def lemmatize(pos_data):
    """
    Lemmatizes the text based on POS tags.

    Args:
        pos_data (list): List of tuples containing word and POS tag pairs.

    Returns:
        str: Lemmatized text.
    """
    # Lemmatize each word using WordNet lemmatizer and its corresponding POS tag
    lemma_rew = " ".join([wordnet_lemmatizer.lemmatize(word, pos=pos) if pos else word for word, pos in pos_data])
    return lemma_rew

# Preprocess the test data
def preprocess_text(text):
    cleaned_text = clean(text)
    pos_tagged = token_stop_pos(cleaned_text)
    lemmatized_text = lemmatize(pos_tagged)
    return lemmatized_text

# Apply preprocessing to test data
df_test['Preprocessed comments'] = df_test['Cleaned comments'].swifter.apply(preprocess_text)

# Get the preprocessed test data
X_test = df_test['Preprocessed comments']

# Get the predicted labels using the pipeline
# Get the predicted labels using the pipeline
y_pred = {}
for label in label_models:
    y_pred[label] = label_models[label].predict(X_test)

# Convert predicted labels to 0 or 1
predicted_labels = {}
for i, comment_id in enumerate(df_test['id']):
    predicted_labels[comment_id] = {
        'toxic': int(y_pred['toxic'][i]),
        'severe_toxic': int(y_pred['severe_toxic'][i]),
        'obscene': int(y_pred['obscene'][i]),
        'threat': int(y_pred['threat'][i]),
        'insult': int(y_pred['insult'][i]),
        'identity_hate': int(y_pred['identity_hate'][i]),
        'misogyny': int(y_pred['misogyny'][i])
    }

# Print the predicted labels
print("Predicted Labels:", predicted_labels)