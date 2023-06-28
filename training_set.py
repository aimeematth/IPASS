import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
import re
import swifter
import joblib

# Download necessary NLTK resources
nltk.download('punkt')                   # Download tokenizer models
nltk.download('stopwords')               # Download stopwords
nltk.download('wordnet')                 # Download WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # Download POS tagger model

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Initialize the WordNet lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Load profane words from a file
with open("profane_words.txt", "r") as file:
    profane_words = {word.strip() for word in file.readlines()}

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("C:\\Users\\lunac\\OneDrive\\Documents\\Schoolprojecten\\IPASS\\train.csv", encoding='latin1', dtype={'comment_text': str})

# Define a function to clean the text by removing non-alphabetic characters
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
df['Cleaned comments'] = df['comment_text'].apply(clean)

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
df['POS tagged'] = df['Cleaned comments'].swifter.apply(token_stop_pos)

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

# Apply lemmatization in parallel using swifter
df['Lemma'] = df['POS tagged'].swifter.apply(lemmatize)

# Create the mapping dictionary for profane word replacement
mapping_dict = {}

# Iterate over the unique vocabulary words in the corpus
unique_words = set(df['Lemma'].str.split(expand=True).stack().unique())
for word in unique_words:
    # Find similar (manipulated) words in the profane words list
    similar_words = [p_word for p_word in profane_words if p_word.lower().startswith(word.lower())]
    if similar_words:
        # Calculate token ratio to determine a match
        token_ratio = max(len(word) / len(p_word) for p_word in similar_words)
        if token_ratio >= 0.8:
            # Map the token to the most similar profane word
            mapping_dict[word.lower()] = max(similar_words, key=len)

# Function to check for profanity in text
def check_profanity(text):
    """
    Checks for profanity in the text.

    Args:
        text (str): Input text to be checked.

    Returns:
        str: 'True' if profanity is found, 'False' otherwise.
    """
    for word in profane_words:
        if re.search(r'\b{}\b'.format(re.escape(word)), text, re.IGNORECASE):
            return 'True'
    return 'False'

# Apply profane word replacement to the 'Lemma' column
df['Processed comments'] = df['Lemma'].apply(check_profanity)

# Function to clean special characters in text
def clean_special_chars(text):
    """
    Cleans special characters in the text.

    Args:
        text (str or tuple): Input text or tuple to be cleaned.

    Returns:
        str or list: Cleaned text or list of cleaned tuples.
    """
    if isinstance(text, tuple):
        return [(re.sub('[^A-Za-z0-9]+', ' ', word), pos) for word, pos in text]
    else:
        return re.sub('[^A-Za-z0-9]+', ' ', text)

# Clean special characters in the 'Processed comments' column
df['Processed comments'] = df['Processed comments'].apply(clean_special_chars)

# Split the dataset into training and evaluation subsets
X = df['Lemma']  # Input features
X = df['Processed comments'].str.lower()  # Processed comments as input features
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'misogyny']]  # Target labels
x_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for the classification task
# Fit a separate model for each label using the training data
label_models = {}
num_epochs = 20  # Number of epochs

for label in y_train.columns:
    print('Training model for label:', label)
    model_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),  # Convert text into a matrix of token counts
        ('tfidf', TfidfTransformer()),      # Apply TF-IDF transformation to the token counts
        ('classifier', LogisticRegression())  # Logistic regression classifier
    ])
    model = model_pipeline.fit(x_train, y_train[label])

    # Training loop with epochs
    for epoch in range(num_epochs):
        model.fit(x_train, y_train[label])
    
    label_models[label] = model


# Evaluate the models on the evaluation subset and calculate accuracy
accuracy = {}
for label, model in label_models.items():
    accuracy[label] = model.score(X_eval, y_eval[label])
    print("Accuracy for label", label, ":", accuracy[label])

# Save the pipeline to a file
joblib.dump(model_pipeline, 'pipeline.pkl')

# Save the label models to individual files
for label, model in label_models.items():
    joblib.dump(model, f'{label}_model.pkl')
