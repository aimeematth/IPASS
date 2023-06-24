import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
import re
import swifter
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

# Load profane words from a file
with open("profane_words.txt", "r") as file:
    profane_words = {word.strip() for word in file.readlines()}

df = pd.read_csv("C:\\Users\\lunac\\OneDrive\\Documents\\Schoolprojecten\\IPASS\\train.csv", encoding='latin1')

# Define a function to clean the text
def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Cleaning the text in the comments column
df['Cleaned comments'] = df['comment_text'].apply(clean)

# POS tagger dictionary
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

# Function for tokenization, stop word removal, POS tagging
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

# Apply tokenization, stop word removal, POS tagging in parallel
df['POS tagged'] = df['Cleaned comments'].swifter.apply(token_stop_pos)

# Function for lemmatization
def lemmatize(pos_data):
    lemmatized = [(wordnet_lemmatizer.lemmatize(word, pos=pos) if pos else word, pos) for word, pos in pos_data]
    return lemmatized

# Apply lemmatization in parallel
df['Lemma'] = df['POS tagged'].swifter.apply(lemmatize)

# Create the mapping dictionary
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

def check_profanity(text):
    for word in profane_words:
        for token, pos in text:
            if re.search(r'\b{}\b'.format(re.escape(word)), token, re.IGNORECASE):
                return 'True'
    return 'False'




# Apply profane word replacement
df['Processed comments'] = df['Lemma'].apply(check_profanity)

# Define a function to clean special characters
def clean_special_chars(text):
    if isinstance(text, tuple):
        return [(re.sub('[^A-Za-z0-9]+', ' ', word), pos) for word, pos in text]
    else:
        return re.sub('[^A-Za-z0-9]+', ' ', text)




# Cleaning special characters in the comments column
df['Processed comments'] = df['Processed comments'].apply(clean_special_chars)

# Split the dataset into training and evaluation subsets
X = df['Processed comments'].str.lower()
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'misogyny']]
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for the classification task
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

label_models = {}
for label in y_train.columns:
    print('Training model for label:', label)
    model_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', LogisticRegression())
    ])
    model = model_pipeline.fit(X_train, y_train[label])
    label_models[label] = model

# Evaluate the models on the evaluation subset
accuracy = {}
for label, model in label_models.items():
    accuracy[label] = model.score(X_eval, y_eval[label])
    print("Accuracy for label", label, ":", accuracy[label])

# Save the pipeline to a file
joblib.dump(pipeline, 'pipeline.pkl')

# Save the label models to individual files
for label, model in label_models.items():
    joblib.dump(model, f'{label}_model.pkl')
