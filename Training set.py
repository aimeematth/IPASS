import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
import re
import swifter

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
    lemma_rew = " ".join([wordnet_lemmatizer.lemmatize(word, pos=pos) if pos else word for word, pos in pos_data])
    return lemma_rew

# Apply lemmatization in parallel
df['Lemma'] = df['POS tagged'].swifter.apply(lemmatize)

# Split the dataset into training and evaluation subsets
X = df['Lemma']
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'misogyny']]
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for the classification task
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Fit a separate model for each label
label_models = {}
for label in y_train.columns:
    print('Training model for label:', label)
    model = pipeline.fit(X_train, y_train[label])
    label_models[label] = model

# Evaluate the models on the evaluation subset
accuracy = {}
for label, model in label_models.items():
    accuracy[label] = model.score(X_eval, y_eval[label])
    print("Accuracy for label", label, ":", accuracy[label])
