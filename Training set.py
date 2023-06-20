import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
import re
import swifter  # Import swifter library for parallel processing

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
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Cleaning the text in the review column
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

print(df.head())

# class YourModel:
#     def __init__(self):
#         self.base_model = Pipeline([
#             ('vect', CountVectorizer()),
#             ('tfidf', TfidfTransformer()),
#             ('clf', LogisticRegression())
#         ])
    
#     def _get_base_model_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#         train_x = X.copy()
#         train_y = y[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

#         # Filter out toxic comments
#         toxic_mask = (train_y == 0).all(axis=1)
#         train_x = train_x.loc[toxic_mask]
#         train_y = train_y.loc[toxic_mask]

#         train_x = train_x.reset_index(drop=True)
#         train_y = train_y.reset_index(drop=True)
        
#         return train_x, train_y['toxic']  # Returning the 'toxic' column for compatibility

#     def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
#         train_x, train_y = self._get_base_model_data(X, y)
#         self.base_model.fit(train_x, train_y, **kwargs)

# df = pd.read_csv("C:\\Users\\lunac\\OneDrive\\Documents\\Schoolprojecten\\IPASS\\train.csv", encoding='latin1')

# X = df['comment_text']
# y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Perform oversampling on the training set
# oversampler = RandomOverSampler(random_state=42)
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.to_frame(), y_train['toxic'])

# model = YourModel()

# # Fit the model
# model.fit(X_train_resampled.squeeze(), y_train_resampled)

# # Evaluate the model
# accuracy = model.base_model.score(X_test, y_test['toxic'])
# print("Label-wise accuracy:", accuracy)
