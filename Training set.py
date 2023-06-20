import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Tuple
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    
    # Perform stemming using Snowball stemmer
    stemmer = SnowballStemmer('english')
    text = " ".join(stemmer.stem(word) for word in text.split())
    
    return text



class YourModel:
    def __init__(self):
        self.base_model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression())
        ])
    
    def _get_base_model_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        train_x = X.copy()
        train_y = y[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

        # Filter out toxic comments
        toxic_mask = (train_y == 0).all(axis=1)
        train_x = train_x.loc[toxic_mask]
        train_y = train_y.loc[toxic_mask]

        train_x = train_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        
        return train_x, train_y['toxic']  # Returning the 'toxic' column for compatibility

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_x, train_y = self._get_base_model_data(X, y)
        self.base_model.fit(train_x, train_y, **kwargs)

df = pd.read_csv("C:\\Users\\lunac\\OneDrive\\Documents\\train.csv", encoding='latin1')
df['comment_text'] = df['comment_text'].apply(preprocess_text)

X = df['comment_text']
y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling on the training set
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.to_frame(), y_train['toxic'])

model = YourModel()

# Fit the model
model.fit(X_train_resampled.squeeze(), y_train_resampled)

# Evaluate the model
accuracy = model.base_model.score(X_test, y_test['toxic'])
print("Label-wise accuracy:", accuracy)
