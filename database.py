import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Define a function to clean the text
    def clean(text):
        text = re.sub('[^A-Za-z]+', ' ', text)
        return text

    # Cleaning the text
    cleaned_text = clean(text)

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

    # Apply tokenization, stop word removal, POS tagging
    pos_tagged = token_stop_pos(cleaned_text)

    # Function for lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatize(pos_data):
        lemma_rew = " ".join([wordnet_lemmatizer.lemmatize(word, pos=pos) if pos else word for word, pos in pos_data])
        return lemma_rew

    # Apply lemmatization
    lemmatized_text = lemmatize(pos_tagged)

    return lemmatized_text

def insert_message(conn, cursor, user_id, message, censored_message):
    insert_query = "INSERT INTO messages (user_id, message, processed, censored_message) VALUES (%s, %s, %s, %s)"
    cursor.execute(insert_query, (user_id, message, True, censored_message))
    conn.commit()


def close_connection(conn, cursor):
    cursor.close()
    conn.close()
