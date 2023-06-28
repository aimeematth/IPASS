import joblib
import database
import training_set
import psycopg2
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Load the trained models from disk
label_models = {}
for label in ['toxic']:
    model = joblib.load(f"{label}_model.pkl")  # Specify the correct path of the trained models
    label_models[label] = model

# Connect to the PostgreSQL database
conn = psycopg2.connect(host='localhost', dbname='chat_comments', user='postgres', password='134340')
cursor = conn.cursor()

# Preprocess text function
def preprocess_text(text):
    """
    Preprocesses the text by cleaning, tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    def clean(text):
        text = re.sub('[^A-Za-z]+', ' ', text)
        return text

    cleaned_text = clean(text)

    pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

    def token_stop_pos(text):
        tags = pos_tag(word_tokenize(text))
        newlist = []
        for word, tag in tags:
            if word.lower() not in set(stopwords.words('english')):
                newlist.append(tuple([word, pos_dict.get(tag[0])]))
        return newlist

    pos_tagged = token_stop_pos(cleaned_text)

    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatize(pos_data):
        lemma_rew = " ".join([wordnet_lemmatizer.lemmatize(word, pos=pos) if pos else word for word, pos in pos_data])
        return lemma_rew

    lemmatized_text = lemmatize(pos_tagged)

    return lemmatized_text

# Login process
username = input("Enter your username: ") #example : StarGazer92
email = input("Enter your email: ") #example : stargazer92@example.com
user_id = input("Enter your user ID: ") #example : 4352

# Check if the user exists in the database
select_user_query = "SELECT id FROM users WHERE id = %s AND username = %s AND email = %s"
cursor.execute(select_user_query, (user_id, username, email))
user_exists = cursor.fetchone()

if not user_exists:
    print("User not found. Please provide valid credentials.")
    exit()

print("Login successful.")

# Chat loop
while True:
    # Get user input
    user_input = input("Enter your message (or 'exit' to quit): ")

    # Check if the user wants to exit
    if user_input.lower() == "exit":
        break

    preprocessed_message = preprocess_text(user_input)

    # Use the loaded models to predict the toxicity labels for the preprocessed message
    predictions = {}
    for label, model in label_models.items():
        prediction = model.predict([preprocessed_message])
        predictions[label] = prediction[0]

    # Determine if the message is toxic
    is_toxic = max(predictions.values()) == 1

    # Perform censorship or any other action based on the predicted labels
    censored_message = user_input

    #   Tokenize the message into individual words
    words = word_tokenize(censored_message)

    for label, prediction in predictions.items():
        print(f"Label: {label}, Prediction: {prediction}")
        if prediction == 1:
            # Perform censorship or any other action
            censored_word = label.replace('_', ' ')
            for i in range(len(words)):
                # Check if the word matches the toxic label
                if words[i].lower() == censored_word.lower():
                    # Censor the word by replacing it with asterisks
                    words[i] = '*' * len(words[i])

    # Reconstruct the censored message
    censored_message = ' '.join(words)

    # Print the censored message
    print("Censored message:", censored_message)
    # Insert the user's message into the database
    database.insert_message(conn, cursor, user_id, user_input, censored_message, is_toxic)

# Close the database connection
database.close_connection(conn, cursor)
