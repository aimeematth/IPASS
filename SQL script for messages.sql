CREATE TABLE users(
    id SERIAL PRIMARY KEY,
    username text,
    email text
);




CREATE TABLE messages(
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    message text,
    processed BOOLEAN,
    censored_message text
);
