from yelpapi import YelpAPI
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

api_key = "OOpEeCR5qwIhPANDoLPwFZGeIwKCNz4LBwzhFOjVPTSXZAciFi4vuvrAyOX0kJ1gBH88Spo5547wl6PdB1oix-yrk1Un65DXH1Gv5cj83QZtrgJKwzyFuDrA6-BMZXYx"

yelp_api = YelpAPI(api_key)
search_term = 'sushi'
location_term = 'El Paso, TX'

# List of restaurant names to include
restaurant_names = [
    'el-charlatan-socorro',
    'sunnys-sushi-horizon-city-2',
    'poke-bar-el-paso-10',
    'tugboat-fish-and-chips-el-paso',
    'kees-teriyaki-and-sushi-el-paso'
]

# Number of reviews to analyze for each restaurant
reviews_limit = 3

for restaurant_name in restaurant_names:
    # Initialize a Counter to store word frequencies for each restaurant
    word_freq_counter = Counter()

    # Extract reviews directly from the reviews response for the specific restaurant
    reviews_response = yelp_api.reviews_query(id=restaurant_name, limit=reviews_limit)

    # Analyze the reviews from the specific restaurant
    for review in reviews_response['reviews']:
        # Tokenize the text to words and convert to lowercase
        tokens = nltk.word_tokenize(re.sub(r'[^\w\s]', '', review['text'].lower()))

        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in stop_words]

        # Update word frequencies
        word_freq_counter.update(filtered_tokens)

    # Save the most common words and their frequencies to a file for each restaurant
    filename = f'word_frequencies_{restaurant_name}.txt'
    with open(filename, 'w') as output_file:
        for word, freq in word_freq_counter.most_common():
            output_file.write(f'{word}: {freq}\n')
