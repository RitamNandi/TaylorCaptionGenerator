from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd

# Imports for preprocessing:
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Preprocessing: lowercase all text, remove punctuation and special characters, tokenize, remove stopwords
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load and preprocess the lyrics dataset
lyrics_df = pd.read_csv('taylor_lyrics_songs_albumnames.csv') # read saved dataset
lyrics_df['processed'] = lyrics_df['lyric'].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(lyrics_df['processed'])

def process_image(image_file_path):
    raw_image = Image.open(image_file_path).convert('RGB')

    # conditional image captioning
    text = "A picture of"
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    caption = caption[12:] # remove the "a picture of" from the front of the string caption

    input_vector = tfidf_vectorizer.transform([preprocess_text(caption)])

    similarities = cosine_similarity(input_vector, tfidf_matrix)

    closest_lyric_index = similarities.argmax()
    closest_lyric = lyrics_df.iloc[closest_lyric_index]['lyric']

    formatted_lyric = f"'{lyrics_df.iloc[closest_lyric_index]['lyric']}' from {lyrics_df.iloc[closest_lyric_index]['song_title']} from album: {lyrics_df.iloc[closest_lyric_index]['album']}"

    if lyrics_df.iloc[closest_lyric_index]['album'] != "Unreleased Songs" and lyrics_df.iloc[closest_lyric_index]['song_title'] != "Unreleased Songs [Discography List]":
        return formatted_lyric, lyrics_df.iloc[closest_lyric_index]['album']
    else:
        return closest_lyric, lyrics_df.iloc[closest_lyric_index]['album'] # This will say unreleased songs if it isn't one