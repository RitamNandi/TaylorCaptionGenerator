# Taylor Caption Generator

Upload any image and the web app outputs a Taylor Swift lyric that best captions the photo, includes the song and album if they aren't unreleased. This project is not a ChatGPT wrapper!

**How to run**:
1. `git clone https://github.com/RitamNandi/TaylorCaptionGenerator.git`
2. Install all prereqs with `pip install -r requirements.txt`
3. Run with `flask --app app run`

**Technical Project Overview**:

Uses Salesforce BLIP model to generate a caption describing the uploaded image. 
A CSV is included of Taylor lyrics. Each entry lyric is preprocessed to lowercase, remove punctuation and special characters, remove stopwords, and tokenize. This is vectorized to make a matrix. The generated caption by the BLIP model is also preprocessed this way. The cosine similarity is calculated between the generated caption vector and the lyric matrix. The lyric with the highest similarity to the generated caption is returned. 
Flask used for web app, image is uploaded in main.html, result is outputted in result.html.
