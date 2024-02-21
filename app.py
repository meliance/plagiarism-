# app.py
import os
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load data set
dataset_path = 'dataset'
documents = []
file_names = []

for filename in os.listdir(dataset_path):
    if filename.endswith('.txt'):
        with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8', errors='replace') as file:
            try:
                content = file.read()
                documents.append(content)
                file_names.append(filename)
            except UnicodeDecodeError as e:
                print(f"Error decoding file {filename}: {e}")

# Create a vectorizer and transform the documents into a document-term matrix
vectorizer = CountVectorizer()
document_matrix = vectorizer.fit_transform(documents)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            user_content = uploaded_file.read().decode('utf-8', errors='replace')

            # Transform the user's content into a document-term matrix
            user_vector = vectorizer.transform([user_content])

            # Calculate cosine similarity between the user's content and the dataset
            similarities = cosine_similarity(user_vector, document_matrix)

            # Calculate percentage similarity
            percentage_similarity = np.max(similarities) * 100

            # Get the file name with the highest similarity
            most_similar_file = file_names[np.argmax(similarities)]

            return render_template('index.html', similarity=percentage_similarity, most_similar_file=most_similar_file)

    return render_template('index.html', similarity=None, most_similar_file=None)


if __name__ == '__main__':
    app.run(debug=True)
