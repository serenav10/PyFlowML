from PyFlow.Core import NodeBase
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyFlow.Packages.PyFlowBase.Pins.AnyPin import AnyPin
from PyFlow.Core.Common import PinOptions
from Qt import QtWidgets, QtCore
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from PySide2.QtWidgets import QLabel, QInputDialog, QDialog, QVBoxLayout, QLineEdit, QPushButton
from PySide2.QtCore import Qt
from PyFlow.Core.Common import *
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib.table import table
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.colors as mcolors

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class RelevantTerms(NodeBase):
    def __init__(self, name):
        super(RelevantTerms, self).__init__(name)

        # Define the input and output pins
        self.dataset = self.createInputPin("dataset", 'AnyPin')
        self.word_cloud = self.createOutputPin("word cloud", 'StringPin')
        self.most_relevant_term = self.createOutputPin("most_relevant_term", 'StringPin')

        self.dataset.enableOptions(PinOptions.AllowAny)

    def compute(self):
        if self.dataset is None or not self.dataset.hasConnections():
            print("No dataset input provided.")
            return

        data = self.dataset.getData()
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data
        else:
            print("Invalid dataset input format.")
            return

        # Verify x_data
        print("x_data contents:")
        print(x_data[:10])
        print("y_data contents:")
        print(y_data[:10])

        # Preprocess the text
        preprocessed_texts = []
        lemmatizer = WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words("english")

        for text in x_data:
            # Convert text to lowercase
            text = text.lower()

            # Replace symbols, numbers, and punctuation with whitespaces
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)

            # Tokenize the text using WhitespaceTokenizer
            tokenizer = WhitespaceTokenizer()
            tokens = tokenizer.tokenize(text)

            # Remove stop words
            tokens = [token for token in tokens if token not in stopwords]

            # Lemmatize the tokens
            lemmas = [lemmatizer.lemmatize(token) for token in tokens]

            # Join lemmas back into text
            preprocessed_text = " ".join(lemmas)
            preprocessed_texts.append(preprocessed_text)

        # Print 5 examples of preprocessed texts
        print("Examples of preprocessed texts:")
        for i in range(5):
            print(preprocessed_texts[i])

        # Compute TF-IDF weights
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate the mean TF-IDF weight for each feature across all documents
        tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)

        # Create a dictionary of word frequencies
        word_frequencies = {feature_names[i]: tfidf_means[i] for i in range(len(feature_names))}

        # Generate word cloud based on TF-IDF weighted frequency
        colormap = 'magma'
        wordcloud = WordCloud(width=800, height=400, stopwords=STOPWORDS, colormap=colormap).generate_from_frequencies(word_frequencies)

        # Save word cloud as PNG file
        wordcloud_path = "word_cloud.png"
        wordcloud.to_file(wordcloud_path)

        # Set the path of the word cloud PNG file as data for the output pin
        self.word_cloud.setData(wordcloud_path)

        ################# Dataset examples:TF-IDF close to 1 (relevant term)

        def convert_strings_to_image(strings):
            print("Selected Lemmas:", strings)  # Add this line for debugging purposes

            if not strings:
                print("No strings to display.")
                return None

            # Create a figure with subplots for each string
            fig, axs = plt.subplots(len(strings), 1, figsize=(10, 6))

            # Iterate over the strings and plot them as text in each subplot
            for i, string in enumerate(strings):
                axs[i].axis('off')
                axs[i].text(0.01, 0.01, string, fontsize=18, ha='left', va='center', wrap=True, fontname='DejaVu Sans')

            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=2)

            # Save the figure as a PNG file
            image_path2 = 'Dataset_examples.png'
            plt.savefig(image_path2)
            plt.close()

            return image_path2

        # Sort the word frequencies by TF-IDF value in descending order
        sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

        # Select the top N lemmas with the highest TF-IDF values
        top_n = 10
        selected_lemmas = [word for word, tfidf in sorted_word_frequencies[:top_n]]

        # Generate an image with the selected lemmas
        image_path = convert_strings_to_image(selected_lemmas)
        self.most_relevant_term.setData(image_path)

        #################

    @staticmethod
    def category():
        return '2_Data_Visualization'
