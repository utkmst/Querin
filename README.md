![alt text](https://ibb.co/XZzvGzn)

# Querin
Querin is a Python-based chatbot designed to interact with users, fetch the latest headlines from the BBC website, and retrieve summaries of topics from Wikipedia. It's a helpful tool for quick information retrieval and exploration of current events.

# Features
Conversational Responses: Chat with Querin using a set of predefined patterns to get a natural-sounding response.
Sentiment Analysis: Querin can analyze the sentiment of your input and respond empathetically.
BBC News Headlines: Fetches and summarizes the latest headlines from the BBC main page.
Wikipedia Summaries: Retrieves summaries of requested topics from Wikipedia.
Text Summarization: Uses a transformer-based model to condense lengthy text from articles into concise summaries.

# Requirements 
Python Libraries
To run Querin 2.0, you'll need to install several dependencies. Use the following commands to set up your environment:
pip install nltk requests beautifulsoup4 textblob transformers torch

# Additional Setup Steps
NLTK Data: Download the WordNet data for lemmatization.

import nltk
nltk.download('wordnet')

TextBlob Corpora: Download necessary data for TextBlob to perform sentiment analysis.
python -m textblob.download_corpora

Virtual Environment (Optional): It's recommended to use a virtual environment to manage dependencies:
python -m venv querin_env
source querin_env/bin/activate  # For Linux/Mac
querin_env\Scripts\activate  # For Windows

# Usage
Clone the Repository:

git clone https://github.com/yourusername/Querin.git
cd Querin

Run the Bot: Start Querin 2.0 by running the following command:
python Querin.py

# Interacting with Querin:

Chatting: Type in greetings or questions for casual interaction. For example, "hi", "how are you?", "who created you?", etc.
Sentiment Analysis: Enter any sentence, and Querin will respond based on the detected sentiment.
BBC News Headlines: Type "news" or "latest news" to get the latest BBC headlines, which will be summarized if the content is lengthy.
Wikipedia Summary: Type "what is <topic>" or "who is <person>" to get a Wikipedia summary. Replace <topic> or <person> with the topic of your choice.
Exit the Chat: Type "quit" to end the session.

# Example
Here’s an example interaction with Querin 2.0:

You: hi
Querin 2.0: Hello!

You: news
Querin 2.0: Fetching BBC headlines...
 - Headline 1
 - Headline 2
 - Headline 3

You: what is machine learning
Querin 2.0: Fetching Wikipedia summary for 'machine learning'...
Wikipedia: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.

You: I am feeling great today!
Querin 2.0: I'm glad to hear that!

You: quit
Querin 2.0: Goodbye! Take care.

# Project Structure
Querin.py: Main script containing the chatbot logic, including sentiment analysis, news fetching, and Wikipedia summarization.
README.md: Documentation for the project.

# Future Enhancements
Some potential enhancements for Querin 2.0 include:

Adding support for more news sources beyond BBC.
Expanding Wikipedia search capabilities and accuracy.
Enhancing the chatbot's conversational abilities with additional machine learning models.

# License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


