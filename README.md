![alt text](https://camo.githubusercontent.com/f8b934b838ff8ba845ef97135975f8a99c1e19f3d9450893c54a2868d03b9802/68747470733a2f2f692e6962622e636f2f43356e54536e6a2f51756572696e2d6c6f676f2d66697273742d72656d6f766562672e706e67)

# Querin🦋
Querin is a Python-based chatbot that integrates the power of OpenAI's GPT-2 model for natural language generation, along with various features such as sentiment analysis, Wikipedia summarization, and BBC news retrieval. It is designed to interact with users conversationally, providing useful information and answering questions based on predefined patterns and external sources like news articles and Wikipedia.

# Features
Conversational Responses: Chat with Querin-GPT1.0 using predefined patterns for a smooth conversational experience.

Sentiment Analysis: Analyzes the sentiment of user input and responds empathetically.

GPT-2 Powered Responses: Generates responses dynamically based on user input using the GPT-2 model.

BBC News Headlines: Fetches and summarizes the latest headlines from BBC news.

Wikipedia Summaries: Retrieves detailed summaries of requested topics from Wikipedia.

Text Summarization: Uses transformer models to condense lengthy articles or text into concise summaries.

# Requirements 
Python Libraries
To run Querin-GPT1.0, you'll need to install several dependencies. Use the following commands to set up your environment:

pip install nltk requests beautifulsoup4 textblob transformers torch feedparser colorama

Additionally, ensure that the necessary NLTK data and TextBlob corpora are downloaded:

import nltk
nltk.download('wordnet')

For TextBlob:
python -m textblob.download_corpora

Optional: Setting Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

python -m venv querin_env

source querin_env/bin/activate  # For Linux/Mac

querin_env\Scripts\activate  # For Windows


# Usage
Clone the Repository:

git clone https://github.com/ashuredd/Querin.git

cd Querin

Run the Bot: Start Querin-GPT1.0 by running the following command:

python Querin-GPT1.0.py

Once the bot is running, you can interact with it directly through the console.

# Example Interactions

Here’s an example of how you can interact with Querin-GPT1.0:

You: hi

Querin-GPT1.0: Hello!

You: news

Querin-GPT1.0: Fetching BBC headlines...

Headline 1 Headline 2 Headline 3

You: what is machine learning

Querin-GPT1.0: Fetching Wikipedia summary for 'machine learning'... Wikipedia: "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."

You: I am feeling great today!

Querin-GPT1.0: I'm glad to hear that!

You: quit

Querin-GPT1.0: Goodbye! Take care.

# Project Structure
Querin-GPT1.0.py: Main script containing the chatbot logic, including sentiment analysis, news fetching, and Wikipedia summarization.

Querin2.0.py: The original version of Querin, without GPT-model.

README.md: Documentation for the project.

# Key Components

GPT-2 Response Generation:
The chatbot generates responses dynamically using the GPT-2 model. The model is fine-tuned to produce conversational responses and handle questions related to space travel, general information, and more.

Predefined Patterns for Basic Responses:
Uses regular expressions to match user input to predefined patterns for personalized, quick responses (e.g., greeting, asking for the bot's name, etc.).

Sentiment Analysis:
TextBlob is used to analyze the sentiment of the user’s input and provide empathetic or positive responses based on the input's emotional tone.

Fetching News and Wikipedia Information:
The chatbot fetches the latest news from the BBC using RSS feeds and summarizes articles for users. Additionally, it retrieves and summarizes Wikipedia pages based on user queries.

# Future Enhancements
Some potential enhancements for future versions of Querin-GPT1.0 include:

Adding support for more news sources beyond BBC.

Expanding Wikipedia search capabilities for more accurate and detailed summaries.

Enhancing the conversational abilities of the chatbot by integrating additional machine learning models.

Allowing for deeper integrations, such as retrieving more structured data or improving memory of past interactions.

# License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


