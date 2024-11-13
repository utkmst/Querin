import string
import warnings
import nltk
import requests
import feedparser
import time
import sys
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.chat.util import Chat, reflections
from textblob import TextBlob
from transformers import pipeline
from colorama import Fore, Style
from urllib.parse import quote

warnings.filterwarnings('ignore')

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Define patterns for the chatbot
patterns = [
    [r"my name is (.*)", ["Hello %1, nice to meet you!"]],
    [r"can i ask you a question?", ["Of course!", "Sure!", "Yes, you can ask me whatever you want!"]],
    [r"thank you|thanks|thx|appreciate it", ["You're welcome!"]],
    [r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]],
    [r"how are you(.*)", ["I'm Querin 2.0, but I'm doing great! How about you?"]],
    [r"what is your name?", ["I am a chatbot designed by Querin Corporation. What's yours?"]],
    [r"quit", ["Bye! It was nice talking to you. Have a great day!"]],
    [r"what (.*) you like to do?", ["I enjoy chatting with humans like you!", "I love learning new words."]],
    [r"do you like (.*)?", ["I don't have preferences, but I think %1 sounds interesting!"]],
    [r"who (.*) created you?",
     ["I was created by programmers from Querin Corporation", "I was made by tech enthusiasts!"]],
    [r"(.*) movie recommendations?",
     ["I recommend 'Inception' for movies!"]], 
    [r"(.*) book recommendations?", ["Try '1984' by George Orwell if you like books."]], 
    [r"(.*) song recommendations?", ["Listen APT. by Rose & Bruno Mars"]]
]

# Reflections for natural language adjustments
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}
def format_paragraphs(text, max_length=80):
    """Format the text to ensure it wraps at a specified maximum length."""
    words = text.split()
    formatted_text = ""
    current_line = ""

    for word in words:
        # Check if adding the next word would exceed the max length
        if len(current_line) + len(word) + 1 > max_length:
            formatted_text += current_line + "\n"  # Add the current line and a newline
            current_line = word  # Start a new line with the current word
        else:
            if current_line:  # If current_line is not empty, add a space before the word
                current_line += " "
            current_line += word

    # Add any remaining text in current_line
    if current_line:
        formatted_text += current_line

    return formatted_text

# Sentiment-based response function
def respond_to_sentiment(user_input):
    analysis = TextBlob(user_input)
    if analysis.sentiment.polarity > 0:
        return "I'm glad to hear that!"
    elif analysis.sentiment.polarity < 0:
        return "I'm sorry to hear that. Hope things get better!"
    else:
        return "Thanks for sharing that."

# Fetch text from a website based on the user's query
def fetch_from_source(query):
    if "news" in query or "latest" in query:
        return fetch_news_article()
    elif "explain" in query or "what is" in query or "who is" in query or "where is" in query:
        return fetch_wikipedia_summary(query)
    else:
        return "I'm not sure which source to use. Could you clarify your question?"

# Fetch latest news article
def fetch_news_article():
    feed_url = 'http://feeds.bbci.co.uk/news/rss.xml'
    feed = feedparser.parse(feed_url)
    response = requests.get(feed_url)
    response.raise_for_status()
    articles = []

    for entry in feed.entries[:5]:
        title = f"{Style.BRIGHT}{Fore.CYAN}{entry.title}{Style.RESET_ALL}"
        summary = entry.summary if 'summary' in entry else "No summary available."

        article_text = f"{title}\n{summary}"
        articles.append(format_paragraphs(article_text, max_length=80))
    return "\n\n".join(articles) if articles else "No news available at the moment."

# Fetch a Wikipedia summary based on the user's query
def fetch_wikipedia_summary(query):
    search_query = query.replace("explain", "").replace("what is", "").replace("who is","").replace("where is","").strip()
    search_query = quote(search_query)
    # Use the Wikipedia API to search for the page
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json"
    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_results = search_response.json()
        if search_results['query']['search']:
            page_title = search_results['query']['search'][0]['title']
            page_url = f"https://en.wikipedia.org/wiki/{quote(page_title)}"
            page_response = requests.get(page_url)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = ' '.join([para.get_text() for para in paragraphs[:2]])
            formatted_content = format_paragraphs(text_content, max_length=80)
            return formatted_content
        else:
            return "I couldn't find relevant information on Wikipedia for that query."

    except requests.RequestException:
        return "I couldn't find relevant information on Wikipedia for that query."

# Generate a resource-based response
def resource_response_enhanced(user_response):
    querin_response = fetch_from_source(user_response)
    follow_up = "Would you like more information or details on a related topic?"
    return querin_response + "\n" + follow_up

# Tokenization and normalization helpers
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Check if input is a query for information
def is_query(user_input):
    query_keywords = ["information", "tell me about", "news", "what is", "who is", "explain", "where is"]
    return any(keyword in user_input.lower() for keyword in query_keywords)

# Initialize Chat object
chatbot = Chat(patterns, reflections)

def slow_typing(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Main conversation loop
print("Welcome to Querin 2.0! Type 'quit' to exit.")

while True:
    user_input = input(f"{Style.BRIGHT}{Fore.GREEN}You: {Style.RESET_ALL}")

    # Exit condition    
    if user_input.lower() in ["quit", "exit", "bye"]:
        print(f"{Style.BRIGHT}{Fore.RED}Querin 2.0: {Style.RESET_ALL}", end="")
        slow_typing("Goodbye! Take care. See you later, alligator!", delay=0.03)
        break

    # Check for predefined patterns
    pattern_response = chatbot.respond(user_input)

    if pattern_response:
        print(f"{Style.BRIGHT}{Fore.RED}Querin 2.0:{Style.RESET_ALL}", end=" ")  
    
        # Apply slow typing only to the response
        slow_typing(pattern_response, delay=0.03)

    elif is_query(user_input):
        print(f"{Style.BRIGHT}{Fore.YELLOW}Querin 2.0:{Style.RESET_ALL}", end=" ")
        slow_typing((resource_response_enhanced(user_input)), delay=0.01)

    else:
        print(f"{Style.BRIGHT}{Fore.RED}Querin 2.0:{Style.RESET_ALL}", end=" ")
        slow_typing((respond_to_sentiment(user_input)), delay=0.03)

