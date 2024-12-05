import string
import warnings
import nltk
import requests
import feedparser
import time
import sys
import re
import random
import json
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.chat.util import Chat, reflections
from textblob import TextBlob
from transformers import pipeline
from colorama import Fore, Style
from urllib.parse import quote
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from translate import Translator


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token
model = GPT2LMHeadModel.from_pretrained("gpt2")

config_file_path = '/home/ashuredd/Querin/Querin/config.json'

config = {}
try:
    with open(config_file_path, 'r') as file:
        if file.readable() and file.seek(0, 2) != 0:  # Check if file is not empty
            file.seek(0)  # Reset file pointer to the beginning
            config = json.load(file)
        else:
            raise ValueError("Configuration file is empty")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except FileNotFoundError:
    print("Configuration file not found")
except ValueError as e:
    print(e)

api_key = config.get("WEATHER_API_KEY")
if not api_key:
    print("API key not found in config.json.")
    exit(1)  # Exit the script if the API key is not found

with open('localization.json', 'r', encoding='utf-8') as file:
    translations = json.load(file)

def get_translations(language):
    return translations.get(language, translations['en'])

def translate_text(text: str, target_language: str) -> str:
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

def detect_language(text: str) -> str:
    return "en"

def generate_gpt2_response(user_input, max_length=100):
    # Tokenize the input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    
    # Generate response with attention mask and pad_token_id
    output = model.generate(
        input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # Set attention mask
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        num_beams=1,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id, # Set the pad token
        early_stopping=True  
    )
    
    # Decode the generated text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if response.lower().startswith(user_input.lower()):
        response = response[len(user_input):].strip()

    return response

language_prompt = """
Please select a language:
en = English
es = Spanish
tr = Turkish
de = German
kr = Korean
jp = Japanese
ch = Chinese
ru = Russian
gr = Greek
id = Indonesian
Enter the code corresponding to your language choice: """

try:
    language_code = input(language_prompt).strip()
except (EOFError, KeyboardInterrupt):
    print("\nInput interrupted. Exiting the program.")
    exit(1)

if language_code not in translations:
    print("Selected language is not supported. Defaulting to English.")
    language_code = 'en'

lang_data = translations.get(language_code, translations['en'])

warnings.filterwarnings('ignore')

# Load summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define patterns for the chatbot
# Define patterns for the chatbot
patterns = [  
    [r"my name is (.*)", ["Hello %1, nice to meet you!"]],
    [r'Who am I?', ['You haven’t told me your name yet!', 'I don’t know your name yet. Can you tell me?']],
    [r'My name is (.*)', ['Nice to meet you, %s! What would you like to know?', 'Hello %s, what can I help you with today?']],
    [r'Goodbye', ['Goodbye! It was nice talking to you.', 'Take care!']],
    [r"can i ask you a question?", ["Of course!", "Sure!", "Yes, you can ask me whatever you want!"]],
    [r"thank you|thanks|thx|appreciate it", ["You're welcome!"]],
    [r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]],
    [r"how are you(.*)", ["I'm Querin-GPT1.0, but I'm doing great! How about you?"]],
    [r"what is your name?", ["I am a chatbot designed by Querin Corporation. What's yours?"]],
    [r"quit", ["Bye! It was nice talking to you. Have a great day!"]],
    [r"what (.*) you like to do?", ["I enjoy chatting with humans like you!", "I love learning new words."]],
    [r"do you like (.*)?", ["I don't have preferences, but I think %1 sounds interesting!"]],
    [r"who (.*) created you?", ["I was created by programmers from Querin Corporation", "I was made by tech enthusiasts!"]],
    [r"(.*) movie recommendations?", ["I recommend 'Inception' for movies!"]], 
    [r"(.*) book recommendations?", ["Try '1984' by George Orwell if you like books."]], 
    [r"(.*) song recommendations?", ["Listen APT. by Rose & Bruno Mars"]],  
]

#Memory to store contextual information
session_memory = {}

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

patterns = translations[language_code].get('patterns', [])
reflections.update(translations[language_code].get('reflections', {}))
sentiments = translations[language_code].get('sentiments', {})

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
    analysis = sentiment_analyzer(user_input)[0]
    if analysis['label'] == '5 stars':
        return random.choice(sentiments.get('positive', ["I'm glad to hear that!"]))
    elif analysis['label'] == '1 star':
        return random.choice(sentiments.get('negative', ["I'm sorry to hear that."]))
    else:
        return random.choice(sentiments.get('neutral', ["I see."]))
'''
# Fetch text from a website based on the user's query
def fetch_from_source(query):
    if "news" in query or "latest" in query:
        return fetch_news_article()
    else:
        return "I'm not sure which source to use. Could you clarify your question?"
'''

# Fetch latest news article
def fetch_news_article():
    feed_url = lang_data.get('news_feed_url', 'http://feeds.bbci.co.uk/news/rss.xml')
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
def fetch_wikipedia_summary(query, lang='en'):
    search_query = query.replace("explain", "").replace("what is", "").replace("who is","").replace("where is","").strip()
    search_query = quote(search_query)
    # Use the Wikipedia API to search for the page
    search_url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json"
    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_results = search_response.json()     
        if search_results['query']['search']:
            page_title = search_results['query']['search'][0]['title']       
            page_url = f"https://{lang}.wikipedia.org/wiki/{quote(page_title)}"
            page_response = requests.get(page_url)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = soup.find_all('p')

            # Initialize a list to store valid paragraphs' text
            valid_paragraphs = []

            # Loop through all paragraphs (or the first ones, as needed)
            for para in paragraphs:
                # Extract the text and strip whitespace
                para_text = para.get_text(separator=" ",strip=True)

                # Check if the paragraph has meaningful content and doesn't have 'mw-empty-elt' class
                if para_text and len(para_text) > 50 and 'mw-empty-elt' not in para.get('class', []):
                    valid_paragraphs.append(para_text)

                # Stop once we've collected two valid paragraphs
                if sum(len(p) for p in valid_paragraphs) >= 300:
                    break
            infobox = soup.find('table', class_='infobox')
            birth_date = None
            if infobox:
                birth_date_row = infobox.find('span', class_='bday')
                if birth_date_row:
                    birth_date = birth_date_row.get_text(strip=True)

            # Join the valid paragraphs into a single string
            text_content = ' '.join(valid_paragraphs)
            formatted_content = format_paragraphs(text_content, max_length=80)
            if birth_date:
                session_memory['birth_date'] = birth_date 

            # If text_content is empty, print a message or handle it accordingly
            if formatted_content:
                return formatted_content
            else:
                print("No meaningful content found in the selected paragraphs.")
                      
        else:
            return "I couldn't find relevant information on Wikipedia for that query."

    except requests.RequestException:
        return "There was an issue reaching Wikipedia for that query."

import requests

def get_weather(city: str, api_key: str, target_lang: str) -> str:
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": city,
        "aqi": "yes"  # Set "yes" if you want air quality data
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        if 'location' in data and 'current' in data:
            location = data['location']['name']
            current = data['current']
            condition = current['condition']['text']
            temp_c = current['temp_c']
            feelslike_c = current['feelslike_c']
            humidity = current['humidity']
            wind_kph = current['wind_kph']
            
            weather_info = (f"Weather in {location}: {condition}, "
                            f"Temperature: {temp_c}°C, Feels like: {feelslike_c}°C, "
                            f"Humidity: {humidity}%, Wind: {wind_kph} kph")
            translator = Translator(to_lang=target_lang)
            translated_weather_info = translator.translate(weather_info)
            return translated_weather_info
        else:
            return "Unexpected response format from the weather service."
    except requests.RequestException as e:
        return f"There was an issue reaching the weather service: {e}"
    except KeyError as e:
        return f"Unexpected response format: {e}"


def chatbot_response(user_input):
    global session_memory
    user_language = language_code
    question_prefixes = lang_data['question_prefixes']
    weather_prefixes = lang_data['weather_prefixes']
    user_input_lower = user_input.lower()

    for prefix in weather_prefixes:
        if user_input_lower.startswith(prefix) or user_input_lower.endswith(prefix) or prefix in user_input_lower:
            city = user_input_lower.replace(prefix, "").strip()
            api_key = config.get("WEATHER_API_KEY")
            if not api_key:
                return "API key not found in the configuration"
            return get_weather(city, api_key, target_lang=language_code)

    
    for prefix in lang_data['news_prefixes']:
        if user_input_lower.startswith(prefix) or user_input_lower.endswith(prefix) or prefix in user_input_lower:
            return fetch_news_article()
    
    for prefix in lang_data['question_prefixes']:
        if user_input_lower.startswith(prefix) or user_input_lower.endswith(prefix) or prefix in user_input_lower:
            return fetch_wikipedia_summary(user_input, lang=language_code)
            
    # Check if the user says "who am I" and return the stored name if available
    if "who am i" in user_input_lower:
        if 'user_name' in session_memory:
            return f"Your name is {session_memory['user_name']}."
        else:
            return "You haven't told me your name yet."

    # Handle name introduction (update user_name in session_memory)
    if "my name is" in user_input_lower:
        name = user_input.replace("my name is", "").strip()
        session_memory['user_name'] = name  # Store the user's name in session memory
        response = f"Hello {name}, nice to meet you!"
        return translate_text(response, user_language)
    
    if "when was" in user_input_lower and session_memory.get("birth_date"):
        response = f"{session_memory['last_topic']} was born on {session_memory['birth_date']}."
        return translate_text(response, user_language)
    
    
    
    # Handle query check with session memory update
    if is_query(user_input):
        for prefix in question_prefixes:
            if user_input_lower.startswith(prefix):
                session_memory['last_topic'] = user_input[len(prefix):].strip()
                break
        response = resource_response_enhanced(user_input)
        return translate_text(response, user_language)
    
    # Check for pattern response
    pattern_response = get_pattern_response(user_input)
    if pattern_response:
        return pattern_response
    
    # Check for sentiment response
    sentiment_response = respond_to_sentiment(user_input)
    if sentiment_response != "Thanks for sharing that.":  # Default text means no sentiment
        return sentiment_response

    # Generate a GPT-2 response if no patterns or specific responses are matched
    return generate_gpt2_response(user_input)

# Generate a resource-based response
'''
def resource_response_enhanced(user_response):
    querin_response = fetch_from_source(user_response)
    follow_up = "Would you like more information or details on a related topic?"
    if session_memory.get('last_topic'):
        follow_up += f" Last time, you asked about '{session_memory['last_topic']}'."
    return querin_response + "\n" + follow_up
'''
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

def get_pattern_response(user_input):
    for pattern, responses in patterns:
        match = re.match(pattern, user_input.lower())
        if match:
            response = random.choice(responses)
            if match.groups():
                try:
                    response = response % tuple(match.groups())
                except (TypeError, ValueError):
                    pass
            return response
    return None

def slow_typing(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Main conversation loop
welcome = translations[language_code]['welcome']
print(welcome)

while True:
    user_input = input(f"{Style.BRIGHT}{Fore.GREEN}You: {Style.RESET_ALL}")

    # Exit condition    
    if user_input.lower() in ["quit", "exit", "bye"]:
        print(f"{Style.BRIGHT}{Fore.RED}Querin-GPT1.0: {Style.RESET_ALL}", end="")
        goodbye_message = translations[language_code]['goodbye']
        slow_typing(goodbye_message, delay=0.03)
        break

    # Check for predefined patterns
    response = chatbot_response(user_input)

    if response:
        print(f"{Style.BRIGHT}{Fore.RED}Querin-GPT1.0:{Style.RESET_ALL}", end=" ")  
    
        # Apply slow typing only to the response
        slow_typing(response, delay=0.03)

    elif is_query(user_input):
        print(f"{Style.BRIGHT}{Fore.YELLOW}Querin-GPT1.0:{Style.RESET_ALL}", end=" ")
        slow_typing((resource_response_enhanced(user_input)), delay=0.01)

    else:
        response = respond_to_sentiment(user_input)
        print(f"{Style.BRIGHT}{Fore.RED}Querin-GPT1.0:{Style.RESET_ALL}", end=" ")
        slow_typing(response, delay=0.03)
