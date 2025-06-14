![alt text](querin.gif)

# Querin🦋
Querin is a multilingual, Python-based chatbot leveraging OpenAI's GPT-2 model to provide intelligent, conversational interactions. With features like sentiment analysis, real-time weather updates, news retrieval, Wikipedia summaries, and language translation, Querin is a versatile assistant designed to cater to diverse user needs.

# Features
**🌐 Multilingual Support**
Users can interact in multiple languages, including English, Spanish, Turkish, German, Korean, Japanese, Chinese, Russian, Greek, and Indonesian.

Custom translations for chatbot responses based on user-selected language.

**🤖 GPT-2 Powered Conversations**

Querin dynamically generates conversational responses using the GPT-2 model.

Handles free-form user input with context-aware and human-like replies.

**📰 News Fetching**

Retrieves the latest news articles from RSS feeds (default: BBC News).

Summarizes news headlines for quick access.

**📖 Wikipedia Summaries**

Fetches and summarizes relevant Wikipedia articles based on user queries.

Supports localized summaries by fetching content in the user’s selected language.

**🌦️ Real-Time Weather Updates**

Provides detailed weather updates for any city using the WeatherAPI.

Data includes current conditions, temperature, humidity, wind speed, and more.

Translates weather information into the user’s chosen language.

**🧠 Sentiment Analysis**

Analyzes user sentiment using a BERT-based model.

Empathizes with user emotions and provides supportive or neutral responses accordingly.

**🔄 Dynamic Translations**

Integrated translation support for translating responses into the user’s preferred language.

**💬 Pattern-Based Responses**

Recognizes predefined patterns to provide quick and direct responses (e.g., greetings, farewells, recommendations).

**📜 Summarization**

Uses transformer models for text summarization, condensing lengthy articles into concise summaries.

# Requirements
## Dependencies

Install the following Python libraries to run Querin:

```bash
pip install nltk requests beautifulsoup4 textblob transformers torch feedparser colorama translate
```
Ensure required corpora are downloaded:
```python
import nltk
nltk.download('wordnet')
```
## For **TextBlob** corpora:
```bash
python -m textblob.download_corpora
```
## API Key
Querin requires an API key for weather data. Add your WeatherAPI key to config.json in the following format:
```json
{
  "WEATHER_API_KEY": "your_api_key_here"
}
```
## File Requirements

config.json: Stores API keys and configurations.

localization.json: Contains translations for supported languages.

# Usage
## Setup
1. Clone the Repository:
```bash
git clone https://github.com/utkmst/Querin.git
cd Querin
```
2. Run the Bot: Start Querin-GPT1.0 by running the following command:
```bash
python Querin-GPT1.0.py
```
(if you want the original version, without GPT model):
```bash
python Querin2.0.py
```


## Interaction

Select your language from the prompt.

Engage with Querin by typing inputs like:

**"What's the weather in London?"**
**"Fetch me the latest news."**
**"Tell me about machine learning."**

Exit the chatbot by typing:
```plaintext
quit
```
# Example Interactions

Here’s an example of how you can interact with Querin-GPT1.0:
```plaintext
You: hi
Querin-GPT1.0: Hello! How can I assist you today?

You: What's the weather in Tokyo?
Querin-GPT1.0: Weather in Tokyo: Sunny, Temperature: 20°C, Feels like: 18°C, Humidity: 50%, Wind: 10 kph.

You: news
Querin-GPT1.0: Fetching BBC headlines...
1. Headline: "Breaking News Title"
2. Summary: "A short summary of the article."

You: Explain artificial intelligence
Querin-GPT1.0: Fetching Wikipedia summary for 'artificial intelligence'...
"Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems."
```

# Project Structure
**Querin-GPT1.0.py**: Main script containing the chatbot logic, language translation, and external API integrations.

**config.json**: Configuration file for storing API keys.

**localization.json**: File containing translations for multilingual support.

**Querin2.0.py**: The original version of Querin, without GPT-model.

**README.md**: Documentation for the project.

**backlog**: To do list.

**LICENSE**: MIT License

# Future Enhancements

Voice Interaction: Adding speech recognition and text-to-speech capabilities.

Expanded News Sources: Support for additional RSS feeds.

Advanced Context Memory: Enhancing chatbot memory for long-term user interactions.

Custom Model Fine-Tuning: Fine-tuning GPT-2 on domain-specific datasets for specialized use cases.

# License
This project is licensed under the MIT License - see the **[LICENSE](./LICENSE)** file for details.


