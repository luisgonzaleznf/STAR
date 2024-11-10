# grok_api.py

import os
from dotenv import load_dotenv
import openai
import json

# Load environment variables and initialize API key and base URL
def initialize_grok_api():
    """
    Initialize environment variables and set up the OpenAI API key and base URL for GROK.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Access the OpenAI API key from environment
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set 'XAI_API_KEY' in your .env file.")

    # Set OpenAI API key and base URL
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    client.base_url = "https://api.x.ai/v1"

    return client

# Function to get completion from the GROK model
def get_grok_completion(client, message):
    """
    Sends a chat completion request to the GROK model with the provided message.
    
    Parameters:
    client (openai.OpenAI): Initialized OpenAI client with base URL and API key.
    message (str): The user's input message for the GROK API.
    
    Returns:
    dict: The JSON response from the GROK model.
    """
    model = "grok-beta"
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}]
        )
        return completion.model_dump_json(indent=2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def extract_grok_content(response_json):
    """
    Extracts and returns the message content from the GROK completion response JSON.
    
    Parameters:
    response_json (str): JSON response string from the GROK API.
    
    Returns:
    str: The content of the message from the GROK response.
    """
    try:
        data = json.loads(response_json)
        message_content = data["choices"][0]["message"]["content"]
        return message_content
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        return None

def generate_summary(client, transcription, formatted_duration, num_speakers, language):
    """
    Generates a summary of the conversation transcription based on the metadata and transcription provided.
    
    Parameters:
    client (openai.OpenAI): Initialized OpenAI client with base URL and API key.
    transcription (str): The text transcription of the audio.
    formatted_duration (str): The length of the audio in hours and minutes (e.g., '1h 20m').
    num_speakers (int): The number of distinct speakers.
    language (str): The language of the conversation.
    
    Returns:
    dict: The JSON response from the GROK model containing the summary.
    """
    # Define the prompt structure
    prompt_intro = """Summarize this conversation or talk transcription, focusing on the main ideas, topics discussed, and any conclusions or next steps presented. Provide additional context about the conversation using the metadata of the audio file:

- Include the length of the audio, formatted in hours and minutes (e.g., 1h 20m).
- Specify the number of distinct speakers involved.
- Indicate the language in which the conversation took place.

Present the summary in a concise and easy-to-read format. The summary must be in the same language as the conversation transcription (the one specified in the metadata).

"""
    # Construct the metadata prompt
    prompt_metadata = f"""Audio Metadata:

Duration: {formatted_duration}
Number of speakers: {num_speakers}
Language: {language}

Transcription:

"""
    # Combine the full message for the GROK model
    message = prompt_intro + prompt_metadata + transcription

    # Call the completion function and parse the result
    response_json = get_grok_completion(client, message)
    if response_json:
        return extract_grok_content(response_json)
    return None