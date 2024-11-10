import os

### GENERATE FOLDER STRUCTURE ###

# List of folders to create
folders = ["audio", "parameters", "chunks", "scripts", "transcription", "vectors", "video"]

# Create each folder in the current directory
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Folder structure created successfully.")





### MOVE AUDIO FILE (OR GENERATE AUDIO OUT OF VIDEO IF NECESSARY) ###

import os
import shutil
import sys  # For aborting the process

# Supported file extensions
audio_extensions = ('.mp3', '.wav')
video_extensions = ('.mp4', '.mkv', '.avi', '.mov')  # Add more as needed

# Paths
source_dir = '.'  # Current directory
audio_dir = 'audio'
parameters_dir = 'parameters'

# Check if at least one audio and one parameters file exist
def check_required_files():
    has_audio = any(file.lower().endswith(audio_extensions) for file in os.listdir(source_dir))
    has_parameters = any(file.lower().endswith(".txt") for file in os.listdir(source_dir))
    
    if not has_audio:
        print("Error: Audio file is missing. Make sure to include either an mp3 file or a wav.")
        sys.exit(1)  # Exit the script with an error code if audio files are missing
    
    if not has_parameters:
        print("Error: Parameters file is missing. Make sure to include a parameters.txt file that defines the number of speakers (1 to 5), language of the speaker/s and .")
        sys.exit(1)  # Exit the script with an error code if parameters files are missing

# Function to detect and process files
def process_files():
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Check for audio files
        if file_name.lower().endswith(audio_extensions):
            print("Detected audio file.")
            shutil.copy(file_path, audio_dir)
        
        # Check for parameters files
        elif file_name.lower().endswith(".txt"):
            print("Detected parameters file.")
            shutil.copy(file_path, parameters_dir)
        
        """
        
        FUTURE IMPLEMENTATION

        from moviepy.editor import VideoFileClip

        # Check for video files
        elif file_name.lower().endswith(video_extensions):
            print("Detected video file.")
            # Copy the video file to the audio folder
            shutil.copy(file_path, audio_dir)
            
            # Extract audio and save as mp3
            video_clip = VideoFileClip(file_path)
            audio_output_path = os.path.join(audio_dir, os.path.splitext(file_name)[0] + '.mp3')
            video_clip.audio.write_audiofile(audio_output_path)
            video_clip.close()

        """

# Run the check
check_required_files()

# Run the process
process_files()

print("File processing complete.")





### PERFORM TRANSCRIPTION AND DIARIZATION ###

import warnings

# Suppress specific warnings by category
warnings.filterwarnings("ignore")

# speaker_diarization_transcription.py

import os
from dotenv import load_dotenv
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from collections import defaultdict

class SpeakerDiarizationTranscription:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Access the Hugging Face token
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        # Check if the token was loaded correctly
        if self.huggingface_token:
            print("Hugging Face token loaded successfully.")
        else:
            raise ValueError("Failed to load Hugging Face token. Check your .env file.")

        # Initialize the models
        self.pipeline = self.load_pyannote_model()
        self.pipe = self.load_whisper_model()

    def load_pyannote_model(self):
        # Initialize the pyannote pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=self.huggingface_token)

        # Check if CUDA is available and move the pipeline to GPU if it is
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print("Pyannote pipeline moved to GPU.")
        else:
            print("CUDA is not available. The pyannote pipeline will run on the CPU.")

        return pipeline

    def convert_to_wav(self, audio_file):
        # Check if the file is mp3 and convert to wav if necessary
        if audio_file.lower().endswith('.mp3'):
            audio = AudioSegment.from_file(audio_file)
            wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_file, format="wav")
            return wav_file
        else:
            return audio_file
        
    """ DEVELOP CODE FOR 1 SPEAKER HERE. ITS AN EXCEPTION WHERE WE DONT NEED TO DIARIZE """
    
    def perform_diarization(self, wav_file, num_speakers=None):
        # Initialize an empty list to store the diarization entries
        diarization_list = []

        # Apply the pipeline to the wav file
        diarization = self.pipeline(wav_file, num_speakers=num_speakers)

        # Iterate over the diarization results and build the list of dictionaries
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Create a dictionary for each diarization segment
            diarization_entry = {
                'start': turn.start,
                'stop': turn.end,
                'speaker': speaker
            }
            # Add the dictionary to the list
            diarization_list.append(diarization_entry)

        return diarization_list

    def load_whisper_model(self):
        # Set device and data types
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Specify the model ID
        model_id = "openai/whisper-large-v2"  # Updated to a valid model ID

        # Load the model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)

        # Load the processor
        processor = AutoProcessor.from_pretrained(model_id)

        # Create the pipeline with additional parameters
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps="word",  # Ensure word-level timestamps are returned
            chunk_length_s=30,         # Adjust for long audio files
            stride_length_s=(5, 5),    # Overlapping chunks to improve accuracy
        )

        print("Whisper model loaded successfully.")

        return pipe

    def transcribe_audio(self, wav_file, language):
        # Transcribe the audio file and get detailed results
        result = self.pipe(wav_file, return_timestamps=True, generate_kwargs={"language": language})

        transcription = []

        # Check if the result contains 'chunks' directly
        if 'chunks' in result:
            # Process the chunks directly
            transcription = result['chunks']
        elif 'segments' in result:
            # The 'segments' key exists, process each segment
            for segment in result['segments']:
                segment_start = segment['start']
                # Each segment may have multiple chunks
                for chunk in segment['chunks']:
                    # Adjust the chunk's timestamps to be absolute
                    t_start, t_end = chunk['timestamp']
                    chunk['timestamp'] = (t_start + segment_start, t_end + segment_start)
                    transcription.append(chunk)
        else:
            # Fallback to processing the entire result
            transcription = [{'timestamp': (result['start'], result['end']), 'text': result['text']}]

        return transcription

    def assign_speakers_to_transcription(self, diarization, transcription):
        # List to store results with assigned speakers
        assigned_transcription = []
        
        # Loop over each transcription segment to find the primary speaker
        for segment in transcription:
            segment_start, segment_end = segment['timestamp']
            segment_duration = segment_end - segment_start

            # Track overlapping time for each speaker in the current segment
            speaker_overlap = {}

            # Loop over each diarization entry to calculate overlap with the transcription segment
            for entry in diarization:
                entry_start, entry_end = entry['start'], entry['stop']
                
                # Calculate the overlap duration
                overlap_start = max(segment_start, entry_start)
                overlap_end = min(segment_end, entry_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # If there is overlap, accumulate it by speaker
                if overlap_duration > 0:
                    speaker = entry['speaker']
                    if speaker not in speaker_overlap:
                        speaker_overlap[speaker] = 0
                    speaker_overlap[speaker] += overlap_duration

            # Determine the primary speaker based on the highest overlap percentage
            if speaker_overlap:
                main_speaker = max(speaker_overlap, key=speaker_overlap.get)
                max_percentage = (speaker_overlap[main_speaker] / segment_duration) * 100
            else:
                main_speaker = None  # No clear speaker found
                max_percentage = 0

            # Append the transcription segment with identified speaker
            assigned_transcription.append({
                'timestamp': segment['timestamp'],
                'text': segment['text'],
                'main_speaker': main_speaker,
                'main_speaker_percentage': max_percentage
            })

        return assigned_transcription


    def get_diarization_and_transcription(self, audio_file, num_speakers, language):
        # Convert to wav if necessary
        wav_file = self.convert_to_wav(audio_file)

        # Perform diarization
        print("Starting speaker diarization...")
        diarization_list = self.perform_diarization(wav_file, num_speakers)
        print("Speaker diarization completed.")

        # Transcribe audio
        print("Starting transcription...")
        transcription = self.transcribe_audio(wav_file, language)
        print("Transcription completed.")

        # Clean up the temporary wav file if it was created
        if wav_file != audio_file:
            os.remove(wav_file)

        return diarization_list, transcription

    def process_audio(self, audio_file, num_speakers, language):
        # Convert to wav if necessary
        wav_file = self.convert_to_wav(audio_file)

        # Perform diarization
        print("Starting speaker diarization...")
        diarization_list = self.perform_diarization(wav_file, num_speakers)
        print("Speaker diarization completed.")

        # Transcribe audio
        print("Starting transcription...")
        transcription = self.transcribe_audio(wav_file, language)
        print("Transcription completed.")

        # Assign speakers to transcription
        print("Mapping speakers to transcription...")
        combined_result = self.assign_speakers_to_transcription(diarization_list, transcription)
        print("Speaker mapping completed.")

        # Clean up the temporary wav file if it was created
        if wav_file != audio_file:
            os.remove(wav_file)

        return combined_result
    


    

### INITIALIZE MODELS ###
    
import sys
import logging
from contextlib import redirect_stdout

# Avoid info logs
logging.getLogger("speechbrain").setLevel(logging.WARNING)
sdt = SpeakerDiarizationTranscription()




### READ FILES INFO ###

# Define the path to the parameters file
parameters_file = os.path.join('parameters', 'parameters.txt')

# Initialize variables to store values
num_speakers = None
language = None

# Read the file and extract values
with open(parameters_file, 'r') as file:
    for line in file:
        # Ignore comments and empty lines
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        # Extract num_speakers
        if line.startswith('num_speakers'):
            num_speakers = int(line.split('=')[1].strip())
        # Extract language
        elif line.startswith('language'):
            language = line.split('=')[1].strip().strip('"')

# Print the variables to confirm
#print("Number of speakers:", num_speakers)
#print("Language:", language)

# Directory where audio files are stored
audio_folder = 'audio'

# List of complete paths for audio files
audio_files = [os.path.join(audio_folder, file) for file in os.listdir(audio_folder) if file.endswith(('.mp3', '.wav'))]
audio_file = audio_files[0]

#print(audio_file)

# Get duration in seconds
duration_in_seconds = len(AudioSegment.from_file(audio_file)) / 1000

# Convert to hours, minutes, and seconds
hours, remainder = divmod(duration_in_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

# Format as HH:MM:SS h
formatted_duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02} h"
#print(f"Duration: {formatted_duration}")




### GENERATE TRANSCRIPTION WITH DIARIZATION ###

# Process the audio file
combined_result = sdt.process_audio(audio_file, num_speakers, language)

# Initialize an empty list to store the merged result
merged_result = []

# Keep track of the last speaker to detect changes
last_speaker = None
current_text = ""
current_timestamp = None

for chunk in combined_result:
    speaker = chunk['main_speaker']
    text = chunk['text']
    timestamp = chunk['timestamp']

    # If this speaker is the same as the last one, concatenate the text
    if speaker == last_speaker:
        current_text += " " + text  # Add space between concatenated text segments
    else:
        # If a new speaker starts, save the previous speaker's concatenated text
        if last_speaker is not None:
            merged_result.append({
                'speaker': last_speaker,
                'text': current_text,
                'timestamp': current_timestamp
            })
        
        # Start a new text segment for the current speaker
        last_speaker = speaker
        current_text = text
        current_timestamp = timestamp  # Capture timestamp of the first chunk for this speaker

# Append the final speaker's text
if last_speaker is not None:
    merged_result.append({
        'speaker': last_speaker,
        'text': current_text,
        'timestamp': current_timestamp
    })

print("Here is a transcription snippet:\n")

# Display the merged result
for entry in merged_result[:6]:
    print(f"{entry['speaker']}: {entry['text']}")



### GIVE OPTION TO NAME SPEAKERS ###

# Ask the user if they want to name the speakers
user_input = input(f"Based on this transcription chunk, would you like to name the {num_speakers} speakers? (Y/N): ")
name_speakers = user_input.strip().upper() == "Y"

# If the user wants to name the speakers, ask for each speaker's name
speaker_names = []
if name_speakers:
    # Assume num_speakers is defined somewhere earlier in your code
    for i in range(num_speakers):
        speaker_name = input(f"Name for SPEAKER_{i:02}: ")
        speaker_names.append(speaker_name.strip().upper())
else:
    speaker_names = None



### REPLACE SPEAKER NAMES IF NECESSARY ###

def replace_speaker_names(merged_result, speaker_names=None):

    if speaker_names == None:
        return merged_result
    # Create a dictionary to map SPEAKER_XX to provided names
    speaker_map = {f"SPEAKER_{str(i).zfill(2)}": name for i, name in enumerate(speaker_names)}
    
    # Replace speaker names in merged_result
    for entry in merged_result:
        if entry['speaker'] in speaker_map:
            entry['speaker'] = speaker_map[entry['speaker']]
    
    return merged_result

# Example usage:
# Assuming merged_result is the result from the code you shared

# Call the function
updated_result = replace_speaker_names(merged_result, speaker_names)




### STORE TRANSCRIPTION FILE ###

# Specify the file path
base_name = os.path.basename(audio_file)
filename = os.path.splitext(base_name)[0]
transcription_path = f"transcription/{filename}.txt"

# Write the merged result to the file
with open(transcription_path, "w", encoding="utf-8") as file:
    for entry in merged_result:
        # Format the timestamp into start and end time with a clearer structure
        start_time, end_time = entry['timestamp']
        start_time_formatted = f"{int(start_time // 60)}:{int(start_time % 60):02}.{int((start_time % 1) * 100):02}"
        end_time_formatted = f"{int(end_time // 60)}:{int(end_time % 60):02}.{int((end_time % 1) * 100):02}"
        
        # Storing the speaker, timestamp, and text in a more readable way
        file.write(f"{entry['speaker']} ({start_time_formatted} - {end_time_formatted}): {entry['text']}\n")

print("Transcription saved succesfully.")



### CHUNK TRANSCRIPTION FILE ###
try:
    with open(transcription_path, 'r') as file:
        word_counts = [len(line.split()) for line in file if line.strip()]
    average_words_per_line = sum(word_counts) / len(word_counts) if word_counts else 0
except Exception as e:
    average_words_per_line = str(e)

# Calculate the minimum and maximum lines per chunk based on the target word count
min_lines = 200 / average_words_per_line
max_lines = 400 / average_words_per_line

# Calculate the average of these two values
average_lines_per_chunk = round((min_lines + max_lines) / 2)

# Calculate the minimum and maximum lines for overlap based on the 50-100 word range
min_overlap_lines = 50 / average_words_per_line
max_overlap_lines = 100 / average_words_per_line

# Calculate the average of these two values
average_overlap_lines = (min_overlap_lines + max_overlap_lines) / 2

# Round the results to the nearest integer
min_overlap_lines_rounded = round(min_overlap_lines)
max_overlap_lines_rounded = round(max_overlap_lines)
average_overlap_lines_rounded = round(average_overlap_lines)

average_overlap_lines_rounded

# Set chunk and overlap sizes
chunk_size = average_lines_per_chunk
overlap_size = average_overlap_lines_rounded

# Read the transcription file
try:
    with open(transcription_path, 'r') as file:
        lines = file.readlines()

    # Generate chunks
    chunks = []
    i = 0
    while i < len(lines):
        # Define the chunk from the current position to chunk_size lines ahead
        chunk = lines[i:i + chunk_size]
        chunks.append("".join(chunk))  # Join lines in the chunk to create a single string
        
        # Move index forward by chunk_size - overlap_size to allow overlap
        i += chunk_size - overlap_size

except FileNotFoundError:
    print("File not found. Please check the file path.")




### STORE CHUNKS ###

# Define output directory for chunks
chunks_dir = "chunks"

# Save each chunk as an individual file
for idx, chunk in enumerate(chunks, 1):
    file_name = f"{idx}.txt"
    file_path = os.path.join(chunks_dir, file_name)
    
    with open(file_path, 'w') as f:
        f.write(chunk)
    
print("All chunks successfully saved.")




### GENERATE SUMMARY WITH GROK ###
import sys
sys.path.append('/mnt/c/Users/luisg/Desktop/STAR/STAR/scripts')

# Now you can import grok as if it's in the same directory
import grok
    
print("Generating summary with Grok.")

# Example usage
client = grok.initialize_grok_api()

# Get transcription
try:
    with open(transcription_path, 'r', encoding='utf-8') as file:
        transcription = file.read()
    transcription  # Displaying content to verify successful read
except FileNotFoundError:
    transcription = "File not found. Please check the file path and try again."
    print("Transcription file not found. Please check the file path and try again.")
    sys.exit(1)  # Exit the script with an error code if parameters files are missing

summary = grok.generate_summary(client, transcription, formatted_duration, num_speakers, language)

if summary:
    # Define the full path for the output file
    summary_path = f"chunks/summary.txt"

    # Write the message content to the specified file
    with open(summary_path, "w") as file:
        file.write(summary)

    print(f"Summary saved to {summary_path}")
else:
    print("Error generating summary with Grok.")
    sys.exit(1)  # Exit the script with an error code if parameters files are missing




### VECTORIZE SUMMARY AND CHUNKS AND STORE VECTORS ###
    
print("Generating vectors.")

import torch
from transformers import AutoModel, AutoTokenizer

model_name = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Check if a GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Assuming you want to use TF-IDF
import pandas as pd

# Define output directory for chunks
chunks_dir = "chunks"
vectors_dir = "vectors"

# Function to process each text file and convert to embeddings
def process_text_files(chunks_dir, vectors_dir):
    # Iterate through each file in the chunks directory
    for file_name in os.listdir(chunks_dir):
        if file_name.endswith(".txt"):
            # Read file content
            file_path = os.path.join(chunks_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Tokenize and get embeddings, move inputs to the same device as the model
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Save embeddings to individual numpy files
            npy_file_path = os.path.join(vectors_dir, f"{os.path.splitext(file_name)[0]}.npy")
            np.save(npy_file_path, embeddings)

# Run the function
process_text_files(chunks_dir, vectors_dir)
    
print("Vectors generated.")
    
print("The tool is ready! VideoQ&A.py to ask information about this video using Grok.")