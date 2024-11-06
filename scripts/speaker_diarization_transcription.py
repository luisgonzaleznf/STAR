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

    def perform_diarization(self, audio_file):
        # Convert mp3 to wav
        audio = AudioSegment.from_file(audio_file)
        wav_file = audio_file.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_file, format="wav")

        # Initialize an empty list to store the diarization entries
        diarization_list = []

        # Apply the pipeline to the wav file
        diarization = self.pipeline(wav_file)

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

        # Clean up the temporary wav file
        os.remove(wav_file)

        return diarization_list

    def load_whisper_model(self):
        # Set device and data types
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Specify the model ID
        model_id = "openai/whisper-large-v3-turbo"

        # Load the model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)

        # Load the processor
        processor = AutoProcessor.from_pretrained(model_id)

        # Create the pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        print("Whisper model loaded successfully.")

        return pipe

    def transcribe_audio(self, audio_file):
        # Transcribe an audio file directly in the pipeline call
        result = self.pipe(audio_file, return_timestamps=True)
        transcription = result["chunks"]  # Extract transcription chunks
        return transcription

    def assign_speakers_to_transcription(self, diarization, transcription):
        """Assign speaker labels to transcription chunks based on diarization."""
        def get_overlap(a_start, a_end, b_start, b_end):
            """Calculate the overlap duration between two time intervals."""
            overlap_start = max(a_start, b_start)
            overlap_end = min(a_end, b_end)
            return max(0.0, overlap_end - overlap_start)

        # For each transcription chunk
        for chunk in transcription:
            t_start, t_end = chunk['timestamp']
            # Keep track of overlapping durations per speaker
            overlaps = defaultdict(float)
            for diar in diarization:
                d_start = diar['start']
                d_end = diar['stop']
                speaker = diar['speaker']
                overlap = get_overlap(t_start, t_end, d_start, d_end)
                if overlap > 0:
                    overlaps[speaker] += overlap
            # Determine the speaker with the maximum overlap
            if overlaps:
                assigned_speaker = max(overlaps.items(), key=lambda x: x[1])[0]
            else:
                assigned_speaker = 'Unknown'
            # Assign the speaker to the chunk
            chunk['speaker'] = assigned_speaker
        return transcription

    def process_audio(self, audio_file):
        # Perform diarization
        print("Starting speaker diarization...")
        diarization_list = self.perform_diarization(audio_file)
        print("Speaker diarization completed.")

        # Transcribe audio
        print("Starting transcription...")
        transcription = self.transcribe_audio(audio_file)
        print("Transcription completed.")

        # Assign speakers to transcription
        print("Assigning speakers to transcription...")
        combined_result = self.assign_speakers_to_transcription(diarization_list, transcription)
        print("Speaker assignment completed.")

        return combined_result

def main():
    # Path to your audio file
    audio_file = "/mnt/c/Users/luisg/Desktop/STAR/STAR/audio_samples/conversation_sample_2.mp3"  # Replace with your audio file path

    # Create an instance of the class
    sdt = SpeakerDiarizationTranscription()

    # Process the audio file
    combined_result = sdt.process_audio(audio_file)

    # Print the results
    for chunk in combined_result:
        print(f"Speaker {chunk['speaker']}: {chunk['text']}")

if __name__ == "__main__":
    main()
