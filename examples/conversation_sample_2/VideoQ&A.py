### GENERATE INITIAL MESSAGE WITH GROK ###
import sys
import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer

sys.path.append('/mnt/c/Users/luisg/Desktop/STAR/STAR/scripts')

# Now you can import grok as if it's in the same directory
import grok

# Example usage
client = grok.initialize_grok_api()

# Extract summary from the folder
summary_path = "chunks/summary.txt"
with open(summary_path, "r") as file:
    summary = file.read()

init_message = grok.generate_RAG_initial_message(client, summary)
    
print(init_message)

while True:

    user_question = input("")

    ### GENERATE RAG PHRASES WITH GROK ###

    # Extract summary from the folder
    summary_path = "chunks/summary.txt"
    with open(summary_path, "r") as file:
        summary = file.read()

    # Extract summary from the folder
    chunk_1_path = "chunks/1.txt"
    with open(chunk_1_path, "r") as file:
        chunk_1 = file.read()

    rag_phrases_json = grok.generate_RAG_phrases(client, user_question, summary, chunk_1)

    #print(rag_phrases_json)

    phrase_list = grok.extract_phrases_from_json(rag_phrases_json)

    ### PERFORM VECTOR SIMILARITY SEARCH ###

    model_name = "BAAI/bge-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Define a function to vectorize texts
    def vectorize_texts(texts, tokenizer, model):
        # Tokenize all texts at once
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # Get embeddings without gradient calculation
        with torch.no_grad():
            outputs = model(**inputs)
            # Calculate mean embedding for each text
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: (n_texts, hidden_dim)
        return embeddings
    
    # List of texts to vectorize
    phrase_list.append(user_question)

    #print(phrase_list)

    # Vectorize the texts
    text_embeddings = vectorize_texts(phrase_list, tokenizer, model)

    # Directory containing .npy vector files
    vector_dir = 'vectors'

    # Initialize a list to store all similarity results
    all_similarities = []

    # Threshold for similarity scores
    similarity_threshold = 0.78

    # Initialize a set to store unique filenames that meet the threshold
    chunk_set = set()

    # Calculate cosine similarity for each text against each .npy vector
    for npy_file in os.listdir(vector_dir):
        if npy_file.endswith(".npy"):
            # Load the vector from .npy file
            npy_vector = torch.tensor(np.load(os.path.join(vector_dir, npy_file)))
            
            # Ensure npy_vector is 2D for compatibility
            if npy_vector.dim() == 1:
                npy_vector = npy_vector.unsqueeze(0)
            
            # Compute cosine similarity for each text embedding with the current npy vector
            similarities = torch.nn.functional.cosine_similarity(text_embeddings, npy_vector, dim=1)
            
            # Check if any similarity meets or exceeds the threshold
            if any(similarity.item() >= similarity_threshold for similarity in similarities):
                # Add the filename without the .npy extension to the set
                chunk_set.add(os.path.splitext(npy_file)[0])

    # Output the set of chunk_set
    # print(f"Files with similarity ≥ {similarity_threshold}:", chunk_set)

    # Initialize a list to store the content of each chunk
    chunk_list = []

    # Set chunks directory
    chunk_directory = 'chunks'

    # Read each chunk file and store its content
    for chunk_id in chunk_set:
        file_path = os.path.join(chunk_directory, f"{chunk_id}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                chunk_content = file.read()
                chunk_list.append(chunk_content)
        except FileNotFoundError:
            chunk_list.append(f"Chunk file {chunk_id}.txt not found.")

    grok_RAG_response = grok.respond_RAG_question(client, user_question, chunk_list)

    print("\n")

    print(grok_RAG_response)