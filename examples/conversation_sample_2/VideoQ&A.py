### INITIAL MESSAGE WITH GROK ###
import sys
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

user_input = input("")