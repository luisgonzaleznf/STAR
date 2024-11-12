# STAR
 Speech Transcription, Analysis, and Retrieval, using whisper, Grok and RAG

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Demo](#demo)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Introduction
The STAR project aims to make audio and video content more accessible by providing users with tools to search and analyze media data efficiently. Instead of manually listening to or taking notes from an entire media piece, STAR allows users to query specific information and receive precise responses. This approach empowers users to save time and enhances accessibility across various fields, including research, journalism, and education.

## Features
- **Transcription and Analysis**: Converts audio and video files into text for easy analysis.
- **Natural Language Querying**: Allows users to ask specific questions and get answers directly from the media content.
- **Summarization**: Automatically summarizes lengthy audio or video files, providing key insights and main points.
- **Keyword Extraction**: Identifies and highlights important keywords and phrases from the content.
- **Multi-language Support**: Works with multiple languages, broadening accessibility for diverse users.
- **Search and Timestamping**: Search for topics, phrases, or keywords within the content, with timestamped results for easy navigation.

## Installation
Get STAR up and running on your local machine by following these steps.

### Prerequisites
Ensure you have the following installed:
- [Python](https://www.python.org/) - v3.10
- [ffmpeg](https://ffmpeg.org/) - for processing audio and video
- **Hugging Face Token** - for accessing Hugging Face models and services. You can obtain a token by signing up on [Hugging Face](https://huggingface.co/).
- **X_AI Token** - for using X_AI capabilities. Sign up and obtain a token from [X_AI](https://x.ai/).

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/luisgonzaleznf/STAR
   ```
2. **Navigate to the project folder:**
   ```bash
   cd STAR
   ```
3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables (see [Configuration](#configuration) for details):**
5. **Store an audio file and a parameters.txt file in the folder**
5. **Run the main scripts in order:**
   ```bash
   python GenerateMentalMap.py
   python VideoQ&A.py
   ```

## Configuration
STAR uses several environment variables to configure different aspects of the project.

### Required Environment Variables
- `HUGGINGFACE_TOKEN` - Your token for Hugging Face services.
- `X_AI_TOKEN` - Your token for X_AI services.

## Contributing
Guidelines for contributing to the project.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add a new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

---

**Optional Sections**

- **Known Issues**: List any current bugs or issues in the project.
- **Roadmap**: Outline future features and improvements.
- **Acknowledgments**: Give credit to any contributors or libraries used.
  
---

Feel free to customize this template to fit your project’s specific needs!
