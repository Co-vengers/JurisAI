# JurisAI

A smart, conversational chatbot powered by Flask and Google's Gemini API that allows you to ask questions and get context-aware answers directly from the text of the Indian Constitution.

This project uses a Retrieval-Augmented Generation (RAG) system, ensuring that the chatbot's answers are fast, accurate, and based solely on the provided constitutional text.

![Indian Constitution Chatbot Interface](https://i.imgur.com/KjWv9rF.png)

## Features

- **Advanced Conversational AI:** Ask complex questions in natural language and get precise answers.
- **Retrieval-Augmented Generation (RAG):** Instead of sending the entire constitution to the AI, the app first finds the most relevant articles using vector embeddings, leading to faster and more accurate responses.
- **Dynamic Web Interface:** A clean and responsive UI with smooth animations, built with Flask and modern CSS.
- **Persistent Chat History:** Your conversation is saved in your session, allowing you to pick up where you left off.
- **Clear History:** Easily clear the entire conversation with a single click.

***

## How It Works

The chatbot's intelligence comes from a Retrieval-Augmented Generation (RAG) pipeline:

1.  **Preprocessing & Embedding:** On startup, the application reads the `indian_constitution copy.csv` file. The text of each article is converted into a numerical representation (vector embedding) using Google's embedding model. These embeddings are stored in memory.
2.  **Retrieval:** When you ask a question, your query is also converted into an embedding. The application then performs a **cosine similarity search** to compare your query's embedding against all the article embeddings to find the most relevant articles.
3.  **Augmentation & Generation:** The text of these top relevant articles is inserted into a carefully crafted prompt along with your original question. This focused context is then sent to the Gemini Pro model, which generates a final, accurate answer based *only* on the provided information.

***

## Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

- Python 3.8+ and pip
- A Google Gemini API key.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Co-vengers/Indian_constitution_chatbot.git](https://github.com/Co-vengers/Indian_constitution_chatbot.git)
    cd Indian_constitution_chatbot
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The required libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` file, you can install them manually: `pip install flask python-dotenv google-generativeai pandas numpy`)*

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of your project directory and add your API key:
    ```env
    GOOGLE_API_KEY="your_google_gemini_api_key_here"
    ```

5.  **Ensure Data File is Present:**
    Make sure the `indian_constitution copy.csv` file is in the root directory of the project.

### Running the Application

1.  **Start the Flask Server:**
    ```bash
    python app.py
    ```
2.  **Access the Chatbot:**
    Open your web browser and go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

***


## Project File Structure
```text
├── app.py
├── config.py
├── views.py
├── utils.py
├── conversation_history.json
├── indian_constitution copy.csv
├── LICENSE
├── README.md
├── requirements.txt
├── .env
├── templates/
│   └── index.html
```

## License
This project is licensed for educational purposes only.

