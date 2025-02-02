
# Semantic Book Recommender with LLMs

## 📖 Overview
The Semantic Book Recommender leverages Large Language Models (LLMs) and vector search to provide personalized book recommendations based on user queries. This system processes book descriptions, extracts emotional tones, and categorizes books to enhance discoverability. Using FAISS for vector storage, Google Generative AI for embeddings, and Gradio for a user-friendly interface, this project enables users to find books that match their interests and emotions seamlessly.

## 🏗 Key Components
This project consists of five major components:

1. **Text Data Cleaning** - Preprocessing and cleaning book descriptions.
   - Code: [`data-exploration.ipynb`](data-exploration.ipynb)
2. **Semantic Search & Vector Database** - Using FAISS to create a searchable book database.
   - Code: [`vector-search.ipynb`](vector-search.ipynb)
   - Functionality: Enables searching for books based on natural language queries (e.g., *"a book about a person seeking revenge"*).
3. **Text Classification (Zero-Shot Learning)** - Classifying books as *Fiction* or *Non-Fiction*.
   - Code: [`text-classification.ipynb`](text-classification.ipynb)
   - Functionality: Adds an extra filtering facet for users.
4. **Sentiment Analysis & Emotion Extraction** - Analyzing book emotions (joy, sadness, suspense, etc.).
   - Code: [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb)
   - Functionality: Enables users to sort books based on emotional tones.
5. **Web Application (Gradio UI)** - An interactive dashboard for book recommendations.
   - Code: [`gradio-dashboard.py`](gradio-dashboard.py)

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PriyanshuDey23/LLM-Semantic-Book-Recommender.git
cd Semantic-Book-Recommender
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.11 installed. Then, install required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file in the project root and add your Google API key:
```ini
GOOGLE_API_KEY=your_api_key_here
```

### 4️⃣ Run the Gradio App
```bash
python app.py
```
The app will be available at: `http://127.0.0.1:7860`

## 🎭 Features
- **AI-Powered Recommendations**: Uses semantic search to find books based on user input.
- **Emotional Tone Filtering**: Sorts books based on happiness, suspense, sadness, etc.
- **Category-Based Selection**: Users can filter by genre.
- **Interactive Web UI**: Built with Gradio for an intuitive experience.

## 🔗 Dependencies
- `pandas`
- `numpy`
- `dotenv`
- `langchain`
- `FAISS`
- `gradio`
- `GoogleGenerativeAIEmbeddings`

To install all dependencies manually, run:
```bash
pip install pandas numpy python-dotenv langchain faiss-cpu gradio
```

## 📌 License
This project is open-source and licensed under the MIT License.

## 🤝 Contributing
We welcome contributions! Feel free to fork the repo, open issues, and submit pull requests.

---
### 🚀 Happy Reading & Coding! 📚