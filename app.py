import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
import gradio as gr

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load books dataset
books = pd.read_csv("E:/LLM-Semantic-Book-Recommender/Data/books_with_emotions.csv")



# Ensure the thumbnail column exists and update to large_thumbnail
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "E:/LLM-Semantic-Book-Recommender/cover-not-found.jpg",
    books["thumbnail"] + "&fife=w800"
)


# Vector Data base
# Load and process book descriptions
raw_documents = TextLoader("E:/LLM-Semantic-Book-Recommender/Data/tagged_description.txt", encoding="utf-8").load()

# Split documents into chunks for FAISS
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# FAISS storage path
FAISS_INDEX_PATH = "E:/LLM-Semantic-Book-Recommender/Faiss_Index"

# Check if FAISS index already exists
if os.path.exists(FAISS_INDEX_PATH + ".index"):
    print("🔄 Loading existing FAISS index...")
    db_books = FAISS.load_local(FAISS_INDEX_PATH, embedding_model)
else:
    print("🆕 Creating a new FAISS index...")
    db_books = FAISS.from_documents(documents, embedding_model)
    db_books.save_local(FAISS_INDEX_PATH)



# Semantic recommendation from books data set,apply filtering based on category and sort based on emotional tone 
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs if rec.page_content.strip()]

    if "isbn13" in books.columns:
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k) # Get the rows related to the book
    else:
        raise KeyError("Column 'isbn13' is missing from books dataset")

    # Filtering(Probability)
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


# UI components for Gradio
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] if pd.notna(row["description"]) else "No description available." # Get Description
        truncated_description = " ".join(description.split()[:30]) + "..." # Split the description , if descrption has more than 30 words then we give ... at end

        authors_str = row["authors"] if pd.notna(row["authors"]) else "Unknown" #Author
        authors_split = authors_str.split(";") # If the book has more than 1 author , combine with ; in between

        # If it has 2 author 
        if len(authors_split) == 2: 
            authors_str = f"{authors_split[0]} and {authors_split[1]}" # Separate 2 author with and
        
        # If it has more than 2 authors
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}" # all author except the last author are separated by , and last author is added using and

        # 1 author
        else:
            authors_str = row["authors"]

        # Display all the information in the caption
        caption = f"{row['title']} by {authors_str}: {truncated_description}" # Title ,author, description
        results.append((row["large_thumbnail"], caption)) # Large thumbnail along with caption
    return results



# UI components
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="🔍 Describe a book:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="📂 Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="🎭 Select an emotional tone:", value="All")
        submit_button = gr.Button("✨ Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="📚 Recommended Books", columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch(server_name="127.0.0.1", server_port=7860, debug=True)
