import os
import numpy as np
import faiss
import pickle
import openai
from dotenv import load_dotenv
import fitz  # PyMuPDF für PDF-Verarbeitung

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"  # Or another suitable GPT model
INDEX_PATH = 'Index/faiss_index.pkl'


def load_index(index_path):
    """
    Loads the FAISS index, index data, and vectors from a file.

    Args:
        index_path (str): The path to the file containing the index.

    Returns:
        tuple: A tuple containing the index data, vectors, and the FAISS index.
    """
    try:
        with open(index_path, 'rb') as f:
            index_data, vectors, index = pickle.load(f)
        print(f"FAISS index loaded from {index_path}")
        return index_data, vectors, index
    except Exception as e:
        print(f"Error loading index from {index_path}: {e}")
        return None, None, None


def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Gets the embedding for the given text using the OpenAI API.

    Args:
        text (str): The text to embed.
        model (str): The name of the OpenAI embedding model to use.

    Returns:
        list: The embedding for the given text.
    """
    try:
        text = text.replace("\n", " ")
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None


def search_index(index, vectors, index_data, query, top_k=1):
    """
    Searches the FAISS index for the most similar vectors to the query vector.

    Args:
        index (faiss.Index): The FAISS index to search.
        vectors (numpy.ndarray): The numpy array of vectors.
        index_data (list): The list of dictionaries containing the file path, page number, and text chunk.
        query (str): The query string.
        top_k (int): The number of most similar vectors to retrieve.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              pdf_path, page, text, type, and page_content for the top_k results.
    """
    query_vector = get_embedding(query)
    if query_vector is None:
        print("Error: Could not generate embedding for the query.")
        return []

    query_vector = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, top_k)  # D: distances, I: indices

    results = []
    for i in range(len(I[0])):
        index_result = I[0][i]
        result = index_data[index_result]
        results.append(result)

    return results


def highlight_and_save_pdf(original_pdf_path, text_to_highlight, page_number, output_pdf_path):
    """
    Highlights the given text in the specified page of a PDF and saves it.

    Args:
        original_pdf_path (str): Path to the original PDF file.
        text_to_highlight (str): The text to be highlighted.
        page_number (int): The 1-based page number where the text is located.
                           (Assumes 'page' from index_data is 1-based)
        output_pdf_path (str): Path to save the highlighted PDF.
    Returns:
        bool: True if highlighting and saving was successful, False otherwise.
    """
    doc = None  # Initialize doc to None for robust error handling
    try:
        doc = fitz.open(original_pdf_path)
        if not doc:
            print(f"Error: Could not open PDF file at {original_pdf_path}")
            return False

        # PyMuPDF page numbers are 0-indexed
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= len(doc):
            print(f"Error: Invalid page number {page_number} for PDF {original_pdf_path} with {len(doc)} pages.")
            doc.close()
            return False

        page = doc.load_page(page_idx)
        
        # Attempt to find the exact text
        text_instances = page.search_for(text_to_highlight)

        # Fallback: If not found, try replacing newlines in the search text with spaces
        # This can help if the chunking process joined lines that are separate in the PDF
        if not text_instances:
            print(f"Warning: Exact text chunk not found on page {page_number}. Trying with newlines replaced by spaces.")
            text_instances = page.search_for(text_to_highlight.replace("\n", " "))

        if not text_instances:
            print(f"Warning: Text '{text_to_highlight[:100].replace(chr(10), ' ')}...' not found on page {page_number} of {original_pdf_path} even after modification.")
            # Further fallbacks could be implemented here (e.g., searching sentence by sentence)
            doc.close()
            return False # Do not save if text not found

        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update() # Apply the highlighting

        doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
        print(f"Highlighted PDF saved as {output_pdf_path}")
        doc.close()
        return True

    except Exception as e:
        print(f"Error processing PDF {original_pdf_path} for highlighting: {e}")
        if doc: # Ensure document is closed if it was opened
            doc.close()
        return False


def generate_answer(query, context, model=LLM_MODEL):
    """
    Generates an answer to the query using the OpenAI LLM, given the context.

    Args:
        query (str): The query string.
        context (str): The context string.
        model (str): The name of the OpenAI LLM to use.

    Returns:
        str: The generated answer.
    """
    try:
        prompt = f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        # print(f"Prompt for LLM:\n{prompt}\n") # Optional: for debugging

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I am sorry, I could not generate an answer at this time."


def main():
    """
    Main function to load the FAISS index, query it, highlight the top result in PDF,
    and generate an answer.
    """
    index_data, vectors, index = load_index(INDEX_PATH)

    if index is None:
        print("Error: Could not load the FAISS index. Exiting.")
        return

    query = input("Enter your query: ")

    # 1. Perform Retrieval
    results = search_index(index, vectors, index_data, query, top_k=1)

    if results:
        top_result = results[0]
        print(f"\n--- Top Retrieval Result ---")
        print(f"Source PDF: {top_result['pdf_path']}")
        print(f"Page: {top_result['page']}")
        print(f"Retrieved Text Snippet (first 200 chars): {top_result['text'][:200]}...\n")

        # 2. Create Highlighted PDF
        original_pdf_path = top_result['pdf_path']
        text_to_highlight = top_result['text']
        
        # Assuming 'page' in index_data is 1-based.
        # If it's 0-based, adjust highlight_and_save_pdf or the page_num_to_highlight value.
        page_num_to_highlight = top_result['page'] 

        # Construct the output PDF path (e.g., "path/to/document-highlighted.pdf")
        base, ext = os.path.splitext(original_pdf_path)
        output_pdf_path = f"{base}-highlighted{ext}"

        print(f"Attempting to highlight text in '{original_pdf_path}' (page {page_num_to_highlight}) and save to '{output_pdf_path}'...")
        highlight_success = highlight_and_save_pdf(original_pdf_path, text_to_highlight, page_num_to_highlight, output_pdf_path)

        if highlight_success:
            print(f"Successfully created highlighted PDF: {output_pdf_path}")
        else:
            print(f"Failed to create or save highlighted PDF for {original_pdf_path}. The original PDF was not modified.")
        print("--- End of Highlighting Process ---\n")

        # 3. Generate Answer using LLM (based on the retrieved text)
        context = ""
        # The loop will run once as results contains only the top_result due to top_k=1
        for i, result_item in enumerate(results): 
            context += f"Important Information: {result_item['text']}\n"
            if result_item.get('type') == 'bullet_point' and 'page_content' in result_item: # Added .get for safety
                context += f"Page Content for the bulletpoint above: {result_item['page_content']}\n"
            context += "-" * 20 + "\n"

        answer = generate_answer(query, context)
        print("\n--- Generated Answer ---")
        print(answer)
        print("--- End of Answer ---")
    else:
        print("No results found for your query.")


if __name__ == "__main__":
    main()
