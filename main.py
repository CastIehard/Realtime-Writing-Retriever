import os
import numpy as np
import faiss
import pickle
import openai
from dotenv import load_dotenv
import fitz  # PyMuPDF für PDF-Verarbeitung
import re

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
INDEX_PATH = 'Index/faiss_index.pkl'
HIGHLIGHTED_PDF_DIR = "Highlighted_PDFs"


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


def highlight_and_save_pdf(original_pdf_path, text_to_highlight, page_number, max_words_to_highlight=100):
    """
    Highlights the given text in the specified page of a PDF and saves it to a separate directory.
    Implements a robust search by finding the longest contiguous sequence of words
    from the beginning and end of the text, then connecting them.

    Args:
        original_pdf_path (str): Path to the original PDF file.
        text_to_highlight (str): The text to be highlighted.
        page_number (int): The 1-based page number where the text is located.
                           (Assumes 'page' from index_data is 1-based)
        max_words_to_highlight (int): Maximum number of words to highlight. Prevents full-page highlighting.

    Returns:
        str: The path to the highlighted PDF if successful, None otherwise.
    """
    # Create the highlighted PDF directory if it doesn't exist
    if not os.path.exists(HIGHLIGHTED_PDF_DIR):
        os.makedirs(HIGHLIGHTED_PDF_DIR)

    # Construct the output PDF path in the highlighted PDF directory
    base, ext = os.path.splitext(os.path.basename(original_pdf_path))  # Use basename to avoid path issues
    output_pdf_path = os.path.join(HIGHLIGHTED_PDF_DIR, f"{base}-highlighted{ext}")

    doc = None  # Initialize doc to None for robust error handling
    try:
        doc = fitz.open(original_pdf_path)
        if not doc:
            print(f"Error: Could not open PDF file at {original_pdf_path}")
            return None

        # PyMuPDF page numbers are 0-indexed
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= len(doc):
            print(f"Error: Invalid page number {page_number} for PDF {original_pdf_path} with {len(doc)} pages.")
            doc.close()
            return None

        page = doc.load_page(page_idx)
        text_to_highlight = text_to_highlight.strip()  # Remove leading/trailing whitespace
        words = text_to_highlight.split()
        num_words = len(words)

        # Check if the text is too long to highlight (potential full-page highlight)
        if num_words > max_words_to_highlight:
            print(f"Warning: Text is too long ({num_words} words). Refusing to highlight to prevent full-page highlighting.")
            doc.close()
            return None

        # --- Forward Search ---
        forward_rects = []
        forward_text = ""
        for i in range(num_words):
            subsequence = " ".join(words[:i+1])
            rects = page.search_for(subsequence)
            if rects:
                forward_rects = rects
                forward_text = subsequence
            else:
                break

        # --- Backward Search ---
        backward_rects = []
        backward_text = ""
        for i in range(num_words):
            subsequence = " ".join(words[num_words-i-1:])
            rects = page.search_for(subsequence)
            if rects:
                backward_rects = rects
                backward_text = subsequence
            else:
                break

        # --- Combine Results ---
        all_rects = []
        if forward_rects:
            print(f"Forward search found: '{forward_text}'")
            all_rects.extend(forward_rects)
        if backward_rects and backward_text != forward_text: # Avoid double highlighting same text
            print(f"Backward search found: '{backward_text}'")
            all_rects.extend(backward_rects)

        # --- Connect the Middle (if needed) ---
        if forward_rects and backward_rects and forward_text != backward_text:
            # Try to find the text between forward and backward
            middle_words = words[len(forward_text.split()):num_words - len(backward_text.split())]
            if middle_words:
                middle_text = " ".join(middle_words)
                middle_rects = page.search_for(middle_text)
                if middle_rects:
                    print(f"Middle search found: '{middle_text}'")
                    all_rects.extend(middle_rects)
                else:
                    print("Warning: Could not find text in the middle.")

        if not all_rects:
            print("Warning: No text found to highlight.")
            doc.close()
            return None

        # --- Highlight ---
        for inst in all_rects:
            highlight = page.add_highlight_annot(inst)
            highlight.update()  # Apply the highlighting

        doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
        print(f"Highlighted PDF saved as {output_pdf_path}")
        doc.close()
        return output_pdf_path

    except Exception as e:
        print(f"Error processing PDF {original_pdf_path} for highlighting: {e}")
        if doc:  # Ensure document is closed if it was opened
            doc.close()
        return None


def generate_answer(query, context, model=LLM_MODEL):
    """
    Generates an answer to the query using the OpenAI LLM, given the context.
    The LLM is instructed to provide guidance based on the code snippet and relevant guidelines.

    Args:
        query (str): The query string (e.g., user's question about the code).
        context (str): The context string, containing both the code snippet and relevant guidelines.
        model (str): The name of the OpenAI LLM to use.

    Returns:
        str: The generated answer, providing guidance or "none" if the guidelines are irrelevant.
    """
    try:
        prompt = (
            "You are a coding assistant that helps users write code that adheres to company guidelines.\n"
            "The user will provide a code snippet and relevant guidelines. Your task is to analyze the code and guidelines, "
            "and provide specific, actionable advice to the user based on the guidelines.\n"
            "Focus on identifying potential violations of the guidelines in the code and suggesting concrete steps to address them.\n"
            "If the guidelines are directly relevant to the code and can help the user improve it, provide guidance based on the guidelines. "
            "For example, if the user is writing a function to validate travel expenses and the guidelines state 'Reimbursement is allowed for 4-star hotels or below', "
            "and the code doesn't check the hotel star rating, you should respond with something like: 'Remember to check that the hotel is 4 stars or below in your code. I highlighted the relevant part in the PDF.'\n"
            "If the guidelines are not relevant to the code, or if you cannot provide any specific guidance, respond with 'none'.\n"
            "Do not provide general information or summaries of the guidelines. Only provide specific advice that is directly applicable to the code.\n"
            "\n"
            "Context:\n{context}\n"
            "\n"
            "Question:\n{query}\n"
            "\n"
            "Answer:"
        )

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt.format(context=context, query=query)}],
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

    #load query from exapmple_code.txt
    query_file_path = 'example_code_2.txt'
    with open(query_file_path, 'r') as f:
        query = f.read().strip()


    # 1. Perform Retrieval
    results = search_index(index, vectors, index_data, query, top_k=1)

    if results:
        top_result = results[0]
        print(f"\n--- Top Retrieval Result ---")
        print(f"Source PDF: {top_result['pdf_path']}")
        print(f"Page: {top_result['page']}")
        print(f"Retrieved Information:\n\n{top_result['text']}\n")

        # 2. Create Highlighted PDF
        original_pdf_path = top_result['pdf_path']
        text_to_highlight = top_result['text']
        page_num_to_highlight = top_result['page']

        highlighted_pdf_path = highlight_and_save_pdf(original_pdf_path, text_to_highlight, page_num_to_highlight)

        if highlighted_pdf_path:
            print(f"Successfully created highlighted PDF: {highlighted_pdf_path}")
        else:
            print(f"Failed to create or save highlighted PDF for {original_pdf_path}. The original PDF was not modified.")
        print("--- End of Highlighting Process ---\n")

        # 3. Generate Answer using LLM (based on the retrieved text)
        context = ""
        # The loop will run once as results contains only the top_result due to top_k=1
        for i, result_item in enumerate(results):
            context += f"Highlightet Information in pdf {result_item['text']}\n"
            if result_item.get('type') == 'bullet_point' and 'page_content' in result_item:  # Added .get for safety
                context += f"Page Content for the highlighted text above: {result_item['page_content']}\n"
            context += "-" * 20 + "\n"

        answer = generate_answer(query, context)
        print("\n--- Generated Answer ---")
        print(answer)
        print("--- End of Answer ---")
    else:
        print("No results found for your query.")


if __name__ == "__main__":
    main()
