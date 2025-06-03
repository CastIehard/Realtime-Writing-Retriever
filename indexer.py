import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import openai
from PyPDF2 import PdfReader
import re
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
FILES_PATH = 'Docs'
INDEX_PATH = 'Index/faiss_index.pkl'


def extract_text_from_file(file_path, file_format):
    """
    Extracts text and page information from a file based on its format.

    Args:
        file_path (str): The path to the file.
        file_format (str): The format of the file (e.g., '.pdf', '.md').

    Returns:
        list: A list of tuples, where each tuple contains (page_number, text, page_content).
              page_content is the full text of the page.

    Raises:
        ValueError: If the file format is not supported.
    """
    match file_format:
        case '.pdf':
            page_texts = []
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        page_texts.append((page_num + 1, text, text))  # Page numbers are 1-based, text is both chunk and page content
                print(f"Extracted {len(page_texts)} pages from {file_path}")
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
                return []  # Return an empty list in case of an error
            return page_texts
        case '.md':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"Extracted {len(text)} characters from {file_path}")
                return [(1, text, text)]  # Treat the entire markdown file as one page, text is both chunk and page content
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
                return []  # Return an empty list in case of an error
        case _:
            raise ValueError(f"Unsupported file format: {file_format}")


def split_text(text, file_format, max_chunk_size=5000, overlap_size=0):
    """
    Splits the text into smaller chunks based on the file format,
    considering headers and bullet points.

    Args:
        text (str): The text to split.
        file_format (str): The format of the file (e.g., '.pdf', '.md').
        max_chunk_size (int): The maximum size of each chunk.
        overlap_size (int): The size of the overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    def further_split(chunks, symbol):
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                sub_chunks = re.split(symbol, chunk)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) > max_chunk_size:
                        for i in range(0, len(sub_chunk), max_chunk_size - overlap_size):
                            new_chunks.append(sub_chunk[i:i + max_chunk_size])
                    else:
                        new_chunks.append(sub_chunk)
            else:
                new_chunks.append(chunk)
        return new_chunks

    if file_format == '.pdf':
        chunks = re.split(r'\n\d+\s', text)# First split by big headers
        if any(len(chunk) > max_chunk_size for chunk in chunks):# Then split by sub headers if the chunks are still too big
            chunks = further_split(chunks, r'\n\d+\.\d+\s')
    elif file_format == '.md':
        chunks = text.split('# ')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '## ')
        if any(len(chunk) > max_chunk_size for chunk in chunks):
            chunks = further_split(chunks, '\n')

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            for i in range(0, len(chunk), max_chunk_size - overlap_size):
                final_chunks.append(chunk[i:i + max_chunk_size])
        else:
            final_chunks.append(chunk)
    return final_chunks


def extract_bullet_point_sections(text, bullet_point_symbols=r"[•*-]\s+"):
    """
    Extracts sections of text that start with a bullet point.
    Stops at the next empty line or the next bullet point.

    Args:
        text (str): The text to analyze.
        bullet_point_symbols (str): Regular expression for bullet point symbols.

    Returns:
        list: A list of text sections that start with a bullet point.
    """
    bullet_point_sections = []
    pattern = re.compile(bullet_point_symbols, re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        start = match.start()
        # Find the next empty line after the bullet
        next_empty = text.find('\n', start)
        # Find the next bullet after this one
        next_bullet = matches[i + 1].start() if i + 1 < len(matches) else -1

        # Determine the end: the earliest of next empty line or next bullet
        ends = [pos for pos in [next_empty, next_bullet] if pos != -1]
        end = min(ends) if ends else len(text)

        bullet_point_sections.append(text[start:end].strip())

    return bullet_point_sections


def get_all_docs(FILES_PATH, extensions):
    """
    Gets all file paths with the specified extensions in the given directory.

    Args:
        FILES_PATH (str): The path to the directory.
        extensions (tuple): A tuple of file extensions to include.

    Returns:
        list: A list of file paths.
    """
    try:
        file_paths = []
        for root, _, files in os.walk(FILES_PATH):
            for file in files:
                if file.endswith(extensions):
                    file_paths.append(os.path.join(root, file))
        print(f"Found {len(file_paths)} files in {FILES_PATH}")
        if not file_paths:
            print(f"No files found in {FILES_PATH} with extensions {extensions}")
        else:
            for file_path in file_paths:
                print(f"- {file_path}")
        print('\n')
        return file_paths
    except Exception as e:
        print(f"Error getting files from {FILES_PATH}: {e}")
        return []


def create_faiss_index(vectors):
    """
    Creates a FAISS index from the given vectors.

    Args:
        vectors (list): A list of vectors.

    Returns:
        tuple: A tuple containing the numpy array of vectors and the FAISS index.
    """
    vectors_np = np.array(vectors).astype('float32')
    faiss.normalize_L2(vectors_np)
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)
    print(f"FAISS index created with {vectors_np.shape[0]} vectors of dimension {vectors_np.shape[1]}")
    return vectors_np, index


def save_index(index_data, vectors, index, INDEX_PATH):
    """
    Saves the FAISS index, vectors, and index data to a file.

    Args:
        index_data (list): A list of dictionaries containing the file path, page number, and text chunk.
        vectors (numpy.ndarray): The numpy array of vectors.
        index (faiss.Index): The FAISS index.
        INDEX_PATH (str): The path to the file to save the index to.
    """
    try:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump((index_data, vectors, index), f)
        print(f"FAISS index and vectors saved to {INDEX_PATH}")
    except Exception as e:
        print(f"Error saving index to {INDEX_PATH}: {e}")


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

def main():
    """
    Main function to extract text from files, embed them using OpenAI API,
    create a FAISS index, and save the index.
    """
    # Get all file paths
    file_paths = get_all_docs(FILES_PATH, extensions=('.pdf', '.md'))

    if not file_paths:
        print("No files found to process. Exiting.")
        return

    index_data = []  # List to store file path, page number, and text chunk
    all_vectors = []

    for file_path in file_paths:
        file_format = os.path.splitext(file_path)[1]
        page_texts = extract_text_from_file(file_path, file_format)
        
        if not page_texts:
            print(f"Skipping {file_path} due to extraction errors.")
            continue

        # Embed the texts and store the data
        for page_num, text, page_content in page_texts:
            # First, split the text into standard chunks
            chunks = split_text(text, file_format, max_chunk_size=5000, overlap_size=500)
            if chunks:
                print(f"Splitting text from {file_path} page {page_num} into {len(chunks)} chunks")
                for chunk in chunks:
                    vector = get_embedding(chunk)
                    if vector:
                        all_vectors.append(vector)

                        index_data.append({
                            'pdf_path': file_path,
                            'page': page_num,
                            'text': chunk,
                            'type': 'paragraph',  # Mark as paragraph chunk
                            'page_content': "" # Empty for paragraph
                        })
                    else:
                        print(f"Skipping chunk due to embedding error.")
            else:
                print(f"Skipping page {page_num} from {file_path} due to no chunks.")

            # Now, extract and embed bullet point sections
            bullet_point_sections = extract_bullet_point_sections(text)
            print(f"Extracted {len(bullet_point_sections)} bullet point sections from {file_path} page {page_num}")
            for section in bullet_point_sections:
                vector = get_embedding(section)
                if vector:
                    all_vectors.append(vector)
                    index_data.append({
                        'pdf_path': file_path,
                        'page': page_num,
                        'text': section,
                        'type': 'bullet_point',  # Mark as bullet point chunk
                        'page_content': page_content # Add page content for bullet points
                    })
                else:
                    print(f"Skipping bullet point section due to embedding error.")

    if not all_vectors:
        print("No vectors created. Check your documents and embedding process. Exiting.")
        return

    print(f"\nDone extracting and embedding, creating FAISS index using {EMBEDDING_MODEL}")

    # Create FAISS index
    vectors_np, index = create_faiss_index(all_vectors)

    # Save the FAISS index, index data, and vectors
    save_index(index_data, vectors_np, index, INDEX_PATH)
    print("FAISS index created and saved successfully.")


if __name__ == "__main__":
    main()
