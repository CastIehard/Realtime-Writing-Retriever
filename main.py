import os
import numpy as np
import faiss
import pickle
import openai
from dotenv import load_dotenv
import fitz  # PyMuPDF für PDF-Verarbeitung
import re
import tkinter as tk
from tkinter import scrolledtext, filedialog, Label, Frame
from PIL import Image, ImageTk  # Pillow library for image handling
import io  # For handling image data in memory
import threading  # For running tasks in the background

load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
INDEX_PATH = 'Index/faiss_index.pkl'
HIGHLIGHTED_PDF_DIR = "Highlighted_PDFs"

# Ensure the highlighted PDF directory exists
if not os.path.exists(HIGHLIGHTED_PDF_DIR):
    os.makedirs(HIGHLIGHTED_PDF_DIR)


def load_index(index_path):
    """Loads the FAISS index, index data, and vectors from a file."""
    try:
        with open(index_path, 'rb') as f:
            index_data, vectors, index = pickle.load(f)
        print(f"FAISS index loaded from {index_path}")
        return index_data, vectors, index
    except Exception as e:
        print(f"Error loading index from {index_path}: {e}")
        return None, None, None


def get_embedding(text, model=EMBEDDING_MODEL):
    """Gets the embedding for the given text using the OpenAI API."""
    try:
        text = text.replace("\n", " ")
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None


def search_index(index, vectors, index_data, query, top_k=1):
    """Searches the FAISS index for the most similar vectors to the query vector."""
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
    """Highlights the given text in the specified page of a PDF and saves it."""
    # Construct the output PDF path in the highlighted PDF directory

    output_pdf_path = os.path.join(HIGHLIGHTED_PDF_DIR, "temp_pdf_highlighted.pdf")

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
    """Generates an answer to the query using the OpenAI LLM, given the context."""
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


class PDFViewer(tk.Frame):
    """A Tkinter frame to display a PDF page as an image."""

    def __init__(self, parent, pdf_path="", page_num=0, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.image_label = Label(self)
        self.image_label.pack(fill="both", expand=True)
        self.update_image()

    def update_image(self):
        """Updates the image displayed in the frame."""
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            self.image_label.config(image="", text="No PDF to display.")
            self.image_label.image = None  # Clear the reference
            return

        try:
            doc = fitz.open(self.pdf_path)
            page = doc.load_page(self.page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img, text="")
            self.image_label.image = img  # Keep a reference!
            doc.close()
        except Exception as e:
            self.image_label.config(image="", text=f"Error displaying PDF: {e}")
            self.image_label.image = None


class MainUI:
    """Main UI class for the application."""

    def __init__(self, root):
        self.root = root
        root.title("Code Analysis and PDF Highlighting")

        # Define font sizes
        default_font_size = 10  # Assuming a default base size
        large_font_size = default_font_size * 2
        label_font = ("Arial", large_font_size, "bold")
        text_font = ("Arial", large_font_size)


        # Load Index
        self.index_data, self.vectors, self.index = load_index(INDEX_PATH)
        if self.index is None:
            print("Error: Could not load the FAISS index. Exiting.")
            root.destroy()  # Close the window if index loading fails
            return

        # UI Components

        # LLM Response on Top Right
        self.llm_response_label = Label(root, text="LLM Response:", font=label_font)
        self.llm_response_label.grid(row=0, column=1, sticky="w", padx=10, pady=(10,0)) # Added padding

        self.llm_response_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=5, state="disabled", font=text_font) # Adjusted height for larger font
        self.llm_response_text.grid(row=1, column=1, padx=10, pady=(5, 0), sticky="nsew")

        # PDF Viewer below LLM Response
        self.pdf_frame = Frame(root)
        self.pdf_frame.grid(row=2, column=1, padx=10, pady=(5, 10), sticky="nsew")

        self.pdf_viewer = PDFViewer(self.pdf_frame, width=400, height=300) # Adjusted height
        self.pdf_viewer.pack(fill="both", expand=True)

        # Code Input on the Left
        self.code_label = Label(root, text="Code Input:", font=label_font)
        self.code_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,0)) # Added padding

        self.code_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10, font=text_font) # Adjusted height
        self.code_input.grid(row=1, column=0, rowspan=3, padx=10, pady=(5, 10), sticky="nsew") # rowspan adjusted
        self.code_input.bind("<KeyRelease>", self.on_code_change)  # Trigger on code changes

        # Layout Configuration
        root.grid_rowconfigure(1, weight=0)  # LLM response row, less weight
        root.grid_rowconfigure(2, weight=1)  # PDF viewer row, more weight
        root.grid_rowconfigure(3, weight=0)  # Empty row if rowspan=3 for code_input is used for spacing
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)


        # Initialize PDF path and viewer
        self.pdf_path = ""  # No default PDF
        self.pdf_viewer.pdf_path = self.pdf_path
        self.pdf_viewer.update_image()

        self.original_pdf_path = ""  # Store the original path
        self.text_to_highlight = ""  # Initialize text to highlight
        self.page_num_to_highlight = 0  # Initialize page number

        # Initialize the LLM response
        self.llm_response = ""

    def on_code_change(self, event=None):
        """Handles changes in the code input."""
        code = self.code_input.get("1.0", tk.END)
        lines = code.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "" and lines[-2].strip() == "":
            code_to_process = code.strip()
            if code_to_process: # Only process if there's actual code
                self.process_code(code_to_process)
            else: # If stripping results in empty string, treat as cleared
                self.clear_pdf_and_response()
        elif not code.strip():  # If the code input is empty
            self.clear_pdf_and_response()


    def process_code(self, code):
        """Processes the code, retrieves information, and updates the UI."""
        # Run this in a separate thread to prevent UI from freezing
        threading.Thread(target=self.run_analysis, args=(code,), daemon=True).start()

    def run_analysis(self, code):
        """Runs the code analysis and updates the UI."""
        if not code: # Should be caught by on_code_change, but good safeguard
            self.root.after(0, self.clear_pdf_and_response)
            self.root.after(0, lambda: self.update_llm_response("No code provided."))
            return

        # 1. Perform Retrieval
        results = search_index(self.index, self.vectors, self.index_data, code, top_k=1)

        if results:
            top_result = results[0]
            self.original_pdf_path = top_result['pdf_path']
            self.text_to_highlight = top_result['text']
            self.page_num_to_highlight = top_result['page']

            # 2. Generate Answer using LLM
            context = f"Highlighted Information in pdf: {self.text_to_highlight}\n"
            if top_result.get('type') == 'bullet_point' and 'page_content' in top_result:
                context += f"Page Content for the highlighted text above: {top_result['page_content']}\n"

            answer = generate_answer(code, context)

            # 3. Highlight PDF and Update UI
            highlighted_pdf_path = highlight_and_save_pdf(self.original_pdf_path, self.text_to_highlight, self.page_num_to_highlight)

            if highlighted_pdf_path:
                self.root.after(0, lambda: self.update_pdf_viewer(highlighted_pdf_path, self.page_num_to_highlight -1))
            else:
                self.root.after(0, lambda: self.update_pdf_viewer(self.original_pdf_path, self.page_num_to_highlight -1))

            self.root.after(0, lambda: self.update_llm_response(answer))

        else:
            self.root.after(0, lambda: self.update_llm_response("No relevant information found."))
            self.root.after(0, self.clear_pdf_and_response)


    def update_pdf_viewer(self, pdf_path, page_num=0):
        """Updates the PDF viewer with the new PDF path and page number."""
        self.pdf_viewer.pdf_path = pdf_path
        self.pdf_viewer.page_num = page_num
        self.pdf_viewer.update_image()

    def update_llm_response(self, response):
        """Updates the LLM response text in the UI."""
        self.llm_response_text.config(state="normal")  # Enable editing
        self.llm_response_text.delete("1.0", tk.END)  # Clear previous text
        self.llm_response_text.insert(tk.END, response)  # Insert new text
        self.llm_response_text.config(state="disabled")  # Disable editing

    def clear_pdf_and_response(self):
        """Clears the PDF viewer and LLM response."""
        self.update_pdf_viewer("", 0)
        self.update_llm_response("")


if __name__ == "__main__":
    root = tk.Tk()
    ui = MainUI(root)
    root.mainloop()
