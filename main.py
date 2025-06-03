import os
import numpy as np
import faiss
import pickle
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
INDEX_PATH = 'Index/faiss_index.pkl'

def load_index(index_path):
    with open(index_path, 'rb') as f:
        index_data, vectors, index = pickle.load(f)
    return index_data, vectors, index

def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def search_index(index, vectors, index_data, query, top_k=1):
    query_vector = get_embedding(query)
    query_vector = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, top_k)
    results = []
    for i in range(len(I[0])):
        index_result = I[0][i]
        result = index_data[index_result]
        results.append(result)
    return results

def generate_checklist_questions(user_code, model=LLM_MODEL):
    prompt = f"""You are an expert assistant for software developers. 
A developer is currently working on the following code and must comply with company guidelines (which you do not know yet). 
Based on the code, generate a list of important questions the developer should consider to ensure compliance with typical company guidelines. 
The questions should be specific to the context of the code and help the developer avoid common mistakes or oversights. 
Return the questions as a numbered list.

Developer's code:
{user_code}

Questions:"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def get_guideline_hints(questions, index, vectors, index_data, top_k=1):
    hints = []
    for question in questions:
        results = search_index(index, vectors, index_data, question, top_k=top_k)
        for result in results:
            hint = {
                "question": question,
                "guideline": result['text'],
                "source": f"{os.path.basename(result['pdf_path'])}, page {result['page']}"
            }
            if result.get('type') == 'bullet_point' and result.get('page_content'):
                hint["page_content"] = result['page_content']
            else:
                hint["page_content"] = None
            hints.append(hint)
    return hints

def generate_user_hint(user_code, hints, model=LLM_MODEL):
    findings = ""
    for i, hint in enumerate(hints, 1):
        findings += f"{i}. {hint['question']}: {hint['guideline']} (Source: {hint['source']})\n"
        if hint.get("page_content"):
            findings += f"   Full page context:\n{hint['page_content']}\n"
    prompt = f"""You are an assistant for software developers. 
A developer is working on the following code. 
Below you find relevant sections from the company guidelines that answer specific questions about compliance. 
If a full page context is provided, use it to give a more precise and helpful hint. 
Based on this information, give the developer a helpful, concise hint about what to check or improve in their code to ensure compliance. 
Refer to the relevant guideline(s) in your answer.

Developer's code:
{user_code}

Relevant guideline findings:
{findings}

Hint:"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def main():
    # Load FAISS index
    index_data, vectors, index = load_index(INDEX_PATH)
    if index is None:
        print("Error: Could not load the FAISS index. Exiting.")
        return

    # Get user code (for demo, you can hardcode or use input)
    print("Paste your code:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    user_code = "\n".join(lines)

    # Step 1: Generate checklist questions
    questions_str = generate_checklist_questions(user_code)
    print("\nGenerated checklist questions:\n", questions_str)
    questions = [q.strip().split('.', 1)[1].strip() for q in questions_str.split('\n') if q.strip() and q[0].isdigit()]

    # Step 2: Search guidelines for each question
    hints = get_guideline_hints(questions, index, vectors, index_data, top_k=1)

    # Step 3: Generate user hint
    user_hint = generate_user_hint(user_code, hints)
    print("\nUser Hint:\n", user_hint)

if __name__ == "__main__":
    main()
