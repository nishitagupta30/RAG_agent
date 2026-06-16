import ollama
import numpy as np

# start ollama - ollama serve
# dwnload something in ollama - ollama pull <model-name>

def ask_rag(question, chunks, chunk_embeddings):
    # Step A: Retrieve the best matching chunk
    best_chunk = find_best_chunk(question, chunks, chunk_embeddings)
    
    # Step B: Build a prompt that includes the retrieved context
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the given information."

Context:
{best_chunk}

Question: {question}

Answer:"""
    
    # Step C: Send to the LLM
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['message']['content']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_best_chunk(question, chunks, chunk_embeddings):
    # Embed the question
    question_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question)['embedding']
    
    # Compare against every chunk
    similarities = []
    for emb in chunk_embeddings:
        sim = cosine_similarity(question_embedding, emb)
        similarities.append(sim)
    
    # Find the best match
    best_index = np.argmax(similarities)
    
    print("Similarity scores:")
    for i, sim in enumerate(similarities):
        print(f"  Chunk {i+1}: {sim:.4f}")
    
    return chunks[best_index]


def load_and_chunk(filepath, chunk_size=1):
    with open(filepath, "r") as f:
        content = f.read()
    
    # Split by empty lines (paragraphs)
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    
    return chunks


# Test it
chunks = load_and_chunk("knowledge.txt")

chunk_embeddings = []
for chunk in chunks:
    response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
    chunk_embeddings.append(response['embedding'])

question = "Who leads the technology team?"
answer = ask_rag(question, chunks, chunk_embeddings)

print(f"Question: {question}")
print(f"Answer: {answer}")