import ollama
import numpy as np

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

print(f"Total chunks: {len(chunks)}")
print()
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    print("---")

chunk_embeddings = []
for chunk in chunks:
    response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
    chunk_embeddings.append(response['embedding'])

print(f"Number of embeddings: {len(chunk_embeddings)}")
print(f"Embedding size: {len(chunk_embeddings[0])}")
print(f"First 5 numbers: {chunk_embeddings[0][:5]}")

question = "Who leads the technology team?"
best_chunk = find_best_chunk(question, chunks, chunk_embeddings)

print(f"\nQuestion: {question}")
print(f"Best matching chunk: {best_chunk}")