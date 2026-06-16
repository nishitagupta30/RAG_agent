import ollama


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