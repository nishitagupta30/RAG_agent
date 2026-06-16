# Step 1 — Load and chunk the document

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

# This is paragraph chunking
# This is the simplest chunking strategy — paragraph based. In real systems there are smarter strategies like:
# Paragraph chunking - Split by blank lines (what we just did)
# Fixed size chunking - Every N words/characters
# Semantic chunking - Split by meaning using an AI model
# Sliding window - Overlapping chunks so context isn't lost at boundaries