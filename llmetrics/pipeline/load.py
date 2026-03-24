"""Load docs/AIEngineeringBook.pdf into ChromaDB, chunked by paragraph."""
import pypdf
import chromadb

PDF_PATH = "docs/AIEngineeringBook.pdf"
COLLECTION = "documents"  # must match CHROMA_COLLECTION in .env
CHUNK_SIZE = 500           # characters per chunk


def extract_chunks(path: str, chunk_size: int) -> list[str]:
    reader = pypdf.PdfReader(path)
    chunks = []
    buffer = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        buffer += " " + text
    # Split into fixed-size chunks with a small overlap
    words = buffer.split()
    current = []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-20:]  # 20-word overlap
            length = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    print(f"Extracting chunks from {PDF_PATH}...")
    chunks = extract_chunks(PDF_PATH, CHUNK_SIZE)
    print(f"  {len(chunks)} chunks created")

    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(COLLECTION)

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )
        print(f"  Loaded {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{COLLECTION}' now has {collection.count()} documents.")


if __name__ == "__main__":
    main()