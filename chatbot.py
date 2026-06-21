import boto3
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==========================
# STEP 1: Download PDF from S3
# ==========================

bucket_name = "avinashh-rag-s3-docs"
file_key = "s3_user_guide.pdf"

s3 = boto3.client("s3")

os.makedirs("documents", exist_ok=True)

local_file = "documents/s3_user_guide.pdf"

print("Downloading PDF from S3...")

s3.download_file(
    bucket_name,
    file_key,
    local_file
)

print("✅ PDF Downloaded Successfully!")

# ==========================
# STEP 2: Read PDF
# ==========================

print("Reading PDF...")

reader = PdfReader(local_file)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("✅ PDF Loaded Successfully!")

# ==========================
# STEP 3: Create Chunks
# ==========================

chunk_size = 1000

chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]

    if len(chunk.strip()) > 100:
        chunks.append(chunk)

print(f"✅ Total Chunks Created: {len(chunks)}")

# ==========================
# STEP 4: Load AI Model
# ==========================

print("Loading AI Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("✅ Model Loaded!")

# ==========================
# STEP 5: Create Embeddings
# ==========================

print("Creating Embeddings...")

chunk_embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("✅ Embeddings Ready!")

# ==========================
# STEP 6: Chat Loop
# ==========================

while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert question into embedding
    question_embedding = model.encode([question])

    # Calculate similarity
    similarities = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    # Best matching chunk
    best_index = np.argmax(similarities)

    answer = chunks[best_index]

    print("\n" + "=" * 60)
    print("📘 ANSWER")
    print("=" * 60)

    print(answer[:1500])

    print("\nSimilarity Score:",
          round(float(similarities[best_index]), 3))