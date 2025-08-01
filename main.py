import os
import json
import re
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import io
from pydantic import BaseModel

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found.")
os.environ["GOOGLE_API_KEY"] = google_api_key

# Khởi tạo Gemini client
genai.configure(api_key=google_api_key)

# --- 1. Define file paths ---
pdf_path = "data/trang-15-18.pdf"
ocr_cache_file = pdf_path + ".ocr_cache.json"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Error: Document '{pdf_path}' not found. Please check the path and filename.")


# -------------------------------------- Helper functions -------------------------------------------
def clean_text(text):
    """Clean text by removing noise and normalizing whitespace."""
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)  # Combine split words
    text = re.sub(r'[^\w\s\.\!\?]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'(?<![\.\!\?])\n', ' ', text)  # Remove \n not after . ! ?
    text = re.sub(r'\n{2,}', '\n', text).strip()  # Normalize multiple newlines
    return text


def filter_header_footer(ocr_text, previous_text=None):
    """Filter out header and footer based on keywords and previous page content."""
    lines = ocr_text.split('\n')
    if len(lines) > 2:
        if any("Section" in line or "Chapter" in line or "Copyright" in line for line in lines[:2]):
            lines = lines[2:]
        if any("Copyright" in line for line in lines[-2:]):
            lines = lines[:-2]
    if previous_text:
        common_start = os.path.commonprefix([ocr_text, previous_text])
        if len(common_start) > 20:
            lines = lines[len(common_start.split('\n')):]
        common_end = os.path.commonprefix([ocr_text[::-1], previous_text[::-1]])
        if len(common_end) > 10:
            lines = lines[:-len(common_end.split('\n')[::-1])]
    return '\n'.join(lines)


def extract_text_from_json(ocr_result):
    """Extract text from various JSON structures returned by Gemini."""
    if not isinstance(ocr_result, dict):
        return ""
    text_parts = []

    def recursive_extract(obj):
        if isinstance(obj, str):
            text_parts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                recursive_extract(value)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)

    recursive_extract(ocr_result)
    return " ".join(text_parts).strip()


def _read_document_with_gemini_ocr(pdf_path: str) -> list[Document]:
    """Reads a PDF document, extracts text with PyPDFLoader, and uses Gemini for OCR on pages with potential image content."""
    all_pages_langchain_documents = []
    previous_text = None

    doc = fitz.open(pdf_path)
    total_pages_in_pdf = len(doc)

    loader = PyPDFLoader(pdf_path)
    pypdf_documents = loader.load()

    if len(pypdf_documents) < total_pages_in_pdf:
        for _ in range(total_pages_in_pdf - len(pypdf_documents)):
            pypdf_documents.append(Document(page_content="", metadata={}))

    # Khởi tạo Gemini model
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    for i in range(total_pages_in_pdf):
        print(f"Processing page {i + 1}/{total_pages_in_pdf}")

        page = doc[i]
        page_height = page.rect.height
        clip = fitz.Rect(0, 100, page.rect.width, page_height - 100)  # Remove header/footer
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        page_content = pypdf_documents[i].page_content
        page_metadata = pypdf_documents[i].metadata

        # Perform OCR với Gemini
        try:
            response = gemini_model.generate_content(
                [
                    "Extract all text from this image and return it in a structured JSON format with a single field 'page_content' containing all the text.",
                    {"mime_type": "image/png", "data": img_data}
                ],
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            ocr_result = json.loads(response.text)
            if isinstance(ocr_result, dict) and "page_content" in ocr_result:
                ocr_text = ocr_result["page_content"]
            else:
                ocr_text = extract_text_from_json(ocr_result)  # Fallback to extract text from any JSON structure
                print(f"Unexpected OCR response format on page {i + 1}, extracted text: {ocr_text[:50]}...")

            if ocr_text and len(ocr_text) > 20:
                page_content = page_content + "\n" + ocr_text if page_content else ocr_text
            else:
                print(f"OCR text too short or empty on page {i + 1}: {ocr_text}")
        except Exception as ocr_e:
            print(f"Gemini OCR error on page {i + 1}: {ocr_e}")

        page_content = filter_header_footer(page_content, previous_text)
        previous_text = page_content
        page_content = clean_text(page_content)
        page_metadata.update({'source': pdf_path, 'page': i, 'page_label': str(i + 1)})

        all_pages_langchain_documents.append(Document(page_content=page_content, metadata=page_metadata))

    doc.close()
    return all_pages_langchain_documents


# ========================================logic step================================================

# --- 2. Reading Document with caching ---
documents = []
if os.path.exists(ocr_cache_file):
    try:
        with open(ocr_cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in cached_data]
        print(f"Loaded {len(documents)} pages from cache.")
    except Exception as e:
        print(f"Cache load failed '{ocr_cache_file}': {e}. Performing OCR again.")
        documents = _read_document_with_gemini_ocr(pdf_path)
        cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
        with open(ocr_cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=4)
else:
    documents = _read_document_with_gemini_ocr(pdf_path)
    cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(ocr_cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, ensure_ascii=False, indent=4)
    print(f"Created cache with {len(documents)} pages.")

# --- 3. Chunk Document ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
splits = text_splitter.split_documents(documents)
print(f"Completed document preprocessing. Total chunks created: {len(splits)}.")

# --- 4. Create Embeddings and save to Vector Store ---
print("Creating embeddings and saving to Vector Store (ChromaDB)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector Store created and saved successfully.")

# --- 5. Initialize Gemini LLM ---
print("Initializing Gemini model...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- 6. Setup Retriever ---
print("Setting up Retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- 7. Building the RAG Chain ---
print("Building RAG chain...")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent assistant. Answer the question solely based on the provided context. "
               "Do not use external ascetic knowledge unless explicitly allowed. "
               "If the context is unclear, contains errors, or lacks relevant information, try to infer from valid parts. "
               "If no relevant information is available or the context is empty, say 'I don’t have sufficient information.'"),
    ("user", "Context:\n{context}\n\nQuestion: {input}")
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("RAG chain is ready.")

# --- 8. Question and answer loop ---
print("\n--- Ready for your questions! Type 'exit' to quit ---")
while True:
    user_question = input("Your question: ")
    if user_question.lower() == "exit":
        print("Thank you for using. Goodbye!")
        break

    try:
        print(f"Processing question: {user_question}")
        response = retrieval_chain.invoke({"input": user_question})
        print("\nGemini response:")
        print(response["answer"])
        print("-" * 50)
    except Exception as e:
        print(f"Error processing question: {e}")
        print("An error occurred while processing your question. Please check API key and internet connection.")