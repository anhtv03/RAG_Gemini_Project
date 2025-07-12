import os
import json
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import re

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- 1. Get a document file path and define an OCR cache file path ---
pdf_path = "data/trang-8-58.pdf"
ocr_cache_file = pdf_path + ".ocr_cache.json"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Lỗi: Không tìm thấy tài liệu '{pdf_path}'. Vui lòng kiểm tra lại đường dẫn và tên file.")

# --- Helper function to read and OCR PDF ---
def _read_document_with_ocr_fallback(pdf_path: str) -> list[Document]:
    """
    Reads a PDF document, attempts to extract text with PyPDFLoader,
    and falls back to EasyOCR for pages with insufficient content.
    Caches OCR reader for efficiency and preprocesses text.
    """
    all_pages_langchain_documents = []
    reader = None  # EasyOCR Reader is initialized once

    doc = fitz.open(pdf_path)
    total_pages_in_pdf = len(doc)

    # Load initial content from PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pypdf_documents = loader.load()

    # Ensure pypdf_documents list matches total_pages_in_pdf
    if len(pypdf_documents) < total_pages_in_pdf:
        for _ in range(total_pages_in_pdf - len(pypdf_documents)):
            pypdf_documents.append(Document(page_content="", metadata={}))

    for i in range(total_pages_in_pdf):
        print(f"Đang xử lý trang {i + 1}/{total_pages_in_pdf}")
        page_content = pypdf_documents[i].page_content
        page_metadata = pypdf_documents[i].metadata

        # Attempt OCR if PyPDFLoader yields too little content
        if len(page_content.strip()) < 50:
            if reader is None:
                reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_data))
            image_np = np.array(image)

            try:
                ocr_result = reader.readtext(image_np)
                ocr_text = ' '.join([item[1] for item in ocr_result if item[1].strip()])
                if ocr_text:
                    if len(page_content.strip()) < 50:
                        page_content = ocr_text
                    else:
                        page_content = page_content + "\n" + ocr_text
            except Exception as ocr_e:
                print(f"Lỗi khi OCR trang {i + 1}: {ocr_e}")

        # Preprocess text: remove extra whitespace and handle \n intelligently
        page_content = re.sub(r'\s+', ' ', page_content.strip())  # Replace multiple whitespace with single space
        page_content = re.sub(r'(?<![\.\!\?])\n', ' ', page_content)  # Remove \n not preceded by . ! ?
        page_content = re.sub(r'\n{2,}', '\n', page_content).strip()  # Normalize multiple newlines to single

        # Ensure metadata is present for source and page number
        if 'source' not in page_metadata:
            page_metadata['source'] = pdf_path
        if 'page' not in page_metadata:
            page_metadata['page'] = i

        all_pages_langchain_documents.append(
            Document(page_content=page_content, metadata=page_metadata))

    doc.close()
    return all_pages_langchain_documents

# --- 2. Reading Document (with caching logic) ---
documents = []
if os.path.exists(ocr_cache_file):
    try:
        with open(ocr_cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            documents = [Document(page_content=item['page_content'], metadata=item['metadata'])
                         for item in cached_data]
    except Exception as e:
        print(f"Lỗi khi tải file cache '{ocr_cache_file}': {e}. Đang tiến hành OCR lại.")
        documents = _read_document_with_ocr_fallback(pdf_path)
        cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
        with open(ocr_cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=4)
else:
    documents = _read_document_with_ocr_fallback(pdf_path)
    cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(ocr_cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, ensure_ascii=False, indent=4)

# --- 3. Chunk Document ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
splits = text_splitter.split_documents(documents)
print(f"Đã hoàn tất tiền xử lý tài liệu. Tổng số đoạn tài liệu đã tạo: {len(splits)}.")

# 4. Create Embeddings and save to Vector Store (ChromaDB)
print("Đang tạo embeddings và lưu vào Vector Store (ChromaDB)...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Đường dẫn lưu trữ database
)
print("Đã tạo và lưu Vector Store thành công.")

# 5. Initialize Gemini's Large Language Model (LLM)
print("Đang khởi tạo mô hình Gemini...")
# llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

# 6. Setup Retriever
print("Đang thiết lập Retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Lấy 3 đoạn tài liệu liên quan nhất

# 7. Building the RAG Chain (Retrieval Chain)
print("Đang xây dựng chuỗi RAG...")
# Định nghĩa prompt cho Gemini
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một trợ lý thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
               "Nếu bạn không biết câu trả lời từ ngữ cảnh, hãy nói rằng bạn không có thông tin."),
    ("user", "Ngữ cảnh:\n{context}\n\nCâu hỏi: {input}")
])

# Tạo chain để kết hợp tài liệu được truy xuất với prompt và LLM
document_chain = create_stuff_documents_chain(llm, prompt)

# Tạo chain tổng thể bao gồm truy xuất và tạo câu trả lời
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("Chuỗi RAG đã sẵn sàng.")

# 8. Question and answer loop
print("\n--- Sẵn sàng nhận câu hỏi của bạn! Gõ 'thoat' để thoát ---")
while True:
    user_question = input("Bạn hỏi: ")
    if user_question.lower() == "thoat":
        print("Cảm ơn bạn đã sử dụng. Tạm biệt!")
        break

    try:
        print("Đang xử lý câu hỏi của bạn...")
        response = retrieval_chain.invoke({"input": user_question})
        print("\nGemini trả lời:")
        print(response["answer"])
        print("-" * 50)
    except Exception as e:
        print(f"Đã xảy ra lỗi khi xử lý câu hỏi: {e}")
        print("Vui lòng kiểm tra lại API Key và kết nối internet.")