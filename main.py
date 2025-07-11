import os
import json
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

print("Đang khởi tạo ứng dụng RAG với LangChain và Gemini...")

# 1. Get a document file path
pdf_path = "data/data_document_root.pdf"
ocr_cache_file = pdf_path + ".ocr_cache.json"

if not os.path.exists(pdf_path):
    print(f"Lỗi: Không tìm thấy tài liệu '{pdf_path}'. Vui lòng kiểm tra lại đường dẫn và tên file.")
    exit()

def read_document_with_ocr_fallback(pdf_path):
    all_pages_langchain_documents = []
    reader = None

    try:
        print(f"Đang tải tài liệu từ '{pdf_path}' bằng PyPDFLoader...")
        loader = PyPDFLoader(pdf_path)
        pypdf_documents = loader.load()
        print(f"Đã trích xuất văn bản từ {len(pypdf_documents)} trang bằng PyPDFLoader.")

        doc = fitz.open(pdf_path)
        total_pages_in_pdf = len(doc)

        print(f"Tổng số trang trong PDF gốc: {total_pages_in_pdf}")

        if len(pypdf_documents) < total_pages_in_pdf:
            for _ in range(total_pages_in_pdf - len(pypdf_documents)):
                pypdf_documents.append(Document(page_content="", metadata={}))

        for i in range(total_pages_in_pdf):
            page_content = pypdf_documents[i].page_content
            page_metadata = pypdf_documents[i].metadata

            if len(page_content.strip()) < 50:
                print(f"Trang {i + 1} có ít nội dung, đang thử OCR...")
                if reader is None:
                    print("Đang khởi tạo EasyOCR (lần đầu sẽ tải model)...")
                    reader = easyocr.Reader(['en'])

                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")

                image = Image.open(io.BytesIO(img_data))
                image_np = np.array(image)

                try:
                    ocr_result = reader.readtext(image_np)
                    ocr_text = ' '.join([item[1] for item in ocr_result if item[1].strip()])
                    if ocr_text:
                        print(f"Đã OCR trang {i + 1} thành công ({len(ocr_text)} ký tự).")
                        if len(page_content.strip()) < 50:
                            pypdf_documents[i].page_content = ocr_text
                        else:
                            pypdf_documents[i].page_content = page_content + "\n" + ocr_text
                    else:
                        print(f"OCR trang {i + 1} không tìm thấy văn bản.")
                except Exception as ocr_e:
                    print(f"Lỗi khi OCR trang {i + 1}: {ocr_e}")
            else:
                print(f"Trang {i + 1} có đủ nội dung từ PyPDFLoader.")

            if 'source' not in page_metadata:
                page_metadata['source'] = pdf_path
            if 'page' not in page_metadata:
                page_metadata['page'] = i

            all_pages_langchain_documents.append(
                Document(page_content=pypdf_documents[i].page_content, metadata=page_metadata))

        doc.close()
        return all_pages_langchain_documents

    except Exception as e:
        print(f"Lỗi khi tải hoặc xử lý tài liệu: {e}")
        exit()


# --- 2. Reading Document (Logic kiểm tra cache) ---
documents = []
if os.path.exists(ocr_cache_file):
    print(f"Đang tải nội dung tài liệu từ file cache '{ocr_cache_file}'...")
    try:
        with open(ocr_cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            # Chuyển dữ liệu từ JSON thành các đối tượng Document của LangChain
            documents = [Document(page_content=item['page_content'], metadata=item['metadata'])
                         for item in cached_data]
        print(f"Đã tải {len(documents)} trang từ file cache.")
    except Exception as e:
        print(f"Lỗi khi tải file cache: {e}. Đang tiến hành OCR lại.")
        documents = read_document_with_ocr_fallback(pdf_path)
        # Lưu kết quả OCR vào cache
        cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
        with open(ocr_cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu {len(documents)} trang vào file cache '{ocr_cache_file}'.")
else:
    print("File cache không tồn tại. Đang tiến hành đọc và OCR tài liệu...")
    documents = read_document_with_ocr_fallback(pdf_path)
    # Lưu kết quả OCR vào cache sau khi xử lý
    cached_data = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(ocr_cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_data, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu {len(documents)} trang vào file cache '{ocr_cache_file}'.")

print(f"Đã có tổng cộng {len(documents)} trang tài liệu sau khi đọc và OCR (hoặc tải từ cache).")

# --- 3. Chunk Document ---
print("Đang chia nhỏ tài liệu thành các đoạn...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
chunks = text_splitter.split_documents(documents)
print(f"Đã chia {len(documents)} trang thành {len(chunks)} đoạn tài liệu.")
print(f"Tổng số đoạn tài liệu đã tạo: {len(chunks)}.")

# Bạn có thể in một vài đoạn mẫu để kiểm tra
if chunks:
    print("\nMột số đoạn mẫu:")
    doc = chunks[10]
    print(doc.page_content)
    print(f"Nguồn: {doc.metadata}")