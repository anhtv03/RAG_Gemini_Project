import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import easyocr
import fitz # PyMuPDF
from PIL import Image
import io
import numpy as np


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = google_api_key

print("Đang khởi tạo ứng dụng RAG với LangChain và Gemini...")

#1. get a document file path
pdf_path = "data/Foundations+of+SW+Testing+ISTQB+Certification+by+Erik+van+Veenendaal+2019-compressed.pdf"
if not os.path.exists(pdf_path):
    print(f"Lỗi: Không tìm thấy tài liệu '{pdf_path}'. Vui lòng kiểm tra lại đường dẫn và tên file.")
    exit()

# 2. reading document
def read_document_with_ocr_fallback(pdf_path):
    all_pages_content = []
    reader = None  # Khởi tạo EasyOCR Reader một lần duy nhất

    try:
        print(f"Đang tải tài liệu từ '{pdf_path}' bằng PyPDFLoader...")
        loader = PyPDFLoader    (pdf_path)
        pypdf_documents = loader.load()
        print(f"Đã trích xuất văn bản từ {len(pypdf_documents)} trang bằng PyPDFLoader.")

        doc = fitz.open(pdf_path)
        total_pages_in_pdf = len(doc)

        print(f"Tổng số trang trong PDF gốc: {total_pages_in_pdf}")

        for i in range(total_pages_in_pdf):
            page_content = pypdf_documents[i].page_content if i < len(pypdf_documents) else ""

            # Kiểm tra xem PyPDFLoader có trích xuất được nội dung đáng kể không
            # Một ngưỡng nhỏ để xác định liệu trang có trống rỗng hoặc chỉ có rất ít ký tự
            if len(page_content.strip()) < 50:  # Nếu văn bản trống hoặc quá ngắn, thử OCR
                print(f"Trang {i + 1} có ít nội dung, đang thử OCR...")
                if reader is None:
                    print("Đang khởi tạo EasyOCR (lần đầu sẽ tải model)...")
                    reader = easyocr.Reader(['en'])  # Chỉ khởi tạo một lần

                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Tăng độ phân giải để OCR tốt hơn
                img_data = pix.tobytes("png")

                image = Image.open(io.BytesIO(img_data))
                image_np = np.array(image)

                try:
                    ocr_result = reader.readtext(image_np)
                    ocr_text = ' '.join([item[1] for item in ocr_result if item[1].strip()])
                    if ocr_text:
                        print(f"Đã OCR trang {i + 1} thành công ({len(ocr_text)} ký tự).")
                        # Gán nội dung OCR vào page_content. Bạn có thể chọn cách kết hợp (nối thêm hoặc thay thế)
                        # Ở đây, tôi sẽ thay thế nếu OCR có nội dung đáng kể và PyPDFLoader không có.
                        if len(page_content.strip()) < 50:  # Nếu PyPDFLoader không có, dùng OCR
                            pypdf_documents[i].page_content = ocr_text
                        else:  # Nếu PyPDFLoader có một chút nhưng OCR tốt hơn (tùy bạn quyết định)
                            pypdf_documents[i].page_content = page_content + "\n" + ocr_text  # Nối thêm
                    else:
                        print(f"OCR trang {i + 1} không tìm thấy văn bản.")
                except Exception as ocr_e:
                    print(f"Lỗi khi OCR trang {i + 1}: {ocr_e}")
            else:
                print(f"Trang {i + 1} có đủ nội dung từ PyPDFLoader.")

        doc.close()
        return pypdf_documents

    except Exception as e:
        print(f"Lỗi khi tải hoặc xử lý tài liệu: {e}")
        exit()


documents = read_document_with_ocr_fallback(pdf_path)
print(f"Đã có tổng cộng {len(documents)} trang tài liệu sau khi đọc và OCR.")


#3. chuck document
# print("Đang chia nhỏ tài liệu thành các đoạn...")
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     add_start_index=True,
# )
# chucks   = text_splitter.split_documents(documents)
# print(f"Split {len(documents)} document into {len(chucks)} chunks.")
# print(f"Đã tạo {len(chucks)} đoạn tài liệu.")

