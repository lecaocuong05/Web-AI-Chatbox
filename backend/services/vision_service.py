from ollama import chat

VISION_PROMPT = """
Bạn là hệ thống OCR tài liệu.

NHIỆM VỤ:

Đầu tiên hãy tự xác định ảnh thuộc loại nào sau đây:
- Bảng dữ liệu (TABLE)
- Sơ đồ/Lưu đồ (FLOWCHART)
- Ảnh thông thường (IMAGE)

KHÔNG cần trả lời loại ảnh.

========================
1. Nếu là BẢNG DỮ LIỆU

- OCR toàn bộ bảng.
- Xuất dưới dạng Markdown.
- Giữ nguyên hàng và cột.
- Không bỏ sót dòng.
- Không bỏ sót cột.
- Không giải thích.
- Không tóm tắt.
- Không diễn giải.

Ví dụ:
| STT | Tên | Kiểu |
|-----|-----|------|
| 1 | Operation | Int(4) |

========================
2. Nếu là SƠ ĐỒ hoặc LƯU ĐỒ
Không được tóm tắt.
Hãy chuyển sơ đồ thành văn bản có cấu trúc.
Theo đúng mẫu:
Thành phần:
- ...
Luồng xử lý:
1. ...
2. ...
3. ...
Nếu có điều kiện rẽ nhánh thì ghi rõ.
Nếu có mũi tên thì giữ đúng thứ tự của mũi tên.
Không tự suy diễn.
Không bổ sung kiến thức ngoài ảnh.
========================
3. Nếu là ẢNH THÔNG THƯỜNG
OCR toàn bộ chữ nhìn thấy.
Sau đó mô tả ngắn gọn nội dung ảnh.
Không suy diễn.
========================
4. Nếu chỉ là logo, icon hoặc hình trang trí
Trả lời đúng:
Ảnh này không chứa thông tin có giá trị tra cứu.
========================

QUY TẮC CHUNG
- Không bịa thêm.
- Không sử dụng kiến thức bên ngoài.
- Chỉ được mô tả những gì nhìn thấy.
- Không thêm lời mở đầu.
- Không thêm lời kết.
"""

def describe_image(image_path):
    try:
        print("===== QWEN-VL BẮT ĐẦU =====")
        response = chat(
            model="qwen2.5vl:32b",
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [image_path]
                }
            ]
        )
        print("===== QWEN-VL XONG =====")
        text = response["message"]["content"]
        print("=" * 80)
        print("QWEN-VL OCR RESULT")
        print(text)
        print("=" * 80)
        with open("vision_output.txt", "w", encoding="utf8") as f:
            f.write(text)
        return text
    except Exception as e:
        print("Vision Error:", e)
        return ""