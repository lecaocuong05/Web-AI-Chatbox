# Hàm lấy ảnh trong PDF
import fitz
import os

def extract_images(pdf_path, output_dir):
    pdf = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok= True)
    image_paths = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        images = page.get_images(full = True)

        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(pdf, xref)

            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_path = os.path.join(
                output_dir,
                f"page_{page_index+1}_{img_index+1}.png"
            )
            pix.save(image_path)
            image_paths.append(image_path)
            pix = None
    pdf.close()
    return image_paths