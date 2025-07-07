import cv2
import pytesseract
import json
from ultralytics import YOLO
from PIL import Image
import numpy as np



def load_model(model_path='yolov8n-seg.pt'):
    """Загружает модель YOLO"""
    return YOLO(model_path)

def detect_blocks(model, image_path):
    """Детектирует блоки на изображении"""
    results = model(image_path)
    return results

def extract_text_from_blocks(results, lang='rus+eng'):
    """Извлекает текст из обнаруженных блоков"""
    output = []
    
    for result in results:
        img = result.orig_img  #
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped_img = img[y1:y2, x1:x2]
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, lang=lang)
            output.append({
                "bbox": [x1, y1, x2, y2],
                "text": text.strip()
            })
    
    return output

def save_results(output, filename='output.json'):
    """Сохраняет результаты в JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def main():
    # Путь к изображению
    image_path = 'example.jpg'  # Замените на свой файл
    
    # 1. Загружаем модель
    model = load_model()
    
    # 2. Детектируем блоки
    results = detect_blocks(model, image_path)
    
    # 3. Извлекаем текст из блоков
    output = extract_text_from_blocks(results)
    
    # 4. Выводим и сохраняем результаты
    for i, item in enumerate(output, 1):
        print(f"Блок {i}:")
        print(f"Координаты: {item['bbox']}")
        print(f"Текст: {item['text']}\n")
    
    save_results(output)
    print("Результаты сохранены в output.json")

if __name__ == "__main__":
    main()