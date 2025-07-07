from ultralytics import YOLO
import os
from pathlib import Path
import pytesseract
import cv2
import numpy as np
from symspellpy import SymSpell, Verbosity
import json

# Инициализация SymSpell для исправления опечаток
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_ru_82_765.txt", term_index=0, count_index=1, encoding="utf-8")

# Конфигурация
CONFIG = {
    "dataset_path": "/Users/vladislavbaanov/Desktop/practice/dataset",
    "output_dir": "Rez",
    "test_image": "/Users/vladislavbaanov/Desktop/practice/test.png"
}

def preprocess_for_ocr(image):
    """Подготовка изображения для распознавания текста"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def correct_spelling(text, lang="ru"):
    """Исправление опечаток с учетом контекста"""
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def save_results(data, filename="results.txt"):
    """Сохранение результатов в файл"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["output_dir"], filename), "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"Блок {item['block_id']}:\n")
            f.write(f"Координаты: {item['bbox']}\n")
            f.write(f"Текст: {item['text']}\n")
            f.write(f"Исправленный текст: {item['corrected_text']}\n\n")

def main():
    print("="*50)
    print("Инициализация системы распознавания текста")
    
    # Проверка датасета
    train_path = Path(os.path.join(CONFIG["dataset_path"], "images/train"))
    val_path = Path(os.path.join(CONFIG["dataset_path"], "images/val"))
    
    print(f"\nНайдено:\nТренировочных изображений: {len(os.listdir(train_path))}\nВалидационных: {len(os.listdir(val_path))}")
    print("="*50)

    # Инициализация модели YOLO
    model = YOLO('yolov8n.pt')
    
    try:
        # Обучение модели
        print("\nЗапуск обучения модели...")
        model.train(
            data=os.path.join(CONFIG["dataset_path"], "data.yaml"),
            epochs=100,
            imgsz=640,
            batch=8,
            name='my_model',
        )
        
        # Загрузка лучшей модели
        best_model = YOLO('detect/my_model/weights/best.pt')
        
        # Распознавание на тестовом изображении
        if os.path.exists(CONFIG["test_image"]):
            img = cv2.imread(CONFIG["test_image"])
            results = best_model.predict(img, save=True, conf=0.5)
            
            output_data = []
            for i, det in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                cropped = img[y1:y2, x1:x2]
                processed = preprocess_for_ocr(cropped)
                
                # Распознавание текста
                raw_text = pytesseract.image_to_string(processed, lang='rus+eng').strip()
                corrected_text = correct_spelling(raw_text)
                
                output_data.append({
                    "block_id": i+1,
                    "bbox": [x1, y1, x2, y2],
                    "text": raw_text,
                    "corrected_text": corrected_text
                })
                
                print(f"\nБлок {i+1}:")
                print(f"Координаты: ({x1}, {y1})-({x2}, {y2})")
                print(f"Исходный текст: {raw_text}")
                print(f"Исправленный текст: {corrected_text}")
            
            # Сохранение результатов
            save_results(output_data)
            print(f"\nВсе результаты сохранены в папку '{CONFIG['output_dir']}'")
            
            # Дополнительно сохраняем в JSON
            with open(os.path.join(CONFIG["output_dir"], "results.json"), "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
        else:
            print(f"\nОшибка: Тестовое изображение не найдено {CONFIG['test_image']}")
    
    except Exception as e:
        print(f"\nОшибка: {str(e)}")

if __name__ == "__main__":
    # Проверка зависимостей
    try:
        pytesseract.get_tesseract_version()
        main()
    except Exception as e:
        print(f"Требуемые компоненты не установлены: {e}")
        print("Установите Tesseract: brew install tesseract tesseract-lang")
        print("Установите SymSpell: pip install symspellpy")