import json
import re

def lev(str_1, str_2):
    n, m = len(str_1), len(str_2)
    if n > m:
        str_1, str_2 = str_2, str_1
        n, m = m, n
    matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        matrix[i][0] = i
    for j in range(n + 1):
        matrix[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str_2[i - 1] == str_1[j - 1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = 1 + min(matrix[i][j-1],matrix[i-1][j],matrix[i-1][j-1])
    
    return int(matrix[m][n])

with open("rez.txt.webRes") as rez:
    rez = json.load(rez)

def extract_text(rez):
    if isinstance(rez, dict):
        if "Text" in rez:
            text = rez["Text"]
            cleaned_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ ]', '', text)
            print(cleaned_text)
        for value in rez.values():
            extract_text(value)
    elif isinstance(rez, list):
        for item in rez:
            extract_text(item)

extract_text(rez)

def get_text_words_from_json(rez, words_list):
    if isinstance(rez, dict):
        if "Text" in rez:
            words_list.extend(re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', rez["Text"]))
        for value in rez.values():
            get_text_words_from_json(value, words_list)
    elif isinstance(rez, list):
        for item in rez:
            get_text_words_from_json(item, words_list)


text_words = []
get_text_words_from_json(rez, text_words)

def replace_word_in_json(rez, original, replacement):
    if isinstance(rez, dict):
        for key, value in rez.items():
            if isinstance(value, str):
                rez[key] = value.replace(original, replacement)
            else:
                replace_word_in_json(value, original, replacement)
    elif isinstance(rez, list):
        for i in range(len(rez)):
            if isinstance(rez[i], str):
                rez[i] = rez[i].replace(original, replacement)
            else:
                replace_word_in_json(rez[i], original, replacement)
                
with open("russian.utf-8") as f:
    for n in text_words:
        target_word = n
    
    for word in f:
        word1 = word.strip()
        equal = lev(word1,target_word)
        print(f"between {word1} and {target_word} -> {equal}")
        if equal == 0:
            print("Same word")
            break

        if equal <= 2:
            break