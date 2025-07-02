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
                matrix[i][j] = 1 + min(matrix[i][j-1], matrix[i-1][j], matrix[i-1][j-1])
    
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

def get_text(rez, words_list):
    if isinstance(rez, dict):
        if "Text" in rez:
            words_list.extend(re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', rez["Text"]))
        for value in rez.values():
            get_text(value, words_list)
    elif isinstance(rez, list):
        for item in rez:
            get_text(item, words_list)

text_words = []
get_text(rez, text_words)

def replace(rez, original, replacement):
    if isinstance(rez, dict):
        for key, value in rez.items():
            if isinstance(value, str):
                rez[key] = value.replace(original, replacement)
            else:
                replace(value, original, replacement)
    elif isinstance(rez, list):
        for i in range(len(rez)):
            if isinstance(rez[i], str):
                rez[i] = rez[i].replace(original, replacement)
            else:
                replace(rez[i], original, replacement)

with open("russian.utf-8") as f:
    dictionary_words = [word.strip() for word in f]
    
    for target_word in text_words:
        for dict_word in dictionary_words:
            equal = lev(dict_word, target_word)
            print(f"between {dict_word} and {target_word} -> {equal}")
            if equal == 0:
                print("Same word")
                break
            if equal <= 2:
                print(f"Replacing '{target_word}' with '{dict_word}'")
                replace(rez, target_word, dict_word)
                break


with open("rez_modified.txt.webRes", 'w') as out_file:
    json.dump(rez, out_file, ensure_ascii=False, indent=2)

