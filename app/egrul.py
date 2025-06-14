import fitz  # pymupdf
import re
import requests
import time
import json
from typing import List, Dict



BASE_URL = "https://egrul.nalog.ru"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
session = requests.Session()


# --- Работа с API ФНС ---
def search(query):
    response = session.post(f"{BASE_URL}/", headers=HEADERS, data={"query": query})
    response.raise_for_status()
    data = response.json()
    print(f"🔍 Поиск: {query} → Ответ: {data}")
    return data["t"]

def wait_for_result(t):
    poll_url = f"{BASE_URL}/search-result/{t}"
    while True:
        r = session.get(poll_url)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "wait":
            time.sleep(1)
        elif data.get("rows"):
            return data["rows"][0]["t"]
        else:
            raise Exception("Ошибка: пустой ответ или неизвестный статус")

def request_vyp(t):
    url = f"{BASE_URL}/vyp-request/{t}"
    r = session.get(url)
    r.raise_for_status()
    return r.json()["t"]

def download_pdf(t):
    url = f"{BASE_URL}/vyp-download/{t}"
    r = session.get(url)
    r.raise_for_status()
    return r.content

def get_pdf_by_inn_or_name(query):
    try:
        t1 = search(query)
        t2 = wait_for_result(t1)
        t3 = request_vyp(t2)
        return download_pdf(t3)
    except Exception as e:
        print(f"[!] Ошибка при получении PDF по запросу {query}: {e}")
        return None


# --- Парсинг PDF ---
def parse_pdf_to_lines(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)
    lines = []
    for page in doc:
        lines.extend(page.get_text().splitlines())
    return lines

def find_inn_above_or_below(lines, index, lookahead=30):
    # Вниз
    for j in range(index, min(index + lookahead, len(lines))):
        match_inline = re.search(r'ИНН.*?(\d{10,12})', lines[j])
        if match_inline:
            return match_inline.group(1)
        if lines[j].strip().upper() == "ИНН" and j + 1 < len(lines):
            match = re.search(r'\d{10,12}', lines[j + 1])
            if match:
                return match.group()

    # Вверх
    for j in range(index - 1, max(index - lookahead, -1), -1):
        match_inline = re.search(r'ИНН.*?(\d{10,12})', lines[j])
        if match_inline:
            return match_inline.group(1)
        if lines[j].strip().upper() == "ИНН" and j + 1 < len(lines):
            match = re.search(r'\d{10,12}', lines[j + 1])
            if match:
                return match.group()

    return None

def find_share_nearby(lines, start_index):
    for j in range(start_index, min(start_index + 30, len(lines))):
        if "Номинальная стоимость доли" in lines[j]:
            if j + 1 < len(lines):
                match = re.search(r'\d+', lines[j + 1])
                if match:
                    return match.group()
    return None

def extract_fio_block(lines, index):
    try:
        return f"{lines[index + 3].strip()} {lines[index + 4].strip()} {lines[index + 5].strip()}"
    except IndexError:
        return None

def extract_owners_from_pdf(pdf_bytes, level=0, visited_inn=None):
    visited_inn = visited_inn or set()
    owners = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    lines = [line.strip() for page in doc for line in page.get_text().splitlines()]

    i = 0
    while i < len(lines):
        # 1. Физическое лицо
        if lines[i].upper() == "ФАМИЛИЯ" and i + 5 < len(lines):
            fio = f"{lines[i+1]} {lines[i+3]} {lines[i+5]}"
            share = find_share_nearby(lines, i)
            owners.append({"ФИО": fio, "ИНН": None, "Доля (руб)": share})
            print(f"{'  '*level}👤 Найден физ.лицо: {fio}, Доля: {share}")
            i += 6
            continue

        # 2. Юрлицо — ищем по словам "ООО", "АО", "ПАО"
        if re.search(r'\b(ООО|АО|ПАО)\b', lines[i]):
            org_name = lines[i]
            inn = find_inn_above_or_below(lines, i)
            print(f"{'  '*level}🏢 Найдено юрлицо: {org_name}, ИНН: {inn}")
            if inn and inn not in visited_inn:
                visited_inn.add(inn)
                print(f"{'  '*level}🔁 Запрашиваем выписку по ИНН {inn} ({org_name})")
                child_pdf = get_pdf_by_inn_or_name(inn)
                if child_pdf:
                    child_owners = extract_owners_from_pdf(child_pdf, level+1, visited_inn)
                    owners.extend(child_owners)
                else:
                    print(f"{'  '*level}⚠️ Не удалось получить PDF по ИНН {inn}")
            else:
                print(f"{'  '*level}↪️ Пропускаем ИНН {inn} (уже обработан)")
        i += 1

    print(f"{'  '*level}✅ Найдено физических лиц на этом уровне: {len(owners)}")
    return owners

def get_owners(bin: str) -> List[Dict]:
    pdf = get_pdf_by_inn_or_name(bin)
    owners = extract_owners_from_pdf(pdf)
    return owners 