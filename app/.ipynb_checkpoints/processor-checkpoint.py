
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes 
import pytesseract
from langchain.prompts import PromptTemplate 
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
import os

def process_pdf(pdf_bytes: bytes) -> dict:
    full_text = read_pdf(pdf_bytes)
    return process_text(full_text)

def process_text(file_text: str) -> dict:
    prompt = get_prompt()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = Ollama(model="llama3:70b-instruct-q2_K", base_url=base_url, num_ctx=8192)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    answer = llm_chain.run(document=file_text)
    cleaned_answer = remove_extra_text(answer)
    # Пытаемся распарсить JSON
    try:
        answer_json = json.loads(cleaned_answer)
        error = None
    except json.JSONDecodeError as e:
        answer_json = None
        error = str(e)

    # Собираем финальный ответ
    return {
        "result": answer_json,     # объект или null
        "error": error,            # строка или null
        "result_raw": answer # строка как есть
    }


# ЧИТАЕМ ДОГОВОР
def read_pdf(pdf_bytes: bytes) -> str:
    # 3. Преобразуем PDF → PIL-страницы
    try:
        pages = convert_from_bytes(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения PDF: {e}")


    FULL_TEXT = ''
    # Обрабатываем каждую страницу
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang='rus+eng')  # если нужен русский и английский
        #print(f'--- Страница {i+1} ---\n{text}\n')
        # Можно сохранить текст в файл
        with open(f'page_{i+1}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        FULL_TEXT += text
    return FULL_TEXT


def get_prompt() -> PromptTemplate:
    #СОЗДАЕМ PROMPT
    TEMPLATE = TEMPLATE = """
  Ты — JSON-extractor для валютного контроля.  
  Ответ ДОЛЖЕН быть валидным JSON-объектом, БЕЗ комментариев, Markdown-блоков, «Вот:», «```json» и т. д.  
  Первая строка ответа должна начинаться символом «{{», последняя — «}}».  
  Любой другой текст запрещён.
 
  ---

  Проанализируй один валютный договор – **полный плоский текст PDF, без исходной разметки**.

  Документ (вставлен ниже):
  {document}

  Документ может быть на русском или английском; даты и суммы встречаются в обоих форматах (дд.мм.гггг / YYYY-MM-DD, «1 000,50» / “1,000.50”).  
  В договоре могут фигурировать поставка товара **или** оказание услуг, поэтому роли сторон описываются синонимами (Поставщик / Продавец / Экспортёр — Supplier / Seller, Покупатель — Buyer, Заказчик — Customer и т. д.).

  ---

  ### Верни строго следующий JSON-объект

  ```json
  {{
  "contractNumber": string | null,    //Формулировки вида «Договор № …», «Contract No. …», «Agreement …»
  "contractDate":   string | null,    // Дата рядом с номером, фразы «от», «dated», «дата заключения», ISO-8601: YYYY-MM-DD, YYYY-MM или YYYY
  "buyer":          string | null,    //Полное юр. название покупателя (Buyer / Customer / Покупатель / Заказчик)
  "seller":         string | null,    //Полное юр. название продавца (Seller / Supplier / Поставщик / Экспортёр)
  "operationType":  "import" | "export" | null, //Определи: если резидент КZ/RU покупатель → **import**; если резидент КZ/RU продавец → **export**
  "contractAmount": number | null,    // Итоговая сумма контракта («Total amount», «Сумма договора»). Без разделителей тысяч
  "currency":       string | null,    // Код валюты (USD, EUR, KZT, RUB …) рядом с суммой
  "repatriationTerm":    string | null, //Фразы «срок репатриации», «repatriation term», «срок выполнения договори» и тд.
  "counterpartyName":    string | null, // Название иностранного контрагента (если отличается от seller/buyer)
  "counterpartyCountry": string | null,   // Страна контрагента, двухбуквенный код ISO-3166-1 (RU, KZ, CN …)
  "counterpartyBank":    string | null,   //Название банка контрагента в реквизитах («Bank», «Банк» …)
  "buyerInn":   string | null,   // поле ИНН, только если сторона-покупатель из России (RU); иначе null
  "sellerInn":  string | null    // поле ИНН, только если сторона-продавец/поставщик из России (RU); иначе null
  }}
  """

    return PromptTemplate(
        template=TEMPLATE,
        input_variables=["document"],  # единственный инпут – сам текст PDF
    )

def remove_extra_text(text: str) -> str:
    start = text.find("{")
    end   = text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 and end > start else ""
