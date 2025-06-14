# PDF Compliance Rules Checker

Этот проект позволяет обрабатывать текст PDF-документов и проверять их на соответствие валютному и договорному комплаенсу с помощью локальной LLM-модели (через Ollama).

## ⚡️ Быстрый старт

### 1. Установите [Ollama](https://ollama.com/)

- Для Linux/macOS:
    ```sh
    curl -fsSL https://ollama.com/install.sh | sh
    ```
- Для Windows — скачайте [инсталлятор с официального сайта](https://ollama.com/download).

### 2. Скачайте нужную модель

- Запустите Ollama и скачайте модель:
    ```sh
    ollama pull llama3:70b-instruct-q2_K
    ```
  > _Это большая модель, загрузка может занять время._

### 3. Запустите Ollama

- Запустите Ollama сервер:
    ```sh
    ollama serve
    ```
  По умолчанию он будет доступен на `http://localhost:11434`.

### 4. Установите зависимости Python

- Рекомендуется использовать виртуальное окружение:
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
- Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

### 5. Запуск приложения

- Стандартный запуск (например, если используется FastAPI):
    ```sh
    uvicorn main:app --reload
    ```
  По умолчанию приложение будет работать на `http://localhost:8000`.

---

## ⚙️ Переменные окружения

Если Ollama работает на другой машине или нестандартном порту, укажите адрес Ollama в переменной окружения:
```sh
export OLLAMA_BASE_URL="http://<IP_или_HOST>:11434"