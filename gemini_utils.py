# geminijudge/gemini_utils.py
import streamlit as st
import google.generativeai as genai
import time
from io import BytesIO
import os
from typing import List, Optional, Any
import traceback

# --- Конфигурация Gemini ---
DEFAULT_GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
GENERATION_CONFIG = {
    "temperature": 0.7,
}

# --- Логирование ---
def log_info(message: str):
    if "log_messages" not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append(("info", message))

def log_success(message: str):
    if "log_messages" not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append(("success", message))

def log_warning(message: str):
    if "log_messages" not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append(("warning", message))

def log_error(message: str):
    if "log_messages" not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append(("error", message))

# --- Функции для работы с Gemini ---
def configure_gemini_api() -> bool:
    api_key = st.session_state.get("api_key_input")
    if not api_key:
        log_error("API ключ Google AI не предоставлен.")
        st.session_state.gemini_configured = False
        return False
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
        log_success("Gemini API успешно сконфигурирован.")
        return True
    except Exception as e:
        log_error(f"Ошибка конфигурации Gemini API: {e}")
        st.session_state.gemini_configured = False
        return False

def get_gemini_model() -> Optional[genai.GenerativeModel]:
    if not st.session_state.get("gemini_configured", False):
        # Попытка авто-конфигурации, если ключ есть, но флаг не выставлен
        if st.session_state.get("api_key_input") and not configure_gemini_api():
             log_error("Авто-конфигурация Gemini API не удалась.")
             return None
        elif not st.session_state.get("api_key_input"):
            # Это сообщение уже будет в configure_gemini_api, но можно оставить для ясности
            # log_error("Gemini API не сконфигурирован: отсутствует API ключ.")
            return None # configure_gemini_api уже залогирует ошибку
            
    model_name = os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL_NAME)
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings=SAFETY_SETTINGS,
            generation_config=GENERATION_CONFIG
        )
        # log_info(f"Модель Gemini '{model_name}' инициализирована.") # Можно убрать для минимизации логов
        return model
    except Exception as e:
        log_error(f"Ошибка инициализации модели Gemini '{model_name}': {e}")
        return None

def upload_file_to_gemini(
    uploaded_file_st_obj: st.runtime.uploaded_file_manager.UploadedFile,
    display_name_prefix: str = "doc"
) -> Optional[genai.types.File]:
    model = get_gemini_model() # Проверка конфигурации косвенно здесь
    if not model: 
        log_error("Невозможно загрузить файл: модель Gemini не инициализирована или API не сконфигурирован.")
        return None

    file_display_name = f"{display_name_prefix}_{uploaded_file_st_obj.name}"
    log_info(f"Загрузка файла: {file_display_name} ({uploaded_file_st_obj.type})")
    
    file_bytes = uploaded_file_st_obj.getvalue()
    
    try:
        gemini_file = genai.upload_file(
            path=BytesIO(file_bytes),
            display_name=file_display_name,
            mime_type=uploaded_file_st_obj.type
        )
        log_info(f"Файл '{gemini_file.display_name}' (ID: {gemini_file.name}) отправлен на сервер. Ожидание обработки...")

        delay_seconds = 4 # Немного уменьшим задержку
        max_retries = 10  # 10 * 4 = 40 секунд
        retries = 0
        while gemini_file.state.name == "PROCESSING" and retries < max_retries:
            time.sleep(delay_seconds)
            gemini_file = genai.get_file(name=gemini_file.name) 
            log_info(f"Статус '{gemini_file.display_name}': {gemini_file.state.name} ({retries+1})")
            retries += 1
        
        if gemini_file.state.name == "ACTIVE":
            log_success(f"Файл '{gemini_file.display_name}' активен.")
            return gemini_file
        elif gemini_file.state.name == "PROCESSING":
            log_warning(f"Файл '{gemini_file.display_name}' еще обрабатывается. Это может повлиять на результат.")
            return gemini_file 
        else: 
            error_message = f"Ошибка обработки файла '{gemini_file.display_name}'. Состояние: {gemini_file.state.name}."
            try: # Более безопасный способ получить детали ошибки
                if hasattr(gemini_file, 'error') and gemini_file.error:
                    error_details = getattr(gemini_file.error, 'message', str(gemini_file.error))
                    error_message += f" Детали: {error_details}"
                elif hasattr(gemini_file, 'file_error') and gemini_file.file_error:
                    error_details = getattr(gemini_file.file_error, 'message', str(gemini_file.file_error))
                    error_message += f" Детали: {error_details}"
            except Exception as e_detail:
                log_warning(f"Не удалось получить детали ошибки для файла: {e_detail}")
            log_error(error_message)
            return None

    except Exception as e:
        log_error(f"Исключение при загрузке/обработке '{file_display_name}': {type(e).__name__} - {e}")
        # log_error(f"Полный стектрейс: {traceback.format_exc()}") # Можно раскомментировать для глубокой отладки
        return None

def generate_text_from_model(
    prompt_text: str,
    files_for_context: Optional[List[genai.types.File]] = None
) -> Optional[str]:
    model = get_gemini_model()
    if not model:
        log_error("Генерация невозможна: модель Gemini не инициализирована.")
        return None

    request_parts: List[Any] = [prompt_text]
    active_files_for_request = []
    if files_for_context:
        active_files_for_request = [f for f in files_for_context if f and hasattr(f, 'state') and f.state.name == "ACTIVE"]
        if len(active_files_for_request) != len(files_for_context):
            log_warning("Не все предоставленные файлы были активны. Используются только активные.")
        if not active_files_for_request and files_for_context: # Были файлы, но ни один не активен
            log_warning("Контекстные файлы были предоставлены, но ни один из них не активен. Генерация без файлового контекста.")
            # Можно решить прервать генерацию, если файлы обязательны
            # return None 
    
    log_info(f"Запрос к модели. Промпт: {len(prompt_text)} симв. Файлов: {len(active_files_for_request)}.")
    request_parts.extend(active_files_for_request)

    try:
        response = model.generate_content(request_parts)

        if not response.parts: # Проверяем, есть ли вообще части в ответе
            block_reason = "Причина неизвестна (ответ пуст)"
            finish_reason_val = "Причина неизвестна"
            
            # Пытаемся получить больше информации из prompt_feedback и candidates
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason_val = candidate.finish_reason.name
                if finish_reason_val != "STOP" and finish_reason_val != "MAX_TOKENS": # Добавляем MAX_TOKENS как валидную причину остановки
                     block_reason += f" (Finish Reason: {finish_reason_val})"
                # Можно также проверить safety_ratings, если они есть
                # if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                #     ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in candidate.safety_ratings])
                #     block_reason += f" [Safety: {ratings_str}]"
            
            log_warning(f"Gemini вернул пустой ответ. {block_reason}")
            return None
        
        generated_text = response.text 
        log_success(f"Gemini сгенерировал ответ ({len(generated_text)} симв.).")
        return generated_text

    except Exception as e:
        log_error(f"Ошибка при генерации контента Gemini: {type(e).__name__} - {e}")
        # log_error(f"Полный стектрейс: {traceback.format_exc()}")
        return None