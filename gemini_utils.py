# geminijudge/gemini_utils.py
import streamlit as st
import google.generativeai as genai
import time
from io import BytesIO
import os
from typing import List, Optional, Any
import traceback

# --- Имена моделей ---
# Можно переопределить через переменные окружения
MODEL_NAME_FOR_GENERATION_DEFAULT = "gemini-2.0-flash-lite"
MODEL_NAME_FOR_EVALUATION_DEFAULT = "gemini-2.5-flash-preview-04-17" 


SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
GENERATION_CONFIG_DEFAULTS = { # Общие настройки, можно переопределить для каждой модели
    "temperature": 0.6, # Чуть ниже для большей предсказуемости
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 4096, # Увеличим для потенциально длинных обоснований
}

# --- Логирование (без изменений) ---
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

def get_gemini_model(model_type: str = "generation") -> Optional[genai.GenerativeModel]:
    """
    Получает инициализированную модель Gemini.
    model_type: "generation" для генерации примеров, "evaluation" для оценки.
    """
    if not st.session_state.get("gemini_configured", False):
        if st.session_state.get("api_key_input") and not configure_gemini_api():
             log_error("Авто-конфигурация Gemini API не удалась.")
             return None
        elif not st.session_state.get("api_key_input"):
            return None
            
    if model_type == "generation":
        model_name = os.getenv("GEMINI_MODEL_GENERATION", MODEL_NAME_FOR_GENERATION_DEFAULT)
    elif model_type == "evaluation":
        model_name = os.getenv("GEMINI_MODEL_EVALUATION", MODEL_NAME_FOR_EVALUATION_DEFAULT)
    else:
        log_error(f"Неизвестный тип модели запрошен: {model_type}")
        return None
    
    # Можно добавить специфичные generation_config для разных моделей
    current_generation_config = GENERATION_CONFIG_DEFAULTS.copy()
    if model_type == "evaluation":
        current_generation_config["temperature"] = 0.3 # Для более точной оценки, меньше "творчества"

    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings=SAFETY_SETTINGS,
            generation_config=current_generation_config
        )
        log_info(f"Модель Gemini '{model_name}' (для {model_type}) инициализирована.")
        return model
    except Exception as e:
        log_error(f"Ошибка инициализации модели '{model_name}' (для {model_type}): {e}")
        return None

def upload_file_to_gemini(
    uploaded_file_st_obj: st.runtime.uploaded_file_manager.UploadedFile,
    display_name_prefix: str = "doc"
) -> Optional[genai.types.File]:
    # Для загрузки файла не обязательно указывать конкретную модель, т.к. это File API
    # Но конфигурация API все равно должна быть выполнена
    if not st.session_state.get("gemini_configured", False):
        log_error("Невозможно загрузить файл: Gemini API не сконфигурирован.")
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

        delay_seconds = 4 
        max_retries = 10  
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
            try:
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
        return None

def generate_text_from_model(
    prompt_text: str,
    model_type: str, # "generation" или "evaluation"
    files_for_context: Optional[List[genai.types.File]] = None
) -> Optional[str]:
    model = get_gemini_model(model_type=model_type)
    if not model:
        log_error(f"Генерация (тип: {model_type}) невозможна: модель не инициализирована.")
        return None

    request_parts: List[Any] = [prompt_text]
    active_files_for_request = []
    if files_for_context:
        active_files_for_request = [f for f in files_for_context if f and hasattr(f, 'state') and f.state.name == "ACTIVE"]
        if len(active_files_for_request) != len(files_for_context):
            log_warning(f"Для модели ({model_type}): Не все предоставленные файлы были активны. Используются только активные.")
        if not active_files_for_request and files_for_context:
            log_warning(f"Для модели ({model_type}): Контекстные файлы были предоставлены, но ни один из них не активен.")
    
    log_info(f"Запрос к модели ({model.model_name}, тип: {model_type}). Промпт: {len(prompt_text)} симв. Файлов: {len(active_files_for_request)}.")
    request_parts.extend(active_files_for_request)

    try:
        response = model.generate_content(request_parts)

        if not response.parts:
            block_reason = "Причина неизвестна (ответ пуст)"
            finish_reason_val = "Причина неизвестна"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason_val = candidate.finish_reason.name
                if finish_reason_val != "STOP" and finish_reason_val != "MAX_TOKENS":
                     block_reason += f" (Finish Reason: {finish_reason_val})"
            log_warning(f"Модель ({model.model_name}) вернула пустой ответ. {block_reason}")
            return None
        
        generated_text = response.text 
        log_success(f"Модель ({model.model_name}) сгенерировала ответ ({len(generated_text)} симв.).")
        return generated_text
    except Exception as e:
        log_error(f"Ошибка при генерации контента моделью ({model.model_name if model else 'N/A'}): {type(e).__name__} - {e}")
        return None