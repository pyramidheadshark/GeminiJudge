# geminijudge/app.py
import streamlit as st
from dotenv import load_dotenv
import os

import prompts
import gemini_utils

load_dotenv()

def initialize_session_state():
    defaults = {
        "api_key_input": os.getenv("GOOGLE_API_KEY", ""),
        "gemini_configured": False,
        "uploaded_st_files": [],
        "processed_gemini_files": [],
        "user_prompt_input": "",
        "model_a_response_input": "",
        "num_incorrect_samples_input": 2, # Уменьшим по умолчанию, т.к. оценка сложнее
        "generated_incorrect_responses": [],
        "evaluation_result_id": None,
        "evaluation_rationale": "", # Для хранения обоснования
        "all_responses_for_evaluation": {},
        "log_messages": [],
        "app_run_id": 0,
        "processing_complete": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.api_key_input and not st.session_state.gemini_configured:
        gemini_utils.configure_gemini_api()

initialize_session_state()

# --- Логика Приложения ---
def handle_file_uploads_and_processing(uploaded_st_files_list: list) -> bool:
    st.session_state.processed_gemini_files = []
    if not uploaded_st_files_list:
        gemini_utils.log_info("Контекстные файлы не предоставлены.")
        return True
    all_successful = True
    for i, st_file in enumerate(uploaded_st_files_list):
        gemini_utils.log_info(f"Обработка файла {i+1}/{len(uploaded_st_files_list)}: {st_file.name}")
        gemini_file_obj = gemini_utils.upload_file_to_gemini(st_file, display_name_prefix=f"doc{i+1}")
        if gemini_file_obj and hasattr(gemini_file_obj, 'state') and gemini_file_obj.state.name == "ACTIVE":
            st.session_state.processed_gemini_files.append(gemini_file_obj)
        elif gemini_file_obj:
            gemini_utils.log_warning(f"Файл {st_file.name} загружен, но не активен (состояние: {gemini_file_obj.state.name}).")
            all_successful = False 
        else:
            gemini_utils.log_error(f"Не удалось обработать файл {st_file.name}.")
            all_successful = False
    if all_successful and st.session_state.processed_gemini_files:
        gemini_utils.log_success(f"{len(st.session_state.processed_gemini_files)} файлов успешно подготовлены.")
    elif st.session_state.processed_gemini_files:
         gemini_utils.log_warning(f"Подготовлено {len(st.session_state.processed_gemini_files)} из {len(uploaded_st_files_list)} файлов.")
    elif uploaded_st_files_list:
        gemini_utils.log_error("Ни один из файлов не был успешно обработан.")
        all_successful = False
    return all_successful

def generate_and_parse_incorrect_responses_logic(user_prompt: str, model_a_response: str, num_samples: int) -> bool:
    st.session_state.generated_incorrect_responses = []
    prompt_text = prompts.get_generate_incorrect_answers_prompt(user_prompt, model_a_response, num_samples)
    
    raw_response = gemini_utils.generate_text_from_model(
        prompt_text, model_type="generation", # Используем модель для генерации
        files_for_context=st.session_state.processed_gemini_files
    )
    if not raw_response:
        gemini_utils.log_error("Не получены 'неправильные' ответы от модели генерации.")
        return False

    parsed_responses = []
    current_answer_accumulator = []
    for line in raw_response.splitlines():
        if line.startswith(prompts.INCORRECT_ANSWER_PARSING_PREFIX):
            if current_answer_accumulator:
                parsed_responses.append("\n".join(current_answer_accumulator).strip())
            current_answer_accumulator = [line.replace(prompts.INCORRECT_ANSWER_PARSING_PREFIX, "", 1).strip()]
        elif current_answer_accumulator:
            current_answer_accumulator.append(line.strip())
    if current_answer_accumulator:
        parsed_responses.append("\n".join(current_answer_accumulator).strip())
    st.session_state.generated_incorrect_responses = [resp for resp in parsed_responses if resp]

    if not st.session_state.generated_incorrect_responses:
        gemini_utils.log_warning("Не удалось извлечь 'неправильные' ответы.")
        return False
    gemini_utils.log_success(f"Извлечено {len(st.session_state.generated_incorrect_responses)} 'неправильных' ответов.")
    return True

def evaluate_all_responses_logic(user_prompt: str, model_a_response: str) -> bool:
    st.session_state.evaluation_result_id = None
    st.session_state.evaluation_rationale = "" # Сброс обоснования
    st.session_state.all_responses_for_evaluation = {}

    all_responses_dict = {prompts.MODEL_A_ANSWER_ID: model_a_response}
    for i, resp_text in enumerate(st.session_state.generated_incorrect_responses):
        all_responses_dict[prompts.get_incorrect_answer_id(i)] = resp_text
    st.session_state.all_responses_for_evaluation = all_responses_dict

    text_block_for_prompt = ""
    for identifier, text in all_responses_dict.items():
        text_block_for_prompt += f"{identifier}:\n{text}\n---\n"
    
    prompt_text = prompts.get_evaluate_responses_prompt(user_prompt, text_block_for_prompt)
    
    full_evaluation_response = gemini_utils.generate_text_from_model(
        prompt_text, model_type="evaluation", # Используем модель для оценки
        files_for_context=st.session_state.processed_gemini_files
    )

    if not full_evaluation_response:
        gemini_utils.log_error("Не получен ответ от оценочной модели.")
        return False
    
    parts = full_evaluation_response.split(prompts.EVALUATION_SECTION_DELIMITER, 1)
    chosen_id_raw = parts[0].strip()
    rationale_text = parts[1].strip() if len(parts) > 1 else ""

    chosen_id = chosen_id_raw.split()[0] if chosen_id_raw else ""

    if chosen_id in all_responses_dict:
        st.session_state.evaluation_result_id = chosen_id
        st.session_state.evaluation_rationale = rationale_text
        gemini_utils.log_success(f"Оценочная модель выбрала ID: '{chosen_id}'. Обоснование получено.")
        if not rationale_text:
            gemini_utils.log_warning("Обоснование от оценочной модели пустое, хотя ID выбран.")
    else:
        gemini_utils.log_warning(f"Оценочная модель вернула '{chosen_id_raw}', ID не распознан.")
        # Попытка найти по тексту (менее надежно, но как запасной вариант)
        found_by_text = False
        for id_key, text_val in all_responses_dict.items():
            # Сравниваем только первую часть ответа модели (до разделителя) с текстами кандидатов
            if chosen_id_raw.strip() == text_val.strip(): 
                st.session_state.evaluation_result_id = id_key
                st.session_state.evaluation_rationale = rationale_text # Все равно сохраняем обоснование, если есть
                gemini_utils.log_warning(f"ID '{id_key}' определен по совпадению текста ответа (опасно).")
                found_by_text = True
                break
        if not found_by_text:
            st.session_state.evaluation_rationale = full_evaluation_response # Сохраняем весь ответ как "обоснование" для отладки
            return False
            
    return True

# --- UI: Боковая Панель (Конфигурация и Ввод) ---
with st.sidebar:
    st.header("⚖️ GeminiJudge")
    st.caption("v2.0 - Двухмодельная оценка с обоснованием")

    with st.expander("🔑 API Ключ Google AI", expanded=not st.session_state.gemini_configured):
        # ... (код для API ключа без изменений) ...
        api_key_val = st.text_input(
            "Ваш API ключ:", type="password", value=st.session_state.api_key_input, 
            key=f"api_key_field_{st.session_state.app_run_id}", help="Можно установить через GOOGLE_API_KEY в .env"
        )
        if api_key_val != st.session_state.api_key_input:
            st.session_state.api_key_input = api_key_val
            st.session_state.gemini_configured = False 
            gemini_utils.configure_gemini_api()
            st.rerun()
        if st.session_state.gemini_configured: st.success("API готов.")
        elif st.session_state.api_key_input: st.error("API не настроен. Проверьте ключ.")


    st.session_state.uploaded_st_files = st.file_uploader(
        "1. Контекстные документы:", accept_multiple_files=True, key=f"file_uploader_{st.session_state.app_run_id}"
    )
    
    st.session_state.user_prompt_input = st.text_area(
        "2. Ваш Промпт:", value=st.session_state.user_prompt_input, height=100, key=f"user_prompt_area_{st.session_state.app_run_id}"
    )
    st.session_state.model_a_response_input = st.text_area(
        "3. Ответ 'Модели А':", value=st.session_state.model_a_response_input, height=100, key=f"model_a_response_area_{st.session_state.app_run_id}"
    )
    st.session_state.num_incorrect_samples_input = st.number_input(
        "4. Кол-во 'неправильных' примеров:", min_value=1, max_value=4, # Макс 4 для компактности UI
        value=st.session_state.num_incorrect_samples_input, step=1, key=f"num_incorrect_{st.session_state.app_run_id}"
    )

    run_button_disabled = not st.session_state.gemini_configured or \
                          not st.session_state.user_prompt_input.strip() or \
                          not st.session_state.model_a_response_input.strip()

    if st.button("🚀 Запустить Оценку", type="primary", use_container_width=True, disabled=run_button_disabled, key=f"run_button_{st.session_state.app_run_id}"):
        st.session_state.app_run_id += 1
        st.session_state.log_messages = [] 
        st.session_state.processing_complete = False
        gemini_utils.log_info("=== Новый сеанс GeminiJudge ===")
        # Флаг для запуска обработки в основном потоке
        st.session_state.processing_initiate = True 
        st.rerun() # Перезапускаем, чтобы основной поток увидел флаг

    st.divider()
    st.subheader("Журнал операций")
    # ... (код для журнала без изменений) ...
    if st.session_state.log_messages:
        for level, message in reversed(st.session_state.log_messages):
            if level == "info": st.sidebar.info(message)
            elif level == "success": st.sidebar.success(message)
            elif level == "warning": st.sidebar.warning(message)
            elif level == "error": st.sidebar.error(message)
        if st.sidebar.button("Очистить журнал", key=f"clear_log_{st.session_state.app_run_id}", use_container_width=True):
            st.session_state.log_messages = []
            st.rerun()
    else:
        st.sidebar.caption("Журнал пуст.")


# --- UI: Основная Область (Результаты и Статус) ---
st.title("Результаты Оценки GeminiJudge")

if "processing_initiate" not in st.session_state:
    st.session_state.processing_initiate = False

if st.session_state.processing_initiate:
    overall_success = True
    st.session_state.generated_incorrect_responses = []
    st.session_state.evaluation_result_id = None
    st.session_state.evaluation_rationale = ""
    st.session_state.all_responses_for_evaluation = {}

    with st.status("Этап 0: Подготовка файлов...", expanded=True) as status_files:
        # ... (логика обработки файлов без изменений) ...
        if not handle_file_uploads_and_processing(st.session_state.uploaded_st_files):
            st.error("Проблема с подготовкой файлов.")
            status_files.update(label="Ошибка подготовки файлов!", state="error", expanded=True)
            overall_success = False
        elif not st.session_state.processed_gemini_files and st.session_state.uploaded_st_files:
             st.warning("Файлы были предоставлены, но ни один не активен.")
             status_files.update(label="Файлы не активны!", state="warning", expanded=True)
        else:
            status_files.update(label="Файлы подготовлены/пропущены.", state="complete", expanded=False)


    if overall_success:
        with st.status(f"Этап 1: Генерация {st.session_state.num_incorrect_samples_input} 'неправильных' ответов (модель: {os.getenv('GEMINI_MODEL_GENERATION', gemini_utils.MODEL_NAME_FOR_GENERATION_DEFAULT)})...", expanded=True) as status_gen:
            # ... (логика генерации неправильных ответов без изменений) ...
            if not generate_and_parse_incorrect_responses_logic(st.session_state.user_prompt_input, st.session_state.model_a_response_input, st.session_state.num_incorrect_samples_input):
                st.warning("Проблема с генерацией 'неправильных' ответов.")
                status_gen.update(label="Ошибка генерации 'неправильных'!", state="warning", expanded=True)
            else:
                status_gen.update(label="'Неправильные' ответы сгенерированы!", state="complete", expanded=False)
    
    if overall_success: 
        with st.status(f"Этап 2: Оценка всех ответов (модель: {os.getenv('GEMINI_MODEL_EVALUATION', gemini_utils.MODEL_NAME_FOR_EVALUATION_DEFAULT)})...", expanded=True) as status_eval:
            # ... (логика оценки ответов без изменений) ...
            if not evaluate_all_responses_logic(st.session_state.user_prompt_input, st.session_state.model_a_response_input):
                st.error("Не удалось получить оценку от Gemini.")
                status_eval.update(label="Ошибка оценки!", state="error", expanded=True)
                overall_success = False
            else:
                status_eval.update(label="Ответы оценены!", state="complete", expanded=False)

    
    if overall_success:
        gemini_utils.log_success("=== Сеанс GeminiJudge завершен успешно! ===")
        st.balloons()
    else:
        gemini_utils.log_error("=== Сеанс GeminiJudge завершен с ошибками/предупреждениями. ===")
    
    st.session_state.processing_complete = True
    st.session_state.processing_initiate = False 
    st.rerun() 

# Отображение результатов
if st.session_state.processing_complete:
    if not st.session_state.generated_incorrect_responses and not st.session_state.evaluation_result_id:
        st.info("Нет данных для отображения. Запустите оценку.")

    # Отображение выбранного ответа и обоснования СНАЧАЛА
    if st.session_state.evaluation_result_id:
        st.header("🏆 Итоговый Выбор и Обоснование от Gemini")
        chosen_id_display = st.session_state.evaluation_result_id
        
        if st.session_state.evaluation_result_id == prompts.MODEL_A_ANSWER_ID:
            st.success(f"**Выбран: {chosen_id_display} (Ответ 'Модели А')**")
        elif st.session_state.evaluation_result_id.startswith("НЕПРАВИЛЬНЫЙ_ОТВET_"):
            st.warning(f"**Выбран: {chosen_id_display} (Один из 'неправильных' вариантов!)**")
        else: 
            st.info(f"**Выбран: {chosen_id_display}**")

        chosen_text = st.session_state.all_responses_for_evaluation.get(st.session_state.evaluation_result_id)
        if chosen_text:
            with st.expander("Показать текст выбранного ответа", expanded=False):
                st.markdown(chosen_text)
        
        if st.session_state.evaluation_rationale:
            st.subheader("📝 Обоснование Выбора:")
            with st.container(border=True):
                st.markdown(st.session_state.evaluation_rationale)
        else:
            st.caption("Обоснование не было предоставлено или не удалось извлечь.")
        st.divider()

    # Затем отображение всех ответов (включая "неправильные")
    if st.session_state.all_responses_for_evaluation:
        st.header("🗂️ Все Рассмотренные Ответы")
        
        # Ответ Модели А
        with st.expander(f"Ответ 'Модели А' ({prompts.MODEL_A_ANSWER_ID})", expanded=False):
            st.markdown(st.session_state.all_responses_for_evaluation.get(prompts.MODEL_A_ANSWER_ID, "Текст не найден"))

        # Сгенерированные "неправильные" ответы
        if st.session_state.generated_incorrect_responses:
            st.subheader("Сгенерированные 'Неправильные' Ответы:")
            num_cols = min(len(st.session_state.generated_incorrect_responses), 3) # Максимум 3 колонки
            cols_incorrect = st.columns(num_cols) 
            
            for i, resp_text in enumerate(st.session_state.generated_incorrect_responses):
                with cols_incorrect[i % num_cols]:
                    incorrect_id = prompts.get_incorrect_answer_id(i)
                    exp_title = f"Плохой ответ #{i + 1}"
                    # Не выделяем здесь, т.к. основной выбор уже показан выше
                    # is_chosen = st.session_state.evaluation_result_id == incorrect_id 
                    # if is_chosen: exp_title = f"✔️ {exp_title} (Выбран)"
                    
                    with st.container(border=True, height=250): # Ограничим высоту для компактности
                        st.markdown(f"**{exp_title} ({incorrect_id})**")
                        st.caption(resp_text)
else:
    if not any(st.session_state.log_messages):
         st.info("Настройте параметры в боковой панели слева и запустите оценку.")