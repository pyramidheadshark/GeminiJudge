# geminijudge/app.py
import streamlit as st
from dotenv import load_dotenv
import os
import traceback # Для отладки, если понадобится

# Импорт модулей проекта
import prompts
import gemini_utils

# --- Начальная конфигурация и Инициализация Session State ---
load_dotenv()

def initialize_session_state():
    defaults = {
        "api_key_input": os.getenv("GOOGLE_API_KEY", ""),
        "gemini_configured": False,
        "uploaded_st_files": [], # Список UploadedFile объектов от Streamlit
        "processed_gemini_files": [], # Список genai.File объектов после обработки Gemini
        "user_prompt_input": "",
        "model_a_response_input": "",
        "num_incorrect_samples_input": 3,
        "generated_incorrect_responses": [],
        "evaluation_result_id": None,
        "all_responses_for_evaluation": {},
        "log_messages": [],
        "app_run_id": 0, # Для сброса состояния виджетов при необходимости
        "processing_complete": False # Флаг для отображения результатов
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.api_key_input and not st.session_state.gemini_configured:
        gemini_utils.configure_gemini_api()

initialize_session_state()

# --- Функции для Логики Приложения (не UI) ---

def handle_file_uploads_and_processing(uploaded_st_files_list: list) -> bool:
    # Сначала очистим список ранее обработанных файлов (локально)
    st.session_state.processed_gemini_files = []

    if not uploaded_st_files_list:
        gemini_utils.log_info("Контекстные файлы не предоставлены.")
        return True # Нет файлов - не ошибка, но и обрабатывать нечего.

    all_successful = True
    # Используем with st.spinner в основном потоке UI для лучшего отображения
    
    for i, st_file in enumerate(uploaded_st_files_list):
        gemini_utils.log_info(f"Обработка файла {i+1}/{len(uploaded_st_files_list)}: {st_file.name}")
        # Отображение прогресса в главном UI через st.status будет лучше
        gemini_file_obj = gemini_utils.upload_file_to_gemini(st_file, display_name_prefix=f"doc{i+1}")
        
        if gemini_file_obj and hasattr(gemini_file_obj, 'state') and gemini_file_obj.state.name == "ACTIVE":
            st.session_state.processed_gemini_files.append(gemini_file_obj)
        elif gemini_file_obj: # Файл загружен, но не ACTIVE
            gemini_utils.log_warning(f"Файл {st_file.name} загружен, но не активен (состояние: {gemini_file_obj.state.name}). Может быть не использован.")
            # Решаем, добавлять ли такой файл в список processed_gemini_files.
            # Для простоты пока не будем добавлять, если не ACTIVE.
            # st.session_state.processed_gemini_files.append(gemini_file_obj) # Если хотим все равно попробовать
            all_successful = False 
        else: # gemini_file_obj is None
            gemini_utils.log_error(f"Не удалось обработать файл {st_file.name}.")
            all_successful = False
            
    if all_successful and st.session_state.processed_gemini_files:
        gemini_utils.log_success(f"{len(st.session_state.processed_gemini_files)} файлов успешно подготовлены.")
    elif st.session_state.processed_gemini_files:
         gemini_utils.log_warning(f"Подготовлено {len(st.session_state.processed_gemini_files)} из {len(uploaded_st_files_list)} файлов. Некоторые могут быть недоступны.")
    elif uploaded_st_files_list: # Были файлы, но ни один не обработался
        gemini_utils.log_error("Ни один из файлов не был успешно обработан.")
        all_successful = False
        
    return all_successful

def generate_and_parse_incorrect_responses_logic(user_prompt: str, model_a_response: str, num_samples: int) -> bool:
    st.session_state.generated_incorrect_responses = []
    prompt_for_incorrect = prompts.get_generate_incorrect_answers_prompt(user_prompt, model_a_response, num_samples)
    
    raw_response = gemini_utils.generate_text_from_model(
        prompt_for_incorrect, files_for_context=st.session_state.processed_gemini_files
    )

    if not raw_response:
        gemini_utils.log_error("Не получены 'неправильные' ответы от Gemini.")
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
        gemini_utils.log_warning("Не удалось извлечь 'неправильные' ответы (проверьте формат).")
        # st.sidebar.expander("Сырой ответ Gemini (генерация плохих)", expanded=False).text_area("",raw_response, height=100, key=f"raw_bad_gen_fail_{st.session_state.app_run_id}")
        return False
    
    gemini_utils.log_success(f"Извлечено {len(st.session_state.generated_incorrect_responses)} 'неправильных' ответов.")
    if len(st.session_state.generated_incorrect_responses) != num_samples:
        gemini_utils.log_warning(f"Запрошено {num_samples}, но получено/распарсено {len(st.session_state.generated_incorrect_responses)}.")
    return True

def evaluate_all_responses_logic(user_prompt: str, model_a_response: str) -> bool:
    st.session_state.evaluation_result_id = None
    st.session_state.all_responses_for_evaluation = {}

    all_responses_dict = {prompts.MODEL_A_ANSWER_ID: model_a_response}
    for i, resp_text in enumerate(st.session_state.generated_incorrect_responses):
        all_responses_dict[prompts.get_incorrect_answer_id(i)] = resp_text
    st.session_state.all_responses_for_evaluation = all_responses_dict

    text_block_for_prompt = ""
    for identifier, text in all_responses_dict.items():
        text_block_for_prompt += f"{identifier}:\n{text}\n---\n"
    
    prompt_for_evaluation = prompts.get_evaluate_responses_prompt(user_prompt, text_block_for_prompt)
    
    chosen_id_raw = gemini_utils.generate_text_from_model(
        prompt_for_evaluation, files_for_context=st.session_state.processed_gemini_files
    )

    if not chosen_id_raw:
        gemini_utils.log_error("Не получена оценка от Gemini.")
        return False
    
    chosen_id = chosen_id_raw.strip().split()[0] if chosen_id_raw.strip() else "" # Берем первое слово

    if chosen_id in all_responses_dict:
        st.session_state.evaluation_result_id = chosen_id
        gemini_utils.log_success(f"Gemini выбрал ID: '{chosen_id}'")
    else:
        gemini_utils.log_warning(f"Gemini вернул '{chosen_id_raw.strip()}', ID не распознан. Проверьте ответ Gemini.")
        # st.sidebar.expander("Сырой ответ Gemini (оценка)", expanded=False).text_area("", chosen_id_raw, height=100, key=f"raw_eval_fail_{st.session_state.app_run_id}")
        # Пытаемся найти по тексту, если модель вернула текст вместо ID
        found_by_text = False
        for id_key, text_val in all_responses_dict.items():
            if chosen_id_raw.strip() == text_val.strip():
                st.session_state.evaluation_result_id = id_key
                gemini_utils.log_warning(f"ID '{id_key}' определен по совпадению текста ответа.")
                found_by_text = True
                break
        if not found_by_text:
            return False # Оценка не удалась
            
    return True

# --- UI: Боковая Панель (Конфигурация и Ввод) ---
with st.sidebar:
    st.header("⚖️ GeminiJudge")
    st.caption("Настройки и Входные Данные")

    with st.expander("🔑 API Ключ Google AI", expanded=not st.session_state.gemini_configured):
        api_key_val = st.text_input(
            "Ваш API ключ:",
            type="password",
            value=st.session_state.api_key_input,
            key=f"api_key_field_{st.session_state.app_run_id}",
            help="Можно установить через GOOGLE_API_KEY в .env"
        )
        if api_key_val != st.session_state.api_key_input:
            st.session_state.api_key_input = api_key_val
            st.session_state.gemini_configured = False # Сброс при смене ключа
            gemini_utils.configure_gemini_api() # Попытка настроить сразу
            st.rerun()

        if st.session_state.gemini_configured:
            st.success("API готов.")
        elif st.session_state.api_key_input: # Ключ введен, но конфигурация не удалась
             st.error("API не настроен. Проверьте ключ.")
        # else: # Ключ не введен
            # st.info("Введите API ключ.")

    st.session_state.uploaded_st_files = st.file_uploader(
        "1. Контекстные документы (PDF, TXT, DOCX и др.):",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.app_run_id}"
    )
    if st.session_state.uploaded_st_files:
        st.caption(f"Выбрано файлов: {len(st.session_state.uploaded_st_files)}")
        # Опционально: показать имена файлов
        # for f in st.session_state.uploaded_st_files: st.caption(f"- {f.name[:30]}")
    
    st.session_state.user_prompt_input = st.text_area(
        "2. Ваш Промпт:",
        value=st.session_state.user_prompt_input,
        height=100,
        key=f"user_prompt_area_{st.session_state.app_run_id}"
    )
    st.session_state.model_a_response_input = st.text_area(
        "3. Ответ 'Модели А' (эталонный):",
        value=st.session_state.model_a_response_input,
        height=100,
        key=f"model_a_response_area_{st.session_state.app_run_id}"
    )
    st.session_state.num_incorrect_samples_input = st.number_input(
        "4. Кол-во 'неправильных' примеров:",
        min_value=1, max_value=5,
        value=st.session_state.num_incorrect_samples_input, step=1,
        key=f"num_incorrect_{st.session_state.app_run_id}"
    )

    run_button_disabled = not st.session_state.gemini_configured or \
                          not st.session_state.user_prompt_input.strip() or \
                          not st.session_state.model_a_response_input.strip()
                          # Файлы не обязательны, но если их нет, модель может работать хуже

    if st.button("🚀 Запустить Оценку", type="primary", use_container_width=True, disabled=run_button_disabled, key=f"run_button_{st.session_state.app_run_id}"):
        st.session_state.app_run_id += 1
        st.session_state.log_messages = [] # Очищаем логи для нового запуска
        st.session_state.processing_complete = False # Сброс флага
        gemini_utils.log_info("=== Новый сеанс GeminiJudge ===")

        overall_success = True
        
        # Этапы обработки вынесены в основную часть для использования st.status
        # Здесь мы просто инициируем процесс

    st.divider()
    st.subheader("Журнал операций")
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

# Логика выполнения основного пайплайна, если кнопка была нажата
# Проверяем, была ли кнопка нажата по изменению app_run_id или другому флагу
# Этот блок будет выполняться ПОСЛЕ того, как UI сайдбара отрендерился и кнопка нажата

# Для корректной работы st.status, он должен быть вызван до того, как функция, которую он отслеживает, завершится.
# Поэтому, если кнопка была нажата, мы выполняем всю логику здесь.
# Используем "processing_initiate" как флаг, что кнопка была нажата в предыдущем rerun
if "processing_initiate" not in st.session_state:
    st.session_state.processing_initiate = False

if st.session_state.get(f"run_button_{st.session_state.app_run_id-1}"): # Проверка нажатия кнопки из предыдущего рерана
    st.session_state.processing_initiate = True
    # Сброс флага кнопки, чтобы не запускать повторно без нового нажатия
    # Это сложная часть Streamlit - управление состоянием кнопки
    # Лучше использовать флаг, который устанавливается при нажатии кнопки
    
if st.session_state.processing_initiate:
    overall_success = True
    # Сбрасываем результаты перед новым запуском из этой точки
    st.session_state.generated_incorrect_responses = []
    st.session_state.evaluation_result_id = None
    st.session_state.all_responses_for_evaluation = {}

    with st.status("Этап 0: Подготовка файлов...", expanded=True) as status_files:
        if not handle_file_uploads_and_processing(st.session_state.uploaded_st_files):
            st.error("Проблема с подготовкой файлов.")
            status_files.update(label="Ошибка подготовки файлов!", state="error", expanded=True)
            overall_success = False
        elif not st.session_state.processed_gemini_files and st.session_state.uploaded_st_files:
             st.warning("Файлы были предоставлены, но ни один не активен.")
             status_files.update(label="Файлы не активны!", state="warning", expanded=True)
             # overall_success можно не менять, если файлы не критичны
        else:
            status_files.update(label="Файлы подготовлены/пропущены.", state="complete", expanded=False)

    if overall_success:
        with st.status(f"Этап 1: Генерация {st.session_state.num_incorrect_samples_input} 'неправильных' ответов...", expanded=True) as status_gen:
            if not generate_and_parse_incorrect_responses_logic(st.session_state.user_prompt_input, st.session_state.model_a_response_input, st.session_state.num_incorrect_samples_input):
                st.warning("Проблема с генерацией 'неправильных' ответов.")
                status_gen.update(label="Ошибка генерации 'неправильных'!", state="warning", expanded=True)
                # Не ставим overall_success = False, оценка Model A все еще может быть полезна
            else:
                status_gen.update(label="'Неправильные' ответы сгенерированы!", state="complete", expanded=False)
    
    if overall_success: 
        with st.status("Этап 2: Оценка всех ответов...", expanded=True) as status_eval:
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
    st.session_state.processing_initiate = False # Сбросить флаг
    st.rerun() # Чтобы обновить UI и логи в сайдбаре

# Отображение результатов, если обработка завершена
if st.session_state.processing_complete:
    if not st.session_state.generated_incorrect_responses and not st.session_state.evaluation_result_id:
        st.info("Нет данных для отображения. Запустите оценку.")
    
    if st.session_state.generated_incorrect_responses:
        st.subheader("Сгенерированные 'Неправильные' Ответы:")
        cols_incorrect = st.columns(len(st.session_state.generated_incorrect_responses))
        for i, resp_text in enumerate(st.session_state.generated_incorrect_responses):
            with cols_incorrect[i % len(cols_incorrect)]: # Распределяем по колонкам
                exp_title = f"Плохой ответ #{i + 1}"
                is_chosen = st.session_state.evaluation_result_id == prompts.get_incorrect_answer_id(i)
                if is_chosen: exp_title = f"✔️ {exp_title} (Выбран)"
                
                with st.container(border=True):
                    st.markdown(f"**{exp_title}**")
                    st.caption(resp_text[:150] + "..." if len(resp_text) > 150 else resp_text) # Краткий предпросмотр
                    # Для полного текста можно использовать expander или modal
                    # st.expander("Показать полностью").markdown(resp_text)


    if st.session_state.all_responses_for_evaluation:
        st.divider()
        st.subheader("Итоговый Выбор Gemini:")
        if st.session_state.evaluation_result_id:
            chosen_text = st.session_state.all_responses_for_evaluation.get(st.session_state.evaluation_result_id)
            
            result_prefix = "🤔"
            chosen_id_display = st.session_state.evaluation_result_id
            
            if st.session_state.evaluation_result_id == prompts.MODEL_A_ANSWER_ID:
                st.success(f"🏆 Gemini выбрал: **{chosen_id_display}** (Ответ 'Модели А')")
            elif st.session_state.evaluation_result_id and st.session_state.evaluation_result_id.startswith("НЕПРАВИЛЬНЫЙ_ОТВЕТ_"):
                st.warning(f"⚠️ Gemini выбрал: **{chosen_id_display}** (Один из 'неправильных' вариантов!)")
            else: 
                st.info(f"{result_prefix} Gemini выбрал: **{chosen_id_display}**")

            if chosen_text:
                with st.container(border=True):
                    st.markdown("**Текст выбранного ответа:**")
                    st.markdown(chosen_text)
            else:
                st.error(f"Не найден текст для ID: {st.session_state.evaluation_result_id}")
        elif st.session_state.all_responses_for_evaluation : # Если были ответы, но оценка не удалась
            st.warning("Gemini не сделал выбор или произошла ошибка при оценке.")
else:
    if not any(st.session_state.log_messages): # Показываем только если логов еще нет
         st.info("Настройте параметры в боковой панели слева и запустите оценку.")