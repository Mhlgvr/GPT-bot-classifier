import time
import uvicorn
import logging
import psycopg2

import uuid
from fastapi import FastAPI, HTTPException

from . import database
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, PROXY_URL, OPEN_AI_API_KEY
from .gpt_api import query_openai_with_context
from .schemas import IncomingMessage, Prediction, GetMessageResponseModel, GetMessageRequestModel
from .model_inference import classify_text


logger = logging.getLogger(__name__)

app = FastAPI(
    title="GPT/BERT Service",
    description="Сервис для генерации ответов и классификации",
)



@app.on_event("startup")
def on_startup() -> None:
    """
    Запуск приложения FastAPI.
    Выполняем проверку доступности PostgreSQL в цикле (на всякий случай)
    После успешного соединения инициализируем базу.
    """
    while True:
        try:
            conn = psycopg2.connect(
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            logger.warning("Waiting for PostgreSQL to become available...")
            time.sleep(2)

    # Инициализация БД
    database.init_db()


@app.post("/get_message", response_model=GetMessageResponseModel)
def get_message(body: GetMessageRequestModel) -> GetMessageResponseModel:
    """
    Эндпоинт, принимающий сообщение от пользователя и возвращающий ответ GPT.

    Действия:
    1. Сохраняет входное сообщение (participant_index=0) в БД.
    2. Загружает весь контекст диалога (user + assistant) и формирует запрос к GPT.
    3. Генерирует ответ OpenAI ChatCompletion.
    4. Сохраняет ответ бота (participant_index=1) в БД.
    5. Возвращает ответ и dialog_id.
    """
    # Сохраняем новое пользовательское сообщение
    user_msg_id = body.last_message_id or uuid.uuid4()
    database.insert_message(
        msg_id=user_msg_id,
        dialog_id=body.dialog_id,
        text=body.last_msg_text,
        participant_index=0
    )

    response_from_openai = "Привет, меня загрузили, но пока не подключили LLM"
    # Генерируем ответ GPT
    if OPEN_AI_API_KEY and PROXY_URL:
        response_from_openai = query_openai_with_context(body, model="gpt-4o")

    # Сохраняем сообщение бота
    bot_msg_id = uuid.uuid4()
    database.insert_message(
        msg_id=bot_msg_id,
        dialog_id=body.dialog_id,
        text=response_from_openai,
        participant_index=1
    )

    return GetMessageResponseModel(
        new_msg_text=response_from_openai,
        dialog_id=body.dialog_id
    )


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Эндпоинт для сохранения сообщения и получения вероятности того,
    что в диалоге участвует бот.

    1. Сохраняем входное сообщение в таблицу `messages`.
    2. Забираем все сообщения данного `dialog_id`.
    3. Применяем zero-shot классификатор.
    4. Возвращаем объект `Prediction`.
    """

    database.insert_message(
        msg_id=msg.id,
        text=msg.text,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index
    )

    # Загружаем весь диалог
    conversation_text = database.select_messages_by_dialog(msg.dialog_id)
    if not conversation_text:
        raise HTTPException(
            status_code=404,
            detail="No messages found for this dialog_id"
        )

    is_bot_probability = classify_text(conversation_text)
    prediction_id = uuid.uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
