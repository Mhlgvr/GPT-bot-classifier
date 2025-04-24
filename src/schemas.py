from pydantic import BaseModel, UUID4, StrictStr
from typing import Optional
 

class GetMessageRequestModel(BaseModel):
    """
    Входная модель (POST /get_message):
    - dialog_id: UUID текущего диалога
    - last_msg_text: последнее сообщение пользователя
    - last_message_id: (опционально) ID этого сообщения
    """
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: Optional[UUID4] = None


class GetMessageResponseModel(BaseModel):
    """
    Ответная модель (POST /get_message):
    - new_msg_text: текст, сгенерированный ботом
    - dialog_id: UUID диалога
    """
    new_msg_text: StrictStr
    dialog_id: UUID4



class IncomingMessage(BaseModel):
    """
    Входная схема одного сообщения, которое нужно сохранить
    и на основании которого проводится классификация диалога.
    """
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    """
    Результат классификации:
    - id: уникальный идентификатор предсказания
    - message_id: UUID сообщения, на которое мы отвечаем
    - dialog_id: ID диалога
    - participant_index: индекс участника
    - is_bot_probability: вероятность, что в диалоге присутствует бот
    """
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float
