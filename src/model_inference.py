import logging
from transformers import pipeline
from .config import MODEL_NAME, CANDIDATE_LABELS

logger = logging.getLogger(__name__)
_classifier = None


def load_model():
    """
    Загружает (или возвращает уже загруженный) zero-shot классификатор.
    """
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot-classification pipeline...")
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1
        )
    return _classifier


def format_conversation(messages):
    """
    Формирует строки диалога для последующего анализа.
    Пример:
        "0: Привет\n1: Здравствуйте!\n..."
    """
    return "\n".join(
        [f"{msg['participant_index']}: {msg['text']}" for msg in messages]
    )


def classify_text(messages) -> float:
    """
    Прогоняет диалог через zero-shot классификатор и возвращает
    вероятность, что в диалоге есть бот.
    """
    classifier = load_model()
    conversation_text = format_conversation(messages)
    prompt = f"Определи, есть ли ai-бот в диалоге:\n\n{conversation_text}"

    result = classifier(
        prompt,
        candidate_labels=CANDIDATE_LABELS
    )

    bot_index = result["labels"].index(CANDIDATE_LABELS[0])
    return result["scores"][bot_index]
