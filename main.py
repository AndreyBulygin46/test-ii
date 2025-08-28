import streamlit as st
from transformers import pipeline
import re
from googletrans import Translator


def is_russian(text):
    return any('\u0400' <= char <= '\u04FF' for char in text)

def translate(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='ru', dest='en')
        return translated.text
    except Exception as e:
        st.error(f"Ошибка перевода: {e}")
        return text

# Инициализация модели
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

classifier = load_model()

st.title("Оценка звонков")
st.markdown("Вставьте текст разговора для анализа")

user_input = st.text_area("Текст разговора", height=300)

if st.button("Анализировать"):
    if not user_input.strip():
        st.warning("Введите текст для анализа")
        st.stop()

    # Предобработка текста
    clean_text = re.sub(r'\s+', ' ', user_input).strip()

    if is_russian(clean_text):
        st.info("Текст на русском языке. Перевод в английский...")
        clean_text = translate(clean_text)

    result = classifier(clean_text)[0]
    sentiment = result['label'].lower()

    recommendations = {
        "label_2": ["Позитивно",
            "Хорошо! Поддерживайте такой тон общения.",
            "Удостоверьтесь, что клиент чувствует себя комфортно."
        ],
        "label_1": [ "Нейтрально",
            "Добавьте больше эмпатии в ответы.",
            "Попробуйте использовать больше вопросов для уточнения потребностей."
        ],
        "label_0": [ "Негативно",
            "Избегайте категоричных формулировок.",
            "Проявите больше терпимости к возражениям клиента."
        ]
    }

    st.markdown(f"### Оценка тона: **{recommendations[sentiment][0]}** ({result['score']:.2f})")
    st.markdown("### Рекомендации:")
    for rec in recommendations[sentiment][1:]:
        st.markdown(f"- {rec}")
