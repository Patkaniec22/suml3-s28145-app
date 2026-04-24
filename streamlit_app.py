import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(
    page_title="EN → DE Translator",
    page_icon="🌍",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_translator():
    """Ładuje tokenizer i model tłumaczący EN -> DE z Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    return tokenizer, model
 
 
def translate(text: str) -> str:
    tokenizer, model = load_translator()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
 
 
@st.cache_resource(show_spinner=False)
def load_sentiment_classifier():
    return pipeline("sentiment-analysis")

st.title("GB-> DE Tłumacz angielsko-niemiecki")
st.image(
    "https://images.unsplash.com/photo-1527866959252-deab85ef7d1b?w=800",
    caption="Translate with Hugging Face",
    use_container_width=True,
)
 
st.success("Gratulacje! Z powodzeniem uruchomiłeś aplikację 🎉")
st.header(" Wybierz funkcję")
 
option = st.selectbox(
    "Opcje",
    [
        "Tłumaczenie tekstu (EN → DE)",
        "Wydźwięk emocjonalny tekstu (eng)",
    ],
)
 
 
if option == "Tłumaczenie tekstu (EN → DE)":
    st.subheader("Tłumaczenie z angielskiego na niemiecki")
 
    text = st.text_area(
        label="Wpisz tekst po angielsku:",
        placeholder="Hello, how are you today?",
        height=150,
    )
 
    if st.button("Przetłumacz"):
        if not text.strip():
            st.warning("Najpierw wpisz jakiś tekst do przetłumaczenia.")
        else:
            try:
                with st.spinner("Trwa ładowanie modelu i tłumaczenie... "):
                    translation = translate(text)
 
                st.success("Gotowe! Oto tłumaczenie:")
                st.text_area(
                    label="Tłumaczenie (DE):",
                    value=translation,
                    height=150,
                )
                st.balloons()
            except Exception as e:
                st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")
 
 
elif option == "Wydźwięk emocjonalny tekstu (eng)":
    st.subheader("Analiza sentymentu")
 
    text = st.text_area(label="Wpisz tekst po angielsku:")
 
    if text:
        try:
            with st.spinner("Analizuję sentyment..."):
                classifier = load_sentiment_classifier()
                answer = classifier(text)
            st.success("Gotowe!")
            st.write(answer)
        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")


st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "Aplikacja stworzona na poczet zadanie SUML_3. Streamlit<br>"
    "<b>Numer indeksu: s28145</b>"
    "</p>",
    unsafe_allow_html=True,
)
