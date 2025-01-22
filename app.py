import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile

# Configuración del modelo en CPU por defecto para evitar problemas de CUDA
def process_audio(audio_path, task, model_size, language):
    # Especificamos device="cpu" y compute_type="int8" para mejor compatibilidad
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    if task == "Transcripción":
        segments, _ = model.transcribe(audio_path, language=language)
        return "".join([segment.text + "\n" for segment in segments])
    else:
        segments, _ = model.translate(audio_path)
        return "".join([segment.text + "\n" for segment in segments])

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def main():
    st.title("Transcripción y Traducción de Audio")
    
    # Configuración
    task = st.selectbox(
        "Selecciona la tarea",
        ["Transcripción", "Traducción"]
    )
    
    # Limitamos los modelos a smaller ones para mejor rendimiento
    model_size = st.selectbox(
        "Selecciona el modelo",
        ["small", "tiny"]
    )
    
    language = st.selectbox(
        "Selecciona el idioma",
        ["Autodetect", "Spanish", "English"]
    )
    
    language = None if language == "Autodetect" else language.lower()
    
    uploaded_file = st.file_uploader("Sube tu archivo de audio", type=['mp3', 'wav', 'm4a'])
    
    if uploaded_file and st.button("Procesar"):
        with st.spinner('Procesando audio...'):
            temp_path = save_uploaded_file(uploaded_file)
            
            try:
                result = process_audio(temp_path, task, model_size, language)
                st.text_area("Resultado:", result, height=300)
                
                st.download_button(
                    label="Descargar resultado",
                    data=result,
                    file_name="resultado.txt",
                    mime="text/plain"
                )
            
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

if __name__ == "__main__":
    main()
