import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import warnings
import os
from PIL import Image
warnings.filterwarnings('ignore')


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .header-text {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .subheader-text {
        font-size: 1.1em;
        color: #b0b0b0;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(15, 52, 96, 0.5);
        padding: 5px;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff;
        color: #1a1a2e;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        color: #1a1a2e;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        width: 100%;
    }
    
    .divider {
        border-top: 1px solid #00d4ff;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load all trained models"""
    try:
        with open('best_nb_model.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        
        lstm_model = load_model('best_lstm_model.h5', compile=False)
        
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        return nb_model, lstm_model, tokenizer, label_encoder, tfidf_vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)


nb_model, lstm_model, tokenizer, label_encoder, tfidf_vectorizer = load_artifacts()


# ============================================================================
# PREPROCESSING
# ============================================================================
slangwords = {"리": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah"}


def remove_noise(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text


def to_lowercase(text):
    return text.lower()


def normalize_slang(text):
    words = text.split()
    fixed_words = []
    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)
    return ' '.join(fixed_words)


def tokenize_words(text):
    return word_tokenize(text)


def remove_stopwords(text):
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    filtered = [txt for txt in text if txt not in listStopwords]
    return filtered


def stemmingText(text):
    try:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    except:
        return text


def reconstruct_text(list_words):
    return ' '.join(word for word in list_words)


def preprocess_text(text):
    try:
        if not isinstance(text, str) or len(str(text).strip()) == 0:
            return ""
        
        cleaned_text = remove_noise(text)
        lowercased_text = to_lowercase(cleaned_text)
        normalized_text = normalize_slang(lowercased_text)
        tokens = tokenize_words(normalized_text)
        filtered_tokens = remove_stopwords(tokens)
        stemmed_text = stemmingText(reconstruct_text(filtered_tokens))
        return stemmed_text
    except:
        return ""


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_sentiment_nb(text):
    try:
        processed_text = preprocess_text(text)
        if not processed_text:
            return 'neutral'
        prediction = nb_model.predict([processed_text])
        sentiment = label_encoder.inverse_transform(prediction)
        return sentiment[0]
    except:
        return 'neutral'


def predict_sentiment_lstm(text, max_seq_len=200):
    try:
        processed_text = preprocess_text(text)
        if not processed_text:
            return 'neutral'
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_seq_len, padding='post', truncating='post')
        prediction = lstm_model.predict(padded_sequence, verbose=0)
        sentiment_idx = np.argmax(prediction, axis=1)
        sentiment = label_encoder.inverse_transform(sentiment_idx)
        return sentiment[0]
    except:
        return 'neutral'


def get_sentiment_emoji(sentiment):
    if sentiment == 'positive':
        return '😊'
    elif sentiment == 'negative':
        return '😞'
    else:
        return '😐'


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<div class="header-text">🎯 Sentimen Analisis Review APK Play Store</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader-text">Aplikasi untuk menganalisis sentimen dari ulasan aplikasi Mobile Legends di Google Play Store</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Analisis Teks Tunggal", "📊 Analisis File CSV", "📈 Statistik", "🎨 Visualisasi"])
    
    # ================================================================
    # TAB 1: SINGLE TEXT ANALYSIS
    # ================================================================
    with tab1:
        st.header("Analisis Sentimen untuk Teks Tunggal")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "🔤 Masukkan ulasan aplikasi Anda:",
                placeholder="Contoh: Game ini sangat bagus, saya suka sekali!",
                height=150,
                key='single_text'
            )
        
        with col2:
            st.info("""
            **📖 Tips:**
            - Masukkan review dalam Bahasa Indonesia
            - Gunakan kata-kata yang jelas
            - Bisa berupa review panjang atau pendek
            """)
        
        if st.button("🚀 Analisis Sentimen", key='analyze_single', use_container_width=True):
            if user_input.strip():
                with st.spinner('⏳ Menganalisis teks...'):
                    nb_sentiment = predict_sentiment_nb(user_input)
                    lstm_sentiment = predict_sentiment_lstm(user_input)
                    
                    st.markdown("---")
                    st.subheader("📊 Hasil Analisis Sentimen")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🤖 Naive Bayes Model")
                        emoji = get_sentiment_emoji(nb_sentiment)
                        
                        if nb_sentiment == 'positive':
                            st.success(f"### {emoji} Sentimen: **POSITIVE** 🎉", icon="✅")
                        elif nb_sentiment == 'negative':
                            st.error(f"### {emoji} Sentimen: **NEGATIVE** 😞", icon="❌")
                        else:
                            st.info(f"### {emoji} Sentimen: **NEUTRAL** 😐", icon="ℹ️")
                        
                        st.markdown(f"""
                        <div style="background: rgba(0,212,255,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;">
                        <p style="color: #00d4ff; font-weight: 600;">Akurasi: 71.43%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 🧠 LSTM Deep Learning Model")
                        emoji = get_sentiment_emoji(lstm_sentiment)
                        
                        if lstm_sentiment == 'positive':
                            st.success(f"### {emoji} Sentimen: **POSITIVE** 🎉", icon="✅")
                        elif lstm_sentiment == 'negative':
                            st.error(f"### {emoji} Sentimen: **NEGATIVE** 😞", icon="❌")
                        else:
                            st.info(f"### {emoji} Sentimen: **NEUTRAL** 😐", icon="ℹ️")
                        
                        st.markdown(f"""
                        <div style="background: rgba(0,212,255,0.1); padding: 10px; border-radius: 8px; margin-top: 10px;">
                        <p style="color: #00d4ff; font-weight: 600;">Akurasi: 82.13%</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Silakan masukkan teks untuk dianalisis!")
    
    # ================================================================
    # TAB 2: CSV ANALYSIS (FIXED - NO DUPLICATE CHART ERROR)
    # ================================================================
    with tab2:
        st.header("Analisis Sentimen dari File CSV")
        
        uploaded_file = st.file_uploader(
            "📤 Unggah file CSV Anda",
            type=["csv"],
            help="File CSV harus memiliki kolom yang berisi teks review"
        )
        
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
                
                st.subheader("📋 Preview Data")
                st.dataframe(df_uploaded.head(), use_container_width=True)
                st.info(f"📊 Total baris: {len(df_uploaded)}")
                
                text_column = st.selectbox(
                    "🔍 Pilih kolom yang berisi teks ulasan:",
                    df_uploaded.columns,
                    help="Pilih kolom yang berisi review text"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    sample_size = st.number_input(
                        "Jumlah data yang dianalisis:",
                        min_value=1,
                        max_value=len(df_uploaded),
                        value=min(50, len(df_uploaded))
                    )
                
                with col2:
                    st.info(f"💾 Akan menganalisis {sample_size} dari {len(df_uploaded)} baris")
                
                if st.button("🚀 Mulai Analisis CSV", key='analyze_csv', use_container_width=True):
                    if text_column and text_column in df_uploaded.columns:
                        df_sample = df_uploaded.iloc[:sample_size].copy()
                        
                        df_sample = df_sample[df_sample[text_column].notna()]
                        df_sample = df_sample[df_sample[text_column].astype(str).str.strip() != '']
                        
                        if len(df_sample) == 0:
                            st.error("❌ Semua data kosong atau null!")
                        else:
                            with st.spinner(f'⏳ Menganalisis {len(df_sample)} data...'):
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                results_list = []
                                error_count = 0
                                
                                for idx, row in df_sample.iterrows():
                                    try:
                                        text = str(row[text_column]).strip()
                                        
                                        if text and len(text) > 0:
                                            nb_sent = predict_sentiment_nb(text)
                                            lstm_sent = predict_sentiment_lstm(text)
                                            
                                            results_list.append({
                                                'Text': text[:100],
                                                'Naive Bayes': nb_sent,
                                                'LSTM': lstm_sent
                                            })
                                    except:
                                        error_count += 1
                                        continue
                                    
                                    progress = (idx + 1) / len(df_sample)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Diproses: {idx + 1}/{len(df_sample)}")
                                
                                results_df = pd.DataFrame(results_list)
                            
                            status_text.empty()
                            
                            if len(results_df) > 0:
                                st.success(f"✅ Analisis selesai! {len(results_df)} data berhasil diproses")
                                if error_count > 0:
                                    st.warning(f"⚠️ {error_count} data tidak bisa diproses")
                                
                                st.markdown("---")
                                st.subheader("✅ Hasil Analisis")
                                st.dataframe(results_df, use_container_width=True, height=300)
                                
                                # ============= CHARTS WITH UNIQUE KEYS =============
                                st.markdown("---")
                                st.subheader("📊 Distribusi Sentimen")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("### 🤖 Naive Bayes")
                                    nb_counts = results_df['Naive Bayes'].value_counts()
                                    
                                    fig_nb = go.Figure()
                                    fig_nb.add_trace(go.Bar(
                                        x=nb_counts.index,
                                        y=nb_counts.values,
                                        marker=dict(
                                            color=['#4caf50' if s == 'positive' else '#f44336' if s == 'negative' else '#2196f3' for s in nb_counts.index]
                                        ),
                                        text=nb_counts.values,
                                        textposition='auto'
                                    ))
                                    fig_nb.update_layout(
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=False,
                                        xaxis_title="Sentimen",
                                        yaxis_title="Jumlah"
                                    )
                                    st.plotly_chart(fig_nb, use_container_width=True, key='nb_chart_csv')
                                
                                with col2:
                                    st.write("### 🧠 LSTM")
                                    lstm_counts = results_df['LSTM'].value_counts()
                                    
                                    fig_lstm = go.Figure()
                                    fig_lstm.add_trace(go.Bar(
                                        x=lstm_counts.index,
                                        y=lstm_counts.values,
                                        marker=dict(
                                            color=['#4caf50' if s == 'positive' else '#f44336' if s == 'negative' else '#2196f3' for s in lstm_counts.index]
                                        ),
                                        text=lstm_counts.values,
                                        textposition='auto'
                                    ))
                                    fig_lstm.update_layout(
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=False,
                                        xaxis_title="Sentimen",
                                        yaxis_title="Jumlah"
                                    )
                                    st.plotly_chart(fig_lstm, use_container_width=True, key='lstm_chart_csv')
                                
                                # ============= STATISTICS =============
                                st.markdown("---")
                                st.subheader("📈 Statistik")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Teranalisis", len(results_df))
                                
                                with col2:
                                    agreement = (results_df['Naive Bayes'] == results_df['LSTM']).sum()
                                    pct = (agreement / len(results_df) * 100) if len(results_df) > 0 else 0
                                    st.metric("Model Agreement", f"{pct:.1f}%")
                                
                                with col3:
                                    positive_count = (results_df['Naive Bayes'] == 'positive').sum()
                                    st.metric("Positif (%)", f"{positive_count/len(results_df)*100:.1f}%")
                                
                                # ============= DOWNLOAD =============
                                st.markdown("---")
                                csv_output = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="💾 Unduh Hasil Analisis CSV",
                                    data=csv_output,
                                    file_name="analisis_sentimen_hasil.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.error("❌ Tidak ada data yang berhasil diproses!")
                    else:
                        st.error("❌ Kolom teks tidak ditemukan!")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tips: Pastikan file CSV valid")

        # ================================================================
    # TAB 3: STATISTICS & DATASET OVERVIEW
    # ================================================================
    with tab3:
        st.header("📈 Statistik Model & Dataset Overview")
        
        # =============== DATASET OVERVIEW SECTION ===============
        st.subheader("📊 Dataset Overview")
        st.markdown("---")
        
        # Dataset Info in Columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Total Reviews", "60,000")
        
        with col2:
            st.metric("🌍 Data Source", "Google Play Store")
        
        with col3:
            st.metric("📝 Bahasa", "Bahasa Indonesia")
        
        with col4:
            st.metric("📅 Timeline", "2023-2025")
        
        # Dataset Details Box
        st.markdown("")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            ### 📋 Dataset Characteristics
            
            **Sentiment Distribution:**
            - 🟢 Positive: ~50-60%
            - 🔴 Negative: ~20-30%
            - ⚫ Neutral: ~15-25%
            
            **Data Quality:**
            - Cleaning: ✅ Noise removal applied
            - Preprocessing: ✅ 6-stage pipeline
            - Validation: ✅ Train-test split (80-20)
            """)
        
        with col2:
            st.info("""
            ### 🔧 Preprocessing Pipeline
            
            **6 Tahap Preprocessing:**
            1. 🗑️ Noise Removal
            2. 🔤 Lowercasing
            3. 🌐 Slang Normalization
            4. ✂️ Tokenization
            5. ⏭️ Stopwords Removal
            6. 🔗 Stemming (Sastrawi)
            
            **Result:** Clean & normalized text features
            """)
        
        st.markdown("---")
        
        # =============== MODEL STATISTICS SECTION ===============
        st.subheader("🎯 Model Performance Metrics")
        st.markdown("")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Naive Bayes Accuracy", "71.43%")
        with col2:
            st.metric("🧠 LSTM Accuracy", "82.13%")
        with col3:
            st.metric("📊 Model Agreement", "88.77%")
        
        st.markdown("")
        
        # Model Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Naive Bayes Classifier
            
            **Tipe:** Machine Learning Classifier
            
            **Vectorizer:** TF-IDF (Term Frequency-Inverse Document Frequency)
            
            **Karakteristik:**
            - ⚡ Sangat Cepat
            - 📊 Probabilistic approach
            - 🎯 Akurasi: 71.43%
            - ⚙️ Training time: < 1 detik
            
            **Use Case:** Ketika kecepatan adalah prioritas utama
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 LSTM Deep Learning
            
            **Tipe:** Recurrent Neural Network (RNN)
            
            **Architecture:** Embedding → LSTM Layers → Dense Output
            
            **Karakteristik:**
            - ⏱️ Moderate speed
            - 🧠 Neural network approach
            - ⭐ Akurasi: 82.13% (LEBIH BAIK)
            - ⚙️ Training time: 5-10 menit
            
            **Use Case:** Ketika akurasi adalah prioritas utama
            """)
        
        st.markdown("---")
        
        # =============== IMPROVEMENT SECTION ===============
        st.subheader("📈 Model Comparison & Improvement")
        
        # Create comparison data
        comparison_data = {
            'Model': ['Naive Bayes', 'LSTM'],
            'Accuracy': [71.43, 82.13],
            'Speed': ['Very Fast ⚡', 'Moderate ⏱️'],
            'Best For': ['Speed Priority', 'Accuracy Priority ⭐']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Improvement Metrics
        improvement_col1, improvement_col2, improvement_col3 = st.columns(3)
        
        with improvement_col1:
            st.success("""
            ### ✅ Improvement
            
            **+10.7%** Akurasi LSTM
            
            Dari 71.43% → 82.13%
            """)
        
        with improvement_col2:
            st.info("""
            ### 🔄 Model Agreement
            
            **88.77%** Konsistensi
            
            Kedua model setuju pada 88.77% prediksi
            """)
        
        with improvement_col3:
            st.warning("""
            ### ⏳ Trade-off
            
            **Speed vs Accuracy**
            
            LSTM lebih akurat tapi lebih lambat
            """)
        
        st.markdown("---")
        
        # =============== KEY FINDINGS SECTION ===============
        st.subheader("🔍 Key Findings")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success("✅ **Conclusion**")
        
        with col2:
            st.markdown("""
            LSTM model adalah pilihan optimal untuk sentiment analysis 
            dengan peningkatan akurasi 10.7% dan high confidence dalam prediksi. 
            Preprocessing berkualitas adalah kunci kesuksesan model.
            """)


    
    # ================================================================
    # TAB 3: STATISTICS
    # ================================================================
    with tab3:
        st.header("📈 Statistik Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Naive Bayes Accuracy", "71.43%")
        with col2:
            st.metric("🧠 LSTM Accuracy", "82.13%")
        with col3:
            st.metric("📊 Model Agreement", "88.77%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🤖 Naive Bayes
            - **Tipe**: Machine Learning Classifier
            - **Vectorizer**: TF-IDF
            - **Akurasi**: 71.43%
            - **Kecepatan**: Sangat cepat ⚡
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 LSTM
            - **Tipe**: Deep Learning (RNN)
            - **Architecture**: LSTM Layer
            - **Akurasi**: 82.13% ⭐
            - **Kecepatan**: Moderate ⏱️
            """)
    
    # ================================================================
    # TAB 4: VISUALIZATIONS
    # ================================================================
    with tab4:
        st.header("🎨 Visualisasi Model")
        
        st.markdown("""
        Tempat untuk menampilkan gambar visualisasi dari hasil training model.
        """)
        
        image_files = {
            'confusion_matrix_lstm.png': 'Confusion Matrix - LSTM',
            'confusion_matrix_naive_bayes.png': 'Confusion Matrix - Naive Bayes',
            'class_distribution.png': 'Class Distribution',
            'sentiment_polarity_pie_chart.png': 'Sentiment Distribution'
        }
        
        images_found = False
        for filename, label in image_files.items():
            if os.path.exists(filename):
                try:
                    img = Image.open(filename)
                    st.image(img, caption=label, use_column_width=True)
                    images_found = True
                except:
                    pass
        
        if not images_found:
            st.info("📁 Belum ada gambar visualisasi. Copy PNG files ke folder app untuk menampilkan.")


if __name__ == "__main__":
    main()