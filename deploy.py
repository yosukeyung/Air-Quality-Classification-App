import streamlit as st
import pandas as pd
import plotly.express as px
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go

st.set_page_config(
    page_title="Air Quality Classification",
    page_icon="🌍",
    layout="wide"
)

# --- 1. SETTING UI & CSS ---
def set_ui_style():
    st.markdown("""
    <style>
    
    /* --- STYLING TOMBOL RESET DI SIDEBAR -> MERAH --- */
    [data-testid="stSidebar"] button {
        background-color: transparent !important;
        border: 1px solid #ef4444 !important; /* Garis tepi merah */
        color: #ef4444 !important; /* Teks merah */
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: #fee2e2 !important; /* Merah muda pudar saat kursor diarahkan */
        color: #dc2626 !important;
        border-color: #dc2626 !important;
    }
    
    [data-testid="stSidebar"] button:focus {
        box-shadow: 0 0 0 0.2rem rgba(239, 68, 68, 0.25) !important;
    }
                
    /* Background utama */
    .stApp {
        background: linear-gradient(135deg, #dff5e1, #e0ecff);
        font-family: 'Segoe UI', sans-serif;
    }
    
    .block-container {
        background-color: white;
        padding: 2rem 3rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin-top: 2rem;
        max-width: 1100px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    
    h1 { color: #2c7a7b; font-weight: 700; }
    h2, h3 { color: #2f855a; }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #38b2ac, #3182ce);
        border-radius: 10px;
    }
    
    /* Card style agar lebih proporsional dan clean */
    .card {
        background: #f9fafb;
        padding: 1.5rem 2rem; /* Jarak atas-bawah dirapatkan, kiri-kanan tetap lega */
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    .custom-info {
        background: #e6fffa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #38b2ac;
    }
    
    img {
        border-radius: 15px;
        max-width: 800px !important; 
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* --- CUSTOM NAVBAR --- */
    div[role="radiogroup"] {
        gap: 0.5rem;
    }
    
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    div[role="radiogroup"] > label {
        padding: 10px 15px;
        border-radius: 12px;
        background-color: transparent;
        transition: all 0.2s ease;
        cursor: pointer;
        width: 100%;
    }
    
    div[role="radiogroup"] > label:hover {
        background-color: #e2e8f0;
    }
    
    /* Highlight Menu Aktif: Biru Muda + Garis Tegas di kiri */
    div[role="radiogroup"] > label:has(input:checked) {
        background-color: #dbeafe !important; /* Warna biru muda */
        border-left: 5px solid #3182ce !important; 
    }
    
    div[role="radiogroup"] > label p {
        font-size: 16px;
        font-weight: 500;
        color: #4a5568;
    }
    
    div[role="radiogroup"] > label:has(input:checked) p {
        font-weight: 800 !important;
        color: #1e3a8a !important; 
    }

    /* --- STYLING TOMBOL AKSI (PRIMARY) -> HIJAU --- */
    button[kind="primary"] {
        background-color: #38a169 !important; /* Hijau */
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover {
        background-color: #2f855a !important; /* Hijau gelap saat dihover */
        box-shadow: 0 4px 10px rgba(47, 133, 90, 0.3);
    }
    
    /* --- STYLING TOMBOL RESET -> MERAH --- */
    .reset-btn button {
        background-color: transparent !important;
        border: 1px solid #ef4444 !important; /* Merah */
        color: #ef4444 !important; /* Merah */
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .reset-btn button:hover {
        background-color: #fee2e2 !important; /* Merah muda pudar */
        color: #dc2626 !important;
        border-color: #dc2626 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD DATA (DI-CACHE AGAR CEPAT) ---
@st.cache_data
def load_data():
    # Pastikan file CSV dari Kaggle sudah didownload dan di-rename menjadi 'air_quality.csv'
    return pd.read_csv("air_quality.csv")

# --- 2. HALAMAN HOME ---
def show_home():
    st.title("🌍 Air Quality Classification App")
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        try:
            st.image("Homepage.png", use_container_width=True)
        except:
            st.info("Gambar Homepage.png belum tersedia di folder yang sama.")

    st.markdown("---")
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #2f855a; margin-top: 0; margin-bottom: 15px;">📌 Tentang Aplikasi</h3>
        <p style="margin-bottom: 8px; font-size: 16px;">Aplikasi ini memprediksi tingkat kualitas udara (<b>Air Quality</b>) berdasarkan data lingkungan menggunakan Machine Learning.</p>
        <p style="margin-bottom: 5px; font-size: 16px;">Kategori yang dihasilkan:</p>
        <ul style="margin-top: 0; font-size: 16px; line-height: 1.6;">
            <li>🟢 <b>Good</b></li>
            <li>🟡 <b>Moderate</b></li>
            <li>🟠 <b>Poor</b></li>
            <li>🔴 <b>Hazardous</b></li>
        </ul>
    </div>
    
    <div class="card">
        <h3 style="color: #2f855a; margin-top: 0; margin-bottom: 15px;">👥 Target Pengguna</h3>
        <ul style="margin-bottom: 0; font-size: 16px; line-height: 1.8;">
            <li><b>Masyarakat Umum</b> &rarr; monitoring udara harian</li>
            <li><b>Peneliti & Akademisi</b> &rarr; analisis data lingkungan</li>
            <li><b>Pemerintah</b> &rarr; pengambilan kebijakan tata kota</li>
        </ul>
    </div>
    
    <div class="card">
        <h3 style="color: #2f855a; margin-top: 0; margin-bottom: 15px;">📊 Dataset</h3>
        <p style="font-size: 16px; margin-bottom: 10px;">Dataset berisi <b>5000 data</b> terkait kualitas udara dan polusi.</p>
        <div class="custom-info" style="margin: 10px 0 15px 0;">
            🔗 <b>Sumber Dataset:</b> 
            <a href="https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment" target="_blank" style="color: #3182ce; font-weight: 600; text-decoration: none;">Air Quality and Pollution Assessment (Kaggle)</a>
        </div>
        <p style="margin-bottom: 5px; font-size: 16px;">Fitur yang digunakan:</p>
        <ul style="margin-top: 0; margin-bottom: 0; font-size: 16px; line-height: 1.8;">
            <li>🌡️ Temperature & Humidity</li>
            <li>🌫️ PM2.5, PM10, NO2, SO2, CO</li>
            <li>🏭 Proximity to Industry</li>
            <li>👥 Population Density</li>
        </ul>
    </div>
    
    <!-- Bagian Team Members Ditambahkan Di Sini -->
    <div class="card">
        <h3 style="color: #2f855a; margin-top: 0; margin-bottom: 15px;">👨‍💻 Team Members</h3>
        <p style="color: #4a5568; font-weight: 600; font-size: 16px; margin-top: 0; margin-bottom: 15px;">🏛️ BINUS University</p>
        <ul style="margin-bottom: 0; font-size: 16px; line-height: 1.8; list-style-type: none; padding-left: 0;">
            <li>👤 <b>Yosuke Yung</b> - 2802428066</li>
            <li>👤 <b>Wisely Janson Halim</b> - 2802467382</li>
            <li>👤 <b>Marcellino Varian Saputra</b> - 2802457652</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- 3. HALAMAN EDA ---
def show_eda():
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    # Load dataset otomatis
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("⚠️ File **air_quality.csv** tidak ditemukan! Pastikan Anda sudah mengunduh dataset dari Kaggle dan menyimpannya di folder yang sama dengan aplikasi ini.")
        return

    # 1. Tinjauan Dataset
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Tinjauan Dataset")
    
    col_chk1, col_chk2, col_chk3 = st.columns(3)
    with col_chk1: show_data = st.checkbox("Lihat Potongan Dataset")
    with col_chk2: show_info = st.checkbox("Lihat Informasi Data")
    with col_chk3: show_desc = st.checkbox("Lihat Statistik")

    if show_data:
        st.write("**Potongan Dataset (5 Baris Pertama):**")
        st.dataframe(df.head())
    if show_info:
        st.write("**Informasi Dataset (df.info):**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    if show_desc:
        st.write("**Statistik Deskriptif (df.describe):**")
        st.dataframe(df.describe())
    st.markdown('</div>', unsafe_allow_html=True)

    # Pisahkan kolom numerik dan kategorikal
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    all_cols = df.columns.tolist()

    # 2. Analisa Univariat
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Analisis Univariat")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uni_feature = st.selectbox("Pilih Fitur:", all_cols, key="uni_feat")
    with col_u2:
        uni_plot = st.selectbox("Pilih Jenis Visualisasi:", ["Histogram", "Boxplot", "Bar Chart"], key="uni_plot")
    
    if uni_plot == "Histogram":
        fig_uni = px.histogram(df, x=uni_feature, color_discrete_sequence=['#2c7a7b'])
    elif uni_plot == "Boxplot":
        fig_uni = px.box(df, y=uni_feature, color_discrete_sequence=['#38b2ac'])
    else:
        val_counts = df[uni_feature].value_counts().reset_index()
        val_counts.columns = [uni_feature, 'Count']
        fig_uni = px.bar(val_counts, x=uni_feature, y='Count', color_discrete_sequence=['#3182ce'])
    
    st.plotly_chart(fig_uni, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Analisa Bivariat
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3. Analisis Bivariat")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        bi_feat_x = st.selectbox("Pilih Fitur X:", all_cols, key="bi_x")
    with col_b2:
        bi_feat_y = st.selectbox("Pilih Fitur Y:", all_cols, key="bi_y")
    with col_b3:
        bi_plot = st.selectbox("Pilih Jenis Visualisasi:", ["Scatter Plot", "Boxplot"], key="bi_plot")

    if bi_plot == "Scatter Plot":
        fig_bi = px.scatter(df, x=bi_feat_x, y=bi_feat_y, color_discrete_sequence=['#2c7a7b'])
    else:
        fig_bi = px.box(df, x=bi_feat_x, y=bi_feat_y, color_discrete_sequence=['#38b2ac'])
    
    st.plotly_chart(fig_bi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Analisa Multivariat (Heatmap)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4. Analisis Multivariat (Heatmap Korelasi)")
    show_heatmap = st.checkbox("Tampilkan Heatmap Korelasi")
    if show_heatmap:
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Teal")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Butuh minimal 2 kolom numerik untuk membuat heatmap.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Feature Selection ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("5. Feature Selection")
    
    all_features = [col for col in df.columns if col != 'Air Quality']
    
    # Gunakan key agar status toggle dan multiselect tersimpan
    use_recommended = st.toggle("Gunakan *Recommended Features*", value=True, key="toggle_rec")
    
    default_selection = all_features if use_recommended else st.session_state.get('selected_features', [])
    
    selected_features = st.multiselect(
        "Pilih Fitur (X):", 
        all_features, 
        default=default_selection, 
        key="ms_features"
    )

    if st.button("Konfirmasi Feature Selection", type="primary"):
        if selected_features:
            st.session_state['selected_features'] = selected_features
            st.session_state['feature_confirmed'] = True
            st.success(f"✅ {len(selected_features)} Fitur dikonfirmasi!")
        else:
            st.error("Pilih minimal satu fitur.")

    # TAMPILAN PERSISTEN: Muncul terus selama sudah pernah dikonfirmasi
    if st.session_state.get('feature_confirmed'):
        st.write("### 📋 Fitur yang Digunakan:")
        st.dataframe(df[st.session_state['selected_features']].head())
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. HALAMAN PREPROCESSING ---
def show_preprocessing():
    st.title("🛠️ Data Preprocessing")
    
    if 'feature_confirmed' not in st.session_state or not st.session_state['feature_confirmed']:
        st.warning("🔒 Silakan selesaikan dan konfirmasi **Feature Selection** di halaman EDA terlebih dahulu.")
        return

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("⚠️ File air_quality.csv tidak ditemukan.")
        return

    X = df[st.session_state['selected_features']]
    y = df['Air Quality']

    # 1. Train-Test Split (TAMBAHKAN KEY AGAR INPUT TIDAK HILANG)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Train-Test Split")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 50, 20, key="prep_test_size") / 100
    with col2:
        random_state = st.number_input("Random State", value=42, key="prep_random_state")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Scaling (TAMBAHKAN KEY)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Data Scaling")
    scaler_type = st.selectbox("Pilih Scaler:", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="prep_scaler_type")
    
    # Penjelasan kegunaan scaling
    if scaler_type == "StandardScaler":
        st.info("💡 **StandardScaler**: Mengubah data sehingga memiliki rata-rata 0 dan standar deviasi 1. Sangat cocok jika data fitur memiliki distribusi yang mendekati normal (Gaussian).")
    elif scaler_type == "MinMaxScaler":
        st.info("💡 **MinMaxScaler**: Mengubah skala data ke dalam rentang tertentu, biasanya 0 hingga 1. Cocok digunakan jika distribusi data tidak normal atau algoritma yang akan digunakan sangat sensitif terhadap besaran nilai.")
    else:
        st.info("💡 **RobustScaler**: Melakukan pemusatan dan penskalaan berdasarkan persentil (median dan kuartil). Sangat tangguh (robust) dan direkomendasikan jika dataset memiliki banyak *outlier* (pencilan).")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Label Encoding Target
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3. Target Encoding")
    st.write("Mengubah kategori Kualitas Udara menjadi angka (Label Encoding):")
    
    mapping_data = {
        "Category": ["🟢 Good", "🔴 Hazardous", "🟡 Moderate", "🟠 Poor"],
        "Encoded Value": [0, 1, 2, 3]
    }
    st.table(pd.DataFrame(mapping_data))
    
    mapping = {'Good': 0, 'Hazardous': 1, 'Moderate': 2, 'Poor': 3}
    y_encoded = y.map(mapping)

    if st.button("Jalankan Preprocessing", type="primary"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
        
        # Inisialisasi Scaler
        if scaler_type == "StandardScaler": 
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler": 
            scaler = MinMaxScaler()
        else: 
            scaler = RobustScaler()
        
        # Scaling fitur dan mengembalikannya dalam bentuk DataFrame agar tabelnya rapi
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        st.success("✨ Preprocessing Selesai!")
        
        # Simpan hasil preprocessing ke session_state untuk digunakan di page Model
        st.session_state['preprocessed_data'] = {
            'X_train': X_train_scaled, 
            'X_test': X_test_scaled,
            'y_train': y_train.reset_index(drop=True), 
            'y_test': y_test.reset_index(drop=True)
        }
        st.session_state['scaler'] = scaler
        st.session_state['preprocessing_done'] = True 
        
    st.markdown('</div>', unsafe_allow_html=True) # Tutup div card di sini

    # --- KODE UNTUK MENAMPILKAN OUTPUT DI LUAR TOMBOL AGAR PERSISTEN ---
    if st.session_state.get('preprocessing_done'):
        # Ambil data dari session state
        data = st.session_state['preprocessed_data']
        X_train_scaled = data['X_train']
        X_test_scaled = data['X_test']
        y_train_res = data['y_train']
        y_test_res = data['y_test']

        st.markdown("---")
        st.write("### 📊 Hasil Data Preprocessing")
        st.write("Berikut adalah pratinjau data setelah dilakukan splitting dan scaling:")
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Jumlah Data Training", len(X_train_scaled))
        col_res2.metric("Jumlah Data Testing", len(X_test_scaled))

        tab1, tab2, tab3, tab4 = st.tabs(["X_train", "X_test", "y_train", "y_test"])
        
        with tab1:
            st.write("**Data Fitur Training (Scaled):**")
            st.dataframe(X_train_scaled)
        with tab2:
            st.write("**Data Fitur Testing (Scaled):**")
            st.dataframe(X_test_scaled)
        with tab3:
            st.write("**Data Target Training (Encoded):**")
            st.dataframe(y_train_res)
        with tab4:
            st.write("**Data Target Testing (Encoded):**")
            st.dataframe(y_test_res)

# --- 5. HALAMAN MODEL ---
def show_model():
    st.title("⚙️ Model Training")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("🔒 Ups! Data belum siap. Silakan selesaikan tahap **Preprocessing** terlebih dahulu ya!")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Konfigurasi Model Machine Learning")
    
    model_choice = st.selectbox("Pilih Model:", ["Logistic Regression", "Random Forest", "XGBoost"], key="model_choice")
    
    st.markdown("**⚙️ Pengaturan Parameter (Hyperparameters):**")
    
    if model_choice == "Logistic Regression":
        st.info("💡 **Logistic Regression**: Model klasifikasi linear yang cepat dan efisien.")
        col1, col2 = st.columns(2)
        with col1:
            lr_c = st.select_slider("Nilai C (Inverse Regularization):", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key="lr_c")
        with col2:
            lr_iter = st.slider("Max Iterations:", min_value=100, max_value=2000, value=1000, step=100, key="lr_iter")
            
    elif model_choice == "Random Forest":
        st.info("💡 **Random Forest**: Model ensemble berbasis Decision Tree yang tangguh terhadap overfitting.")
        col1, col2 = st.columns(2)
        with col1:
            rf_estimators = st.slider("Jumlah Pohon (n_estimators):", min_value=50, max_value=500, value=100, step=50, key="rf_est")
        with col2:
            rf_depth = st.slider("Kedalaman Maksimal (max_depth):", min_value=3, max_value=20, value=10, key="rf_depth")
            
    else: # XGBoost
        st.info("💡 **XGBoost**: Algoritma boosting tingkat lanjut yang sangat akurat dan presisi.")
        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_estimators = st.slider("Jumlah Estimator:", min_value=50, max_value=500, value=100, step=50, key="xgb_est")
        with col2:
            xgb_lr = st.select_slider("Learning Rate:", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key="xgb_lr")
        with col3:
            xgb_depth = st.slider("Max Depth:", min_value=3, max_value=15, value=6, key="xgb_depth")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Mulai Training")
    st.write("Klik tombol di bawah untuk melatih model dengan parameter di atas.")
    
    if st.button("🚀 Train Model Sekarang!", type="primary"):
        X_train = st.session_state['preprocessed_data']['X_train']
        y_train = st.session_state['preprocessed_data']['y_train']
        
        with st.status("👩‍💻 Menjalankan proses training... (Tunggu sebentar ya!)", expanded=True) as status:
            st.write("🔍 Memuat data preprocessed...")
            time.sleep(0.5)
            
            if model_choice == "Logistic Regression":
                st.write("🧮 Mengonfigurasi persamaan Logistic Regression...")
                model = LogisticRegression(C=lr_c, max_iter=lr_iter, random_state=42)
            elif model_choice == "Random Forest":
                st.write("🌲 Menanam pohon-pohon keputusan (Decision Trees)...")
                model = RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth, random_state=42)
            else:
                st.write("🔥 Mempersiapkan XGBoost (Siap-siap, ini mungkin agak lama)...")
                model = XGBClassifier(n_estimators=xgb_estimators, learning_rate=xgb_lr, max_depth=xgb_depth, random_state=42, eval_metric='mlogloss')
            
            time.sleep(0.5)
            st.write("🧠 Model sedang belajar dan mengenali pola lingkungan...")
            model.fit(X_train, y_train)
            
            st.session_state['trained_model'] = model
            st.session_state['model_name'] = model_choice
            st.session_state['model_trained'] = True # Tanda bahwa model sudah selesai di-training
            
            # Jika training model baru, hapus hasil evaluasi lama supaya tidak nyangkut
            if 'eval_results' in st.session_state:
                del st.session_state['eval_results']
            
            status.update(label="✨ Yeay! Training Selesai!", state="complete", expanded=False)
        
        if model_choice == "XGBoost":
            st.snow() 
            st.toast("🔥 BOOM! XGBoost super cerdas berhasil dilatih!", icon="😎")
        else:
            st.balloons()
            st.toast("🎉 Yuhuu! Model berhasil pintar!", icon="🥳")
            
    # TAMPILAN PERSISTEN: Muncul terus selama model sudah ditraining
    if st.session_state.get('model_trained'):
        st.success(f"✅ Model **{st.session_state['model_name']}** siap digunakan!")
        st.info("👉 **Langkah Selanjutnya:** Lanjut ke menu **Evaluation** di *sidebar* untuk melihat akurasi dan laporannya!")
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- 6. HALAMAN EVALUATION ---
def show_evaluation():
    st.title("📈 Model Evaluation")
    
    if 'trained_model' not in st.session_state:
        st.warning("🔒 Ups! Kamu belum melatih model. Silakan ke menu **Model** untuk melakukan training terlebih dahulu.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Prediksi Data Testing")
    
    model = st.session_state['trained_model']
    model_name = st.session_state['model_name']
    X_test = st.session_state['preprocessed_data']['X_test']
    y_test = st.session_state['preprocessed_data']['y_test']
    
    if 'eval_results' not in st.session_state:
        with st.spinner("Menghitung hasil evaluasi..."):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            st.session_state['eval_results'] = {
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
    y_pred = st.session_state['eval_results']['y_pred']
    y_proba = st.session_state['eval_results']['y_proba']
    
    st.success(f"✅ Evaluasi menggunakan model **{model_name}** berhasil dilakukan pada {len(y_test)} data testing!")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. Pilih Metrik Evaluasi")
    
    metrics_options = ["Classification Report", "Confusion Matrix", "ROC-AUC Curve"]
    
    # Menjaga state metrik agar persisten
    if 'selected_metrics' not in st.session_state:
        st.session_state['selected_metrics'] = []
        
    selected_metrics = st.multiselect(
        "Pilih metrik:", 
        metrics_options, 
        default=st.session_state['selected_metrics']
    )
    # Update session state secara manual setiap ada perubahan
    st.session_state['selected_metrics'] = selected_metrics
    st.markdown('</div>', unsafe_allow_html=True)

    class_names = ['Good', 'Hazardous', 'Moderate', 'Poor']

    if "Classification Report" in selected_metrics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📑 Classification Report")
        st.info("💡 **Precision**: Akurasi tebakan positif. **Recall**: Data aktual yang berhasil ditebak. **F1-Score**: Keseimbangan keduanya.")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.background_gradient(cmap='Greens').format(precision=3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if "Confusion Matrix" in selected_metrics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            x=class_names, y=class_names,
            labels=dict(x="Tebakan Model", y="Data Asli", color="Jumlah Data")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if "ROC-AUC Curve" in selected_metrics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 ROC-AUC Curve (One-vs-Rest)")
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        fig_roc = go.Figure()
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{class_names[i]} (AUC = {roc_auc:.3f})"))
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. HALAMAN TESTING ---
def show_testing():
    st.title("🧪 Testing Prediction")
    
    if 'trained_model' not in st.session_state:
        st.warning("🔒 Selesaikan training model terlebih dahulu di halaman Model!")
        return
    if 'scaler' not in st.session_state:
        st.error("⚠️ Scaler tidak ditemukan! Pastikan sudah menyelesaikan Preprocessing.")
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Input Data Lingkungan Baru")
    st.write("Masukkan nilai fitur-fitur di bawah ini. Tekan tombol `Tab` pada keyboard untuk berpindah ke kolom berikutnya.")
    
    # Konfigurasi batas atas/bawah
    feature_configs = {
        'Temperature': {'label': 'Temperature (°C)', 'min': -20.0, 'max': 60.0, 'val': 28.0, 'step': 0.1},
        'Humidity': {'label': 'Humidity (%)', 'min': 0.0, 'max': 100.0, 'val': 60.0, 'step': 1.0},
        'PM2.5': {'label': 'PM2.5 (µg/m³)', 'min': 0.0, 'max': 500.0, 'val': 15.0, 'step': 1.0},
        'PM10': {'label': 'PM10 (µg/m³)', 'min': 0.0, 'max': 500.0, 'val': 25.0, 'step': 1.0},
        'NO2': {'label': 'NO2 (ppb)', 'min': 0.0, 'max': 200.0, 'val': 20.0, 'step': 1.0},
        'SO2': {'label': 'SO2 (ppb)', 'min': 0.0, 'max': 200.0, 'val': 10.0, 'step': 1.0},
        'CO': {'label': 'CO (ppm)', 'min': 0.0, 'max': 50.0, 'val': 1.5, 'step': 0.1},
        'Proximity_to_Industrial_Areas': {'label': 'Proximity to Industry (km)', 'min': 0.0, 'max': 100.0, 'val': 5.0, 'step': 0.1},
        'Population_Density': {'label': 'Population Density (orang/km²)', 'min': 0, 'max': 50000, 'val': 500, 'step': 10}
    }

    # URUTAN SPESIFIK sesuai permintaan agar Tabbing urut dari Kiri ke Kanan
    ordered_features = [
        'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 
        'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density'
    ]
    
    selected_features = st.session_state['selected_features']
    # Filter fitur berdasarkan urutan spesifik
    display_features = [f for f in ordered_features if f in selected_features]
    # Jika ada fitur tambahan yang tidak terdaftar di ordered_features (untuk safety)
    display_features += [f for f in selected_features if f not in ordered_features]

    user_inputs = {}
    
    # RENDER BARIS DEMI BARIS (Mengatasi isu tabbing atas ke bawah)
    for i in range(0, len(display_features), 2):
        cols = st.columns(2)
        # Looping 2 kolom untuk baris saat ini
        for j in range(2):
            if i + j < len(display_features):
                feature = display_features[i + j]
                config = feature_configs.get(feature, {'label': feature, 'min': 0.0, 'max': 1000.0, 'val': 0.0, 'step': 1.0})
                
                with cols[j]:
                    if isinstance(config['val'], int):
                        user_inputs[feature] = st.number_input(
                            config['label'], 
                            min_value=int(config['min']), max_value=int(config['max']), 
                            value=int(config['val']), step=int(config['step']), 
                            key=f"test_input_{feature}"
                        )
                    else:
                        user_inputs[feature] = st.number_input(
                            config['label'], 
                            min_value=float(config['min']), max_value=float(config['max']), 
                            value=float(config['val']), step=float(config['step']), 
                            key=f"test_input_{feature}"
                        )
            
    if st.button("🔮 Hasilkan Prediksi", type="primary", use_container_width=True):
        with st.spinner("Menganalisis data..."):
            input_df = pd.DataFrame([user_inputs])
            
            # Reorder input dataframe agar urutannya sama persis dengan X_train
            X_train_cols = st.session_state['preprocessed_data']['X_train'].columns
            input_df = input_df[X_train_cols]
            
            scaler = st.session_state['scaler']
            input_scaled = scaler.transform(input_df)
            
            model = st.session_state['trained_model']
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            reverse_mapping = {0: "🟢 Good", 1: "🔴 Hazardous", 2: "🟡 Moderate", 3: "🟠 Poor"}
            result_label = reverse_mapping.get(pred, "Unknown")
            confidence = max(proba) * 100
            
            st.session_state['test_prediction'] = {
                'label': result_label,
                'confidence': confidence,
                'inputs': input_df
            }
    st.markdown('</div>', unsafe_allow_html=True)

    if 'test_prediction' in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Hasil Prediksi")
        
        res = st.session_state['test_prediction']
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #e0ecff; border-radius: 15px; border: 2px solid #3182ce;">
            <h3 style="color: #2c7a7b; margin-bottom: 5px;">Kualitas Udara Diprediksi:</h3>
            <h1 style="font-size: 3.5rem; margin: 0;">{res['label']}</h1>
            <p style="font-size: 1.2rem; color: #2f855a; margin-top: 10px;">Tingkat Kepercayaan (Confidence Score): <b>{res['confidence']:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---
def main():
    set_ui_style()
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Home"

    # --- SETUP SIDEBAR BARU ---
    st.sidebar.markdown(
        "<h2 style='text-align: center; color: #2c7a7b; font-weight: 800; "
        "border-bottom: 2px solid #38b2ac; padding-bottom: 10px; margin-bottom: 20px;'>"
        "Navigation</h2>", 
        unsafe_allow_html=True
    )
    
    # Menu menggunakan ikon agar mirip gambar referensi
    menu = ["🏠 Home", "📊 EDA", "🛠️ Preprocessing", "⚙️ Model", "📈 Evaluation", "🧪 Testing"]
    
    choice = st.sidebar.radio("Menu", menu, key="current_page", label_visibility="collapsed")

    # Memberikan Jarak Kosong agar Tombol Reset ada di paling bawah
    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid #cbd5e1; margin-bottom: 15px;'>", unsafe_allow_html=True)
    
    # Tombol Reset (Akan berwarna merah dari CSS)
    st.sidebar.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.sidebar.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.session_state['current_page'] = "🏠 Home"
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # --- PENGARAHAN HALAMAN ---
    if choice == "🏠 Home": 
        show_home()
    elif choice == "📊 EDA": 
        show_eda()
    elif choice == "🛠️ Preprocessing": 
        show_preprocessing()
    elif choice == "⚙️ Model":
        show_model()
    elif choice == "📈 Evaluation":
        show_evaluation()
    elif choice == "🧪 Testing":
        show_testing()

if __name__ == "__main__":
    main()