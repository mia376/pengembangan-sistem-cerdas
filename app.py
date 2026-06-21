import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# KONFIGURASI
# =====================================================

st.set_page_config(
    page_title="Klasterisasi Padi Sumbar",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
<style>

/* Background Utama */
.stApp {
    background: linear-gradient(
        135deg,
        #f1f8e9 0%,
        #dcedc8 30%,
        #c8e6c9 60%,
        #a5d6a7 100%
    );
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1b5e20 0%,
        #2e7d32 50%,
        #388e3c 100%
    );
}

/* Tulisan Sidebar */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Judul */
h1 {
    color: #1b5e20 !important;
    font-weight: 800;
}

/* Sub Judul */
h2, h3 {
    color: #2e7d32 !important;
}

/* Metric Card */
[data-testid="metric-container"] {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f1f8e9
    );

    border: 2px solid #81c784;

    padding: 20px;

    border-radius: 18px;

    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 15px;
}

/* Success Box */
.stAlert {
    border-radius: 15px;
}

/* Selectbox */
.stSelectbox > div > div {
    border-radius: 12px;
}

/* Radio Button Area */
div[role="radiogroup"] {
    background-color: rgba(255,255,255,0.25);
    padding: 10px;
    border-radius: 10px;
}

/* Download Button */
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #2e7d32,
        #66bb6a
    );

    color: white;

    border: none;

    border-radius: 12px;

    font-weight: bold;

    height: 50px;
}

/* Hover Download */
.stDownloadButton button:hover {
    background: linear-gradient(
        90deg,
        #1b5e20,
        #4caf50
    );
}

/* Tombol Umum */
.stButton button {
    background: linear-gradient(
        90deg,
        #43a047,
        #81c784
    );

    color: white;

    border-radius: 12px;

    border: none;
}

/* Plotly Container */
.element-container {
    border-radius: 15px;
}

/* Garis Pemisah */
hr {
    border: 1px solid #81c784;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #e8f5e9;
}

::-webkit-scrollbar-thumb {
    background: #4caf50;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    def baca_file(path, tahun):

        df = pd.read_excel(
            path,
            skiprows=4,
            header=None
        )

        df = df.iloc[:, :4]

        df.columns = [
            "Kabupaten_Kota",
            "Luas_Panen",
            "Produksi",
            "Produktivitas"
        ]

        df["Tahun"] = tahun

        df = df[
            df["Kabupaten_Kota"] !=
            "Provinsi Sumatera Barat"
        ]

        return df

    data = []

    for tahun in [2021, 2022, 2023, 2024, 2025]:

        path = f"data/{tahun}.xlsx"

        data.append(
            baca_file(path, tahun)
        )

    return pd.concat(
        data,
        ignore_index=True
    )

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("🌾 Klasterisasi Produksi Padi Sumatera Barat")
st.markdown(
    "Menggunakan Algoritma K-Means"
)

# =====================================================
# MENU
# =====================================================

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Dashboard",
        "Clustering",
        "Detail Kabupaten"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.header("📊 Dashboard")

    tahun = st.selectbox(
        "Pilih Tahun",
        sorted(df["Tahun"].unique())
    )

    data_tahun = df[
        df["Tahun"] == tahun
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Jumlah Daerah",
        len(data_tahun)
    )

    c2.metric(
        "Total Produksi",
        f"{data_tahun['Produksi'].sum():,.0f} Ton"
    )

    c3.metric(
        "Total Luas Panen",
        f"{data_tahun['Luas_Panen'].sum():,.0f} Ha"
    )

    c4.metric(
        "Produktivitas",
        f"{data_tahun['Produktivitas'].mean():.2f}"
    )

    fig = px.bar(
        data_tahun.sort_values(
            "Produksi",
            ascending=False
        ),
        x="Kabupaten_Kota",
        y="Produksi",
        title=f"Produksi Padi Tahun {tahun}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# CLUSTERING
# =====================================================

elif menu == "Clustering":

    st.header("🤖 K-Means Clustering")

    tahun = st.selectbox(
        "Pilih Tahun",
        sorted(df["Tahun"].unique())
    )

    data_tahun = df[
        df["Tahun"] == tahun
    ].copy()

    k = st.slider(
        "Jumlah Cluster",
        2,
        5,
        3
    )

    X = data_tahun[
        [
            "Luas_Panen",
            "Produksi",
            "Produktivitas"
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    data_tahun["Cluster"] = model.fit_predict(
        X_scaled
    )

    # ==========================
    # UBAH NOMOR CLUSTER
    # MENJADI TINGGI/SEDANG/RENDAH
    # ==========================

    rata_cluster = (
        data_tahun.groupby("Cluster")["Produksi"]
        .mean()
        .sort_values(ascending=False)
    )

    nama_cluster = {}

    if k == 3:

        nama_cluster[
            rata_cluster.index[0]
        ] = "Tinggi"

        nama_cluster[
            rata_cluster.index[1]
        ] = "Sedang"

        nama_cluster[
            rata_cluster.index[2]
        ] = "Rendah"

    else:

        for i, c in enumerate(
            rata_cluster.index
        ):
            nama_cluster[c] = f"Cluster {i+1}"

    data_tahun[
        "Kategori"
    ] = data_tahun[
        "Cluster"
    ].map(
        nama_cluster
    )

    score = silhouette_score(
        X_scaled,
        data_tahun["Cluster"]
    )

    st.success(
        f"Silhouette Score : {score:.3f}"
    )

    st.subheader(
        "Hasil Clustering"
    )

    st.dataframe(
        data_tahun[
            [
                "Kabupaten_Kota",
                "Luas_Panen",
                "Produksi",
                "Produktivitas",
                "Kategori"
            ]
        ],
        use_container_width=True
    )

  
    fig = px.scatter(
        data_tahun,
        x="Produksi",
        y="Produktivitas",
        color="Kategori",
        size="Luas_Panen",
        hover_name="Kabupaten_Kota",
        title="Visualisasi Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = data_tahun.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Hasil",
        csv,
        "hasil_clustering.csv",
        "text/csv"
    )

# =====================================================
# DETAIL DAERAH
# =====================================================

elif menu == "Detail Kabupaten":

    st.header(
        "🏙 Detail Kabupaten/Kota"
    )

    daerah = st.selectbox(
        "Pilih Daerah",
        sorted(
            df[
                "Kabupaten_Kota"
            ].unique()
        )
    )

    detail = df[
        df[
            "Kabupaten_Kota"
        ] == daerah
    ]

    st.dataframe(
        detail,
        use_container_width=True
    )

    fig = px.line(
        detail,
        x="Tahun",
        y="Produksi",
        markers=True,
        title=f"Tren Produksi {daerah}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )