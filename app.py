import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model dan encoder
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('loo_encoder.pkl', 'rb') as f:
    loo_encoder = pickle.load(f)

with open('ordinal_encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Atur halaman
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="wide")

# Sidebar info
with st.sidebar:
    st.image("car.jpg", width=200)
    st.markdown("""
    ## ℹ️ Tentang Aplikasi
    Aplikasi ini menggunakan model Machine Learning untuk memprediksi harga mobil bekas di Arab Saudi.
    
    **Gunakan formulir di kanan untuk mengisi fitur mobil.**
    
    **Model:** XGBoost  
    **Data:** Mobil Bekas Arab Saudi
    """)

# Gambar header
# st.image("car.jpg", use_column_width=True)

# Judul
st.markdown("<h1 style='text-align: center; color: #0A9396;'>🚘 Prediksi Harga Mobil Bekas di Arab Saudi</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Form input
with st.form("form_prediksi"):
    st.markdown("### 📋 Masukkan Detail Mobil")

    col1, col2, col3 = st.columns(3)

    with col1:
        type_ = st.selectbox("🚗 Tipe Mobil", ['C300', 'Sunny', 'Elantra', 'Accord', 'Land Cruiser', 'Impala', 'Yaris', 'Camry',
    'Patrol', 'Tahoe', 'Corolla', 'Copper', 'Prado', 'Civic', 'Furniture', 'RX',
    'Yukon', 'Bus Urvan', 'Aurion', 'Malibu', 'Rav4', 'CX9', 'Expedition', 'ES',
    'Cadenza', 'Tucson', 'Platinum', 'G80', 'Accent', 'Sonata', 'LX', 'GX', 'Azera',
    'CT-S', 'EC7', 'ZS', 'Kona', 'Grand Cherokee', 'S', 'M', 'Charger', 'Taurus', 'E',
    'Royal', 'Picanto', 'Power', 'Datsun', 'F-Pace', 'Hilux', 'Suburban', 'Explorer',
    'Range Rover', 'FJ', 'Senta fe', 'Optima', 'GS8', 'Maxima', 'Caprice',
    'Challenger', 'Camaro', 'Symbol', 'Fluence', '6', 'RX5', 'Avalon', 'APV', '3008',
    'Cerato', 'Traverse', 'Sierra', 'F150', 'Genesis', 'NX', 'C', 'G70', 'Flex', 'UX',
    'Cores', 'Creta', 'Rio', 'Odyssey', 'Sylvian Bus', 'H1', 'Ciocca',
    'Land Cruiser Pickup', 'Cressida', 'Duster', 'Seven', 'GLC', 'Carnival', 'EC8',
    'H6', '300', 'The 7', 'Z370', 'Spark', 'Attrage', 'Focus', 'X-Trail', 'Forester',
    'Pick up', 'The 4', 'GS', 'Pajero', 'Acadia', 'City', 'Echo Sport', 'Vego', 'CLA',
    'The 5', 'Silverado', 'Cherokee', 'Altima', 'X', 'Navigator', 'Wrangler', 'XT5',
    'Cruze', 'Navara', 'Gran Max', 'Innova', 'Aveo', 'Soul', 'Sportage', 'Montero',
    'Prestige', 'Sentra', 'Dokker', 'Veloster', 'Fusion', 'Land Cruiser 70',
    'Pathfinder', 'Seltos', 'Behbehani', 'Victoria', 'LS', 'CX5', 'Emgrand',
    'Carenz', 'SEL', 'The 6', 'Marquis', 'H2', 'Talisman', 'Mustang', '5008', 'A',
    'T77', 'Optra', 'Safrane', 'QX', 'Tiggo', 'Durango', 'Eado', 'MKS', 'CT5',
    'Panamera', 'CS35', 'Coolray', 'Countryman', 'D-MAX', 'Partner', 'Capture',
    '301', 'Pilot', 'Previa', 'X-Terra', 'Other', 'CS75', 'Outlander', 'Sorento',
    'Touareg', 'Safari', 'HRV', '3', 'Q5', 'CT4', 'MKX', 'S5', 'A6', 'X7', 'Rush', '2',
    'Delta', 'VTC', 'IS', 'Cayenne', 'Blazer', 'CS35 Plus', 'KICKS', 'Q', 'CS85',
    'Armada', 'Echo', 'Avanza', 'Terios Ground', 'CX3', 'S300', 'Koleos', 'Compass',
    'Edge', 'A8', 'Hiace', 'Lumina', 'ML', '500', 'Macan', 'Passat', 'CLS', 'Stinger',
    'Viano', 'A3', 'RX8', 'RC', 'Escalade', 'MKZ', 'CC', 'C-HR', 'Terrain', 'Mohave',
    'Savana', 'CL', 'Ram', 'Coaster', 'Vitara', 'Juke', 'C200', 'FX', 'Fleetwood',
    'Milan', 'Dyna', 'Cayman', 'Boxer', 'ATS', 'Cadillac', 'Grand Marquis', 'H3',
    'K5', '5', 'Trailblazer', 'Prius', 'CS95', 'F3', 'A5', 'Gamble', 'The 3', 'L200',
    '360', 'Jimny', 'XJ', 'LF X60', 'Van', 'Envoy', 'Patriot', 'GL', 'The M',
    'Flying Spur', 'Cayenne S', 'ASX', 'Golf', 'Coupe S', 'Doblo', 'Bus County',
    'Ranger', 'GLE', 'H-2', 'Z', 'Avante', 'B50', 'Grand Vitara', 'Mini Van',
    'Nativa', 'Beetle', 'Ertiga', '4Runner', 'GS3', 'Quattroporte', 'Azkarra', 'XF',
    'A7', 'Gloria', 'Tuscani', 'Kaptiva', 'Murano', 'DB9', 'Jetta', 'Opirus', 'CRV',
    'Montero2', 'i40', 'Tiguan', 'Logan', 'Town Car', 'Lancer', 'Abeka', 'Dzire',
    'Terios', 'Cayenne Turbo', 'Mini Cooper', 'Z350', 'Nitro', 'Van R', 'Crosstour',
    'SX4', 'Suvana', 'Liberty', 'Coupe', 'Prestige Plus', 'X40', 'Colorado', 'CT6',
    'Fabia', 'Megane', 'Q7', 'Daily', 'Carens', 'A4', 'GC7', 'G330', 'H9', 'Sedona',
    'Cayenne Turbo GTS', 'SRT', 'HS', "D'max", 'Pegas', 'DTS', 'Superb', 'Veracruz',
    '307', 'CX7', 'QQ', 'L300', 'Galant'])
        make = st.selectbox("🏷️ Merek Mobil", ['Chrysler', 'Nissan', 'Hyundai', 'Honda', 'Toyota', 'Chevrolet', 'MINI', 'Lexus',
    'GMC', 'Mazda', 'Ford', 'Kia', 'Genesis', 'Cadillac', 'Geely', 'MG', 'Jeep',
    'Mercedes', 'INFINITI', 'Dodge', 'Great Wall', 'Jaguar', 'Land Rover', 'GAC',
    'Renault', 'Suzuki', 'Peugeot', 'Changan', 'HAVAL', 'BMW', 'Mitsubishi',
    'Subaru', 'Zhengzhou', 'Lincoln', 'Daihatsu', 'FAW', 'Chery', 'Porsche', 'Isuzu',
    'Volkswagen', 'Audi', 'Fiat', 'Mercury', 'Classic', 'Hummer', 'BYD', 'Maserati',
    'Lifan', 'Bentley', 'Foton', 'Aston Martin', 'Other', 'Victory Auto', 'Škoda',
    'Iveco'])
        region = st.selectbox("📍 Wilayah", [ 'Riyadh', 'Jeddah', 'Dammam', 'Al-Medina', 'Qassim', 'Jazan', 'Tabouk', 'Aseer',
    'Al-Ahsa', 'Taef', 'Sabya', 'Makkah', 'Khobar', 'Abha', 'Al-Baha', 'Yanbu',
    'Hail', 'Al-Namas', 'Jubail', 'Al-Jouf', 'Hafar Al-Batin', 'Najran', 'Arar',
    'Wadi Dawasir', 'Besha', 'Qurayyat', 'Sakaka'])

    with col2:
        color = st.selectbox("🎨 Warna Mobil", [ 'Black', 'Silver', 'Grey', 'Navy', 'White', 'Bronze', 'Another Color', 'Golden',
    'Brown', 'Blue', 'Red', 'Oily', 'Green', 'Orange', 'Yellow'])
        gear_type = st.selectbox("⚙️ Transmisi", ['Automatic', 'Manual'])
        origin = st.selectbox("🌍 Asal Mobil", ['Saudi', 'Gulf Arabic', 'Other'])

    with col3:
        options = st.selectbox("🛠️ Opsi Tambahan", ['Full', 'Standard', 'Semi Full'])
        fuel_type = st.selectbox("⛽ Jenis Bahan Bakar", ['Gas', 'Diesel', 'Hybrid'])
        year = st.slider("📆 Tahun Mobil", 1972, 2021, 2015)

    col4, col5 = st.columns(2)
    with col4:
        engine_size = st.number_input("🔧 Ukuran Mesin (L)", min_value=1.0, max_value=9.0, step=0.1, value=2.0)
    with col5:
        mileage = st.number_input("🛣️ Jarak Tempuh (km)", min_value=100, max_value=4500000, step=1000, value=50000)

    submitted = st.form_submit_button("🔍 Prediksi Harga")

# Proses prediksi
if submitted:
    input_df = pd.DataFrame([{
        'Make': make,
        'Type': type_,
        'Year': year,
        'Origin': origin,
        'Color': color,
        'Options': options,
        'Engine_Size': engine_size,
        'Fuel_Type': fuel_type,
        'Gear_Type': gear_type,
        'Mileage': mileage,
        'Region': region,
        'Price': 0
    }])

    df_transformed = loo_encoder.transform(input_df)
    df_transformed = ordinal_encoder.transform(df_transformed)
    df_transformed[['Mileage', 'Year', 'Engine_Size']] = scaler.transform(
        df_transformed[['Mileage', 'Year', 'Engine_Size']]
    )
    df_transformed.drop(columns=['Price'], inplace=True)

    pred = model.predict(df_transformed)
    st.markdown(f"""
    <div style='padding:20px; background-color:#DFF5E1; border-radius:10px; text-align:center'>
        <h2>💰 Perkiraan Harga Mobil:</h2>
        <h1 style='color:#219EBC;'>SAR {int(pred[0]):,}</h1>
    </div>
    """, unsafe_allow_html=True)
