"""
automate_NamaSiswa.py
Script otomatisasi preprocessing dataset Heart Disease UCI
Konversi dari proses eksperimen di notebook menjadi pipeline otomatis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.
    
    Args:
        filepath: Path ke file CSV dataset
    Returns:
        DataFrame dataset
    """
    logger.info(f"Memuat dataset dari: {filepath}")
    
    if not os.path.exists(filepath):
        # Buat dataset simulasi jika file tidak ditemukan
        logger.warning(f"File tidak ditemukan. Membuat dataset simulasi...")
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            'age': np.random.randint(29, 77, n),
            'sex': np.random.randint(0, 2, n),
            'cp': np.random.randint(0, 4, n),
            'trestbps': np.random.randint(94, 200, n),
            'chol': np.random.randint(126, 564, n),
            'fbs': np.random.randint(0, 2, n),
            'restecg': np.random.randint(0, 3, n),
            'thalach': np.random.randint(71, 202, n),
            'exang': np.random.randint(0, 2, n),
            'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
            'slope': np.random.randint(0, 3, n),
            'ca': np.random.randint(0, 4, n),
            'thal': np.random.choice([0, 1, 2, 3], n),
            'target': np.random.randint(0, 2, n)
        })
        # Tambahkan missing values simulasi
        df.loc[np.random.choice(df.index, 5), 'chol'] = np.nan
        df.loc[np.random.choice(df.index, 3), 'thalach'] = np.nan
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset simulasi dibuat dan disimpan di {filepath}")
    else:
        df = pd.read_csv(filepath)
    
    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan imputasi median untuk fitur numerik.
    
    Args:
        df: DataFrame input
    Returns:
        DataFrame tanpa missing values
    """
    logger.info("Menangani missing values...")
    df_clean = df.copy()
    
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df_clean.isnull().sum().sum()
    
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            logger.info(f"  Kolom '{col}': mengisi {df[col].isnull().sum()} nilai kosong dengan median={median_val:.2f}")
    
    missing_after = df_clean.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus data duplikat.
    
    Args:
        df: DataFrame input
    Returns:
        DataFrame tanpa duplikat
    """
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    logger.info(f"Menghapus duplikat: {before} -> {after} baris ({before - after} dihapus)")
    return df_clean


def handle_outliers(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Clipping (Winsorization).
    
    Args:
        df: DataFrame input
        features: List fitur numerik yang akan diproses
    Returns:
        DataFrame dengan outlier yang sudah ditangani
    """
    logger.info("Menangani outlier dengan IQR Clipping...")
    df_clean = df.copy()
    
    for feature in features:
        if feature in df_clean.columns:
            Q1 = df_clean[feature].quantile(0.25)
            Q3 = df_clean[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[feature] = df_clean[feature].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"  {feature}: clip ke [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df_clean


def encode_features(df: pd.DataFrame, categorical_ohe: list) -> pd.DataFrame:
    """
    Melakukan One-Hot Encoding pada fitur kategorikal.
    
    Args:
        df: DataFrame input
        categorical_ohe: List kolom untuk One-Hot Encoding
    Returns:
        DataFrame setelah encoding
    """
    logger.info(f"Melakukan One-Hot Encoding pada: {categorical_ohe}")
    
    # Filter hanya kolom yang ada
    valid_cols = [col for col in categorical_ohe if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols, dtype=int)
    
    logger.info(f"Kolom sebelum encoding: {df.shape[1]} -> setelah: {df_encoded.shape[1]}")
    return df_encoded


def split_data(df: pd.DataFrame, target_col: str = 'target',
               test_size: float = 0.2, random_state: int = 42):
    """
    Membagi data menjadi train dan test set.
    
    Args:
        df: DataFrame input
        target_col: Nama kolom target
        test_size: Proporsi data test
        random_state: Random state untuk reproduktibilitas
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Membagi data: train={1-test_size:.0%}, test={test_size:.0%}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   numerical_features: list):
    """
    Melakukan standarisasi pada fitur numerik.
    
    Args:
        X_train: Training features
        X_test: Testing features
        numerical_features: List fitur numerik yang akan distandarisasi
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    logger.info(f"Standarisasi fitur: {numerical_features}")
    
    # Filter hanya kolom yang ada
    valid_features = [f for f in numerical_features if f in X_train.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[valid_features] = scaler.fit_transform(X_train[valid_features])
    X_test_scaled[valid_features] = scaler.transform(X_test[valid_features])
    
    logger.info("Standarisasi selesai!")
    return X_train_scaled, X_test_scaled, scaler


def save_outputs(X_train, X_test, y_train, y_test, scaler,
                 output_dir: str = 'heart_disease_preprocessing'):
    """
    Menyimpan semua output preprocessing.
    
    Args:
        X_train, X_test, y_train, y_test: Split dataset
        scaler: Fitted scaler
        output_dir: Direktori output
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Menyimpan output ke: {output_dir}/")
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Simpan dataset lengkap
    train_full = X_train.copy()
    train_full['target'] = y_train.values
    train_full.to_csv(f'{output_dir}/train_preprocessed.csv', index=False)
    
    test_full = X_test.copy()
    test_full['target'] = y_test.values
    test_full.to_csv(f'{output_dir}/test_preprocessed.csv', index=False)
    
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f'{output_dir}/feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    
    logger.info(f"Output tersimpan:")
    logger.info(f"  - X_train.csv: {X_train.shape}")
    logger.info(f"  - X_test.csv:  {X_test.shape}")
    logger.info(f"  - y_train.csv: {y_train.shape}")
    logger.info(f"  - y_test.csv:  {y_test.shape}")
    logger.info(f"  - scaler.pkl")
    logger.info(f"  - feature_columns.pkl")
    logger.info(f"  - train_preprocessed.csv")
    logger.info(f"  - test_preprocessed.csv")


def preprocess(input_path: str = 'heart_disease_raw/heart_disease.csv',
               output_dir: str = 'heart_disease_preprocessing'):
    """
    Fungsi utama pipeline preprocessing data.
    
    Args:
        input_path: Path ke dataset raw
        output_dir: Direktori untuk menyimpan hasil preprocessing
    Returns:
        X_train, X_test, y_train, y_test yang sudah siap dilatih
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING HEART DISEASE DATASET")
    logger.info("=" * 60)
    
    # Step 1: Load data
    df = load_data(input_path)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 4: Handle outliers
    outlier_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    df = handle_outliers(df, outlier_features)
    
    # Step 5: Encode categorical features
    categorical_ohe = ['cp', 'restecg', 'slope', 'thal']
    df = encode_features(df, categorical_ohe)
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(df, target_col='target')
    
    # Step 7: Scale features
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    X_train, X_test, scaler = scale_features(X_train, X_test, numerical_features)
    
    # Step 8: Save outputs
    save_outputs(X_train, X_test, y_train, y_test, scaler, output_dir)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI! Data siap untuk pelatihan model.")
    logger.info("=" * 60)
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing Heart Disease Dataset')
    parser.add_argument('--input', type=str,
                        default='heart_disease_raw/heart_disease.csv',
                        help='Path ke file dataset raw')
    parser.add_argument('--output', type=str,
                        default='heart_disease_preprocessing',
                        help='Direktori output hasil preprocessing')
    
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = preprocess(args.input, args.output)
