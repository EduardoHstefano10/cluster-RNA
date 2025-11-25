"""
Modelos de Machine Learning V2 para el Sistema de Alerta Temprana
Versión actualizada que trabaja con variables categóricas y PostgreSQL
Incluye Red Neuronal Artificial (RNA) y Clustering K-means
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime
# from database import EstudiantesDB  # <-- Comentar esta línea si no usas PostgreSQL


class CategoricalRiskPredictor:
    """
    Modelo de Red Neuronal para predicción de riesgo académico
    Trabaja con variables categóricas
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.85)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.training_timestamp = None

    def encode_categorical_features(self, df, fit=False):
        """
        Codificar variables categóricas a numéricas
        """
        # Definir todas las variables categóricas
        categorical_features = [
            'sueno_horas', 'actividad_fisica', 'alimentacion', 'estilo_de_vida',
            'estres_academico', 'apoyo_familiar', 'bienestar', 'asistencia',
            'horas_estudio', 'interes_academico', 'rendimiento_academico',
            'historial_academico', 'carga_laboral', 'beca', 'deudor'
        ]

        df_encoded = df.copy()

        for feature in categorical_features:
            if feature not in df.columns:
                continue

            if fit:
                # Crear nuevo encoder
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df[feature].fillna('Unknown'))
                self.label_encoders[feature] = le
            else:
                # Usar encoder existente
                if feature in self.label_encoders:
                    le = self.label_encoders[feature]
                    # Manejar categorías desconocidas
                    df_encoded[feature] = df[feature].fillna('Unknown').apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df_encoded

    def create_risk_labels_from_categorical(self, df):
        """
        Crear etiquetas de riesgo basadas en variables categóricas
        Riesgo: 0=Sin riesgo, 1=Leve, 2=Moderado, 3=Alto, 4=Crítico
        """
        risk_scores = np.zeros(len(df))

        # Si ya existe riesgo_predicho, usarlo
        if 'riesgo_predicho' in df.columns and df['riesgo_predicho'].notna().any():
            risk_map_reverse = {
                'Sin_riesgo': 0,
                'Riesgo_leve': 1,
                'Riesgo_moderado': 2,
                'Riesgo_alto': 3,
                'Riesgo_critico': 4
            }
            return df['riesgo_predicho'].map(risk_map_reverse).fillna(2).astype(int).values

        # Generar labels basado en las variables categóricas
        # Factor 1: Estrés académico
        if 'estres_academico' in df.columns:
            estres_map = {'Leve': 0, 'Moderado': 1, 'Alto': 2, 'Severo': 3, 'Crítico': 4}
            risk_scores += df['estres_academico'].map(estres_map).fillna(1)

        # Factor 2: Rendimiento académico
        if 'rendimiento_academico' in df.columns:
            rendimiento_map = {
                'Logro_destacado': 0,
                'Previsto': 0,
                'En_proceso': 1,
                'En_inicio': 2
            }
            risk_scores += df['rendimiento_academico'].map(rendimiento_map).fillna(1)

        # Factor 3: Asistencia
        if 'asistencia' in df.columns:
            asistencia_map = {'Constante': 0, 'Frecuente': 0, 'Irregular': 2, 'Nula': 3}
            risk_scores += df['asistencia'].map(asistencia_map).fillna(1)

        # Factor 4: Bienestar
        if 'bienestar' in df.columns:
            bienestar_map = {'Saludable': 0, 'Moderado': 1, 'En_riesgo': 2}
            risk_scores += df['bienestar'].map(bienestar_map).fillna(1)

        # Factor 5: Apoyo familiar
        if 'apoyo_familiar' in df.columns:
            apoyo_map = {'Fuerte': 0, 'Moderado': 1, 'Escaso': 2, 'Nulo': 3}
            risk_scores += df['apoyo_familiar'].map(apoyo_map).fillna(1)

        # Factor 6: Deudor
        if 'deudor' in df.columns:
            deudor_map = {
                'Sin_deuda': 0,
                'Retraso_leve': 1,
                'Retraso_moderado': 2,
                'Retraso_crítico': 3
            }
            risk_scores += df['deudor'].map(deudor_map).fillna(0)

        # Clasificar riesgo basado en score acumulado
        risk_labels = np.zeros(len(df), dtype=int)
        risk_labels[risk_scores <= 2] = 0  # Sin riesgo
        risk_labels[(risk_scores > 2) & (risk_scores <= 5)] = 1  # Leve
        risk_labels[(risk_scores > 5) & (risk_scores <= 8)] = 2  # Moderado
        risk_labels[(risk_scores > 8) & (risk_scores <= 11)] = 3  # Alto
        risk_labels[risk_scores > 11] = 4  # Crítico

        return risk_labels

    def train_from_dataframe(self, df):
        """
        Entrenar el modelo con un DataFrame
        """
        if len(df) < 10:
            raise ValueError("Se necesitan al menos 10 muestras para entrenar el modelo")

        print(f"📊 Entrenando con {len(df)} muestras...")

        # Codificar variables categóricas
        df_encoded = self.encode_categorical_features(df, fit=True)

        # Crear etiquetas de riesgo
        y = self.create_risk_labels_from_categorical(df)

        # Seleccionar solo columnas numéricas
        X = df_encoded.select_dtypes(include=[np.number]).drop(
            columns=['id', 'cluster_asignado', 'probabilidad_desercion'],
            errors='ignore'
        )

        # Guardar nombres de features
        self.feature_columns = X.columns.tolist()

        # Normalizar
        X_scaled = self.scaler.fit_transform(X)

        # Aplicar PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Entrenar clustering
        self.kmeans.fit(X_pca)

        # Dividir datos para entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Entrenar red neuronal
        self.neural_network.fit(X_train, y_train)

        # Evaluar
        train_score = self.neural_network.score(X_train, y_train)
        test_score = self.neural_network.score(X_test, y_test)

        self.is_trained = True
        self.training_timestamp = datetime.now()

        print(f"✅ Modelo entrenado exitosamente!")
        print(f"   📈 Train accuracy: {train_score:.4f}")
        print(f"   📉 Test accuracy: {test_score:.4f}")
        print(f"   🔍 Componentes PCA: {self.pca.n_components_}")
        print(f"   🕐 Timestamp: {self.training_timestamp}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_components': self.pca.n_components_,
            'n_samples': len(df),
            'timestamp': str(self.training_timestamp)
        }

    def train_from_database(self):
        """
        Entrenar el modelo directamente desde la base de datos PostgreSQL
        """
        print("🔄 Conectando a PostgreSQL para obtener datos de entrenamiento...")

        db = EstudiantesDB()

        try:
            # Obtener todos los datos de estudiantes
            df = db.get_all_students()

            if df.empty:
                raise ValueError("No hay datos en la base de datos para entrenar")

            print(f"✅ Datos cargados: {len(df)} estudiantes")

            # Entrenar
            results = self.train_from_dataframe(df)

            db.close()
            return results

        except Exception as e:
            db.close()
            raise Exception(f"Error al entrenar desde base de datos: {e}")

    def predict_risk_from_categorical(self, student_data):
        """
        Predecir riesgo para un estudiante usando variables categóricas
        student_data: dict con las variables del estudiante
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")

        # Convertir a DataFrame
        df = pd.DataFrame([student_data])

        # Codificar variables categóricas
        df_encoded = self.encode_categorical_features(df, fit=False)

        # Seleccionar features usadas en entrenamiento
        X = df_encoded[self.feature_columns]

        # Normalizar
        X_scaled = self.scaler.transform(X.values)

        # Aplicar PCA
        X_pca = self.pca.transform(X_scaled)

        # Predecir riesgo
        risk_level = self.neural_network.predict(X_pca)[0]
        risk_proba = self.neural_network.predict_proba(X_pca)[0]

        # Predecir cluster
        cluster = self.kmeans.predict(X_pca)[0]

        # Mapear nivel de riesgo a texto
        risk_map = {
            0: 'Sin riesgo',
            1: 'Riesgo leve',
            2: 'Riesgo moderado',
            3: 'Riesgo alto',
            4: 'Riesgo crítico'
        }

        # Calcular probabilidad de deserción
        desertion_probability = (
            sum(risk_proba[i] for i in range(2, min(len(risk_proba), 5)))
        ) * 100

        return {
            'risk_level': int(risk_level),
            'risk_label': risk_map.get(risk_level, 'Sin riesgo'),
            'risk_probability': float(risk_proba[risk_level]) * 100,
            'desertion_probability': float(desertion_probability),
            'cluster': int(cluster),
            'cluster_name': self.get_cluster_name(cluster),
            'pca_components': X_pca.tolist()[0]
        }

    def get_cluster_name(self, cluster_id):
        """Obtener nombre descriptivo del cluster"""
        cluster_names = {
            0: 'C2 - Estrés académico',
            1: 'C1 - Compromiso alto',
            2: 'C3 - Riesgo acumulado'
        }
        return cluster_names.get(cluster_id, f'Cluster {cluster_id}')

    def save_model(self, path='models'):
        """Guardar modelo entrenado"""
        if not os.path.exists(path):
            os.makedirs(path)

        joblib.dump(self.scaler, f'{path}/scaler_v2.pkl')
        joblib.dump(self.pca, f'{path}/pca_v2.pkl')
        joblib.dump(self.kmeans, f'{path}/kmeans_v2.pkl')
        joblib.dump(self.neural_network, f'{path}/neural_network_v2.pkl')
        joblib.dump(self.label_encoders, f'{path}/label_encoders_v2.pkl')
        joblib.dump(self.feature_columns, f'{path}/feature_columns_v2.pkl')

        # Guardar metadatos
        metadata = {
            'training_timestamp': str(self.training_timestamp),
            'n_features': len(self.feature_columns),
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, f'{path}/metadata_v2.pkl')

        print(f"✅ Modelo V2 guardado en {path}/")

    def load_model(self, path='models'):
        """Cargar modelo guardado"""
        try:
            self.scaler = joblib.load(f'{path}/scaler_v2.pkl')
            self.pca = joblib.load(f'{path}/pca_v2.pkl')
            self.kmeans = joblib.load(f'{path}/kmeans_v2.pkl')
            self.neural_network = joblib.load(f'{path}/neural_network_v2.pkl')
            self.label_encoders = joblib.load(f'{path}/label_encoders_v2.pkl')
            self.feature_columns = joblib.load(f'{path}/feature_columns_v2.pkl')

            # Cargar metadatos
            metadata = joblib.load(f'{path}/metadata_v2.pkl')
            self.training_timestamp = metadata.get('training_timestamp')
            self.is_trained = True

            print(f"✅ Modelo V2 cargado desde {path}/")
            print(f"   🕐 Entrenado: {self.training_timestamp}")
            return True
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            return False


def auto_train_model(force_retrain=False):
    """
    Entrenar automáticamente el modelo desde la base de datos
    """
    print("=" * 60)
    print("🤖 SISTEMA DE REENTRENAMIENTO AUTOMÁTICO")
    print("=" * 60)

    model = CategoricalRiskPredictor()

    # Verificar si existe modelo previo
    if not force_retrain and os.path.exists('models/neural_network_v2.pkl'):
        print("📁 Modelo existente detectado, cargando...")
        if model.load_model():
            print("✅ Modelo cargado exitosamente")
            return model
        else:
            print("⚠️  Error al cargar modelo, reentrenando...")

    # Entrenar desde base de datos
    try:
        print("🔄 Iniciando entrenamiento desde PostgreSQL...")
        results = model.train_from_database()

        # Guardar modelo
        model.save_model()

        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"   Muestras: {results['n_samples']}")
        print(f"   Precisión (train): {results['train_accuracy']:.2%}")
        print(f"   Precisión (test): {results['test_accuracy']:.2%}")
        print(f"   Componentes PCA: {results['n_components']}")
        print("=" * 60)

        return model

    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
        print("   El sistema continuará sin modelo de predicción")
        return None


if __name__ == "__main__":
    # Entrenar modelo automáticamente
    model = auto_train_model(force_retrain=True)

    if model:
        # Ejemplo de predicción
        ejemplo_estudiante = {
            'sueno_horas': 'Entre_6_8h',
            'actividad_fisica': 'Moderado',
            'alimentacion': 'Moderada',
            'estilo_de_vida': 'Moderado',
            'estres_academico': 'Moderado',
            'apoyo_familiar': 'Fuerte',
            'bienestar': 'Saludable',
            'asistencia': 'Constante',
            'horas_estudio': 'De_1_3h',
            'interes_academico': 'Muy_motivado',
            'rendimiento_academico': 'Previsto',
            'historial_academico': 'Entre_11_15',
            'carga_laboral': 'No_trabaja',
            'beca': 'Parcial',
            'deudor': 'Sin_deuda'
        }

        prediccion = model.predict_risk_from_categorical(ejemplo_estudiante)
        print("\n📊 Predicción de ejemplo:")
        print(f"   Riesgo: {prediccion['risk_label']}")
        print(f"   Probabilidad deserción: {prediccion['desertion_probability']:.1f}%")
        print(f"   Cluster: {prediccion['cluster_name']}")
