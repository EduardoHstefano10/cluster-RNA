"""
Modelos de Machine Learning para el Sistema de Alerta Temprana
Incluye Red Neuronal Artificial (RNA) y Clustering K-means
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


class StudentRiskPredictor:
    """
    Modelo de Red Neuronal para predicción de riesgo académico
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.85)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.is_trained = False

    def create_risk_labels(self, data):
        """
        Crear etiquetas de riesgo basadas en múltiples criterios
        Riesgo: 0=Sin riesgo, 1=Leve, 2=Moderado, 3=Alto, 4=Crítico
        """
        risk_scores = np.zeros(len(data))

        # Factor 1: Promedio ponderado
        risk_scores += np.where(data['Promedio_ponderado'] < 12, 2, 0)
        risk_scores += np.where(data['Promedio_ponderado'] < 14, 1, 0)

        # Factor 2: Asistencia
        risk_scores += np.where(data['Asistencia'] < 75, 2, 0)
        risk_scores += np.where(data['Asistencia'] < 85, 1, 0)

        # Factor 3: Cursos desaprobados
        risk_scores += np.where(data['Cursos_desaprobados'] > 2, 2, 0)
        risk_scores += np.where(data['Cursos_desaprobados'] > 1, 1, 0)

        # Factor 4: Porcentaje de créditos aprobados
        risk_scores += np.where(data['Porcentaje_creditos_aprobados'] < 60, 2, 0)
        risk_scores += np.where(data['Porcentaje_creditos_aprobados'] < 70, 1, 0)

        # Factor 5: Índice de regularidad
        risk_scores += np.where(data['indice_regularidad'] < 50, 2, 0)
        risk_scores += np.where(data['indice_regularidad'] < 60, 1, 0)

        # Clasificar riesgo basado en score acumulado
        risk_labels = np.zeros(len(data), dtype=int)
        risk_labels[risk_scores <= 1] = 0  # Sin riesgo
        risk_labels[(risk_scores > 1) & (risk_scores <= 3)] = 1  # Leve
        risk_labels[(risk_scores > 3) & (risk_scores <= 5)] = 2  # Moderado
        risk_labels[(risk_scores > 5) & (risk_scores <= 7)] = 3  # Alto
        risk_labels[risk_scores > 7] = 4  # Crítico

        return risk_labels

    def train(self, data_path='estudiantes_data.csv'):
        """
        Entrenar el modelo con los datos
        """
        # Cargar datos
        data = pd.read_csv(data_path)
        data = data.dropna()

        # Crear etiquetas de riesgo
        risk_labels = self.create_risk_labels(data)

        # Preparar features
        X = data.select_dtypes(include=[np.number]).values
        y = risk_labels

        # Normalizar
        X_scaled = self.scaler.fit_transform(X)

        # Aplicar PCA
        X_pca = self.pca.fit_transform(X_scaled)

        # Entrenar clustering
        self.kmeans.fit(X_pca)

        # Dividir datos para entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        # Entrenar red neuronal
        self.neural_network.fit(X_train, y_train)

        # Evaluar
        train_score = self.neural_network.score(X_train, y_train)
        test_score = self.neural_network.score(X_test, y_test)

        self.is_trained = True

        print(f"Modelo entrenado exitosamente!")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        print(f"Componentes PCA: {self.pca.n_components_}")

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_components': self.pca.n_components_
        }

    def predict_risk(self, student_data):
        """
        Predecir riesgo para un estudiante
        student_data: dict con las variables del estudiante
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train() primero.")

        # Convertir a DataFrame
        df = pd.DataFrame([student_data])

        # Asegurar orden de columnas
        expected_cols = [
            'Promedio_ponderado', 'Creditos_matriculados',
            'Porcentaje_creditos_aprobados', 'Cursos_desaprobados',
            'Asistencia', 'Retiros_cursos', 'Edad',
            'Horas_trabajo_semana', 'Anio_ingreso',
            'Numero_ciclos_academicos', 'Cursos_matriculados_ciclo',
            'Horas_estudio_semana', 'indice_regularidad',
            'Intentos_aprobacion_curso', 'Nota_promedio'
        ]

        # Reordenar columnas
        df = df[expected_cols]

        # Normalizar
        X_scaled = self.scaler.transform(df.values)

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
        desertion_probability = (risk_proba[2] + risk_proba[3] + risk_proba[4]) * 100 if len(risk_proba) > 2 else 0

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
        """
        Obtener nombre descriptivo del cluster
        """
        cluster_names = {
            0: 'C2 - Estrés académico',
            1: 'C1 - Compromiso alto',
            2: 'C3 - Riesgo acumulado'
        }
        return cluster_names.get(cluster_id, f'Cluster {cluster_id}')

    def get_cluster_description(self, cluster_id):
        """
        Obtener descripción del cluster
        """
        descriptions = {
            0: 'Estudiantes con buen rendimiento pero alta carga laboral que genera estrés académico',
            1: 'Estudiantes con alto compromiso, buena asistencia y rendimiento sobresaliente',
            2: 'Estudiantes con múltiples factores de riesgo acumulados que requieren intervención'
        }
        return descriptions.get(cluster_id, 'Cluster sin descripción')

    def save_model(self, path='models'):
        """
        Guardar modelo entrenado
        """
        if not os.path.exists(path):
            os.makedirs(path)

        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.pca, f'{path}/pca.pkl')
        joblib.dump(self.kmeans, f'{path}/kmeans.pkl')
        joblib.dump(self.neural_network, f'{path}/neural_network.pkl')

        print(f"Modelo guardado en {path}/")

    def load_model(self, path='models'):
        """
        Cargar modelo guardado
        """
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.pca = joblib.load(f'{path}/pca.pkl')
        self.kmeans = joblib.load(f'{path}/kmeans.pkl')
        self.neural_network = joblib.load(f'{path}/neural_network.pkl')
        self.is_trained = True

        print(f"Modelo cargado desde {path}/")


# Función auxiliar para entrenar y guardar modelo
def train_and_save_model():
    """
    Entrenar el modelo y guardarlo
    """
    print("Iniciando entrenamiento del modelo...")
    model = StudentRiskPredictor()
    results = model.train('estudiantes_data.csv')
    model.save_model()
    print("Modelo entrenado y guardado exitosamente!")
    return model, results


if __name__ == "__main__":
    # Entrenar modelo
    model, results = train_and_save_model()

    # Ejemplo de predicción
    ejemplo_estudiante = {
        'Promedio_ponderado': 15.5,
        'Creditos_matriculados': 20,
        'Porcentaje_creditos_aprobados': 75,
        'Cursos_desaprobados': 2,
        'Asistencia': 85,
        'Retiros_cursos': 1,
        'Edad': 21,
        'Horas_trabajo_semana': 15,
        'Anio_ingreso': 2015,
        'Numero_ciclos_academicos': 10,
        'Cursos_matriculados_ciclo': 6,
        'Horas_estudio_semana': 17,
        'indice_regularidad': 65,
        'Intentos_aprobacion_curso': 1,
        'Nota_promedio': 16
    }

    prediccion = model.predict_risk(ejemplo_estudiante)
    print("\nPredicción de ejemplo:")
    print(prediccion)
