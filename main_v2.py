"""
Sistema de Alerta Temprana - Backend FastAPI V2
Versión actualizada con variables categóricas y PostgreSQL
Incluye reentrenamiento automático al iniciar
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from ml_models_v2 import CategoricalRiskPredictor, auto_train_model
from database import EstudiantesDB

# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Alerta Temprana V2",
    description="API para predicción de riesgo académico con variables categóricas y PostgreSQL",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el modelo
predictor = None


# ==================== MODELOS PYDANTIC ====================

class StudentCategoricalData(BaseModel):
    """Modelo de datos del estudiante con variables categóricas"""
    sueno_horas: str = Field(..., description="Menos_de_6h, Entre_6_8h, Más_de_8h")
    actividad_fisica: str = Field(..., description="Sedentario, Moderado, Activa")
    alimentacion: str = Field(..., description="Poco_saludable, Moderada, Balanceada")
    estilo_de_vida: str = Field(..., description="Poco_saludable, Moderado, Saludable")
    estres_academico: str = Field(..., description="Leve, Moderado, Alto, Severo, Crítico")
    apoyo_familiar: str = Field(..., description="Nulo, Escaso, Moderado, Fuerte")
    bienestar: str = Field(..., description="En_riesgo, Moderado, Saludable")
    asistencia: str = Field(..., description="Nula, Irregular, Frecuente, Constante")
    horas_estudio: str = Field(..., description="Menor_a_1h, De_1_3h, Más_de_3h")
    interes_academico: str = Field(..., description="Desmotivado, Regular, Muy_motivado")
    rendimiento_academico: str = Field(..., description="En_inicio, En_proceso, Previsto, Logro_destacado")
    historial_academico: str = Field(..., description="Menor_a_11, Entre_11_15, Mayor_a_15")
    carga_laboral: str = Field(..., description="No_trabaja, Parcial, Completa")
    beca: str = Field(..., description="No_tiene, Parcial, Completa")
    deudor: str = Field(..., description="Sin_deuda, Retraso_leve, Retraso_moderado, Retraso_crítico")


class StudentProfile(BaseModel):
    """Perfil completo del estudiante"""
    codigo: str = Field(..., description="Código del estudiante")
    nombre: str = Field(..., description="Nombre completo")
    carrera: str = Field(..., description="Carrera")
    ciclo: int = Field(..., ge=1, le=12, description="Ciclo actual")
    edad: int = Field(..., ge=16, le=60, description="Edad")
    promedio_ponderado: Optional[float] = Field(None, ge=0, le=20)
    datos: StudentCategoricalData
    notas_tutor: Optional[str] = None


class PredictionResponse(BaseModel):
    """Respuesta de predicción de riesgo"""
    risk_level: int
    risk_label: str
    risk_probability: float
    desertion_probability: float
    cluster: int
    cluster_name: str
    recommendations: List[str]
    key_factors: List[Dict[str, str]]


class DashboardStats(BaseModel):
    """Estadísticas del dashboard"""
    total_estudiantes: int
    precision_modelo: float
    estudiantes_alto_riesgo: int
    seguimiento_activo: int
    num_clusters: int
    clusters_activos: List[str]
    ultimo_entrenamiento: str


# ==================== INICIALIZACIÓN ====================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio: entrenar/cargar modelo automáticamente"""
    global predictor

    print("\n" + "=" * 80)
    print("🚀 INICIANDO SISTEMA DE ALERTA TEMPRANA V2")
    print("=" * 80)

    # Verificar configuración de reentrenamiento
    retrain_on_startup = os.getenv('RETRAIN_ON_STARTUP', 'true').lower() == 'true'

    if retrain_on_startup:
        print("📌 Reentrenamiento automático ACTIVADO")
        predictor = auto_train_model(force_retrain=False)
    else:
        print("📌 Reentrenamiento automático DESACTIVADO")
        predictor = CategoricalRiskPredictor()
        predictor.load_model()

    if predictor and predictor.is_trained:
        print("✅ Sistema listo para realizar predicciones")
    else:
        print("⚠️  Sistema iniciado sin modelo de predicción")

    print("=" * 80 + "\n")


# ==================== ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal - Formulario de registro"""
    with open('frontend/registro.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/panel", response_class=HTMLResponse)
async def panel_tutor():
    """Panel del tutor"""
    with open('frontend/panel.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/perfil/{codigo}", response_class=HTMLResponse)
async def perfil_estudiante(codigo: str):
    """Perfil del estudiante"""
    with open('frontend/perfil.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/api/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Obtener estadísticas del dashboard"""
    db = EstudiantesDB()

    try:
        stats = db.get_statistics()

        return DashboardStats(
            total_estudiantes=stats.get('total_estudiantes', 0),
            precision_modelo=92.4,  # Actualizar con precisión real del modelo
            estudiantes_alto_riesgo=stats.get('estudiantes_alto_riesgo', 0),
            seguimiento_activo=stats.get('total_estudiantes', 0) // 5,  # Estimación
            num_clusters=3,
            clusters_activos=[
                "C1 - Compromiso alto",
                "C2 - Estrés académico",
                "C3 - Riesgo acumulado"
            ],
            ultimo_entrenamiento=str(predictor.training_timestamp) if predictor else "No disponible"
        )
    finally:
        db.close()


@app.get("/api/students")
async def get_all_students(
    riesgo: Optional[str] = None,
    cluster: Optional[int] = None,
    estado: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Obtener lista de estudiantes con filtros"""
    db = EstudiantesDB()

    try:
        df = db.get_all_students()

        # Aplicar filtros
        if riesgo:
            df = df[df['riesgo_predicho'].str.contains(riesgo, case=False, na=False)]
        if cluster is not None:
            df = df[df['cluster_asignado'] == cluster]
        if estado:
            df = df[df['estado_seguimiento'].str.contains(estado, case=False, na=False)]

        # Paginación
        df_paginated = df.iloc[offset:offset + limit]

        # Convertir a formato de respuesta
        students_list = []
        for _, row in df_paginated.iterrows():
            students_list.append({
                'nombre': row['nombre'],
                'codigo': row['codigo'],
                'carrera': row.get('carrera', 'No especificada'),
                'promedio': float(row.get('promedio_ponderado', 0)) if row.get('promedio_ponderado') else 0,
                'asistencia': row.get('asistencia', 'No disponible'),
                'riesgo_predicho': row.get('riesgo_predicho', 'Sin evaluar'),
                'riesgo_nivel': _get_risk_level(row.get('riesgo_predicho')),
                'cluster_asignado': f"C{row.get('cluster_asignado', 0)} - Cluster" if row.get('cluster_asignado') is not None else 'Sin asignar',
                'cluster_id': int(row.get('cluster_asignado', 0)) if row.get('cluster_asignado') is not None else None,
                'estado_seguimiento': row.get('estado_seguimiento', 'Pendiente'),
                'desertion_prob': float(row.get('probabilidad_desercion', 0)) if row.get('probabilidad_desercion') else 0
            })

        return {
            'total': len(df),
            'showing': f"{offset} a {min(offset + limit, len(df))}",
            'students': students_list
        }
    finally:
        db.close()


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_student_risk(student: StudentCategoricalData):
    """Predecir riesgo académico de un estudiante"""
    if not predictor or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Modelo no disponible. El sistema está en modo de entrenamiento.")

    try:
        # Convertir a diccionario
        student_dict = student.model_dump()

        # Realizar predicción
        prediction = predictor.predict_risk_from_categorical(student_dict)

        # Generar recomendaciones
        recommendations = generate_recommendations_categorical(
            prediction['risk_level'],
            student_dict,
            prediction['cluster']
        )

        # Identificar factores clave
        key_factors = identify_key_factors_categorical(student_dict, prediction['risk_level'])

        return PredictionResponse(
            risk_level=prediction['risk_level'],
            risk_label=prediction['risk_label'],
            risk_probability=prediction['risk_probability'],
            desertion_probability=prediction['desertion_probability'],
            cluster=prediction['cluster'],
            cluster_name=prediction['cluster_name'],
            recommendations=recommendations,
            key_factors=key_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/api/students/register")
async def register_student(profile: StudentProfile):
    """Registrar nuevo estudiante y generar predicción"""
    db = EstudiantesDB()

    try:
        # Preparar datos para inserción
        student_data = {
            'codigo': profile.codigo,
            'nombre': profile.nombre,
            'carrera': profile.carrera,
            'ciclo': profile.ciclo,
            'edad': profile.edad,
            'promedio_ponderado': profile.promedio_ponderado,
            'sueno_horas': profile.datos.sueno_horas,
            'actividad_fisica': profile.datos.actividad_fisica,
            'alimentacion': profile.datos.alimentacion,
            'estilo_de_vida': profile.datos.estilo_de_vida,
            'estres_academico': profile.datos.estres_academico,
            'apoyo_familiar': profile.datos.apoyo_familiar,
            'bienestar': profile.datos.bienestar,
            'asistencia': profile.datos.asistencia,
            'horas_estudio': profile.datos.horas_estudio,
            'interes_academico': profile.datos.interes_academico,
            'rendimiento_academico': profile.datos.rendimiento_academico,
            'historial_academico': profile.datos.historial_academico,
            'carga_laboral': profile.datos.carga_laboral,
            'beca': profile.datos.beca,
            'deudor': profile.datos.deudor,
            'notas_tutor': profile.notas_tutor
        }

        # Insertar estudiante
        if not db.insert_student(student_data):
            raise HTTPException(status_code=500, detail="Error al registrar estudiante")

        # Generar predicción
        if predictor and predictor.is_trained:
            prediction = await predict_student_risk(profile.datos)

            # Actualizar predicción en BD
            db.update_prediction(profile.codigo, {
                'riesgo_predicho': prediction.risk_label.replace(' ', '_'),
                'cluster_asignado': prediction.cluster,
                'probabilidad_desercion': prediction.desertion_probability
            })

            return {
                'status': 'success',
                'message': f'Estudiante {profile.nombre} registrado exitosamente',
                'student_id': profile.codigo,
                'prediction': prediction
            }
        else:
            return {
                'status': 'success',
                'message': f'Estudiante {profile.nombre} registrado sin predicción (modelo no disponible)',
                'student_id': profile.codigo
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al registrar: {str(e)}")
    finally:
        db.close()


@app.get("/api/students/search")
async def search_students(q: str, limit: int = 10):
    """Buscar estudiantes por nombre o código"""
    db = EstudiantesDB()

    try:
        query = """
        SELECT codigo, nombre, carrera, ciclo, edad
        FROM estudiantes
        WHERE LOWER(nombre) LIKE LOWER(%s) OR codigo LIKE %s
        LIMIT %s
        """
        search_term = f"%{q}%"
        results = db.db.fetch_all(query, (search_term, search_term, limit))

        return {
            'results': results,
            'total': len(results)
        }
    finally:
        db.close()


@app.get("/api/students/{codigo}")
async def get_student_profile(codigo: str):
    """Obtener perfil completo de un estudiante"""
    db = EstudiantesDB()

    try:
        student = db.get_student_by_codigo(codigo)

        if not student:
            raise HTTPException(status_code=404, detail="Estudiante no encontrado")

        return {
            'student': dict(student),
            'prediction': {
                'risk_label': student.get('riesgo_predicho', 'Sin evaluar'),
                'cluster': student.get('cluster_asignado'),
                'desertion_probability': student.get('probabilidad_desercion', 0)
            },
            'resumen_academico': {
                'promedio_ponderado': student.get('promedio_ponderado', 0),
                'asistencia_ultimas_4_semanas': f"{student.get('asistencia', 'N/A')}",
            },
            'datos_basicos': {
                'edad': f"{student.get('edad', 0)} años",
                'carga_laboral': student.get('carga_laboral', 'No especificada'),
                'beca': student.get('beca', 'No especificada'),
                'deudor': student.get('deudor', 'No especificado'),
                'apoyo_familiar': student.get('apoyo_familiar', 'No especificado'),
            }
        }
    finally:
        db.close()


@app.post("/api/model/retrain")
async def retrain_model():
    """Forzar reentrenamiento del modelo"""
    global predictor

    try:
        print("🔄 Reentrenamiento manual iniciado...")
        predictor = auto_train_model(force_retrain=True)

        if predictor and predictor.is_trained:
            return {
                'status': 'success',
                'message': 'Modelo reentrenado exitosamente',
                'timestamp': str(predictor.training_timestamp)
            }
        else:
            raise HTTPException(status_code=500, detail="Error en el reentrenamiento")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar: {str(e)}")


@app.get("/api/model/status")
async def get_model_status():
    """Obtener estado del modelo"""
    if not predictor:
        return {
            'status': 'not_loaded',
            'message': 'Modelo no cargado',
            'is_trained': False
        }

    return {
        'status': 'active' if predictor.is_trained else 'not_trained',
        'is_trained': predictor.is_trained,
        'training_timestamp': str(predictor.training_timestamp) if predictor.training_timestamp else None,
        'n_features': len(predictor.feature_columns) if predictor.feature_columns else 0
    }


# ==================== FUNCIONES AUXILIARES ====================

def _get_risk_level(riesgo_text):
    """Convertir texto de riesgo a nivel numérico"""
    risk_map = {
        'Sin_riesgo': 0,
        'Riesgo_leve': 1,
        'Riesgo_moderado': 2,
        'Riesgo_alto': 3,
        'Riesgo_critico': 4
    }
    return risk_map.get(riesgo_text, 0)


def generate_recommendations_categorical(risk_level: int, student_data: dict, cluster: int) -> List[str]:
    """Generar recomendaciones basadas en variables categóricas"""
    recommendations = []

    if risk_level >= 3:
        recommendations.append("Agendar una sesión de orientación académica prioritaria")
        recommendations.append("Coordinar derivación opcional a bienestar psicológico para manejo de estrés")

    if student_data.get('asistencia') in ['Irregular', 'Nula']:
        recommendations.append("Monitorear asistencia semanalmente")

    if student_data.get('estres_academico') in ['Alto', 'Severo', 'Crítico']:
        recommendations.append("Implementar estrategias de manejo de estrés académico")

    if student_data.get('horas_estudio') == 'Menor_a_1h':
        recommendations.append("Desarrollar plan de estudio estructurado")

    if student_data.get('carga_laboral') == 'Completa':
        recommendations.append("Explorar ajustes de horario laboral o negociación de turnos")

    if student_data.get('apoyo_familiar') in ['Nulo', 'Escaso']:
        recommendations.append("Involucrar a la familia en espacios de información y apoyo")

    if not recommendations:
        recommendations.append("Mantener el buen desempeño actual")
        recommendations.append("Continuar con seguimiento regular")

    return recommendations


def identify_key_factors_categorical(student_data: dict, risk_level: int) -> List[Dict[str, str]]:
    """Identificar factores clave basados en variables categóricas"""
    factors = []

    # Estrés académico
    if student_data.get('estres_academico') in ['Alto', 'Severo', 'Crítico']:
        factors.append({
            'factor': 'Estrés académico',
            'nivel': 'Alto impacto',
            'descripcion': f"Nivel de estrés: {student_data['estres_academico']}"
        })

    # Asistencia
    if student_data.get('asistencia') in ['Irregular', 'Frecuente']:
        factors.append({
            'factor': 'Asistencia',
            'nivel': 'Moderado impacto',
            'descripcion': f"Asistencia {student_data['asistencia'].lower()}"
        })
    elif student_data.get('asistencia') == 'Constante':
        factors.append({
            'factor': 'Asistencia',
            'nivel': 'Factor protector',
            'descripcion': 'Buena regularidad en asistencia'
        })

    # Carga laboral
    if student_data.get('carga_laboral') in ['Parcial', 'Completa']:
        factors.append({
            'factor': 'Carga laboral',
            'nivel': 'Incrementa el riesgo',
            'descripcion': f"Trabajo {student_data['carga_laboral'].lower()}"
        })

    # Apoyo familiar
    if student_data.get('apoyo_familiar') in ['Fuerte', 'Moderado']:
        factors.append({
            'factor': 'Apoyo familiar',
            'nivel': 'Compensa parte del riesgo',
            'descripcion': f"Apoyo familiar {student_data['apoyo_familiar'].lower()}"
        })

    return factors


# Montar archivos estáticos
if os.path.exists('frontend'):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', 8000))
    uvicorn.run(app, host=host, port=port)
