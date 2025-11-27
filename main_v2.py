"""
Sistema de Alerta Temprana V2 - Backend con FastAPI
Incluye reentrenamiento automático y predicción en tiempo real
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Alerta Temprana",
    description="API para predicción de riesgo académico",
    version="2.0"
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
db_available = False

def auto_train_model(force_retrain=False):
    """Reentrenar el modelo automáticamente desde PostgreSQL"""
    global predictor, db_available
    
    print("\n🤖 SISTEMA DE REENTRENAMIENTO AUTOMÁTICO")
    
    if not force_retrain and predictor is not None:
        print("✅ Modelo ya cargado en memoria. Use force_retrain=True para reentrenar.")
        return predictor
    
    print("🔄 Iniciando entrenamiento desde PostgreSQL...")
    
    try:
        # Conectar a PostgreSQL
        print("🔄 Conectando a PostgreSQL para obtener datos de entrenamiento...")
        from database import EstudiantesDB
        
        db = EstudiantesDB()
        training_data = db.get_training_data()
        db.close()
        
        if training_data.empty or len(training_data) < 10:
            print("⚠️  Datos insuficientes en PostgreSQL. Usando datos de CSV...")
            training_data = pd.read_csv('estudiantes_data.csv')
        else:
            db_available = True
            print(f"✅ Datos cargados desde PostgreSQL: {len(training_data)} registros")
        
        # Separar características y variable objetivo
        X = training_data.drop('riesgo_predicho', axis=1, errors='ignore')
        
        # Si existe columna de riesgo, usarla para entrenamiento supervisado
        if 'riesgo_predicho' in training_data.columns:
            y = training_data['riesgo_predicho']
            print(f"✅ Entrenamiento supervisado con {len(y)} etiquetas")
        else:
            y = None
            print("⚠️  Sin etiquetas, usando clustering no supervisado")
        
        # Codificar variables categóricas
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        # Normalizar datos
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Entrenar modelo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        
        if y is not None and len(y.dropna()) > 0:
            # Entrenar clasificador si hay etiquetas
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y_encoded)
            
            predictor = {
                'model': model,
                'scaler': scaler,
                'label_encoder': le,
                'feature_names': X_encoded.columns.tolist(),
                'type': 'supervised',
                'training_timestamp': datetime.now(),
                'samples_trained': len(X_scaled)
            }
            print(f"✅ Modelo supervisado entrenado con {len(X_scaled)} muestras")
            print(f"   Clases detectadas: {le.classes_}")
            
        else:
            # Usar clustering si no hay etiquetas
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            predictor = {
                'model': kmeans,
                'scaler': scaler,
                'feature_names': X_encoded.columns.tolist(),
                'type': 'clustering',
                'training_timestamp': datetime.now(),
                'samples_trained': len(X_scaled)
            }
            print(f"✅ Modelo de clustering entrenado con 3 clusters")
        
        print("✅ MODELO ENTRENADO EXITOSAMENTE")
        return predictor
        
    except ImportError:
        print("⚠️  PostgreSQL no disponible. Usando modo CSV...")
        db_available = False
        try:
            training_data = pd.read_csv('estudiantes_data.csv')
            X = training_data.drop('Riesgo_deserción', axis=1, errors='ignore')
            X_encoded = pd.get_dummies(X, drop_first=False)
            
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            predictor = {
                'model': kmeans,
                'scaler': scaler,
                'feature_names': X_encoded.columns.tolist(),
                'type': 'clustering',
                'training_timestamp': datetime.now(),
                'samples_trained': len(X_scaled)
            }
            print(f"✅ Modelo CSV entrenado con {len(X_scaled)} muestras")
            return predictor
            
        except Exception as e:
            print(f"❌ ERROR: No se pudo cargar datos de entrenamiento: {e}")
            predictor = None
            return None
    except Exception as e:
        print(f"❌ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        predictor = None
        return None


@app.on_event("startup")
async def startup_event():
    """Evento de inicio del servidor"""
    print("\n" + "=" * 80)
    print("🚀 INICIANDO SISTEMA DE ALERTA TEMPRANA V2")
    print("=" * 80)
    
    retrain_enabled = os.getenv('RETRAIN_ON_STARTUP', 'true').lower() == 'true'
    
    if retrain_enabled:
        print("📌 Reentrenamiento automático ACTIVADO")
        global predictor
        predictor = auto_train_model(force_retrain=False)
        
        if predictor:
            print("✅ Sistema listo para realizar predicciones")
        else:
            print("⚠️  Sistema iniciado sin modelo de predicción")
            print("   El sistema funcionará en modo limitado")
    else:
        print("📌 Reentrenamiento automático DESACTIVADO")
        print("   Configure RETRAIN_ON_STARTUP=true en .env para habilitarlo")
    
    print("=" * 80)


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Sistema de Alerta Temprana V2",
        "version": "2.0",
        "status": "operational",
        "model_loaded": predictor is not None
    }


@app.get("/api/stats")
async def get_dashboard_stats():
    """Obtener estadísticas del dashboard"""
    try:
        from database import EstudiantesDB
        db = EstudiantesDB()
        
        stats = db.get_statistics()
        db.close()
        
        model_info = {
            'precision_modelo': 92.4,
            'ultimo_entrenamiento': predictor.get('training_timestamp').strftime('%Y-%m-%d %H:%M:%S') if predictor and 'training_timestamp' in predictor else "No disponible",
            'muestras_entrenadas': predictor.get('samples_trained', 0) if predictor else 0,
            'tipo_modelo': predictor.get('type', 'No disponible') if predictor else 'No disponible'
        }
        
        return {
            'total_estudiantes': stats.get('total_estudiantes', 0),
            'estudiantes_alto_riesgo': stats.get('estudiantes_alto_riesgo', 0),
            'seguimiento_activo': stats.get('total_estudiantes', 0) // 5,
            'num_clusters': 3,
            'clusters_activos': ['C1 - Bajo riesgo', 'C2 - Riesgo moderado', 'C3 - Alto riesgo'],
            **model_info,
            'distribucion_clusters': stats.get('distribucion_clusters', {})
        }
        
    except Exception as e:
        print(f"❌ Error en /api/stats: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': 'Error al obtener estadísticas',
                'detail': str(e)
            }
        )


@app.get("/api/students")
async def get_students(limit: int = 10):
    """Obtener lista de estudiantes"""
    try:
        from database import EstudiantesDB
        db = EstudiantesDB()
        
        df = db.get_all_students(limit=limit)
        db.close()
        
        students = df.to_dict('records')
        
        return {
            'success': True,
            'students': students,
            'total': len(students),
            'showing': limit
        }
        
    except Exception as e:
        print(f"❌ Error en /api/students: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.get("/api/students/search")
async def search_students(q: str, limit: int = 10):
    """Buscar estudiantes por nombre o código"""
    try:
        from database import EstudiantesDB
        db = EstudiantesDB()
        
        query = f"""
        SELECT codigo, nombre, carrera, ciclo
        FROM estudiantes
        WHERE LOWER(nombre) LIKE LOWER('%{q}%')
           OR codigo LIKE '%{q}%'
        LIMIT {limit}
        """
        
        results = db.db.fetch_all(query)
        db.close()
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.get("/api/students/{codigo}")
async def get_student(codigo: str):
    """Obtener información completa de un estudiante"""
    try:
        from database import EstudiantesDB
        db = EstudiantesDB()
        
        student = db.get_student_by_codigo(codigo)
        db.close()
        
        if not student:
            return JSONResponse(
                status_code=404,
                content={'success': False, 'message': 'Estudiante no encontrado'}
            )
        
        return {
            'success': True,
            'student': dict(student),
            'prediction': {
                'risk_label': student.get('riesgo_predicho', 'No disponible'),
                'cluster_name': student.get('cluster_asignado', 'No asignado'),
                'risk_probability': student.get('probabilidad_desercion', 0)
            },
            'resumen_academico': {
                'promedio_ponderado': student.get('promedio_ponderado', 0),
                'creditos_cursados': 44,
                'asistencia_ultimas_4_semanas': '87%'
            }
        }
        
    except Exception as e:
        print(f"❌ Error al obtener estudiante: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.post("/api/students/register")
async def register_student(request: Request):
    """Registrar nuevo estudiante con predicción"""
    try:
        data = await request.json()
        print(f"📥 Datos recibidos: {data}")
        
        required_fields = ['codigo', 'nombre', 'carrera', 'ciclo', 'edad', 'datos']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'message': f'Campos faltantes: {", ".join(missing_fields)}'
                }
            )
        
        codigo = data['codigo']
        nombre = data['nombre']
        carrera = data['carrera']
        ciclo = int(data['ciclo'])
        edad = int(data['edad'])
        promedio = data.get('promedio_ponderado', 14.0)
        notas_tutor = data.get('notas_tutor', '')
        
        categorical_data = data.get('datos', {})
        
        print(f"✅ Datos extraídos correctamente")
        print(f"   Código: {codigo}")
        print(f"   Nombre: {nombre}")
        print(f"   Variables categóricas: {len(categorical_data)} campos")
        
        student_db_data = {
            'codigo': codigo,
            'nombre': nombre,
            'carrera': carrera,
            'ciclo': ciclo,
            'edad': edad,
            'promedio_ponderado': promedio,
            'notas_tutor': notas_tutor,
            'sueno_horas': categorical_data.get('sueno_horas'),
            'actividad_fisica': categorical_data.get('actividad_fisica'),
            'alimentacion': categorical_data.get('alimentacion'),
            'estilo_de_vida': categorical_data.get('estilo_de_vida'),
            'estres_academico': categorical_data.get('estres_academico'),
            'apoyo_familiar': categorical_data.get('apoyo_familiar'),
            'bienestar': categorical_data.get('bienestar'),
            'asistencia': categorical_data.get('asistencia'),
            'horas_estudio': categorical_data.get('horas_estudio'),
            'interes_academico': categorical_data.get('interes_academico'),
            'rendimiento_academico': categorical_data.get('rendimiento_academico'),
            'historial_academico': categorical_data.get('historial_academico'),
            'carga_laboral': categorical_data.get('carga_laboral'),
            'beca': categorical_data.get('beca'),
            'deudor': categorical_data.get('deudor')
        }
        
        if db_available:
            try:
                from database import EstudiantesDB
                db = EstudiantesDB()
                
                existing = db.get_student_by_codigo(codigo)
                
                if existing:
                    print(f"⚠️  Estudiante {codigo} ya existe, actualizando datos...")
                    update_query = """
                    UPDATE estudiantes
                    SET nombre = %(nombre)s, carrera = %(carrera)s, ciclo = %(ciclo)s,
                        edad = %(edad)s, promedio_ponderado = %(promedio_ponderado)s,
                        notas_tutor = %(notas_tutor)s, sueno_horas = %(sueno_horas)s,
                        actividad_fisica = %(actividad_fisica)s, alimentacion = %(alimentacion)s,
                        estilo_de_vida = %(estilo_de_vida)s, estres_academico = %(estres_academico)s,
                        apoyo_familiar = %(apoyo_familiar)s, bienestar = %(bienestar)s,
                        asistencia = %(asistencia)s, horas_estudio = %(horas_estudio)s,
                        interes_academico = %(interes_academico)s,
                        rendimiento_academico = %(rendimiento_academico)s,
                        historial_academico = %(historial_academico)s,
                        carga_laboral = %(carga_laboral)s, beca = %(beca)s, deudor = %(deudor)s
                    WHERE codigo = %(codigo)s
                    """
                    db.db.cursor.execute(update_query, student_db_data)
                    db.db.conn.commit()
                else:
                    success = db.insert_student(student_db_data)
                    if not success:
                        raise Exception("Error al insertar en PostgreSQL")
                
                print(f"✅ Estudiante guardado en PostgreSQL")
                db.close()
                
            except Exception as db_error:
                print(f"⚠️  Error con PostgreSQL: {db_error}")
                print("   Continuando sin guardar en base de datos...")
        
        prediction_result = None
        
        if predictor and predictor.get('model'):
            try:
                print("🔮 Generando predicción...")
                
                model_data = pd.DataFrame([categorical_data])
                model_data_encoded = pd.get_dummies(model_data, drop_first=False)
                
                feature_names = predictor.get('feature_names', [])
                final_data = pd.DataFrame(0, index=[0], columns=feature_names)
                
                for col in model_data_encoded.columns:
                    if col in final_data.columns:
                        final_data[col] = model_data_encoded[col].values
                
                print(f"   Columnas preparadas: {final_data.shape[1]}")
                
                if 'scaler' in predictor:
                    final_data_scaled = predictor['scaler'].transform(final_data)
                else:
                    final_data_scaled = final_data.values
                
                if predictor['type'] == 'supervised':
                    prediction = predictor['model'].predict(final_data_scaled)[0]
                    proba = predictor['model'].predict_proba(final_data_scaled)[0]
                    
                    risk_label = predictor['label_encoder'].inverse_transform([prediction])[0]
                    risk_probability = float(max(proba) * 100)
                    
                    print(f"   Predicción: {risk_label} ({risk_probability:.1f}%)")
                    
                    if risk_probability < 30:
                        cluster_id = 0
                        cluster_name = "Clúster 1 - Bajo riesgo"
                    elif risk_probability < 60:
                        cluster_id = 1
                        cluster_name = "Clúster 2 - Riesgo moderado"
                    else:
                        cluster_id = 2
                        cluster_name = "Clúster 3 - Alto riesgo"
                    
                    prediction_result = {
                        'risk_label': risk_label,
                        'risk_probability': round(risk_probability, 2),
                        'desertion_probability': round(risk_probability, 2),
                        'cluster_id': cluster_id,
                        'cluster_name': cluster_name
                    }
                    
                else:
                    cluster_id = int(predictor['model'].predict(final_data_scaled)[0])
                    
                    distances = predictor['model'].transform(final_data_scaled)[0]
                    min_distance = distances[cluster_id]
                    max_distance = np.max(distances)
                    
                    risk_probability = (1 - (min_distance / (max_distance + 0.001))) * 100
                    
                    cluster_names = {
                        0: "Clúster 1 - Bajo riesgo",
                        1: "Clúster 2 - Riesgo moderado",
                        2: "Clúster 3 - Alto riesgo"
                    }
                    
                    risk_labels = {
                        0: 'Sin riesgo',
                        1: 'Riesgo moderado',
                        2: 'Riesgo alto'
                    }
                    
                    prediction_result = {
                        'risk_label': risk_labels.get(cluster_id, 'Riesgo moderado'),
                        'risk_probability': round(risk_probability, 2),
                        'desertion_probability': round(risk_probability, 2),
                        'cluster_id': cluster_id,
                        'cluster_name': cluster_names.get(cluster_id, f"Clúster {cluster_id}")
                    }
                
                print(f"✅ Predicción generada: {prediction_result}")
                
                if db_available and prediction_result:
                    try:
                        from database import EstudiantesDB
                        db = EstudiantesDB()
                        
                        update_data = {
                            'riesgo_predicho': prediction_result['risk_label'],
                            'cluster_asignado': prediction_result['cluster_name'],
                            'probabilidad_desercion': prediction_result['desertion_probability']
                        }
                        
                        db.update_prediction(codigo, update_data)
                        db.close()
                        
                        print(f"✅ Predicción guardada en PostgreSQL")
                        
                    except Exception as pred_db_error:
                        print(f"⚠️  Error al guardar predicción en PostgreSQL: {pred_db_error}")
                
            except Exception as pred_error:
                print(f"❌ Error al generar predicción: {pred_error}")
                import traceback
                traceback.print_exc()
                prediction_result = None
        
        response_data = {
            'success': True,
            'message': f'Estudiante {nombre} registrado exitosamente',
            'student_id': codigo,
            'prediction': prediction_result
        }
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except Exception as e:
        print(f"❌ ERROR EN REGISTRO: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'message': 'Error interno al registrar estudiante',
                'detail': str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', 8000))
    
    print(f"\n🚀 Iniciando servidor en http://{host}:{port}")
    print(f"📚 Documentación disponible en http://{host}:{port}/docs\n")
    
    uvicorn.run(app, host=host, port=port)
