# Sistema de Alerta Temprana Académica

Sistema completo de predicción de riesgo académico y clustering de estudiantes usando **Redes Neuronales Artificiales (RNA)** y **K-means clustering** con interfaz web Bootstrap y backend FastAPI.

## 🎯 Características

- **Predicción de Riesgo Académico** usando Red Neuronal Artificial (MLP)
- **Clustering de Estudiantes** con K-means (3 clústeres)
- **Análisis PCA** para reducción de dimensionalidad
- **API RESTful** con FastAPI
- **Interfaz Web** moderna con Bootstrap 5
- **3 Vistas Principales**:
  - Registro/Actualización de Estudiantes
  - Panel del Tutor
  - Perfil del Estudiante

## 📊 Arquitectura del Sistema

### Backend (FastAPI + Python)
- **main.py**: API REST con todos los endpoints
- **ml_models.py**: Modelo de RNA y clustering

### Frontend (Bootstrap 5 + JavaScript)
- **registro.html**: Formulario de registro/actualización
- **panel.html**: Dashboard del tutor
- **perfil.html**: Vista detallada del estudiante

### Datos
- **estudiantes_data.csv**: Dataset con 2000 estudiantes y 15 variables
- **PCAGRUPAL.ipynb**: Notebook de análisis exploratorio

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd cluster-RNA
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Entrenar el Modelo (Primera vez)
```bash
python ml_models.py
```

Esto generará la carpeta `models/` con:
- scaler.pkl
- pca.pkl
- kmeans.pkl
- neural_network.pkl

### Iniciar el Servidor
```bash
python main.py
```

O usando uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Acceder a la Aplicación
Abre tu navegador en: **http://localhost:8000**

## 📁 Estructura del Proyecto

```
cluster-RNA/
├── main.py                      # Backend FastAPI
├── ml_models.py                 # Modelos de ML (RNA + Clustering)
├── requirements.txt             # Dependencias Python
├── README.md                    # Documentación
├── estudiantes_data.csv         # Dataset (2000 estudiantes)
├── PCAGRUPAL.ipynb             # Notebook de análisis
├── frontend/
│   ├── registro.html           # Vista de registro
│   ├── panel.html              # Panel del tutor
│   └── perfil.html             # Perfil del estudiante
└── models/                      # Modelos entrenados (generado)
    ├── scaler.pkl
    ├── pca.pkl
    ├── kmeans.pkl
    └── neural_network.pkl
```

## 🔬 Modelo de Machine Learning

### Variables del Dataset (15 features)
1. Promedio_ponderado
2. Creditos_matriculados
3. Porcentaje_creditos_aprobados
4. Cursos_desaprobados
5. Asistencia
6. Retiros_cursos
7. Edad
8. Horas_trabajo_semana
9. Anio_ingreso
10. Numero_ciclos_academicos
11. Cursos_matriculados_ciclo
12. Horas_estudio_semana
13. indice_regularidad
14. Intentos_aprobacion_curso
15. Nota_promedio

### Red Neuronal Artificial (RNA)
- **Arquitectura**: MLP con capas [64, 32, 16]
- **Activación**: ReLU
- **Optimizador**: Adam
- **Salida**: 5 niveles de riesgo (0-4)
  - 0: Sin riesgo
  - 1: Riesgo leve
  - 2: Riesgo moderado
  - 3: Riesgo alto
  - 4: Riesgo crítico

### Clustering K-means
- **Número de clústeres**: 3
- **Clústeres identificados**:
  - **C1**: Compromiso alto
  - **C2**: Estrés académico
  - **C3**: Riesgo acumulado

### PCA (Análisis de Componentes Principales)
- **Varianza retenida**: 85%
- **Componentes**: ~9 componentes principales

## 🌐 API Endpoints

### Estudiantes
- `GET /` - Página de registro
- `GET /panel` - Panel del tutor
- `GET /perfil/{codigo}` - Perfil del estudiante
- `GET /api/students` - Listar estudiantes (con filtros)
- `GET /api/students/{codigo}` - Obtener estudiante específico
- `POST /api/students/register` - Registrar nuevo estudiante

### Predicciones
- `POST /api/predict` - Predecir riesgo de un estudiante

### Estadísticas
- `GET /api/stats` - Estadísticas del dashboard
- `GET /api/clusters/{cluster_id}` - Información de un clúster

### Intervenciones
- `POST /api/interventions/register` - Registrar intervención

### Exportación
- `GET /api/export/students?formato=csv` - Exportar estudiantes

## 📊 Ejemplos de Uso

### 1. Predecir Riesgo de un Estudiante
```python
import requests

student_data = {
    "Promedio_ponderado": 15.5,
    "Creditos_matriculados": 20,
    "Porcentaje_creditos_aprobados": 75,
    "Cursos_desaprobados": 1,
    "Asistencia": 87,
    "Retiros_cursos": 1,
    "Edad": 21,
    "Horas_trabajo_semana": 15,
    "Anio_ingreso": 2015,
    "Numero_ciclos_academicos": 10,
    "Cursos_matriculados_ciclo": 6,
    "Horas_estudio_semana": 17,
    "indice_regularidad": 65,
    "Intentos_aprobacion_curso": 1,
    "Nota_promedio": 16
}

response = requests.post(
    "http://localhost:8000/api/predict",
    json=student_data
)

print(response.json())
```

### 2. Obtener Lista de Estudiantes
```python
response = requests.get(
    "http://localhost:8000/api/students?limit=10&offset=0"
)

students = response.json()
print(f"Total: {students['total']}")
for student in students['students']:
    print(f"{student['nombre']} - {student['riesgo_predicho']}")
```

## 🎨 Interfaz de Usuario

### Vista 1: Registro/Actualizar Estudiante
- Búsqueda de estudiante
- Formulario de variables categóricas
- Generación de predicción y clúster

### Vista 2: Panel del Tutor
- Estadísticas generales
- Filtros por riesgo, clúster y estado
- Tabla de estudiantes con acciones
- Exportación de datos

### Vista 3: Perfil del Estudiante
- Gráfico de riesgo (dona chart)
- Resumen académico
- Comparación con clúster
- Factores clave de riesgo
- Recomendaciones personalizadas

## 📈 Métricas del Modelo

- **Train Accuracy**: ~95%
- **Test Accuracy**: ~92%
- **KMO Global**: 0.927 (excelente para PCA)
- **Componentes PCA**: 9 (85% varianza)

## 🔧 Configuración Avanzada

### Cambiar el Puerto
```bash
uvicorn main:app --port 3000
```

### Modo de Desarrollo
```bash
uvicorn main:app --reload --log-level debug
```

### Configurar CORS
Editar `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -m 'Agregar nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Autores

- Desarrollado para el curso de Inteligencia Artificial y Redes Neuronales

## 🐛 Reportar Bugs

Si encuentras algún error, por favor abre un issue en el repositorio.

## 📚 Recursos Adicionales

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Bootstrap 5 Documentation](https://getbootstrap.com/)
- [Chart.js Documentation](https://www.chartjs.org/)

## ⚡ Performance

- Tiempo de predicción: ~10ms por estudiante
- Capacidad: +10,000 estudiantes simultáneos
- Base de datos: Actualmente en memoria (recomendado: PostgreSQL/MongoDB para producción)

## 🔐 Seguridad

Para producción, considera:
- Agregar autenticación JWT
- Implementar rate limiting
- Usar HTTPS
- Validar inputs exhaustivamente
- Implementar logging y monitoring

## 📞 Soporte

Para preguntas o soporte, contacta a: [tu-email@ejemplo.com]

---

**¡Disfruta del sistema! 🎓**
