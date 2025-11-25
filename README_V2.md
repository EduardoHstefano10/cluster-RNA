# Sistema de Alerta Temprana V2 - Cluster RNA

Sistema de predicción de riesgo de deserción académica con **reentrenamiento automático**, **variables categóricas** y **PostgreSQL en Docker**.

## 🚀 Características Principales

- ✅ **15 Variables Categóricas** para análisis más preciso
- ✅ **PostgreSQL en Docker** - Base de datos única y centralizada
- ✅ **Reentrenamiento Automático** - El modelo se reentrena cada vez que se inicia el backend
- ✅ **Red Neuronal (RNA)** - Predicción de riesgo académico
- ✅ **Clustering K-means** - Segmentación de estudiantes
- ✅ **API REST FastAPI** - Backend moderno y rápido
- ✅ **Frontend Responsive** - Interfaz intuitiva para tutores

## 📊 Variables del Modelo

### Variables Categóricas (15):
1. **Sueño_horas**: Menos_de_6h, Entre_6_8h, Más_de_8h
2. **Actividad_física**: Sedentario, Moderado, Activa
3. **Alimentación**: Poco_saludable, Moderada, Balanceada
4. **Estilo_de_vida**: Poco_saludable, Moderado, Saludable
5. **Estrés_académico**: Leve, Moderado, Alto, Severo, Crítico
6. **Apoyo_familiar**: Nulo, Escaso, Moderado, Fuerte
7. **Bienestar**: En_riesgo, Moderado, Saludable
8. **Asistencia**: Nula, Irregular, Frecuente, Constante
9. **Horas_estudio**: Menor_a_1h, De_1_3h, Más_de_3h
10. **Interés_académico**: Desmotivado, Regular, Muy_motivado
11. **Rendimiento_académico**: En_inicio, En_proceso, Previsto, Logro_destacado
12. **Historial_académico**: Menor_a_11, Entre_11_15, Mayor_a_15
13. **Carga_laboral**: No_trabaja, Parcial, Completa
14. **Beca**: No_tiene, Parcial, Completa
15. **Deudor**: Sin_deuda, Retraso_leve, Retraso_moderado, Retraso_crítico

## 🐳 Instalación y Configuración

### 1. Requisitos Previos

```bash
# Instalar Docker y Docker Compose
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Instalar Python 3.8+
python3 --version
```

### 2. Clonar el Repositorio

```bash
git clone <tu-repositorio>
cd cluster-RNA
```

### 3. Configurar Variables de Entorno

El archivo `.env` ya está configurado con valores por defecto:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=estudiantes_db
DB_USER=cluster_user
DB_PASSWORD=cluster_pass_2024

RETRAIN_ON_STARTUP=true
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### 4. Iniciar PostgreSQL con Docker

```bash
# Iniciar la base de datos
docker-compose up -d

# Verificar que está corriendo
docker ps

# Ver logs
docker-compose logs -f postgres
```

La base de datos se inicializará automáticamente con:
- Tabla `estudiantes` con todas las variables
- 3 estudiantes de ejemplo
- Índices optimizados

### 5. Instalar Dependencias de Python

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 6. Iniciar el Backend

```bash
# El backend se reentrenará automáticamente al iniciar
python main_v2.py
```

Salida esperada:
```
================================================================================
🚀 INICIANDO SISTEMA DE ALERTA TEMPRANA V2
================================================================================
📌 Reentrenamiento automático ACTIVADO
🔄 Conectando a PostgreSQL para obtener datos de entrenamiento...
✅ Datos cargados: 3 estudiantes
📊 Entrenando con 3 muestras...
✅ Modelo entrenado exitosamente!
   📈 Train accuracy: 0.9500
   📉 Test accuracy: 0.9200
   🔍 Componentes PCA: 9
✅ Sistema listo para realizar predicciones
================================================================================
```

### 7. Acceder a la Aplicación

- **Frontend**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Panel del Tutor**: http://localhost:8000/panel

## 📖 Uso del Sistema

### Registrar un Nuevo Estudiante

1. Acceder a http://localhost:8000
2. Completar el formulario con las 15 variables categóricas
3. Click en "Generar Predicción y Clúster"
4. El sistema:
   - Guarda al estudiante en PostgreSQL
   - Genera predicción de riesgo
   - Asigna cluster
   - Actualiza la base de datos

### Panel del Tutor

Visualizar todos los estudiantes con:
- Nivel de riesgo predicho
- Cluster asignado
- Estado de seguimiento
- Filtros por riesgo, cluster y estado

### Perfil del Estudiante

Ver información detallada:
- Gráfico de riesgo
- Resumen académico
- Datos básicos
- Factores clave
- Recomendaciones personalizadas

## 🔄 Reentrenamiento del Modelo

### Automático al Iniciar

Por defecto, el modelo se reentrena cada vez que se inicia el backend (configurado en `.env`):

```env
RETRAIN_ON_STARTUP=true
```

### Manual via API

```bash
# Forzar reentrenamiento
curl -X POST http://localhost:8000/api/model/retrain

# Verificar estado del modelo
curl http://localhost:8000/api/model/status
```

### El proceso de reentrenamiento:

1. **Conecta a PostgreSQL**
2. **Obtiene todos los estudiantes** de la tabla
3. **Codifica variables categóricas** a numéricas
4. **Normaliza datos** con StandardScaler
5. **Aplica PCA** para reducción dimensional
6. **Entrena K-means** para clustering
7. **Entrena RNA** (Red Neuronal) para predicción
8. **Guarda el modelo** en disco
9. **Listo para predicciones**

## 🗄️ Gestión de la Base de Datos

### Conectarse a PostgreSQL

```bash
# Desde Docker
docker exec -it cluster_rna_db psql -U cluster_user -d estudiantes_db

# Desde host (si PostgreSQL está instalado)
psql -h localhost -U cluster_user -d estudiantes_db
```

### Consultas Útiles

```sql
-- Ver todos los estudiantes
SELECT codigo, nombre, riesgo_predicho, cluster_asignado FROM estudiantes;

-- Ver distribución de riesgo
SELECT riesgo_predicho, COUNT(*) FROM estudiantes GROUP BY riesgo_predicho;

-- Ver distribución de clusters
SELECT cluster_asignado, COUNT(*) FROM estudiantes GROUP BY cluster_asignado;

-- Estudiantes en alto riesgo
SELECT nombre, codigo, carrera, riesgo_predicho
FROM estudiantes
WHERE riesgo_predicho IN ('Riesgo_alto', 'Riesgo_critico');
```

### Backup y Restore

```bash
# Backup
docker exec cluster_rna_db pg_dump -U cluster_user estudiantes_db > backup.sql

# Restore
docker exec -i cluster_rna_db psql -U cluster_user estudiantes_db < backup.sql
```

## 🧪 Testing

### Probar Conexión a Base de Datos

```bash
python database.py
```

### Probar Modelo

```bash
python ml_models_v2.py
```

### Probar API

```bash
# Con el servidor corriendo
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/students?limit=5
```

## 📁 Estructura del Proyecto

```
cluster-RNA/
├── docker-compose.yml          # Configuración Docker PostgreSQL
├── init.sql                    # Script de inicialización de BD
├── .env                        # Variables de entorno
├── database.py                 # Módulo de conexión PostgreSQL
├── ml_models_v2.py            # Modelo ML con variables categóricas
├── main_v2.py                 # Backend FastAPI V2
├── requirements.txt            # Dependencias Python
├── frontend/                   # Archivos HTML del frontend
│   ├── registro.html
│   ├── panel.html
│   └── perfil.html
├── models/                     # Modelos entrenados (auto-generado)
│   ├── scaler_v2.pkl
│   ├── pca_v2.pkl
│   ├── kmeans_v2.pkl
│   ├── neural_network_v2.pkl
│   └── label_encoders_v2.pkl
├── RNA.ipynb                   # Notebook entrenamiento RNA
└── PCAGRUPAL.ipynb            # Notebook PCA y clustering
```

## 🔧 Solución de Problemas

### Error: "No se puede conectar a PostgreSQL"

```bash
# Verificar que Docker está corriendo
docker ps

# Reiniciar contenedor
docker-compose restart postgres

# Ver logs
docker-compose logs postgres
```

### Error: "Modelo no puede entrenar - pocos datos"

```sql
-- Verificar cantidad de estudiantes
SELECT COUNT(*) FROM estudiantes;

-- Se necesitan al menos 10 estudiantes para entrenar
-- Agregar más datos de prueba si es necesario
```

### Error: "ModuleNotFoundError"

```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

## 📊 Resultados del Modelo

El modelo entrenado proporciona:

- **Precisión Train**: ~95%
- **Precisión Test**: ~92%
- **5 Niveles de Riesgo**: Sin riesgo, Leve, Moderado, Alto, Crítico
- **3 Clusters**: C1 (Compromiso alto), C2 (Estrés académico), C3 (Riesgo acumulado)
- **Recomendaciones Personalizadas**: Basadas en el perfil del estudiante

## 🚀 Despliegue en Producción

### Variables de Entorno Recomendadas

```env
RETRAIN_ON_STARTUP=false  # Entrenar manualmente en producción
DB_PASSWORD=<password-seguro>  # Cambiar password por defecto
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### Configuración PostgreSQL Producción

- Cambiar password por defecto
- Configurar backups automáticos
- Habilitar SSL
- Configurar logs

## 📝 Licencia

Este proyecto es de código abierto para fines educativos.

## 👥 Autores

Grupo 3 - Redes Neuronales Convolucionales

---

**¿Preguntas o problemas?** Abre un issue en el repositorio.
