# Sistema de Alerta Temprana - Instrucciones de Instalación

## 📋 Descripción

Sistema completo de alerta temprana para estudiantes que incluye:
- **Registro HTML**: Formulario para registrar nuevos estudiantes
- **Panel del Tutor**: Vista consolidada de todos los estudiantes
- **Perfil del Estudiante**: Vista detallada con predicción de riesgo
- **Base de Datos PostgreSQL**: Almacenamiento persistente de datos
- **Sistema de Entrenamientos**: Historial de entrenamientos del modelo

## 🚀 Instalación Rápida

### 1. Iniciar PostgreSQL

```bash
sudo service postgresql start
# O en sistemas con systemd:
sudo systemctl start postgresql
```

### 2. Configurar la Base de Datos

Ejecuta el script de configuración:

```bash
./setup_database.sh
```

Este script automáticamente:
- ✅ Crea el usuario `cluster_user`
- ✅ Crea la base de datos `estudiantes_db`
- ✅ Crea las tablas necesarias (estudiantes, entrenamientos_modelo)
- ✅ Inserta datos de ejemplo

### 3. Instalar Dependencias de Python

```bash
pip install fastapi uvicorn psycopg2-binary pandas python-dotenv
```

### 4. Iniciar el Servidor

```bash
python main.py
```

El servidor estará disponible en: **http://localhost:8000**

## 🎯 Flujo de Uso

### Paso 1: Registro de Estudiante
1. Abre http://localhost:8000/ (formulario de registro)
2. Completa los datos del estudiante:
   - Código, nombre, carrera, ciclo
   - Variables categóricas (15 variables del modelo)
3. Click en "Generar Predicción y Clúster"
4. El sistema automáticamente:
   - ✅ Guarda el estudiante en PostgreSQL
   - ✅ Genera predicción de riesgo
   - ✅ Asigna un clúster
   - ✅ Almacena toda la información

### Paso 2: Panel del Tutor
1. Abre http://localhost:8000/panel
2. Verás:
   - Estadísticas generales
   - Lista de estudiantes con sus predicciones
   - Filtros por riesgo, clúster y estado
3. Click en el ícono de ojo para ver el perfil completo

### Paso 3: Perfil del Estudiante
1. Desde el panel, click en ver perfil
2. Verás:
   - Predicción de riesgo con gráfico
   - Resumen académico
   - Datos básicos y contexto
   - Clúster asignado con comparación
   - Factores clave de riesgo
   - Recomendaciones del tutor

## 📊 Estructura de la Base de Datos

### Tabla: `estudiantes`
Almacena toda la información de los estudiantes:
- Datos básicos (código, nombre, carrera, ciclo, edad)
- Variables categóricas (15 variables del modelo)
- Variables numéricas adicionales
- Resultados del modelo (riesgo_predicho, cluster_asignado, probabilidad_desercion)
- Metadatos (notas_tutor, fecha_registro, ultima_actualizacion, estado_seguimiento)

### Tabla: `entrenamientos_modelo`
Almacena el historial de entrenamientos del modelo:
- fecha_entrenamiento
- num_estudiantes_entrenamiento
- precision_modelo
- metricas_json (accuracy, precision, recall, f1-score)
- modelo_version
- observaciones
- ruta_modelo

## 🔧 Configuración

El archivo `.env` contiene la configuración:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=estudiantes_db
DB_USER=cluster_user
DB_PASSWORD=cluster_pass_2024

MODEL_PATH=models/
RETRAIN_ON_STARTUP=true
MIN_SAMPLES_FOR_TRAINING=10

SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

## 🧪 Verificar Instalación

Para verificar que todo está funcionando:

```bash
# 1. Verificar PostgreSQL
pg_isready -h localhost -p 5432

# 2. Verificar conexión a la BD
python -c "from database import EstudiantesDB; db = EstudiantesDB(); stats = db.get_statistics(); print(f'Total estudiantes: {stats}'); db.close()"

# 3. Iniciar servidor
python main.py
```

## 📝 Endpoints API

- `GET /` - Formulario de registro HTML
- `GET /panel` - Panel del tutor HTML
- `GET /perfil/{codigo}` - Perfil del estudiante HTML
- `GET /api/stats` - Estadísticas del dashboard
- `GET /api/students` - Lista de estudiantes (con filtros)
- `POST /api/students/register` - Registrar nuevo estudiante
- `GET /api/students/search?q={query}` - Buscar estudiantes
- `GET /api/students/{codigo}` - Obtener perfil de estudiante
- `GET /api/export/students?formato=csv` - Exportar listado

## 🎨 Características

✅ **Registro completo**: 15 variables categóricas del modelo bayesiano
✅ **Predicción automática**: Calcula riesgo y asigna clúster al registrar
✅ **Persistencia en PostgreSQL**: Todos los datos se guardan en la BD
✅ **Panel interactivo**: Filtros, búsqueda y navegación
✅ **Perfil detallado**: Visualización completa con gráficos
✅ **Exportación**: Descarga listado en CSV
✅ **Historial de entrenamientos**: Guarda cada entrenamiento del modelo

## ⚠️ Solución de Problemas

### PostgreSQL no inicia
```bash
# Verificar logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Reiniciar servicio
sudo service postgresql restart
```

### Error de conexión a la BD
```bash
# Verificar que el usuario y BD existan
sudo -u postgres psql -c "\du"  # Lista usuarios
sudo -u postgres psql -c "\l"   # Lista bases de datos
```

### Reinstalar desde cero
```bash
# Eliminar BD y usuario
sudo -u postgres dropdb estudiantes_db
sudo -u postgres dropuser cluster_user

# Ejecutar setup nuevamente
./setup_database.sh
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que PostgreSQL esté corriendo
2. Revisa los logs del servidor (`python main.py`)
3. Verifica las credenciales en `.env`
4. Asegúrate de que todas las dependencias estén instaladas

---

**¡Listo para usar!** 🎉

El sistema ahora está completamente integrado con PostgreSQL y guarda cada registro en la base de datos.
