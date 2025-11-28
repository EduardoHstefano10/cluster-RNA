# 📊 Instrucciones para Cargar Datos de Entrenamiento

Este documento explica cómo poblar la base de datos con datos de entrenamiento de referencia para que el sistema de comparación de modelos funcione correctamente.

## 🎯 Objetivo

Llenar la tabla `entrenamientos_modelo` con datos históricos de entrenamientos del modelo ML para:
- Mostrar el progreso del modelo a lo largo del tiempo
- Permitir comparaciones entre diferentes versiones
- Proporcionar métricas de referencia para futuros entrenamientos
- Retroalimentar y mejorar los nodos del modelo

## 📁 Archivos Disponibles

### 1. `insert_training_data.sql` (RECOMENDADO)
Script SQL que puede ejecutarse directamente en cualquier cliente PostgreSQL.

**Datos que inserta:**
- 10 registros de entrenamiento históricos
- Desde hace 6 meses hasta hace 3 días
- Precisión progresiva de 68% a 89%
- Versiones del modelo desde v1.0 hasta v2.5
- Métricas completas (accuracy, precision, recall, f1-score)

### 2. `insert_training_data.py` (ALTERNATIVA)
Script Python para ejecutar desde el código backend.

## 🚀 Métodos de Ejecución

### Opción 1: Ejecutar SQL directamente (MÁS FÁCIL)

#### A. Desde Railway Dashboard
1. Ve a tu proyecto en Railway
2. Abre el servicio de PostgreSQL
3. Haz clic en "Query"
4. Copia y pega el contenido de `insert_training_data.sql`
5. Ejecuta el script
6. Verifica los resultados

#### B. Desde psql (línea de comandos)
```bash
# Conectar a Railway PostgreSQL
psql postgresql://postgres:PASSWORD@HOST:PORT/railway

# Ejecutar el script
\i insert_training_data.sql

# O en una sola línea
psql postgresql://postgres:PASSWORD@HOST:PORT/railway -f insert_training_data.sql
```

#### C. Desde DBeaver, pgAdmin, o DataGrip
1. Conectar a la base de datos de Railway
2. Abrir el archivo `insert_training_data.sql`
3. Ejecutar el script completo
4. Verificar los resultados

### Opción 2: Ejecutar script Python

```bash
# Asegurarse de tener las dependencias instaladas
pip install psycopg2-binary python-dotenv

# Configurar archivo .env con credenciales de Railway
cp .env.example .env
# Editar .env con tus credenciales reales

# Ejecutar el script
python insert_training_data.py
```

## 📊 Datos que se Insertarán

### Resumen de Entrenamientos

| Versión | Estudiantes | Precisión | Fecha Relativa | Estado |
|---------|------------|-----------|----------------|---------|
| v1.0-baseline | 150 | 68.45% | Hace 6 meses | Histórico |
| v1.1-improved | 200 | 71.82% | Hace 5.8 meses | Histórico |
| v1.2-optimized | 300 | 74.35% | Hace 4 meses | Histórico |
| v1.3-tuned | 350 | 77.23% | Hace 3.6 meses | Histórico |
| v2.0-categorical | 450 | 80.12% | Hace 2 meses | Histórico |
| v2.1-enhanced | 500 | 82.45% | Hace 1.6 meses | Histórico |
| v2.2-stable | 600 | 84.67% | Hace 1 mes | Producción |
| v2.3-optimized | 650 | 86.23% | Hace 20 días | Producción |
| v2.4-production | 700 | 87.45% | Hace 10 días | Producción |
| v2.5-latest | 750 | 88.92% | Hace 3 días | Producción |

### Progresión de Métricas

Los datos muestran una mejora progresiva del modelo:
- **Precisión mínima**: 68.45% (v1.0-baseline)
- **Precisión máxima**: 88.92% (v2.5-latest)
- **Precisión promedio**: ~80%
- **Mejora total**: +20.47 puntos porcentuales

### Métricas Incluidas por Entrenamiento

Cada entrenamiento incluye:
- `train_accuracy`: Precisión en datos de entrenamiento
- `test_accuracy`: Precisión en datos de prueba
- `precision`: Precisión del clasificador
- `recall`: Recall/Sensibilidad
- `f1_score`: F1-Score (media armónica)
- `pca_components`: Número de componentes PCA
- `training_samples`: Cantidad de estudiantes usados
- `configuration`: Descripción de la configuración del modelo

## ✅ Verificación

Después de ejecutar el script, verifica que los datos se insertaron correctamente:

```sql
-- Verificar total de entrenamientos
SELECT COUNT(*) FROM entrenamientos_modelo;
-- Debe retornar: 10

-- Ver resumen de métricas
SELECT
    MIN(precision_modelo) as precision_minima,
    MAX(precision_modelo) as precision_maxima,
    ROUND(AVG(precision_modelo)::numeric, 2) as precision_promedio
FROM entrenamientos_modelo;

-- Ver todos los entrenamientos
SELECT
    id,
    TO_CHAR(fecha_entrenamiento, 'YYYY-MM-DD') as fecha,
    num_estudiantes_entrenamiento as estudiantes,
    ROUND(precision_modelo, 2) as precision,
    modelo_version
FROM entrenamientos_modelo
ORDER BY fecha_entrenamiento DESC;
```

## 🎨 Visualización en la Interfaz

Una vez ejecutado el script, la página de **"Comparación de Modelos"** mostrará:

1. **Métricas Generales**:
   - Total de entrenamientos: 10
   - Precisión promedio: ~80%
   - Precisión máxima: 88.92%
   - Precisión mínima: 68.45%

2. **Historial de Entrenamientos**:
   - Tabla con todos los entrenamientos
   - Ordenados del más reciente al más antiguo
   - Métricas detalladas por cada uno

## 🔄 Actualización Futura

Cuando ejecutes nuevos entrenamientos reales del modelo:

1. El sistema guardará automáticamente los entrenamientos usando la clase `EntrenamientosDB` en `database.py`
2. Los nuevos entrenamientos se agregarán a la tabla sin eliminar los históricos
3. Las métricas se actualizarán en tiempo real

Ejemplo de código para guardar un entrenamiento:

```python
from database import EntrenamientosDB

db = EntrenamientosDB()

entrenamiento_data = {
    'num_estudiantes': 800,
    'precision': 89.5,  # en porcentaje
    'metricas': {
        'train_accuracy': 0.92,
        'test_accuracy': 0.895,
        'precision': 0.91,
        'recall': 0.88,
        'f1_score': 0.895
    },
    'version': 'v3.0',
    'observaciones': 'Nuevo entrenamiento con datos actualizados',
    'ruta_modelo': 'models/v3.0/model.pkl'
}

db.guardar_entrenamiento(entrenamiento_data)
db.close()
```

## ⚠️ Notas Importantes

1. **Limpieza de datos**: El script SQL incluye un `DELETE FROM entrenamientos_modelo` al inicio. Si quieres mantener entrenamientos existentes, comenta esa línea.

2. **Fechas relativas**: Las fechas se calculan relativamente a la fecha actual usando `NOW() - INTERVAL 'X days'`, por lo que siempre serán relevantes.

3. **Datos de prueba**: Estos son datos de referencia simulados. En producción, los entrenamientos reales reemplazarán gradualmente estos datos.

4. **Formato JSON**: Las métricas se almacenan en formato JSONB de PostgreSQL, lo que permite consultas eficientes y flexibles.

## 🆘 Solución de Problemas

### Error: "relation entrenamientos_modelo does not exist"
**Solución**: Ejecuta primero `init.sql` para crear las tablas:
```bash
psql postgresql://... -f init.sql
```

### Error: Connection refused
**Solución**: Verifica que:
- El archivo `.env` tenga las credenciales correctas de Railway
- Tengas acceso a internet
- Las credenciales no hayan expirado

### La interfaz no muestra datos
**Solución**:
1. Verifica que los datos se insertaron: `SELECT COUNT(*) FROM entrenamientos_modelo;`
2. Reinicia el servidor web
3. Revisa la consola del navegador para errores
4. Verifica que el endpoint de la API funcione correctamente

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs de la base de datos
2. Verifica las credenciales en `.env`
3. Prueba la conexión manualmente con psql
4. Consulta la documentación de Railway para problemas de conectividad

---

**¡Listo!** Con estos datos, tu sistema de comparación de modelos tendrá información histórica para retroalimentar y mejorar continuamente. 🚀
