# 🚂 Configuración de Base de Datos en Railway

## 📋 Pasos para configurar tu aplicación con Railway PostgreSQL

### 1. Obtener las Variables de Entorno de Railway

En tu proyecto de Railway, ve a la pestaña **Variables** y copia las siguientes variables:

- `DATABASE_URL` o `DATABASE_PUBLIC_URL`
- `PGHOST`
- `PGPORT`
- `PGDATABASE`
- `PGUSER`
- `PGPASSWORD`

### 2. Configurar el archivo .env

Crea o actualiza el archivo `.env` en la raíz del proyecto con los valores reales de Railway:

```bash
# Variables de Railway PostgreSQL
DATABASE_URL=postgresql://usuario:password@host:puerto/database
PGHOST=tu-host.railway.app
PGPORT=5432
PGDATABASE=railway
PGUSER=postgres
PGPASSWORD=tu-password-real
```

**IMPORTANTE:** Reemplaza los valores con los que obtienes de Railway. El archivo `.env` ya está en `.gitignore`, por lo que no se subirá al repositorio.

### 3. Inicializar la Base de Datos

Ejecuta el script de configuración para crear las tablas:

```bash
python setup_database.py
```

Este script:
- ✅ Se conecta a PostgreSQL en Railway
- ✅ Ejecuta el script `init.sql` para crear las tablas
- ✅ Crea algunos datos de ejemplo
- ✅ Verifica que todo esté configurado correctamente

### 4. Cargar los Datos del CSV

Una vez que las tablas estén creadas, carga los datos del CSV:

```bash
python load_csv_to_db.py
```

Este script:
- ✅ Lee el archivo `data/estudiantes_data.csv`
- ✅ Convierte los datos numéricos a categorías
- ✅ Inserta todos los estudiantes en la base de datos
- ✅ Muestra estadísticas de la carga

### 5. Iniciar el Servidor

Finalmente, inicia el servidor FastAPI:

```bash
python main.py
```

O usando uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Configuración en Railway (Producción)

Para desplegar en Railway:

### 1. Variables de Entorno en Railway

Railway automáticamente configura las variables de PostgreSQL cuando añades el servicio de base de datos. Asegúrate de que tu servicio web tenga acceso a estas variables.

### 2. Archivo Procfile (si es necesario)

Crea un archivo `Procfile` en la raíz:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### 3. Requirements.txt

Asegúrate de tener todas las dependencias en `requirements.txt`:

```
fastapi
uvicorn
psycopg2-binary
python-dotenv
pandas
pydantic
```

### 4. Script de Inicio

Railway ejecutará automáticamente:
1. Instalará las dependencias de `requirements.txt`
2. Ejecutará el comando especificado en `Procfile`

**IMPORTANTE:** Debes ejecutar manualmente `setup_database.py` y `load_csv_to_db.py` desde la consola de Railway la primera vez:

```bash
# Desde la consola de Railway
python setup_database.py
python load_csv_to_db.py
```

## 🌐 Verificar la Conexión

Para verificar que todo funciona:

```bash
# Probar la conexión a la base de datos
python database.py
```

Deberías ver:
```
✅ Conexión exitosa a PostgreSQL
📊 Estadísticas:
  Total estudiantes: X
  Alto riesgo: Y
  Clusters: {...}
```

## 📝 Notas Importantes

1. **Seguridad:** Nunca subas el archivo `.env` al repositorio. Ya está en `.gitignore`.

2. **Variables de Railway:** Railway regenera automáticamente las variables de entorno cuando agregas el servicio PostgreSQL. Cópialas directamente desde la interfaz de Railway.

3. **Conexión desde Local:** Para conectarte desde tu máquina local a Railway:
   - Usa `DATABASE_PUBLIC_URL` que permite conexiones externas
   - Asegúrate de tener conexión a Internet

4. **Primera Carga:** Solo necesitas ejecutar `setup_database.py` y `load_csv_to_db.py` una vez. Después, los datos persisten en Railway.

## 🆘 Solución de Problemas

### Error: "could not translate host name"
- Verifica que tienes conexión a Internet
- Comprueba que las variables de Railway estén correctamente copiadas en `.env`

### Error: "permission denied"
- Verifica que el usuario de PostgreSQL tenga permisos
- Railway debería configurar esto automáticamente

### Error: "module not found"
- Instala las dependencias: `pip install -r requirements.txt`

### Los datos no se cargan
- Verifica que el archivo `data/estudiantes_data.csv` exista
- Revisa que la ruta en `load_csv_to_db.py` sea correcta

## ✅ Checklist de Configuración

- [ ] Crear servicio PostgreSQL en Railway
- [ ] Copiar variables de entorno de Railway a `.env`
- [ ] Ejecutar `pip install -r requirements.txt`
- [ ] Ejecutar `python setup_database.py`
- [ ] Ejecutar `python load_csv_to_db.py`
- [ ] Ejecutar `python main.py` o desplegar en Railway
- [ ] Verificar que el backend responde en `/api/stats`
- [ ] Verificar que los estudiantes aparecen en `/api/students`

¡Listo! Tu aplicación debería estar conectada a Railway PostgreSQL. 🎉
