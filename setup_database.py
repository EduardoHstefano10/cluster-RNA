"""
Script para inicializar la base de datos en Railway
Ejecuta el script init.sql para crear las tablas necesarias
"""

import psycopg2
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

# Cargar variables de entorno
load_dotenv()


def get_db_connection():
    """Obtener conexión a PostgreSQL (soporta DATABASE_URL y variables individuales)"""
    
    # Opción 1: Usar DATABASE_URL si está disponible
    database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_PUBLIC_URL')
    
    if database_url:
        print("🔗 Usando DATABASE_URL para conectar...")
        return psycopg2.connect(database_url)
    
    # Opción 2: Usar variables individuales
    print("🔗 Usando variables individuales para conectar...")
    return psycopg2.connect(
        host=os.getenv('PGHOST', os.getenv('DB_HOST', 'localhost')),
        port=int(os.getenv('PGPORT', os.getenv('DB_PORT', 5432))),
        database=os.getenv('PGDATABASE', os.getenv('DB_NAME', 'railway')),
        user=os.getenv('PGUSER', os.getenv('DB_USER', 'postgres')),
        password=os.getenv('PGPASSWORD', os.getenv('DB_PASSWORD', ''))
    )


def setup_database():
    """Inicializar base de datos ejecutando init.sql"""
    print("🚀 Iniciando configuración de base de datos en Railway...")
    
    # Conectar a la base de datos
    print("🔌 Conectando a PostgreSQL...")
    try:
        conn = get_db_connection()
        conn.set_session(autocommit=True)
        cursor = conn.cursor()
        print("✅ Conexión establecida exitosamente")
        
        # Mostrar información de conexión
        database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_PUBLIC_URL')
        if database_url:
            parsed = urlparse(database_url)
            print(f"📍 Host: {parsed.hostname}")
            print(f"📍 Puerto: {parsed.port}")
            print(f"📍 Base de datos: {parsed.path.lstrip('/')}")
            print(f"📍 Usuario: {parsed.username}")
        else:
            print(f"📍 Host: {os.getenv('PGHOST', os.getenv('DB_HOST'))}")
            print(f"📍 Puerto: {os.getenv('PGPORT', os.getenv('DB_PORT'))}")
            print(f"📍 Base de datos: {os.getenv('PGDATABASE', os.getenv('DB_NAME'))}")
            print(f"📍 Usuario: {os.getenv('PGUSER', os.getenv('DB_USER'))}")
        
    except Exception as e:
        print(f"❌ Error al conectar a PostgreSQL: {e}")
        print("\n💡 Asegúrate de que:")
        print("   1. Las variables de entorno estén configuradas correctamente en .env")
        print("   2. Tienes acceso a Internet para conectar con Railway")
        print("   3. Las credenciales de Railway son correctas")
        return False
    
    # Leer script SQL
    print("\n📂 Leyendo script init.sql...")
    try:
        with open('init.sql', 'r', encoding='utf-8') as f:
            sql_script = f.read()
        print("✅ Script SQL leído correctamente")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo init.sql")
        cursor.close()
        conn.close()
        return False
    
    # Ejecutar script SQL
    print("\n🔧 Ejecutando script SQL...")
    try:
        cursor.execute(sql_script)
        print("✅ Script SQL ejecutado exitosamente")
    except Exception as e:
        print(f"❌ Error al ejecutar script SQL: {e}")
        cursor.close()
        conn.close()
        return False
    
    # Verificar que las tablas fueron creadas
    print("\n🔍 Verificando tablas creadas...")
    try:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print("✅ Tablas creadas:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("⚠️  No se encontraron tablas en la base de datos")
            
    except Exception as e:
        print(f"❌ Error al verificar tablas: {e}")
    
    # Verificar datos de ejemplo
    print("\n📊 Verificando datos de ejemplo...")
    try:
        cursor.execute("SELECT COUNT(*) FROM estudiantes;")
        count = cursor.fetchone()[0]
        print(f"✅ Total de estudiantes en la base de datos: {count}")
        
        if count > 0:
            cursor.execute("SELECT codigo, nombre FROM estudiantes LIMIT 3;")
            students = cursor.fetchall()
            print("   Ejemplos:")
            for student in students:
                print(f"   - {student[0]}: {student[1]}")
    except Exception as e:
        print(f"⚠️  Error al verificar datos: {e}")
    
    # Cerrar conexión
    cursor.close()
    conn.close()
    print("\n✅ Configuración de base de datos completada")
    print("🎉 ¡Base de datos lista para usar!")
    
    return True


if __name__ == "__main__":
    try:
        success = setup_database()
        if success:
            print("\n" + "="*50)
            print("✅ Base de datos inicializada correctamente")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ Hubo errores durante la inicialización")
            print("="*50)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
