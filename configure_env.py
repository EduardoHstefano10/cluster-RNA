"""
Script interactivo para configurar las variables de entorno de Railway
Ejecuta: python configure_env.py
"""

import os
import sys

def configure_env():
    """Configurar archivo .env de manera interactiva"""
    print("=" * 60)
    print("🚂 CONFIGURACIÓN DE RAILWAY POSTGRESQL")
    print("=" * 60)
    print()
    print("📋 Necesitas obtener las credenciales desde Railway:")
    print("   1. Ve a: https://railway.app")
    print("   2. Abre tu proyecto")
    print("   3. Click en 'PostgreSQL'")
    print("   4. Ve a la pestaña 'Variables' o 'Connect'")
    print()
    print("=" * 60)
    print()

    # Preguntar qué opción prefiere
    print("¿Cómo quieres configurar la conexión?")
    print("1. Usar DATABASE_URL completo (RECOMENDADO)")
    print("2. Usar variables individuales (PGHOST, PGPORT, etc.)")
    print()

    opcion = input("Selecciona una opción (1 o 2): ").strip()

    env_lines = [
        "# 🚂 Variables de Entorno para Railway PostgreSQL",
        "# Generado automáticamente por configure_env.py",
        "",
    ]

    if opcion == "1":
        print()
        print("📝 Copia el DATABASE_URL desde Railway")
        print("   Debería verse así:")
        print("   postgresql://postgres:contraseña@host.railway.app:5432/railway")
        print()
        database_url = input("Pega aquí tu DATABASE_URL: ").strip()

        if database_url and database_url.startswith("postgresql://"):
            env_lines.extend([
                f"DATABASE_URL={database_url}",
                "",
                "# Variables individuales (opcionales)",
                "# PGHOST=",
                "# PGPORT=5432",
                "# PGDATABASE=railway",
                "# PGUSER=postgres",
                "# PGPASSWORD=",
            ])
            success = True
        else:
            print("❌ DATABASE_URL no válido. Debe empezar con 'postgresql://'")
            success = False

    elif opcion == "2":
        print()
        print("📝 Ingresa las siguientes variables (déjalas vacías para valores por defecto):")
        print()

        pghost = input("PGHOST (ejemplo: monorail.proxy.rlwy.net): ").strip()
        pgport = input("PGPORT [5432]: ").strip() or "5432"
        pgdatabase = input("PGDATABASE [railway]: ").strip() or "railway"
        pguser = input("PGUSER [postgres]: ").strip() or "postgres"
        pgpassword = input("PGPASSWORD: ").strip()

        if pghost and pgpassword:
            env_lines.extend([
                "# Variables individuales",
                f"PGHOST={pghost}",
                f"PGPORT={pgport}",
                f"PGDATABASE={pgdatabase}",
                f"PGUSER={pguser}",
                f"PGPASSWORD={pgpassword}",
                "",
                "# Variables alternativas",
                "DB_HOST=${PGHOST}",
                "DB_PORT=${PGPORT}",
                "DB_NAME=${PGDATABASE}",
                "DB_USER=${PGUSER}",
                "DB_PASSWORD=${PGPASSWORD}",
            ])
            success = True
        else:
            print("❌ PGHOST y PGPASSWORD son obligatorios")
            success = False
    else:
        print("❌ Opción no válida")
        success = False

    if success:
        # Guardar archivo .env
        with open('.env', 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_lines) + '\n')

        print()
        print("=" * 60)
        print("✅ Archivo .env configurado correctamente")
        print("=" * 60)
        print()
        print("🧪 Probando conexión...")
        print()

        # Intentar probar la conexión
        try:
            import psycopg2
            from dotenv import load_dotenv
            load_dotenv()

            database_url = os.getenv('DATABASE_URL')
            if database_url:
                conn = psycopg2.connect(database_url)
            else:
                conn = psycopg2.connect(
                    host=os.getenv('PGHOST'),
                    port=int(os.getenv('PGPORT', 5432)),
                    database=os.getenv('PGDATABASE', 'railway'),
                    user=os.getenv('PGUSER', 'postgres'),
                    password=os.getenv('PGPASSWORD', '')
                )

            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()

            print("✅ ¡Conexión exitosa!")
            print(f"📊 PostgreSQL versión: {version[:50]}...")
            print()
            print("🎉 Ya puedes ejecutar:")
            print("   python setup_database.py")
            print("   python load_csv_to_db.py")
            print()

        except ImportError:
            print("⚠️  No se pudo verificar la conexión (falta psycopg2-binary o python-dotenv)")
            print("   Instala con: pip install psycopg2-binary python-dotenv")
        except Exception as e:
            print(f"❌ Error al conectar: {e}")
            print()
            print("💡 Verifica que:")
            print("   1. Las credenciales sean correctas")
            print("   2. Tengas acceso a Internet")
            print("   3. El servicio de Railway esté activo")
    else:
        print()
        print("=" * 60)
        print("❌ Configuración cancelada")
        print("=" * 60)
        print("   Ejecuta de nuevo el script para reintentar")

if __name__ == "__main__":
    try:
        configure_env()
    except KeyboardInterrupt:
        print("\n\n❌ Configuración cancelada por el usuario")
        sys.exit(1)
