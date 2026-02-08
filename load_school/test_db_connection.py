#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки подключения к PostgreSQL и наличия необходимых таблиц.
"""

import psycopg2
import sys


# Параметры подключения к PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'education_db',
    'user': 'postgres',
    'password': 'your_password'
}


def test_connection():
    """
    Проверяет подключение к базе данных и наличие необходимых таблиц.
    """
    print("="*60)
    print("🔌 ПРОВЕРКА ПОДКЛЮЧЕНИЯ К БАЗЕ ДАННЫХ POSTGRESQL")
    print("="*60)
    
    try:
        # Пытаемся подключиться
        print("\n1️⃣  Попытка подключения к базе данных...")
        print(f"   Host: {DB_CONFIG['host']}")
        print(f"   Port: {DB_CONFIG['port']}")
        print(f"   Database: {DB_CONFIG['database']}")
        print(f"   User: {DB_CONFIG['user']}")
        
        conn = psycopg2.connect(**DB_CONFIG)
        print("   ✅ Подключение успешно установлено!")
        
        cursor = conn.cursor()
        
        # Проверяем версию PostgreSQL
        print("\n2️⃣  Проверка версии PostgreSQL...")
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"   ✅ {version}")
        
        # Проверяем наличие схемы edu
        print("\n3️⃣  Проверка наличия схемы 'edu'...")
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = 'edu';
        """)
        schema_exists = cursor.fetchone()
        
        if schema_exists:
            print("   ✅ Схема 'edu' существует")
        else:
            print("   ❌ Схема 'edu' НЕ НАЙДЕНА!")
            print("   💡 Выполните SQL-скрипт create_database.sql")
            conn.close()
            sys.exit(1)
        
        # Проверяем наличие таблиц
        print("\n4️⃣  Проверка наличия таблиц...")
        
        required_tables = ['region', 'municipality', 'school']
        
        for table in required_tables:
            cursor.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'edu' AND table_name = '{table}';
            """)
            table_exists = cursor.fetchone()
            
            if table_exists:
                # Подсчитываем количество записей
                cursor.execute(f"SELECT COUNT(*) FROM edu.{table};")
                count = cursor.fetchone()[0]
                print(f"   ✅ Таблица 'edu.{table}' существует ({count} записей)")
            else:
                print(f"   ❌ Таблица 'edu.{table}' НЕ НАЙДЕНА!")
                print("   💡 Выполните SQL-скрипт create_database.sql")
                conn.close()
                sys.exit(1)
        
        # Проверяем структуру таблиц
        print("\n5️⃣  Проверка структуры таблиц...")
        
        # Таблица region
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'edu' AND table_name = 'region'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print(f"   📋 Таблица 'region':")
        for col_name, col_type in columns:
            print(f"      - {col_name}: {col_type}")
        
        # Таблица municipality
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'edu' AND table_name = 'municipality'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print(f"\n   📋 Таблица 'municipality':")
        for col_name, col_type in columns:
            print(f"      - {col_name}: {col_type}")
        
        # Таблица school
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'edu' AND table_name = 'school'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print(f"\n   📋 Таблица 'school':")
        for col_name, col_type in columns:
            print(f"      - {col_name}: {col_type}")
        
        # Проверяем внешние ключи
        print("\n6️⃣  Проверка внешних ключей...")
        cursor.execute("""
            SELECT
                tc.constraint_name, 
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' 
              AND tc.table_schema = 'edu'
            ORDER BY tc.table_name;
        """)
        
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            for fk in foreign_keys:
                print(f"   ✅ {fk[1]}.{fk[2]} -> {fk[3]}.{fk[4]}")
        else:
            print("   ⚠️  Внешние ключи не найдены")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*60)
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*60)
        print("\n💡 База данных готова к загрузке данных.")
        print("   Запустите скрипт: python load_schools_to_db.py")
        print("="*60)
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ ОШИБКА ПОДКЛЮЧЕНИЯ К БД:")
        print(f"   {e}")
        print("\n💡 Возможные причины:")
        print("   - PostgreSQL не запущен")
        print("   - Неверные параметры подключения (хост, порт, БД, пользователь, пароль)")
        print("   - Нет прав доступа для пользователя")
        print("   - База данных не существует")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ НЕОЖИДАННАЯ ОШИБКА:")
        print(f"   {e}")
        sys.exit(1)


if __name__ == '__main__':
    test_connection()
