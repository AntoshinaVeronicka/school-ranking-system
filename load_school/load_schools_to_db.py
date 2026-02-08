#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обработки данных об образовательных учреждениях из Excel-файла
и загрузки их в реляционную базу данных PostgreSQL.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import re
import sys


# ========== НАСТРОЙКИ ==========

# Параметры подключения к PostgreSQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'education_db',
    'user': 'postgres',
    'password': 'your_password'
}

# Файлы для обработки
EXCEL_FILES = [
    '/mnt/user-data/uploads/Приморский_край.xlsx',
    '/mnt/user-data/uploads/Хабаровский_край.xlsx'
]


# ========== ФУНКЦИИ ОБРАБОТКИ ДАННЫХ ==========

def standardize_school_name(name):
    """
    Приводит полное официальное наименование школы к стандартному сокращённому виду.
    
    Args:
        name (str): Полное название школы
        
    Returns:
        str: Сокращённое название школы
    """
    if pd.isna(name) or not isinstance(name, str):
        return name
    
    # Удаляем лишние пробелы
    name = ' '.join(name.split())
    
    # Словарь замен (регистронезависимые)
    replacements = {
        r'Муниципальное\s+автономное\s+общеобразовательное\s+учреждение': 'МАОУ',
        r'Муниципальное\s+бюджетное\s+общеобразовательное\s+учреждение': 'МБОУ',
        r'Муниципальное\s+казенное\s+общеобразовательное\s+учреждение': 'МКОУ',
        r'Государственное\s+общеобразовательное\s+учреждение': 'ГОУ',
        r'Государственное\s+бюджетное\s+общеобразовательное\s+учреждение': 'ГБОУ',
        r'Федеральное\s+государственное\s+автономное\s+образовательное\s+учреждение\s+высшего\s+образования': 'ФГАОУ ВО',
        r'Средняя\s+общеобразовательная\s+школа': 'СОШ',
        r'Общеобразовательная\s+школа': 'ОШ',
        r'средняя\s+общеобразовательная\s+школа': 'СОШ',
        r'общеобразовательная\s+школа': 'ОШ'
    }
    
    # Применяем замены
    result = name
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Удаляем лишние кавычки в начале/конце строки
    result = result.strip('"').strip("'").strip('«').strip('»')
    
    # Удаляем множественные пробелы
    result = ' '.join(result.split())
    
    return result


def extract_municipality_name(municipality_full):
    """
    Извлекает название муниципального образования из строки вида '(10) Город Владивосток'.
    
    Args:
        municipality_full (str): Полная строка с кодом и названием
        
    Returns:
        str: Название муниципального образования
    """
    if pd.isna(municipality_full) or not isinstance(municipality_full, str):
        return municipality_full
    
    # Удаляем код в скобках, если он есть
    match = re.match(r'\(\d+\)\s*(.+)', municipality_full)
    if match:
        return match.group(1).strip()
    
    return municipality_full.strip()


def extract_school_name(school_full):
    """
    Извлекает название школы из строки вида '(8) МБОУ Гимназия № 1 г. Владивосток'.
    
    Args:
        school_full (str): Полная строка с кодом и названием
        
    Returns:
        str: Название школы
    """
    if pd.isna(school_full) or not isinstance(school_full, str):
        return school_full
    
    # Удаляем код в скобках, если он есть
    match = re.match(r'\(\d+\)\s*(.+)', school_full)
    if match:
        return match.group(1).strip()
    
    return school_full.strip()


def read_excel_files(file_paths):
    """
    Читает Excel-файлы и объединяет данные.
    
    Args:
        file_paths (list): Список путей к Excel-файлам
        
    Returns:
        pd.DataFrame: Объединённый DataFrame с данными
    """
    all_data = []
    
    for file_path in file_paths:
        print(f"\n📖 Читаю файл: {file_path}")
        
        # Определяем регион из имени файла
        if 'Приморский' in file_path:
            region_name = 'Приморский край'
        elif 'Хабаровский' in file_path:
            region_name = 'Хабаровский край'
        else:
            # Пытаемся извлечь из файла
            region_name = file_path.split('/')[-1].replace('.xlsx', '').replace('_', ' ')
        
        # Читаем файл, пропуская заголовочные строки
        df = pd.read_excel(file_path, header=None)
        
        # Находим строку с заголовками (обычно строка 5-7)
        header_row = None
        for idx in range(min(10, len(df))):
            if df.iloc[idx, 1] is not None and 'Муниципальное образование' in str(df.iloc[idx, 1]):
                header_row = idx
                break
        
        if header_row is None:
            print(f"⚠️  Не найдена строка с заголовками в файле {file_path}")
            continue
        
        # Находим начало данных (обычно через 1-2 строки после заголовка)
        data_start_row = header_row + 2
        
        # Читаем данные
        df_data = df.iloc[data_start_row:].copy()
        
        # Отбираем нужные столбцы (1 - муниципалитет, 2 - школа)
        df_data = df_data[[1, 2]].copy()
        df_data.columns = ['municipality', 'school']
        
        # Фильтруем пустые строки и служебные строки
        df_data = df_data.dropna(subset=['municipality', 'school'])
        df_data = df_data[~df_data['municipality'].astype(str).str.contains('ВСЕГО', case=False, na=False)]
        df_data = df_data[~df_data['school'].astype(str).str.contains('ВСЕГО', case=False, na=False)]
        
        # Фильтруем строки с нумерацией (например, "1" и "2")
        df_data = df_data[~((df_data['municipality'].astype(str).str.strip().str.isdigit()) & 
                            (df_data['school'].astype(str).str.strip().str.isdigit()))]
        
        # Добавляем регион
        df_data['region'] = region_name
        
        # Извлекаем чистые названия
        df_data['municipality'] = df_data['municipality'].apply(extract_municipality_name)
        df_data['school'] = df_data['school'].apply(extract_school_name)
        
        # Обрабатываем названия школ
        df_data['school'] = df_data['school'].apply(standardize_school_name)
        
        all_data.append(df_data)
        print(f"✅ Прочитано строк: {len(df_data)}")
    
    # Объединяем все данные
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Всего строк прочитано: {len(combined_df)}")
        return combined_df
    else:
        print("❌ Нет данных для обработки")
        return pd.DataFrame()


def insert_regions(conn, regions):
    """
    Вставляет регионы в таблицу edu.region.
    
    Args:
        conn: Подключение к БД
        regions (list): Список уникальных названий регионов
        
    Returns:
        int: Количество добавленных регионов
    """
    cursor = conn.cursor()
    
    # Подготавливаем данные для вставки
    values = [(r,) for r in regions]
    
    # SQL запрос с ON CONFLICT DO NOTHING
    query = """
        INSERT INTO edu.region (name)
        VALUES %s
        ON CONFLICT (name) DO NOTHING
    """
    
    execute_values(cursor, query, values)
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    
    return inserted_count


def get_region_ids(conn, regions):
    """
    Получает ID регионов по их названиям.
    
    Args:
        conn: Подключение к БД
        regions (list): Список названий регионов
        
    Returns:
        dict: Словарь {название_региона: region_id}
    """
    cursor = conn.cursor()
    
    query = "SELECT region_id, name FROM edu.region WHERE name = ANY(%s)"
    cursor.execute(query, (list(regions),))
    
    region_dict = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.close()
    
    return region_dict


def insert_municipalities(conn, municipalities_data):
    """
    Вставляет муниципалитеты в таблицу edu.municipality.
    
    Args:
        conn: Подключение к БД
        municipalities_data (list): Список кортежей (region_id, municipality_name)
        
    Returns:
        int: Количество добавленных муниципалитетов
    """
    cursor = conn.cursor()
    
    # SQL запрос с ON CONFLICT DO NOTHING
    query = """
        INSERT INTO edu.municipality (region_id, name)
        VALUES %s
        ON CONFLICT (region_id, name) DO NOTHING
    """
    
    execute_values(cursor, query, municipalities_data)
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    
    return inserted_count


def get_municipality_ids(conn, region_municipalities):
    """
    Получает ID муниципалитетов.
    
    Args:
        conn: Подключение к БД
        region_municipalities (list): Список кортежей (region_id, municipality_name)
        
    Returns:
        dict: Словарь {(region_id, municipality_name): municipality_id}
    """
    cursor = conn.cursor()
    
    query = "SELECT municipality_id, region_id, name FROM edu.municipality"
    cursor.execute(query)
    
    municipality_dict = {(row[1], row[2]): row[0] for row in cursor.fetchall()}
    cursor.close()
    
    return municipality_dict


def insert_schools(conn, schools_data):
    """
    Вставляет школы в таблицу edu.school.
    
    Args:
        conn: Подключение к БД
        schools_data (list): Список кортежей (municipality_id, school_name)
        
    Returns:
        int: Количество добавленных школ
    """
    cursor = conn.cursor()
    
    # SQL запрос с ON CONFLICT DO NOTHING
    query = """
        INSERT INTO edu.school (municipality_id, full_name, is_active)
        VALUES %s
        ON CONFLICT (municipality_id, full_name) DO NOTHING
    """
    
    # Добавляем is_active = True
    schools_data_with_active = [(m_id, name, True) for m_id, name in schools_data]
    
    execute_values(cursor, query, schools_data_with_active)
    inserted_count = cursor.rowcount
    conn.commit()
    cursor.close()
    
    return inserted_count


def load_data_to_db(df, db_config):
    """
    Загружает данные в базу данных PostgreSQL.
    
    Args:
        df (pd.DataFrame): DataFrame с данными
        db_config (dict): Параметры подключения к БД
    """
    print("\n" + "="*60)
    print("🔌 Подключение к базе данных PostgreSQL...")
    print("="*60)
    
    try:
        # Подключаемся к БД
        conn = psycopg2.connect(**db_config)
        print("✅ Подключение установлено")
        
        # 1. Вставка регионов
        print("\n📍 Обработка регионов...")
        unique_regions = df['region'].unique().tolist()
        print(f"   Уникальных регионов: {len(unique_regions)}")
        
        inserted_regions = insert_regions(conn, unique_regions)
        print(f"   Добавлено новых регионов: {inserted_regions}")
        
        # Получаем ID регионов
        region_ids = get_region_ids(conn, unique_regions)
        print(f"   Всего регионов в БД: {len(region_ids)}")
        
        # 2. Вставка муниципалитетов
        print("\n🏛️  Обработка муниципалитетов...")
        
        # Подготавливаем данные для муниципалитетов
        municipalities = df[['region', 'municipality']].drop_duplicates()
        municipalities_data = [
            (region_ids[row['region']], row['municipality'])
            for _, row in municipalities.iterrows()
            if row['region'] in region_ids
        ]
        
        print(f"   Уникальных муниципалитетов: {len(municipalities_data)}")
        
        inserted_municipalities = insert_municipalities(conn, municipalities_data)
        print(f"   Добавлено новых муниципалитетов: {inserted_municipalities}")
        
        # Получаем ID муниципалитетов
        municipality_ids = get_municipality_ids(conn, municipalities_data)
        print(f"   Всего муниципалитетов в БД: {len(municipality_ids)}")
        
        # 3. Вставка школ
        print("\n🏫 Обработка школ...")
        
        # Подготавливаем данные для школ
        schools_data = []
        for _, row in df.iterrows():
            region_id = region_ids.get(row['region'])
            if region_id:
                municipality_key = (region_id, row['municipality'])
                municipality_id = municipality_ids.get(municipality_key)
                if municipality_id:
                    schools_data.append((municipality_id, row['school']))
        
        # Удаляем дубликаты
        schools_data = list(set(schools_data))
        
        print(f"   Уникальных школ: {len(schools_data)}")
        
        inserted_schools = insert_schools(conn, schools_data)
        print(f"   Добавлено новых школ: {inserted_schools}")
        
        # Закрываем соединение
        conn.close()
        print("\n✅ Данные успешно загружены в базу данных!")
        
        # Итоговая статистика
        print("\n" + "="*60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("="*60)
        print(f"   Регионов обработано: {len(unique_regions)}")
        print(f"   Новых регионов добавлено: {inserted_regions}")
        print(f"   Муниципалитетов обработано: {len(municipalities_data)}")
        print(f"   Новых муниципалитетов добавлено: {inserted_municipalities}")
        print(f"   Школ обработано: {len(schools_data)}")
        print(f"   Новых школ добавлено: {inserted_schools}")
        print("="*60)
        
    except psycopg2.Error as e:
        print(f"\n❌ Ошибка при работе с БД: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)


# ========== ГЛАВНАЯ ФУНКЦИЯ ==========

def main():
    """
    Главная функция скрипта.
    """
    print("="*60)
    print("🚀 ЗАГРУЗКА ДАННЫХ ОБ ОБРАЗОВАТЕЛЬНЫХ УЧРЕЖДЕНИЯХ В БД")
    print("="*60)
    
    # Читаем данные из Excel
    df = read_excel_files(EXCEL_FILES)
    
    if df.empty:
        print("\n❌ Нет данных для загрузки")
        sys.exit(1)
    
    # Показываем примеры обработанных данных
    print("\n📋 Примеры обработанных данных:")
    print(df.head(10).to_string(index=False))
    
    # Загружаем в БД
    load_data_to_db(df, DB_CONFIG)
    
    print("\n✅ Скрипт успешно завершён!")


if __name__ == '__main__':
    main()


# ========== РЕКОМЕНДАЦИИ ПО ПОДГОТОВКЕ EXCEL-ФАЙЛА ==========
"""
Чтобы скрипт работал корректно и без ошибок, рекомендуется подготовить Excel-файл следующим образом:

1. СТРУКТУРА ДАННЫХ:
   - Убедитесь, что данные начинаются с понятной строки заголовков
   - Заголовки должны содержать текст "Муниципальное образование" и "Образовательная организация"
   - Данные должны начинаться через 1-2 строки после заголовков

2. ФОРМАТ ЯЧЕЕК:
   - Удалите все объединённые ячейки (Merge cells)
   - Уберите сложное форматирование (цвета, границы не критичны, но могут замедлить чтение)
   - Установите формат ячеек как "Текст" для названий регионов, муниципалитетов и школ

3. КОНСИСТЕНТНОСТЬ ДАННЫХ:
   - Названия регионов должны быть записаны единообразно во всех строках
   - Названия муниципалитетов не должны иметь опечаток или разных вариантов написания
   - Избегайте лишних пробелов в начале/конце названий

4. СТРУКТУРА СТОЛБЦОВ:
   - Столбец 1 (B): Муниципальное образование
   - Столбец 2 (C): Название школы
   - Коды в скобках типа "(10) Город Владивосток" допустимы - скрипт их обработает

5. ОЧИСТКА ДАННЫХ:
   - Удалите пустые строки между данными
   - Удалите строки с промежуточными итогами ("ВСЕГО по...")
   - Проверьте, что все строки с данными имеют заполненные значения в обоих столбцах

6. КОДИРОВКА:
   - Убедитесь, что файл сохранён в формате .xlsx (не .xls)
   - Кириллица должна отображаться корректно

7. ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДАЦИИ:
   - Если в файле несколько листов, данные должны быть на первом листе
   - Не используйте формулы в ячейках с данными - только текст
   - Стандартизируйте форматы названий перед импортом, если возможно

ПРИМЕР ПРАВИЛЬНОЙ СТРУКТУРЫ:

Строка 5-6: Заголовки
| Муниципальное образование | Образовательная организация (школа) |
|--------------------------|-------------------------------------|
Строка 8+: Данные
| (10) Город Владивосток | (8) МБОУ Гимназия № 1 г. Владивосток |
| (13) Уссурийский ГО    | (286) МБОУ СОШ № 14 г. Уссурийск     |

Следуя этим рекомендациям, вы значительно упростите процесс импорта данных и снизите
вероятность ошибок при работе скрипта.
"""
