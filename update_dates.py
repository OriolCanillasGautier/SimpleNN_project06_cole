# ============================================
# Script per actualitzar les dates dels datasets
# ============================================
# Aquest script actualitza les dates dels CSVs per tenir dades més recents
# Això permetrà que el model LSTM faci prediccions correctes per dates futures

import pandas as pd
from datetime import datetime, timedelta

def update_sales_data1():
    """
    Actualitza les dates de sales_data1.csv
    Dates originals: 2003-2005 -> Noves dates: 2023-2025
    """
    print("📊 Actualitzant sales_data1.csv...")
    
    # Provar diferents encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv('sales_data1.csv', encoding=encoding)
            print(f"   Encoding utilitzat: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    # La columna de data és ORDERDATE
    # Format: "2/24/2003 0:00"
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], format='%m/%d/%Y %H:%M')
    
    # Calcular l'offset d'anys per arribar a 2025-2027
    # Com les dates actuals són 2023-2025, afegim 2 anys més
    years_offset = 2
    
    # Actualitzar la data
    df['ORDERDATE'] = df['ORDERDATE'] + pd.DateOffset(years=years_offset)
    
    # Actualitzar també les columnes YEAR_ID
    df['YEAR_ID'] = df['YEAR_ID'] + years_offset
    
    # Formatejar la data al format original
    df['ORDERDATE'] = df['ORDERDATE'].dt.strftime('%m/%d/%Y %H:%M')
    
    # Guardar
    df.to_csv('sales_data1.csv', index=False, encoding='utf-8')
    
    print(f"✅ sales_data1.csv actualitzat! Dates ara: 2025-2027")
    print(f"   - Files processades: {len(df)}")
    print(f"   - Anys únics: {sorted(df['YEAR_ID'].unique())}")


def update_sales_data2():
    """
    Actualitza les dates de sales_data2.csv
    Dates originals: 2019 -> Noves dates: 2025
    """
    print("\n📊 Actualitzant sales_data2.csv...")
    
    df = pd.read_csv('sales_data2.csv')
    
    # La columna de data és "Order Date"
    # Format: "2019-01-22 21:25:00"
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    
    # Calcular l'offset d'anys per arribar a 2026-2027
    # Com les dates actuals són 2024-2025, afegim 2 anys més
    years_offset = 2
    
    # Actualitzar la data
    df['Order Date'] = df['Order Date'] + pd.DateOffset(years=years_offset)
    
    # Formatejar la data al format original
    df['Order Date'] = df['Order Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Guardar
    df.to_csv('sales_data2.csv', index=False)
    
    # Mostrar rang de dates
    dates = pd.to_datetime(df['Order Date'])
    print(f"✅ sales_data2.csv actualitzat!")
    print(f"   - Files processades: {len(df)}")
    print(f"   - Rang de dates: {dates.min()} a {dates.max()}")


if __name__ == '__main__':
    print("=" * 60)
    print("🔄 ACTUALITZACIÓ DE DATES DELS DATASETS")
    print("=" * 60)
    print("Aquest script actualitzarà les dates dels CSVs perquè")
    print("els models LSTM puguin fer prediccions correctes per 2025-2026")
    print("=" * 60)
    
    update_sales_data1()
    update_sales_data2()
    
    print("\n" + "=" * 60)
    print("✅ ACTUALITZACIÓ COMPLETADA!")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Ara has de re-entrenar els models!")
    print("   Executa els notebooks sales_data1.ipynb i sales_data2.ipynb")
    print("   per entrenar els models amb les noves dates.")
    print("=" * 60)
