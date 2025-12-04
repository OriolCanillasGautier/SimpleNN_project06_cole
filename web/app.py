# ============================================
# FLASK APP - Predicció de Vendes
# ============================================
# Aplicació web per predir vendes usant els models entrenats

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os

# Intentar importar keras (pot no estar disponible per models sklearn)
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

app = Flask(__name__)

# Ruta base dels models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Carregar models i preprocessadors al iniciar
def load_model_components(dataset_name):
    """Carrega el model i components per un dataset específic"""
    try:
        # Determinar si és model sklearn o keras
        sklearn_model_path = os.path.join(MODELS_DIR, f'{dataset_name}_model.pkl')
        keras_model_path = os.path.join(MODELS_DIR, f'{dataset_name}_model.keras')
        
        # Paths amb prefix del dataset
        scaler_path = os.path.join(MODELS_DIR, f'{dataset_name}_scaler.pkl')
        mappings_path = os.path.join(MODELS_DIR, f'{dataset_name}_mappings.pkl')
        encoders_path = os.path.join(MODELS_DIR, f'{dataset_name}_label_encoders.pkl')
        stats_path = os.path.join(MODELS_DIR, f'{dataset_name}_product_stats.csv')
        
        # Fallback per sales_data2 (els fitxers originals sense prefix)
        if dataset_name == 'sales_data2':
            if not os.path.exists(keras_model_path):
                keras_model_path = os.path.join(MODELS_DIR, 'sales_model.keras')
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
            if not os.path.exists(mappings_path):
                mappings_path = os.path.join(MODELS_DIR, 'mappings.pkl')
            if not os.path.exists(encoders_path):
                encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
            if not os.path.exists(stats_path):
                stats_path = os.path.join(MODELS_DIR, 'product_stats.csv')
        
        # Carregar model (sklearn o keras)
        model = None
        model_type = None
        
        if os.path.exists(sklearn_model_path):
            model = joblib.load(sklearn_model_path)
            model_type = 'sklearn'
        elif os.path.exists(keras_model_path) and KERAS_AVAILABLE:
            model = keras.models.load_model(keras_model_path)
            model_type = 'keras'
        else:
            raise FileNotFoundError(f"No s'ha trobat cap model per {dataset_name}")
        
        scaler = joblib.load(scaler_path)
        mappings = joblib.load(mappings_path)
        encoders = joblib.load(encoders_path)
        product_stats = pd.read_csv(stats_path)
        
        return {
            'model': model,
            'model_type': model_type,
            'scaler': scaler,
            'mappings': mappings,
            'encoders': encoders,
            'product_stats': product_stats,
            'loaded': True
        }
    except Exception as e:
        print(f"Error carregant {dataset_name}: {e}")
        return {'loaded': False, 'error': str(e)}

# Cache dels models
models_cache = {}

def get_model(dataset_name):
    if dataset_name not in models_cache:
        models_cache[dataset_name] = load_model_components(dataset_name)
    return models_cache[dataset_name]


@app.route('/')
def index():
    """Pàgina principal"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint per fer prediccions"""
    try:
        data = request.json
        date_str = data.get('date')
        dataset = data.get('dataset', 'sales_data2')
        product = data.get('product', 'all')
        
        # Parsejar la data
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Carregar model
        components = get_model(dataset)
        if not components.get('loaded'):
            return jsonify({
                'success': False,
                'error': f"Model no disponible per {dataset}: {components.get('error', 'Unknown error')}"
            })
        
        model = components['model']
        model_type = components.get('model_type', 'keras')
        scaler = components['scaler']
        mappings = components['mappings']
        product_stats = components['product_stats']
        encoders = components['encoders']
        
        # Filtrar productes si cal
        if product != 'all':
            stats_to_use = product_stats[product_stats['Product'] == product]
        else:
            stats_to_use = product_stats
        
        predictions = []
        feature_columns = mappings['feature_columns']
        
        for _, row in stats_to_use.iterrows():
            # Crear features segons el dataset
            if dataset == 'sales_data2':
                # Features per sales_data2 (Neural Network)
                features = {
                    'Order ID': 150000,
                    'Purchase Address': 0,
                    'Quantity Ordered': row['Quantity Ordered'],
                    'Price Each': row['Price Each'],
                    'Cost price': row.get('Cost price', row['Price Each'] * 0.6),
                    'MONTH': date.month,
                    'DAY': date.day,
                    'YEAR': date.year,
                    'TIME': 0,
                    'CATEGORY_ID': row.get('CATEGORY_ID', 1),
                    'PRODUCT_ID': row.get('PRODUCT_ID', 1)
                }
            else:
                # Features per sales_data1 (RandomForest)
                # Usem les columnes definides en el mappings
                features = {}
                for col in feature_columns:
                    if col == 'MONTH':
                        features[col] = date.month
                    elif col == 'DAY':
                        features[col] = date.day
                    elif col == 'YEAR':
                        features[col] = date.year
                    elif col == 'QUANTITYORDERED':
                        features[col] = row['Quantity Ordered']
                    elif col == 'PRICEEACH':
                        features[col] = row['Price Each']
                    elif col in row.index:
                        features[col] = row[col]
                    else:
                        # Valors per defecte per columnes no trobades
                        features[col] = 0
            
            # Crear DataFrame amb les features
            try:
                X = pd.DataFrame([features])[feature_columns]
            except KeyError as e:
                # Si falten columnes, omplir amb 0
                X = pd.DataFrame([features])
                for col in feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_columns]
            
            # Predir segons el tipus de model
            if model_type == 'sklearn':
                pred = model.predict(X)[0]
            else:
                # Keras - normalitzar primer
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled, verbose=0)[0][0]
            
            # Obtenir categoria
            category = 'N/A'
            if 'catégorie' in row.index:
                category = row['catégorie']
            elif 'Category' in row.index:
                category = row['Category']
            
            predictions.append({
                'product': row['Product'],
                'category': category,
                'predicted_turnover': max(0, float(pred)),
                'avg_price': float(row['Price Each']),
                'avg_quantity': float(row['Quantity Ordered'])
            })
        
        # Calcular totals
        total_turnover = sum(p['predicted_turnover'] for p in predictions)
        
        return jsonify({
            'success': True,
            'date': date_str,
            'dataset': dataset,
            'predictions': predictions,
            'total_turnover': total_turnover,
            'num_products': len(predictions)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/products/<dataset>')
def get_products(dataset):
    """Retorna la llista de productes disponibles"""
    try:
        components = get_model(dataset)
        if not components.get('loaded'):
            return jsonify({'success': False, 'products': []})
        
        products = components['product_stats']['Product'].tolist()
        return jsonify({
            'success': True,
            'products': ['all'] + sorted(products)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model-info/<dataset>')
def model_info(dataset):
    """Retorna informació del model"""
    try:
        components = get_model(dataset)
        if not components.get('loaded'):
            return jsonify({'success': False, 'loaded': False})
        
        mappings = components['mappings']
        return jsonify({
            'success': True,
            'loaded': True,
            'training_date': mappings.get('training_date', 'N/A'),
            'metrics': mappings.get('model_metrics', {}),
            'num_products': len(components['product_stats']),
            'categories': list(mappings.get('category_mapping', {}).keys())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Iniciant servidor de predicció de vendes...")
    print("=" * 60)
    
    # Verificar models disponibles
    for dataset in ['sales_data1', 'sales_data2']:
        components = get_model(dataset)
        if components.get('loaded'):
            print(f"✅ {dataset}: Model carregat correctament")
        else:
            print(f"❌ {dataset}: {components.get('error', 'No disponible')}")
    
    print("=" * 60)
    print("📍 Obre el navegador a: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, port=5000)
