# Sales Prediction with Random Forest

Aquest projecte utilitza un model de Random Forest per predir les vendes basant-se en dades històriques de vendes.

## Descripció

El projecte analitza un conjunt de dades de vendes (`sales_data_sample.csv`) i entrena un model de Random Forest Regressor per predir els valors de vendes. Inclou:

- Preprocessament de dades (conversió de dates, encoding de variables categòriques)
- Entrenament del model amb Random Forest (500 arbres)
- Avaluació del model amb mètriques R² i RMSE
- Visualitzacions de prediccions vs. valors reals i anàlisi de residuals

## Requisits

- Python 3.7+
- Les dependències es troben a `requirements.txt`

## Instal·lació

1. Clona el repositori:
```bash
git clone https://github.com/OriolCanillasGautier/SimpleNN_project06_cole.git
cd SimpleNN_project06_cole
```

2. Instal·la les dependències:
```bash
pip install -r requirements.txt
```

## Ús

Obre el notebook `proves.ipynb` amb Jupyter Notebook o VS Code i executa les cel·les en ordre:

1. **Cel·la 1**: Importació de llibreries
2. **Cel·la 2**: Càrrega de dades
3. **Cel·la 3**: Preprocessament de dates i filtratge de dades
4. **Cel·la 4**: Entrenament del model Random Forest
5. **Cel·la 5**: Avaluació i visualització dels resultats

## Estructura del Projecte

```
SimpleNN_project06_cole/
│
├── proves.ipynb              # Notebook principal amb l'anàlisi
├── sales_data_sample.csv     # Dataset de vendes
├── requirements.txt          # Dependències de Python
├── README.md                 # Aquest fitxer
└── LICENSE                   # Llicència del projecte
```

## Resultats

El model genera:
- **R² Score**: Mesura de la qualitat de les prediccions
- **RMSE**: Error quadràtic mitjà de les prediccions
- **Gràfics**:
  - Prediccions vs. valors reals
  - Anàlisi de residuals

## Tecnologies Utilitzades

- **NumPy**: Càlculs numèrics
- **Pandas**: Manipulació de dades
- **Matplotlib**: Visualització
- **Scikit-learn**: Model de Machine Learning (Random Forest)

## Autor

Cole - [OriolCanillasGautier](https://github.com/OriolCanillasGautier)

## Llicència

Aquest projecte està llicenciat segons els termes especificats al fitxer LICENSE.