# Retail AI Insights: Inteligencia de Clientes con ML Híbrido

##  Resumen del Proyecto

Este MVP fue desarrollado durante un **Reto Crítico de 6 horas** en un Bootcamp Empresarial. El objetivo fue transformar datos transaccionales crudos (50,000 registros) en una herramienta estratégica de toma de decisiones para el sector Retail. Implementamos un **motor de Machine Learning Híbrido** que permite segmentar clientes automáticamente y entender las reglas de negocio detrás de cada grupo.

##  Metodología y Arquitectura

El proyecto sigue una arquitectura limpia, separando la lógica de datos de la interfaz de usuario para garantizar escalabilidad y mantenimiento:

1. **Ingeniería RFM**: Procesamiento de datos para calcular Recencia, Frecuencia y Valor Monetario por cliente.
2. **Aprendizaje No Supervisado (K-Means)**: Descubrimiento de 4 clústeres naturales de comportamiento.
3. **IA Explicable (Decision Tree)**: Modelo supervisado para extraer reglas lógicas y validar la segmentación con una división de datos 70/30.
4. **Optimización de Rendimiento**: Ingesta de datos vía CSV con almacenamiento en caché (`@st.cache_data`) para respuestas en milisegundos.

## Resultados de Impacto (KPIs)

- **Exactitud del Modelo (Accuracy)**: 90.1% en la clasificación de clientes nuevos.
- **Alcance del Análisis**: 5,708 clientes únicos identificados.
- **Ventas Totales Analizadas**: $7,492,725.
- **Ticket Promedio Global**: $1,312.67.
- **Segmentación Crítica**: Identificación de 1,017 clientes VIP que sostienen la rentabilidad del negocio.

##  Tecnologías Utilizadas

- **Lenguaje**: Python 3.9+
- **Análisis de Datos**: Pandas, Numpy
- **Machine Learning**: Scikit-Learn (KMeans, DecisionTreeClassifier)
- **Visualización**: Plotly Express, Matplotlib
- **Interfaz**: Streamlit (Layout basado en Tabs corporativos)

##  Estructura del Repositorio

```
IA-BOOTCAMP/
├── .venv/               # Entorno virtual
├── app.py               # Interfaz de usuario y Dashboard
├── logic.py             # Motor de Machine Learning y RFM
├── Data_clean_model.csv # Dataset optimizado de 50k registros
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación estratégica
```

## ⏱️ Ejecución del Sprint (Scrum)

- **11:00 AM**: Ingesta de datos y limpieza de outliers.
- **01:00 PM**: Desarrollo del motor RFM y Clustering.
- **03:00 PM**: Entrenamiento del modelo supervisado y validación de métricas.
- **04:30 PM**: Pulido de UI/UX y optimización de carga.

## 🏁 Cómo Ejecutar el Proyecto

1. **Clona el repositorio**:
   ```bash
   git clone <url-del-repositorio>
   cd IA-BOOTCAMP
   ```

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecuta la aplicación**:
   ```bash
   streamlit run app.py
   ```

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de crear un pull request.

##  Licencia

Este proyecto fue desarrollado con fines educativos durante un bootcamp empresarial.

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub
