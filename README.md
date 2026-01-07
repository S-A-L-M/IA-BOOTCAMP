# Retail AI Insights: Customer Intelligence with Hybrid ML

## Project Overview

This MVP was developed during a **6-hour Critical Challenge** in a Corporate Bootcamp. The goal was to transform raw transactional data (50,000 records) into a strategic decision-making tool for the Retail sector. We implemented a **Hybrid Machine Learning engine** that automatically segments customers and uncovers the business rules behind each group.

## Methodology and Architecture

The project follows a clean architecture, separating data logic from the user interface to ensure scalability and maintainability:

1. **RFM Engineering**: Data processing to calculate Recency, Frequency, and Monetary value per customer.  
2. **Unsupervised Learning (K-Means)**: Discovery of 4 natural behavior clusters.  
3. **Explainable AI (Decision Tree)**: Supervised model to extract logical rules and validate segmentation with a 70/30 data split.  
4. **Performance Optimization**: Data ingestion via CSV with caching (`@st.cache_data`) for millisecond response times.

## Impact Results (KPIs)

- **Model Accuracy**: 90.1% in classifying new customers.  
- **Analysis Scope**: 5,708 unique customers identified.  
- **Total Sales Analyzed**: $7,492,725.  
- **Global Average Ticket**: $1,312.67.  
- **Critical Segmentation**: Identification of 1,017 VIP customers sustaining business profitability.

## Technologies Used

- **Language**: Python 3.9+  
- **Data Analysis**: Pandas, Numpy  
- **Machine Learning**: Scikit-Learn (KMeans, DecisionTreeClassifier)  
- **Visualization**: Plotly Express, Matplotlib  
- **Interface**: Streamlit (corporate tab-based layout)

## Repository Structure

```
IA-BOOTCAMP/
├── .venv/               # Entorno virtual
├── app.py               # Interfaz de usuario y Dashboard
├── logic.py             # Motor de Machine Learning y RFM
├── Data_clean_model.csv # Dataset optimizado de 50k registros
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación estratégica
```

## Sprint Execution (Scrum)

- **11:00 AM**: Data ingestion and outlier cleaning.  
- **01:00 PM**: Development of RFM engine and clustering.  
- **03:00 PM**: Training of supervised model and metric validation.  
- **04:30 PM**: UI/UX refinement and load optimization.

## How to Run the Project

1. **Clone the repository**:  
```bash
   git clone <url-del-repositorio>
   cd IA-BOOTCAMP
   ```

Install dependencies:
```bash
   pip install -r requirements.txt
   ```
Run the application:
```bash
   streamlit run app.py
   ```
Contributions
Contributions are welcome. Please open an issue to discuss major changes before submitting a pull request.

License
This project was developed for educational purposes during a corporate bootcamp.

If you found this project useful, consider giving it a star on GitHub.
