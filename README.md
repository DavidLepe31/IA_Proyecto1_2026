# 🎫 HelpDesk AI — Clasificación de Tickets con Naïve Bayes

**Universidad Rafael Landívar · Facultad de Ingeniería · Inteligencia Artificial 2026**

---

## Descripción

Sistema de enrutamiento automático de tickets de soporte al cliente.
Clasifica solicitudes en **11 categorías** usando **Naïve Bayes Multinomial implementado desde cero**, sin ninguna librería de Machine Learning.

| Campo            | Detalle                                                               |
|------------------|-----------------------------------------------------------------------|
| Dataset          | Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face)  |
| URL Dataset      | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| Registros        | 26,872 solicitudes reales con etiquetas verificadas                   |
| Categorías       | 11 clases                                                             |
| Algoritmo        | Naïve Bayes Multinomial (implementación manual)                       |
| Evaluación       | K-Folds Cross Validation (K=5, manual)                               |
| Backend          | Python + Flask                                                        |
| Frontend         | HTML5 + CSS3 + JavaScript vanilla                                     |

---

## Categorías (11 clases)

| Categoría              | Descripción                                       |
|------------------------|---------------------------------------------------|
| ORDER                  | Pedidos: crear, modificar, rastrear               |
| BILLING                | Facturas, recibos, estados de cuenta              |
| SHIPPING               | Opciones y direcciones de envío                   |
| REFUND                 | Reembolsos y devoluciones                         |
| ACCOUNT                | Cuenta, contraseña, perfil de usuario             |
| CONTACT                | Contactar agente humano o soporte                 |
| DELIVERY               | Seguimiento y fechas de entrega                   |
| FEEDBACK               | Quejas, sugerencias y reseñas                     |
| PAYMENT                | Métodos de pago y transacciones                   |
| CANCELLATION_REQUEST   | Cancelar órdenes o suscripciones                  |
| TECHNICAL_SUPPORT      | Errores, bugs y problemas técnicos                |

---

## Estructura del Proyecto

```
helpdesk-ai/
│
├── naive_bayes.py          # Motor de inferencia completo (backend IA)
├── app.py                  # Servidor Flask (API REST)
├── requirements.txt        # Dependencias Python
│
├── model.pkl               # Modelo entrenado (generado automáticamente)
├── metrics.json            # Resultados K-Folds y métricas de evaluación
│
├── bitext_dataset.csv      # Dataset (descargar con instrucciones abajo)
│
├── templates/
│   └── index.html          # Interfaz web del sistema de tickets
│
├── static/                 # Archivos estáticos (imágenes, CSS adicional)
│
└── README.md               # Este archivo
```

---

## Instalación y Ejecución

### Paso 1 — Instalar dependencias

```bash
pip install flask nltk
```

### Paso 2 — Descargar el dataset Bitext

```bash
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
ds['train'].to_csv('bitext_dataset.csv', index=False)
print('Listo: bitext_dataset.csv')
"
```

> **Nota:** Si no puedes descargar el dataset, el sistema funciona igualmente
> con un dataset sintético representativo de las mismas 11 categorías.

### Paso 3 — Entrenar el modelo

```bash
# Con el dataset Bitext real (recomendado):
python naive_bayes.py bitext_dataset.csv

# Sin dataset (usa datos sintéticos para pruebas):
python naive_bayes.py
```

Esto genera automáticamente:
- `model.pkl`    — modelo entrenado serializado
- `metrics.json` — métricas K-Folds y evaluación completa

### Paso 4 — Iniciar el servidor web

```bash
python app.py
```

Abrir el navegador en: **http://localhost:5000**

> ⚠️ Los textos deben ingresarse en **inglés** (el dataset Bitext es en inglés).

---

## API REST

### `POST /classify` — Clasificar un ticket

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"subject": "Cannot login", "description": "I forgot my password and cannot reset it"}'
```

**Respuesta:**
```json
{
  "ticket_id":   "TKT-A3F9C1",
  "subject":     "Cannot login",
  "description": "I forgot my password and cannot reset it",
  "category":    "ACCOUNT",
  "probabilities": {
    "ACCOUNT":           84.2,
    "TECHNICAL_SUPPORT": 9.8,
    "CONTACT":           3.1,
    "ORDER":             1.4,
    ...
  },
  "timestamp": "2026-04-15 10:22:00"
}
```

### `GET /metrics` — Métricas del modelo

```bash
curl http://localhost:5000/metrics
```

### `GET /tickets` — Historial de tickets

```bash
curl http://localhost:5000/tickets
```

### `POST /retrain` — Reentrenar el modelo

```bash
# Con CSV real:
curl -X POST http://localhost:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "bitext_dataset.csv"}'

# Con datos sintéticos:
curl -X POST http://localhost:5000/retrain \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Algoritmo — Descripción Técnica

### 1. Preprocesamiento (6 pasos)

```
Texto crudo
  │
  ├─ Paso 1: Eliminar placeholders  {{Order Number}}, {{Name}}, etc.
  │          (regex: \{\{[^}]+\}\})
  │
  ├─ Paso 2: Convertir a minúsculas
  │
  ├─ Paso 3: Eliminar caracteres no alfabéticos ([^a-z\s])
  │
  ├─ Paso 4: Tokenización  (NLTK word_tokenize)
  │
  ├─ Paso 5: Eliminar stopwords  (NLTK english) + tokens cortos (len ≤ 2)
  │
  └─ Paso 6: Stemming  (Porter Stemmer)
             "running" → "run"  |  "payments" → "payment"
             │
             └─→  Lista de tokens normalizados
```

### 2. Bag of Words

Construcción del vocabulario a partir del corpus de entrenamiento.
Cada documento se representa como un vector de frecuencias de tokens.

### 3. Naïve Bayes Multinomial

**Probabilidad a priori:**
```
log P(c) = log( |docs_c| / |docs_total| )
```

**Verosimilitud con Laplace Smoothing (α = 1):**
```
log P(w|c) = log( (count(w,c) + α) / (N_c + α · |V|) )
```

**Inferencia con suma de logaritmos:**
```
log P(c|d) ∝ log P(c) + Σᵢ log P(wᵢ|c)
```

### 4. K-Folds Cross Validation (K = 5)

1. Mezclar el dataset aleatoriamente (semilla fija = 42)
2. Dividir en 5 particiones iguales
3. Para cada fold i: entrenar en los otros 4, validar en el i
4. Promediar métricas y calcular desviación estándar (varianza)

### 5. Métricas

| Métrica       | Fórmula                                  |
|---------------|------------------------------------------|
| Precisión     | TP / (TP + FP)                           |
| Recall        | TP / (TP + FN)                           |
| F1-Score      | 2 × P × R / (P + R)                     |
| Accuracy      | Σ TP_c  /  total_muestras               |
| Macro F1      | promedio(F1_c) sobre las 11 clases      |

---

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│                  USUARIO                        │
│  Ingresa: subject + description del problema    │
└──────────────────────┬──────────────────────────┘
                       │  HTTP POST /classify
                       ▼
┌─────────────────────────────────────────────────┐
│              FRONTEND  (index.html)             │
│  HTML5 + CSS3 + JavaScript vanilla              │
│  • Formulario de ticket (ID, asunto, desc.)     │
│  • Barras de probabilidad por clase             │
│  • Historial de tickets clasificados            │
└──────────────────────┬──────────────────────────┘
                       │  JSON request/response
                       ▼
┌─────────────────────────────────────────────────┐
│              BACKEND  (app.py — Flask)          │
│  REST API: /classify /metrics /tickets          │
└──────────────────────┬──────────────────────────┘
                       │  Python function call
                       ▼
┌─────────────────────────────────────────────────┐
│         MOTOR DE INFERENCIA (naive_bayes.py)    │
│                                                 │
│  preprocess(text)                               │
│      └→ NaiveBayesClassifier.predict_text()     │
│             └→ predict_proba()                  │
│                  └→ Σ log P(c) + Σ log P(w|c)  │
│                                                 │
│  Cargado desde: model.pkl                       │
└──────────────────────┬──────────────────────────┘
                       │  load / save
                       ▼
┌─────────────────────────────────────────────────┐
│              PERSISTENCIA                       │
│  model.pkl      — modelo serializado (pickle)   │
│  metrics.json   — métricas K-Folds y evaluación │
└─────────────────────────────────────────────────┘
```

---

## Tecnologías

| Componente    | Tecnología                                    |
|---------------|-----------------------------------------------|
| Backend IA    | Python 3.10 — implementación 100% manual      |
| Tokenización  | NLTK word_tokenize + stopwords (solo esto)    |
| Stemming      | NLTK Porter Stemmer                           |
| Servidor web  | Flask 2.3+                                    |
| Frontend      | HTML5 + CSS3 + JavaScript ES6 vanilla         |
| Persistencia  | pickle (model.pkl) + JSON (metrics.json)      |
| Dataset       | Bitext (Hugging Face, 26,872 registros, 11 clases) |

---

## Restricciones Técnicas Aplicadas

- ❌ **No se usa** scikit-learn, TensorFlow, Keras, PyTorch, Hugging Face (para clasificación)
- ✅ **NLTK** se usa **exclusivamente** para tokenización y stopwords
- ✅ Naïve Bayes implementado manualmente desde cero
- ✅ K-Folds implementado manualmente (sin librerías de validación)
- ✅ Todas las métricas calculadas manualmente (sin sklearn.metrics)

---

*Proyecto — Inteligencia Artificial · Universidad Rafael Landívar · Primer Semestre 2026*
