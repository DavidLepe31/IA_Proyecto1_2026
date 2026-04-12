"""
╔══════════════════════════════════════════════════════════════╗
║   Motor de Inferencia — Naïve Bayes Multinomial              ║
║   Clasificación de Solicitudes a Mesa de Ayuda               ║
║   Universidad Rafael Landívar · IA 2026                      ║
║                                                              ║
║   Dataset : Bitext Customer Support LLM Chatbot Dataset      ║
║   URL     : huggingface.co/datasets/bitext/                  ║
║             Bitext-customer-support-llm-chatbot-training-    ║
║             dataset                                          ║
║   Registros: 26,872  ·  Categorías: 11                       ║
║                                                              ║
║   Implementado 100% manualmente — sin scikit-learn,          ║
║   TensorFlow, PyTorch, Keras ni equivalentes.                ║
║   NLTK permitido solo para tokenización y stopwords.         ║
╚══════════════════════════════════════════════════════════════╝
"""

import re
import math
import json
import pickle
import random
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
#  NLTK — solo tokenización y stopwords (según normas del curso)
# ══════════════════════════════════════════════════════════════
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus   import stopwords
    from nltk.stem     import PorterStemmer

    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    STOP_WORDS     = set(stopwords.words("english"))
    stemmer        = PorterStemmer()
    NLTK_AVAILABLE = True

except Exception:
    # Fallback si NLTK no está instalado
    NLTK_AVAILABLE = False
    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","you","your","yours","he","him",
        "his","she","her","it","its","they","them","their","what","which","who",
        "this","that","these","those","am","is","are","was","were","be","been",
        "being","have","has","had","do","does","did","will","would","shall",
        "should","may","might","must","can","could","a","an","the","and","but",
        "or","nor","for","so","yet","at","by","in","of","on","to","up","as",
        "into","with","about","above","after","before","between","out","over",
        "then","when","where","why","how","all","both","each","few","more",
        "most","other","some","such","no","not","only","own","same","than",
        "too","very","just","because","through","during",
    }
    stemmer = None


# ══════════════════════════════════════════════════════════════
#  CATEGORÍAS DEL DATASET BITEXT (11 clases)
# ══════════════════════════════════════════════════════════════
CATEGORIES = [
    "ORDER",
    "BILLING",
    "SHIPPING",
    "REFUND",
    "ACCOUNT",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "PAYMENT",
    "CANCELLATION_REQUEST",
    "TECHNICAL_SUPPORT",
]

# Mapeo de etiquetas raw del CSV Bitext → nuestras 11 categorías
BITEXT_LABEL_MAP = {
    "place_order":              "ORDER",
    "track_order":              "ORDER",
    "change_order":             "ORDER",
    "check_invoices":           "BILLING",
    "get_invoice":              "BILLING",
    "check_payment_methods":    "PAYMENT",
    "payment_issue":            "PAYMENT",
    "check_refund_policy":      "REFUND",
    "get_refund":               "REFUND",
    "track_refund":             "REFUND",
    "set_up_shipping_address":  "SHIPPING",
    "delivery_options":         "DELIVERY",
    "delivery_period":          "DELIVERY",
    "create_account":           "ACCOUNT",
    "delete_account":           "ACCOUNT",
    "edit_account":             "ACCOUNT",
    "recover_password":         "ACCOUNT",
    "registration_problems":    "ACCOUNT",
    "switch_account":           "ACCOUNT",
    "newsletter_subscription":  "ACCOUNT",
    "contact_customer_service": "CONTACT",
    "contact_human_agent":      "CONTACT",
    "complaint":                "FEEDBACK",
    "review":                   "FEEDBACK",
    "cancel_order":             "CANCELLATION_REQUEST",
}


# ══════════════════════════════════════════════════════════════
#  1. PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════

# Regex para eliminar placeholders del tipo {{Order Number}}, {{Name}}
# Son específicos del dataset Bitext y no aportan información léxica.
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")


def preprocess(text):
    """
    Pipeline de preprocesamiento completo (6 pasos):

    Paso 1 — Eliminar placeholders {{...}}
              Caractéristicos del dataset Bitext.
              Ej: 'Your order {{Order Number}} has shipped'
                  → 'Your order   has shipped'

    Paso 2 — Minúsculas
              Normalización de capitalización.

    Paso 3 — Eliminar caracteres no alfabéticos
              Se conservan solo letras a-z y espacios.

    Paso 4 — Tokenización (NLTK word_tokenize)
              Divide el texto en tokens individuales.

    Paso 5 — Eliminar stopwords y tokens cortos (len <= 2)
              Las stopwords son palabras funcionales sin
              valor discriminativo (the, a, is, in, ...).

    Paso 6 — Stemming con Porter Stemmer
              Reduce cada token a su raíz morfológica.
              Ej: 'running' → 'run', 'payments' → 'payment'

    Retorna: lista de tokens limpios y normalizados.
    """
    # Paso 1
    text = PLACEHOLDER_RE.sub(" ", text)
    # Paso 2
    text = text.lower()
    # Paso 3
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+",      " ", text).strip()
    # Paso 4
    tokens = word_tokenize(text) if NLTK_AVAILABLE else text.split()
    # Paso 5
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    # Paso 6
    if NLTK_AVAILABLE and stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


# ══════════════════════════════════════════════════════════════
#  2. BAG OF WORDS
# ══════════════════════════════════════════════════════════════

def build_vocabulary(corpus):
    """
    Construye el vocabulario del corpus de entrenamiento.

    Recorre todos los documentos tokenizados y asigna un índice
    entero único a cada token nuevo encontrado.

    Parámetro:
        corpus : lista de listas de tokens (corpus tokenizado)

    Retorna:
        dict   palabra (str) → índice (int)
    """
    vocab = {}
    idx   = 0
    for tokens in corpus:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


# ══════════════════════════════════════════════════════════════
#  3. NAÏVE BAYES MULTINOMIAL — IMPLEMENTACIÓN MANUAL
# ══════════════════════════════════════════════════════════════

class NaiveBayesClassifier:
    """
    Clasificador Naïve Bayes Multinomial implementado desde cero.

    Técnicas implementadas:
    ──────────────────────────────────────────────────────────
    Bag of Words
        Cada documento se representa como vector de frecuencias
        de tokens sobre el vocabulario construido del corpus.

    Laplace Smoothing (α = 1 por defecto)
        P(w|c) = (count(w,c) + α) / (N_c + α · |V|)

        Evita probabilidad cero para palabras no vistas en
        entrenamiento. Sin Laplace, una sola palabra ausente
        anularía toda la probabilidad del documento.

    Suma de Logaritmos
        score(c|d) = log P(c) + Σᵢ log P(wᵢ|c)

        Reemplaza el producto de probabilidades pequeñas por
        una suma de logaritmos. Evita underflow numérico al
        trabajar con vocabularios de miles de palabras donde
        P(w₁|c) · P(w₂|c) · ... puede volverse 0.0 en float64.
    """

    def __init__(self, alpha=1.0):
        self.alpha               = alpha  # coeficiente Laplace
        self.vocab               = {}     # palabra → índice
        self.class_log_prior     = {}     # log P(c)
        self.word_log_likelihood = {}     # {clase: {idx: log P(w|c)}}
        self.classes             = []     # lista de clases únicas
        self.vocab_size          = 0      # tamaño del vocabulario |V|

    # ── Entrenamiento ─────────────────────────────────────────
    def train(self, X_tokens, y):
        """
        Entrena el clasificador con el corpus tokenizado.

        Calcula:
            1. Vocabulario con build_vocabulary()
            2. log P(c)   para cada clase c
            3. log P(w|c) para cada palabra w en el vocabulario

        Parámetros:
            X_tokens : list[list[str]] — documentos tokenizados
            y        : list[str]       — etiquetas de clase
        """
        self.classes    = sorted(set(y))
        n_docs          = len(y)
        self.vocab      = build_vocabulary(X_tokens)
        self.vocab_size = len(self.vocab)

        # Agrupar documentos por clase
        class_docs = {c: [] for c in self.classes}
        for tokens, label in zip(X_tokens, y):
            class_docs[label].append(tokens)

        # ── log P(c) — probabilidad a priori de cada clase ────
        # Estimación de máxima verosimilitud:
        #     P(c) = |documentos de clase c| / |total documentos|
        for c in self.classes:
            self.class_log_prior[c] = math.log(len(class_docs[c]) / n_docs)

        # ── log P(w|c) — verosimilitud con Laplace Smoothing ──
        # Para cada clase c y cada palabra w del vocabulario:
        #   P(w|c) = (count(w,c) + α) / (Σcount(·,c) + α·|V|)
        self.word_log_likelihood = {}
        for c in self.classes:
            word_counts = defaultdict(int)
            total_words = 0
            for tokens in class_docs[c]:
                for t in tokens:
                    if t in self.vocab:
                        word_counts[self.vocab[t]] += 1
                        total_words += 1

            denom = total_words + self.alpha * self.vocab_size
            ll    = {}
            for word, idx in self.vocab.items():
                count   = word_counts.get(idx, 0)
                ll[idx] = math.log((count + self.alpha) / denom)
            # Probabilidad para palabras fuera de vocabulario (OOV)
            ll["__oov__"] = math.log(self.alpha / denom)
            self.word_log_likelihood[c] = ll

    # ── Inferencia ────────────────────────────────────────────
    def predict_proba(self, tokens):
        """
        Calcula el log-score de cada clase para un documento.

        Fórmula (Bayes Naive con log):
            log P(c|d) ∝ log P(c) + Σᵢ log P(wᵢ|c)

        La clase MAP (maximum a posteriori) es la de mayor score.
        El uso de suma de logaritmos evita underflow numérico.

        Retorna:
            dict  clase → log-score  (float, negativo)
        """
        scores = {}
        for c in self.classes:
            score = self.class_log_prior[c]
            ll    = self.word_log_likelihood[c]
            oov   = ll["__oov__"]
            for t in tokens:
                if t in self.vocab:
                    score += ll.get(self.vocab[t], oov)
                else:
                    score += oov
            scores[c] = score
        return scores

    def predict(self, tokens):
        """Retorna la clase con mayor log-score (clase MAP)."""
        scores = self.predict_proba(tokens)
        return max(scores, key=scores.get)

    def predict_text(self, text):
        """
        Pipeline completo texto crudo → (categoría, probabilidades%).

        1. Aplica preprocess() al texto de entrada.
        2. Calcula log-scores con predict_proba().
        3. Convierte log-scores a probabilidades via softmax:
               exp_s[c] = exp(score[c] - max_score)   (estabilidad numérica)
               P(c|d)   = exp_s[c] / Σ exp_s

        Retorna:
            tuple (str, dict)
                str  — categoría predicha
                dict — clase → probabilidad (%)
        """
        tokens     = preprocess(text)
        log_scores = self.predict_proba(tokens)

        max_s  = max(log_scores.values())
        exp_s  = {c: math.exp(s - max_s) for c, s in log_scores.items()}
        total  = sum(exp_s.values())
        probs  = {c: round(v / total * 100, 2) for c, v in exp_s.items()}

        predicted = max(log_scores, key=log_scores.get)
        return predicted, probs


# ══════════════════════════════════════════════════════════════
#  4. K-FOLDS CROSS VALIDATION — IMPLEMENTACIÓN MANUAL
# ══════════════════════════════════════════════════════════════

def k_folds_split(n, k=5, seed=42):
    """
    Divide n índices en k folds de tamaño aproximadamente igual.

    Algoritmo:
        1. Generar indices = [0, 1, ..., n-1]
        2. Mezclar aleatoriamente (seed fija → reproducibilidad)
        3. Dividir en k bloques contiguos de tamaño floor(n/k)
           El último bloque absorbe el residuo n % k
        4. Para cada fold i:
               val_indices   = bloque i
               train_indices = todos los otros bloques

    Parámetros:
        n    : número total de muestras
        k    : número de folds (mínimo 5 según requisitos)
        seed : semilla aleatoria para reproducibilidad

    Retorna:
        list de k tuplas (train_indices, val_indices)
    """
    rng       = random.Random(seed)
    indices   = list(range(n))
    rng.shuffle(indices)

    fold_size = n // k
    folds     = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else n
        val   = indices[start:end]
        train = indices[:start] + indices[end:]
        folds.append((train, val))
    return folds


# ══════════════════════════════════════════════════════════════
#  5. MÉTRICAS DE EVALUACIÓN
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, classes):
    """
    Calcula el conjunto completo de métricas de evaluación
    para un clasificador multiclase.

    ──────────────────────────────────────────────────────────
    Matriz de Confusión
        M[i][j] = nº de veces que la clase real i
                  fue predicha como clase j
        La diagonal principal M[c][c] son los aciertos (TP).

    Para cada clase c:
        TP  = M[c][c]
        FP  = Σ_{j≠c} M[j][c]   (otras clases predichas como c)
        FN  = Σ_{j≠c} M[c][j]   (c predicha como otra clase)

        Precisión  = TP / (TP + FP)
            "De todo lo que predije como c, ¿qué fracción era c?"

        Recall     = TP / (TP + FN)
            "De todo lo que era c, ¿qué fracción predije correctamente?"

        F1-Score   = 2 · P · R / (P + R)
            Media armónica de Precisión y Recall.

    Accuracy global  = Σ_c TP_c  /  total_muestras
        "Fracción de predicciones correctas sobre todas las clases."

    Macro F1  = promedio(F1_c)  para todas las clases c
        Trata a todas las clases por igual sin importar su frecuencia.
    ──────────────────────────────────────────────────────────

    Retorna dict con:
        confusion_matrix  — dict bidimensional clase×clase
        per_class         — dict clase → {precision, recall, f1, tp, fp, fn}
        accuracy          — float
        macro_f1          — float
    """
    # Construir Matriz de Confusión
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm.get(t, {}):
            cm[t][p] += 1

    # Métricas por clase
    per_class = {}
    for c in classes:
        tp   = cm[c][c]
        fp   = sum(cm[o][c] for o in classes if o != c)
        fn   = sum(cm[c][o] for o in classes if o != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    correct  = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes)

    return {
        "confusion_matrix": cm,
        "per_class":        per_class,
        "accuracy":         round(accuracy, 4),
        "macro_f1":         round(macro_f1, 4),
    }


# ══════════════════════════════════════════════════════════════
#  6. PERSISTENCIA DEL MODELO
# ══════════════════════════════════════════════════════════════

def save_model(model, path="model.pkl"):
    """
    Serializa el modelo entrenado a disco usando pickle.

    El archivo .pkl contiene el objeto NaiveBayesClassifier
    completo: vocabulario, probabilidades a priori, verosimilitudes
    por clase y todos los metadatos necesarios para inferencia.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Modelo guardado  → '{path}'")


def load_model(path="model.pkl"):
    """
    Carga y deserializa un modelo previamente guardado.
    Retorna un objeto NaiveBayesClassifier listo para predecir.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[✓] Modelo cargado   ← '{path}'")
    return model


# ══════════════════════════════════════════════════════════════
#  7. CARGA DEL DATASET BITEXT
# ══════════════════════════════════════════════════════════════

def load_bitext_dataset(path):
    """
    Carga el dataset Bitext desde CSV local.

    ┌─────────────────────────────────────────────────────────┐
    │  INSTRUCCIONES PARA DESCARGAR EL DATASET (1 sola vez):  │
    │                                                         │
    │  pip install datasets                                   │
    │                                                         │
    │  python -c "                                            │
    │  from datasets import load_dataset                      │
    │  ds = load_dataset(                                     │
    │    'bitext/Bitext-customer-support-llm-chatbot-         │
    │     training-dataset'                                   │
    │  )                                                      │
    │  ds['train'].to_csv('bitext_dataset.csv', index=False)  │
    │  print('Dataset listo: bitext_dataset.csv')             │
    │  "                                                      │
    └─────────────────────────────────────────────────────────┘

    Columnas usadas del CSV:
        'instruction' → texto de la solicitud del cliente
        'category'    → etiqueta raw (se mapea con BITEXT_LABEL_MAP)

    Nota: los placeholders {{Order Number}} se eliminan en preprocess().
    """
    import csv
    X, y = [], []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text    = row.get("instruction", "").strip()
                raw_cat = row.get("category",    "").strip()
                mapped  = BITEXT_LABEL_MAP.get(raw_cat)
                if not mapped:
                    up = raw_cat.upper()
                    mapped = up if up in CATEGORIES else None
                if text and mapped:
                    X.append(text)
                    y.append(mapped)

        print(f"[✓] Dataset Bitext cargado: {len(X)} registros")
        dist = defaultdict(int)
        for lbl in y:
            dist[lbl] += 1
        print("[i] Distribución:")
        mx = max(dist.values())
        for c in sorted(dist):
            bar = "▓" * (dist[c] * 25 // mx)
            print(f"    {c:25s}  {dist[c]:5d}  {bar}")
        return X, y

    except FileNotFoundError:
        print(f"[!] Archivo '{path}' no encontrado.")
        print("[!] Sigue las instrucciones del README para descargar el dataset.")
        print("[i] Usando dataset sintético de demostración...\n")
        return generate_synthetic_dataset()


def generate_synthetic_dataset(n=3300):
    """
    Dataset sintético con las 11 categorías Bitext.

    Propósito: ejecutar y demostrar el sistema cuando el CSV
    real no está disponible. Para el proyecto final se usa
    el dataset Bitext con 26,872 registros reales.
    """
    templates = {
        "ORDER": [
            "I want to place a new order for the item on your site",
            "How do I order multiple items at the same time here",
            "I need to change the quantity on my existing order",
            "Can I modify my order before it ships out today",
            "I placed an order but did not receive confirmation email",
            "Where can I check the status of my recent order",
            "I want to reorder the same items from last month",
            "My order confirmation shows the wrong item was selected",
            "Can I add more products to an order I just submitted",
            "The order form keeps showing an error when submitted",
        ],
        "BILLING": [
            "I need a copy of my invoice for the last payment made",
            "Can you send me the billing statement for my account",
            "My invoice shows the wrong amount for services rendered",
            "I need an itemized bill for my company expense report",
            "Can I get a receipt for the payment I made recently",
            "The billing address on my account needs to be updated",
            "I did not receive my monthly invoice via email",
            "My billing cycle date does not match what was agreed",
            "Please resend the invoice for my order from last week",
            "I need a duplicate invoice for my tax records",
        ],
        "SHIPPING": [
            "I need to change my shipping address for the order",
            "What shipping options do you offer for my location",
            "Can I upgrade to express shipping on my current order",
            "My package was delivered to the wrong address",
            "Do you ship internationally to my country",
            "How do I set up a new default shipping address",
            "The shipping cost seems higher than I expected",
            "I want to add a delivery instruction for the courier",
            "Is free shipping available above a certain amount",
            "Can I ship to a PO box or does it need a street address",
        ],
        "REFUND": [
            "I would like to request a refund for my recent purchase",
            "How long does the refund process usually take",
            "I returned the item but have not received my refund yet",
            "Can you check the status of my pending refund request",
            "I was charged but never received the product I ordered",
            "The item I received was damaged, I need a full refund",
            "My refund was processed for the wrong amount",
            "I want to return the product because it does not fit",
            "The refund was supposed to appear by now but nothing",
            "Can I get store credit instead of a card refund",
        ],
        "ACCOUNT": [
            "I cannot log in to my account and need help accessing it",
            "I want to delete my account and all my personal data",
            "How do I change the email address linked to my account",
            "I forgot my password and need to reset it right now",
            "I need to create a new account for my business use",
            "My account has been locked and I cannot access it",
            "I want to update my personal information in the system",
            "How do I enable two-factor authentication on my account",
            "I accidentally created two accounts and want to merge them",
            "My account shows incorrect information that needs fixing",
        ],
        "CONTACT": [
            "I need to speak with a human agent as soon as possible",
            "Can you give me the phone number for customer service",
            "I want to contact the support team about my problem",
            "How do I reach a real person at your company right now",
            "Please connect me with a live agent who can help me",
            "What is the best way to contact support department today",
            "I need direct contact information for the billing team",
            "Is there a way to chat live with a customer service agent",
            "I have been trying to reach someone for three days",
            "What are your support hours so I can call at right time",
        ],
        "DELIVERY": [
            "When will my package be delivered to my home address",
            "My delivery is late and I need to know what happened",
            "Can I schedule a specific delivery time for my package",
            "The delivery driver left my package in the wrong place",
            "My estimated delivery date has passed with no package",
            "I need to reschedule my delivery to a different date",
            "How do I track the live delivery progress of my order",
            "My neighbor received my package by mistake what do I do",
            "The tracking says delivered but I never got the package",
            "Can I pick up my package at the warehouse instead",
        ],
        "FEEDBACK": [
            "I want to leave a positive review for the excellent service",
            "I have a serious complaint about the product quality received",
            "The customer service I received was very disappointing",
            "I am very satisfied with my experience and want to share",
            "I have suggestions that could improve your service overall",
            "The product did not meet the expectations from your website",
            "Your support team was incredibly helpful and responsive",
            "I want to report a bad experience with a staff member",
            "Everything worked perfectly and I am very happy with order",
            "I think the checkout process could be simplified",
        ],
        "PAYMENT": [
            "My payment was declined even though my card is valid",
            "I want to add a new credit card as a payment method",
            "Can I pay using a bank transfer instead of credit card",
            "I was charged twice for the same transaction this month",
            "My payment is not going through and I do not know why",
            "I want to set up automatic recurring payments for my plan",
            "I need to update my expired credit card details on file",
            "Does your platform accept PayPal as a payment option",
            "I want to pay with a gift card, is that possible",
            "The checkout keeps failing at the payment step with error",
        ],
        "CANCELLATION_REQUEST": [
            "I want to cancel my subscription effective immediately",
            "How do I cancel my order before it has been shipped",
            "Please cancel my account and stop all future charges",
            "I need to cancel my service before the next billing cycle",
            "Can I cancel the automatic renewal of my subscription",
            "I want to unsubscribe and permanently close my account",
            "I changed my mind and want to cancel the order I placed",
            "Please confirm that my cancellation request was received",
            "I want to cancel but keep my data for possible return",
            "How long does it take to process a cancellation request",
        ],
        "TECHNICAL_SUPPORT": [
            "The application keeps crashing every time I try to open it",
            "I cannot access my account because of a technical error",
            "The website is not loading correctly on my browser",
            "I am getting an error message when I try to check out",
            "My account settings are not saving when I click save",
            "The mobile app stopped working after the latest update",
            "The search feature is not returning any results at all",
            "I cannot upload my profile picture due to a technical error",
            "The page keeps timing out before I complete my purchase",
            "Notifications are not working even though they are enabled",
        ],
    }

    rng       = random.Random(42)
    X, y      = [], []
    per_class = n // len(CATEGORIES)

    for label in CATEGORIES:
        texts = templates.get(label, [f"I need help with {label.lower()} issue"])
        for _ in range(per_class):
            base  = rng.choice(texts)
            words = base.split()
            if len(words) > 5 and rng.random() < 0.35:
                cut  = rng.randint(4, len(words))
                base = " ".join(words[:cut])
            X.append(base)
            y.append(label)

    combined = list(zip(X, y))
    rng.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# ══════════════════════════════════════════════════════════════
#  8. PIPELINE COMPLETO DE ENTRENAMIENTO Y EVALUACIÓN
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(dataset_path=None, k=5, save_path="model.pkl"):
    """
    Ejecuta el pipeline completo de principio a fin:

    1. Carga del dataset (Bitext CSV o sintético)
    2. Preprocesamiento de todos los textos
    3. K-Folds Cross Validation manual (K = 5 mínimo)
    4. Métricas: Precisión, Recall, F1, Accuracy, Macro F1
    5. Análisis de varianza entre folds (media y σ)
    6. Matriz de Confusión
    7. Entrenamiento del modelo final (100% de los datos)
    8. Guardado: model.pkl  y  metrics.json

    Retorna: (modelo_final, resumen_métricas)
    """
    print("=" * 66)
    print("  NAÏVE BAYES MULTINOMIAL — CLASIFICACIÓN DE TICKETS")
    print("  Dataset : Bitext Customer Support LLM Chatbot Dataset")
    print("  Clases  : 11 categorías")
    print("=" * 66)

    # 1 — Carga
    if dataset_path:
        X_raw, y = load_bitext_dataset(dataset_path)
    else:
        print("[i] Dataset no especificado → usando sintético\n")
        X_raw, y = generate_synthetic_dataset()

    print(f"\n[i] Registros totales  : {len(X_raw)}")
    dist = defaultdict(int)
    for lbl in y:
        dist[lbl] += 1
    mx = max(dist.values())
    print("[i] Distribución:")
    for c in sorted(dist):
        bar = "▓" * (dist[c] * 28 // mx)
        print(f"    {c:25s}  {dist[c]:5d}  {bar}")

    # 2 — Preprocesamiento
    print("\n[→] Preprocesando textos...")
    X_tokens = [preprocess(t) for t in X_raw]
    print(f"[✓] Listo.  Ejemplo: '{X_raw[0][:55]}...'")
    print(f"             tokens: {X_tokens[0][:7]}")

    active_classes = sorted(set(y))

    # 3 — K-Folds
    print(f"\n[→] K-Folds Cross Validation  (K={k})...")
    folds        = k_folds_split(len(X_tokens), k=k)
    fold_results = []
    all_true, all_pred = [], []

    for fi, (tr_idx, vl_idx) in enumerate(folds):
        X_tr = [X_tokens[i] for i in tr_idx]
        y_tr = [y[i]        for i in tr_idx]
        X_vl = [X_tokens[i] for i in vl_idx]
        y_vl = [y[i]        for i in vl_idx]

        clf  = NaiveBayesClassifier(alpha=1.0)
        clf.train(X_tr, y_tr)
        y_pd = [clf.predict(t) for t in X_vl]

        m = compute_metrics(y_vl, y_pd, active_classes)
        fold_results.append(m)
        all_true.extend(y_vl)
        all_pred.extend(y_pd)
        print(f"  Fold {fi+1}/{k}  →  "
              f"Accuracy: {m['accuracy']:.4f}  |  Macro F1: {m['macro_f1']:.4f}")

    # 4 — Análisis de varianza
    avg_acc = sum(r["accuracy"]  for r in fold_results) / k
    avg_mf1 = sum(r["macro_f1"] for r in fold_results) / k
    std_acc = math.sqrt(sum((r["accuracy"] - avg_acc)**2  for r in fold_results) / k)
    std_mf1 = math.sqrt(sum((r["macro_f1"] - avg_mf1)**2 for r in fold_results) / k)

    print("\n" + "─" * 66)
    print("  RESUMEN — ANÁLISIS DE VARIANZA ENTRE FOLDS")
    print("─" * 66)
    print(f"  Accuracy  promedio : {avg_acc:.4f}   (σ = {std_acc:.4f})")
    print(f"  Macro F1  promedio : {avg_mf1:.4f}   (σ = {std_mf1:.4f})")

    # 5 — Métricas por clase
    gm = compute_metrics(all_true, all_pred, active_classes)

    print("\n" + "─" * 66)
    print("  MÉTRICAS POR CLASE")
    print("─" * 66)
    print(f"  {'Clase':25s}  {'Prec':>7} {'Recall':>7} {'F1':>7}  {'TP':>6} {'FP':>5} {'FN':>5}")
    print(f"  {'─'*25}  {'─'*7} {'─'*7} {'─'*7}  {'─'*6} {'─'*5} {'─'*5}")
    for c in active_classes:
        m = gm["per_class"][c]
        print(f"  {c:25s}  {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f}  {m['tp']:>6d} {m['fp']:>5d} {m['fn']:>5d}")
    print(f"\n  Accuracy global : {gm['accuracy']:.4f}")
    print(f"  Macro F1        : {gm['macro_f1']:.4f}")

    # 6 — Matriz de Confusión
    print("\n" + "─" * 66)
    print("  MATRIZ DE CONFUSIÓN  (fila=real, col=predicho)")
    print("─" * 66)
    short  = {c: c[:7] for c in active_classes}
    header = "           " + "  ".join(f"{short[c]:>8}" for c in active_classes)
    print(header)
    cm = gm["confusion_matrix"]
    for tc in active_classes:
        row = f"  {short[tc]:8s}  " + "  ".join(
            f"{cm[tc].get(pc, 0):>8d}" for pc in active_classes
        )
        print(row)

    # 7 — Modelo final
    print("\n[→] Entrenando modelo final (100% de los datos)...")
    final_model = NaiveBayesClassifier(alpha=1.0)
    final_model.train(X_tokens, y)
    save_model(final_model, save_path)

    # 8 — Guardar métricas
    summary = {
        "dataset":         dataset_path or "synthetic",
        "k_folds":         k,
        "total_records":   len(X_raw),
        "num_categories":  len(active_classes),
        "categories":      active_classes,
        "avg_accuracy":    round(avg_acc, 4),
        "std_accuracy":    round(std_acc, 4),
        "avg_macro_f1":    round(avg_mf1, 4),
        "std_macro_f1":    round(std_mf1, 4),
        "fold_accuracies": [r["accuracy"] for r in fold_results],
        "fold_macro_f1s":  [r["macro_f1"]  for r in fold_results],
        "global_metrics":  gm,
    }
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[✓] metrics.json guardado")
    return final_model, summary


# ══════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Uso:
    #   python naive_bayes.py                    ← dataset sintético
    #   python naive_bayes.py bitext_dataset.csv ← dataset real Bitext
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    model, _     = train_and_evaluate(dataset_path=dataset_path)

    # Demo de predicciones
    print("\n" + "=" * 66)
    print("  DEMO DE PREDICCIONES")
    print("=" * 66)
    demos = [
        ("ORDER",                "I want to track my recent order, it has not arrived yet"),
        ("BILLING",              "I need an invoice for the payment I made last week"),
        ("SHIPPING",             "My package was shipped to the wrong address by mistake"),
        ("REFUND",               "I want a refund, the product I received was damaged"),
        ("ACCOUNT",              "I cannot login to my account, please help me reset password"),
        ("CONTACT",              "I need to speak with a human agent as soon as possible"),
        ("DELIVERY",             "When will my package be delivered, it is very late"),
        ("FEEDBACK",             "I have a complaint about the very poor service quality"),
        ("PAYMENT",              "My payment keeps getting declined even though card is valid"),
        ("CANCELLATION_REQUEST", "Please cancel my subscription immediately and stop charges"),
        ("TECHNICAL_SUPPORT",    "The application crashes every time I try to open it"),
    ]
    correct = 0
    print(f"  {'Esperado':25s}  {'Predicho':25s}  OK   Confianza")
    print(f"  {'─'*25}  {'─'*25}  ──   ─────────")
    for expected, text in demos:
        pred, probs = model.predict_text(text)
        ok       = "✓" if pred == expected else "✗"
        correct += pred == expected
        print(f"  {expected:25s}  {pred:25s}   {ok}   {probs.get(pred,0):.1f}%")
    print(f"\n  Accuracy demo: {correct}/{len(demos)}  ({correct/len(demos)*100:.1f}%)")
