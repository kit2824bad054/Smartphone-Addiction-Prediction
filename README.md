# Screen Signal — Smartphone Addiction Risk Predictor

A lightweight, production-grade, 100% client-side machine learning application that predicts smartphone addiction risk and explains the behavioral drivers behind device dependency.

The application runs entirely in the browser using pre-trained Logistic Regression parameters—no backend server, no Python runtime in production, and no third-party API keys required.

---

## Project Structure

```
smartphone-addiction-predictor/
├── data/
│   ├── cleaned_smartphone_addiction.csv   # Cleaned 5-feature dataset with target labels
│   └── teen_phone_addiction_raw.csv       # Original dataset downloaded from Kaggle
├── train_model.py                         # Python training script (scikit-learn + pandas)
├── index.html                             # "Screen Signal" web application (HTML/CSS/JS)
└── README.md                              # Project documentation and retraining guide
```

---

## Step 1 — Dataset Source & Feature Mapping

### 1.1 Dataset Source
- **Dataset:** *Teen Smartphone Usage and Addiction Impact Dataset* (Kaggle public dataset)
- **Records:** 3,000 real individual behavioral observations across 25 metrics.
- **Location:** `data/teen_phone_addiction_raw.csv` and `data/cleaned_smartphone_addiction.csv`

### 1.2 Feature Mapping
To fulfill the 5 key behavioral dimensions of smartphone addiction, original dataset fields were mapped as follows:

| Target Feature Name | Dataset Source Column | Type | Description / Transformation |
| :--- | :--- | :--- | :--- |
| `screen_time_hours` | `Daily_Usage_Hours` | Continuous (`0.0`–`14.0`) | Total daily active screen-on time in hours. |
| `unlocks_per_day` | `Phone_Checks_Per_Day` | Integer (`5`–`250`) | Number of times the phone is unlocked or checked per day. |
| `social_media_hours` | `Time_on_Social_Media` | Continuous (`0.0`–`10.0`) | Daily hours spent actively using social media platforms. |
| `night_usage_ratio` | `Screen_Time_Before_Bed` & `Daily_Usage_Hours` | Percentage (`0%`–`100%`) | Night-time usage share: `clip((Screen_Time_Before_Bed / max(Daily_Usage, 0.5)) * 100, 0, 100)`. |
| `sleep_hours` | `Sleep_Hours` | Continuous (`3.0`–`10.0`) | Nightly restorative sleep duration in hours. |

### 1.3 Target Label Creation
Grounded in digital well-being research (e.g., SAS-SV scale criteria), smartphone addiction risk is driven by elevated screen time, compulsive unlocking, heavy social media use, high late-night bedtime usage, and chronic sleep deprivation.

A standardized composite usage score is computed:
$$\text{Composite Score} = 0.30 \cdot z_{\text{screen}} + 0.20 \cdot z_{\text{unlocks}} + 0.20 \cdot z_{\text{social}} + 0.15 \cdot z_{\text{night}} - 0.15 \cdot z_{\text{sleep}}$$

This score is binned into three balanced tertiles:
- **Low Risk (Class 0):** Lowest 33.3% of composite usage (1,000 samples)
- **Moderate Risk (Class 1):** Middle 33.3% of composite usage (1,000 samples)
- **High Risk (Class 2):** Highest 33.3% of composite usage (1,000 samples)

---

## Step 2 — Model Training & Evaluation

The script `train_model.py` loads the dataset, cleans missing values, standardizes the features with `StandardScaler`, and trains both a **Multinomial Logistic Regression** model and a **Random Forest Classifier** on an 80/20 train/test stratified split.

### 2.1 Model Performance Comparison

| Model | Test Accuracy | Weighted F1 Score | Characteristics |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (Multinomial Softmax)** | **98.83%** | **0.9884** | Fully linear, interpretable log-odds, exportable to client-side JS. |
| **Random Forest (100 Trees)** | **86.67%** | **0.8675** | Non-linear ensemble comparison baseline. |

### 2.2 Classification Reports (Test Set: 600 samples)

#### Logistic Regression
```
              precision    recall  f1-score   support

         Low       1.00      0.98      0.99       200
    Moderate       0.98      0.99      0.98       200
        High       0.99      0.99      0.99       200

    accuracy                           0.99       600
   macro avg       0.99      0.99      0.99       600
weighted avg       0.99      0.99      0.99       600
```

#### Random Forest
```
              precision    recall  f1-score   support

         Low       0.90      0.89      0.89       200
    Moderate       0.79      0.82      0.80       200
        High       0.92      0.89      0.91       200

    accuracy                           0.87       600
   macro avg       0.87      0.87      0.87       600
weighted avg       0.87      0.87      0.87       600
```

### 2.3 Exported JavaScript Constants

The script formats and outputs the exact array constants for direct copy-pasting into `index.html`:

```javascript
// Feature Order:
// ['screen_time_hours', 'unlocks_per_day', 'social_media_hours', 'night_usage_ratio', 'sleep_hours']

const FEATURE_NAMES = ['screen_time_hours', 'unlocks_per_day', 'social_media_hours', 'night_usage_ratio', 'sleep_hours'];
const RISK_CLASSES = ['Low', 'Moderate', 'High'];
const SCALER_MEAN = [5.020458, 82.850417, 2.512917, 24.637704, 6.512917];
const SCALER_SCALE = [1.95197, 37.745989, 0.995414, 19.645943, 1.485679];
const LOGREG_COEF = [
  [-7.290602, -5.078497, -5.063988, -3.588382, 3.78962],  // Class 0 (Low Risk)
  [-0.086272, -0.003047, -0.06108, -0.012278, 0.047962],  // Class 1 (Moderate Risk)
  [7.376874, 5.081544, 5.125068, 3.60066, -3.837582],     // Class 2 (High Risk)
];
const LOGREG_INTERCEPT = [-1.423745, 2.957644, -1.533899];
```

#### Behavioral Coefficient Interpretation (High Risk Class):
- `screen_time_hours` ($+7.38$): The strongest driver of device addiction.
- `social_media_hours` ($+5.13$) & `unlocks_per_day` ($+5.08$): Strong compulsive habit reinforcers.
- `night_usage_ratio` ($+3.60$): Bedtime screen use significantly accelerates risk.
- `sleep_hours` ($-3.84$): Protective factor; reduced sleep increases addiction severity.

---

## Step 3 — Web Application: "Screen Signal" (`index.html`)

### 3.1 Design Aesthetics
- **Theme:** Dark theme with deep slate surfaces (`#090d12`, `#0f1620`, `#15202c`).
- **Accent Color:** Strict single accent color in **Teal** (`#14b8a6`, `#2dd4bf`, `#0f766e`).
- **Typography:** Modern clean sans-serif (`Inter`, system UI font stack) and `JetBrains Mono` for numeric metrics.
- **Interactivity:** Snappy, sub-millisecond local inference with no unnecessary animation or lagging transitions.
- **Responsiveness:** Full multi-column layout on desktop, smoothly collapsing to a single-column layout on mobile devices.

### 3.2 Client-Side Mathematical Logic
1. **Standardization:**
   $$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$
2. **Softmax Multinomial Logits:**
   $$z_c = b_c + \sum_{i=0}^4 W_{c,i} \cdot \tilde{x}_i \quad \text{for } c \in \{0, 1, 2\}$$
   $$P(c) = \frac{e^{z_c - \max(z)}}{\sum_{k} e^{z_k - \max(z)}}$$
3. **0–100 Risk Score:**
   $$\text{Risk Score} = \text{clamp}\Big(\text{round}\big(P(\text{Moderate}) \cdot 50 + P(\text{High}) \cdot 100\big),\, 0,\, 100\Big)$$
4. **Feature Contribution Calculation:**
   $$\text{Impact}_i = W_{2, i} \cdot \tilde{x}_i$$
   Positive impacts highlight specific behaviors driving risk upward, while negative impacts denote healthy, protective factors.
5. **Personalized Action Plan:**
   The app inspects the top 2 positive risk contributors and generates customized, actionable behavioral interventions in real time.

---

## Step 4 — How to Run & Retrain

### 4.1 Running the Web App Locally
Because `index.html` uses 100% client-side JavaScript, you can run it instantly using any of the following methods:

#### Method A: Open Directly in Browser
Double-click `index.html` or open it directly in Google Chrome, Microsoft Edge, Firefox, or Safari:
```
file:///c:/Users/Dharani/OneDrive/Documents/Smartphone%20Addiction/Smartphone-Addiction-Prediction/index.html
```

#### Method B: Using Python's Built-in HTTP Server
```bash
python -m http.server 8080
```
Then navigate to: `http://localhost:8080/`

---

### 4.2 Retraining the Model with New Data
If you update or replace the dataset:

1. Place the new CSV file in the `data/` directory.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The script will print the new evaluation metrics and output the updated JavaScript constants:
   ```
   const SCALER_MEAN = [...];
   const SCALER_SCALE = [...];
   const LOGREG_COEF = [...];
   const LOGREG_INTERCEPT = [...];
   ```
4. Open `index.html` in an editor.
5. Locate lines ~380–405 inside `<script>` and replace `SCALER_MEAN`, `SCALER_SCALE`, `LOGREG_COEF`, and `LOGREG_INTERCEPT` with the freshly printed arrays.
6. Save `index.html` and refresh your browser. Your updated model is live!

---

## Technical Specifications & Zero-Dependency Guarantee
- **Backend:** None. Zero network requests, zero telemetry, zero cookies.
- **Privacy:** 100% on-device local computation. No user data ever leaves the browser.
- **Dependencies:** Standard library HTML5, CSS3, ES6+ JavaScript.
- **Training Environment:** Python 3.10+, `scikit-learn >= 1.2`, `pandas >= 2.0`, `numpy >= 1.24`.