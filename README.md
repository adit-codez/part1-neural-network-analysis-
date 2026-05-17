# 📊 Customer Churn Prediction — Neural Network Analysis

> **Part 1: Neural Network Fundamentals and Training Behavior Analysis**

---

## 📌 Problem Statement

Predict whether a telecom customer will **churn** (leave the service) or **stay**, based on their usage patterns, plan details, and behavioral signals. This is a **binary classification** problem.

**Target Variable:** `churn` (0 = stays, 1 = churns)

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Total Records | 2,000 |
| Total Features | 16 input + 1 target |
| Churn Rate | ~2.3% (highly imbalanced) |
| Missing Values | None |

### Features

| Feature | Type | Description |
|---|---|---|
| `region` | Categorical | Geographic region |
| `plan_type` | Categorical | Basic / Standard / Premium / Enterprise |
| `contract_type` | Categorical | Month-to-month / One-year / Two-year |
| `payment_method` | Categorical | UPI / Wallet / Debit Card / etc. |
| `autopay_enabled` | Binary | Whether autopay is on |
| `tenure_months` | Numerical | How long the customer has been subscribed |
| `monthly_charges_inr` | Numerical | Monthly bill amount |
| `avg_login_days_per_month` | Numerical | Platform engagement |
| `support_tickets_last_90_days` | Numerical | Customer support activity |
| `payment_delay_days` | Numerical | Average delay in payments |
| `data_usage_gb` | Numerical | Monthly data usage |
| `satisfaction_score` | Numerical | Satisfaction rating (1–10) |
| `last_complaint_days_ago` | Numerical | Days since last complaint |
| `discount_percent` | Numerical | Discount on plan |
| `referral_count` | Numerical | Number of referrals made |

---

## 🔧 Task 2: Data Preprocessing

Steps applied:
1. **Drop** non-informative `customer_id` column
2. **Encode** categorical features using `LabelEncoder`
3. **Scale** all numerical features with `StandardScaler`
4. **Split**: 80% train / 20% test (stratified)
5. **Class imbalance**: Handle using `class_weight='balanced'` in the model

---

## 🧠 Task 3: Neural Network Architecture

```
Input (15 features)
      │
Dense(128) → ReLU → BatchNorm → Dropout(0.4)
      │
Dense(64) → ReLU → BatchNorm → Dropout(0.3)
      │
Dense(32) → ReLU → Dropout(0.2)
      │
Dense(1) → Sigmoid
      │
Output (churn probability)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy, AUC

---

## 📈 Task 4: Training & Evaluation

- Trained for up to 50 epochs with EarlyStopping
- Evaluated on held-out test set
- Confusion matrix + classification report included

---

## 🔬 Task 5: Hyperparameter Experiments

Three experiments run:

| Exp | Hidden Layers | Neurons | LR | Batch | Activation |
|---|---|---|---|---|---|
| 1 (Baseline) | 3 | 128→64→32 | 0.001 | 32 | ReLU |
| 2 (Deeper) | 4 | 256→128→64→32 | 0.001 | 32 | ReLU |
| 3 (Low LR) | 3 | 128→64→32 | 0.0001 | 64 | ReLU |
| 4 (ELU act.) | 3 | 128→64→32 | 0.001 | 32 | ELU |

Results are saved in `results/model_comparison_table.csv`.

---

## 💡 Task 6: Final Reflection

### What role do weights and biases play?
**Weights** determine how strongly each input feature influences the next layer. They are the learnable parameters that the model adjusts during training. **Biases** add a constant offset to each neuron's output, allowing the activation function to shift left or right. Together, they define the transformation `output = activation(W·input + b)`. Without well-tuned weights and biases, the network cannot capture patterns in data.

### Why is an activation function required?
Without activation functions, stacking multiple linear layers would still produce only a linear output — no matter how many layers you add. Activation functions (ReLU, Sigmoid, ELU) introduce **non-linearity**, enabling the network to learn complex, non-linear patterns like the interaction between payment delay, satisfaction score, and churn probability.

### What happens when learning rate is too high or too low?
- **Too high**: The optimizer overshoots the minimum — loss oscillates wildly or diverges entirely. The model never converges.
- **Too low**: Training becomes extremely slow. The model may get stuck in a local minimum, or not learn at all within a reasonable number of epochs.
- **Ideal**: Loss decreases smoothly and plateaus at a good minimum. Typically 0.001 (Adam) is a good starting point.

### Did the model show overfitting or underfitting?
The dataset has a **severe class imbalance** (~97% non-churn). Without class weighting, the model would underfit by predicting "no churn" for everything and still achieving 97% accuracy. With `class_weight='balanced'`, the model properly learns both classes. Minor overfitting may occur in deeper experiments — controlled using Dropout and EarlyStopping. The baseline model shows a healthy training/validation gap.

---

## 📁 Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```


