import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import joblib
import os
from scipy.stats import uniform, randint

def run_modeling():
   
    df = pd.read_csv("data/processed/final_dataset.csv")
    X = df.drop("has_insurance", axis=1)
    y = df["has_insurance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Distribución de clases (Train):")
    print(y_train.value_counts(normalize=True))
    balance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Ratio Clase 0 / Clase 1: {balance_ratio:.2f}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    for name, model in models.items():
        print(f"Evaluando modelo: {name}")
        auc_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        print(f"AUC promedio (CV): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"Reporte de clasificación (Test) para {name}:")
        print(classification_report(y_test, y_pred))
        final_auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC final en Test: {final_auc:.4f}")

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Curva ROC - {name}")
        plt.savefig(f"outputs/figures/roc_{name.replace(' ', '_').lower()}.png")
        plt.close()

        results[name] = {"model": model, "auc": final_auc}

    print("Ajuste de hiperparámetros para XGBoost:")
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'scale_pos_weight': [balance_ratio]
    }

    random_search = RandomizedSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        param_distributions=param_dist,
        n_iter=50,
        scoring='roc_auc',
        cv=cv_strategy,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train_res, y_train_res)
    print("Mejores hiperparámetros encontrados:")
    print(random_search.best_params_)

    best_xgb = random_search.best_estimator_

    y_pred_best = best_xgb.predict(X_test)
    y_proba_best = best_xgb.predict_proba(X_test)[:, 1]
    final_auc_best = roc_auc_score(y_test, y_proba_best)
    print("Reporte mejor XGBoost en Test Set:")
    print(classification_report(y_test, y_pred_best))
    print(f"AUC-ROC final (Mejor XGBoost Test): {final_auc_best:.4f}")

    joblib.dump(best_xgb, "outputs/models/xgboost_tuned_model.pkl")
    print("Modelo XGBoost ajustado guardado.")

    print("Calculando SHAP values para XGBoost ajustado...")
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary_bar_xgboost_tuned.png")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary_detail_xgboost_tuned.png")
    plt.close()
    print("Gráficos SHAP guardados.")

if __name__ == "__main__":
    run_modeling()
