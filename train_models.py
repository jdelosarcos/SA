from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import RANDOM_STATE, load_excel, clean_student_data, get_modeling_data, TARGET

MODELS_DIR = Path('models')
OUTPUTS_DIR = Path('outputs')
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def make_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])


def safe_cv(y):
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(3, min_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def model_zoo():
    return {
        'Logistic Regression Balanced': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest Balanced': RandomForestClassifier(n_estimators=80, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=1),
        'Extra Trees Balanced': ExtraTreesClassifier(n_estimators=80, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1),
    }


def build_pipeline(model_name, X):
    return Pipeline([('preprocessor', make_preprocessor(X)), ('model', model_zoo()[model_name])])


def train_all_from_raw(raw_df, feature_set='academic'):
    clean_df = clean_student_data(raw_df)
    X, y, features, modeling_df, classes = get_modeling_data(clean_df, feature_set=feature_set)

    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
    }
    rows = []
    cv = safe_cv(y)
    for model_name in model_zoo():
        try:
            pipe = build_pipeline(model_name, X)
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
            rows.append({
                'Model': model_name,
                'Accuracy_mean': float(np.mean(scores['test_accuracy'])),
                'Balanced_Accuracy_mean': float(np.mean(scores['test_balanced_accuracy'])),
                'Precision_macro_mean': float(np.mean(scores['test_precision_macro'])),
                'Recall_macro_mean': float(np.mean(scores['test_recall_macro'])),
                'F1_macro_mean': float(np.mean(scores['test_f1_macro'])),
                'F1_macro_std': float(np.std(scores['test_f1_macro'])),
            })
        except Exception as e:
            rows.append({'Model': model_name, 'Error': str(e)})

    results = pd.DataFrame(rows).sort_values(['F1_macro_mean', 'Balanced_Accuracy_mean'], ascending=False, na_position='last')
    if results.dropna(subset=['F1_macro_mean']).empty:
        raise RuntimeError('No model successfully trained. Check class distribution and dataset format.')

    best = results.dropna(subset=['F1_macro_mean']).iloc[0]
    best_pipe = build_pipeline(best['Model'], X)
    best_pipe.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODELS_DIR / 'outcome_model.pkl')
    joblib.dump(features, MODELS_DIR / 'feature_list.pkl')
    metadata = {
        'feature_set': feature_set,
        'target': TARGET,
        'classes': list(classes),
        'best_model': str(best['Model']),
        'n_records_used': int(len(y)),
        'class_distribution': pd.Series(y).value_counts().to_dict(),
    }
    (MODELS_DIR / 'model_metadata.json').write_text(json.dumps(metadata, indent=2))
    clean_df.to_csv(OUTPUTS_DIR / 'cleaned_student_dataset.csv', index=False)
    results.to_csv(OUTPUTS_DIR / 'model_comparison.csv', index=False)
    return best_pipe, results, metadata, clean_df


def train_all(data_path='data/StudProfile.xlsx', feature_set='academic'):
    raw = load_excel(data_path)
    return train_all_from_raw(raw, feature_set=feature_set)

if __name__ == '__main__':
    model, results, metadata, clean_df = train_all()
    print('Training complete')
    print(metadata)
    print(results)
