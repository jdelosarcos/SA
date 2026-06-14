import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from preprocessing import clean_student_data, get_modeling_data, TARGET
from train_models import train_all_from_raw

st.set_page_config(page_title='Student Degree Outcome Dashboard', layout='wide')

MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)
DEFAULT_DATA = Path('data/StudProfile.xlsx')

st.title('Student Degree Outcome Prediction Dashboard')
st.caption('Streamlit-ready version: upload dataset, view patterns, train model, predict outcomes, and generate research-ready tables.')

@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def clean_data_cached(raw_df):
    return clean_student_data(raw_df)


def load_default_or_upload():
    uploaded = st.sidebar.file_uploader('Upload Excel dataset', type=['xlsx', 'xls'])
    if uploaded is not None:
        return read_excel_bytes(uploaded.getvalue()), uploaded.name
    if DEFAULT_DATA.exists():
        return pd.read_excel(DEFAULT_DATA), str(DEFAULT_DATA)
    return None, None


def show_metric_cards(clean_df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f'{clean_df.shape[0]:,}')
    c2.metric('Columns', f'{clean_df.shape[1]:,}')
    c3.metric('Completed', int((clean_df[TARGET] == 'Completed').sum()))
    c4.metric('Stopped', int((clean_df[TARGET] == 'Stopped').sum()))


def model_exists():
    return (MODELS_DIR / 'outcome_model.pkl').exists() and (MODELS_DIR / 'feature_list.pkl').exists()


def load_model_package():
    model = joblib.load(MODELS_DIR / 'outcome_model.pkl')
    features = joblib.load(MODELS_DIR / 'feature_list.pkl')
    meta_path = MODELS_DIR / 'model_metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, features, metadata

with st.sidebar:
    st.header('Menu')
    page = st.radio('Go to', ['Overview', 'Feature Relationships', 'Train Model', 'Predict Outcomes', 'Association Rules', 'Interpretation'])
    feature_set = st.selectbox('Feature set for model', ['academic', 'enrollment'])

raw_df, source_name = load_default_or_upload()
if raw_df is None:
    st.info('Please upload your Excel file in the sidebar. The app is ready and not loading indefinitely.')
    st.stop()

try:
    clean_df = clean_data_cached(raw_df)
except Exception as e:
    st.error('The uploaded file was read, but preprocessing failed. Please check if the columns match the student profile template.')
    st.exception(e)
    st.stop()

if page == 'Overview':
    st.subheader('Dataset Overview')
    st.write(f'Loaded source: `{source_name}`')
    show_metric_cards(clean_df)

    status_counts = clean_df[TARGET].value_counts().reset_index()
    status_counts.columns = ['Degree Status', 'Count']
    st.plotly_chart(px.bar(status_counts, x='Degree Status', y='Count', text='Count', title='Degree Status Distribution'), use_container_width=True)
    st.info('Use F1-macro and balanced accuracy when classes are imbalanced.')

    missing = (clean_df.isna().mean() * 100).sort_values(ascending=False).head(20).reset_index()
    missing.columns = ['Variable', 'Missing Percent']
    st.plotly_chart(px.bar(missing, x='Missing Percent', y='Variable', orientation='h', title='Top Missing Variables'), use_container_width=True)
    st.dataframe(clean_df.head(50), use_container_width=True)

elif page == 'Feature Relationships':
    st.subheader('Feature Relationships')
    model_df = clean_df[clean_df[TARGET].isin(['Completed', 'Stopped', 'Shifted'])].copy()
    categorical_options = [c for c in ['course', 'shs_strand', 'has_scholarship', 'has_extracurricular', 'family_income', 'residence', 'mother_occ_group', 'father_occ_group', 'performance_1st_year'] if c in model_df.columns]
    numeric_options = [c for c in ['age_first_year', 'hs_average', 'income_rank', 'intro_computing', 'programming', 'gwa_1st_year', 'extracurricular_count'] if c in model_df.columns]

    if categorical_options:
        feature = st.selectbox('Categorical feature', categorical_options)
        pct = pd.crosstab(model_df[feature], model_df[TARGET], normalize='index').mul(100).reset_index()
        long = pct.melt(id_vars=feature, var_name='Degree Status', value_name='Percentage')
        st.plotly_chart(px.bar(long, x=feature, y='Percentage', color='Degree Status', title=f'Degree Status by {feature}'), use_container_width=True)
        st.dataframe(pct.round(2), use_container_width=True)

    if numeric_options:
        num = st.selectbox('Numeric feature', numeric_options)
        st.plotly_chart(px.box(model_df, x=TARGET, y=num, points='all', title=f'{num} by Degree Status'), use_container_width=True)
        st.info('Overlapping boxplots indicate weak class separability, which explains lower accuracy.')

elif page == 'Train Model':
    st.subheader('Train Lightweight Model')
    st.write('This cloud-safe version uses class-weighted Logistic Regression, Random Forest, and Extra Trees. It avoids heavy cloud bottlenecks.')
    if st.button('Train model now', type='primary'):
        try:
            with st.spinner('Training. This should finish shortly...'):
                model, results, metadata, cleaned = train_all_from_raw(raw_df, feature_set=feature_set)
            st.success('Training completed. Model files were saved in the app runtime.')
            st.json(metadata)
            st.dataframe(results, use_container_width=True)
        except Exception as e:
            st.error('Training failed. See details below.')
            st.exception(e)
    elif model_exists():
        model, features, metadata = load_model_package()
        st.success('Saved model found.')
        st.json(metadata)
    else:
        st.warning('No trained model found yet. Click Train model now.')

elif page == 'Predict Outcomes':
    st.subheader('Predict Outcomes')
    if not model_exists():
        st.warning('Please train the model first in the Train Model page.')
        st.stop()
    model, features, metadata = load_model_package()
    available = [f for f in features if f in clean_df.columns]
    X = clean_df[available].copy()
    result_df = clean_df.copy()
    result_df['Predicted_Degree_Status'] = model.predict(X)
    if hasattr(model.named_steps.get('model'), 'predict_proba'):
        try:
            proba = model.predict_proba(X)
            result_df['Prediction_Confidence'] = proba.max(axis=1)
        except Exception:
            pass

    pred_counts = result_df['Predicted_Degree_Status'].value_counts().reset_index()
    pred_counts.columns = ['Predicted Status', 'Count']
    st.plotly_chart(px.pie(pred_counts, names='Predicted Status', values='Count', title='Predicted Outcome Distribution'), use_container_width=True)

    valid = result_df[TARGET].isin(metadata.get('classes', []))
    if valid.sum() > 0:
        y_true = result_df.loc[valid, TARGET]
        y_pred = result_df.loc[valid, 'Predicted_Degree_Status']
        metrics = pd.DataFrame([{
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }])
        st.dataframe(metrics, use_container_width=True)
        labels = metadata.get('classes', [])
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        st.plotly_chart(px.imshow(cm_df, text_auto=True, title='Confusion Matrix'), use_container_width=True)

    st.dataframe(result_df.head(100), use_container_width=True)
    st.download_button('Download predictions CSV', result_df.to_csv(index=False).encode('utf-8'), 'student_degree_predictions.csv', 'text/csv')

elif page == 'Association Rules':
    st.subheader('Association Rule Mining')
    df = clean_df[clean_df[TARGET].isin(['Completed', 'Stopped', 'Shifted'])].copy()
    min_support = st.slider('Minimum support', 0.01, 0.20, 0.04, 0.01)
    min_conf = st.slider('Minimum confidence', 0.30, 0.95, 0.50, 0.05)
    if st.button('Generate rules'):
        try:
            d = df.copy()
            d['age_band'] = pd.cut(d['age_first_year'], bins=[0,18,21,99], labels=['Age <=18', 'Age 19-21', 'Age >=22'])
            d['hs_average_band'] = pd.cut(d['hs_average'], bins=[0,79,89,100], labels=['HS low', 'HS average', 'HS high'])
            cols = [c for c in ['degree_status','course','shs_strand','has_scholarship','has_extracurricular','family_income','residence','mother_occ_group','father_occ_group','performance_1st_year','programming_performance','intro_performance','age_band','hs_average_band'] if c in d.columns]
            transactions = []
            for _, row in d[cols].iterrows():
                transactions.append([f'{col}={row[col]}' for col in cols if pd.notna(row[col]) and str(row[col]) != 'Unknown'])
            te = TransactionEncoder()
            basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
            itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
            rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
            status_items = {f'degree_status={c}' for c in ['Completed', 'Stopped', 'Shifted']}
            rules = rules[rules['consequents'].apply(lambda x: len(set(x) & status_items) > 0)].copy()
            if rules.empty:
                st.warning('No status-related rules found. Lower support or confidence.')
                st.stop()
            rules['Antecedents'] = rules['antecedents'].apply(lambda x: ' AND '.join(sorted(list(x))))
            rules['Consequents'] = rules['consequents'].apply(lambda x: ' AND '.join(sorted(list(x))))
            show = rules.sort_values(['lift','confidence','support'], ascending=False)[['Antecedents','Consequents','support','confidence','lift']].head(30)
            st.dataframe(show, use_container_width=True)
            st.plotly_chart(px.bar(show.head(12).iloc[::-1], x='lift', y='Antecedents', color='Consequents', orientation='h', title='Top Rules by Lift'), use_container_width=True)
        except Exception as e:
            st.error('Rule generation failed.')
            st.exception(e)

elif page == 'Interpretation':
    st.subheader('Research Interpretation Guide')
    st.markdown(f'''
    **Dataset distribution:** `{clean_df[TARGET].value_counts().to_dict()}`

    **Interpretation:** If the classes are imbalanced or overlapping, low accuracy does not automatically mean the study failed. Report balanced accuracy, F1-macro, feature relationships, and association rules.

    **For Chapter 4:** The dashboard supports three objectives: identifying influential features, predicting completion/stopping/shifting, and analyzing patterns through association rules.
    ''')
