import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.ml.predict import predict, preprocess_for_inference, load_pipeline
from src.ml.explain import derive_rules, explain_instance
from src.talk_to_data.nl_to_sql import ask, narrate_shap_explanation, SAMPLE_QUERIES
from src.utils.audit import get_recent_predictions, get_recent_chats
from src.utils.config import config
from src.utils.helpers import load_artifact
st.set_page_config(page_title='Credit Risk Intelligence Platform', layout='wide')
st.title('AI-Powered Credit Risk Intelligence Platform')

@st.cache_resource
def load_cached_artifacts():
    model, encoders, feature_cols = load_pipeline()
    medians_path = f'{config.MODELS_DIR}/medians.joblib'
    medians = load_artifact(medians_path) if os.path.exists(medians_path) else {}
    calibrator_path = f'{config.MODELS_DIR}/calibrator.joblib'
    calibrator = load_artifact(calibrator_path) if os.path.exists(calibrator_path) else None
    return (model, encoders, feature_cols, calibrator, medians)
model, encoders, feature_cols, calibrator, medians = load_cached_artifacts()
section = st.sidebar.radio('Navigate', ['EDA', 'Risk Prediction', 'Explainability', 'Business Rules', 'Talk to Data', 'Audit Log'])
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'notebooks', 'figures')
if section == 'EDA':
    st.header('Exploratory Data Analysis')
    st.caption('Key empirical risk relationships identified across the Home Credit dataset.')
    if not os.path.exists(FIGURES_DIR):
        st.info('Run notebooks/eda.ipynb first to generate figures.')
    else:
        insight_figures = [('insight1_income_default.png', 'Default rate by income bracket', 'Lower-income brackets show elevated default rates; non-linear risk distribution across upper income tiers.'), ('insight2_employment_default.png', 'Default rate by employment length', 'Shorter tenure correlates strongly with default risk; applicants with <1 year show highest default probability.'), ('insight3_education_default.png', 'Default rate by education type', 'Clear educational gradient: lower secondary exhibits highest default rates, academic degree holders lowest.'), ('insight4a_bureau_active_ratio.png', 'Default rate by bureau active credit ratio', 'Higher proportions of actively open bureau lines indicate elevated credit utilization and risk.'), ('insight4b_bureau_dpd_ratio.png', 'Default rate by bureau avg DPD ratio', 'Historical days-past-due frequency across external bureau accounts is a powerful default predictor.'), ('insight5_payment_delay.png', 'Default rate by avg payment delay', 'Payment delay duration on prior internal installments correlates monotonically with default.'), ('insight6_cc_utilization.png', 'Default rate by credit card utilization', 'High revolving credit card balance utilization signals liquidity strain and heightened risk.'), ('insight7_pos_dpd.png', 'Default rate by POS/cash loan DPD', 'Past delinquencies on point-of-sale and cash loans directly elevate current default probability.')]
        found_any = False
        for filename, caption_title, description in insight_figures:
            path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(path):
                found_any = True
                st.subheader(caption_title)
                st.image(path, caption=description, use_container_width=True)
        if not found_any:
            st.info('Run notebooks/eda.ipynb first to generate figures.')
elif section == 'Risk Prediction':
    st.header('Risk Prediction')
    st.caption('Upload applicant data in CSV format to score with the calibrated LightGBM model.')
    uploaded = st.file_uploader('Upload applicant data (CSV)', type='csv')
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        with st.spinner('Scoring applicants and applying Platt calibration...'):
            preprocessed_df = preprocess_for_inference(raw_df, encoders=encoders, feature_cols=feature_cols, medians=medians)
            results = predict(raw_df, model=model, encoders=encoders, feature_cols=feature_cols, calibrator=calibrator, medians=medians)
        st.session_state['raw_df'] = raw_df
        st.session_state['preprocessed_df'] = preprocessed_df
        st.session_state['results'] = results
        if config.ID_COL in raw_df.columns:
            st.session_state['id_list'] = raw_df[config.ID_COL].astype(str).tolist()
        else:
            st.session_state['id_list'] = [f'Row {i}' for i in range(len(raw_df))]
        st.success(f'Successfully scored {len(raw_df)} applicants.')
        st.subheader('Prediction Results')
        st.dataframe(results, use_container_width=True)
        st.subheader('Risk Band Distribution')
        band_counts = results['risk_band'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
        st.bar_chart(band_counts)
elif section == 'Explainability':
    st.header('Explainability (SHAP)')
    st.caption("Inspect individual feature contributions driving an applicant's default probability.")
    preprocessed_df = st.session_state.get('preprocessed_df')
    results = st.session_state.get('results')
    id_list = st.session_state.get('id_list')
    if preprocessed_df is None or results is None or id_list is None:
        st.info('Upload data in Risk Prediction first.')
    else:
        selected_id = st.selectbox('Select Applicant (SK_ID_CURR)', id_list)
        selected_idx = id_list.index(selected_id)
        applicant_row = preprocessed_df.iloc[[selected_idx]]
        applicant_result = results.iloc[selected_idx]
        with st.spinner('Computing SHAP explanations...'):
            contributions = explain_instance(applicant_row)
        st.subheader('Top 10 Feature Contributions (SHAP)')
        top10 = contributions.head(10)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ['#d32f2f' if v > 0 else '#1976d2' for v in top10.values]
        top10.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
        ax.set_xlabel('SHAP Contribution (Red: increases risk | Blue: decreases risk)')
        ax.set_title(f'SHAP Drivers for Applicant {selected_id}')
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        col1, col2 = st.columns(2)
        prob = float(applicant_result['default_probability'])
        band = str(applicant_result['risk_band'])
        col1.metric('Calibrated Default Probability', f'{prob:.2%}')
        col2.metric('Assigned Risk Band', band)
        st.subheader('AI Analysis')
        try:
            with st.spinner('Generating AI explanation...'):
                narration = narrate_shap_explanation(contributions)
                st.markdown(narration)
        except Exception as e:
            st.info(f'AI narration unavailable ({e}). Refer to the SHAP contribution chart above.')
elif section == 'Business Rules':
    st.header('Derived Business Rules')
    st.caption('Transparent surrogate decision-tree rules and descriptive feature drivers.')
    RULES_SAMPLE_SIZE = 500

    @st.cache_data(show_spinner=False)
    def _load_rules_sample():
        import sqlite3
        db_path = config.SQL_DB_PATH
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        try:
            total = pd.read_sql_query('SELECT COUNT(*) AS n FROM applicants', conn).iloc[0]['n']
            sample_df = pd.read_sql_query(f'SELECT * FROM applicants ORDER BY RANDOM() LIMIT {RULES_SAMPLE_SIZE}', conn)
        finally:
            conn.close()
        return (sample_df, int(total))
    result = _load_rules_sample()
    if result is None:
        st.info('SQLite database not found. Run the data pipeline first to build `sql/credit_risk.db`.')
    else:
        sample_raw, total_rows = result
        with st.spinner(f'Extracting rules from {len(sample_raw)} applicants (sampled from {total_rows:,} total)...'):
            X_sample = preprocess_for_inference(sample_raw, encoders=encoders, feature_cols=feature_cols, medians=medians)
            rules = derive_rules(X_sample)
        fidelity = rules['surrogate']['fidelity']
        st.metric('Surrogate Fidelity', f'{fidelity:.1%}')
        st.caption(f'Fidelity measures how closely the surrogate decision tree approximates the LightGBM base model on a {len(X_sample)}-row random sample (drawn from {total_rows:,} training records). Values of 85–95% indicate a reliable policy approximation; 100% on small samples may indicate overfitting.')
        st.subheader('Surrogate Decision-Tree Rules')
        st.code(rules['surrogate']['rules_text'], language='text')
        st.subheader('Descriptive SHAP Summary')
        for r in rules['descriptive']:
            st.write(f'- {r}')
elif section == 'Talk to Data':
    st.header('Talk to Data')
    st.caption('Ask natural language questions about applicant credit profiles and portfolio risk.')
    st.subheader('Suggested Queries')

    def set_query(query_text):
        st.session_state['chat_input'] = query_text
        st.session_state.pop('last_response', None)
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUERIES):
        with cols[i % 3]:
            st.button(q, key=f'sample_q_{i}', use_container_width=True, on_click=set_query, args=(q,))
    question = st.text_input('Ask a question about the applicant data:', key='chat_input')
    if st.button('Run Query', type='primary'):
        if question and question.strip():
            with st.spinner('Processing natural language query...'):
                response = ask(question.strip())
            st.session_state['last_response'] = response
        else:
            st.warning('Please enter a question or select a suggested query above.')
    if 'last_response' in st.session_state:
        response = st.session_state['last_response']
        if 'error' in response:
            st.error(response['error'])
        else:
            st.subheader('Answer')
            st.write(response['answer'])
            with st.expander('Show generated SQL'):
                st.code(response['sql'], language='sql')
            st.subheader('Query Result')
            st.dataframe(response['result'], use_container_width=True)
elif section == 'Audit Log':
    st.header('Audit Trail')
    st.caption('Immutable record of scored applicants and natural language chatbot exchanges.')
    tab1, tab2 = st.tabs(['Predictions', 'Chat History'])
    with tab1:
        pred_rows = get_recent_predictions()
        if pred_rows:
            pred_df = pd.DataFrame(pred_rows, columns=['id', 'timestamp', 'applicant_id', 'default_probability', 'risk_band', 'top_shap_drivers'])
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.info('No predictions logged yet.')
    with tab2:
        chat_rows = get_recent_chats()
        if chat_rows:
            chat_df = pd.DataFrame(chat_rows, columns=['id', 'timestamp', 'question', 'generated_sql', 'answer', 'error'])
            st.dataframe(chat_df, use_container_width=True)
        else:
            st.info('No chat history yet.')