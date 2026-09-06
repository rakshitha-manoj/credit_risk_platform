import pandas as pd
from src.utils.docker_utils import resolve_data_path
from src.utils.logger import get_logger
logger = get_logger(__name__)
DPD_STATUS_VALUES = {'1', '2', '3', '4', '5'}

def load_application(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('application_train.csv', data_dir)
    df = pd.read_csv(path)
    logger.info(f'Loaded application_train: {df.shape}')
    return df

def load_bureau_balance_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('bureau_balance.csv', data_dir)
    bb = pd.read_csv(path)
    bb['is_dpd'] = bb['STATUS'].isin(DPD_STATUS_VALUES).astype(int)
    agg = bb.groupby('SK_ID_BUREAU').agg(bb_months_on_record=('MONTHS_BALANCE', 'count'), bb_dpd_ratio=('is_dpd', 'mean')).reset_index()
    logger.info(f'Aggregated bureau_balance to bureau-level: {agg.shape}')
    return agg

def load_bureau_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('bureau.csv', data_dir)
    bureau = pd.read_csv(path)
    try:
        bb_agg = load_bureau_balance_agg(data_dir)
        bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    except FileNotFoundError as e:
        logger.warning(f'bureau_balance not found, skipping DPD enrichment: {e}')
        bureau['bb_dpd_ratio'] = pd.NA
    agg = bureau.groupby('SK_ID_CURR').agg(bureau_count=('SK_ID_BUREAU', 'count'), bureau_credit_active_ratio=('CREDIT_ACTIVE', lambda x: (x == 'Active').mean()), bureau_avg_credit_sum=('AMT_CREDIT_SUM', 'mean'), bureau_avg_days_credit=('DAYS_CREDIT', 'mean'), bureau_total_overdue=('AMT_CREDIT_SUM_OVERDUE', 'sum'), bureau_avg_dpd_ratio=('bb_dpd_ratio', 'mean')).reset_index()
    logger.info(f'Aggregated bureau (+ bureau_balance): {agg.shape}')
    return agg

def load_previous_application_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('previous_application.csv', data_dir)
    prev = pd.read_csv(path)
    agg = prev.groupby('SK_ID_CURR').agg(prev_app_count=('SK_ID_PREV', 'count'), prev_approved_ratio=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').mean()), prev_avg_credit=('AMT_CREDIT', 'mean')).reset_index()
    logger.info(f'Aggregated previous_application: {agg.shape}')
    return agg

def load_installments_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('installments_payments.csv', data_dir)
    inst = pd.read_csv(path)
    inst['payment_delay'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['underpayment'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    agg = inst.groupby('SK_ID_CURR').agg(avg_payment_delay=('payment_delay', 'mean'), max_payment_delay=('payment_delay', 'max'), avg_underpayment=('underpayment', 'mean')).reset_index()
    logger.info(f'Aggregated installments_payments: {agg.shape}')
    return agg

def load_pos_cash_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('POS_CASH_balance.csv', data_dir)
    pos = pd.read_csv(path)
    agg = pos.groupby('SK_ID_CURR').agg(pos_contract_count=('SK_ID_PREV', 'nunique'), pos_avg_dpd=('SK_DPD', 'mean'), pos_max_dpd=('SK_DPD', 'max'), pos_completed_ratio=('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').mean())).reset_index()
    logger.info(f'Aggregated POS_CASH_balance: {agg.shape}')
    return agg

def load_credit_card_agg(data_dir: str) -> pd.DataFrame:
    path = resolve_data_path('credit_card_balance.csv', data_dir)
    cc = pd.read_csv(path)
    cc['utilization'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, pd.NA)
    agg = cc.groupby('SK_ID_CURR').agg(cc_avg_balance=('AMT_BALANCE', 'mean'), cc_avg_utilization=('utilization', 'mean'), cc_avg_dpd=('SK_DPD', 'mean'), cc_max_dpd=('SK_DPD', 'max')).reset_index()
    logger.info(f'Aggregated credit_card_balance: {agg.shape}')
    return agg
AUX_LOADERS = (load_bureau_agg, load_previous_application_agg, load_installments_agg, load_pos_cash_agg, load_credit_card_agg)

def build_full_dataset(data_dir: str) -> pd.DataFrame:
    df = load_application(data_dir)
    for loader_fn in AUX_LOADERS:
        try:
            aux = loader_fn(data_dir)
            df = df.merge(aux, on='SK_ID_CURR', how='left')
        except FileNotFoundError as e:
            logger.warning(f'Skipping optional table: {e}')
    logger.info(f'Final joined dataset: {df.shape}')
    return df
if __name__ == '__main__':
    from src.utils.config import config
    data = build_full_dataset(config.DATA_DIR)
    print(data.head())
    print(f'\nTotal columns: {len(data.columns)}')