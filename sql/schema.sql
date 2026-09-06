-- Schema for the applicants table used by the Talk-to-Data module.

CREATE TABLE IF NOT EXISTS applicants (
    SK_ID_CURR INTEGER PRIMARY KEY,
    TARGET INTEGER,
    AMT_INCOME_TOTAL REAL,
    AMT_CREDIT REAL,
    AMT_ANNUITY REAL,
    NAME_EDUCATION_TYPE TEXT,
    DAYS_EMPLOYED REAL,
    default_probability REAL,
    risk_band TEXT,
    -- bureau + bureau_balance derived
    bureau_count INTEGER,
    bureau_credit_active_ratio REAL,
    bureau_avg_credit_sum REAL,
    bureau_avg_days_credit REAL,
    bureau_total_overdue REAL,
    bureau_avg_dpd_ratio REAL,
    -- previous_application derived
    prev_app_count INTEGER,
    prev_approved_ratio REAL,
    prev_avg_credit REAL,
    -- installments_payments derived
    avg_payment_delay REAL,
    max_payment_delay REAL,
    avg_underpayment REAL,
    -- POS_CASH_balance derived
    pos_contract_count INTEGER,
    pos_avg_dpd REAL,
    pos_max_dpd REAL,
    pos_completed_ratio REAL,
    -- credit_card_balance derived
    cc_avg_balance REAL,
    cc_avg_utilization REAL,
    cc_avg_dpd REAL,
    cc_max_dpd REAL
);
