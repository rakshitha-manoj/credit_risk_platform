import re
from openai import OpenAI
from src.talk_to_data.prompt_templates import build_nl_to_sql_prompt, build_answer_prompt, build_self_correction_prompt, build_shap_narration_prompt
from src.talk_to_data.query_runner import run_query
from src.utils.config import config
from src.utils.logger import get_logger
logger = get_logger(__name__)
_DESTRUCTIVE_PATTERN = re.compile('\\b(delete|drop|truncate|alter|insert|update|create|attach|detach|exec|execute|grant|revoke|shutdown|kill|destroy|remove|erase|wipe|purge)\\b', re.IGNORECASE)
_INJECTION_PATTERN = re.compile('(ignore\\s+(previous|above|all)\\s+(instructions?|prompts?|rules?)|you\\s+are\\s+now|act\\s+as|pretend|jailbreak|bypass|override|disregard)', re.IGNORECASE)

def validate_question(question: str) -> str | None:
    if _DESTRUCTIVE_PATTERN.search(question):
        return '⚠️ This query appears to request a destructive operation (e.g. delete, drop, update). Only read-only analytical questions are supported.'
    if _INJECTION_PATTERN.search(question):
        return '⚠️ This query was flagged as a potential prompt-injection attempt. Please rephrase your analytical question.'
    if len(question) > 500:
        return '⚠️ Your question is too long. Please keep it under 500 characters.'
    return None
TABLE_SCHEMA = '\nSK_ID_CURR (applicant id), TARGET (0/1 defaulted), AMT_INCOME_TOTAL,\nAMT_CREDIT, AMT_ANNUITY, NAME_EDUCATION_TYPE, DAYS_EMPLOYED,\ndefault_probability, risk_band (Low/Medium/High).\n\nEngineered aggregate columns:\n- bureau_*: bureau_count, bureau_credit_active_ratio, bureau_avg_credit_sum,\n  bureau_avg_days_credit, bureau_total_overdue, bureau_avg_dpd_ratio\n- prev_*: prev_app_count, prev_approved_ratio, prev_avg_credit\n- avg_payment_delay, max_payment_delay, avg_underpayment\n- pos_*: pos_contract_count, pos_avg_dpd, pos_max_dpd, pos_completed_ratio\n- cc_*: cc_avg_balance, cc_avg_utilization, cc_avg_dpd, cc_max_dpd\n'
MAX_CORRECTION_ATTEMPTS = 1

def _client() -> OpenAI:
    base_url = None
    api_key = config.OPENAI_API_KEY
    if config.LLM_PROVIDER == 'groq':
        base_url = 'https://api.groq.com/openai/v1'
        api_key = config.GROQ_API_KEY or config.OPENAI_API_KEY
    return OpenAI(api_key=api_key, base_url=base_url)

def _clean_sql(raw: str) -> str:
    return raw.replace('```sql', '').replace('```', '').strip()

def _call_llm(prompt: str, temperature: float=0) -> str:
    response = _client().chat.completions.create(model=config.LLM_MODEL, messages=[{'role': 'user', 'content': prompt}], temperature=temperature)
    return response.choices[0].message.content.strip()

def generate_sql(question: str) -> str:
    prompt = build_nl_to_sql_prompt(question, TABLE_SCHEMA)
    sql = _clean_sql(_call_llm(prompt, temperature=0))
    logger.info(f'Generated SQL: {sql}')
    return sql

def correct_sql(question: str, sql: str, error: str) -> str:
    prompt = build_self_correction_prompt(question, sql, error, TABLE_SCHEMA)
    corrected = _clean_sql(_call_llm(prompt, temperature=0))
    logger.info(f'Self-corrected SQL: {corrected}')
    return corrected

def synthesize_answer(question: str, result_df, sql: str='') -> str:
    MAX_ROWS_FOR_LLM = 50
    if result_df.empty:
        result_str = 'No rows returned.'
    elif len(result_df) > MAX_ROWS_FOR_LLM:
        result_str = result_df.head(MAX_ROWS_FOR_LLM).to_string(index=False) + f'\n\n... ({len(result_df)} total rows, showing first {MAX_ROWS_FOR_LLM})'
    else:
        result_str = result_df.to_string(index=False)
    prompt = build_answer_prompt(question, result_str, sql=sql)
    return _call_llm(prompt, temperature=0.2)

def _run_with_self_correction(question: str, sql: str):
    attempt_sql = sql
    last_error = None
    for attempt in range(MAX_CORRECTION_ATTEMPTS + 1):
        try:
            result_df = run_query(attempt_sql)
            return (attempt_sql, result_df, None)
        except Exception as e:
            last_error = str(e)
            logger.warning(f'SQL attempt {attempt} failed: {last_error}')
            if attempt < MAX_CORRECTION_ATTEMPTS:
                attempt_sql = correct_sql(question, attempt_sql, last_error)
    return (attempt_sql, None, last_error)

def ask(question: str) -> dict:
    rejection = validate_question(question)
    if rejection is not None:
        logger.warning(f'Question rejected by guardrail: {question!r}')
        return {'question': question, 'sql': '', 'error': rejection}
    try:
        sql = generate_sql(question)
        final_sql, result_df, error = _run_with_self_correction(question, sql)
        if error is not None:
            try:
                from src.utils.audit import init_audit_tables, log_chat
                init_audit_tables()
                log_chat(question, sql=final_sql, error=error)
            except Exception:
                pass
            return {'question': question, 'sql': final_sql, 'error': error}
        answer = synthesize_answer(question, result_df, sql=final_sql)
        try:
            from src.utils.audit import init_audit_tables, log_chat
            init_audit_tables()
            log_chat(question, sql=final_sql, answer=answer)
        except Exception:
            pass
        return {'question': question, 'sql': final_sql, 'result': result_df, 'answer': answer}
    except Exception as exc:
        logger.error(f'Unexpected error in ask(): {exc}', exc_info=True)
        friendly = 'Something went wrong while processing your question. Please try rephrasing or using a simpler query.'
        return {'question': question, 'sql': '', 'error': friendly}

def narrate_shap_explanation(shap_contributions) -> str:
    top = shap_contributions.head(5)
    summary_lines = [f'{feature}: {('+' if value > 0 else '')}{value:.3f}' for feature, value in top.items()]
    prompt = build_shap_narration_prompt('\n'.join(summary_lines))
    return _call_llm(prompt, temperature=0.2)
SAMPLE_QUERIES = ['How many applicants fall into the High risk band?', 'What is the average income of applicants who defaulted?', 'Which education type has the highest default rate?', 'How many applicants have more than 2 prior loan applications?', 'What is the average credit amount for Low risk applicants?', 'What is the average credit card utilization for High risk applicants?']