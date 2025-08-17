import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pandas.tseries.offsets import MonthBegin
import smtplib
from email.message import EmailMessage
import shutil
import subprocess

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def ensure_downloads_folder():
    downloads = Path.home() / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return downloads

def main():
    start_time = time.time()
    downloads_folder = ensure_downloads_folder()

    # === 1. DATA LOADING ===
    contracts_url = "https://raw.githubusercontent.com/Vintaos/credit-analyst-case/main/contracts_details.csv"
    payments_url = "https://raw.githubusercontent.com/Vintaos/credit-analyst-case/main/contract_payments.csv"

    try:
        contracts_df = pd.read_csv(contracts_url)
        payments_df = pd.read_csv(payments_url)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # === 2. DATA CLEANING & WRANGLING ===
    contracts_df.rename(columns={'contractid': 'contract_id'}, inplace=True)
    contracts_df.drop_duplicates(subset=['contract_id'], inplace=True)

    for df in [contracts_df, payments_df]:
        for col in df.columns:
            if 'date' in col.lower() or 'month' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

    contracts_df['down_payment_usd'] = contracts_df['down_payment_usd'].fillna(0)
    payments_df['total_paid'] = payments_df['total_paid'].fillna(0)

    if 'contract_value_usd' not in contracts_df.columns:
        contracts_df['contract_value_usd'] = 0.0

    contracts_df['first_full_month'] = contracts_df['registration_date'] + pd.offsets.MonthBegin(1)
    contracts_df['first_full_month'] = contracts_df['first_full_month'].dt.to_period('M').dt.to_timestamp()

    # === 3. JOIN DATA ===
    cumulative_paid = payments_df.groupby('contract_id')['total_paid'].sum().reset_index()
    contracts_df = pd.merge(contracts_df, cumulative_paid, on='contract_id', how='left')
    contracts_df['total_paid'] = contracts_df['total_paid'].fillna(0)
    contracts_df['cumulative_paid'] = contracts_df['total_paid'] + contracts_df['down_payment_usd']

    # === 4. SORT PAYMENTS FOR EASY REVIEW ===
    payments_df = payments_df.sort_values(by=['contract_id', 'pay_month']).reset_index(drop=True)

    # === 5. ADD COMPANY TO PAYMENTS_DF ===
    payments_df = payments_df.merge(
        contracts_df[['contract_id', 'company']],
        on='contract_id',
        how='left'
    )

    # === 6. KPI CALCULATIONS ===

    # --- 6.1 Collection Rate Trend ---
    contracts_df['start_month'] = contracts_df['registration_date'].dt.to_period('M')
    contracts_df['end_month'] = (contracts_df['registration_date'] + pd.to_timedelta(contracts_df['contract_tenor_days'], unit='D')).dt.to_period('M')

    expected_rows = []
    for _, row in contracts_df.iterrows():
        if pd.isnull(row['start_month']) or pd.isnull(row['end_month']):
            continue
        months = pd.period_range(row['start_month'], row['end_month'], freq='M')
        for m in months:
            expected_rows.append({
                'contract_id': row['contract_id'],
                'company': row['company'],
                'month': m.to_timestamp(),
                'expected_payment': round(row['daily_payment_amount_usd'] * m.days_in_month, 2)
            })
    expected_df = pd.DataFrame(expected_rows)

    payments_df['month'] = payments_df['pay_month'].dt.to_period('M').dt.to_timestamp()
    actual_df = payments_df.groupby(['contract_id', 'company', 'month'])['total_paid'].sum().reset_index()
    actual_df['total_paid'] = actual_df['total_paid'].round(2)

    trend_df = pd.merge(expected_df, actual_df, on=['contract_id', 'company', 'month'], how='left')
    trend_df['total_paid'] = trend_df['total_paid'].fillna(0).round(2)

    # --- 6.2 Filter to Financed Contracts Only ---
    financed_contracts = contracts_df[contracts_df['contract_type'] == 'FINANCED']['contract_id'].unique()
    trend_df = trend_df[trend_df['contract_id'].isin(financed_contracts)]

    trend_summary = trend_df.groupby(['company', 'month']).agg(
        expected_payment=('expected_payment', 'sum'),
        total_paid=('total_paid', 'sum')
    ).reset_index()

    trend_summary['collection_rate'] = (trend_summary['total_paid'] / trend_summary['expected_payment']) * 100
    trend_summary['collection_rate'] = trend_summary['collection_rate'].round(2)
    trend_summary['expected_payment'] = trend_summary['expected_payment'].round(2)
    trend_summary['total_paid'] = trend_summary['total_paid'].round(2)
    trend_summary = trend_summary[trend_summary['month'] <= pd.Timestamp('2025-12-31')]

    # --- 6.3 Repayment Rate Trend ---
    trend_summary = trend_summary.sort_values(['company', 'month'])
    trend_summary['cum_total_paid'] = trend_summary.groupby('company')['total_paid'].cumsum()
    trend_summary['cum_expected_payment'] = trend_summary.groupby('company')['expected_payment'].cumsum()
    trend_summary['repayment_rate'] = (trend_summary['cum_total_paid'] / trend_summary['cum_expected_payment']) * 100
    trend_summary['repayment_rate'] = trend_summary['repayment_rate'].round(2)

    # --- 6.4 Portfolio at Risk (PAR) ---
    trend_df = trend_df.sort_values(['contract_id', 'month'])
    trend_df['missed_payment'] = trend_df['total_paid'] < trend_df['expected_payment']
    trend_df['consec_missed'] = 0
    for contract in trend_df['contract_id'].unique():
        missed = 0
        for idx in trend_df[trend_df['contract_id'] == contract].index:
            if trend_df.loc[idx, 'missed_payment']:
                missed += 1
            else:
                missed = 0
            trend_df.loc[idx, 'consec_missed'] = missed

    def par_bucket(months):
        if months >= 4:
            return 'PAR120'
        elif months == 3:
            return 'PAR90'
        elif months == 2:
            return 'PAR60'
        elif months == 1:
            return 'PAR30'
        else:
            return 'Current'

    trend_df['PAR_bucket'] = trend_df['consec_missed'].apply(par_bucket)
    par_summary = trend_df.groupby(['company', 'month', 'PAR_bucket']).agg(
        par_amount=('expected_payment', 'sum')
    ).reset_index()
    par_pivot = par_summary.pivot_table(index=['company', 'month'], columns='PAR_bucket', values='par_amount', fill_value=0).reset_index()
    max_month = payments_df['pay_month'].max()
    par_pivot = par_pivot[par_pivot['month'] <= max_month]

    # --- 6.5 Add PAR% Columns ---
    par_buckets = ['PAR30', 'PAR60', 'PAR90', 'PAR120', 'Current']
    for bucket in par_buckets:
        if bucket not in par_pivot.columns:
            par_pivot[bucket] = 0.0
    par_buckets = ['PAR30', 'PAR60', 'PAR90', 'PAR120', 'Current']
    for bucket in par_buckets:
        if bucket not in par_pivot.columns:
            par_pivot[bucket] = 0.0
    par_pivot['Total_PAR'] = par_pivot[par_buckets].sum(axis=1)
    for bucket in par_buckets:
        par_pivot[f'{bucket}_pct'] = np.where(
            par_pivot['Total_PAR'] > 0,
            (par_pivot[bucket] / par_pivot['Total_PAR']) * 100,
            0
        ).round(2)









    # --- 6.6 Vintage Analysis ---
    contracts_df['vintage'] = contracts_df['registration_date'].dt.to_period('M')
    trend_df = trend_df.merge(
        contracts_df[['contract_id', 'vintage', 'registration_date', 'contract_tenor_days', 'company']],
        on='contract_id',
        how='left'
    )

    # Guarantee 'company' is present and filled
    if 'company' not in trend_df.columns or trend_df['company'].isnull().all():
        trend_df = trend_df.drop(columns=['company'], errors='ignore')
        trend_df = trend_df.merge(
            contracts_df[['contract_id', 'company']],
            on='contract_id',
            how='left'
        )

    trend_df['months_on_book'] = ((trend_df['month'] - trend_df['registration_date']).dt.days // 30)
    trend_df['tenor_months'] = (trend_df['contract_tenor_days'] // 30)
    trend_df = trend_df[(trend_df['months_on_book'] >= 0) & (trend_df['months_on_book'] <= trend_df['tenor_months'])]

    vintage_analysis = trend_df.groupby(['company', 'vintage', 'months_on_book']).agg(
        expected_payment=('expected_payment', 'sum'),
        total_paid=('total_paid', 'sum')
    ).reset_index()
    vintage_analysis['collection_rate'] = (vintage_analysis['total_paid'] / vintage_analysis['expected_payment']) * 100
    vintage_analysis['collection_rate'] = vintage_analysis['collection_rate'].round(2)

    # --- 6.7 Product Performance Analysis ---
    if 'baseunit_productname' in contracts_df.columns and 'product_family' in contracts_df.columns:
        trend_df = trend_df.merge(
            contracts_df[['contract_id', 'baseunit_productname', 'product_family']],
            on='contract_id',
            how='left'
        )
        product_summary = trend_df.groupby(['company', 'month', 'baseunit_productname', 'product_family']).agg(
            expected_payment=('expected_payment', 'sum'),
            total_paid=('total_paid', 'sum')
        ).reset_index()
        product_summary['collection_rate'] = (product_summary['total_paid'] / product_summary['expected_payment']) * 100
        product_summary['collection_rate'] = product_summary['collection_rate'].round(2)
    else:
        product_summary = pd.DataFrame()

    # --- 6.8 FPD KPI Analysis & FPD Rate Calculation ---
    FPD_CUTOFF_DAY = 5  # 5th of the month

    def get_fpd_month(reg_date):
        if pd.isnull(reg_date):
            return pd.NaT
        if reg_date.day <= FPD_CUTOFF_DAY:
            return (reg_date + pd.offsets.MonthBegin(1)).replace(day=1)
        else:
            return (reg_date + pd.offsets.MonthBegin(2)).replace(day=1)

    contracts_df['fpd_month'] = contracts_df['registration_date'].apply(get_fpd_month)
    financed = contracts_df[contracts_df['contract_type'] == 'FINANCED'].copy()
    payments_df['pay_month_only'] = payments_df['pay_month'].dt.to_period('M').dt.to_timestamp()

    fpd_payments = payments_df.merge(
        financed[['contract_id', 'company', 'fpd_month', 'registration_date']],
        on=['contract_id', 'company'],
        how='inner'
    )
    fpd_payments = fpd_payments[fpd_payments['pay_month_only'] == fpd_payments['fpd_month']]

    fpd_paid = fpd_payments.groupby('contract_id')['total_paid'].sum().reset_index().rename(columns={'total_paid': 'fpd_total_paid'})

    fpd_expected = trend_df.merge(
        financed[['contract_id', 'company', 'fpd_month', 'registration_date']],
        left_on=['contract_id', 'company', 'month'],
        right_on=['contract_id', 'company', 'fpd_month'],
        how='inner'
    )
    if 'registration_date_x' in fpd_expected.columns:
        fpd_expected = fpd_expected.rename(columns={'registration_date_x': 'registration_date'})
    elif 'registration_date_y' in fpd_expected.columns:
        fpd_expected = fpd_expected.rename(columns={'registration_date_y': 'registration_date'})
    fpd_expected = fpd_expected[['contract_id', 'expected_payment', 'company', 'registration_date', 'fpd_month']]

    fpd_final = pd.merge(fpd_expected, fpd_paid, on='contract_id', how='left')
    fpd_final['fpd_total_paid'] = fpd_final['fpd_total_paid'].fillna(0)
    fpd_final['FPD_flag'] = fpd_final['fpd_total_paid'] < fpd_final['expected_payment']
    fpd_final['reg_month'] = fpd_final['registration_date'].dt.to_period('M').dt.to_timestamp()

    FPD_rate_trend = fpd_final.groupby(['company', 'reg_month', 'fpd_month']).agg(
        contracts=('contract_id', 'count'),
        FPD_count=('FPD_flag', 'sum')
    ).reset_index()
    FPD_rate_trend['FPD_rate'] = (FPD_rate_trend['FPD_count'] / FPD_rate_trend['contracts']) * 100
    FPD_rate_trend['FPD_rate'] = FPD_rate_trend['FPD_rate'].round(2)

    # --- 6.9 Add FPD Zero Rate ---
    fpd_final['FPD_zero_flag'] = fpd_final['fpd_total_paid'] == 0
    FPD_zero_trend = fpd_final.groupby(['company', 'reg_month', 'fpd_month']).agg(
        FPD_zero_count=('FPD_zero_flag', 'sum')
    ).reset_index()
    FPD_rate_trend = FPD_rate_trend.merge(FPD_zero_trend, on=['company', 'reg_month', 'fpd_month'], how='left')
    FPD_rate_trend['FPD_zero_count'] = FPD_rate_trend['FPD_zero_count'].fillna(0).astype(int)
    FPD_rate_trend['FPD_zero_rate'] = (FPD_rate_trend['FPD_zero_count'] / FPD_rate_trend['contracts']) * 100
    FPD_rate_trend['FPD_zero_rate'] = FPD_rate_trend['FPD_zero_rate'].round(2)

    # === 7. CASH FLOW ANALYSIS BY COMPANY, PRODUCT FAMILY, PRODUCT NAME ===

    # --- 7.1 Prepare Disbursements (Outflows) ---
    disbursements = contracts_df.groupby(
        ['company', 'product_family', 'baseunit_productname', contracts_df['registration_date'].dt.to_period('M').dt.to_timestamp()]
    ).agg(
        disbursed_amount=('contract_value_usd', 'sum')
    ).reset_index().rename(columns={'registration_date': 'month'})

    # --- 7.2 Expected Inflows (Scheduled Payments) ---
    if 'baseunit_productname' not in trend_df.columns or 'product_family' not in trend_df.columns:
        trend_df = trend_df.merge(
            contracts_df[['contract_id', 'baseunit_productname', 'product_family']],
            on='contract_id',
            how='left'
        )
    expected_inflows = trend_df.groupby(
        ['company', 'product_family', 'baseunit_productname', 'month']
    ).agg(
        expected_payment=('expected_payment', 'sum')
    ).reset_index()

    # --- 7.3 Actual Inflows (Payments Received) ---
    actual_inflows = trend_df.groupby(
        ['company', 'product_family', 'baseunit_productname', 'month']
    ).agg(
        total_paid=('total_paid', 'sum')
    ).reset_index()

    # --- 7.4 Merge All Into Single Cash Flow Table ---
    cash_flow = pd.merge(disbursements, expected_inflows, on=['company', 'product_family', 'baseunit_productname', 'month'], how='outer')
    cash_flow = pd.merge(cash_flow, actual_inflows, on=['company', 'product_family', 'baseunit_productname', 'month'], how='outer')
    cash_flow = cash_flow.sort_values(['company', 'product_family', 'baseunit_productname', 'month']).fillna(0)

    # --- 7.5 Calculate Net Cash Flow ---
    cash_flow['net_cash_flow'] = cash_flow['total_paid'] - cash_flow['disbursed_amount']

    # --- 7.6 Add Cumulative Net Cash Flow Per Group ---
    cash_flow['cumulative_net_cash_flow'] = cash_flow.groupby(['company', 'product_family', 'baseunit_productname'])['net_cash_flow'].cumsum()

    # === 8. CASH FLOW PROJECTION & PORTFOLIO VALUATION ===

    # --- 8.1 Set Projection Period: August 2025 to July 2026 (12 months) ---
    projection_start = pd.Timestamp('2025-08-01')
    projection_months = pd.date_range(projection_start, periods=12, freq='MS')

    projection_rows = []
    for month in projection_months:
        for _, row in contracts_df.iterrows():
            reg_date = row['registration_date']
            tenor_days = row['contract_tenor_days']
            end_date = reg_date + pd.Timedelta(days=tenor_days) if pd.notnull(reg_date) and pd.notnull(tenor_days) else None
            if pd.notnull(reg_date) and pd.notnull(tenor_days) and reg_date <= month <= end_date:
                expected_payment = row['daily_payment_amount_usd'] * month.days_in_month
                projection_rows.append({
                    'month': month,
                    'contract_id': row['contract_id'],
                    'company': row['company'],
                    'expected_payment': expected_payment
                })

    projection_df = pd.DataFrame(projection_rows)
    monthly_projection = projection_df.groupby('month')['expected_payment'].sum().reset_index()
    monthly_projection['expected_payment'] = monthly_projection['expected_payment'].round(2)

    # --- 8.2 Discount Cash Flows to Present Value (Simple DCF) ---
    discount_rate = 0.15  # 15% annual discount rate
    monthly_discount = (1 + discount_rate) ** (1/12) - 1
    monthly_projection['discount_factor'] = [(1 + monthly_discount) ** -i for i in range(1, len(monthly_projection)+1)]
    monthly_projection['discounted_cashflow'] = monthly_projection['expected_payment'] * monthly_projection['discount_factor']

    portfolio_value = monthly_projection['discounted_cashflow'].sum()
    print(f"\nEstimated portfolio value (12-month discounted cash flow): ${portfolio_value:,.2f}")

    portfolio_value_df = pd.DataFrame({
        "Metric": ["Estimated Portfolio Value (12-month discounted cash flow)"],
        "Value": [portfolio_value]
    })

    

    # === 9. SAVE ALL KPI DATA TO COMMON EXCEL ===
    file_path = downloads_folder / "dlight_cleaned_data.xlsx"
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        contracts_df.to_excel(writer, sheet_name='Contracts_Cleaned', index=False)
        payments_df.to_excel(writer, sheet_name='Payments_Cleaned', index=False)
        trend_summary.to_excel(writer, sheet_name='Collection_Rate_Trend', index=False)
        trend_summary[['company', 'month', 'repayment_rate']].to_excel(writer, sheet_name='Repayment_Rate_Trend', index=False)
        par_pivot.to_excel(writer, sheet_name='PAR_Trend', index=False)
        vintage_analysis.to_excel(writer, sheet_name='Vintage_Analysis', index=False)
        product_summary.to_excel(writer, sheet_name='Product_Collection_Rate', index=False)
        FPD_rate_trend.to_excel(writer, sheet_name='FPD_Rate_Trend', index=False)
        cash_flow.to_excel(writer, sheet_name='Cash_Flow', index=False)
        projection_df.to_excel(writer, sheet_name='Cashflow_Projection', index=False)
        portfolio_value_df.to_excel(writer, sheet_name='Portfolio_Value', index=False)
        #monthly_writeoff.to_excel(writer, sheet_name='Write_Offs', index=False)
    




# Define your local repo path (update this to your actual local path)
    local_repo_path = Path(r"C:\Users\Finance Trainee\Documents\GitHub\credit-analyst-case")  # <-- CHANGE THIS!
    repo_excel_path = local_repo_path / "dlight_cleaned_data.xlsx"

    # Copy the Excel file to your repo folder
    shutil.copy(file_path, repo_excel_path)
    print(f"✅ Excel file copied to repo at {repo_excel_path}")

    # Automate git add, commit, and push
    try:
        subprocess.run(["git", "add", str(repo_excel_path)], cwd=local_repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update: latest dlight_cleaned_data.xlsx"], cwd=local_repo_path, check=True)
        subprocess.run(["git", "push"], cwd=local_repo_path, check=True)
        print("✅ Excel file committed and pushed to GitHub.")
    except Exception as e:
        print(f"⚠️ Git automation failed: {e}")






    # --- SEND EMAIL WITHOUT ATTACHMENT ---
    def send_email_without_attachment(subject, body, to_email):
        sender_email = "vaaketch@gmail.com"
        sender_password = ""  # Use an app password, not your main password
    
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email
        msg.set_content(body)
    
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"✅ Email sent to {to_email} with subject: {subject}")
    
        # Call the email function after saving the Excel file
    send_email_without_attachment(
        subject="Congratulations",
        body=(
            "Dear Vincent, we are pleased hahaha. am waiting for good tech news.\n\n"
            "Download the latest Excel file here:\n"
            "https://github.com/Vintaos/credit-analyst-case/raw/main/dlight_cleaned_data.xlsx"
        ),
        to_email="vaaketch@gmail.com"
    )







        













    end_time = time.time()
    print(f"✅ Data processing complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()


# === END OF SCRIPT ===
