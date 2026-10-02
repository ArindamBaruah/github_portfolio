import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
 
from gbm_factory import fit_best_gbm, backend_name
from garch_module import fit_garch11, garch_sigma2_series, garch_one_step_forecast
from nifty_pipeline_v2 import CONFIG, load_and_merge, engineer_features
 
N_DAYS = 365  # size of the report window (trading days), counted back from CONFIG['end_date']
 
 
def build_report(df, data, feat_cols, cfg, n_days=N_DAYS):
    n = len(data)
    train_win, purge = cfg['train_window'], cfg['purge']
 
    if n_days >= n:
        raise ValueError(f"Only {n} usable feature rows available, cannot report {n_days} days.")
 
    test_idx = np.arange(n - n_days, n)
    train_end = test_idx[0] - purge
    train_start = max(0, train_end - train_win)
    if train_end - train_start < 100:
        raise ValueError("Not enough history before the report window to train a model "
                          f"(only {train_end - train_start} rows). Reduce n_days or train_window.")
    tr_idx = np.arange(train_start, train_end)
 
    print(f"[report] Training window: rows {train_start}-{train_end - 1} "
          f"({data['Date'].iloc[train_start].date()} -> {data['Date'].iloc[train_end - 1].date()}), "
          f"{len(tr_idx)} rows")
    print(f"[report] Test window ({n_days} days): "
          f"{data['Date'].iloc[test_idx[0]].date()} -> {data['Date'].iloc[test_idx[-1]].date()}")
 
    Xtr, Xte = data.iloc[tr_idx][feat_cols].values, data.iloc[test_idx][feat_cols].values
    y_dir_tr = data.iloc[tr_idx]['target_dir'].values
    y_dH_tr = data.iloc[tr_idx]['target_dH'].values
    y_dL_tr = data.iloc[tr_idx]['target_dL'].values
    y_gap_tr = data.iloc[tr_idx]['target_gap'].values
 
    # ---- GARCH(1,1) fit on TRAIN returns only; roll sigma2 forward through the test window
    #      using realized returns available at t-1 (no leakage) ----
    r_pct_tr = data.iloc[tr_idx]['daily_return'].values * 100
    omega, alpha, beta = fit_garch11(r_pct_tr)
    sigma2_tr = garch_sigma2_series(r_pct_tr, omega, alpha, beta)
    r_pct_te = data.iloc[test_idx]['daily_return'].values * 100
    sigma2_te = np.zeros(len(test_idx))
    last_sigma2, last_r = sigma2_tr[-1], r_pct_tr[-1]
    for i in range(len(test_idx)):
        s2 = garch_one_step_forecast(last_r, last_sigma2, omega, alpha, beta)
        sigma2_te[i] = s2
        last_sigma2, last_r = s2, r_pct_te[i]
 
    Xtr_garch = np.column_stack([Xtr, sigma2_tr])
    Xte_garch = np.column_stack([Xte, sigma2_te])
 
    # ---- Fit the model set ONCE on the training window ----
    clf, _, backend = fit_best_gbm(Xtr, y_dir_tr, task='clf')
    regH, _, _ = fit_best_gbm(Xtr_garch, y_dH_tr, task='reg')
    regL, _, _ = fit_best_gbm(Xtr_garch, y_dL_tr, task='reg')
    regGap, _, _ = fit_best_gbm(Xtr, y_gap_tr, task='reg')
 
    p_up = clf.predict_proba(Xte)[:, 1]
    pred_dH = regH.predict(Xte_garch)
    pred_dL = regL.predict(Xte_garch)
    pred_gap = regGap.predict(Xte)
 
    # ---- Expected Close: prob-weighted trailing up/down intraday return, computed from
    #      TRAIN data only (last 250 train rows), exactly as in final_forecast() ----
    train_dates = data.iloc[tr_idx]['Date']
    df_tr = df[df['Date'].isin(train_dates)]
    intraday_tr = (np.log(df_tr['C'] / df_tr['O'])).tail(250)
    up_avg = intraday_tr[intraday_tr > 0].mean()
    down_avg = intraday_tr[intraday_tr <= 0].mean()
 
    # ---- Actual OHLC + anchor (prior actual close) per test date, pulled from df ----
    df_idx = df.set_index('Date')
    test_dates = data.iloc[test_idx]['Date'].values
 
    rows = []
    for i, date in enumerate(test_dates):
        date = pd.Timestamp(date)
        anchor_close = df_idx.loc[date, 'prev_C']          # actual close AS OF the day being predicted
        actual_O = df_idx.loc[date, 'O']
        actual_H = df_idx.loc[date, 'H']
        actual_L = df_idx.loc[date, 'L']
        actual_next_close = df_idx.loc[date, 'C']           # actual realized close for that day
 
        O_pred = anchor_close * np.exp(pred_gap[i])
        H_pred = O_pred * np.exp(pred_dH[i])
        L_pred = O_pred * np.exp(pred_dL[i])
        exp_intraday_ret = p_up[i] * up_avg + (1 - p_up[i]) * down_avg
        C_pred = O_pred * np.exp(exp_intraday_ret)
 
        predicted_up = C_pred > anchor_close
        predicted_down = C_pred < anchor_close
        actual_up = actual_next_close > anchor_close
        actual_down = actual_next_close < anchor_close
 
        correct = "Yes" if (predicted_down and actual_down) or (predicted_up and actual_up) else "No"
 
        if predicted_up:
            confidence = "High" if p_up[i] > 0.55 else "Low"
        elif predicted_down:
            confidence = "High" if p_up[i] < 0.45 else "Low"
        else:
            confidence = "Low"
 
        rows.append(dict(
            Date=date.date(),
            Anchor_Close=round(anchor_close, 2),
            Predicted_Open=round(O_pred, 2),
            Predicted_High=round(H_pred, 2),
            Predicted_Low=round(L_pred, 2),
            Predicted_Close=round(C_pred, 2),
            Actual_Open=round(actual_O, 2),
            Actual_High=round(actual_H, 2),
            Actual_Low=round(actual_L, 2),
            Actual_Close=round(actual_next_close, 2),
            P_Close_gt_Open=round(p_up[i], 4),
            Predicted_Direction="Up" if predicted_up else ("Down" if predicted_down else "Flat"),
            Actual_Direction="Up" if actual_up else ("Down" if actual_down else "Flat"),
            Correct=correct,
            Confidence=confidence,
        ))
 
    report = pd.DataFrame(rows)
    return report, dict(backend=backend, omega=omega, alpha=alpha, beta=beta,
                         up_avg=up_avg, down_avg=down_avg,
                         train_start_date=data.iloc[train_start]['Date'].date(),
                         train_end_date=data.iloc[train_end - 1]['Date'].date())
 
 
def summarize(report):
    n = len(report)
    acc = (report['Correct'] == 'Yes').mean()
    print(f"\n[report] {n} days | directional accuracy: {acc:.1%}")
    for conf in ['High', 'Low']:
        sub = report[report['Confidence'] == conf]
        if len(sub):
            sub_acc = (sub['Correct'] == 'Yes').mean()
            print(f"  Confidence={conf:<5} n={len(sub):3d}  accuracy={sub_acc:.1%}")
 
 
def write_xlsx(report, meta, cfg, out_path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Prediction Report'
 
    header_font = Font(name='Arial', bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='1F4E78')
    body_font = Font(name='Arial', size=10)
    center = Alignment(horizontal='center')
 
    yes_fill = PatternFill('solid', fgColor='C6EFCE')  # green
    no_fill = PatternFill('solid', fgColor='FFC7CE')   # red
    high_fill = PatternFill('solid', fgColor='FFEB9C')  # amber
    yes_font = Font(name='Arial', size=10, color='006100')
    no_font = Font(name='Arial', size=10, color='9C0006')
 
    # ---- Title / meta block ----
    ws['A1'] = 'Nifty 60-Day Prediction Report'
    ws['A1'].font = Font(name='Arial', bold=True, size=14)
    ws['A2'] = f"Cutoff date: {cfg.get('end_date')}   |   Report window: {report['Date'].iloc[0]} -> {report['Date'].iloc[-1]}"
    ws['A2'].font = Font(name='Arial', italic=True, size=10)
    ws['A3'] = (f"Model trained once on {meta['train_start_date']} -> {meta['train_end_date']} "
                f"({cfg['train_window']} rows, GBM backend: {meta['backend']}); applied unchanged to each test day.")
    ws['A3'].font = Font(name='Arial', italic=True, size=9)
    ws.merge_cells('A1:O1')
    ws.merge_cells('A2:O2')
    ws.merge_cells('A3:O3')
 
    header_row = 5
    cols = list(report.columns)
    for j, col in enumerate(cols, start=1):
        c = ws.cell(row=header_row, column=j, value=col.replace('_', ' '))
        c.font = header_font
        c.fill = header_fill
        c.alignment = center
 
    for i, row in report.iterrows():
        r = header_row + 1 + i
        for j, col in enumerate(cols, start=1):
            val = row[col]
            c = ws.cell(row=r, column=j, value=val)
            c.font = body_font
            c.alignment = center
            if col == 'P_Close_gt_Open':
                c.number_format = '0.0%'
            if col == 'Correct':
                if val == 'Yes':
                    c.fill, c.font = yes_fill, yes_font
                else:
                    c.fill, c.font = no_fill, no_font
            if col == 'Confidence' and val == 'High':
                c.fill = high_fill
 
    # ---- Summary block ----
    sum_row = header_row + len(report) + 2
    n = len(report)
    acc = (report['Correct'] == 'Yes').mean()
    ws.cell(row=sum_row, column=1, value='Summary').font = Font(name='Arial', bold=True, size=12)
    ws.cell(row=sum_row + 1, column=1, value=f"Overall directional accuracy: {acc:.1%} ({(report['Correct']=='Yes').sum()}/{n})").font = body_font
    r = sum_row + 2
    for conf in ['High', 'Low']:
        sub = report[report['Confidence'] == conf]
        if len(sub):
            sub_acc = (sub['Correct'] == 'Yes').mean()
            ws.cell(row=r, column=1,
                    value=f"Confidence={conf}: n={len(sub)}, accuracy={sub_acc:.1%}").font = body_font
            r += 1
    ws.cell(row=r + 1, column=1,
            value=("Note: Predicted_Close is derived using TODAY's GIFT Nifty / US overnight "
                   "columns as a stand-in for the actual forecast day's pre-open prints (same "
                   "convention as nifty_pipeline_v2.py's final_forecast()); this is a like-for-like "
                   "backtest of the pipeline's methodology, not a live point-in-time forecast archive.")
            ).font = Font(name='Arial', italic=True, size=9)
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=15)
    ws.cell(row=r + 1, column=1).alignment = Alignment(wrap_text=True)
 
    widths = [12, 13, 15, 14, 13, 15, 12, 12, 11, 13, 15, 18, 15, 9, 11]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
 
    wb.save(out_path)
 
 
if __name__ == '__main__':
    cfg = CONFIG
    print(f"[gbm backend] {backend_name()}")
 
    df = load_and_merge(cfg)
    data, feat_cols = engineer_features(df)
    print(f"[features] {len(feat_cols)} features, {len(data)} usable rows, "
          f"{data['Date'].min().date()} -> {data['Date'].max().date()}")
 
    report, meta = build_report(df, data, feat_cols, cfg, n_days=N_DAYS)
    summarize(report)
 
    out_csv = 'Nifty_60Day_Prediction_Report.csv'
    report.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
 
    out_xlsx = 'Nifty_60Day_Prediction_Report.xlsx'
    write_xlsx(report, meta, cfg, out_xlsx)
    print(f"[saved] {out_xlsx}")