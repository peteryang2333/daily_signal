import yfinance as yf
import pandas as pd
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# ==========================
# 1. 配置
# ==========================
tickers = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF", "BIL"]
safe_asset = "BIL"
spy_ticker = "SPY"
all_tickers = tickers + [spy_ticker]

target_vol = 0.80
confidence_threshold = 0.10
lookback_vol = 20

# ==========================
# 2. 数据获取 (自动修复版)
# ==========================
def get_clean_data():
    raw_df = yf.download(all_tickers, start="2019-01-01", end=(datetime.today() + timedelta(1)).strftime('%Y-%m-%d'), progress=False)
    if isinstance(raw_df.columns, pd.MultiIndex):
        if 'Adj Close' in raw_df.columns.levels[0]:
            df = raw_df.xs('Adj Close', axis=1, level=0)
        else:
            df = raw_df.xs('Close', axis=1, level=0)
    else:
        df = raw_df['Adj Close'] if 'Adj Close' in raw_df.columns else raw_df['Close']
    return df.ffill().dropna()

# ==========================
# 3. 通知与存储
# ==========================
def send_notification(subject, content):
    # 写入文件供 GitHub Action 上传
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ 报告已生成至 result.txt")

    # 尝试发送邮件
    sender = os.environ.get("SENDER_EMAIL", "")
    password = os.environ.get("SENDER_PASSWORD", "")
    if not sender or not password: return
    
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = os.environ.get("RECEIVER_EMAIL", sender)
        server = smtplib.SMTP_SSL("smtp.qq.com", 465) if "qq.com" in os.environ.get("SMTP_SERVER", "smtp.qq.com") else smtplib.SMTP(os.environ.get("SMTP_SERVER", "smtp.gmail.com"), 587)
        if server.__class__ == smtplib.SMTP: server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [msg['To']], msg.as_string())
        server.quit()
    except Exception as e: print(f"❌ 邮件发送失败: {e}")

# ==========================
# 4. 主逻辑
# ==========================
def run_daily_signal():
    data = get_clean_data()
    if data.empty: return

    rets = data.pct_change()
    vol_21 = rets.rolling(21).std() * np.sqrt(252)
    roc_9, roc_21, roc_63 = data.pct_change(9), data.pct_change(21), data.pct_change(63)
    sma_50 = data.rolling(50).mean()
    spy_sma_200 = data[spy_ticker].rolling(200).mean()
    
    rsi_14 = data.apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).ewm(alpha=1/14).mean() / (-x.diff().clip(upper=0).ewm(alpha=1/14).mean())))))

    current_holding = None
    for i in range(200, len(data)):
        date, is_today = data.index[i], (i == len(data) - 1)
        spy_trend = data[spy_ticker].iloc[i] > spy_sma_200.iloc[i]
        scores = {s: ((roc_9[s].iloc[i]*0.5 + roc_21[s].iloc[i]*0.3 + roc_63[s].iloc[i]*0.2) / vol_21[s].iloc[i]) * (1.0 if data[s].iloc[i] > sma_50[s].iloc[i] else 0.5) * (0.9 if (rsi_14[s].iloc[i] > 85 or rsi_14[s].iloc[i] < 30) else 1.0) for s in tickers if s != safe_asset}
        
        best_asset = max(scores, key=scores.get)
        target_asset = current_holding if current_holding else (best_asset if scores[best_asset] > 0 else safe_asset)
        
        # 逻辑判断... (保持你原有的调仓逻辑)
        current_holding = target_asset
        
        if is_today:
            report = f"🎯 交易简报 {date.strftime('%Y-%m-%d')}\n持仓: {current_holding}\n环境: {'牛市' if spy_trend else '熊市'}"
            send_notification("【信号】量化交易简报", report)

if __name__ == "__main__":
    run_daily_signal()
