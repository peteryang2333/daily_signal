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

# ==========================
# 2. 核心数据获取与修复 (重点修复区)
# ==========================
def get_clean_data():
    raw_df = yf.download(all_tickers, start="2019-01-01", end=(datetime.today() + timedelta(1)).strftime('%Y-%m-%d'), progress=False)
    
    # 自动处理多层索引：尝试寻找 'Adj Close'，如果找不到则尝试 'Close'
    if isinstance(raw_df.columns, pd.MultiIndex):
        # 尝试通过 xs 提取列，或者直接按层级提取
        if 'Adj Close' in raw_df.columns.levels[0]:
            df = raw_df.xs('Adj Close', axis=1, level=0)
        else:
            df = raw_df.xs('Close', axis=1, level=0)
    else:
        # 单层索引处理
        if 'Adj Close' in raw_df.columns:
            df = raw_df['Adj Close']
        else:
            df = raw_df['Close']
    return df.ffill().dropna()

# ==========================
# 3. 原有逻辑保持不变
# ==========================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", SENDER_EMAIL)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = 465 if "qq.com" in SMTP_SERVER else 587 

def send_notification(subject, content):
    if not SENDER_EMAIL or not SENDER_PASSWORD: return
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) if SMTP_PORT == 465 else smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        if SMTP_PORT != 465: server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        server.quit()
    except Exception as e: print(f"❌ 邮件发送失败: {e}")

def run_daily_signal():
    data = get_clean_data()
    if data.empty: return

    # 计算指标
    rets = data.pct_change()
    vol_21 = rets.rolling(window=21).std() * np.sqrt(252)
    roc_9, roc_21, roc_63 = data.pct_change(9), data.pct_change(21), data.pct_change(63)
    sma_50 = data.rolling(50).mean()
    spy_sma_200 = data[spy_ticker].rolling(200).mean()

    # 循环推演逻辑 (保留你原本的业务代码)
    current_holding = None
    for i in range(200, len(data)):
        # ... (此处保持你原来代码中的 3. 状态推演逻辑不变) ...
        # 注意：这里面的 data[s] 和 rets[s] 现在都能直接工作了
        pass

if __name__ == "__main__":
    run_daily_signal()
