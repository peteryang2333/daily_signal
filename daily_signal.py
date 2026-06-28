import yfinance as yf
import pandas as pd
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# ==========================
# 1. 策略全局参数配置
# ==========================
tickers = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF", "BIL"]
safe_asset = "BIL"
spy_ticker = "SPY"
all_tickers = tickers + [spy_ticker]

target_vol = 0.80
confidence_threshold = 0.10
lookback_vol = 20

# ==========================
# 2. 邮件通知配置
# ==========================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", SENDER_EMAIL)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = 465 if "qq.com" in SMTP_SERVER else 587 

def send_notification(subject, content):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("⚠️ 未检测到邮箱配置，跳过邮件发送。")
        return
        
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        server.quit()
        print("✅ 策略执行完毕，邮件通知已发送！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

def run_daily_signal():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在获取最新行情数据...")
    
    # 修复逻辑：使用 auto_adjust=True 并处理可能的多层索引
    raw_data = yf.download(all_tickers, start="2019-01-01", end=(datetime.today() + timedelta(1)).strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Close']
    else:
        data = raw_data
        
    data = data.ffill().dropna()

    if data.empty:
        print("❌ 数据下载失败。")
        return

    # 技术指标计算
    rets = data.pct_change()
    vol_21 = rets.rolling(window=21).std() * np.sqrt(252) 
    vol_21 = vol_21.replace(0, 1.0) 

    roc_9 = data.pct_change(periods=9)
    roc_21 = data.pct_change(periods=21)
    roc_63 = data.pct_change(periods=63)
    sma_50 = data.rolling(window=50).mean()
    spy_sma_200 = data[spy_ticker].rolling(window=200).mean()

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    rsi_14 = data.apply(calculate_rsi)

    # 状态推演
    current_holding = None
    for i in range(200, len(data)):
        date = data.index[i]
        is_today = (i == len(data) - 1)
        spy_trend = data[spy_ticker].iloc[i] > spy_sma_200.iloc[i]
        scores = {}
        
        for s in tickers:
            if s == safe_asset: continue
            fast, med, slow = roc_9[s].iloc[i], roc_21[s].iloc[i], roc_63[s].iloc[i]
            v, r, price, sma = vol_21[s].iloc[i], rsi_14[s].iloc[i], data[s].iloc[i], sma_50[s].iloc[i]
            risk_adj_mom = ((fast * 0.5) + (med * 0.3) + (slow * 0.2)) / v
            trend_score = 1.0 if price > sma else 0.5
            rsi_penalty = 0.9 if (r > 85 or r < 30) else 1.0
            scores[s] = risk_adj_mom * trend_score * rsi_penalty
        
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_asset, best_score = sorted_assets[0]
        
        prev_holding = current_holding
        target_asset = current_holding
        current_score = scores.get(current_holding, -999) if current_holding else -999
        
        if current_holding is None:
            target_asset = best_asset if best_score > 0 else safe_asset
        else:
            if current_holding == safe_asset:
                if best_score > 0.02: target_asset = best_asset
            else:
                if best_score > current_score * (1 + confidence_threshold): target_asset = best_asset
                elif current_score < -0.02: target_asset = safe_asset
                    
        if not spy_trend and target_asset != safe_asset:
            uup_score, target_score = scores.get("UUP", -999), scores.get(target_asset, -999)
            if uup_score > 0 and uup_score > target_score: target_asset = "UUP"
            elif target_score < 0: target_asset = safe_asset

        curr_vol = rets[target_asset].iloc[i-lookback_vol+1:i+1].std() * np.sqrt(252) if (target_asset and target_asset != safe_asset) else 0.0
        target_weight = min(1.0, target_vol / curr_vol) if curr_vol > 0 else 1.0
        target_weight = 1.0 if target_asset == safe_asset else target_weight
        
        current_holding = target_asset
        safe_weight = max(0.0, 1.0 - target_weight) if target_asset != safe_asset else 1.0

        if is_today:
            report = [f"🎯 评估日期: {date.strftime('%Y-%m-%d')}", f"环境: {'牛市' if spy_trend else '熊市'}", 
                      f"决断: {'维持 ' + target_asset if prev_holding == target_asset else '切换至 ' + target_asset}"]
            send_notification("【信号】量化交易简报", "\n".join(report))

if __name__ == "__main__":
    run_daily_signal()
