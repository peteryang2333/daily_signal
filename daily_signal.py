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
# 2. 邮件通知配置 (通过环境变量获取)
# ==========================
# GitHub Actions 会通过 env 注入这些变量，确保密码安全
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", SENDER_EMAIL) # 默认发给自己
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.qq.com") # 默认使用QQ邮箱服务器
# QQ邮箱通常使用465端口(SSL)，Gmail通常使用587端口(TLS)
SMTP_PORT = 465 if "qq.com" in SMTP_SERVER else 587 

def send_notification(subject, content):
    """发送邮件通知的最简封装"""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("⚠️ 未检测到邮箱配置(SENDER_EMAIL/SENDER_PASSWORD)，跳过邮件发送。")
        return
        
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        
        # 根据端口选择不同的安全连接方式
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        server.quit()
        print("✅ 策略执行完毕，邮件通知已成功发送！")
    except Exception as e:
        print(f"❌ 邮件发送失败，错误信息: {e}")

def run_daily_signal():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在获取最新行情数据...")
    # 获取数据，多获取几天确保均线计算充足
    data = yf.download(all_tickers, start="2019-01-01", end=(datetime.today() + timedelta(1)).strftime('%Y-%m-%d'), progress=False)['Adj Close']
    data = data.ffill().dropna()

    if data.empty:
        error_msg = "❌ 数据下载失败或今天是节假日，无法生成信号。"
        print(error_msg)
        send_notification("【策略警告】数据获取失败", error_msg)
        return

    # ==========================
    # 2. 技术指标计算 (向量化)
    # ==========================
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

    # ==========================
    # 3. 状态推演 (静默遍历历史)
    # ==========================
    current_holding = None # 初始状态
    
    # 从第200天开始推演，一直推演到“今天”
    for i in range(200, len(data)):
        date = data.index[i]
        is_today = (i == len(data) - 1) # 判断是否为最后一条数据（即今天）
        
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
        best_asset, best_score = sorted_assets[0][0], sorted_assets[0][1]
        
        prev_holding = current_holding
        target_asset = current_holding
        current_score = scores.get(current_holding, -999) if current_holding else -999
        
        # 10% 摩擦门槛逻辑
        if current_holding is None:
            target_asset = best_asset if best_score > 0 else safe_asset
        else:
            if current_holding == safe_asset:
                if best_score > 0.02: target_asset = best_asset
            else:
                if best_score > current_score * (1 + confidence_threshold): target_asset = best_asset
                elif current_score < -0.02: target_asset = safe_asset
                    
        # 熊市防守逻辑
        if not spy_trend and target_asset != safe_asset:
            uup_score, target_score = scores.get("UUP", -999), scores.get(target_asset, -999)
            if uup_score > 0 and uup_score > target_score: target_asset = "UUP"
            elif target_score < 0: target_asset = safe_asset

        # 波动率控仓计算
        curr_vol = 0.0
        if target_asset and target_asset != safe_asset:
            curr_vol = rets[target_asset].iloc[i-lookback_vol+1:i+1].std() * np.sqrt(252)
            target_weight = min(1.0, target_vol / curr_vol) if curr_vol > 0 else 1.0
        elif target_asset == safe_asset:
            target_weight = 1.0
        else:
            target_weight = 0.0
            
        current_holding = target_asset
        safe_weight = max(0.0, 1.0 - target_weight) if target_asset != safe_asset else 1.0

        # ==========================
        # 4. 只在今天（最新日）输出操作简报并发送通知
        # ==========================
        if is_today:
            # 使用列表收集所有日志，方便一次性发送邮件
            report_lines = []
            def log_msg(text):
                print(text)
                report_lines.append(text)

            log_msg("\n" + "="*60)
            log_msg(f" 🎯 The Omniscient Paradox | 实盘交易指令室")
            log_msg(f" 📅 评估日期: {date.strftime('%Y-%m-%d')} (盘尾)")
            log_msg("="*60)
            
            market_state = "🟢 牛市 (均线之上)" if spy_trend else "🔴 熊市预警 (均线之下，触发防御)"
            log_msg(f"[宏观环境] SPY 状态: {market_state}")
            log_msg(f"[昨日持仓] {prev_holding if prev_holding else '空仓'}")
            log_msg("-" * 60)
            
            log_msg("[各标的动能打分榜] (前3名):")
            for rank, (ast, sc) in enumerate(sorted_assets[:3]):
                log_msg(f"  {rank+1}. {ast:4} | 得分: {sc:7.3f} | 波动率: {vol_21[ast].iloc[i]*100:5.1f}% | RSI: {rsi_14[ast].iloc[i]:5.1f}")
            log_msg("-" * 60)
            
            # 判断是否需要动作
            if prev_holding == target_asset:
                log_msg(f"🔔 【今日决断】: 维持当前持仓，无需轮动！")
                log_msg(f"   👉 继续持有: {target_asset}")
                if target_asset != safe_asset:
                    log_msg(f"   👉 仓位配置: {target_asset} 分配 {target_weight*100:.1f}%, 现金/国债(BIL) 分配 {safe_weight*100:.1f}%")
            else:
                log_msg(f"🚨 【今日决断】: 触发调仓信号，请执行交易！")
                log_msg(f"   🛒 卖出指令: 清仓 {prev_holding if prev_holding else '无'}")
                log_msg(f"   🛒 买入指令: 建仓 {target_asset}")
                if target_asset != safe_asset:
                    log_msg(f"   👉 仓位配置: 将总资金的 {target_weight*100:.1f}% 买入 {target_asset}")
                    if safe_weight > 0.05:
                        log_msg(f"   👉 闲置处理: 将剩余 {safe_weight*100:.1f}% 资金买入 BIL 吃利息")
            log_msg("="*60 + "\n")
            
            # 打包日志发送邮件通知
            email_subject = f"【量化信号】{date.strftime('%Y-%m-%d')} 调仓指令"
            email_body = "\n".join(report_lines)
            send_notification(email_subject, email_body)

if __name__ == "__main__":
    run_daily_signal()
