import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import time
import shutil
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# ==========================
# 1. 全局配置参数 (对齐 QuantConnect)
# ==========================
tickers = ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF", "BIL"]
safe_asset = "BIL"
spy_ticker = "SPY"
all_tickers = tickers + [spy_ticker]

target_vol = 0.80               # 目标年化波动率
confidence_threshold = 0.10     # 调仓摩擦阈值 (10%)
lookback_vol = 20               # 仓位计算波动率回溯期

# ==========================
# 2. 强健的数据获取模块
# ==========================
def _clear_yf_cache():
    """清理 yfinance 的 SQLite 缓存，规避 'database is locked' 问题"""
    for pattern in [
        os.path.expanduser("~/.cache/py-yfinance"),
        os.path.join(os.environ.get("XDG_CACHE_HOME", ""), "py-yfinance"),
    ]:
        if pattern and os.path.isdir(pattern):
            try:
                shutil.rmtree(pattern)
                print(f"🧹 已清理 yfinance 缓存: {pattern}")
            except OSError:
                pass


def get_clean_data(max_retries=3):
    # 获取比200天更多的数据以满足长期均线和历史模拟
    # threads=False 规避 yfinance 多线程共享 SQLite 缓存导致的 'database is locked'
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_df = yf.download(
                all_tickers,
                start="2019-01-01",
                end=(datetime.today() + timedelta(1)).strftime('%Y-%m-%d'),
                progress=False,
                threads=False,
            )

            if isinstance(raw_df.columns, pd.MultiIndex):
                if 'Adj Close' in raw_df.columns.levels[0]:
                    df = raw_df.xs('Adj Close', axis=1, level=0)
                else:
                    df = raw_df.xs('Close', axis=1, level=0)
            else:
                df = raw_df['Adj Close'] if 'Adj Close' in raw_df.columns else raw_df['Close']

            df = df.ffill().dropna()

            # 校验：所有标的都要有数据，缺列或空表都算失败（避免部分下载失败被吞掉）
            missing = [t for t in all_tickers if t not in df.columns]
            if df.empty or missing:
                raise RuntimeError(f"数据不完整，缺失: {missing or '全部'}")

            return df
        except Exception as e:
            last_err = e
            print(f"⚠️ 第 {attempt}/{max_retries} 次下载失败: {e}")
            _clear_yf_cache()
            if attempt < max_retries:
                wait = 15 * attempt
                print(f"⏳ {wait}s 后重试…")
                time.sleep(wait)

    print(f"❌ 重试 {max_retries} 次后仍失败: {last_err}")
    return pd.DataFrame()

# ==========================
# 3. 结果保存与邮件通知模块
# ==========================
def send_notification(subject, content):
    # 1. 强制写入 txt 文件供 GitHub Artifacts 保存下载
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✅ 报告已生成并保存至 result.txt")
    print(content)  # 在 GitHub Actions 日志中直接打印，方便免下载查看

    # 2. 尝试发送邮件
    sender = os.environ.get("SENDER_EMAIL", "")
    password = os.environ.get("SENDER_PASSWORD", "")
    if not sender or not password: 
        print("⚠️ 未配置邮箱机密，跳过邮件发送。")
        return
        
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = os.environ.get("RECEIVER_EMAIL", sender)
        server_url = os.environ.get("SMTP_SERVER", "smtp.qq.com")
        
        if "qq.com" in server_url:
            server = smtplib.SMTP_SSL(server_url, 465)
        else:
            server = smtplib.SMTP(server_url, 587)
            server.starttls()
            
        server.login(sender, password)
        server.sendmail(sender, [msg['To']], msg.as_string())
        server.quit()
        print("✅ 邮件通知已成功发送！")
    except Exception as e: 
        print(f"❌ 邮件发送失败: {e}")

# ==========================
# 4. 指标计算与核心策略逻辑 (1:1 还原 QC)
# ==========================
def calculate_rsi(series, period=14):
    """标准的 Wilder's RSI 计算法，与 QC 默认对齐"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_daily_signal():
    data = get_clean_data()
    if data.empty: 
        print("❌ 数据获取失败（已重试仍不可用），以非零码退出以便 Actions 标记失败")
        sys.exit(1)

    # --- A. 向量化计算所有技术指标 ---
    rets = data.pct_change()
    vol_21 = rets.rolling(window=21).std() * np.sqrt(252)
    roc_9 = data.pct_change(periods=9)
    roc_21 = data.pct_change(periods=21)
    roc_63 = data.pct_change(periods=63)
    sma_50 = data.rolling(window=50).mean()
    spy_sma_200 = data[spy_ticker].rolling(window=200).mean()
    rsi_14 = data.apply(calculate_rsi)

    # --- B. 历史重演状态机 ---
    current_holding = None
    
    # 必须从第 200 天开始循环，确保指标充分预热且状态机连续推演
    for i in range(200, len(data)):
        date = data.index[i]
        is_today = (i == len(data) - 1)
        spy_trend = data[spy_ticker].iloc[i] > spy_sma_200.iloc[i]
        
        scores = {}
        for s in tickers:
            if s == safe_asset: continue
            
            # 提取当日指标
            fast, med, slow = roc_9[s].iloc[i], roc_21[s].iloc[i], roc_63[s].iloc[i]
            vol = vol_21[s].iloc[i] if pd.notna(vol_21[s].iloc[i]) and vol_21[s].iloc[i] > 0 else 1.0
            rsi = rsi_14[s].iloc[i]
            price = data[s].iloc[i]
            sma = sma_50[s].iloc[i]
            
            # 动能打分 (Risk-Adjusted Momentum)
            weighted_mom = (fast * 0.5) + (med * 0.3) + (slow * 0.2)
            risk_adj_mom = weighted_mom / vol
            
            # 趋势与震荡惩罚
            trend_score = 1.0 if price > sma else 0.5
            rsi_penalty = 0.9 if (rsi > 85 or rsi < 30) else 1.0
            
            # 最终得分
            scores[s] = risk_adj_mom * trend_score * rsi_penalty
            
        # 排序寻找最强资产
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_asset, best_score = sorted_assets[0]
        
        prev_holding = current_holding
        target_asset = current_holding
        current_score = scores.get(current_holding, -999) if current_holding else -999
        
        # --- C. 核心决策树 (完全对应 QC 的 rebalance) ---
        
        # 1. 基础摩擦与切换逻辑
        if current_holding is None:
            target_asset = best_asset if best_score > 0 else safe_asset
        else:
            if current_holding == safe_asset:
                if best_score > 0.02: 
                    target_asset = best_asset
            else:
                if best_score > current_score * (1 + confidence_threshold): 
                    target_asset = best_asset
                elif current_score < -0.02: 
                    target_asset = safe_asset

        # 2. 熊市防守逻辑 (Spy Trend)
        if not spy_trend and target_asset != safe_asset:
            uup_score = scores.get("UUP", -999)
            target_score = scores.get(target_asset, -999)
            
            if uup_score > 0 and uup_score > target_score:
                target_asset = "UUP"
            elif target_score < 0:
                target_asset = safe_asset

        # 3. 仓位比例计算 (Target Volatility Sizing)
        if target_asset and target_asset != safe_asset:
            # 提取过去 20 天的日收益率来计算近期实际波动率
            window_rets = rets[target_asset].iloc[i - lookback_vol + 1 : i + 1]
            curr_vol = window_rets.std() * np.sqrt(252)
            target_weight = min(1.0, target_vol / curr_vol) if curr_vol > 0 else 1.0
        elif target_asset == safe_asset:
            target_weight = 1.0
        else:
            target_weight = 0.0

        current_holding = target_asset
        safe_weight = max(0.0, 1.0 - target_weight) if target_asset != safe_asset else 1.0

        # --- D. 仅在最后一天(今天)生成详尽简报 ---
        if is_today:
            # 判断今日具体动作
            if prev_holding == target_asset:
                action_str = f"【维持持有】 {target_asset}"
            else:
                action_str = f"【执行调仓】 卖出 {prev_holding if prev_holding else '无'} -> 买入 {target_asset}"

            # 格式化动能打分榜 (TOP 3)
            score_lines = "\n".join([f"  🔸 {idx+1}. {s:5} | 综合得分: {sc:7.3f} | 21日波动: {vol_21[s].iloc[i]*100:5.1f}% | RSI: {rsi_14[s].iloc[i]:4.1f}" for idx, (s, sc) in enumerate(sorted_assets[:3])])
            
            report = (
                f"============================================\n"
                f" 🎯 The Omniscient Paradox | 实盘交易指令\n"
                f" 📅 评估日期: {date.strftime('%Y-%m-%d')} (盘尾)\n"
                f"============================================\n\n"
                f"📊 【宏观环境与状态】\n"
                f"  • 大盘环境: {'🟢 牛市 (SPY > 200均线)' if spy_trend else '🔴 熊市预警 (SPY < 200均线，触发防守机制)'}\n"
                f"  • 昨日持仓: {prev_holding if prev_holding else '空仓'}\n\n"
                f"⚡ 【今日操作指令】\n"
                f"  • 指令动作: {action_str}\n"
                f"  • 今日目标: {target_asset}\n"
                f"  • 仓位分配: {target_weight*100:5.1f}% 资金买入 {target_asset}\n"
                f"              {safe_weight*100:5.1f}% 资金买入 {safe_asset} (闲置吃息)\n\n"
                f"🏆 【资产动能打分 TOP 3】\n"
                f"{score_lines}\n\n"
                f"============================================"
            )
            
            # 调用发送/保存模块
            send_notification(f"【量化信号】{date.strftime('%Y-%m-%d')} 调仓指令", report)

if __name__ == "__main__":
    run_daily_signal()
