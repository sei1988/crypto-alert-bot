#!/usr/bin/env python3
"""
Crypto Alert Bot — BTC, SOL, ETH
Στρατηγικές: Breakout, EMA Crossover, RSI, Volume Spike, Pivot Points, Candlestick Patterns
Timeframes: 15m, 1H, 4H
"""

import asyncio
import logging
import time
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SYMBOLS, TIMEFRAMES, CHECK_INTERVAL_SECONDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({'enableRateLimit': True})


# ─── TELEGRAM ────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info(f"✅ Telegram sent: {message[:60]}...")
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")


# ─── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol} {timeframe}: {e}")
        return pd.DataFrame()


# ─── INDICATORS ──────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMAs
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema100'] = ta.ema(df['close'], length=100)
    df['ema200'] = ta.ema(df['close'], length=200)

    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)

    # Volume MA
    df['vol_ma20'] = df['volume'].rolling(20).mean()

    # ATR for breakout threshold
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = atr

    # Pivot Points (last completed session)
    df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
    df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
    df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
    df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))

    return df


# ─── STRATEGIES ──────────────────────────────────────────────────────────────

def check_ema_crossover(df: pd.DataFrame, symbol: str, tf: str) -> list:
    alerts = []
    if len(df) < 3:
        return alerts

    prev, curr = df.iloc[-2], df.iloc[-1]

    # Bullish crossover: EMA20 crosses above EMA50
    if prev['ema20'] <= prev['ema50'] and curr['ema20'] > curr['ema50']:
        alerts.append({
            'type': '📈 EMA CROSSOVER — BULLISH',
            'detail': f"EMA20 ({curr['ema20']:.1f}) ⬆️ πάνω από EMA50 ({curr['ema50']:.1f})",
            'bias': 'LONG',
            'price': curr['close']
        })

    # Bearish crossover: EMA20 crosses below EMA50
    elif prev['ema20'] >= prev['ema50'] and curr['ema20'] < curr['ema50']:
        alerts.append({
            'type': '📉 EMA CROSSOVER — BEARISH',
            'detail': f"EMA20 ({curr['ema20']:.1f}) ⬇️ κάτω από EMA50 ({curr['ema50']:.1f})",
            'bias': 'SHORT',
            'price': curr['close']
        })

    return alerts


def check_rsi(df: pd.DataFrame, symbol: str, tf: str) -> list:
    alerts = []
    if len(df) < 2:
        return alerts

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = curr['rsi']

    if prev['rsi'] <= 30 and rsi > 30:
        alerts.append({
            'type': '🟢 RSI OVERSOLD EXIT',
            'detail': f"RSI ανέβηκε πάνω από 30 → {rsi:.1f} (Potential LONG)",
            'bias': 'LONG',
            'price': curr['close']
        })
    elif rsi < 25:
        alerts.append({
            'type': '🔴 RSI EXTREME OVERSOLD',
            'detail': f"RSI = {rsi:.1f} — Ακραία oversold ζώνη",
            'bias': 'LONG WATCH',
            'price': curr['close']
        })
    elif prev['rsi'] >= 70 and rsi < 70:
        alerts.append({
            'type': '🔴 RSI OVERBOUGHT EXIT',
            'detail': f"RSI έπεσε κάτω από 70 → {rsi:.1f} (Potential SHORT)",
            'bias': 'SHORT',
            'price': curr['close']
        })
    elif rsi > 75:
        alerts.append({
            'type': '⚠️ RSI EXTREME OVERBOUGHT',
            'detail': f"RSI = {rsi:.1f} — Ακραία overbought ζώνη",
            'bias': 'SHORT WATCH',
            'price': curr['close']
        })

    return alerts


def check_breakout(df: pd.DataFrame, symbol: str, tf: str, lookback: int = 20) -> list:
    alerts = []
    if len(df) < lookback + 2:
        return alerts

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    window = df.iloc[-(lookback+1):-1]

    highest = window['high'].max()
    lowest = window['low'].min()
    atr = curr['atr'] if not pd.isna(curr['atr']) else 0

    # Bullish breakout
    if prev['close'] <= highest and curr['close'] > highest + (atr * 0.1):
        alerts.append({
            'type': '🚀 BREAKOUT — BULLISH',
            'detail': f"Έσπασε πάνω από {lookback}-candle high: {highest:.1f}",
            'bias': 'LONG',
            'price': curr['close']
        })

    # Bearish breakdown
    elif prev['close'] >= lowest and curr['close'] < lowest - (atr * 0.1):
        alerts.append({
            'type': '💥 BREAKDOWN — BEARISH',
            'detail': f"Έσπασε κάτω από {lookback}-candle low: {lowest:.1f}",
            'bias': 'SHORT',
            'price': curr['close']
        })

    return alerts


def check_volume_spike(df: pd.DataFrame, symbol: str, tf: str, multiplier: float = 2.5) -> list:
    alerts = []
    if len(df) < 21:
        return alerts

    curr = df.iloc[-1]
    vol = curr['volume']
    vol_ma = curr['vol_ma20']

    if pd.isna(vol_ma) or vol_ma == 0:
        return alerts

    ratio = vol / vol_ma
    if ratio >= multiplier:
        direction = "🟢 BULLISH" if curr['close'] > curr['open'] else "🔴 BEARISH"
        alerts.append({
            'type': f'📊 VOLUME SPIKE {direction}',
            'detail': f"Όγκος {ratio:.1f}x πάνω από MA20 ({vol:.0f} vs {vol_ma:.0f})",
            'bias': 'LONG' if curr['close'] > curr['open'] else 'SHORT',
            'price': curr['close']
        })

    return alerts


def check_pivot_levels(df: pd.DataFrame, symbol: str, tf: str) -> list:
    alerts = []
    if len(df) < 2:
        return alerts

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    price = curr['close']

    levels = {
        'R2': curr['r2'],
        'R1': curr['r1'],
        'Pivot': curr['pivot'],
        'S1': curr['s1'],
        'S2': curr['s2'],
    }

    for name, level in levels.items():
        if pd.isna(level):
            continue
        proximity = abs(price - level) / level * 100

        if proximity < 0.15:  # Within 0.15% of level
            if price > level:
                alerts.append({
                    'type': f'📍 PIVOT LEVEL — {name}',
                    'detail': f"Τιμή κοντά σε {name} ({level:.1f}) — πάνω | Potential resistance",
                    'bias': 'SHORT WATCH',
                    'price': price
                })
            else:
                alerts.append({
                    'type': f'📍 PIVOT LEVEL — {name}',
                    'detail': f"Τιμή κοντά σε {name} ({level:.1f}) — κάτω | Potential support",
                    'bias': 'LONG WATCH',
                    'price': price
                })

    return alerts


def check_candlestick_patterns(df: pd.DataFrame, symbol: str, tf: str) -> list:
    alerts = []
    if len(df) < 3:
        return alerts

    c = df.iloc[-1]
    p = df.iloc[-2]
    pp = df.iloc[-3]

    body = abs(c['close'] - c['open'])
    full_range = c['high'] - c['low']
    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']

    if full_range == 0:
        return alerts

    # Hammer (bullish)
    if (lower_wick > body * 2 and
            upper_wick < body * 0.5 and
            c['close'] > c['open']):
        alerts.append({
            'type': '🔨 PATTERN — HAMMER',
            'detail': f"Bullish Hammer σχηματίστηκε — potential reversal πάνω",
            'bias': 'LONG',
            'price': c['close']
        })

    # Shooting Star (bearish)
    elif (upper_wick > body * 2 and
          lower_wick < body * 0.5 and
          c['close'] < c['open']):
        alerts.append({
            'type': '⭐ PATTERN — SHOOTING STAR',
            'detail': f"Bearish Shooting Star — potential reversal κάτω",
            'bias': 'SHORT',
            'price': c['close']
        })

    # Bullish Engulfing
    if (p['close'] < p['open'] and  # prev bearish
            c['close'] > c['open'] and  # curr bullish
            c['open'] < p['close'] and
            c['close'] > p['open']):
        alerts.append({
            'type': '🕯️ PATTERN — BULLISH ENGULFING',
            'detail': f"Bullish Engulfing — δυνατό reversal signal ⬆️",
            'bias': 'LONG',
            'price': c['close']
        })

    # Bearish Engulfing
    elif (p['close'] > p['open'] and  # prev bullish
          c['close'] < c['open'] and  # curr bearish
          c['open'] > p['close'] and
          c['close'] < p['open']):
        alerts.append({
            'type': '🕯️ PATTERN — BEARISH ENGULFING',
            'detail': f"Bearish Engulfing — δυνατό reversal signal ⬇️",
            'bias': 'SHORT',
            'price': c['close']
        })

    # Doji
    if body < full_range * 0.1:
        alerts.append({
            'type': '➕ PATTERN — DOJI',
            'detail': f"Doji σχηματίστηκε — αβεβαιότητα αγοράς, περίμενε επόμενο candle",
            'bias': 'NEUTRAL',
            'price': c['close']
        })

    return alerts


# ─── ALERT FORMATTING ─────────────────────────────────────────────────────────

def format_alert(alert: dict, symbol: str, timeframe: str) -> str:
    coin = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
    emoji_tf = {'15m': '⚡', '1h': '🕐', '4h': '📊'}.get(timeframe, '📌')
    bias_emoji = {'LONG': '🟢', 'SHORT': '🔴', 'LONG WATCH': '🟡', 'SHORT WATCH': '🟡', 'NEUTRAL': '⚪'}.get(alert['bias'], '⚪')

    return (
        f"{'='*30}\n"
        f"{alert['type']}\n"
        f"{'='*30}\n"
        f"🪙 <b>{coin}</b> | {emoji_tf} {timeframe.upper()}\n"
        f"💰 Τιμή: <b>{alert['price']:,.2f}</b> USDT\n"
        f"📋 {alert['detail']}\n"
        f"{bias_emoji} Bias: <b>{alert['bias']}</b>\n"
        f"🕒 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
    )


# ─── ALERT DEDUPLICATION ──────────────────────────────────────────────────────

sent_alerts = {}

def is_duplicate(key: str, cooldown_minutes: int = 60) -> bool:
    now = time.time()
    if key in sent_alerts:
        if now - sent_alerts[key] < cooldown_minutes * 60:
            return True
    sent_alerts[key] = now
    return False


# ─── MAIN SCAN LOOP ───────────────────────────────────────────────────────────

def scan_symbol(symbol: str, timeframe: str):
    df = fetch_ohlcv(symbol, timeframe)
    if df.empty or len(df) < 50:
        return

    df = add_indicators(df)

    all_alerts = []
    all_alerts += check_ema_crossover(df, symbol, timeframe)
    all_alerts += check_rsi(df, symbol, timeframe)
    all_alerts += check_breakout(df, symbol, timeframe)
    all_alerts += check_volume_spike(df, symbol, timeframe)
    all_alerts += check_pivot_levels(df, symbol, timeframe)
    all_alerts += check_candlestick_patterns(df, symbol, timeframe)

    for alert in all_alerts:
        dedup_key = f"{symbol}_{timeframe}_{alert['type']}"
        if not is_duplicate(dedup_key):
            msg = format_alert(alert, symbol, timeframe)
            send_telegram(msg)
            time.sleep(0.5)


def main():
    logger.info("🚀 Crypto Alert Bot ξεκίνησε!")
    send_telegram(
        "🤖 <b>Crypto Alert Bot Online!</b>\n\n"
        f"📌 Symbols: {', '.join([s.replace('/USDT:USDT','').replace('/USDT','') for s in SYMBOLS])}\n"
        f"⏱ Timeframes: {', '.join(TIMEFRAMES)}\n"
        f"🔄 Scan κάθε {CHECK_INTERVAL_SECONDS}s\n\n"
        "Στρατηγικές:\n"
        "• 📈 EMA Crossover (20/50)\n"
        "• 📊 RSI Overbought/Oversold\n"
        "• 🚀 Breakout/Breakdown\n"
        "• 📊 Volume Spike\n"
        "• 📍 Pivot Points\n"
        "• 🕯️ Candlestick Patterns"
    )

    while True:
        try:
            for symbol in SYMBOLS:
                for tf in TIMEFRAMES:
                    logger.info(f"Scanning {symbol} {tf}...")
                    scan_symbol(symbol, tf)
                    time.sleep(1)

            logger.info(f"✅ Scan complete. Επόμενο σε {CHECK_INTERVAL_SECONDS}s")
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("⛔ Bot σταμάτησε.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == '__main__':
    main()
