#!/usr/bin/env python3
import logging
import time
from datetime import datetime
import ccxt
import pandas as pd
import requests
import pandas_ta as ta
from config import (
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
SYMBOLS, TIMEFRAMES, CHECK_INTERVAL_SECONDS
)

logging.basicConfig(level=logging.INFO, format=’%(asctime)s [%(levelname)s] %(message)s’)
logger = logging.getLogger(**name**)
exchange = ccxt.binance({‘enableRateLimit’: True})
sent_alerts = {}

def send_telegram(message):
url = “https://api.telegram.org/bot” + TELEGRAM_BOT_TOKEN + “/sendMessage”
try:
requests.post(url, json={“chat_id”: TELEGRAM_CHAT_ID, “text”: message, “parse_mode”: “HTML”}, timeout=10)
logger.info(“Sent: “ + message[:50])
except Exception as e:
logger.error(“Telegram error: “ + str(e))

def fetch_ohlcv(symbol, timeframe, limit=200):
try:
raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(raw, columns=[‘timestamp’, ‘open’, ‘high’, ‘low’, ‘close’, ‘volume’])
df[‘timestamp’] = pd.to_datetime(df[‘timestamp’], unit=‘ms’)
df.set_index(‘timestamp’, inplace=True)
return df
except Exception as e:
logger.error(“Fetch error “ + symbol + “ “ + timeframe + “: “ + str(e))
return pd.DataFrame()

def add_indicators(df):
df[‘ema20’] = ta.ema(df[‘close’], length=20)
df[‘ema50’] = ta.ema(df[‘close’], length=50)
df[‘ema100’] = ta.ema(df[‘close’], length=100)
df[‘ema200’] = ta.ema(df[‘close’], length=200)
df[‘rsi’] = ta.rsi(df[‘close’], length=14)
df[‘atr’] = ta.atr(df[‘high’], df[‘low’], df[‘close’], length=14)
df[‘vol_ma20’] = df[‘volume’].rolling(20).mean()
df[‘pivot’] = (df[‘high’].shift(1) + df[‘low’].shift(1) + df[‘close’].shift(1)) / 3
df[‘r1’] = 2 * df[‘pivot’] - df[‘low’].shift(1)
df[‘r2’] = df[‘pivot’] + (df[‘high’].shift(1) - df[‘low’].shift(1))
df[‘s1’] = 2 * df[‘pivot’] - df[‘high’].shift(1)
df[‘s2’] = df[‘pivot’] - (df[‘high’].shift(1) - df[‘low’].shift(1))
return df

def calc_setup(price, atr, bias, tf):
tf_mult = {‘15m’: 1.0, ‘1h’: 1.5, ‘4h’: 2.0}
m = tf_mult.get(tf, 1.0)
if bias == ‘LONG’:
entry = round(price, 2)
sl = round(price - atr * 1.2 * m, 2)
tp1 = round(price + atr * 1.5 * m, 2)
tp2 = round(price + atr * 3.0 * m, 2)
else:
entry = round(price, 2)
sl = round(price + atr * 1.2 * m, 2)
tp1 = round(price - atr * 1.5 * m, 2)
tp2 = round(price - atr * 3.0 * m, 2)
risk = abs(entry - sl)
reward1 = abs(tp1 - entry)
reward2 = abs(tp2 - entry)
rr1 = round(reward1 / risk, 2) if risk > 0 else 0
rr2 = round(reward2 / risk, 2) if risk > 0 else 0
lev = “10x”
if rr1 >= 1.5:
lev = “10x”
if rr2 >= 2.5:
lev = “15x”
return {‘entry’: entry, ‘sl’: sl, ‘tp1’: tp1, ‘tp2’: tp2, ‘rr1’: rr1, ‘rr2’: rr2, ‘leverage’: lev}

def format_signal(signal_type, detail, bias, price, setup, symbol, tf):
coin = symbol.replace(’/USDT:USDT’, ‘’).replace(’/USDT’, ‘’)
tf_emoji = {‘15m’: ‘FAST’, ‘1h’: ‘1H’, ‘4h’: ‘4H’}.get(tf, tf)
bias_arrow = ‘UP’ if bias == ‘LONG’ else ‘DOWN’
msg = “================================\n”
msg += signal_type + “\n”
msg += “================================\n”
msg += “<b>” + coin + “</b> | “ + tf_emoji + “ | <b>” + bias + “</b> “ + bias_arrow + “\n”
msg += detail + “\n”
msg += “––––––––––––––––\n”
msg += “ENTRY:  “ + str(setup[‘entry’]) + “ USDT\n”
msg += “SL:     “ + str(setup[‘sl’]) + “ USDT\n”
msg += “TP1:    “ + str(setup[‘tp1’]) + “ USDT  (R/R “ + str(setup[‘rr1’]) + “:1)\n”
msg += “TP2:    “ + str(setup[‘tp2’]) + “ USDT  (R/R “ + str(setup[‘rr2’]) + “:1)\n”
msg += “––––––––––––––––\n”
msg += “Leverage: “ + setup[‘leverage’] + “\n”
msg += “Close 50% at TP1, leave 50% for TP2\n”
msg += datetime.now().strftime(’%H:%M:%S %d/%m/%Y’)
return msg

def is_duplicate(key, cooldown_min=60):
now = time.time()
if key in sent_alerts and now - sent_alerts[key] < cooldown_min * 60:
return True
sent_alerts[key] = now
return False

def check_ema_crossover(df, symbol, tf):
alerts = []
if len(df) < 3:
return alerts
prev = df.iloc[-2]
curr = df.iloc[-1]
atr = curr[‘atr’] if not pd.isna(curr[‘atr’]) else 0
if prev[‘ema20’] <= prev[‘ema50’] and curr[‘ema20’] > curr[‘ema50’]:
setup = calc_setup(curr[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘EMA CROSSOVER BULLISH’, ‘EMA20 crossed above EMA50’, ‘LONG’, curr[‘close’], setup, symbol, tf))
elif prev[‘ema20’] >= prev[‘ema50’] and curr[‘ema20’] < curr[‘ema50’]:
setup = calc_setup(curr[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘EMA CROSSOVER BEARISH’, ‘EMA20 crossed below EMA50’, ‘SHORT’, curr[‘close’], setup, symbol, tf))
return alerts

def check_rsi(df, symbol, tf):
alerts = []
if len(df) < 2:
return alerts
curr = df.iloc[-1]
prev = df.iloc[-2]
rsi = curr[‘rsi’]
atr = curr[‘atr’] if not pd.isna(curr[‘atr’]) else 0
if prev[‘rsi’] <= 30 and rsi > 30:
setup = calc_setup(curr[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘RSI OVERSOLD EXIT’, ’RSI crossed above 30: ’ + str(round(rsi, 1)), ‘LONG’, curr[‘close’], setup, symbol, tf))
elif rsi < 25:
setup = calc_setup(curr[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘RSI EXTREME OVERSOLD’, ’RSI = ’ + str(round(rsi, 1)), ‘LONG’, curr[‘close’], setup, symbol, tf))
elif prev[‘rsi’] >= 70 and rsi < 70:
setup = calc_setup(curr[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘RSI OVERBOUGHT EXIT’, ’RSI crossed below 70: ’ + str(round(rsi, 1)), ‘SHORT’, curr[‘close’], setup, symbol, tf))
elif rsi > 75:
setup = calc_setup(curr[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘RSI EXTREME OVERBOUGHT’, ’RSI = ’ + str(round(rsi, 1)), ‘SHORT’, curr[‘close’], setup, symbol, tf))
return alerts

def check_breakout(df, symbol, tf, lookback=20):
alerts = []
if len(df) < lookback + 2:
return alerts
curr = df.iloc[-1]
prev = df.iloc[-2]
window = df.iloc[-(lookback+1):-1]
highest = window[‘high’].max()
lowest = window[‘low’].min()
atr = curr[‘atr’] if not pd.isna(curr[‘atr’]) else 0
if prev[‘close’] <= highest and curr[‘close’] > highest + (atr * 0.1):
setup = calc_setup(curr[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘BREAKOUT BULLISH’, ’Broke ’ + str(lookback) + ’-candle high: ’ + str(round(highest, 1)), ‘LONG’, curr[‘close’], setup, symbol, tf))
elif prev[‘close’] >= lowest and curr[‘close’] < lowest - (atr * 0.1):
setup = calc_setup(curr[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘BREAKDOWN BEARISH’, ’Broke ’ + str(lookback) + ’-candle low: ’ + str(round(lowest, 1)), ‘SHORT’, curr[‘close’], setup, symbol, tf))
return alerts

def check_volume_spike(df, symbol, tf, multiplier=2.5):
alerts = []
if len(df) < 21:
return alerts
curr = df.iloc[-1]
vol = curr[‘volume’]
vol_ma = curr[‘vol_ma20’]
if pd.isna(vol_ma) or vol_ma == 0:
return alerts
ratio = vol / vol_ma
atr = curr[‘atr’] if not pd.isna(curr[‘atr’]) else 0
if ratio >= multiplier:
bias = ‘LONG’ if curr[‘close’] > curr[‘open’] else ‘SHORT’
setup = calc_setup(curr[‘close’], atr, bias, tf)
alerts.append(format_signal(‘VOLUME SPIKE’, ’Volume ’ + str(round(ratio, 1)) + ‘x above MA20’, bias, curr[‘close’], setup, symbol, tf))
return alerts

def check_candlestick_patterns(df, symbol, tf):
alerts = []
if len(df) < 3:
return alerts
c = df.iloc[-1]
p = df.iloc[-2]
body = abs(c[‘close’] - c[‘open’])
full_range = c[‘high’] - c[‘low’]
upper_wick = c[‘high’] - max(c[‘close’], c[‘open’])
lower_wick = min(c[‘close’], c[‘open’]) - c[‘low’]
atr = c[‘atr’] if not pd.isna(c[‘atr’]) else 0
if full_range == 0:
return alerts
if lower_wick > body * 2 and upper_wick < body * 0.5 and c[‘close’] > c[‘open’]:
setup = calc_setup(c[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘PATTERN HAMMER’, ‘Bullish Hammer - potential reversal UP’, ‘LONG’, c[‘close’], setup, symbol, tf))
elif upper_wick > body * 2 and lower_wick < body * 0.5 and c[‘close’] < c[‘open’]:
setup = calc_setup(c[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘PATTERN SHOOTING STAR’, ‘Bearish Shooting Star - potential reversal DOWN’, ‘SHORT’, c[‘close’], setup, symbol, tf))
if p[‘close’] < p[‘open’] and c[‘close’] > c[‘open’] and c[‘open’] < p[‘close’] and c[‘close’] > p[‘open’]:
setup = calc_setup(c[‘close’], atr, ‘LONG’, tf)
alerts.append(format_signal(‘PATTERN BULLISH ENGULFING’, ‘Bullish Engulfing - strong reversal UP’, ‘LONG’, c[‘close’], setup, symbol, tf))
elif p[‘close’] > p[‘open’] and c[‘close’] < c[‘open’] and c[‘open’] > p[‘close’] and c[‘close’] < p[‘open’]:
setup = calc_setup(c[‘close’], atr, ‘SHORT’, tf)
alerts.append(format_signal(‘PATTERN BEARISH ENGULFING’, ‘Bearish Engulfing - strong reversal DOWN’, ‘SHORT’, c[‘close’], setup, symbol, tf))
return alerts

def scan_symbol(symbol, timeframe):
df = fetch_ohlcv(symbol, timeframe)
if df.empty or len(df) < 50:
return
df = add_indicators(df)
all_alerts = []
all_alerts += check_ema_crossover(df, symbol, timeframe)
all_alerts += check_rsi(df, symbol, timeframe)
all_alerts += check_breakout(df, symbol, timeframe)
all_alerts += check_volume_spike(df, symbol, timeframe)
all_alerts += check_candlestick_patterns(df, symbol, timeframe)
for msg in all_alerts:
key = symbol + “*” + timeframe + “*” + msg[:40]
if not is_duplicate(key):
send_telegram(msg)
time.sleep(0.5)

def main():
logger.info(“Crypto Alert Bot v2 started!”)
send_telegram(
“<b>Crypto Alert Bot v2 Online!</b>\n\n”
“BTC | SOL | ETH\n”
“15m | 1H | 4H\n\n”
“Each alert includes:\n”
“ENTRY | SL | TP1 | TP2 | R/R | Leverage”
)
while True:
try:
for symbol in SYMBOLS:
for tf in TIMEFRAMES:
logger.info(“Scanning “ + symbol + “ “ + tf)
scan_symbol(symbol, tf)
time.sleep(1)
logger.info(“Scan complete. Next in “ + str(CHECK_INTERVAL_SECONDS) + “s”)
time.sleep(CHECK_INTERVAL_SECONDS)
except KeyboardInterrupt:
logger.info(“Bot stopped.”)
break
except Exception as e:
logger.error(“Error: “ + str(e))
time.sleep(30)

if **name** == ‘**main**’:
main()
