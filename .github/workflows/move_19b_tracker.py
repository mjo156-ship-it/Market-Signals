#!/usr/bin/env python3
"""
MOVE 19B Episode Tracker v1.0  (2026-10-05)
===========================================
Signal 19B: ^MOVE 20-day change > +50%  -> contrarian UPRO entry.

Rules (as backtested 2026-10-05, n=19 episodes 2003-2026):
  * FRESH trigger = 19B goes off -> on at a close.
  * Each fresh trigger: add a UPRO tranche at the NEXT session's close.
  * Every fresh trigger resets the exit for ALL open tranches to
    20 trading sessions after the latest tranche's entry.
  * Exit: sell all UPRO at the close of the exit session.

Stateless: the whole episode is re-derived from ^MOVE history each run,
so there is no state file to commit and a missed run cannot corrupt it.

Modes (argv[1]):
  evening (default) - run after the close. Sends an email when:
       - a fresh trigger fired today (BUY tranche at tomorrow's close)
       - the exit date moved because of a re-trigger
       - 3 sessions before exit (heads-up)
       - 1 session before exit (SELL TOMORROW at the close)
  morning           - sends only on the exit session itself (SELL TODAY).
  status            - print the current episode, never email.
  test              - email the current status unconditionally (setup check).
"""
import os, sys, smtplib
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np, pandas as pd, yfinance as yf

SENDER_EMAIL = os.environ.get('SENDER_EMAIL', '')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', '')
PHONE_EMAIL = os.environ.get('PHONE_EMAIL', '')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'evening'

THRESH = 0.50          # 20d change threshold
LOOKBACK = 20          # sessions for the change
HOLD = 20              # sessions held after the latest entry
ENTRY_LAG = 1          # enter at next session's close
SIZING = ["5% of total portfolio", "+2.5%", "+2.5%"]   # T1 / T2 / T3; cap 10%

# ----------------------------------------------------------------- calendar
FALLBACK_HOLIDAYS = [  # NYSE full closures, used only if pandas_market_calendars is missing
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25','2026-06-19',
    '2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31','2027-06-18',
    '2027-07-05','2027-09-06','2027-11-25','2027-12-24']

def nyse_sessions(start, end):
    try:
        import pandas_market_calendars as mcal
        s = mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(s.index.normalize().tz_localize(None))
    except Exception as e:
        print(f"pandas_market_calendars unavailable ({e}); using fallback holidays")
        bd = pd.bdate_range(start, end)
        return bd[~bd.isin(pd.to_datetime(FALLBACK_HOLIDAYS))]

# ----------------------------------------------------------------- data
def load_move():
    df = yf.download('^MOVE', start='2002-01-01', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    s = df['Close'].dropna()
    s.index = pd.DatetimeIndex(s.index).normalize().tz_localize(None)
    return s

# ----------------------------------------------------------------- episode logic
def build(move, today):
    sessions = nyse_sessions(move.index[0], today + pd.Timedelta(days=120))
    hist = sessions[sessions <= move.index[-1]]
    mv = move.reindex(hist).ffill(limit=3)          # same convention as backtest
    ch20 = mv / mv.shift(LOOKBACK) - 1
    on = (ch20 > THRESH).fillna(False)
    fresh = on & ~on.shift(1, fill_value=False)
    pos = {d: i for i, d in enumerate(sessions)}

    episodes, cur = [], None
    for d in fresh[fresh].index:
        ent = pos[d] + ENTRY_LAG
        if cur and ent <= cur['exit']:
            cur['prev_exit'] = cur['exit']
            cur['triggers'].append(d); cur['entries'].append(ent); cur['exit'] = ent + HOLD
        else:
            if cur: episodes.append(cur)
            cur = {'triggers': [d], 'entries': [ent], 'exit': ent + HOLD, 'prev_exit': None}
    if cur: episodes.append(cur)
    return sessions, mv, ch20, on, episodes

def main():
    today = pd.Timestamp(date.today())
    move = load_move()
    sessions, mv, ch20, on, eps = build(move, today)
    last = mv.index[-1]
    is_session = today in sessions
    # index of "today" in session terms (last completed close for evening runs)
    t_idx = sessions.get_indexer([today])[0] if is_session else sessions.searchsorted(today) - 1
    S = lambda i: sessions[i].strftime('%a %b %d, %Y')

    stale = (last < today) and MODE == 'evening' and is_session
    lines = [f"^MOVE close {last.date()}: {mv.iloc[-1]:.1f} | 20d change {ch20.iloc[-1]*100:+.1f}% "
             f"(threshold +{THRESH*100:.0f}%) | 19B {'ON' if on.iloc[-1] else 'off'}"]
    if stale:
        lines.append("NOTE: ^MOVE has not printed today's close yet - trigger check uses the prior close.")

    ep = eps[-1] if eps else None
    active = ep is not None and ep['exit'] >= t_idx and ep['triggers'][0] <= sessions[t_idx]
    alerts = []   # (subject, sms_text)

    if active:
        left = ep['exit'] - t_idx
        lines.append(f"\nACTIVE EPISODE (started {ep['triggers'][0].date()}), {len(ep['triggers'])} fresh trigger(s):")
        for k, (d, e) in enumerate(zip(ep['triggers'], ep['entries'])):
            lines.append(f"  T{k+1}: trigger {d.date()} -> buy UPRO at close {S(e)}  [size {SIZING[min(k,2)]}]")
        lines.append(f"EXIT: sell ALL UPRO at the close on {S(ep['exit'])}  ({left} session(s) from today)")
        lines.append("Any new fresh trigger before then pushes the exit 20 sessions past the new entry.")

        fired_today = ep['triggers'][-1] == last and last == sessions[t_idx]
        if MODE == 'evening':
            if fired_today:
                k = len(ep['triggers'])
                subj = f"🟢 19B FRESH TRIGGER T{k}: buy UPRO tranche at tomorrow's close ({SIZING[min(k-1,2)]})"
                if ep['prev_exit'] is not None and k > 1:
                    subj += f" | exit moved {sessions[ep['prev_exit']].date()} -> {sessions[ep['exit']].date()}"
                alerts.append((subj, f"19B T{k}: buy UPRO at close {sessions[ep['entries'][-1]].date()}. Exit {sessions[ep['exit']].date()}"))
            elif left == 3:
                alerts.append((f"🟡 19B exit in 3 sessions: sell UPRO at close {S(ep['exit'])}",
                               f"19B: sell UPRO at close {sessions[ep['exit']].date()} (3 sessions)"))
            elif left == 1:
                alerts.append((f"🔴 19B SELL TOMORROW: all UPRO at the close {S(ep['exit'])}",
                               f"19B: SELL ALL UPRO TOMORROW at close ({sessions[ep['exit']].date()})"))
        elif MODE == 'morning' and left == 0 and is_session:
            alerts.append((f"🔴 19B SELL TODAY: all UPRO at the close ({S(ep['exit'])})",
                           "19B: SELL ALL UPRO TODAY at the close"))
    else:
        lines.append("\nNo active 19B episode.")
        if ep is not None:
            lines.append(f"Last episode: triggers {[d.date().isoformat() for d in ep['triggers']]}, exited {sessions[ep['exit']].date()}")

    body = "\n".join(lines) + ("\n\nRule: fresh trigger = 19B off->on. Tranche at next close; "
            "exit 20 sessions after the latest entry. Signal-monitor only (^MOVE not in Composer).")
    print(body)
    if MODE == 'test':
        send("🧪 19B tracker test - current episode status", body, "19B tracker test OK")
        return
    if MODE == 'status' or not alerts:
        print("\n(no alert sent)")
        return
    for subj, sms in alerts:
        send(subj, body, sms)

def send(subject, body, sms):
    print(f"\n>>> ALERT: {subject}")
    if not (SENDER_EMAIL and SENDER_PASSWORD and RECIPIENT_EMAIL):
        print("Email not configured - console only."); return
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        for to, subj, txt in [(RECIPIENT_EMAIL, subject, body)] + ([(PHONE_EMAIL, '', sms)] if PHONE_EMAIL else []):
            m = MIMEMultipart(); m['From'] = SENDER_EMAIL; m['To'] = to; m['Subject'] = subj
            m.attach(MIMEText(txt, 'plain')); server.send_message(m)
        server.quit(); print("Sent.")
    except Exception as e:
        print(f"Email failed: {e}")

if __name__ == '__main__':
    main()
