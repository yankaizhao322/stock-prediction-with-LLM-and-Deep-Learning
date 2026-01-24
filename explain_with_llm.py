
# -*- coding: utf-8 -*-
import os, re, argparse, numpy as np, pandas as pd, pytz, torch
from collections import Counter
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


def clean_text(s: str) -> str:
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"[@#]\w+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def try_vader():
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

def compute_sentiment(texts: List[str], vader=None) -> Dict[str, float]:
    if not texts:
        return dict(mean=0.0, pos_ratio=0.0)
    if vader is not None:
        scores = [vader.polarity_scores(t)["compound"] for t in texts]
        arr = np.array(scores, dtype=float)
        return dict(mean=float(arr.mean()), pos_ratio=float((arr > 0.05).mean()))
    pos = {"beat","record","surge","bull","strong","positive","up","buy","growth","upgrade"}
    neg = {"miss","recall","lawsuit","bear","drop","negative","down","sell","downgrade","risk"}
    sc=[]
    for t in texts:
        tl=t.lower()
        p=sum(w in tl for w in pos)
        n=sum(w in tl for w in neg)
        sc.append(p-n)
    arr=np.array(sc,dtype=float)
    return dict(
        mean=float(arr.mean() if len(arr) else 0.0),
        pos_ratio=float((arr>0).mean() if len(arr) else 0.0)
    )

def top_keywords(texts: List[str], k: int = 12) -> List[str]:
    if not texts:
        return []
    stop=set("""
    a an the and or of for to in on with at from as is are was were be been being this that these those
    it its he's she's they're we're you're i've we've you've you'll won't can't don't didn't isn't aren't
    about into over under between after before during against within without while by via per amp
    """.split())
    words=[]
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z\-']{1,}", t):
            wl=w.lower()
            if wl not in stop and len(wl)>=3:
                words.append(wl)
    return [w for w,_ in Counter(words).most_common(k)]

def previous_trading_day(target_date: str, price_index: pd.Index) -> pd.Timestamp:
    T = pd.to_datetime(target_date).normalize()
    dates = pd.to_datetime(price_index)
    prior = dates[dates < T]
    if len(prior) == 0:
        raise ValueError("No earlier trading day found (check price data range).")
    return prior.max()

def load_prices(prices_csv: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(prices_csv)
    if "Stock Name" in df.columns:
        df = df[df["Stock Name"] == ticker]
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.drop_duplicates("Date").set_index("Date").sort_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    return df.dropna(subset=["Close"])

def load_tweets(tweets_csv: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(tweets_csv)
    if "Stock Name" in df.columns:
        df = df[df["Stock Name"] == ticker]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Tweet"] = df["Tweet"].astype(str).map(clean_text)
    return df.dropna(subset=["Date","Tweet"])

def load_predictions(pred_csv: str) -> pd.DataFrame:
    preds = pd.read_csv(pred_csv)
    if "date" not in preds.columns:
        raise ValueError("predictions.csv must contain column 'date'")
    preds["date"] = pd.to_datetime(preds["date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    if (
        "pred_prob_up" not in preds.columns
        and "pred_label" not in preds.columns
        and "pred_price" not in preds.columns
    ):
        raise ValueError(
            "predictions.csv must contain 'pred_prob_up' or 'pred_label' (optional: 'pred_price')"
        )
    return preds

def collect_evidence(pred_date: str, tweets_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict:
    prev_day = previous_trading_day(pred_date, prices_df.index)
    tz_ny = pytz.timezone("America/New_York")
    ts = pd.to_datetime(tweets_df["Date"], errors="coerce", utc=True)
    local_ts = ts.dt.tz_convert(tz_ny)
    mask = (
        (local_ts.dt.tz_localize(None).dt.normalize() == prev_day)
        & (local_ts.dt.hour < 16)
    )
    day_tweets = tweets_df.loc[mask, "Tweet"].dropna().astype(str).tolist()

    uniq, seen = [], set()
    for t in day_tweets:
        key=t.lower()
        if key and key not in seen:
            uniq.append(t)
            seen.add(key)
            if len(uniq) >= 8:
                break

    vader = try_vader()
    sent = compute_sentiment(uniq, vader=vader)
    kw   = top_keywords(uniq, k=12)

    ret_today = (
        float(prices_df["Close"].pct_change().loc[prev_day])
        if prev_day in prices_df.index else 0.0
    )

    vol_rel=None
    if "Volume" in prices_df.columns:
        ma20 = prices_df["Volume"].rolling(20).mean()
        if prev_day in prices_df.index and not np.isnan(ma20.loc[prev_day]):
            vol_rel = float(prices_df["Volume"].loc[prev_day] / ma20.loc[prev_day])

    return {
        "evidence_day": prev_day.strftime("%Y-%m-%d"),
        "tweets_sample": uniq,
        "sent_mean": sent["mean"],
        "sent_pos_ratio": sent["pos_ratio"],
        "keywords": kw,
        "price_change_today": ret_today,
        "volume_rel_mean": vol_rel,
    }

def gen_explanation(tok, lm, ticker: str, pred_date: str, prob_up: float, thr: float, ev: Dict) -> str:
    bullets = [
        f"- Evidence day (previous trading day): {ev['evidence_day']}",
        f"- Pre-close sentiment mean: {ev['sent_mean']:.3f}; positive ratio: {ev['sent_pos_ratio']:.1%}",
        f"- Same-day price change: {ev['price_change_today']:+.2%}"
        + (
            f"; volume / 20-day average: {ev['volume_rel_mean']:.2f}x"
            if ev['volume_rel_mean'] is not None else ""
        )
    ]
    if ev["keywords"]:
        bullets.append(f"- Keywords (by frequency): {', '.join(ev['keywords'][:10])}")
    if ev["tweets_sample"]:
        bullets.append("- Representative tweets (before market close):")
        for t in ev["tweets_sample"]:
            bullets.append(f"  • {t[:140]}")

    system = (
        "You are a rigorous financial research assistant. "
        "Based on the evidence, produce verifiable and restrained explanations. "
        "Answer in English and avoid investment advice."
    )

    user = f"""Asset: {ticker}
Prediction target date (next trading day): {pred_date}
Model output: probability of price increase {prob_up:.1%} (threshold {thr:.2f})

Evidence:
{os.linesep.join(bullets)}

Please answer in bullet points:
1) Key drivers for a potential price increase or decrease tomorrow (based on keywords and tweet semantics);
2) Uncertainty and opposing signals (if any);
3) Events to monitor (earnings, regulation, supply chain, macro, etc.);
4) One-sentence assessment of evidence coverage (tweet volume, balance, potential bias).
"""
    msgs=[
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]
    ids = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(lm.device)
    out = lm.generate(
        ids,
        max_new_tokens=320,
        temperature=0.2,
        top_p=0.9
    )
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets_csv", type=str, required=True)
    ap.add_argument("--prices_csv", type=str, required=True)
    ap.add_argument("--pred_csv",   type=str, required=True)
    ap.add_argument("--ticker",     type=str, required=True)
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv",    type=str, default="explanations.csv")
    args = ap.parse_args()

    prices = load_prices(args.prices_csv, args.ticker)
    tweets = load_tweets(args.tweets_csv, args.ticker)
    preds  = load_predictions(args.pred_csv)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    lm  = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device=="cuda" else None,
        low_cpu_mem_usage=True
    ).to(args.device)

    rows=[]
    for _, r in preds.iterrows():
        d = str(r["date"])
        prob = float(
            r.get(
                "pred_prob_up",
                1.0 if int(r.get("pred_label",0))==1 else 0.0
            )
        )
        ev = collect_evidence(d, tweets, prices)
        text = gen_explanation(tok, lm, args.ticker, d, prob, args.threshold, ev)
        print("="*100)
        print(f"[{args.ticker}] Target date {d} | P(up) {prob:.1%} | Threshold {args.threshold:.2f}")
        print(text)
        rows.append({
            "date": d,
            "prob_up": prob,
            "explanation": text
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Explanations exported to:", os.path.abspath(args.out_csv))

if __name__ == "__main__":
    main()

with open("explain_with_llm.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Saved to explain_with_llm.py")
