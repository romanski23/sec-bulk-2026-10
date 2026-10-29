"""A7 amendment A7: unified indicator test.
Usage: python -I indicators.py a7_dir quarterly.parquet prices_corrected.parquet filings.parquet out_md [setting]

1. Horizon test: headline winners' first run (setting), observed at -12, -6, 0, +3, +6 months; their matched controls on the same dates.
2. Breakout test: all eligible tickers; breakout month = 6m total return >= 40% and within 10% of the 52-week high, one per 12 months;
   label = A5 month label from month_labels_all (buy vs too_early/too_late; unknown dropped).
3. Discovery/holdout: random halves of tickers (seed 20261029, written to split.csv) and a time holdout (dates >= 2019-01-01).
   Candidates on discovery only; BH 10%; confirmed if same sign and >= half the magnitude on both holdouts with >= 15 cases per group.
4. Composite: logistic regression on confirmed breakout features, fitted on discovery, scored on holdout; hit rate by score quintile.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_monthly, features  # noqa: E402

a7, quarterly, prices, filings, out_md = sys.argv[1:6]
SETTING = sys.argv[6] if len(sys.argv) > 6 else "h25_t25"
RUN_MODE = sys.argv[7] if len(sys.argv) > 7 else "first"      # "first": first XBRL-era run per winner; "all": every run
DEDUP_DAYS = int(sys.argv[8]) if len(sys.argv) > 8 else 365   # episode spacing per stock; 0 = every qualifying month
a7 = Path(a7)
SEED = 20261029
HORIZONS = [-12, -6, 0, 3, 6]
lines = []


def say(s=""):
    lines.append(s); print(s, flush=True)


U = pd.read_parquet(a7 / "universe.parquet")
W = pd.read_parquet(a7 / "winners.parquet")
C = pd.read_parquet(a7 / "controls.parquet")
R = pd.read_parquet(a7 / "runs.parquet"); R = R[R.setting == SETTING]
LA = pd.read_parquet(a7 / "month_labels_all.parquet", columns=["ticker", "month", f"label_{SETTING}"]).rename(columns={f"label_{SETTING}": "label"})
tk = U[["ticker", "cik", "sic"]].drop_duplicates("ticker")          # cik as used at the latest snapshot (predecessor mapping applied there)
W["headline"] = np.where(W.tag == "win_10x10y", W.window_complete_10y, W.window_complete_5y)
WH = W[W.headline]
elig_tickers = sorted(set(U[U.eligible].ticker))
rng = np.random.default_rng(SEED)
half = pd.Series(rng.random(len(elig_tickers)) < 0.5, index=elig_tickers)
split = pd.DataFrame({"ticker": elig_tickers, "set": np.where(half.values, "discovery", "holdout")})
split.to_csv(a7 / "split.csv", index=False)
set_of = dict(zip(split.ticker, split.set))

say(f"# A7 unified indicator test ({SETTING}; runs: {RUN_MODE}; episode spacing: {DEDUP_DAYS} days)\n")
monthly = build_monthly(prices)
Q = pd.read_parquet(quarterly)

# ---------------- 1. horizon dataset ----------------
MIN_RUN_START = pd.Timestamp("2012-07-01")   # XBRL covers all filers from mid-2011; a -12m observation needs fundamentals, so runs from mid-2012
first_runs = R[pd.to_datetime(R.start) >= MIN_RUN_START].sort_values(["ticker", "run_no"])[["ticker", "start", "run_no", "buy_months"]]
if RUN_MODE == "first":
    first_runs = first_runs.drop_duplicates("ticker")
first_runs = first_runs[first_runs.ticker.isin(set(WH.ticker))]
say(f"Horizon test: {'every run' if RUN_MODE == 'all' else 'first run per winner'} starting on/after {MIN_RUN_START.date()} (XBRL constraint); "
    f"{len(first_runs)} runs on {first_runs.ticker.nunique()} winners; run-length distribution (buy months): median {first_runs.buy_months.median():.0f}, "
    f">= 12 months: {(first_runs.buy_months >= 12).mean():.0%}.\n")
obs = []
for r in first_runs.itertuples():
    for h in HORIZONS:
        obs.append(dict(ticker=r.ticker, run_start=r.start, run_no=r.run_no, horizon=h, obs_date=(pd.Timestamp(r.start) + pd.DateOffset(months=h)) + pd.offsets.MonthEnd(0), group="winner", winner=r.ticker))
wobs = pd.DataFrame(obs); wobs["obs_date"] = pd.to_datetime(wobs.obs_date).astype("datetime64[ns]"); wobs["run_start"] = pd.to_datetime(wobs.run_start).astype("datetime64[ns]")
# controls observed on their winner's dates (controls of the winner at its first qualifying snapshot, any tag)
cw = C.merge(WH[["ticker", "snapshot", "tag"]].rename(columns={"ticker": "winner"}), on=["winner", "snapshot", "tag"])
cw = cw.drop_duplicates(["winner", "control"])
cobs = wobs.merge(cw[["winner", "control"]], on="winner").drop(columns=["ticker"]).rename(columns={"control": "ticker"})
# a control observed on the same date for several runs of its winner is kept once per date
cobs = cobs.drop_duplicates(["ticker", "obs_date", "horizon"])
cobs["group"] = "control"
H = pd.concat([wobs, cobs], ignore_index=True).merge(tk, on="ticker", how="left")
H = H[H.cik.notna()]
H = features(H, Q, monthly, filings)
H["set"] = H.ticker.map(set_of).fillna("holdout")
H["time_holdout"] = H.run_start >= "2019-01-01"
H.to_parquet(a7 / f"horizon_features_{RUN_MODE}.parquet", index=False)
say(f"Horizon dataset: {H[H.group == 'winner'].ticker.nunique()} winners (first runs), {H[H.group == 'control'].ticker.nunique()} controls; rows {len(H):,}; "
    f"rows without fundamentals: {H.ttm_end.isna().mean():.0%}, without prices: {H.adj.isna().mean():.0%}")

# ---------------- 2. breakout dataset ----------------
B = monthly[monthly.ticker.isin(elig_tickers) & (monthly.ret_6m >= 0.40) & (monthly.dist_52w_high >= -0.10) & (monthly.mo >= "2010-01-01")].copy()
B = B.sort_values(["ticker", "mo"])
keep = []
last = {}
for r in B.itertuples():
    if DEDUP_DAYS and r.ticker in last and (r.mo - last[r.ticker]).days < DEDUP_DAYS:
        keep.append(False)
    else:
        keep.append(True); last[r.ticker] = r.mo
B = B[keep]
LA["mo"] = pd.to_datetime(LA.month).dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")
B = B.merge(LA[["ticker", "mo", "label"]], on=["ticker", "mo"], how="left")
B = B[B.label.isin(["buy", "too_early", "too_late"])]
B["hit"] = (B.label == "buy").astype(int)
B["obs_date"] = B.month_end
B = B.merge(tk, on="ticker", how="left")
B = B[B.cik.notna()]
B = features(B[["ticker", "cik", "sic", "obs_date", "hit", "label"]], Q, monthly, filings)
B["set"] = B.ticker.map(set_of)
B["time_holdout"] = B.obs_date >= "2019-01-01"
B.to_parquet(a7 / f"breakout_features_d{DEDUP_DAYS}.parquet", index=False)
say(f"Breakout dataset: {len(B):,} breakouts on {B.ticker.nunique():,} tickers; base rate (became a buy month): {B.hit.mean():.0%}; discovery {int((B.set == 'discovery').sum()):,} / holdout {int((B.set == 'holdout').sum()):,}; time holdout {int(B.time_holdout.sum()):,}\n")

# ---------------- helpers ----------------
def winsor(df, cols, ref):
    lo = ref[cols].quantile(0.01); hi = ref[cols].quantile(0.99)
    return df.assign(**{c: df[c].clip(lo[c], hi[c]) for c in cols})


def bh(p, q=0.10):
    p = np.asarray(p); n = len(p); order = np.argsort(p); ranked = np.empty(n, dtype=bool)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    ranked[order] = np.arange(n) < k
    return ranked


def effect_h(df, f):
    w = df.loc[df.group == "winner", f].dropna(); c = df.loc[df.group == "control", f].dropna()
    if len(w) < 15 or len(c) < 15:
        return np.nan, np.nan, len(w), len(c)
    sd = pd.concat([w, c]).std()
    d = (w.median() - c.median()) / sd if sd > 0 else np.nan
    p = stats.mannwhitneyu(w, c, alternative="two-sided").pvalue
    return d, p, len(w), len(c)


def effect_b(df, f):
    x = df[[f, "hit"]].dropna()
    if len(x) < 45:
        return np.nan, np.nan, len(x), np.nan, np.nan
    try:
        t = pd.qcut(x[f].rank(method="first"), 3, labels=False)
    except ValueError:
        return np.nan, np.nan, len(x), np.nan, np.nan
    top = x.hit[t == 2].mean(); bot = x.hit[t == 0].mean()
    ct = pd.crosstab(t, x.hit)
    p = stats.chi2_contingency(ct)[1] if ct.shape == (3, 2) else np.nan
    return top - bot, p, len(x), top, bot


# ---------------- 3a. horizon test ----------------
say("## Horizon test: winners' first run versus matched controls (standardised median difference, winner minus control)\n")
Hd = H[H.set == "discovery"]; Hh = H[H.set == "holdout"]; Ht = H[H.time_holdout]
Hd = winsor(Hd, FEATURES, Hd); Hh = winsor(Hh, FEATURES, Hd); Ht = winsor(Ht, FEATURES, Hd)
rows = []
for h in HORIZONS:
    for f in FEATURES:
        d, p, nw, nc = effect_h(Hd[Hd.horizon == h], f)
        rows.append(dict(horizon=h, feature=f, d_disc=d, p=p, n_w=nw, n_c=nc))
HT = pd.DataFrame(rows)
valid = HT.p.notna()
HT["bh"] = False; HT.loc[valid, "bh"] = bh(HT.loc[valid, "p"].values)
conf = []
for r in HT[HT.bh].itertuples():
    d2, p2, nw2, nc2 = effect_h(Hh[Hh.horizon == r.horizon], r.feature)
    d3, p3, nw3, nc3 = effect_h(Ht[Ht.horizon == r.horizon], r.feature)
    ok = (not np.isnan(d2)) and (not np.isnan(d3)) and np.sign(d2) == np.sign(r.d_disc) and np.sign(d3) == np.sign(r.d_disc) \
         and abs(d2) >= abs(r.d_disc) / 2 and abs(d3) >= abs(r.d_disc) / 2 and min(nw2, nc2, nw3, nc3) >= 15
    conf.append(dict(horizon=r.horizon, feature=r.feature, d_disc=r.d_disc, d_hold=d2, d_time=d3, n_w=r.n_w, n_c=r.n_c, confirmed=ok))
HC = pd.DataFrame(conf)
HT.to_csv(a7 / f"horizon_tests_{RUN_MODE}.csv", index=False); HC.to_csv(a7 / f"horizon_confirmed_{RUN_MODE}.csv", index=False)
say(f"Feature-horizon pairs tested: {int(valid.sum())}; BH-significant on discovery: {int(HT.bh.sum())}; confirmed on both holdouts: {int(HC.confirmed.sum()) if len(HC) else 0}\n")
if len(HC):
    say("| horizon | feature | d discovery | d holdout | d 2019+ | n winners | n controls | confirmed |")
    say("|---|---|---|---|---|---|---|---|")
    for r in HC.sort_values(["confirmed", "horizon", "d_disc"], ascending=[False, True, False]).itertuples():
        say(f"| {r.horizon:+d}m | {r.feature} | {r.d_disc:+.2f} | {r.d_hold:+.2f} | {r.d_time:+.2f} | {r.n_w} | {r.n_c} | {'yes' if r.confirmed else 'no'} |")
# profile table for confirmed features across horizons (all data, descriptive)
if len(HC) and HC.confirmed.any():
    say("\nHorizon profile of confirmed features (median winner vs median control, all sets):\n")
    Hall = winsor(H, FEATURES, Hd)
    say("| feature | " + " | ".join(f"{h:+d}m W / C" for h in HORIZONS) + " |")
    say("|---|" + "---|" * len(HORIZONS))
    for f in HC[HC.confirmed].feature.unique():
        cells = []
        for h in HORIZONS:
            w = Hall[(Hall.horizon == h) & (Hall.group == "winner")][f].median(); c = Hall[(Hall.horizon == h) & (Hall.group == "control")][f].median()
            cells.append(f"{w:.2f} / {c:.2f}")
        say(f"| {f} | " + " | ".join(cells) + " |")

# ---------------- 3b. episode tests: breakout and trough ----------------
def episode_test(B, name, note):
    say(f"\n## {name} test: episodes that became buy months versus those that did not (hit rate top tercile minus bottom tercile of the feature)\n")
    say(note + "\n")
    Bd = B[B.set == "discovery"]; Bh = B[B.set == "holdout"]; Bt = B[B.time_holdout]
    Bd = winsor(Bd, FEATURES, Bd); Bh = winsor(Bh, FEATURES, Bd); Bt = winsor(Bt, FEATURES, Bd)
    say(f"Base rates: discovery {Bd.hit.mean():.0%} (n={len(Bd):,}), holdout {Bh.hit.mean():.0%} (n={len(Bh):,}), 2019+ {Bt.hit.mean():.0%} (n={len(Bt):,})\n")
    rows = []
    for f in FEATURES:
        d, p, n, top, bot = effect_b(Bd, f)
        rows.append(dict(feature=f, diff_disc=d, p=p, n=n, top=top, bot=bot))
    BT = pd.DataFrame(rows); valid = BT.p.notna()
    BT["bh"] = False; BT.loc[valid, "bh"] = bh(BT.loc[valid, "p"].values)
    conf = []
    for r in BT[BT.bh].itertuples():
        d2, p2, n2, t2, b2 = effect_b(Bh, r.feature); d3, p3, n3, t3, b3 = effect_b(Bt, r.feature)
        ok = (not np.isnan(d2)) and (not np.isnan(d3)) and np.sign(d2) == np.sign(r.diff_disc) and np.sign(d3) == np.sign(r.diff_disc) \
             and abs(d2) >= abs(r.diff_disc) / 2 and abs(d3) >= abs(r.diff_disc) / 2 and min(n2, n3) >= 45
        conf.append(dict(feature=r.feature, diff_disc=r.diff_disc, top_disc=r.top, bot_disc=r.bot, diff_hold=d2, top_hold=t2, bot_hold=b2, diff_time=d3, n_disc=r.n, n_hold=n2, n_time=n3, confirmed=ok))
    BC = pd.DataFrame(conf)
    tag = name.lower()
    BT.to_csv(a7 / f"{tag}_tests_d{DEDUP_DAYS}.csv", index=False); BC.to_csv(a7 / f"{tag}_confirmed_d{DEDUP_DAYS}.csv", index=False)
    say(f"Features tested: {int(valid.sum())}; BH-significant on discovery: {int(BT.bh.sum())}; confirmed on both holdouts: {int(BC.confirmed.sum()) if len(BC) else 0}\n")
    if len(BC):
        say("| feature | discovery top / bottom tercile hit rate | holdout top / bottom | 2019+ diff | n discovery | confirmed |")
        say("|---|---|---|---|---|---|")
        for r in BC.sort_values(["confirmed", "diff_disc"], ascending=[False, False]).itertuples():
            say(f"| {r.feature} | {r.top_disc:.0%} / {r.bot_disc:.0%} | {r.top_hold:.0%} / {r.bot_hold:.0%} | {r.diff_time:+.2f} | {r.n_disc:,} | {'yes' if r.confirmed else 'no'} |")
    if len(BC) and BC.confirmed.any():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        feats = list(BC[BC.confirmed].feature)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Bd[feats], Bd.hit)
        sc = model.predict_proba(Bh[feats])[:, 1]; yh = Bh.hit
        qn = pd.qcut(pd.Series(sc).rank(method="first"), 5, labels=False)
        say(f"\nComposite of {len(feats)} confirmed {tag} features (fit on discovery, scored on holdout; base rate {yh.mean():.0%}):\n")
        say("| holdout score quintile | episodes | hit rate | lift |")
        say("|---|---|---|---|")
        for k in range(5):
            m = (qn == k).values
            say(f"| {k + 1} (low to high) | {int(m.sum()):,} | {yh[m].mean():.0%} | {yh[m].mean() / yh.mean():.2f}x |")
        # forward 36-month annualised return by quintile (from monthly adjusted closes), and the 2019+ time holdout on holdout tickers
        fwd = monthly[["ticker", "mo", "adj"]].copy()
        fwd["mo36"] = fwd.mo - pd.DateOffset(months=36)
        f36 = Bh[["ticker", "mo", "adj"]].merge(fwd[["ticker", "mo36", "adj"]].rename(columns={"mo36": "mo", "adj": "adj36"}), on=["ticker", "mo"], how="left")
        ann36 = (f36.adj36 / f36.adj) ** (1 / 3) - 1
        say("\n| holdout score quintile | median 36m annualised return | share with 36m return < 0 |")
        say("|---|---|---|")
        for k in range(5):
            m = (qn == k).values
            say(f"| {k + 1} | {ann36[m].median():+.0%} | {(ann36[m] < 0).mean():.0%} |")
        tt = Bt[(Bt.set == "holdout")]
        if len(tt) >= 100:
            sct = model.predict_proba(tt[feats])[:, 1]; topt = sct >= np.quantile(sct, 0.8); bott = sct <= np.quantile(sct, 0.2)
            say(f"\n2019+ episodes on holdout tickers (n={len(tt):,}): top-quintile hit rate {tt.hit[topt].mean():.0%}, bottom-quintile {tt.hit[bott].mean():.0%}, base {tt.hit.mean():.0%}")
        coefs = model.named_steps["logisticregression"].coef_[0]
        say("\nStandardised coefficients: " + ", ".join(f"{f} {c:+.2f}" for f, c in sorted(zip(feats, coefs), key=lambda x: -abs(x[1]))))
        unc = set(W[W.outcome_uncertain].ticker); m2 = ~Bh.ticker.isin(unc).values
        sc2 = model.predict_proba(Bh[m2][feats])[:, 1]; top = sc2 >= np.quantile(sc2, 0.8)
        say(f"Holdout excluding outcome_uncertain tickers: top-quintile hit rate {yh[m2][top].mean():.0%} vs base {yh[m2].mean():.0%}")
    else:
        say(f"\nNo confirmed {tag} feature; no composite fitted.")


def episodes(mask_fn, name):
    E_ = monthly[monthly.ticker.isin(elig_tickers) & (monthly.mo >= "2010-01-01")].copy()
    E_ = E_[mask_fn(E_)].sort_values(["ticker", "mo"])
    keep = []; last = {}
    for r in E_.itertuples():
        if DEDUP_DAYS and r.ticker in last and (r.mo - last[r.ticker]).days < DEDUP_DAYS:
            keep.append(False)
        else:
            keep.append(True); last[r.ticker] = r.mo
    E_ = E_[keep].merge(LA[["ticker", "mo", "label"]], on=["ticker", "mo"], how="left")
    E_ = E_[E_.label.isin(["buy", "too_early", "too_late"])]
    E_["hit"] = (E_.label == "buy").astype(int); E_["obs_date"] = E_.month_end
    E_ = E_.merge(tk, on="ticker", how="left"); E_ = E_[E_.cik.notna()]
    E_ = features(E_[["ticker", "cik", "sic", "obs_date", "hit", "label"]], Q, monthly, filings)
    E_["set"] = E_.ticker.map(set_of); E_["time_holdout"] = E_.obs_date >= "2019-01-01"
    E_.to_parquet(a7 / f"{name.lower()}_features_d{DEDUP_DAYS}.parquet", index=False)
    say(f"{name} dataset: {len(E_):,} episodes on {E_.ticker.nunique():,} tickers; base rate {E_.hit.mean():.0%}")
    return E_


episode_test(B, "Breakout", "Breakout = 6-month total return >= 40% and within 10% of the 52-week high; one per 12 months per stock (A7).")
T = episodes(lambda m: (m.dd_from_36m_high <= -0.50) & (m.ret_6m <= -0.30), "Trough")
episode_test(T, "Trough", "Trough = at least 50% below the 36-month high and 6-month return <= -30%; one per 12 months per stock. "
             "Added 29 Oct 2026 after the horizon table showed run starts sit in deep drawdowns; same discipline as the breakout test.")

Path(out_md).write_text("\n".join(lines) + "\n")
