"""Rebuild the charts in assets/ (used by README.md) from the newest log in runs/. Run: python make_readme_figs.py"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_COLORS = {"qwen2.5:0.5b": BLUE, "llama3.2:1b": ORANGE, "qwen2.5:1.5b": AQUA, "llama3.2:3b": YELLOW}
MUTED, INK, INK2, GRID, SURFACE = "#b5b4ae", "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK, "font.size": 9,
})

log = pd.read_json(sorted((ROOT / "runs").glob("*.jsonl"))[-1], lines=True)
MODELS = ["qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:1.5b", "llama3.2:3b"]


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def label_h(ax, bars, fmt):
    for b in bars:
        v = b.get_width()
        ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8)


# 1. Model sizes on disk (real, from `ollama list`)
sizes = {"qwen2.5:0.5b": (0.494, 0.398, "Q4_K_M"), "llama3.2:1b": (1.24, 1.321, "Q8_0"),
         "qwen2.5:1.5b": (1.54, 0.986, "Q4_K_M"), "llama3.2:3b": (3.21, 2.019, "Q4_K_M")}
fig, ax = plt.subplots(figsize=(6.5, 2.6))
names = MODELS[::-1]
bars = ax.barh(names, [sizes[m][1] for m in names], color=[MODEL_COLORS[m] for m in names], height=0.6)
for b, m in zip(bars, names):
    p, gb, q = sizes[m]
    ax.annotate("{:.2f} GB   {:.2f}B params · {}".format(gb, p, q), (b.get_width(), b.get_y() + b.get_height() / 2),
                xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=8)
ax.set_xlim(0, 3.3)
ax.set_xlabel("size on disk (GB)")
ax.grid(axis="y", visible=False)
ax.set_title("Downloaded models: size on disk")
save(fig, "model_sizes.png")

# 2. Where the time goes: prefill vs decode per model (real, mean over the 6 compare prompts)
cmp_ = log[log.experiment == "compare"]
t = cmp_.groupby("model")[["load_s", "prefill_s", "decode_s"]].mean().loc[MODELS]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
y = np.arange(len(MODELS))[::-1]
ax.barh(y, t.prefill_s, color=BLUE, height=0.6, label="prefill (read the prompt)")
ax.barh(y, t.decode_s, left=t.prefill_s + 0.04, color=ORANGE, height=0.6, label="decode (write the answer)")
for yi, (m, r) in zip(y, t.iterrows()):
    ax.annotate("{:.1f}s".format(r.prefill_s + r.decode_s), (r.prefill_s + r.decode_s, yi), xytext=(4, 0),
                textcoords="offset points", va="center", color=INK2, fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(MODELS)
ax.set_xlabel("seconds per prompt (mean of 6 prompts)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, (t.prefill_s + t.decode_s).max() * 1.12)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
ax.set_title("Where the time goes: decode dominates", pad=22)
save(fig, "time_breakdown.png")

# 3. Throughput: prefill vs decode tokens/s (real), two panels, same model order and colors
tp = cmp_.groupby("model")[["prefill_tok_per_s", "decode_tok_per_s"]].mean().loc[MODELS]
fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
for ax, col, title in [(axes[0], "decode_tok_per_s", "Decode speed (tokens/s)"),
                       (axes[1], "prefill_tok_per_s", "Prefill speed (tokens/s)")]:
    bars = ax.barh(names, tp[col].loc[names], color=[MODEL_COLORS[m] for m in names], height=0.6)
    label_h(ax, bars, "{:.1f}")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, tp[col].max() * 1.18)
axes[1].set_yticklabels([])
fig.tight_layout()
save(fig, "throughput.png")

# 4. Cold start: first-call load time (real, warmup calls)
w = log[(log.experiment == "warmup")].drop_duplicates("model").set_index("model").load_s.loc[MODELS]
fig, ax = plt.subplots(figsize=(6.5, 2.2))
bars = ax.barh(names, w.loc[names], color=[MODEL_COLORS[m] for m in names], height=0.6)
label_h(ax, bars, "{:.1f}s")
ax.set_xlabel("seconds to load weights into RAM (first call only)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, w.max() * 1.15)
ax.set_title("Cold start: model load time")
save(fig, "load_time.png")

# 5. Illustrative: temperature reshapes the next-token distribution
tokens = ["the", "a", "my", "an", "her", "this", "one", "some", "…junk"]
logits = np.array([5.0, 4.2, 3.0, 2.4, 2.0, 1.6, 1.0, 0.5, -1.0])
temps = [0.3, 0.7, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(1, len(temps), figsize=(11, 2.4), sharey=True)
for ax, T in zip(axes, temps):
    p = np.exp(logits / T)
    p /= p.sum()
    ax.bar(range(len(tokens)), p, color=BLUE, width=0.7)
    ax.set_title("T = {}".format(T), fontsize=10)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, fontsize=7)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("probability")
fig.suptitle("Illustration: softmax(logits / T) for one next-token choice", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "temperature_concept.png")

# 6. Illustrative: top-p keeps the smallest set of tokens that covers p of the mass
p = np.exp(logits / 1.5)
p /= p.sum()
cum = np.cumsum(p)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.4), sharey=True)
for ax, top_p in zip(axes, [0.1, 0.5, 0.9]):
    keep = np.searchsorted(cum, top_p) + 1
    colors = [BLUE if i < keep else MUTED for i in range(len(tokens))]
    ax.bar(range(len(tokens)), p, color=colors, width=0.7)
    ax.set_title("top_p = {}  →  {} token{} kept".format(top_p, keep, "" if keep == 1 else "s"), fontsize=10)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, fontsize=7)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("probability (T = 1.5)")
fig.suptitle("Illustration: top-p cuts the tail (blue = sampled from, gray = removed)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "top_p_concept.png")

# 7. Real: what happened in the temperature and top_p sweeps
def distinct_2(texts):
    grams = []
    for s in texts:
        w_ = s.lower().split()
        grams += list(zip(w_, w_[1:]))
    return len(set(grams)) / max(len(grams), 1)


def non_ascii_share(texts):
    s = "".join(texts)
    return sum(ord(c) > 127 and c not in "’“”—–…" for c in s) / max(len(s), 1)


def sweep(exp, key):
    d = log[log.experiment == exp].copy()
    d["x"] = d.options.apply(lambda o: o[key])
    g = d.groupby("x").response
    return pd.DataFrame({"unique": g.nunique(), "n": g.size(),
                         "distinct_2": g.apply(lambda s: distinct_2(list(s))),
                         "non_ascii": g.apply(lambda s: non_ascii_share(list(s)))})

ts = sweep("temp_sweep", "temperature")
ps = sweep("top_p_sweep", "top_p")
fig, axes = plt.subplots(2, 2, figsize=(9.5, 4.6), sharex="col")
for col, (df_, xlabel, title) in enumerate([(ts, "temperature  (top_p = 1, top_k off)", "Temperature sweep"),
                                            (ps, "top_p  (temperature = 1.5)", "top_p sweep")]):
    x = np.arange(len(df_))
    top, bot = axes[0, col], axes[1, col]
    top.plot(x, df_.distinct_2, color=BLUE, lw=2, marker="o", ms=6)
    top.set_ylim(0, 1.05)
    top.set_title("{}: variety (distinct-2, 1 = every word pair unique)".format(title), fontsize=9.5)
    bot.plot(x, df_.non_ascii * 100, color=ORANGE, lw=2, marker="s", ms=6)
    bot.set_ylim(0, 6)
    bot.set_title("{}: gibberish (% non-ASCII characters)".format(title), fontsize=9.5)
    bot.set_xticks(x)
    bot.set_xticklabels(["{:g}".format(v) for v in df_.index])
    bot.set_xlabel(xlabel)
fig.tight_layout()
save(fig, "sampling_results.png")

# 8. (moved: quantization charts are built from real results in section 10)

print(ts.round(3), ps.round(3), sep="\n")

# 9. Real: "pick a random number between 1 and 10" — 30 samples per temperature
import re
rn = log[log.experiment == "random_number"].copy()
if len(rn):
    rn["T"] = rn.options.apply(lambda o: o["temperature"])
    rn["n"] = rn.response.apply(lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else None)
    Ts = sorted(rn["T"].unique())
    fig, axes = plt.subplots(1, len(Ts), figsize=(10, 2.5), sharey=True)
    xs = list(range(1, 11))
    for ax, T in zip(np.atleast_1d(axes), Ts):
        c = rn[rn["T"] == T].n.value_counts()
        bars = ax.bar(xs, [c.get(x, 0) for x in xs], color=BLUE, width=0.7)
        for b in bars:
            if b.get_height():
                ax.annotate(int(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.set_title("temperature = {:g}".format(T), fontsize=10)
        ax.set_xticks(xs)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("times picked (of 30)")
    fig.suptitle("llama3.2:1b asked for \"a random number between 1 and 10\"", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "random_number.png")
    print(rn.groupby("T").n.value_counts().unstack(fill_value=0))


# 10. Real: grade the eval answers with the same checkers as the notebook
import json, difflib


def number_is(x):
    def check(resp):
        m = re.search(r"-?\d+(?:\.\d+)?", resp.replace(",", ""))
        return bool(m) and float(m.group()) == float(x)
    return check


def contains(pattern):
    return lambda resp: re.search(pattern, resp, re.IGNORECASE) is not None


def is_json_person(resp):
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    try:
        obj = json.loads(m.group()) if m else None
    except ValueError:
        return False
    return isinstance(obj, dict) and str(obj.get("name", "")).lower() == "ada" and str(obj.get("age")) == "36"


# prompt fragment -> (item name, checker); mirrors EVAL in the notebook
CHECKS = {
    "17 * 24": ("arith_mul", number_is(408)), "15% of 80": ("arith_pct", number_is(12)),
    "capital of Australia": ("capital", contains(r"\bcanberra\b")), "symbol for gold": ("gold", contains(r"\bAu\b")),
    "Pride and Prejudice": ("author", contains("austen")), "into Spanish": ("spanish", contains("gracias")),
    "strawberry": ("letters", number_is(3)), "'planet'": ("reverse", contains(r"\btenalp\b")),
    "Sort alphabetically": ("sort", contains(r"apple,\s*banana,\s*mango,\s*pear")),
    "feathers or": ("feathers", contains(r"\bsame\b")), "Sally": ("sisters", number_is(1)),
    "3:40 PM": ("time", contains(r"\b6:15\b")), "JSON object": ("json", is_json_person),
    "How many cents": ("bat_ball", number_is(5)), "prime numbers": ("primes", number_is(8)),
}


def grade(prompt, response):
    for frag, (name, check) in CHECKS.items():
        if frag in prompt:
            return name, check(response)
    return None, None


ev = log[log.experiment.isin(["quant_eval", "size_eval"])].copy()
ev["item"], ev["correct"] = zip(*[grade(p_, r_) for p_, r_ in zip(ev.prompt, ev.response)])

QUANTS = ["llama3.2:1b-instruct-fp16", "llama3.2:1b-instruct-q8_0", "llama3.2:1b-instruct-q4_K_M",
          "llama3.2:1b-instruct-q4_0", "llama3.2:1b-instruct-q2_K"]
QUANT_GB = {"fp16": 2.48, "q8_0": 1.32, "q4_K_M": 0.81, "q4_0": 0.77, "q2_K": 0.58}   # from /api/tags
qe = ev[ev.experiment == "quant_eval"]
if len(qe):
    dr = log[log.experiment == "quant_drift"]
    ref = dr[dr.model == QUANTS[0]].groupby("prompt").response.first()
    dr = dr.assign(sim=[difflib.SequenceMatcher(None, ref[p_], r_).ratio() for p_, r_ in zip(dr.prompt, dr.response)])
    qs = pd.DataFrame({
        "accuracy": qe.groupby("model").correct.mean(),
        "decode_tok_s": dr.groupby("model").decode_tok_per_s.mean(),
        "similarity": dr.groupby("model").sim.mean(),
    }).loc[QUANTS]
    qs.index = [q.split("-")[-1] for q in qs.index]
    qs["size_GB"] = [QUANT_GB[q] for q in qs.index]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    panels = [("size_GB", "Size on disk (GB)", "{:.2f}"), ("decode_tok_s", "Decode speed (tok/s)", "{:.1f}"),
              ("accuracy", "Eval accuracy (15 items)", "{:.0%}"), ("similarity", "Text similarity to fp16", "{:.2f}")]
    for ax, (col, title, fmt) in zip(axes, panels):
        bars = ax.bar(qs.index, qs[col], color=BLUE, width=0.6)
        for b in bars:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(0, qs[col].max() * 1.18)
    axes[2].set_ylim(0, 1)
    axes[3].set_ylim(0, 1.12)
    fig.suptitle("llama3.2:1b at five quantization levels (measured)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "quant_results.png")
    print(qs.round(3))

# 11. Real: size vs quality. Speed comes from Part 2 (long outputs); Part 5 answers are ~9 tokens, which inflates tok/s.
se = ev[ev.experiment == "size_eval"]
if len(se):
    acc = se.groupby("model").correct.mean().loc[MODELS]
    spd = tp.decode_tok_per_s.loc[MODELS]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    offsets = {"qwen2.5:0.5b": (0, 16, "center"), "llama3.2:1b": (-12, 0, "right"),
               "qwen2.5:1.5b": (8, 6, "left"), "llama3.2:3b": (10, 0, "left")}
    for m in MODELS:
        ax.scatter(spd[m], acc[m], s=60 + 90 * sizes[m][1], color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = offsets[m]
        ax.annotate("{}  {:.0%}".format(m, acc[m]), (spd[m], acc[m]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=9, color=INK)
    ax.set_xlim(2, 14)
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("decode speed, tokens/s (Part 2)  →  faster")
    ax.set_ylabel("eval accuracy  →  better")
    ax.set_title("Speed vs. quality (marker area ∝ size on disk)")
    save(fig, "size_vs_quality.png")
    print(pd.DataFrame({"acc": acc, "tok_s": spd}).round(3))
