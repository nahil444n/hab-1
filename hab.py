"""Score anchors against HAB-1, the Human Anchor Benchmark (README.md).

  python hab.py score PRED.jsonl [--set core|all] [--json OUT.json]       # one system
  python hab.py compare A.jsonl B.jsonl [--set core]                     # B minus A, paired, with 95% intervals
  python hab.py baseline gold|empty|capitalised OUT.jsonl [--set core]   # reference systems, to check the scorer

PRED.jsonl has one line per item: {"id": ..., "anchors": [{"start", "end", "kind"}]}, offsets in the item's text.
Items missing from the file count as empty predictions; the --set decides which items are scored.

What is counted, and why (every rule follows from what each source's labels can support):
- A required span is found when a prediction matches it: character IoU >= 0.5 ("lenient", the main number; it forgives
  "the Beatles" vs "Beatles" and "Dr. Smith" vs "Smith"), or exactly ("strict", also reported). A GUM topic is found
  when a prediction overlaps the mention and covers its head word. Each span and each prediction is used once.
- Name precision only where the source marks every name (scope "names"). A prediction with a name kind (PERSON, ORG,
  LOCATION, EVENT, WORK, PRODUCT, OTHER) that matches no required span is a false positive, unless it overlaps an
  optional span (in GUM: a named mention that is not required, such as a date). CONCEPT predictions never count
  against name precision.
- Topic recall where topics are marked (also reported, outside the headline, with a looser match where one span
  contains the other, "horror" in "horror movies": topic edges differ most between labelling styles): film genres (CCPE-M), algorithms and data structures (StackOverflow), and the
  things GUM's human-written summary mentions. No source marks every topic, so there is no topic precision; GUM gives
  two guardrails instead: the topic precision proxy (the share of CONCEPT predictions on something the human summary
  mentions) and the junk rate (the share of all predictions that touch no mention's head word at all).
- Kind accuracy only on spans whose kind a person assigned (kind_source "human"), never on Wikidata-derived kinds.
- 95% intervals: 1,000 bootstrap resamples of whole conversations, documents or threads within each component.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ITEMS = ROOT / "data" / "items.jsonl"
NAME_KINDS = {"PERSON", "ORG", "LOCATION", "EVENT", "WORK", "PRODUCT", "OTHER"}
COMPONENTS = ["tweetner7", "wnut17", "gum", "ccpe", "conel", "reddit", "stackoverflow"]
NAME_F1_COMPS = ["tweetner7", "wnut17", "gum", "ccpe", "stackoverflow"]   # every name marked: precision measurable
TOPIC_COMPS = ["gum", "ccpe", "stackoverflow"]
N_BOOT = 1000


def load_items(which="core"):
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    return [x for x in items if which == "all" or x.get("core")]


def load_pred(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r.get("anchors", [])
    return out


def iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / (max(a["end"], b["end"]) - min(a["start"], b["start"]))


def overlaps(a, s, e):
    return a["start"] < e and s < a["end"]


def head_match(p, g):
    return overlaps(p, g["start"], g["end"]) and p["start"] <= g["head"][0] and p["end"] >= g["head"][1]


def contains(p, g):
    """One span inside the other ("horror" in "horror movies"), sharing at least one character."""
    return overlaps(p, g["start"], g["end"]) and ((p["start"] <= g["start"] and p["end"] >= g["end"]) or
                                                  (g["start"] <= p["start"] and g["end"] >= p["end"]))


def match(gold, preds, strict=False, contain=False):
    """One-to-one matches {gold index: pred index}: best IoU first."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(preds):
            if strict:
                ok = p["start"] == g["start"] and p["end"] == g["end"]
            elif contain and "head" not in g:
                ok = contains(p, g)
            elif "head" in g:
                ok = head_match(p, g)
            else:
                ok = iou(p, g) >= 0.5
            if ok:
                pairs.append((iou(p, g), gi, pi))
    out, used = {}, set()
    for _, gi, pi in sorted(pairs, reverse=True):
        if gi not in out and pi not in used:
            out[gi] = pi
            used.add(pi)
    return out


def item_counts(it, preds):
    """Raw counts for one item (summed later over items and bootstrap samples)."""
    preds = [p for p in preds if 0 <= p["start"] < p["end"] <= len(it["text"])]
    req = it["required"]
    m = match(req, preds)
    ms = match(req, preds, strict=True)
    c = collections.Counter()
    matched_preds = set(m.values())
    for gi, g in enumerate(req):
        cls = g["class"]
        c[f"{cls}_gold"] += 1
        if gi in m:
            c[f"{cls}_tp"] += 1
            if g.get("kind") and g.get("kind_source") == "human":
                c["kind_n"] += 1
                c["kind_ok"] += preds[m[gi]].get("kind") == g["kind"]
        if gi in ms:
            c[f"{cls}_tp_strict"] += 1
    topics = [g for g in req if g["class"] == "topic"]
    c["topic_tp_contain"] += len(match(topics, preds, contain=True))   # reported variant, not in the headline
    if it["scope"] == "names":
        named_mentions = [x for x in it.get("mentions", []) if x["named"]]
        for pi, p in enumerate(preds):
            if p.get("kind") not in NAME_KINDS:
                continue
            if pi in matched_preds:
                gi = next(g for g, q in m.items() if q == pi)
                if req[gi]["class"] == "name":
                    c["name_pred_tp"] += 1
                continue
            if it["component"] == "gum":
                neutral = any(overlaps(p, x["start"], x["end"]) for x in named_mentions)
            else:
                neutral = any(overlaps(p, o["start"], o["end"]) for o in it["optional"])
            if not neutral:
                c["name_fp"] += 1
    if it["component"] == "gum":
        mentions = it.get("mentions", [])
        for p in preds:
            c["gum_preds"] += 1
            touches = lambda s: s and p["start"] < s[1] and s[0] < p["end"]
            if not any(touches(x["head"]) or touches(x.get("name_span")) for x in mentions):
                c["gum_junk"] += 1   # touches no mention's head word and no name
            if p.get("kind") == "CONCEPT":
                c["gum_concept"] += 1
                c["gum_concept_salient"] += any(x["human_salient"] and p["start"] < x["head"][1] and x["head"][0] < p["end"]
                                                for x in mentions)
    c["preds"] += len(preds)
    return c


def metrics(cs):
    """Component metrics from summed counts."""
    def div(a, b):
        return a / b if b else None
    out = {"name_recall": div(cs["name_tp"], cs["name_gold"]), "name_recall_strict": div(cs["name_tp_strict"], cs["name_gold"]),
           "topic_recall": div(cs["topic_tp"], cs["topic_gold"]), "topic_recall_strict": div(cs["topic_tp_strict"], cs["topic_gold"]),
           "topic_recall_contain": div(cs["topic_tp_contain"], cs["topic_gold"]),
           "kind_accuracy": div(cs["kind_ok"], cs["kind_n"])}
    if cs["name_pred_tp"] + cs["name_fp"] or "scope_names" in cs:
        p = div(cs["name_pred_tp"], cs["name_pred_tp"] + cs["name_fp"])
        r = out["name_recall"]
        out["name_precision"] = p
        out["name_f1"] = 0.0 if r == 0 else ((2 * p * r / (p + r) if p and r else 0.0) if p is not None and r is not None else None)
    if cs["gum_preds"]:
        out["junk_rate"] = div(cs["gum_junk"], cs["gum_preds"])
        out["topic_precision_proxy"] = div(cs["gum_concept_salient"], cs["gum_concept"])
    out["gold_names"], out["gold_topics"], out["predictions"] = cs["name_gold"], cs["topic_gold"], cs["preds"]
    return out


def headline(comp):
    """The scorecard's summary numbers from per-component metrics."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    h = {"name_f1": mean([comp.get(c, {}).get("name_f1") for c in NAME_F1_COMPS]),
         "name_recall": mean([comp.get(c, {}).get("name_recall") for c in COMPONENTS]),
         "topic_recall": mean([comp.get(c, {}).get("topic_recall") for c in TOPIC_COMPS]),
         "topic_recall_contain": mean([comp.get(c, {}).get("topic_recall_contain") for c in TOPIC_COMPS]),
         "junk_rate": comp.get("gum", {}).get("junk_rate"),
         "topic_precision_proxy": comp.get("gum", {}).get("topic_precision_proxy")}
    ka = [(comp[c]["_kind_ok"], comp[c]["_kind_n"]) for c in comp if "_kind_n" in comp[c]]
    h["kind_accuracy"] = sum(a for a, _ in ka) / sum(b for _, b in ka) if ka and sum(b for _, b in ka) else None
    h["hab1"] = mean([h["name_f1"], h["topic_recall"]]) if h["name_f1"] is not None and h["topic_recall"] is not None else None
    return h


def evaluate(items, pred, groups_sample=None):
    """{component: metrics} and the headline, optionally on a bootstrap sample of groups (a Counter of group ids)."""
    per = collections.defaultdict(collections.Counter)
    for it in items:
        w = 1 if groups_sample is None else groups_sample.get(it["group"], 0)
        if not w:
            continue
        c = item_counts(it, pred.get(it["id"], []))
        for k, v in c.items():
            per[it["component"]][k] += v * w
        if it["scope"] == "names":
            per[it["component"]]["scope_names"] = 1
    comp = {}
    for name, cs in per.items():
        comp[name] = metrics(cs)
        comp[name]["_kind_ok"], comp[name]["_kind_n"] = cs["kind_ok"], cs["kind_n"]
    return comp, headline(comp)


def boot_samples(items, seed=2210):
    rng = random.Random(seed)
    by = collections.defaultdict(set)
    for it in items:
        by[it["component"]].add(it["group"])
    by = {c: sorted(g) for c, g in by.items()}
    for _ in range(N_BOOT):
        s = collections.Counter()
        for c, gs in by.items():
            s.update(rng.choice(gs) for _ in gs)
        yield s


def ci(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


def score(pred_path, which="core", out_json=None, quiet=False):
    items = load_items(which)
    pred = load_pred(pred_path)
    comp, head = evaluate(items, pred)
    samples = list(boot_samples(items))
    boots = [evaluate(items, pred, s)[1] for s in samples]
    head_ci = {k: ci([b[k] for b in boots]) for k in head}
    res = {"predictions": str(pred_path), "set": which, "items": len(items), "covered": sum(1 for x in items if x["id"] in pred),
           "headline": {k: (round(v, 4) if v is not None else None) for k, v in head.items()}, "headline_ci95": head_ci,
           "components": {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not k.startswith("_")}
                          for c, m in sorted(comp.items(), key=lambda kv: COMPONENTS.index(kv[0]))}}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    if not quiet:
        print_report(res)
    return res


def f(v):
    return "  —  " if v is None else f"{v:.3f}"


def print_report(res):
    print(f"HAB-1 ({res['set']}: {res['items']} items, {res['covered']} with predictions) — {res['predictions']}")
    h, c = res["headline"], res["headline_ci95"]
    for k in ("hab1", "name_f1", "name_recall", "topic_recall", "topic_recall_contain", "kind_accuracy",
              "topic_precision_proxy", "junk_rate"):
        print(f"  {k:22} {f(h[k])}   95% {c[k]}")
    cols = ["name_precision", "name_recall", "name_f1", "name_recall_strict", "topic_recall", "kind_accuracy", "gold_names", "gold_topics"]
    print(f"  {'component':14}" + "".join(f"{x[:14]:>15}" for x in cols))
    for name, m in res["components"].items():
        print(f"  {name:14}" + "".join(f"{(f(m.get(x)) if not isinstance(m.get(x), int) else str(m.get(x))):>15}" for x in cols))


def compare(a_path, b_path, which="core"):
    items = load_items(which)
    a, b = load_pred(a_path), load_pred(b_path)
    ha, hb = evaluate(items, a)[1], evaluate(items, b)[1]
    diffs = collections.defaultdict(list)
    for s in boot_samples(items):
        xa, xb = evaluate(items, a, s)[1], evaluate(items, b, s)[1]
        for k in ha:
            if xa[k] is not None and xb[k] is not None:
                diffs[k].append(xb[k] - xa[k])
    print(f"B minus A on HAB-1 {which} ({len(items)} items); A = {a_path}, B = {b_path}")
    out = {}
    for k in ("hab1", "name_f1", "name_recall", "topic_recall", "topic_recall_contain", "kind_accuracy",
              "topic_precision_proxy", "junk_rate"):
        d = (hb[k] - ha[k]) if ha[k] is not None and hb[k] is not None else None
        out[k] = {"a": ha[k], "b": hb[k], "diff": d, "ci95": ci(diffs[k])}
        print(f"  {k:22} A {f(ha[k])}  B {f(hb[k])}  B-A {('—' if d is None else f'{d:+.3f}'):>7}   95% {ci(diffs[k])}")
    return out


def baseline(kind, out, which="core"):
    """Reference systems: the gold itself (every required span, with its kind), nothing, and every run of
    capitalised words as a PERSON (a crude name finder that shows what the guardrails catch)."""
    with open(out, "w", encoding="utf-8") as fh:
        for it in load_items(which):
            if kind == "gold":
                anchors = [{"start": s["start"], "end": s["end"], "kind": s["kind"] or ("CONCEPT" if s["class"] == "topic" else "OTHER")}
                           for s in it["required"]]
            elif kind == "empty":
                anchors = []
            else:
                anchors = [{"start": m.start(), "end": m.end(), "kind": "PERSON"}
                           for m in re.finditer(r"\b[A-Z][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)*", it["text"]) if m.start() > 0]
            fh.write(json.dumps({"id": it["id"], "anchors": anchors}) + "\n")
    print(f"{kind} baseline -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("pred")
    s.add_argument("--set", default="core", choices=["core", "all"])
    s.add_argument("--json")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--set", default="core", choices=["core", "all"])
    b = sub.add_parser("baseline")
    b.add_argument("kind", choices=["gold", "empty", "capitalised"])
    b.add_argument("out")
    b.add_argument("--set", default="core", choices=["core", "all"])
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.pred, a.set, a.json)
    elif a.cmd == "compare":
        compare(a.a, a.b, a.set)
    else:
        baseline(a.kind, a.out, a.set)


if __name__ == "__main__":
    sys.exit(main())
