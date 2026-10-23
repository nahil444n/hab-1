"""HAB-1, the Human Anchor Benchmark: an independent test of anchor finding, built only from labels people made
(README.md has the design and the reasons).

  python build.py            # after python download.py; uses the cached Wikidata types in data/

Seven public datasets, downloaded to sources/ by download.py (pinned versions, checked by SHA-256), become one format:
  tweetner7      TweetNER7 test_2021: tweets, 7 name types, exhaustive (names)
  wnut17         WNUT-17 test: Reddit, StackExchange and other social text, 6 name types, exhaustive (names)
  gum            GUM v12 conversation/interview/podcast/vlog: every mention marked, plus what the human summary mentions
  ccpe           CCPE-M: spoken chats between two people about films; names and film genres
  conel          ConEL-2 and ConEL: chats linked to Wikipedia; recall of names only (concept links are optional)
  reddit         Reddit Entity Linking, gold and silver (2 or 3 of 3 workers agree): recall of names only
  stackoverflow  StackOverflow NER test: software names exhaustive; algorithms and data structures as topics

Every item has the text to label, up to three earlier units of its conversation or document as context, and:
  required  spans a system should mark: class "name" (kind, or null when the source's type has no clear kind) or
            "topic" (kind CONCEPT); kind_source "human" (the source's own type) or "wikidata" (ConEL and Reddit, from the
            linked page's class: used to tell names from concepts, never to score kinds)
  optional  spans the source marks that are neither required nor errors (concept links, code identifiers,
            other mentions): a prediction overlapping one is not counted either way
  scope     "names": the source marks every name, so a name-kind prediction that matches nothing is a false positive;
            "none": the source's labels are partial, so only recall is measured
  match     "iou": a prediction matches a span when their character IoU is at least 0.5 (exact match is also
            reported); "head" (GUM topics): it overlaps the mention and covers its head word
GUM items also list every mention with its head, for the junk rate and the topic precision proxy (hab.py).

Writes data/items.jsonl (all units), core_ids.txt (the seeded sample every system labels),
blind_core.jsonl (core text and context only, for labellers), wikidata_types.json (the cache) and manifest.json.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "sources"
OUT = ROOT / "data"
SEED = 2210
CONTEXT_UNITS, CONTEXT_CHARS = 3, 300
CORE = {"tweetner7": 300, "wnut17": 300, "gum": 360, "ccpe": 300, "conel": 250, "reddit": 250, "stackoverflow": 300}
UA = "HAB-1 benchmark build (https://github.com/nahil444n/hab-1)"


def item(iid, comp, group, text, context, required, optional, scope, match, **extra):
    return {"id": iid, "component": comp, "group": group, "text": text,
            "context": [c[:CONTEXT_CHARS] for c in context[-CONTEXT_UNITS:]],
            "required": sorted(required, key=lambda s: (s["start"], s["end"])),
            "optional": sorted(optional, key=lambda s: (s["start"], s["end"])), "scope": scope, "match": match, **extra}


def span(start, end, cls=None, kind=None, src=None, **extra):
    d = {"start": start, "end": end}
    if cls:
        d["class"] = cls
        d["kind"] = kind
        d["kind_source"] = "human"
    if src:
        d["source_type"] = src
    return {**d, **extra}


def join_tokens(tokens):
    """Space-joined text and each token's character offsets."""
    offs, pos = [], 0
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    return " ".join(tokens), offs


def bio(tags):
    """[(start token, end token exclusive, type)] from BIO tags ('B-x', 'I-x', 'O')."""
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur[2] != t[2:])):
            if cur:
                out.append((cur[0], i, cur[2]))
                cur = None
            if t.startswith("B-") or t.startswith("I-"):
                cur = [i, None, t[2:]]
    return out


# ---- names-exhaustive social text ----
TW_KIND = {"corporation": "ORG", "creative_work": "WORK", "event": "EVENT", "group": "ORG", "location": "LOCATION",
           "person": "PERSON", "product": "PRODUCT"}


def tweetner7():
    labels = {v: k for k, v in json.load(open(EXT / "tweetner7/label.json")).items()}

    def clean(t):
        if t.startswith("{@") and t.endswith("@}"):
            return t[2:-2]
        return {"{{URL}}": "https://t.co/x", "{{USERNAME}}": "@user"}.get(t, t)
    for line in open(EXT / "tweetner7/2021.test.json", encoding="utf-8"):
        r = json.loads(line)
        text, offs = join_tokens([clean(t) for t in r["tokens"]])
        req = [span(offs[s][0], offs[e - 1][1], "name", TW_KIND[typ], typ) for s, e, typ in bio([labels[t] for t in r["tags"]])]
        yield item(f"tw-{r['id']}", "tweetner7", f"tw-{r['id']}", text, [], req, [], "names", "iou")


WN_KIND = {"person": "PERSON", "location": "LOCATION", "corporation": "ORG", "group": "ORG", "creative-work": "WORK",
           "product": "PRODUCT"}


def wnut17():
    sents, cur = [], []
    for line in open(EXT / "wnut17/emerging.test.annotated", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        tok, tag = line.split("\t")[:2]
        cur.append((tok, tag))
    if cur:
        sents.append(cur)
    for n, s in enumerate(sents):
        text, offs = join_tokens([t for t, _ in s])
        req = [span(offs[a][0], offs[b - 1][1], "name", WN_KIND[typ], typ) for a, b, typ in bio([g for _, g in s])]
        yield item(f"wn-{n:04d}", "wnut17", f"wn-{n:04d}", text, [], req, [], "names", "iou")


# ---- GUM ----
ENT = re.compile(r"\((\d+)-([^()]*)|(\d+)\)")
GUM_NAME_KIND = {"person": "PERSON", "organization": "ORG", "place": "LOCATION", "event": "EVENT"}
GUM_TOPIC_ETYPES = {"abstract", "object", "substance", "event", "animal", "plant"}


def gum_doc(path):
    """Sentences of one GUM document: text, tokens with character spans, and every entity mention."""
    sents, meta = [], {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# meta::") and " = " in line:
            k, v = line[8:].split(" = ", 1)
            meta[k] = v
        elif line.startswith("# sent_id = "):
            cur = {"id": line[12:], "tokens": [], "mwt": {}, "speaker": None}
        elif line.startswith("# speaker = ") and cur is not None:
            cur["speaker"] = line[12:]
        elif line.startswith("# text = ") and cur is not None:
            cur["text"] = line[9:]
        elif not line and cur is not None:
            sents.append(cur)
            cur = None
        elif line and not line.startswith("#") and cur is not None:
            f = line.split("\t")
            if "-" in f[0]:
                a, b = map(int, f[0].split("-"))
                cur["mwt"][a] = (b, f[1])
            elif "." not in f[0]:
                cur["tokens"].append({"id": int(f[0]), "form": f[1], "upos": f[3], "head": int(f[6]) if f[6].isdigit() else 0,
                                      "misc": f[9]})
    for s in sents:
        text, pos, i = s["text"], 0, 0
        toks = s["tokens"]
        while i < len(toks):
            t = toks[i]
            if t["id"] in s["mwt"]:   # a multiword token: its surface form spans all its parts
                end_id, form = s["mwt"][t["id"]]
                k = text.find(form, pos)
                k = pos if k < 0 else k
                for u in toks[i:]:
                    if u["id"] > end_id:
                        break
                    u["span"] = (k, k + len(form))
                    i += 1
                pos = k + len(form)
                continue
            k = text.find(t["form"], pos)
            k = pos if k < 0 else k
            t["span"] = (k, k + len(t["form"]))
            pos = k + len(t["form"])
            i += 1
        by_id = {t["id"]: t for t in toks}
        open_m, mentions = {}, []
        for t in toks:
            m = re.search(r"Entity=([^|]*)", t["misc"])
            if not m:
                continue
            for g in ENT.finditer(m.group(1)):
                if g.group(1):   # an opening "(id-etype-infstat-salience-centering-minspan-link[-identity]"
                    attrs = g.group(2).split("-")
                    mid = g.group(1)
                    rec = {"eid": mid, "etype": attrs[0], "salience": attrs[2] if len(attrs) > 2 else "",
                           "identity": "-".join(attrs[6:]) if len(attrs) > 6 else "", "first": t["id"]}
                    close_now = g.end() < len(m.group(1)) and m.group(1)[g.end()] == ")"
                    if close_now:
                        rec["last"] = t["id"]
                        mentions.append(rec)
                    else:
                        open_m.setdefault(mid, []).append(rec)
                else:
                    mid = g.group(3)
                    if open_m.get(mid):
                        rec = open_m[mid].pop()
                        rec["last"] = t["id"]
                        mentions.append(rec)
        out = []
        for rec in mentions:
            ids = [i for i in range(rec["first"], rec["last"] + 1) if i in by_id]
            if not ids:
                continue
            heads = [i for i in ids if by_id[i]["head"] not in ids]
            h = by_id[heads[0] if heads else ids[-1]]
            named = bool(rec["identity"]) or h["upos"] == "PROPN"
            name_span = None
            if named:   # the run of proper-noun tokens that is the name ("Noam Chomsky" in "US linguist Noam Chomsky")
                runs, run = [], []
                for i in ids:
                    if by_id[i]["upos"] == "PROPN" or (run and by_id[i]["form"] in {"of", "&", "the", "de", "-"} and
                                                       i + 1 in by_id and by_id[i + 1]["upos"] == "PROPN"):
                        run.append(i)
                    elif run:
                        runs.append(run)
                        run = []
                if run:
                    runs.append(run)
                if runs:
                    ident = set(re.split(r"[_ ]+", rec["identity"].lower())) if rec["identity"] else set()
                    best = max(runs, key=lambda r: (h["id"] in r, len(ident & {by_id[i]["form"].lower() for i in r}), len(r)))
                    name_span = (by_id[best[0]]["span"][0], by_id[best[-1]]["span"][1])
            out.append({"start": by_id[ids[0]]["span"][0], "end": by_id[ids[-1]]["span"][1], "etype": rec["etype"],
                        "head": h["span"], "head_upos": h["upos"], "named": named, "name_span": name_span,
                        "human_salient": rec["salience"][:1] == "s", "salience": rec["salience"].count("s")})
        s["mentions"] = out
    return sents, meta


def gum():
    for path in sorted((EXT / "gum/dep").glob("GUM_*.conllu")):
        doc = path.stem
        genre = doc.split("_")[1]
        sents, _ = gum_doc(path)
        texts = [s["text"] for s in sents]
        for n, s in enumerate(sents):
            req, opt, seen = [], [], set()
            for m in s["mentions"]:
                if m["named"] and m["name_span"] and m["etype"] not in {"time", "quantity"}:
                    key = ("n",) + m["name_span"]
                    if key not in seen:
                        seen.add(key)
                        req.append(span(*m["name_span"], "name", GUM_NAME_KIND.get(m["etype"]), "gum-" + m["etype"]))
                elif (not m["named"] and m["human_salient"] and m["head_upos"] == "NOUN" and m["etype"] in GUM_TOPIC_ETYPES):
                    key = ("t",) + tuple(m["head"])
                    if key not in seen:
                        seen.add(key)
                        req.append(span(m["start"], m["end"], "topic", "CONCEPT", "gum-salient-" + m["etype"], head=list(m["head"])))
                else:
                    opt.append(span(m["start"], m["end"], src="gum-" + m["etype"]))
            mentions = [{"start": m["start"], "end": m["end"], "head": list(m["head"]), "named": m["named"],
                         "name_span": list(m["name_span"]) if m["name_span"] else None,
                         "human_salient": m["human_salient"], "etype": m["etype"]} for m in s["mentions"]]
            yield item(f"gum-{s['id']}", "gum", doc, s["text"], texts[max(0, n - CONTEXT_UNITS):n], req, opt, "names", "iou",
                       genre=genre, speaker=s.get("speaker"), mentions=mentions)


# ---- CCPE-M ----
CC_KIND = {"MOVIE_OR_SERIES": ("name", "WORK"), "PERSON": ("name", "PERSON"), "SOMETHING_ELSE": ("name", None),
           "MOVIE_GENRE_OR_CATEGORY": ("topic", "CONCEPT")}


def ccpe():
    for conv in json.load(open(EXT / "ccpe/data.json", encoding="utf-8")):
        utts = sorted(conv["utterances"], key=lambda u: u["index"])
        history = []
        for u in utts:
            text = u["text"]
            req, seen = [], set()
            for seg in u.get("segments", []):
                if not any(a["annotationType"] == "ENTITY_NAME" for a in seg["annotations"]):
                    continue
                s, e = seg["startIndex"], seg["endIndex"]
                if text[s:e] != seg["text"]:   # a few offsets are off: find the text near them
                    k = text.find(seg["text"], max(0, s - 5))
                    if k < 0:
                        continue
                    s, e = k, k + len(seg["text"])
                et = next(a["entityType"] for a in seg["annotations"] if a["annotationType"] == "ENTITY_NAME")
                if (s, e) in seen or et not in CC_KIND:
                    continue
                seen.add((s, e))
                cls, kind = CC_KIND[et]
                req.append(span(s, e, cls, kind, et))
            yield item(f"cc-{conv['conversationId']}-{u['index']:02d}", "ccpe", conv["conversationId"], text,
                       [f"{h['speaker'].lower()}: {h['text']}" for h in history], req, [], "names", "iou", speaker=u["speaker"])
            history.append(u)


# ---- Wikidata types for the linking datasets ----
# Wikidata classes for each kind, in the order the conventions want: a country or city is a place before it is an
# organisation (Wikidata puts states under "organization" too), a company, school, band or league is an organisation,
# and a building, facility or bare "location" is a place only when nothing else fits (museums, venues, airports).
TOP = {"PERSON": ["Q5", "Q95074", "Q15632617"],
       "GEO": ["Q6256", "Q3624078", "Q515", "Q486972", "Q56061", "Q82794", "Q618123"],
       "ORG": ["Q43229"],
       "WORK": ["Q17537576", "Q7725634", "Q11424", "Q5398426", "Q482994", "Q7366", "Q7889", "Q47461344"],
       "EVENT": ["Q1656682"],
       "PRODUCT": ["Q2424752", "Q7397", "Q9135", "Q35127", "Q9143"],
       "FACILITY": ["Q41176", "Q13226383", "Q17334923", "Q2221906"]}
PRIORITY = ["PERSON", "GEO", "ORG", "WORK", "EVENT", "PRODUCT", "FACILITY"]
KIND_OF = {"GEO": "LOCATION", "FACILITY": "LOCATION"}
TOP_GROUP = {q: g for g, qs in TOP.items() for q in qs}
CACHE_VERSION = 3
PRONOUNS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us",
            "our", "they", "them", "their", "this", "that", "these", "those", "who", "which"}
SMALL = {"of", "the", "and", "in", "on", "a", "an", "for", "to", "de", "la", "le", "von", "van", "&", "at", "by", "du"}


def proper_title(title):
    """Does a Wikipedia title look like a proper name? Several words all capitalised ("Game of Thrones"), or one word
    with a capital inside it or in capitals ("IPhone", "NFL"). Concepts are "Sugar", "Whale meat"."""
    words = re.sub(r"_\(.*\)$", "", title).replace("_", " ").split()
    if len(words) >= 2:
        return all(w[:1].isupper() or w.lower() in SMALL or not w[:1].isalpha() for w in words)
    w = words[0] if words else ""
    return len(w) >= 2 and (w.isupper() or any(ch.isupper() for ch in w[1:]))


def classify(title, mention, wd):
    """(named: True / False / None, kind) for a mention linked to a Wikipedia title.
    - named: an item (not a class) whose instance-of path reaches one of TOP's classes, or a class that is a product
      line, franchise or organisation with a proper-name title (iPhone, Harry Potter);
    - concept (False): any other class (sugar, hotel, dating);
    - unknown (None): anything else, and a lowercase mention of a one-word title ("hello", "world"), which may well be
      the common word. Unknown spans are optional: neither required nor errors."""
    v = wd.get(title)
    if not v or not v.get("qid"):
        return None, None
    groups = {TOP_GROUP[q] for q in v["tops"] if q in TOP_GROUP}
    group = next((g for g in PRIORITY if g in groups), None)
    kind = KIND_OF.get(group, group)
    if group and (not v["is_class"] or (group in ("PRODUCT", "WORK", "ORG") and proper_title(title))):
        if not any(ch.isupper() for ch in mention) and not (len(re.sub(r"_\(.*\)$", "", title).split("_")) >= 2 and proper_title(title)):
            return None, kind
        return True, kind
    return (False, None) if v["is_class"] else (None, None)


def _get(url, params, accept="application/json"):
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA, "Accept": accept})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def wikidata_types(titles, cache_path):
    """{Wikipedia title: {"qid", "tops": [TOP classes its instance-of path reaches], "is_class"}}, cached; classify()
    turns an entry into a name or a concept, so changing the rules needs no new queries."""
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    if cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "items": {}}
    items = cache["items"]
    todo = sorted({t for t in titles if t and t not in items})
    qid = {}
    for k in range(0, len(todo), 50):
        batch = todo[k:k + 50]
        d = _get("https://en.wikipedia.org/w/api.php", {"action": "query", "titles": "|".join(batch), "prop": "pageprops",
                                                         "ppprop": "wikibase_item", "redirects": 1, "format": "json"})
        to_final = {}
        for kind in ("normalized", "redirects"):
            for x in d["query"].get(kind, []):
                to_final[x["from"]] = x["to"]
        final_q = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in d["query"]["pages"].values()}
        for asked in batch:
            name = asked
            for _ in range(3):   # asked -> normalised -> redirect target
                name = to_final.get(name, name)
            qid[asked] = final_q.get(name)
        time.sleep(0.5)
    found = sorted({q for q in qid.values() if q})
    tops, is_class = collections.defaultdict(set), set()
    for k in range(0, len(found), 80):
        vals = " ".join(f"wd:{q}" for q in found[k:k + 80])
        r = _get("https://query.wikidata.org/sparql", {"query": f"SELECT ?item ?top WHERE {{ VALUES ?item {{ {vals} }} VALUES ?top "
                 f"{{ {' '.join('wd:' + q for q in TOP_GROUP)} }} ?item wdt:P31/wdt:P279* ?top }}", "format": "json"},
                 "application/sparql-results+json")
        for b in r["results"]["bindings"]:
            tops[b["item"]["value"].rsplit("/", 1)[1]].add(b["top"]["value"].rsplit("/", 1)[1])
        r = _get("https://query.wikidata.org/sparql", {"query": f"SELECT DISTINCT ?item WHERE {{ VALUES ?item {{ {vals} }} "
                 "?item wdt:P279 ?x }", "format": "json"}, "application/sparql-results+json")
        is_class |= {b["item"]["value"].rsplit("/", 1)[1] for b in r["results"]["bindings"]}
        time.sleep(1.0)
    for t in todo:
        q = qid.get(t)
        items[t] = {"qid": q, "tops": sorted(tops.get(q, set())), "is_class": q in is_class}
    json.dump(cache, open(cache_path, "w"), indent=0, sort_keys=True)
    return items


# ---- ConEL-2 and ConEL ----
def _conel_raw():
    """(source, dialogue id, turns) for ConEL-2's three splits and ConEL's concept/named-entity set."""
    for split in ("Train", "Val", "Test"):
        for d in json.load(open(EXT / f"conel2/ConEL22_EL_{split}.json", encoding="utf-8")):
            yield "conel2", d["dialogue_id"], d["turns"]
    for d in json.load(open(EXT / "conel/ConEL_CNE.json", encoding="utf-8")):
        yield "conel1", f"{d.get('dataset_name', '')}-{d['dialogue_id']}", d["turns"]


def conel_titles():
    return {a["entity"] for _, _, turns in _conel_raw() for t in turns for a in t.get("el_annotations", []) if a.get("entity")}


def conel(types):
    seen_text = set()
    for src, did, turns in _conel_raw():
        turns = sorted(turns, key=lambda t: t["turn_number"])
        texts = []
        for t in turns:
            text = t.get("utterance")
            if not text:   # TREC CAsT system turns hold a passage id only
                continue
            if t["speaker"] == "USER" and "el_annotations" in t and text.strip().lower() not in seen_text:
                seen_text.add(text.strip().lower())
                req, opt = [], []
                for a in t["el_annotations"]:
                    if "span" in a:
                        s, e = a["span"]
                    else:   # ConEL (2021) gives the mention only: its first occurrence
                        s = text.find(a["mention"])
                        if s < 0:
                            continue
                        e = s + len(a["mention"])
                    named, kind = classify(a.get("entity", ""), text[s:e], types)   # ConEL's entity_type is unreliable
                    if named and text[s:e].strip() and text[s:e].strip().lower() not in PRONOUNS:
                        req.append(span(s, e, "name", kind, f"{src}-named", kind_source="wikidata"))
                    else:
                        opt.append(span(s, e, src=f"{src}-{'concept' if named is False else 'unknown'}"))
                for p in t.get("personal_entity_annotations", []) + t.get("personal_entity_annotations_without_eems", []):
                    k = text.find(p.get("personal_entity_mention", "\0"))
                    if k >= 0:
                        opt.append(span(k, k + len(p["personal_entity_mention"]), src="conel-personal"))
                yield item(f"{src}-{did}-{t['turn_number']:02d}", "conel", f"{src}-{did}", text, texts, req, opt, "none", "iou")
            texts.append(text)


# ---- Reddit Entity Linking ----
def _reddit_raw():
    base = EXT / "reddit_el/extracted"
    posts = {}
    for line in open(base / "posts.tsv", encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            posts[f[0]] = {"subreddit": f[1], "text": "\t".join(f[2:])}
    raw = open(base / "comments.tsv", encoding="utf-8").read()
    comments = {}
    for rec in re.split(r"\n(?=[a-z0-9]{5,9}\tt[13]_[a-z0-9]+\t[a-z0-9]+\t)", raw):
        f = rec.split("\t", 4)
        if len(f) == 5:
            comments[f[0]] = {"parent": f[1], "link": f[2], "subreddit": f[3], "text": f[4]}
    ann = collections.defaultdict(list)
    for tier in ("gold", "silver", "bronze"):
        for kind, pool in (("post", posts), ("comment", comments)):
            for line in open(base / f"{tier}_{kind}_annotations.tsv", encoding="utf-8"):
                f = line.rstrip("\n").split("\t")
                if len(f) >= 6 and f[4].isdigit() and f[0] in pool:
                    ann[f[0]].append({"tier": tier, "mention": f[2], "entity": f[3], "start": int(f[4]), "end": int(f[5])})
    return posts, comments, ann


def reddit_titles():
    _, _, ann = _reddit_raw()
    return {a["entity"] for v in ann.values() for a in v if a["tier"] != "bronze"}


def reddit(types):
    posts, comments, ann = _reddit_raw()
    units = [(pid, p["text"], [], pid) for pid, p in posts.items()] + \
            [(cid, c["text"], [posts.get(c["link"], {}).get("text", "")], c["link"]) for cid, c in comments.items()]
    for uid, text, ctx, thread in units:
        if uid not in ann:
            continue
        req, opt, seen = [], [], set()
        for a in ann[uid]:
            s, e = a["start"], a["end"]
            if text[s:e] != a["mention"]:
                k = text.find(a["mention"], max(0, s - 3))
                if k < 0:
                    continue
                s, e = k, k + len(a["mention"])
            if (s, e) in seen:
                continue
            seen.add((s, e))
            named, kind = classify(a["entity"], text[s:e], types)
            if a["tier"] != "bronze" and named and text[s:e].strip().lower() not in PRONOUNS:
                req.append(span(s, e, "name", kind, f"reddit-{a['tier']}", kind_source="wikidata"))
            else:
                opt.append(span(s, e, src=f"reddit-{a['tier']}-{'concept' if named is False else 'other'}"))
        yield item(f"rd-{uid}", "reddit", thread, text, [c for c in ctx if c], req, opt, "none", "iou")


# ---- StackOverflow NER ----
SO_NAME = {"Application": "PRODUCT", "Library": "PRODUCT", "Operating_System": "PRODUCT", "Device": "PRODUCT",
           "Website": "PRODUCT", "Organization": "ORG", "Language": None}
SO_TOPIC = {"Algorithm", "Data_Structure"}
SO_SKIP = {"Code_Block", "Output_Block"}


def stackoverflow():
    sents, cur, qid = [], [], "q0"
    for line in open(EXT / "stackoverflow_ner/test.txt", encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if not line.strip():
            if cur:
                sents.append((qid, cur))
                cur = []
            continue
        if len(f) >= 2 and f[0]:
            cur.append((f[0], f[1]))
    if cur:
        sents.append((qid, cur))
    out, q = [], "q0"
    for _, s in sents:
        toks = [t for t, _ in s]
        if toks[:1] in (["Question_ID"], ["Answer_to_Question_ID"]) and len(toks) >= 3:
            q = toks[2]
            continue
        if toks[:1] in (["Question_URL"], ["Answer_ID"], ["Question_ID"]) or any(g[2:] in SO_SKIP for _, g in s if g != "O"):
            continue
        if len(toks) < 3:
            continue
        out.append((q, s))
    by_q = collections.defaultdict(list)
    for n, (q, s) in enumerate(out):
        text, offs = join_tokens([t for t, _ in s])
        req, opt = [], []
        for a, b, typ in bio([g for _, g in s]):
            sp = (offs[a][0], offs[b - 1][1])
            if typ in SO_NAME:
                req.append(span(*sp, "name", SO_NAME[typ], typ))
            elif typ in SO_TOPIC:
                req.append(span(*sp, "topic", "CONCEPT", typ))
            else:
                opt.append(span(*sp, src=typ))
        yield item(f"so-{n:05d}", "stackoverflow", f"so-{q}", text, list(by_q[q]), req, opt, "names", "iou")
        by_q[q].append(text)


# ---- the core sample ----
def core_sample(items):
    """The units every system labels. Units with anchors are over-drawn where they are rare, so topics get enough
    weight; units without any keep false alarms in the count (precision needs them)."""
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for it in items:
        by[it["component"]].append(it)
    has = lambda x, c: any(s["class"] == c for s in x["required"])
    core = []

    def draw(cand, n):
        cand = [x for x in cand if x["id"] not in picked]
        got = rng.sample(cand, min(n, len(cand)))
        picked.update(x["id"] for x in got)
        return got
    for comp, n in CORE.items():
        pool = sorted(by[comp], key=lambda x: x["id"])
        picked = set()
        if comp == "gum":   # per genre: two thirds with an anchor, one third any sentence of at least four words
            for g in ("conversation", "interview", "podcast", "vlog"):
                cand = [x for x in pool if x["genre"] == g and len(x["text"].split()) >= 4]
                core += draw([x for x in cand if x["required"]], 2 * n // 12) + draw(cand, n // 12)
        elif comp in ("conel", "reddit"):   # recall only: units with at least one required name
            core += draw([x for x in pool if x["required"]], n)
        elif comp in ("ccpe", "stackoverflow"):   # a third with a topic, a third with a name, a third any
            core += draw([x for x in pool if has(x, "topic")], n // 3) + draw([x for x in pool if has(x, "name")], n // 3)
            core += draw(pool, n - 2 * (n // 3))
        else:
            core += draw(pool, n)
    return sorted(x["id"] for x in core)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "wikidata_types.json"
    titles = conel_titles() | reddit_titles()
    types = wikidata_types(titles, cache)
    items = []
    for fn in (tweetner7, wnut17, gum, ccpe):
        items += list(fn())
    items += list(conel(types)) + list(reddit(types))
    items += list(stackoverflow())
    for it in items:   # every span must lie inside its text
        for s in it["required"] + it["optional"]:
            assert 0 <= s["start"] < s["end"] <= len(it["text"]), (it["id"], s)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    core = core_sample(items)
    core_set = set(core)
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({**it, "core": it["id"] in core_set}, ensure_ascii=False) + "\n")
    (OUT / "core_ids.txt").write_text("\n".join(core) + "\n")
    with open(OUT / "blind_core.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            if it["id"] in core_set:
                f.write(json.dumps({"id": it["id"], "component": it["component"], "context": it["context"], "text": it["text"]},
                                   ensure_ascii=False) + "\n")
    stats = {}
    for comp in CORE:
        its = [x for x in items if x["component"] == comp]
        cs = [x for x in its if x["id"] in core_set]
        cnt = lambda xs, c: sum(1 for x in xs for s in x["required"] if s["class"] == c)
        stats[comp] = {"units": len(its), "required_names": cnt(its, "name"), "required_topics": cnt(its, "topic"),
                       "optional": sum(len(x["optional"]) for x in its), "core_units": len(cs),
                       "core_names": cnt(cs, "name"), "core_topics": cnt(cs, "topic"),
                       "names_with_kind": sum(1 for x in its for s in x["required"] if s["class"] == "name" and s["kind"])}
    typed = [classify(t, t, types) for t, v in types.items() if v.get("qid")]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    manifest = {"benchmark": "HAB-1", "built": time.strftime("%Y-%m-%d"), "seed": SEED, "core_sizes": CORE,
                "context_units": CONTEXT_UNITS, "context_chars": CONTEXT_CHARS, "components": stats,
                "wikidata": {"titles": len(types), "resolved": len(typed), "named": sum(1 for n, _ in typed if n),
                             "concepts": sum(1 for n, _ in typed if n is False)},
                "sha256_16": {p.name: digest(p) for p in (OUT / "items.jsonl", OUT / "core_ids.txt", OUT / "blind_core.jsonl")}}
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
