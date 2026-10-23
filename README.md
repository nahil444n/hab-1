# HAB-1: the Human Anchor Benchmark

HAB-1 tests **anchor finding**: marking the phrases in a piece of text that should become nodes in a knowledge graph. Those are the names of people, organisations, places, events, works and products, and the topics the text is about. Every label in HAB-1 was made by people, not by a language model. It is built from seven public datasets of conversation, forum, social and technical text.

- **27,897 units**, each a sentence, turn, tweet or post, with up to three earlier units of its conversation as context.
- **A fixed core sample of 2,060 units** that every system labels, holding 2,622 names and 427 topics (`data/core_ids.txt`, drawn with seed 2210).
- **One scorer** (`hab.py`) for any system that returns character spans with a kind.

This repository holds no one else's text. `download.py` fetches each source from its official home, pinned to the exact version HAB-1 was built from and checked by SHA-256. `build.py` then turns the sources into the benchmark on your machine.

## Quick start

Python 3.8 or later, standard library only.

```bash
python download.py   # about 25 MB into sources/, every file checked against sources.json
python build.py      # writes data/items.jsonl, data/blind_core.jsonl and data/manifest.json
```

Label the core units (`data/blind_core.jsonl` has each unit's text and context, without labels) with your system. Write one line per unit:

```json
{"id": "...", "anchors": [{"start": 0, "end": 17, "kind": "PERSON"}, {"start": 82, "end": 93, "kind": "CONCEPT"}]}
```

`start` and `end` are character offsets into the unit's `text`; the context is only for reading. Then score:

```bash
python hab.py score my_system.jsonl --set core --json my_system.score.json
python hab.py compare system_a.jsonl system_b.jsonl --set core     # B minus A, paired, with 95% intervals
```

**Kinds:**
- **Names:** PERSON, ORG, LOCATION, EVENT, WORK, PRODUCT and OTHER.
- **Topics:** CONCEPT.
- **Any other kind** (dates, numbers, or kinds of your own) is never counted against name precision.

Units missing from your file count as empty predictions.

## The seven components

| Component | Text | Who labelled it | What is required | Core units / names / topics |
|---|---|---|---|---|
| `tweetner7` | Tweets, 2020–21 (TweetNER7, test_2021) | 3 crowd workers per tweet | names, 7 types | 300 / 954 / 0 |
| `wnut17` | Reddit, StackExchange and other social text (WNUT-17 test) | 3 crowd workers per sentence | names, 6 types | 300 / 268 / 0 |
| `gum` | Spoken conversation, interviews, podcasts and vlogs (GUM, 63 documents) | linguistics students; one human-written summary per document | names; **topics:** the non-named things the summary mentions | 360 / 307 / 147 |
| `ccpe` | Spoken chats between two people about films (CCPE-M) | paid annotators | film, person and other names; **topics:** film genres | 300 / 163 / 143 |
| `conel` | Chats between two people, and search-style conversations (ConEL-2 and ConEL) | 3 crowd workers per mention | names (classes from Wikidata) | 250 / 322 / 0 |
| `reddit` | Reddit posts and comments (Reddit Entity Linking, gold and silver) | 3 crowd workers | names (classes from Wikidata) | 250 / 424 / 0 |
| `stackoverflow` | StackOverflow questions and answers (StackOverflow NER test) | 4 computer-science students and an adjudicator | software names and organisations; **topics:** algorithms and data structures | 300 / 184 / 137 |

Each source has spans it marks that are neither required nor errors: concept links, code identifiers and other mentions. A prediction on one of those optional spans is not counted either way.

## How it is scored

- **A required span is found** when a prediction matches it:
  - **lenient,** the main number: character overlap of at least half (IoU ≥ 0.5). It forgives "the Beatles" against "Beatles";
  - **strict:** exact boundaries, also reported;
  - **GUM's topics:** GUM marks whole noun phrases, so a topic is found when a prediction overlaps the phrase and covers its head word.
- **Name precision** counts only where the source marks every name. There, a prediction with a name kind that matches nothing is a false positive. CONCEPT predictions never count against names.
- **Topic recall** is counted on three things:
  - CCPE-M's film genres;
  - StackOverflow's algorithms and data structures;
  - the things GUM's human summary mentions.

  No source marks every topic, so there is no topic precision. GUM gives two guardrails against a system that marks every noun:
  - the **topic precision proxy:** the share of CONCEPT predictions on something the summary mentions;
  - the **junk rate:** the share of predictions that touch no mention at all.
- **Kind accuracy** counts only spans whose type a person assigned. ConEL's and Reddit's classes come from Wikidata (`data/wikidata_types.json`, a cache of 3,541 titles). They decide what counts as a name there, but are never scored as kinds.
- **95% intervals** come from 1,000 bootstrap resamples of whole conversations, documents or threads. `compare` resamples both systems the same way.

| Headline number | What it is |
|---|---|
| **HAB-1** | the mean of name F1 and topic recall; for ranking only |
| Name F1 | lenient, the mean over the five components that mark every name |
| Topic recall | the mean over GUM, CCPE-M and StackOverflow |
| Name recall | the mean over all seven components |
| Kind accuracy | over human-typed spans |
| Topic precision proxy, junk rate | GUM's guardrails |

A system is better only if name F1 and topic recall don't fall and at least one of them rises beyond its interval, without the guardrails getting worse.

**Check your setup.** `hab.py baseline gold|empty|capitalised OUT.jsonl --set core` writes three reference systems. Scored on the core, they give:

| Reference system | HAB-1 | Name F1 | Topic recall | Kind accuracy | Junk rate |
|---|---|---|---|---|---|
| the answer key itself | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| marks nothing | 0.000 | 0.000 | 0.000 | — | — |
| every run of capitalised words, as a person | 0.270 | 0.492 | 0.048 | 0.290 | 0.069 |

## What HAB-1 cannot tell you

- **No human labels exist for AI assistants' replies.** HAB-1 covers people chatting, forums, tweets and spoken conversation.
- **Each source keeps its own conventions.** TweetNER7 labels hashtags as names, GUM's summary-salient things include everyday nouns ("dinner"), and CCPE-M marks only film-related names. Every system is judged by the same rules, so comparisons between systems are fair. Absolute numbers aren't comparable with other benchmarks.
- **Topic labels are partial** and come from three different definitions. Read topic recall as a direction, together with the guardrails.
- **Crowd labels have modest agreement in places.** That is why Reddit uses only its gold and silver annotations.
- **Some text is rebuilt from tokens joined by spaces:** tweets, WNUT-17 and StackOverflow ("# YouTube"), so it reads slightly unnaturally.

## Files

| File | What it holds |
|---|---|
| `download.py`, `sources.json` | the 73 source files: their official URLs at fixed versions, and their SHA-256 |
| `build.py` | sources → `data/items.jsonl` (every unit: text, context, required and optional spans, scope), `data/blind_core.jsonl` (core text and context only) and `data/manifest.json` |
| `hab.py` | the scorer |
| `data/core_ids.txt` | the core sample |
| `data/wikidata_types.json` | the Wikidata classes of the ConEL and Reddit titles, so the build needs no network calls |

A build from the pinned sources should give these SHA-256s (first 16 characters) in `data/manifest.json`:
- `items.jsonl` `cd183713529fdf9a`
- `core_ids.txt` `54e9ef450cdf8669`
- `blind_core.jsonl` `f0420c2d02257a77`

## Sources and licences

HAB-1's code and its own files are under the MIT licence (`LICENSE`). The text and labels you download stay under their sources' terms. Please cite the sources you use.

| Source | Where | Licence |
|---|---|---|
| TweetNER7 (Ushio et al., 2022) | [huggingface.co/datasets/tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7) | "other": tweets, under Twitter's terms |
| WNUT-17 (Derczynski et al., 2017) | [github.com/leondz/emerging_entities_17](https://github.com/leondz/emerging_entities_17) | CC BY 4.0 |
| GUM (Zeldes, 2017) | [github.com/amir-zeldes/gum](https://github.com/amir-zeldes/gum) | annotations CC BY 4.0; texts by source (see GUM's LICENSE.md; podcasts CC BY-NC-SA 4.0) |
| CCPE-M (Radlinski et al., 2019) | [github.com/google-research-datasets/ccpe](https://github.com/google-research-datasets/ccpe) | CC BY 4.0 |
| ConEL-2 (Joko and Hasibi, 2022) | [github.com/informagi/conversational-entity-linking-2022](https://github.com/informagi/conversational-entity-linking-2022) | MIT |
| ConEL (Joko et al., 2021) | [github.com/informagi/conversational-entity-linking](https://github.com/informagi/conversational-entity-linking) | MIT |
| Reddit Entity Linking (Botzer et al., 2021) | [zenodo.org/records/3970806](https://zenodo.org/records/3970806) | CC BY 4.0 |
| StackOverflow NER (Tabassum et al., 2020) | [github.com/jeniyat/StackOverflowNER](https://github.com/jeniyat/StackOverflowNER) | MIT |
| Wikidata | [wikidata.org](https://www.wikidata.org) | CC0 |

Some of the built text is non-commercial: GUM's podcasts are CC BY-NC-SA. Check each source's terms before using the built data commercially.
