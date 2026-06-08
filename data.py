"""Build a chat fine-tuning dataset from the Blackadder scripts.

Reads every ``data/season_*.txt`` file and emits ``data/blackadder.jsonl``.
Each line of the output is a JSON array of two messages::

    [
      {"role": "user", "content": "<the line spoken just before Edmund>"},
      {"role": "assistant", "content": "<Edmund's line>"}
    ]

Design decisions (confirmed with the project owner):

* Edmund is identified by *all* of his aliases, and the alias set is
  resolved per-season because the scripts abbreviate speakers differently
  in each series (and the abbreviations conflict -- e.g. ``B`` means
  *Baldrick* in series 2 & 3, never Blackadder).
* The ``user`` turn is the nearest preceding line spoken by *another*
  character; intervening stage directions are skipped. Consecutive Edmund
  lines are merged into a single assistant turn.

The parser is heuristic -- these are messy fan transcripts -- but it is
conservative about what counts as a speaker so that wrapped dialogue and
scene narration are not mistaken for new turns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_PATH = DATA_DIR / "blackadder.jsonl"

# Speaker labels that refer to Edmund / Blackadder, per season number.
# Single-letter aliases come from the per-episode cast legends in the
# scripts; ``B`` is deliberately excluded everywhere (it is Baldrick).
EDMUND_ALIASES: dict[int, set[str]] = {
    1: {"Edmund", "EBA", "BA", "E"},
    2: {"Edmund", "Blackadder", "BA", "E"},
    3: {"Edmund", "Blackadder", "E"},
    4: {"Edmund", "Blackadder"},
}

# A speaker line: starts at column 0 with a capitalised label of at most a
# few words, an optional "(parenthetical)", then a colon followed by space.
# Requiring each word to be capitalised stops continuation lines such as
# "Minister of Great Britain and Her Empires: Mr. William Pitt" (note the
# lowercase "of"/"and") from being read as a speaker.
SPEAKER_RE = re.compile(
    r"^(?P<label>[A-Z][A-Za-z.'’\-]*(?:\s[A-Z][A-Za-z.'’\-]+){0,3})"
    r"\s*(?:\([^)]*\))?\s*:\s+(?P<text>\S.*)$"
)

# Scene-transition lines that are narration, not dialogue continuation.
SCENE_CUE_RE = re.compile(
    r"^(In the |In a |At the |At Mrs|At Prince|Outside|Inside|Cut to|"
    r"Meanwhile|Back (at|in|to)|The next|That night|Later|Up at|Down at|"
    r"On the |Some time|Scene)",
)

SENTENCE_END = re.compile(r"[.!?…]")
BRACKETS_RE = re.compile(r"\[[^\[\]]*\]")
PARENS_RE = re.compile(r"\([^()]*\)")
# One episode (s1 "The Black Seal") uses "{...}" for stage directions.
CURLY_RE = re.compile(r"\{[^{}]*\}")
# Marks the end of an episode in these transcripts; the next episode begins
# with a fresh intro paragraph and (sometimes) a fresh cast legend.
EPISODE_BREAK = "sharing is caring"


def clean(text: str) -> str:
    """Strip stage directions (``[...]`` and ``(...)``), emphasis markers and
    tidy whitespace. Stage directions may span several script lines, so this
    must run on the *joined* speech, not on individual raw lines."""
    for pattern in (BRACKETS_RE, PARENS_RE, CURLY_RE):
        prev = None
        while prev != text:  # peel nested directions from the inside out
            prev = text
            text = pattern.sub(" ", text)
    text = re.sub(r"[\[\](){}<>*]", "", text)  # drop any unbalanced leftovers
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(label: str) -> str:
    """Drop a trailing parenthetical and surrounding space from a speaker."""
    return re.sub(r"\s*\(.*$", "", label).strip()


def parse_utterances(lines: list[str], aliases: set[str]) -> list[dict]:
    """Turn raw script lines into ordered (speaker, text, is_edmund) entries."""
    utterances: list[dict] = []
    current: dict | None = None
    seen_dialogue = False  # used to skip the cast-legend block of each episode

    def close() -> None:
        """Finalise the open speech: clean the joined text, keep if non-empty."""
        nonlocal current
        if current is not None:
            text = clean(" ".join(current["fragments"]))
            if text:
                utterances.append(
                    {
                        "label": current["label"],
                        "text": text,
                        "is_edmund": current["is_edmund"],
                    }
                )
            current = None

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:  # blank line ends the current speech
            close()
            continue
        if EPISODE_BREAK in stripped.lower():  # end of an episode
            close()
            seen_dialogue = False  # the next episode may have its own legend
            continue
        if stripped.startswith("["):  # pure stage direction / scene change
            close()
            continue

        match = SPEAKER_RE.match(line)
        if match:
            close()
            # Cast legends ("B: Baldrick") precede an episode's dialogue: a
            # short, punctuation-free gloss. Skip them until real dialogue.
            head = clean(match.group("text"))
            if (
                not seen_dialogue
                and not SENTENCE_END.search(head)
                and len(head.split()) <= 5
            ):
                continue
            seen_dialogue = True
            label = normalize_label(match.group("label"))
            # Keep the raw text; a speech may open with only a stage direction
            # and spill its actual words onto the following lines.
            current = {
                "label": label,
                "fragments": [match.group("text")],
                "is_edmund": label in aliases,
            }
            continue

        if SCENE_CUE_RE.match(stripped):  # narration between speeches
            close()
            continue

        if current is not None:  # a wrapped continuation of the current speech
            current["fragments"].append(line)

    close()
    return utterances


def merge_runs(utterances: list[dict]) -> list[dict]:
    """Merge consecutive utterances from the same speaker (Edmund-aware)."""
    merged: list[dict] = []
    for utt in utterances:
        key = "__EDMUND__" if utt["is_edmund"] else utt["label"]
        if merged and merged[-1]["_key"] == key:
            merged[-1]["text"] = f"{merged[-1]['text']} {utt['text']}".strip()
        else:
            merged.append({**utt, "_key": key})
    return merged


def build_pairs(utterances: list[dict]) -> list[list[dict]]:
    """Pair each Edmund line with the preceding other-character line."""
    pairs: list[list[dict]] = []
    for i, utt in enumerate(utterances):
        if not utt["is_edmund"] or i == 0:
            continue
        prev = utterances[i - 1]  # guaranteed non-Edmund after merge_runs
        if prev["is_edmund"] or not prev["text"] or not utt["text"]:
            continue
        pairs.append(
            [
                {"role": "user", "content": prev["text"]},
                {"role": "assistant", "content": utt["text"]},
            ]
        )
    return pairs


def main() -> None:
    season_files = sorted(DATA_DIR.glob("season_*.txt"))
    if not season_files:
        raise SystemExit(f"No season_*.txt files found in {DATA_DIR}")

    all_pairs: list[list[dict]] = []
    for path in season_files:
        season_match = re.search(r"season_(\d+)", path.name)
        season = int(season_match.group(1)) if season_match else 0
        aliases = EDMUND_ALIASES.get(season, {"Edmund", "Blackadder"})

        lines = path.read_text(encoding="utf-8").splitlines()
        utterances = merge_runs(parse_utterances(lines, aliases))
        pairs = build_pairs(utterances)
        all_pairs.extend(pairs)
        print(f"{path.name}: {len(utterances):4d} utterances -> {len(pairs):4d} pairs")

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for pair in all_pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_pairs)} dialogue pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
