#!/usr/bin/env python3
"""Automated short-song album generator for YuE2.

Pipeline stages, each restartable through JSON state on disk:

  1. idea design     an LLM invents songs around high-volume search anchors
  2. batch render    shared-forward YuE2 generation with a duration target
  3. packaging       manifest plus a local HTML player

The render stage is the only part that needs the GPU; ``--stage ideas`` stops
after the text stage so the creative output can be reviewed before rendering.

Each song is built on a *search anchor*: a phrase listeners already type into a
music platform for their own reasons (``歌单``, ``助眠``, ``奶龙``). A song titled
with that phrase is returned by that search, which is what makes it findable.
"""

from __future__ import annotations

import argparse
import html
import http.client
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# One semantic codec token covers 40 ms of rendered audio (the VAE downsamples
# at 1920 samples per frame and the codec emits 25 frames per second).
SECONDS_PER_TOKEN = 0.04

# YuE2 has no duration parameter, and its length is not reliably controllable.
# The semantic stage decides for itself when to stop: with the same lyrics, plan
# and seed, four measured runs produced 35.0, 42.2, 47.0, 48.9 and 55.9 s, plus
# one 130.0 s outlier, and cutting the ABC budget produced a longer song than
# raising it. What the prompt can control is *whether the song resolves*: a song
# left free to reach its end token always sounds finished, while one cut off at a
# token cap stops mid-phrase. Length is therefore requested through the lyric
# budget as a bias, and the semantic stage is only capped when the listener asks
# for a strict limit and accepts the cut ending.
SECONDS_PER_LINE = 9.5
REFERENCE_BPM = 72.0
LINE_OVERHEAD_SECONDS = 4.0
BEATS_PER_LINE = SECONDS_PER_LINE * REFERENCE_BPM / 60.0
NOMINAL_BPM = 100.0
MIN_LYRIC_LINES = 4
MAX_LYRIC_LINES = 16

# Only a runaway guard, and deliberately modest. The ceiling sizes the batch KV
# cache, so raising it to the model's own 9000 default makes admission reject
# every batch and drops throughput to one song at a time (measured: 201.6 s for
# a single song). Measured songs that ended on their own used 875-1399 tokens,
# so this leaves room for the common case while keeping batching alive.
NATURAL_CEILING_FLOOR = 1600
NATURAL_CEILING_FACTOR = 3.0        # of the target length's own token count
# Fraction of the requested length a strict cap keeps, so the cut lands close.
STRICT_LENGTH_TOLERANCE = 0.92

LLM_DEFAULT_BASE = "https://ai.zetx.tech/v1"
LLM_DEFAULT_MODEL = "deepseek-v4.1-flash"
LLM_DEFAULT_EFFORT = "none"
LLM_TIMEOUT = 300
LLM_ATTEMPTS = 3

# This model reasons before answering and the reasoning shares the token
# budget, so the JSON needs a lot of headroom on top of its own size.
TOKENS_PER_SONG = 1100
TOKEN_HEADROOM = 1400

# Segmenting the batch by anchor pool keeps parallel requests from all
# proposing the same handful of obvious anchors.
ANCHOR_POOLS = (
    "platform vocabulary (歌单, 音乐, 铃声, 卡点, 纯音乐, BGM, 白噪音, 无损, 循环播放)",
    "daily routine (助眠, 学习, 通勤, 加班, 早起, 期末, 摸鱼, 打扫, 睡前)",
    "feeling (失眠, 想哭, 释怀, 一个人, 生日, 毕业, 表白, 想念, 放下)",
    "characters and creatures (奶龙, 卡皮巴拉, 线条小狗, Loopy, 玲娜贝儿, 熊猫, 猫meme, 小黄鸭)",
    "calendar and season (新年, 中秋, 圣诞, 夏天, 初雪, 春天, 周末, 假期)",
)

SYSTEM_PROMPT = """You design short songs that listeners discover by accident while searching a music platform.

The mechanic is the anchor: a phrase that already carries huge search volume because
people type it for their own reasons. A song titled with that phrase gets returned by
that search. Examples of the mechanic:
- anchor "歌单" -> a song literally titled 歌单 appears when anyone searches 歌单
- anchor "奶龙" -> fans searching that character meet the song
- anchor "助眠" -> anyone looking for sleep music meets the song

Rules:
- No news, no current events, no trending topics, no real living person's name.
- No crime, disaster, war or tragedy.
- Never imply an official, licensed or endorsed connection to a brand or character.
- The anchor must be something a listener would plausibly type, not a sentence.

The style string format is exactly: language tag, genre, lead vocal, instruments, mood, BPM.
Example: "Mandarin, warm chiptune pop, cute bright lead vocal, toy piano and bouncy synth
bass, playful and sweet, 108 BPM"

Lyrics: use [Verse] and [Chorus] tags, plain singable words. Work the anchor in
naturally; never just chant it. Write exactly the number of sung lines the length asks for,
including their repetitions."""

USER_PROMPT = """Write {count} songs in {language}, favouring that language.

Anchor pool to draw from for this batch: {pool}
{avoid}
{duration}
{structure}
{lyrics}
{brief}
Return JSON only, no prose and no code fence:
{{"songs": [{{"anchor": "...", "title": "...", "highlight": "...", "note": "...", "style": "...", "lyrics": "...", "id": "...", "why_searchable": "..."}}]}}

- "title": what a search for the anchor would plausibly return
- "highlight": one sentence naming the single memorable gimmick that makes it shareable
- "note": one short paragraph in {language} explaining what this song is and why it suits
  what the listener asked for. Address the listener directly; if they asked for nothing
  specific, say what mood the song serves.
- "style": end with the tempo, e.g. "... 96 BPM", because the tempo sets the song's length
- "id": lowercase ascii kebab-case, 2 to 4 words, a transliteration of the title
- "why_searchable": one sentence on why people search the anchor"""

BRIEF_LINE = ("The listener asked for this, and every song must honour it while still "
              "keeping its own search anchor: {brief}")

# Refining an existing song: keep what works, improve the rest.
REFINE_SYSTEM = """You improve an existing short song that a listener already has. Keep what works
and make the rest better. You may change the style (instruments, vocal character, mood, tempo), the
title and the note. You may keep the search anchor or choose a better one that fits the same words.

Rules:
- No news, no current events, no trending topics, no real living person's name.
- No crime, disaster, war or tragedy.
- Never imply an official, licensed or endorsed connection to a brand or character.
- Every version must differ from the others in a way the listener can hear; do not return
  near-duplicates of the same idea.

The style string format is exactly: language tag, genre, lead vocal, instruments, mood, BPM.
Example: "Mandarin, warm chiptune pop, cute bright lead vocal, toy piano and bouncy synth bass,
playful and sweet, 108 BPM\""""

REFINE_USER = """Existing song
anchor: {anchor}
title: {title}
style: {style}
note: {note}
<lyrics>
{lyrics}
</lyrics>

{keep}
{structure}
{direction}
Produce {count} improved versions in {language}. Return JSON only, no prose and no code fence:
{{"songs": [{{"anchor": "...", "title": "...", "highlight": "...", "note": "...", "style": "...", "lyrics": "...", "id": "...", "why_searchable": "..."}}]}}

- "note": one short paragraph in {language} saying what you changed and why, addressed to the listener
- "style": end with the tempo, e.g. "... 96 BPM"
- "id": lowercase ascii kebab-case, 2 to 4 words, unique per version
- "why_searchable": one sentence on why people search the anchor"""

REFINE_KEEP = ("Keep the lyrics EXACTLY as written above: copy that block verbatim into every "
               "version, character for character. Improve only the style, title and note.")
REFINE_FREE = ("You may rewrite the lyrics if that makes the song better, but keep the same subject "
               "and language.")
REFINE_DIRECTION = "The listener asked for this improvement: {brief}"

# The requested length becomes a lyric budget, because length follows how much
# the model sings. Left unchecked it also decides whether the song resolves
# before the runaway guard, so the count is stated as a requirement.
DURATION_LINE = ("Length: aim for about {seconds:.0f} seconds of finished audio. Singing runs "
                 "at roughly {beats:.0f} beats per lyric line, so at the tempo you choose that "
                 "is about {lines} sung lines in total. Do not pad the song to fill time and do "
                 "not cut it short; write exactly the lines the length needs.")

# Intro and outro are a structural preference the plan and the style both carry.
STRUCTURE_WITH = ("Structure: open with a short instrumental intro before the first sung line, "
                  "and close with a brief instrumental outro after the last one.")
STRUCTURE_WITHOUT = ("Structure: no instrumental intro, the singing starts on the first bar, "
                     "and the song ends on the final sung line with nothing after it.")
STRUCTURE_STYLE_WITH = "short instrumental intro and outro"
STRUCTURE_STYLE_WITHOUT = "no intro, vocal starts immediately"

# When the listener wrote the words, they are used verbatim. This also overrides
# the lyric-length instruction above it, which would otherwise contradict.
FIXED_LYRICS_LINE = ("The listener supplied the lyrics. Copy the block below verbatim into every "
                     "song's \"lyrics\" field: do not rewrite, translate, reorder, add or drop "
                     "lines. This overrides any other lyric or line-count instruction. The title "
                     "must still be a short searchable name built from the anchor, never the "
                     "listener's own request. Design the anchor, title, style and note around "
                     "these words.\n"
                     "<lyrics>\n{lyrics}\n</lyrics>")


@dataclass
class Budget:
    """Text budgets and the expected length for one requested duration."""
    seconds: float
    lines: int
    abc_tokens: int


def duration_budget(seconds: float) -> Budget:
    per_line = seconds_per_line(NOMINAL_BPM)
    lines = int(round((seconds - LINE_OVERHEAD_SECONDS) / per_line))
    lines = 2 * max(1, round(lines / 2))          # keep verse and chorus balanced
    lines = max(MIN_LYRIC_LINES, min(MAX_LYRIC_LINES, lines))
    # Generous enough for the plan to finish, which is what yields a clean
    # ending: at 200 tokens the plan was cut and the song ran long and unstable,
    # at 576 (complete) it was the cleanest take measured.
    return Budget(seconds=seconds, lines=lines, abc_tokens=200 + 130 * lines)


def seconds_per_line(bpm: float) -> float:
    """Tempo scaling for the length estimate; BPM is parsed from the style."""
    return BEATS_PER_LINE * 60.0 / max(40.0, min(200.0, bpm))


def style_bpm(style: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*bpm", style, re.I)
    return float(match.group(1)) if match else 100.0


def duration_lines(budget: Budget) -> dict:
    return {"lines": budget.lines // 2, "beats": round(BEATS_PER_LINE)}


def brief_slot(brief: str) -> str:
    brief = (brief or "").strip()
    return BRIEF_LINE.format(brief=brief) if brief else ""


def structure_line(intro_outro: bool) -> str:
    return STRUCTURE_WITH if intro_outro else STRUCTURE_WITHOUT


def structure_suffix(intro_outro: bool) -> str:
    """A short clause for the style string, which conditions the audio model."""
    return STRUCTURE_STYLE_WITH if intro_outro else STRUCTURE_STYLE_WITHOUT


def fixed_lyrics_slot(lyrics: str) -> str:
    lyrics = (lyrics or "").strip()
    return FIXED_LYRICS_LINE.format(lyrics=lyrics) if lyrics else ""


def refine_ideas(client: LlmClient, parent: dict, count: int, language: str,
                 workers: int, chunk: int, brief: str = "",
                 keep_lyrics: bool = True, intro_outro: bool = True,
                 emit=None) -> tuple[list[Idea], list[str]]:
    """Ask for improved versions of one existing song.

    The lyrics and the search anchor are the parts worth preserving, so they are
    handed to the model as the starting point and only the style, title and note
    are expected to move. Each version gets its own seed, so the audio differs.
    """
    language_name = "Chinese (Mandarin)" if language == "zh" else "English"
    lyrics = (parent.get("lyrics") or "").strip()
    template = dict(
        anchor=parent.get("anchor", ""), title=parent.get("title", ""),
        style=parent.get("style", ""), note=parent.get("note", "") or parent.get("highlight", ""),
        lyrics=lyrics, language=language_name,
        keep=REFINE_KEEP if keep_lyrics else REFINE_FREE,
        structure=structure_line(intro_outro),
        direction=REFINE_DIRECTION.format(brief=brief.strip()) + "\n" if brief.strip() else "",
    )
    requests = [(index, min(chunk, count - index)) for index in range(0, count, chunk)]
    found: list[dict] = []
    notes: list[str] = []
    finished = 0
    if emit is not None:
        emit({"type": "status", "stage": "ideas",
              "message": f"正在基于《{parent.get('title', '')}》优化 {count} 个版本"})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(client.chat_json, REFINE_SYSTEM,
                        REFINE_USER.format(count=size, **template),
                        TOKEN_HEADROOM + TOKENS_PER_SONG * size): index
            for index, size in requests
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not end the run
                notes.append(f"chunk {index} failed: {type(exc).__name__}: {exc}")
                payload = {}
            else:
                songs = payload.get("songs")
                if isinstance(songs, list):
                    found.extend(songs)
                else:
                    notes.append(f"chunk {index} returned no songs list")
            finished += 1
            if emit is not None:
                emit({"type": "progress", "stage": "ideas",
                      "done": finished, "total": len(requests),
                      "message": f"已完成 {finished}/{len(requests)} 组优化"})

    ideas: list[Idea] = []
    taken: set[str] = set()
    for raw in found:
        entry = dict(raw) if isinstance(raw, dict) else {}
        # Force the preserved parts before validation, so a model slip cannot
        # spoil words the listener already approved.
        if keep_lyrics and lyrics:
            entry["lyrics"] = lyrics
        if not _text(entry, "anchor"):
            entry["anchor"] = parent.get("anchor", "")
        # A refinement is not bound by the length budget, and its words may be
        # the parent's, so the line-count check does not apply.
        idea, problem = validate_song(entry, taken, None)
        if idea is None:
            notes.append(f"dropped {_text(raw, 'title') or '?'}: {problem}")
            continue
        ideas.append(idea)
    return ideas, notes


def with_structure(style: str, intro_outro: bool) -> str:
    """Add the structure clause, keeping the tempo at the end of the style."""
    clause = structure_suffix(intro_outro)
    match = re.search(r"\d+(?:\.\d+)?\s*BPM\s*$", style, re.I)
    if not match:
        return f"{style}, {clause}"
    head = style[:match.start()].rstrip(" ,")
    return f"{head}, {clause}, {match.group(0).strip()}"


# --------------------------------------------------------------------------
# LLM access
# --------------------------------------------------------------------------

def load_env(path: Path) -> None:
    """Read key=value lines into the environment without overwriting real values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LlmClient:
    base: str
    key: str
    model: str
    timeout: float = LLM_TIMEOUT
    effort: str = LLM_DEFAULT_EFFORT

    def chat(self, system: str, user: str, max_tokens: int) -> str:
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": 1.0,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        if self.effort:
            # Cutting the reasoning pass is the single biggest latency win here.
            payload["reasoning_effort"] = self.effort
        body = json.dumps(payload).encode("utf-8")
        last = None
        for attempt in range(LLM_ATTEMPTS):
            request = urllib.request.Request(
                self.base.rstrip("/") + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                # 5xx and 429 are worth retrying; other 4xx usually are not.
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(
                        f"LLM rejected the request ({exc.code}): "
                        f"{exc.read().decode('utf-8', 'replace')[:400]}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as exc:
                # A dropped connection (RemoteDisconnected, reset, truncated body)
                # is transient and common behind a proxy, so it is retried too.
                last = exc
            else:
                # A malformed or truncated payload is not transient, so report it
                # immediately rather than burning the whole retry budget on it.
                return self._extract(raw)
            if attempt + 1 < LLM_ATTEMPTS:
                time.sleep(2 ** attempt * 3)
        raise RuntimeError(f"LLM request failed after {LLM_ATTEMPTS} attempts: {last}")

    @staticmethod
    def _extract(raw: str) -> str:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON: {raw[:400]}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {raw[:400]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # The model reasons first and the reasoning shares the token budget,
            # so an exhausted budget leaves reasoning but no answer.
            usage = data.get("usage") or {}
            raise RuntimeError(
                "LLM returned no answer text "
                f"(finish_reason={choices[0].get('finish_reason')!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"keys={sorted(message)}); raise max_tokens or lower reasoning effort.")
        return content

    def chat_json(self, system: str, user: str, max_tokens: int) -> dict:
        content = self.chat(system, user, max_tokens).strip()
        if content.startswith("```"):
            block = content.split("```")[1]
            content = block[4:] if block.lstrip().lower().startswith("json") else block
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Salvage the outermost object when the model wraps it in a sentence.
            start, end = content.find("{"), content.rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(content[start:end + 1])


# --------------------------------------------------------------------------
# Idea validation
# --------------------------------------------------------------------------

TITLE_LIMIT = 40
ANCHOR_LIMIT = 14

# An automated writer should not turn crime or tragedy into songs.
UNSUITABLE_MARKERS = (
    "murder", "murderer", "serial killer", "killer", "killed", "killing",
    "shooting", "gunman", "massacre", "genocide", "rape", "rapist",
    "suicide", "torture", "executed", "execution", "hanged", "corpse",
    "war crime", "terrorist", "bombing", "assassination", "pedophile",
    "homicide", "manslaughter", "kidnapped", "受害", "凶杀", "谋杀", "自杀",
    "屠杀", "恐怖袭击", "爆炸案", "枪击", "遗体",
)
UNSUITABLE_PATTERNS = (r"\bdies\b", r"\bdied\b", r"\bdeath\b", r"\bdead\b")


def is_unsuitable(text: str) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in UNSUITABLE_MARKERS):
        return True
    return any(re.search(pattern, lowered) for pattern in UNSUITABLE_PATTERNS)


def slugify(text: str, limit: int = 48) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.casefold()).strip("-")[:limit]


def _comparable(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.casefold())


def echoes_request(title: str, brief: str) -> bool:
    """True when the title is just the listener's request handed back.

    The model does this occasionally, and such a title is neither searchable nor
    a name, so it is treated as a failed idea rather than shipped.
    """
    brief_key = _comparable(brief)
    if len(brief_key) < 6:
        return False
    title_key = _comparable(title)
    return bool(title_key) and (title_key == brief_key or brief_key in title_key)


@dataclass
class Idea:
    anchor: str
    title: str
    highlight: str
    style: str
    lyrics: str
    slug: str
    rationale: str
    seed: int
    note: str = ""

    def request(self) -> dict:
        return {"id": self.slug, "style": self.style, "lyrics": self.lyrics,
                "cot": "full", "seed": self.seed}


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def sung_lines(lyrics: str) -> int:
    """Count lines that are actually sung, not the section tags."""
    return sum(1 for line in lyrics.splitlines()
               if line.strip() and not line.strip().startswith("["))


def validate_song(raw: dict, taken: set[str], budget: Budget | None = None) -> tuple[Idea | None, str]:
    """Reject schema violations and content we will not set to music."""
    if not isinstance(raw, dict):
        return None, "not an object"
    anchor, title = _text(raw, "anchor"), _text(raw, "title")
    highlight, style, lyrics = (_text(raw, "highlight"), _text(raw, "style"),
                                _text(raw, "lyrics"))
    if not (anchor and title and style and lyrics):
        return None, "missing anchor, title, style or lyrics"
    if len(anchor) > ANCHOR_LIMIT or len(title) > TITLE_LIMIT:
        return None, f"anchor or title too long ({anchor!r} / {title!r})"
    if "[verse]" not in lyrics.casefold() or "[chorus]" not in lyrics.casefold():
        return None, "lyrics lack [Verse] or [Chorus]"
    parts = [part.strip() for part in style.split(",") if part.strip()]
    if len(parts) < 4 or not re.search(r"\d+\s*bpm", style, re.I):
        return None, f"style is not a full YuE2 style string: {style!r}"
    # Length follows the lyric budget, so a wrong count is a real defect.
    if budget is not None:
        lines = sung_lines(lyrics)
        if abs(lines - budget.lines) > 2:
            return None, f"{lines} sung lines, wanted about {budget.lines}"
    if is_unsuitable(" ".join((anchor, title, highlight, lyrics))):
        return None, "crime or tragedy subject"

    slug = slugify(_text(raw, "id")) or slugify(title) or f"track-{zlib.crc32(title.encode()):08x}"
    base, suffix = slug, 2
    while slug in taken:
        slug = f"{base}-{suffix}"
        suffix += 1
    taken.add(slug)
    return Idea(anchor=anchor, title=title, highlight=highlight, style=style, lyrics=lyrics,
                slug=slug, rationale=_text(raw, "why_searchable"), note=_text(raw, "note"),
                seed=(zlib.crc32(slug.encode("utf-8")) % (2 ** 31))), ""


def design_ideas(client: LlmClient, count: int, language: str, workers: int,
                 chunk: int, brief: str = "", budget: Budget | None = None,
                 intro_outro: bool = True, lyrics: str = "",
                 emit=None) -> tuple[list[Idea], list[str]]:
    """Ask for the songs in parallel chunks, then keep only valid ones.

    ``lyrics`` supplies the words verbatim, in which case the model designs only
    the searchable parts and the length budget no longer constrains the count.
    """
    budget = budget or duration_budget(30.0)
    fixed = lyrics.strip()
    language = "Chinese (Mandarin)" if language == "zh" else "English"
    requests = [(index, min(chunk, count - index)) for index in range(0, count, chunk)]
    found: list[dict] = []
    notes: list[str] = []
    finished = 0
    if emit is not None:
        emit({"type": "status", "stage": "ideas",
              "message": f"正在构思 {count} 首歌的选题" + ("" if fixed else "和歌词")})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                client.chat_json, SYSTEM_PROMPT,
                USER_PROMPT.format(count=size, language=language,
                                   pool=ANCHOR_POOLS[index % len(ANCHOR_POOLS)],
                                   avoid="Use a different anchor from the other batches.",
                                   duration=DURATION_LINE.format(seconds=budget.seconds,
                                                                 **duration_lines(budget)),
                                   structure=structure_line(intro_outro),
                                   lyrics=fixed_lyrics_slot(fixed),
                                   brief=brief_slot(brief)),
                # The model reasons first and shares this budget, so leave room for both.
                TOKEN_HEADROOM + TOKENS_PER_SONG * size,
            ): index
            for index, size in requests
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not end the run
                notes.append(f"chunk {index} failed: {type(exc).__name__}: {exc}")
                payload = {}
            else:
                songs = payload.get("songs")
                if isinstance(songs, list):
                    found.extend(songs)
                else:
                    notes.append(f"chunk {index} returned no songs list")
            finished += 1
            if emit is not None:
                emit({"type": "progress", "stage": "ideas",
                      "done": finished, "total": len(requests),
                      "message": f"已完成 {finished}/{len(requests)} 组构思"})

    ideas: list[Idea] = []
    taken: set[str] = set()
    anchors: set[str] = set()
    for raw in found:
        idea, problem = validate_song(raw, taken, None if fixed else budget)
        if idea is None:
            notes.append(f"dropped {_text(raw, 'title') or '?'}: {problem}")
            continue
        if idea.anchor in anchors:
            taken.discard(idea.slug)
            notes.append(f"dropped {idea.title}: anchor {idea.anchor!r} already used")
            continue
        if echoes_request(idea.title, brief):
            taken.discard(idea.slug)
            notes.append(f"dropped {idea.title!r}: title just repeats the request")
            continue
        anchors.add(idea.anchor)
        if fixed:
            # The words are the listener's; only the model-written parts are kept.
            idea.lyrics = fixed
        idea.style = with_structure(idea.style, intro_outro)
        ideas.append(idea)
    return ideas, notes


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def semantic_ceiling(seconds: float) -> int:
    """Room to end naturally, sized so the batch KV cache still fits."""
    target = seconds / SECONDS_PER_TOKEN
    return int(max(NATURAL_CEILING_FLOOR, NATURAL_CEILING_FACTOR * target))


def sampling_overrides(values: dict | None = None) -> dict:
    """Only the sampling parameters the caller actually chose.

    Anything absent keeps the checkpoint's own tuned value for that stage, which
    is why these are merged into a dict of budgets rather than built into a
    complete Sampling object.
    """
    out: dict = {}
    for name in ("temperature", "top_k", "repetition_penalty"):
        raw = (values or {}).get(name)
        if raw is None or raw == "":
            continue
        value = float(raw)
        if name == "top_k":
            value = int(value)
        out[name] = value
    return out


def public_song(record: dict) -> dict:
    """Trim a rendered song to what a progress stream needs to display it."""
    return {key: record[key] for key in
            ("slug", "anchor", "title", "highlight", "note", "style", "lyrics", "rationale",
             "audio", "audio_seconds", "truncated")}


def render(ideas: list[Idea], args, emit=None, cancelled=None, pipeline=None) -> dict:
    """Render every idea, reporting progress through ``emit`` as it happens.

    ``generate_batch`` returns a whole batch at once, so stage and token events
    are what keep the display live while a batch is in flight.
    """
    import soundfile as sf
    import torch
    from yue2 import YuE2Pipeline
    from yue2.batching import BatchAdmissionError

    def notify(event: dict) -> None:
        if emit is not None:
            emit(event)

    retryable = (BatchAdmissionError, torch.cuda.OutOfMemoryError)
    budget = duration_budget(args.seconds)
    strict = bool(getattr(args, "strict_duration", False))
    # Dictionaries, so the checkpoint's tuned sampling is merged with the budgets
    # instead of being replaced. A Sampling object here would silently substitute
    # the generic defaults, which makes the melodies wander.
    overrides = sampling_overrides(getattr(args, "sampling", None))
    abc_sampling = {"max_tokens": budget.abc_tokens, **overrides}
    if strict:
        # A hard cut, requested knowingly: the most tokens that fit the target.
        wanted = int(round(args.seconds / SECONDS_PER_TOKEN))
        semantic_sampling = {"min_tokens": int(wanted * STRICT_LENGTH_TOLERANCE),
                             "max_tokens": wanted, **overrides}
    else:
        semantic_sampling = {"max_tokens": semantic_ceiling(args.seconds), **overrides}
    mode = "strict cap" if strict else "natural ending"
    print(f"[render] {len(ideas)} songs, batch={args.batch}, "
          f"~{budget.seconds:.0f}s requested => {budget.lines} lyric lines, "
          f"abc<={budget.abc_tokens}, semantic={semantic_sampling} ({mode})")
    if overrides:
        print(f"[render] sampling overrides: {overrides}")

    output = Path(args.out)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict] = []
    owned = pipeline is None
    if owned:
        notify({"type": "status", "stage": "loading",
                "message": "正在载入音乐模型，首次启动需要一点时间"})
        pipeline = YuE2Pipeline.from_pretrained(
            args.model, vae=args.vae, device="cuda", resident_models=True,
            memory_budget_gib=args.memory_budget_gib,
            progress=getattr(args, "progress", True),
        )
    try:
        start, batch = 0, args.batch
        while start < len(ideas):
            group = ideas[start:start + batch]
            row_slug = {index: idea.slug for index, idea in enumerate(group)}
            counts: dict[int, int] = {}
            notify({"type": "batch", "start": start, "slugs": list(row_slug.values()),
                    "message": f"正在生成第 {start + 1} 至 {start + len(group)} 首"})

            def stage(name, _map=row_slug):
                if ":" in name:
                    head, _, index = name.partition(":")
                    notify({"type": "stage", "slug": _map.get(int(index), ""), "stage": head})
                else:
                    notify({"type": "stage_all", "stage": name,
                            "slugs": list(_map.values())})

            def tokens_batch(row, phase, token, _map=row_slug):
                counts[row] = counts.get(row, 0) + 1
                if counts[row] % 25 == 0:
                    notify({"type": "tokens", "slug": _map.get(row, ""),
                            "phase": phase, "count": counts[row]})

            def tokens_single(phase, token):
                counts[0] = counts.get(0, 0) + 1
                if counts[0] % 25 == 0:
                    notify({"type": "tokens", "slug": row_slug.get(0, ""),
                            "phase": phase, "count": counts[0]})

            requests = [idea.request() for idea in group]
            print(f"[render] batch at {start}: {', '.join(r['id'] for r in requests)}")
            began = time.perf_counter()
            try:
                if len(requests) == 1:
                    # The shared-forward path needs at least two independent requests.
                    songs = [pipeline(**requests[0], abc_sampling=abc_sampling,
                                      semantic_sampling=semantic_sampling,
                                      cancelled=cancelled, on_stage=stage,
                                      on_token=tokens_single)]
                else:
                    songs = pipeline.generate_batch(
                        requests, abc_sampling=abc_sampling,
                        semantic_sampling=semantic_sampling,
                        cancelled=cancelled, on_stage=stage, on_token=tokens_batch,
                    )
            except InterruptedError:
                raise
            except MemoryError as exc:
                # The acoustic stage runs per song, so a smaller batch cannot help.
                # Only shorter lyrics can, which makes the cause worth naming.
                raise MemoryError(
                    "显存不足：歌词太长，编曲阶段放不下。把「希望时长」调小一档再试。"
                    f" ({exc})") from exc
            except retryable as exc:
                if batch == 1:
                    raise
                batch = max(1, batch - 1)
                print(f"[render] batch rejected ({exc}); retrying with batch={batch}")
                notify({"type": "status", "stage": "retry",
                        "message": f"显存不足，改为每批 {batch} 首重试"})
                torch.cuda.empty_cache()
                continue
            print(f"[render] batch finished in {time.perf_counter() - began:.1f}s")
            for index, song in enumerate(songs):
                idea = group[index]
                directory = audio_dir / idea.slug
                directory.mkdir(parents=True, exist_ok=True)
                sf.write(directory / "audio.flac", song.audio, song.sample_rate)
                record = asdict(idea)
                record.update({
                    "audio": str((directory / "audio.flac").relative_to(output)),
                    "audio_seconds": len(song.audio) / song.sample_rate,
                    "truncated": song.truncated,
                    "timing": song.timing,
                    "score": song.semantic.plan.abc,
                })
                rendered.append(record)
                if song.truncated["semantic"]:
                    print(f"[render] warning: {idea.slug} did not resolve; "
                          f"{song.timing.get('semantic', {}).get('output_tokens')} tokens "
                          f"hit the ceiling ({'strict cap' if strict else 'runaway guard'})")
                notify({"type": "song", "song": public_song(record)})
            start += batch
            batch = args.batch
    finally:
        if owned:
            pipeline.close()
    return {"songs": rendered}


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

PLAYER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Auto album</title>
<style>
body{{font:15px/1.6 system-ui,sans-serif;margin:0;padding:32px;background:#101014;color:#e8e8ee}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}}
p.meta{{color:#8b8b99;margin:0 0 24px}}
article{{border:1px solid #26262e;border-radius:10px;padding:16px 18px;margin-bottom:14px;background:#17171d}}
h2{{font-size:16px;margin:0 0 6px;font-weight:600}}
small{{color:#8b8b99}}
b.anchor{{color:#8fd3a8;font-weight:600}}
audio{{width:100%;margin-top:10px}}
pre{{white-space:pre-wrap;color:#a9a9b8;font-size:13px;margin:10px 0 0}}
p.hook{{color:#c9c9d6;font-style:italic;margin:8px 0 0}}
</style></head><body>
<h1>Auto album &mdash; {count} songs</h1>
<p class="meta">target {seconds}s &middot; generated {stamp}</p>
{items}
</body></html>
"""


def package(ideas: list[Idea], result: dict, out: Path, seconds: float) -> Path:
    by_slug = {record["slug"]: record for record in result["songs"]}
    blocks = []
    for idea in ideas:
        record = by_slug.get(idea.slug)
        if record is None:
            continue
        blocks.append(f"""<article>
<h2>{html.escape(idea.title)}</h2>
<small>{record['audio_seconds']:.1f}s &middot; {html.escape(idea.style)}</small>
<p class="hook">{html.escape(idea.note or idea.highlight)}</p>
<audio controls src="{html.escape(record['audio'])}"></audio>
<pre>{html.escape(idea.lyrics)}</pre>
</article>""")
    page = PLAYER.format(count=len(blocks), seconds=seconds,
                         stamp=time.strftime("%Y-%m-%d %H:%M"),
                         items="\n".join(blocks))
    player = out / "player.html"
    player.write_text(page, encoding="utf-8")
    (out / "album.json").write_text(
        json.dumps({"ideas": [asdict(idea) for idea in ideas], **result},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return player


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    # Load .env before argparse so its values can supply the flag defaults.
    root = Path(__file__).resolve().parent.parent
    load_env(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/auto-album"))
    parser.add_argument("--ideas", type=Path, help="reuse an earlier ideas.json instead of calling the LLM")
    parser.add_argument("--count", type=int, default=6, help="songs to design")
    parser.add_argument("--brief", default="", help="free-form listener requirements honoured by every song")
    parser.add_argument("--lyrics", default="", help="lyrics used verbatim instead of written by the model")
    parser.add_argument("--no-intro-outro", dest="intro_outro", action="store_false",
                        help="start singing immediately and end on the last sung line")
    parser.add_argument("--language", choices=("en", "zh"), default="zh")
    parser.add_argument("--plain-log", dest="progress", action="store_false",
                        help="suppress the engine's own progress lines on stderr")
    parser.add_argument("--llm-base", default=os.environ.get("AUTO_ALBUM_LLM_BASE", LLM_DEFAULT_BASE))
    parser.add_argument("--llm-key", default="")
    parser.add_argument("--llm-model", default=os.environ.get("AUTO_ALBUM_LLM_MODEL", LLM_DEFAULT_MODEL))
    parser.add_argument("--llm-effort", default=os.environ.get("AUTO_ALBUM_LLM_EFFORT", LLM_DEFAULT_EFFORT),
                        help="reasoning effort; 'none' is much faster, empty string restores model default")
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=2, help="songs requested per LLM call")
    parser.add_argument("--batch", type=int, default=4, help="songs per GPU batch; reduced automatically on OOM")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="target length; biases the lyrics, since length is not precise")
    parser.add_argument("--strict-duration", action="store_true",
                        help="cap generation to hit the target, accepting a cut ending")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling diversity; omit to keep the checkpoint's tuned value")
    parser.add_argument("--top-k", type=int, default=None,
                        help="candidate range per step; omit to keep the tuned value")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="pressure against recently used tokens; omit to keep the tuned value")
    parser.add_argument("--stage", choices=("ideas", "all"), default="all")
    parser.add_argument("--memory-budget-gib", type=float, default=14.0)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    args = parser.parse_args(argv)
    # The env file is the fallback for sampling, so a saved choice applies here too.
    args.sampling = {
        name: value for name, value in
        (("temperature", args.temperature if args.temperature is not None
          else os.environ.get("AUTO_ALBUM_TEMPERATURE", "")),
         ("top_k", args.top_k if args.top_k is not None
          else os.environ.get("AUTO_ALBUM_TOP_K", "")),
         ("repetition_penalty", args.repetition_penalty
          if args.repetition_penalty is not None
          else os.environ.get("AUTO_ALBUM_REPETITION_PENALTY", "")))
        if value != ""
    }

    args.out.mkdir(parents=True, exist_ok=True)

    if args.ideas:
        payload = json.loads(args.ideas.read_text(encoding="utf-8"))
        # Tolerate files written before newer fields existed.
        fields = Idea.__dataclass_fields__
        ideas = [Idea(**{key: entry[key] for key in fields if key in entry})
                 for entry in payload["ideas"]]
        notes = []
        print(f"[ideas] loaded {len(ideas)} from {args.ideas}")
    else:
        key = args.llm_key or os.environ.get("AUTO_ALBUM_LLM_KEY", "")
        if not key:
            print("[ideas] no API key: set AUTO_ALBUM_LLM_KEY, pass --llm-key, or add it to .env",
                  file=sys.stderr)
            return 2
        client = LlmClient(base=args.llm_base, key=key, model=args.llm_model,
                           effort=args.llm_effort)
        print(f"[ideas] asking {args.llm_model} (effort={args.llm_effort or 'default'}) "
              f"for {args.count} songs ({args.chunk} per call, {args.llm_workers} in parallel)")
        began = time.perf_counter()
        ideas, notes = design_ideas(client, args.count, args.language,
                                    args.llm_workers, args.chunk, args.brief,
                                    duration_budget(args.seconds), args.intro_outro,
                                    args.lyrics)
        print(f"[ideas] {len(ideas)} usable ideas in {time.perf_counter() - began:.1f}s")

    for note in notes:
        print(f"  ! {note}", file=sys.stderr)
    if not ideas:
        print("[ideas] nothing usable to render", file=sys.stderr)
        return 2

    (args.out / "ideas.json").write_text(
        json.dumps({"ideas": [asdict(idea) for idea in ideas]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[ideas] wrote {args.out / 'ideas.json'}")
    for idea in ideas:
        print(f"  {idea.anchor:<8} {idea.title}")

    if args.stage == "ideas":
        return 0

    result = render(ideas, args)
    unresolved = [song["title"] for song in result["songs"] if song["truncated"]["semantic"]]
    if unresolved:
        print(f"[render] {len(unresolved)} song(s) did not resolve: "
              f"{', '.join(unresolved)}; fewer lyric lines per song will fix it", file=sys.stderr)
    player = package(ideas, result, args.out, args.seconds)
    # Browsers refuse media loaded from file:// URLs, so the album needs a
    # local server even though the page itself opens fine.
    print(f"[done] {player}")
    print(f"[play] python -m http.server 8731 --directory \"{args.out}\" "
          f"then open http://127.0.0.1:8731/player.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())