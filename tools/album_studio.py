#!/usr/bin/env python3
"""Local web studio for the auto album generator.

Run it, open the page, type what you want, press generate. The GPU work happens
in one background worker so a single graphics card is never shared, and progress
is streamed to the browser with server-sent events.

    python tools/album_studio.py            # http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_album as album  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs" / "studio"
JOB_ROOT = OUT_ROOT / "jobs"
FAVORITES_PATH = OUT_ROOT / "favorites.json"
ENV_PATH = ROOT / ".env"
UI_PATH = Path(__file__).resolve().parent / "album_studio.html"

ENV_KEYS = {
    "llm_base": "AUTO_ALBUM_LLM_BASE",
    "llm_key": "AUTO_ALBUM_LLM_KEY",
    "llm_model": "AUTO_ALBUM_LLM_MODEL",
    "llm_effort": "AUTO_ALBUM_LLM_EFFORT",
    "temperature": "AUTO_ALBUM_TEMPERATURE",
    "top_k": "AUTO_ALBUM_TOP_K",
    "repetition_penalty": "AUTO_ALBUM_REPETITION_PENALTY",
}
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# No separators or dots, so it is safe to join into a path.
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULTS = {
    "llm_base": album.LLM_DEFAULT_BASE,
    "llm_model": album.LLM_DEFAULT_MODEL,
    "llm_effort": album.LLM_DEFAULT_EFFORT,
    "llm_key": "",
    "temperature": "",
    "top_k": "",
    "repetition_penalty": "",
}
FINISHED = {"succeeded", "failed", "cancelled"}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:6]}{'*' * 8}{key[-4:]}"


def write_env(updates: dict[str, str]) -> None:
    """Update named keys in .env, keeping every other line untouched."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.is_file() else []
    written: set[str] = set()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            name = stripped.split("=", 1)[0].strip()
            if name in updates:
                kept.append(f"{name}={updates[name]}")
                written.add(name)
                continue
        kept.append(line)
    kept.extend(f"{name}={value}" for name, value in updates.items() if name not in written)
    ENV_PATH.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")


class SettingsPayload(BaseModel):
    llm_base: str = Field(default="", max_length=300)
    llm_model: str = Field(default="", max_length=120)
    llm_effort: str = Field(default="", max_length=40)
    llm_key: str = Field(default="", max_length=400)
    # Empty means "keep the checkpoint's own tuned value for each stage".
    temperature: str = Field(default="", max_length=20)
    top_k: str = Field(default="", max_length=20)
    repetition_penalty: str = Field(default="", max_length=20)


SAMPLING_LIMITS = {"temperature": (0.0, 5.0), "top_k": (1.0, 4096.0),
                   "repetition_penalty": (1.0, 3.0)}
SAMPLING_LABELS = {"temperature": "多样性", "top_k": "候选范围",
                   "repetition_penalty": "重复惩罚"}


def clean_sampling(name: str, raw: str) -> str:
    """Return a normalised value or raise, so the UI can report it inline."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    label = SAMPLING_LABELS[name]
    try:
        value = float(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"「{label}」需要填数字") from exc
    low, high = SAMPLING_LIMITS[name]
    if not low <= value <= high:
        raise HTTPException(status_code=400, detail=f"「{label}」的范围是 {low} 到 {high}")
    if name == "top_k":
        return str(int(value))
    return f"{value:g}"


def current_settings() -> dict:
    key = os.environ.get(ENV_KEYS["llm_key"], "")
    return {
        "llm_base": os.environ.get(ENV_KEYS["llm_base"], DEFAULTS["llm_base"]),
        "llm_model": os.environ.get(ENV_KEYS["llm_model"], DEFAULTS["llm_model"]),
        "llm_effort": os.environ.get(ENV_KEYS["llm_effort"], DEFAULTS["llm_effort"]),
        "temperature": os.environ.get(ENV_KEYS["temperature"], ""),
        "top_k": os.environ.get(ENV_KEYS["top_k"], ""),
        "repetition_penalty": os.environ.get(ENV_KEYS["repetition_penalty"], ""),
        "key_masked": mask_key(key),
        "has_key": bool(key),
        "ready": bool(key),
    }


def runtime_info() -> dict:
    """What will actually run the generation, reported without loading any weights.

    The attention method is the one the AR decode graph will pick: FlashAttention
    when the PyTorch build ships that kernel, otherwise the cuDNN kernel. Both run
    on the GPU; this is not a CPU fallback.
    """
    import torch
    from yue2.cuda_graph import flash_attention_available

    if not torch.cuda.is_available():
        return {"available": False, "label": "没有可用的显卡",
                "detail": "生成需要一张支持 BF16 的 NVIDIA 显卡"}
    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    short = name.replace("NVIDIA ", "").replace("GeForce ", "")
    flash = flash_attention_available()
    method = "FlashAttention" if flash else "cuDNN"
    # Both kernels run on the GPU; only the AR decode graph differs.
    detail = (
        f"{name}，{round(props.total_memory / 2 ** 30, 1)} GiB 显存。\n"
        f"运算全部在显卡上进行，注意力使用 {method} 内核。"
        + ("" if flash else
           f"\n这版 PyTorch 没有编译 FlashAttention，所以改用同为 GPU 内核的 "
           f"cuDNN {torch.backends.cudnn.version()}，速度略有差别，画质不受影响。")
    )
    return {
        "available": True,
        "device": name,
        "device_short": short,
        "vram_gib": round(props.total_memory / 2 ** 30, 1),
        "attention": "flash" if flash else "cudnn",
        "method": method,
        "label": f"显卡加速 · {method}",
        "detail": detail,
    }


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

class JobRequest(BaseModel):
    brief: str = Field(default="", max_length=1200)
    # Supplied lyrics are used verbatim, so the length is whatever they sing.
    lyrics: str = Field(default="", max_length=4000)
    count: int = Field(default=4, ge=1)
    # The acoustic stage runs per song, so a very long one can exceed the card's
    # VRAM; that surfaces as a clear error rather than a silent cap here.
    seconds: float = Field(default=30.0, ge=5.0)
    intro_outro: bool = True
    strict_duration: bool = False
    language: str = Field(default="zh", pattern="^(zh|en)$")
    batch: int = Field(default=4, ge=1, le=8)
    ideas_only: bool = False
    # Set both to refine one existing song instead of designing fresh ones.
    parent_job: str = Field(default="", max_length=64)
    parent_slug: str = Field(default="", max_length=64)
    keep_lyrics: bool = True


@dataclass
class Job:
    id: str
    request: dict
    status: str = "queued"
    stage: str = "queued"
    message: str = "排队中"
    error: str = ""
    ideas: list = field(default_factory=list)
    songs: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    events: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id, "status": self.status, "stage": self.stage,
                "message": self.message, "error": self.error,
                "ideas": self.ideas, "songs": self.songs, "notes": self.notes,
                "count": self.request.get("count", 0),
                "ideas_only": self.request.get("ideas_only", False),
                "strict_duration": self.request.get("strict_duration", False),
                "intro_outro": self.request.get("intro_outro", True),
                "created": self.created, "started": self.started,
                "finished": self.finished, "cursor": len(self.events),
            }

    def publish(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)

    def read(self, cursor: int) -> tuple[list, int]:
        with self.lock:
            return self.events[cursor:], len(self.events)

    # -- persistence ------------------------------------------------------
    STATE_FIELDS = ("id", "request", "status", "stage", "message", "error", "ideas",
                    "songs", "notes", "created", "started", "finished")

    def to_state(self) -> dict:
        with self.lock:
            return {name: getattr(self, name) for name in self.STATE_FIELDS}

    @classmethod
    def from_state(cls, state: dict) -> "Job":
        known = {name: state[name] for name in cls.STATE_FIELDS if name in state}
        job = cls(**known)
        if job.status in ("running", "queued"):
            # Nothing is executing after a restart, so a job left mid-flight is
            # reported as interrupted rather than looking like it is still going.
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "服务重启，作业已中断"
            job.finished = job.finished or time.time()
        return job

    def save(self) -> None:
        """Write this job's state so favourites and refinements survive a restart."""
        target = JOB_ROOT / f"{self.id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(self.to_state(), ensure_ascii=False), encoding="utf-8")
        temp.replace(target)


class Studio:
    """One GPU, one job at a time, with the model kept resident between runs."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.pending: queue.Queue[str] = queue.Queue()
        self.pipeline = None
        self._load_history()
        self.worker = threading.Thread(target=self._serve, name="album-worker", daemon=True)
        self.worker.start()

    def _load_history(self) -> None:
        """Restore past jobs so favourites and refinements outlive a restart."""
        if not JOB_ROOT.is_dir():
            return
        restored = []
        for path in JOB_ROOT.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                restored.append(Job.from_state(state))
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                print(f"[studio] skipping unreadable job file {path.name}: {exc}",
                      file=sys.stderr)
        restored.sort(key=lambda job: job.created)
        for job in restored:
            self.jobs[job.id] = job
            self.order.append(job.id)
            job.save()
        if restored:
            print(f"[studio] restored {len(restored)} past job(s) from {JOB_ROOT}",
                  file=sys.stderr)

    # -- job lifecycle ----------------------------------------------------
    def submit(self, request: JobRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request.model_dump())
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        job.publish({"type": "status", "stage": "queued", "message": "排队中"})
        self.pending.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    def recent(self) -> list[dict]:
        with self.lock:
            ids = list(reversed(self.order))[:20]
        return [self.jobs[job_id].snapshot() for job_id in ids]

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        with job.lock:
            if job.status in FINISHED:
                return {"status": job.status}
            job.status = "cancelled"
            job.message = "已取消"
            job.stage = "cancelled"
            job.finished = time.time()
            job.events.append({"type": "status", "stage": "cancelled", "message": "已取消"})
        job.save()
        return {"status": "cancelled"}

    def _serve(self) -> None:
        while True:
            job_id = self.pending.get()
            job = self.jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            try:
                self._run(job)
            except InterruptedError:
                self._settle(job, "cancelled", "已取消")
            except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
                detail = f"{type(exc).__name__}: {exc}"
                print(f"[studio] job {job.id} failed: {detail}", file=sys.stderr)
                self._settle(job, "failed", "生成失败", error=detail)

    def _settle(self, job: Job, status: str, message: str, error: str = "") -> None:
        with job.lock:
            job.status = status
            job.stage = status
            job.message = message
            job.error = error
            job.finished = time.time()
            job.events.append({"type": "status", "stage": status, "message": message,
                               "error": error})
        job.save()

    # -- the actual work -------------------------------------------------
    def _run(self, job: Job) -> None:
        request = dict(job.request)
        out = OUT_ROOT / job.id
        out.mkdir(parents=True, exist_ok=True)

        with job.lock:
            job.status = "running"
            job.started = time.time()

        def emit(event: dict) -> None:
            if event.get("type") == "song":
                slug = event["song"]["slug"]
                event["song"]["audio"] = f"/api/jobs/{job.id}/audio/{slug}"
            with job.lock:
                # Register a finished song before its event can reach a client,
                # or the player asks for audio the job does not yet admit to
                # having and gets a 404, which shows up as 0:00.
                if event.get("type") == "song":
                    job.songs.append(event["song"])
                job.stage = event.get("stage", job.stage)
                if event.get("message"):
                    job.message = event["message"]
                job.events.append(event)
            if event.get("type") == "song":
                # Persist as songs land, so an interrupted job keeps what it finished.
                job.save()

        cancelled = lambda: job.status == "cancelled"  # noqa: E731 - read by the engine

        args = SimpleNamespace(
            out=out, seconds=request["seconds"],
            strict_duration=request.get("strict_duration", False),
            batch=request["batch"], memory_budget_gib=14.0,
            model="m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", progress=False,
            sampling={key: os.environ.get(ENV_KEYS[key], "")
                      for key in ("temperature", "top_k", "repetition_penalty")},
        )
        args.sampling = {name: value for name, value in args.sampling.items() if value != ""}

        client = album.LlmClient(
            base=os.environ.get(ENV_KEYS["llm_base"], DEFAULTS["llm_base"]),
            key=os.environ.get(ENV_KEYS["llm_key"], ""),
            model=os.environ.get(ENV_KEYS["llm_model"], DEFAULTS["llm_model"]),
            effort=os.environ.get(ENV_KEYS["llm_effort"], DEFAULTS["llm_effort"]),
        )
        if not client.key:
            raise RuntimeError("还没有配置 LLM 凭据，请先在设置里填写")

        parent = None
        if request.get("parent_job") and request.get("parent_slug"):
            try:
                parent = parent_idea(request["parent_job"], request["parent_slug"])
            except HTTPException as exc:
                raise RuntimeError("要优化的那首歌已经不在了，请重新生成后再试") from exc

        if parent is not None:
            ideas, notes = album.refine_ideas(
                client, parent, request["count"], request["language"],
                workers=4, chunk=2, brief=request["brief"],
                keep_lyrics=request.get("keep_lyrics", True),
                intro_outro=request.get("intro_outro", True), emit=emit,
            )
        else:
            ideas, notes = album.design_ideas(
                client, request["count"], request["language"],
                workers=4, chunk=2, brief=request["brief"],
                budget=album.duration_budget(request["seconds"]),
                intro_outro=request.get("intro_outro", True),
                lyrics=request.get("lyrics", ""), emit=emit,
            )
        # The LLM calls cannot be interrupted mid-flight, so honour a cancel that
        # arrived during ideation before spending any GPU time.
        if cancelled():
            raise InterruptedError("cancelled during ideation")
        for note in notes:
            print(f"[studio] job {job.id}: {note}", file=sys.stderr)
        if not ideas:
            # Distinguish a connection problem from ideas that were all rejected,
            # because the listener cannot fix the first by rewording a request.
            if any(note.startswith("chunk") and "failed" in note for note in notes):
                raise RuntimeError("连接构思服务失败，请检查网络后重试")
            raise RuntimeError("没有产出可用的选题，请调整要求后重试")

        payload = [asdict(idea) for idea in ideas]
        with job.lock:
            job.ideas = payload
            job.notes = notes
            job.events.append({"type": "ideas", "ideas": payload,
                               "message": f"已构思 {len(payload)} 首歌"})
        job.save()
        if request["ideas_only"]:
            album.package(ideas, {"songs": []}, out, request["seconds"])
            self._settle(job, "succeeded", f"已出 {len(payload)} 个选题")
            return

        emit({"type": "status", "stage": "loading", "message": "正在准备音乐模型"})
        if self.pipeline is None:
            self.pipeline = self._load_pipeline(args)
        result = album.render(ideas, args, emit=emit, cancelled=cancelled,
                              pipeline=self.pipeline)
        if cancelled():
            raise InterruptedError("cancelled")

        # Songs were registered as their events were emitted, so nothing to copy here.
        album.package(ideas, result, out, request["seconds"])
        self._settle(job, "succeeded", f"已生成 {len(job.songs)} 首歌")

    def _load_pipeline(self, args: SimpleNamespace):
        from yue2 import YuE2Pipeline
        print(f"[studio] loading YuE2 weights ({args.memory_budget_gib} GiB budget)", file=sys.stderr)
        started = time.perf_counter()
        options = dict(device="cuda", resident_models=True,
                       memory_budget_gib=args.memory_budget_gib, progress=self.verbose)
        try:
            # The weights are already cached, so resolving them offline is instant
            # and keeps a flaky connection from failing an otherwise local job.
            pipeline = YuE2Pipeline.from_pretrained(
                args.model, vae=args.vae, local_files_only=True, **options)
        except Exception as exc:  # noqa: BLE001 - fall back to a networked resolve
            print(f"[studio] local weights not resolved ({type(exc).__name__}: {exc}); "
                  f"retrying online", file=sys.stderr)
            pipeline = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, **options)
        print(f"[studio] weights ready in {time.perf_counter() - started:.1f}s", file=sys.stderr)
        return pipeline


studio = Studio()


class Favorites:
    """Saved songs, keyed by job and slug, persisted next to the outputs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.items: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, list):
            self.items = [item for item in data
                          if isinstance(item, dict) and isinstance(item.get("job_id"), str)
                          and isinstance(item.get("slug"), str)
                          and SLUG_PATTERN.match(item["slug"])]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    @staticmethod
    def _decorate(item: dict) -> dict:
        return {**item, "audio": f"/api/jobs/{item['job_id']}/audio/{item['slug']}",
                "available": (OUT_ROOT / item["job_id"] / "audio" / item["slug"]
                              / "audio.flac").is_file()}

    def list(self) -> list[dict]:
        with self.lock:
            return [self._decorate(item) for item in self.items]

    def add(self, job_id: str, slug: str) -> list[dict]:
        song = find_song(job_id, slug)
        entry = {"job_id": job_id, "slug": slug, "title": song.get("title", slug),
                 "anchor": song.get("anchor", ""), "note": song.get("note", ""),
                 "audio_seconds": song.get("audio_seconds", 0.0),
                 "added": time.time()}
        with self.lock:
            self.items = [item for item in self.items
                          if not (item["job_id"] == job_id and item["slug"] == slug)]
            self.items.insert(0, entry)
            self._save()
        return self.list()

    def remove(self, job_id: str, slug: str) -> list[dict]:
        with self.lock:
            self.items = [item for item in self.items
                          if not (item["job_id"] == job_id and item["slug"] == slug)]
            self._save()
        return self.list()


def find_song(job_id: str, slug: str) -> dict:
    """Look up a song the job actually produced, and reject anything else."""
    job = studio.get(job_id)
    if not SLUG_PATTERN.match(slug):
        raise HTTPException(status_code=400, detail="bad slug")
    for song in job.snapshot()["songs"]:
        if song["slug"] == slug:
            return song
    raise HTTPException(status_code=404, detail="no such song in this job")


def parent_idea(job_id: str, slug: str) -> dict:
    """The song to refine, taken from whichever list the job populated.

    An ideas-only job has no rendered songs but its ideas carry the same fields,
    so refining from an idea works too.
    """
    job = studio.get(job_id)
    if not SLUG_PATTERN.match(slug):
        raise HTTPException(status_code=400, detail="bad slug")
    snapshot = job.snapshot()
    for entry in list(snapshot["songs"]) + list(snapshot["ideas"]):
        if entry.get("slug") == slug:
            return entry
    raise HTTPException(status_code=404, detail="no such song in this job")


def song_file(job_id: str, slug: str) -> Path:
    """Resolve a stored audio file.

    Authorisation is the file existing, not the job still being in memory, so
    favourites keep playing after a restart. Traversal is prevented by the format
    checks: neither id nor slug may contain a separator or a dot.
    """
    if not JOB_ID_PATTERN.match(job_id) or not SLUG_PATTERN.match(slug):
        raise HTTPException(status_code=400, detail="bad path")
    path = OUT_ROOT / job_id / "audio" / slug / "audio.flac"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="audio not written")
    return path


favorites = Favorites(FAVORITES_PATH)


# --------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------

app = FastAPI(title="Album Studio", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not UI_PATH.is_file():
        raise HTTPException(status_code=500, detail="album_studio.html is missing")
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/settings")
def get_settings() -> dict:
    return {**current_settings(),
            "defaults": {"count": 4, "seconds": 30, "language": "zh", "batch": 4}}


@app.get("/api/runtime")
def get_runtime() -> dict:
    return runtime_info()


@app.post("/api/settings")
def post_settings(payload: SettingsPayload) -> dict:
    updates = {
        ENV_KEYS["llm_base"]: payload.llm_base.strip() or DEFAULTS["llm_base"],
        ENV_KEYS["llm_model"]: payload.llm_model.strip() or DEFAULTS["llm_model"],
        ENV_KEYS["llm_effort"]: payload.llm_effort.strip(),
        ENV_KEYS["temperature"]: clean_sampling("temperature", payload.temperature),
        ENV_KEYS["top_k"]: clean_sampling("top_k", payload.top_k),
        ENV_KEYS["repetition_penalty"]: clean_sampling("repetition_penalty",
                                                       payload.repetition_penalty),
    }
    # An empty key field means "leave the stored key alone".
    if payload.llm_key.strip():
        updates[ENV_KEYS["llm_key"]] = payload.llm_key.strip()
    write_env(updates)
    for name, value in updates.items():
        os.environ[name] = value
    return current_settings()


@app.post("/api/jobs")
def create_job(request: JobRequest) -> dict:
    if not os.environ.get(ENV_KEYS["llm_key"], ""):
        raise HTTPException(status_code=400, detail="请先在设置里填写 LLM 凭据")
    job = studio.submit(request)
    return {"id": job.id}


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": studio.recent()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return studio.get(job_id).snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return studio.cancel(job_id)


@app.api_route("/api/jobs/{job_id}/audio/{slug}", methods=["GET", "HEAD"])
def get_audio(job_id: str, slug: str) -> FileResponse:
    # A plain @app.get leaves HEAD as 405, and some media stacks probe with it.
    path = song_file(job_id, slug)
    return FileResponse(path, media_type="audio/flac", filename=f"{slug}.flac")


@app.get("/api/jobs/{job_id}/player")
def get_player(job_id: str) -> FileResponse:
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="bad job id")
    path = OUT_ROOT / job_id / "player.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="player not written yet")
    return FileResponse(path, media_type="text/html")


# --------------------------------------------------------------------------
# Favorites
# --------------------------------------------------------------------------

class FavoriteRef(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    slug: str = Field(min_length=1, max_length=64)


class FavoriteBatch(BaseModel):
    items: list[FavoriteRef] = Field(min_length=1, max_length=100)


@app.get("/api/favorites")
def list_favorites() -> dict:
    return {"favorites": favorites.list()}


@app.post("/api/favorites")
def add_favorite(ref: FavoriteRef) -> dict:
    return {"favorites": favorites.add(ref.job_id, ref.slug)}


@app.post("/api/favorites/remove")
def remove_favorite(ref: FavoriteRef) -> dict:
    return {"favorites": favorites.remove(ref.job_id, ref.slug)}


@app.post("/api/favorites/download")
def download_favorites(batch: FavoriteBatch) -> FileResponse:
    """Zip the selected songs, skipping any whose audio is already gone."""
    export = OUT_ROOT / "exports"
    export.mkdir(parents=True, exist_ok=True)
    target = export / f"favorites-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    used, missing = set(), 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, ref in enumerate(batch.items, start=1):
            try:
                path = song_file(ref.job_id, ref.slug)
            except HTTPException:
                missing += 1
                continue
            title = find_song(ref.job_id, ref.slug).get("title") or ref.slug
            name = re.sub(r"[\\/:*?\"<>|]", "_", title).strip() or ref.slug
            while name in used:
                name = f"{name}-2"
            used.add(name)
            archive.write(path, arcname=f"{index:02d} {name}.flac")
        if not used:
            archive.writestr("README.txt", "所选歌曲的音频已被清理，没有可下载的文件。\n")
    return FileResponse(target, media_type="application/zip", filename=target.name)


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str, cursor: int = 0) -> StreamingResponse:
    job = studio.get(job_id)

    async def stream():
        index = max(0, cursor)
        idle = 0
        while True:
            events, index = job.read(index)
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            snapshot = job.snapshot()
            if snapshot["status"] in FINISHED and not events:
                yield f"data: {json.dumps({'type': 'end', **snapshot}, ensure_ascii=False)}\n\n"
                break
            idle = idle + 1 if not events else 0
            if idle and idle % 40 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--verbose", action="store_true",
                        help="also let the engine print its own progress lines")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    album.load_env(ENV_PATH)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    studio.verbose = args.verbose

    import uvicorn
    url = f"http://{args.host}:{args.port}"
    print(f"[studio] settings file: {ENV_PATH}")
    print(f"[studio] open {url}")
    if not args.no_browser:
        threading.Timer(1.2, lambda: _open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _open(url: str) -> None:
    import webbrowser
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - opening a browser is a convenience only
        pass


if __name__ == "__main__":
    raise SystemExit(main())