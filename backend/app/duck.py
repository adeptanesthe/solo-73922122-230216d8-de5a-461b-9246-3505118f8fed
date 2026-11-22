"""配乐试听：20 ms 块级增益规划、频带响度约束、混音与冲突定位。

约束（对每一块 20 ms）：
- 配乐增益 g ∈ [0, 1]；
- 与甲 / 乙 / 待复核分段有重叠的块，压低后的配乐在 300～3400 Hz
  频带内的 RMS ≤ 同块原声（录音）在该频带内 RMS 的一半；
- 圈选为提示音的块，增益固定为 1；
- 相邻块之间增益最多下降 0.10、最多上升 0.04。

在这些约束下取全轨逐块可达到的最高增益。该可行域对逐点取最大值封闭，
因此存在唯一的最大解：先从后向前收紧“下降速率”给出的上界，再从前向后
收紧“上升速率”给出的上界。
"""
from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from .audio import AudioError, SAMPLE_RATE, load_wav

BLOCK_MS = 20.0
BLOCK_SAMPLES = int(round(SAMPLE_RATE * BLOCK_MS / 1000.0))  # 320
BAND_LO_HZ = 300.0
BAND_HI_HZ = 3400.0
DUCK_RATIO = 0.5          # 压低后的配乐 ≤ 原声的一半
MAX_GAIN = 1.0
MAX_RISE = 0.04           # 每块最多上升
MAX_FALL = 0.10           # 每块最多下降
SPEECH_LABELS = {"A", "B", "review"}
ALL_LABELS = {"A", "B", "review", "silence"}
FEAS_TOL = 1e-9


class MixInputError(ValueError):
    """配乐试听输入不合法（分段 / 提示音 / 配乐等）。"""


@dataclass
class BlockPlan:
    starts: np.ndarray
    ends: np.ndarray
    speech_blocks: np.ndarray
    cue_blocks: np.ndarray
    caps: np.ndarray
    voice_rms: np.ndarray
    music_rms: np.ndarray


# ---------------------------------------------------------------- 输入解析

def load_music(data: bytes, expected_samples: int) -> np.ndarray:
    """读取并校验配乐：16 kHz、单声道、与录音等长（允许 1 个采样取整误差）。"""
    try:
        music, _sr, _duration = load_wav(data)
    except AudioError as exc:
        raise MixInputError("配乐不合法：" + str(exc))
    if abs(int(music.size) - int(expected_samples)) > 1:
        raise MixInputError(
            "配乐须与录音等长："
            f"录音 {expected_samples / SAMPLE_RATE:.3f} 秒，"
            f"配乐 {music.size / SAMPLE_RATE:.3f} 秒")
    if music.size != expected_samples:
        # 单采样取整误差：补齐 / 截断到录音长度，避免后续分块错位
        if music.size < expected_samples:
            music = np.pad(music, (0, expected_samples - music.size))
        else:
            music = music[:expected_samples]
    return np.asarray(music, dtype=np.float64)


def parse_segments(raw: str, duration: float) -> list[dict]:
    """解析分析结果中的分段列表。"""
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise MixInputError("分段数据不是合法的 JSON")
    if not isinstance(items, list):
        raise MixInputError("分段数据格式错误：应为分段数组")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise MixInputError(f"第 {i + 1} 个分段格式错误")
        label = item.get("label")
        if label not in ALL_LABELS:
            raise MixInputError(f"第 {i + 1} 个分段标签非法：{label!r}")
        try:
            start, end = float(item.get("start")), float(item.get("end"))
        except (TypeError, ValueError):
            raise MixInputError(f"第 {i + 1} 个分段包含非数值时间")
        if not (np.isfinite(start) and np.isfinite(end)):
            raise MixInputError(f"第 {i + 1} 个分段包含非法时间")
        if start < -1e-6 or end > duration + 1e-3 or end - start <= 0.0:
            raise MixInputError(
                f"第 {i + 1} 个分段 [{start:.2f}, {end:.2f}] 越界或时长非法")
        out.append({
            "label": str(label),
            "start": max(0.0, start),
            "end": min(duration, end),
        })
    return out


def parse_cues(raw: str, duration: float) -> list[list[float]]:
    """解析提示音区间并合并重叠 / 相邻区间。"""
    if raw is None or raw.strip() == "":
        return []
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise MixInputError("提示音区间不是合法的 JSON")
    if not isinstance(items, list):
        raise MixInputError("提示音区间格式错误：应为 [[开始, 结束], ...]")
    ranges = []
    for i, item in enumerate(items):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise MixInputError(f"第 {i + 1} 个提示音区间格式错误，应为 [开始, 结束]")
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise MixInputError(f"第 {i + 1} 个提示音区间包含非数值时间")
        if not (np.isfinite(start) and np.isfinite(end)):
            raise MixInputError(f"第 {i + 1} 个提示音区间包含非法时间")
        if start < -1e-6 or end > duration + 1e-3 or end - start <= 0.0:
            raise MixInputError(
                f"第 {i + 1} 个提示音区间 [{start:.2f}, {end:.2f}] 越界或时长非法")
        ranges.append([max(0.0, start), min(duration, end)])

    ranges.sort()
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


# ---------------------------------------------------------------- 分块与响度

def block_bounds(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """均匀 20 ms 分块；最后一块可能不足 20 ms。"""
    n_blocks = max(1, int(math.ceil(n_samples / BLOCK_SAMPLES)))
    starts = np.arange(n_blocks, dtype=np.int64) * BLOCK_SAMPLES
    ends = np.minimum(starts + BLOCK_SAMPLES, n_samples)
    return starts, ends


def _band_sos():
    return butter(4, [BAND_LO_HZ, BAND_HI_HZ], btype="bandpass",
                  fs=SAMPLE_RATE, output="sos")


_BAND_SOS = _band_sos()


def _band_rms(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """每一块单独做零相位 300～3400 Hz 带通后的 RMS（“同块”频带响度）。

    块长固定、增益在块内恒定，因此“第 i 块压低后的配乐频带 RMS”
    恰好等于 gain[i] × 该块配乐的频带 RMS，频带约束可逐块精确换算成增益上限。
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(starts.size, dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg = x[s:e]
        if seg.size == 0:
            out[i] = 0.0
            continue
        filtered = sosfiltfilt(_BAND_SOS, seg)
        out[i] = math.sqrt(float(np.mean(filtered * filtered)))
    return out


def _range_mask(starts, ends, ranges) -> np.ndarray:
    """块与任一时间区间有重叠即为 True。"""
    bs = starts / float(SAMPLE_RATE)
    be = ends / float(SAMPLE_RATE)
    mask = np.zeros(starts.size, dtype=bool)
    for seg in ranges:
        s, e = float(seg["start"] if isinstance(seg, dict) else seg[0]), \
            float(seg["end"] if isinstance(seg, dict) else seg[1])
        mask |= (bs < e - 1e-9) & (be > s + 1e-9)
    return mask


def build_block_plan(voice: np.ndarray, music: np.ndarray,
                     segments: list[dict], cues: list[list[float]]) -> BlockPlan:
    starts, ends = block_bounds(voice.size)
    speech_segments = [s for s in segments if s["label"] in SPEECH_LABELS]
    speech_blocks = _range_mask(starts, ends, speech_segments)
    cue_blocks = _range_mask(starts, ends, cues)

    voice_rms = _band_rms(voice, starts, ends)
    music_rms = _band_rms(music, starts, ends)

    # 与讲话分段重叠的块：压低后配乐在该块的频带 RMS ≤ 0.5 × 同块原声频带 RMS，
    # 即 gain ≤ 0.5 × voice_rms / music_rms（块内配乐静音则无需压低）。
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = DUCK_RATIO * voice_rms / np.where(music_rms > 0.0, music_rms, 1.0)
    caps = np.where(music_rms > 0.0, np.minimum(MAX_GAIN, ratio), MAX_GAIN)
    caps = np.where(speech_blocks, np.clip(caps, 0.0, MAX_GAIN), MAX_GAIN)
    return BlockPlan(starts, ends, speech_blocks, cue_blocks, caps,
                     voice_rms, music_rms)


# ---------------------------------------------------------------- 增益规划

def _cap_bottleneck(caps: np.ndarray, k: int) -> int:
    """解释 upper[k] 为何 < 1：返回真正被压低、且 cap < 1 的块下标。

    upper[k] = min(cap[k], min_{j>k} cap[j] + FALL*(j-k))；
    对本块不压低的块，顺着未来“下降速率”约束找那个 cap 受限的块。
    """
    n = caps.size
    if caps[k] < MAX_GAIN - FEAS_TOL:
        return int(k)
    if k + 1 >= n:
        return int(k)
    js = np.arange(k + 1, n)
    vals = caps[js] + MAX_FALL * (js - k)
    j = int(js[int(np.argmin(vals))])
    return j if caps[j] < MAX_GAIN - FEAS_TOL else int(k)


def plan_gains(caps: np.ndarray, cue_blocks: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """求满足全部约束的逐块最高增益。

    返回 (gains, feasible, bottleneck_blocks)。gains 是整条可行域内的逐点最大解：
    对任何可行解 h 都有 h[i] <= gains[i]，因此提示音块可达 1 当且仅当
    gains 在该块等于 1；否则无解，bottleneck_blocks 给出造成冲突的讲话块。
    """
    caps = np.asarray(caps, dtype=np.float64)
    cue_blocks = np.asarray(cue_blocks, dtype=bool)
    n = caps.size
    if n == 0:
        return np.empty(0), True, np.empty(0, dtype=int)

    # 1) 下降速率给出的“更早的块”上界：g[i] <= g[j] + FALL*(j-i)
    upper = caps.copy()
    for i in range(n - 2, -1, -1):
        upper[i] = min(upper[i], upper[i + 1] + MAX_FALL)

    # 2) 逐块取最高：g[0]=upper[0]，之后受上升速率限制
    gains = np.empty(n, dtype=np.float64)
    gains[0] = upper[0]
    for i in range(1, n):
        gains[i] = min(upper[i], gains[i - 1] + MAX_RISE)

    # 3) 提示音块必须为 1（数值容差内认为 1）
    bad = cue_blocks & (gains < MAX_GAIN - FEAS_TOL)
    if not bool(np.any(bad)):
        gains[cue_blocks] = MAX_GAIN
        return gains, True, np.empty(0, dtype=int)

    # 4) 定位冲突块。g[c] = min_{k<=c}(upper[k] + RISE*(c-k))，
    #    取达到最小值的 k（优先选靠后的，便于解释），再追溯其 cap 瓶颈。
    bottlenecks: list[int] = []
    for c in np.where(bad)[0]:
        c = int(c)
        ks = np.arange(0, c + 1)
        vals = upper[ks] + MAX_RISE * (c - ks)
        k = int(ks[int(np.argmin(vals))])
        bottlenecks.append(_cap_bottleneck(caps, k))
    return gains, False, np.unique(np.asarray(bottlenecks, dtype=int))


def _conflict_intervals(plan: BlockPlan, bottleneck_blocks: np.ndarray,
                        segments: list[dict]) -> list[dict]:
    """把瓶颈块映射回讲话分段，输出可点击复听的冲突时段。"""
    bs = plan.starts / float(SAMPLE_RATE)
    be = plan.ends / float(SAMPLE_RATE)
    selected: dict[tuple, dict] = {}
    for k in np.asarray(bottleneck_blocks, dtype=int):
        t0, t1 = float(bs[k]), float(be[k])
        hit = [s for s in segments
               if s["label"] in SPEECH_LABELS
               and s["start"] < t1 - 1e-9 and s["end"] > t0 + 1e-9]
        if not hit:
            # 理论上提示音冲突总由讲话块引起；保留块级信息兜底
            selected[(round(t0, 3), round(t1, 3), "block")] = {
                "start": round(t0, 3), "end": round(t1, 3),
                "label": "block", "reason": "该块的增益无法在速率限制内满足提示音要求",
            }
        for s in hit:
            key = (round(s["start"], 3), round(s["end"], 3), s["label"])
            selected[key] = {
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "label": s["label"],
                "reason": "该讲话段要求压低配乐，与固定为 1 的提示音或增益速率限制冲突",
            }
    return sorted(selected.values(), key=lambda x: (x["start"], x["end"]))


# ---------------------------------------------------------------- 混音

def render_mix(voice: np.ndarray, music: np.ndarray,
               plan: BlockPlan, gains: np.ndarray):
    """按块应用增益并混音；超过满幅时两轨等比缩小。"""
    gain_signal = np.zeros(voice.size, dtype=np.float64)
    for i, (s, e) in enumerate(zip(plan.starts, plan.ends)):
        gain_signal[s:e] = gains[i]
    mix = np.asarray(voice, dtype=np.float64) + gain_signal * np.asarray(music, dtype=np.float64)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    scale = 1.0 if peak <= 1.0 else 1.0 / peak
    return mix * scale, peak, scale


def wav_bytes(signal: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(buf, SAMPLE_RATE, pcm)
    return buf.getvalue()


# ---------------------------------------------------------------- 对外入口

def build_preview(voice: np.ndarray, music: np.ndarray,
                  segments: list[dict], cues: list[list[float]]) -> dict:
    duration = voice.size / float(SAMPLE_RATE)
    plan = build_block_plan(voice, music, segments, cues)
    gains, feasible, bottlenecks = plan_gains(plan.caps, plan.cue_blocks)

    base = {
        "feasible": bool(feasible),
        "duration": round(duration, 6),
        "block_ms": BLOCK_MS,
        "block_count": int(plan.starts.size),
        "speech": plan.speech_blocks.tolist(),
        "cue": plan.cue_blocks.tolist(),
        "limits": [float(v) for v in plan.caps],
    }
    if not feasible:
        conflicts = _conflict_intervals(plan, bottlenecks, segments)
        return {
            **base,
            "gains": None,
            "message": "无解：提示音要求增益为 1，但相邻讲话段要求压低配乐，"
                       "20 ms 块间增益最多降 0.10 / 升 0.04，无法同时满足。"
                       "已标出造成冲突的讲话时段，请回到原声试听。",
            "conflicts": conflicts,
        }

    mix, peak, scale = render_mix(voice, music, plan, gains)
    return {
        **base,
        "gains": [float(v) for v in gains],
        "mix_base64": base64.b64encode(wav_bytes(mix)).decode("ascii"),
        "mix_peak_before_scale": round(peak, 6),
        "mix_scale": round(scale, 9),
        "stats": {
            "blocks": int(gains.size),
            "speech_blocks": int(plan.speech_blocks.sum()),
            "cue_blocks": int(plan.cue_blocks.sum()),
            "ducked_blocks": int((gains < MAX_GAIN - FEAS_TOL).sum()),
            "min_gain": float(np.min(gains)),
            "max_gain": float(np.max(gains)),
        },
    }
