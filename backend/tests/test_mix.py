"""自测：配乐试听的块级增益规划、频带约束、混音与冲突定位。

运行：python -m pytest backend/tests -v
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
from scipy.optimize import linprog
from scipy.signal import butter, sosfiltfilt

from app.audio import SAMPLE_RATE
from app.duck import (
    BLOCK_SAMPLES,
    MAX_FALL,
    MAX_RISE,
    SPEECH_LABELS,
    block_bounds,
    build_block_plan,
    load_music,
    parse_cues,
    parse_segments,
    plan_gains,
)
from app.main import app

client = TestClient(app)

LO, HI = 300.0, 3400.0


# ---------------------------------------------------------------- 工具

def wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE, stereo: bool = False) -> bytes:
    data = np.clip(y, -1.0, 1.0)
    if stereo:
        data = np.stack([data, data], axis=1)
    buf = io.BytesIO()
    wavfile.write(buf, sr, (data * 32767.0).astype(np.int16))
    return buf.getvalue()


def decode_wav(raw: bytes) -> np.ndarray:
    _sr, y = wavfile.read(io.BytesIO(raw))
    return y.astype(np.float64) / 32768.0


def band_rms_ref(x: np.ndarray) -> float:
    """独立的 300～3400 Hz 带通 RMS 实现（不复用被测代码）。"""
    sos = butter(4, [LO, HI], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    f = sosfiltfilt(sos, x)
    return float(np.sqrt(np.mean(f * f)))


def tone(seconds: float, freq: float, amp: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    return amp * np.sin(2.0 * np.pi * freq * t)


def make_pair(duration=3.0, voice_plan=None, music_amp=0.5, music_freq=1000.0):
    n = int(round(duration * SAMPLE_RATE))
    voice = np.zeros(n)
    for start, end, amp, freq in voice_plan or []:
        v = tone(end - start, freq, amp)
        i0 = int(round(start * SAMPLE_RATE))
        voice[i0:i0 + v.size] += v
    music = tone(duration, music_freq, music_amp)
    return voice, music


def segs_from_plan(duration, plan):
    """plan: [(start, end, label)] 必须铺满全轨。"""
    return [{"label": lab, "start": s, "end": e} for s, e, lab in plan]


def mix_request(voice, music, segments, cues=(), raw_voice=None, raw_music=None,
                sr_voice=SAMPLE_RATE, sr_music=SAMPLE_RATE,
                stereo_voice=False, stereo_music=False):
    files = {
        "file": ("voice.wav", raw_voice if raw_voice is not None else
                 wav_bytes(voice, sr_voice, stereo_voice), "audio/wav"),
        "music": ("music.wav", raw_music if raw_music is not None else
                  wav_bytes(music, sr_music, stereo_music), "audio/wav"),
    }
    data = {"segments": json.dumps(segments), "cues": json.dumps(list(cues))}
    return client.post("/api/mix", files=files, data=data)


# ---------------------------------------------------------------- 单元：规划

def brute_force_gains(caps: np.ndarray) -> np.ndarray:
    """用线性规划独立求逐块最高增益（最大化总和）。"""
    n = caps.size
    c = -np.ones(n)
    A, b = [], []
    for i in range(n):
        row = np.zeros(n); row[i] = 1.0
        A.append(row); b.append(float(caps[i]))          # g <= cap
        if i + 1 < n:
            row = np.zeros(n); row[i] = 1.0; row[i + 1] = -1.0
            A.append(row); b.append(MAX_FALL)            # g[i]-g[i+1] <= 0.10
            row = np.zeros(n); row[i + 1] = 1.0; row[i] = -1.0
            A.append(row); b.append(MAX_RISE)            # g[i+1]-g[i] <= 0.04
            row = np.zeros(n); row[i] = 1.0
            A.append(row); b.append(1.0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0.0, 1.0)] * n, method="highs")
    assert res.status == 0, res.message
    return -res.fun if False else np.clip(res.x, 0.0, 1.0)


def test_gain_plan_matches_linprog():
    rng = np.random.default_rng(7)
    for _ in range(12):
        n = int(rng.integers(5, 40))
        caps = rng.uniform(0.0, 1.0, n)
        caps[rng.random(n) < 0.3] = 1.0
        gains, feasible, _ = plan_gains(caps, np.zeros(n, dtype=bool))
        assert feasible
        ref = brute_force_gains(caps)
        assert np.allclose(gains, ref, atol=1e-6), (gains, ref)


def test_gain_plan_respects_rates_and_caps():
    caps = np.array([1.0, 0.2, 0.2, 1.0, 1.0])
    gains, feasible, _ = plan_gains(caps, np.zeros(5, dtype=bool))
    assert feasible
    assert np.all(gains <= caps + 1e-12)
    d = np.diff(gains)
    assert np.all(d <= MAX_RISE + 1e-12)
    assert np.all(d >= -MAX_FALL - 1e-12)


def test_plan_ducks_speech_and_recovers_after():
    duration = 4.0
    voice, music = make_pair(duration, [(1.0, 2.0, 0.2, 800.0), (3.0, 4.0, 0.2, 800.0)])
    segments = segs_from_plan(duration, [
        (0.0, 1.0, "silence"), (1.0, 2.0, "A"),
        (2.0, 3.0, "silence"), (3.0, 4.0, "B"),
    ])
    body = mix_request(voice, music, segments)
    assert body.status_code == 200, body.json()
    mix = body.json()
    assert mix["feasible"]
    gains = np.array(mix["gains"])
    limits = np.array(mix["limits"])

    assert np.all(gains <= limits + 1e-6)
    assert np.all(gains <= 1.0 + 1e-12)
    d = np.diff(gains)
    assert np.all(d <= MAX_RISE + 1e-9)
    assert np.all(d >= -MAX_FALL - 1e-9)

    # 讲话块内确实被压低
    voice_blocks = np.array(mix["speech"])
    assert voice_blocks[50] and voice_blocks[99]        # 1.00~2.00 s
    assert gains[50] < 0.5 and gains[99] < 0.5
    # 远离讲话的静音块恢复为 1
    assert gains[0] == pytest.approx(1.0)
    assert gains[120] == pytest.approx(1.0)            # 2.4 s 前后静音中段
    # 为了在第二段讲话开始时已压低，增益须从讲话前若干块开始下降
    assert gains[149] < 0.5 and gains[148] > gains[149]
    assert gains[160] < 0.5


def test_band_ducking_constraint_in_rendered_mix():
    duration = 3.0
    voice, music = make_pair(duration, [(0.8, 2.2, 0.25, 600.0)])
    segments = segs_from_plan(duration, [
        (0.0, 0.8, "silence"), (0.8, 2.2, "A"),
        (2.2, 2.6, "review"), (2.6, 3.0, "silence"),
    ])
    body = mix_request(voice, music, segments).json()
    assert body["feasible"]
    gains = np.array(body["gains"])
    starts, ends = block_bounds(voice.size)

    # 独立实现：每块单独做零相位带通，复核“同块”频带响度约束
    ducked = np.zeros(voice.size)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ducked[s:e] = gains[i] * music[s:e]
    checked = 0
    for i in np.where(np.array(body["speech"]))[0]:
        s, e = starts[i], ends[i]
        mv = band_rms_ref(voice[s:e])
        mm = band_rms_ref(ducked[s:e])
        assert mm <= 0.5 * mv + 1e-6, (i, mv, mm, gains[i])
        checked += 1
    assert checked > 50


def test_ducking_holds_for_whole_speech_segments():
    """稳健性：整段讲话（含段边界溢出）上连续测量的频带 RMS 大体也满足 ≤ 0.5。"""
    duration = 3.0
    voice, music = make_pair(duration, [(0.8, 2.2, 0.25, 600.0)])
    segments = segs_from_plan(duration, [
        (0.0, 0.8, "silence"), (0.8, 2.2, "A"),
        (2.2, 2.6, "review"), (2.6, 3.0, "silence"),
    ])
    body = mix_request(voice, music, segments).json()
    gains = np.array(body["gains"])
    starts, ends = block_bounds(voice.size)
    gain = np.zeros(voice.size)
    for i, (s, e) in enumerate(zip(starts, ends)):
        gain[s:e] = gains[i]
    sos = butter(4, [LO, HI], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    fv = sosfiltfilt(sos, voice)
    fd = sosfiltfilt(sos, gain * music)
    for seg in segments:
        if seg["label"] not in ("A", "B", "review"):
            continue
        s = int(seg["start"] * SAMPLE_RATE)
        e = int(seg["end"] * SAMPLE_RATE)
        mv = float(np.sqrt(np.mean(fv[s:e] ** 2)))
        mm = float(np.sqrt(np.mean(fd[s:e] ** 2)))
        # 相邻块能量串扰带来的偏差仅限块边界附近
        assert mm <= 0.5 * mv * 1.2 + 1e-3, (seg, mv, mm)


def test_cue_blocks_fixed_at_one_and_feasible():
    duration = 3.0
    # 提示音放在开头 0.1~0.3 s 的静音里，之后讲话；有充足时间降下来
    voice, music = make_pair(duration, [(1.0, 2.0, 0.2, 800.0)])
    segments = segs_from_plan(duration, [
        (0.0, 1.0, "silence"), (1.0, 2.0, "A"), (2.0, 3.0, "silence"),
    ])
    body = mix_request(voice, music, segments, cues=[[0.1, 0.3]]).json()
    assert body["feasible"], body
    gains = np.array(body["gains"])
    cue = np.array(body["cue"])
    assert cue.sum() >= 9
    assert np.allclose(gains[cue], 1.0)


def test_infeasible_cue_too_close_to_speech_reports_conflict():
    duration = 3.0
    # 讲话 0~1 s 后立刻要求 1.0 的提示音：0.04/块 的上升速率不允许
    voice, music = make_pair(duration, [(0.0, 1.0, 0.2, 800.0)])
    segments = segs_from_plan(duration, [
        (0.0, 1.0, "A"), (1.0, 1.2, "silence"),
        (1.2, 1.3, "review"), (1.3, 3.0, "silence"),
    ])
    resp = mix_request(voice, music, segments, cues=[[1.05, 1.15]])
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert not body["feasible"]
    assert body["gains"] is None and "mix_base64" not in body
    assert "冲突" in body["message"]
    # 冲突应定位到讲话段（0~1 s 的甲），而不是笼统的块
    labels = {(c["label"], round(c["start"], 2), round(c["end"], 2))
              for c in body["conflicts"]}
    assert ("A", 0.0, 1.0) in labels, body["conflicts"]


def test_infeasible_cue_inside_speech_reports_that_segment():
    duration = 2.0
    voice, music = make_pair(duration, [(0.5, 1.5, 0.2, 800.0)])
    segments = segs_from_plan(duration, [
        (0.0, 0.5, "silence"), (0.5, 1.5, "review"), (1.5, 2.0, "silence"),
    ])
    body = mix_request(voice, music, segments, cues=[[0.9, 1.1]]).json()
    assert not body["feasible"]
    assert any(c["label"] == "review" for c in body["conflicts"])


def test_mix_scaled_when_over_full_scale():
    duration = 2.0
    # 开头静音处提示音增益 1，两轨都接近满幅 -> 混音超过 1.0
    voice = tone(duration, 800.0, 0.9)
    music = tone(duration, 1000.0, 0.9)
    segments = segs_from_plan(duration, [(0.0, 2.0, "silence")])
    body = mix_request(voice, music, segments, cues=[[0.0, 2.0]]).json()
    assert body["feasible"]
    assert body["mix_peak_before_scale"] > 1.0
    assert 0.0 < body["mix_scale"] < 1.0
    out = decode_wav(base64.b64decode(body["mix_base64"]))
    assert np.max(np.abs(out)) <= 1.0 + 1e-4
    # 两轨等比缩小：提示音处配乐 / 原声 的幅度比保持 ~1
    gains = np.array(body["gains"])
    assert np.allclose(gains, 1.0)
    seg = slice(int(0.2 * SAMPLE_RATE), int(0.8 * SAMPLE_RATE))
    assert band_rms_ref(out[seg]) == pytest.approx(
        body["mix_scale"] * band_rms_ref((voice + music)[seg]), rel=0.02)


def test_block_grid_is_20ms_and_covers_tail():
    starts, ends = block_bounds(3 * SAMPLE_RATE + 77)
    assert BLOCK_SAMPLES == 320
    assert starts[0] == 0 and ends[-1] == 3 * SAMPLE_RATE + 77
    assert np.all(np.diff(starts) == BLOCK_SAMPLES)
    assert ends[-2] - starts[-2] == BLOCK_SAMPLES
    assert ends[-1] - starts[-1] == 77


# ---------------------------------------------------------------- 接口校验

def _basic():
    duration = 2.0
    voice, music = make_pair(duration, [(0.5, 1.5, 0.2, 800.0)])
    segments = segs_from_plan(duration, [
        (0.0, 0.5, "silence"), (0.5, 1.5, "A"), (1.5, 2.0, "silence"),
    ])
    return voice, music, segments


def test_reject_music_not_same_length():
    voice, music, segments = _basic()
    short = music[:-SAMPLE_RATE]  # 少 1 秒
    resp = mix_request(voice, short, segments)
    assert resp.status_code == 400
    assert "等长" in resp.json()["detail"]


def test_reject_bad_music_format():
    voice, music, segments = _basic()
    assert mix_request(voice, music, segments, sr_music=8000).status_code == 400
    assert mix_request(voice, music, segments, stereo_music=True).status_code == 400
    assert mix_request(voice, music, segments, raw_music=b"nope").status_code == 400


def test_reject_bad_segments_and_cues():
    voice, music, segments = _basic()
    bad_label = [dict(s) for s in segments]
    bad_label[1]["label"] = "C"
    assert mix_request(voice, music, bad_label).status_code == 400

    bad_json = client.post(
        "/api/mix",
        files={"file": ("v.wav", wav_bytes(voice), "audio/wav"),
               "music": ("m.wav", wav_bytes(music), "audio/wav")},
        data={"segments": "not json", "cues": "[]"})
    assert bad_json.status_code == 400

    resp = mix_request(voice, music, segments, cues=[[1.9, 2.5]])
    assert resp.status_code == 400
    assert "越界" in resp.json()["detail"]


def test_cues_are_merged():
    duration = 2.0
    merged = parse_cues("[[0.1, 0.2], [0.2, 0.3], [0.5, 0.6]]", duration)
    assert merged == [[0.1, 0.3], [0.5, 0.6]]


def test_parse_segments_validates_bounds():
    with pytest.raises(Exception):
        parse_segments(json.dumps([{"label": "A", "start": 0.0, "end": 5.0}]), 2.0)


def test_load_music_tolerates_one_sample_rounding():
    duration = 1.0
    music = tone(duration, 500.0, 0.3)
    n_expect = int(duration * SAMPLE_RATE)
    raw = wav_bytes(music[:-1])
    out = load_music(raw, n_expect)
    assert out.size == n_expect


def test_mix_endpoint_with_no_cues_and_silence_only():
    duration = 1.0
    voice, music = make_pair(duration)
    body = mix_request(voice, music, segs_from_plan(duration, [(0.0, 1.0, "silence")])).json()
    assert body["feasible"]
    assert np.allclose(np.array(body["gains"]), 1.0)
    assert body["stats"]["speech_blocks"] == 0


def test_end_to_end_demo_analysis_then_mix():
    """按网页流程：/api/demo -> /api/analyze -> /api/mix（等长配乐）。"""
    demo = client.get("/api/demo").json()
    raw = base64.b64decode(demo["audio_base64"])
    analyzed = client.post("/api/analyze", files={"file": ("demo.wav", raw, "audio/wav")},
                           data={"references": json.dumps(demo["references"])})
    assert analyzed.status_code == 200
    segments = [{k: s[k] for k in ("label", "start", "end")}
                for s in analyzed.json()["segments"]]
    assert any(s["label"] == "A" for s in segments)

    duration = demo["duration"]
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    music = 0.3 * np.sin(2 * np.pi * 523 * t) + 0.12 * np.sin(2 * np.pi * 1046 * t)

    # 提示音放在开头静音，应可行
    ok = client.post("/api/mix",
                     files={"file": ("demo.wav", raw, "audio/wav"),
                            "music": ("m.wav", wav_bytes(music), "audio/wav")},
                     data={"segments": json.dumps(segments),
                           "cues": json.dumps([[0.05, 0.30]])})
    assert ok.status_code == 200, ok.json()
    body = ok.json()
    assert body["feasible"], body.get("conflicts")
    assert body["block_count"] == int(np.ceil(len(t) / BLOCK_SAMPLES))
    out = decode_wav(base64.b64decode(body["mix_base64"]))
    assert out.size == decode_wav(raw).size   # 长度与录音一致
    gains = np.array(body["gains"])
    assert np.all(gains <= np.array(body["limits"]) + 1e-6)
    deltas = np.diff(gains)
    assert np.all(deltas <= MAX_RISE + 1e-9) and np.all(deltas >= -MAX_FALL - 1e-9)
    assert np.all(gains[np.array(body["cue"])] == 1.0)

    # 提示音落在甲讲话段里，应无解并给出冲突讲话段
    bad = client.post("/api/mix",
                      files={"file": ("demo.wav", raw, "audio/wav"),
                             "music": ("m.wav", wav_bytes(music), "audio/wav")},
                      data={"segments": json.dumps(segments),
                            "cues": json.dumps([[1.0, 1.2]])})
    bad_body = bad.json()
    assert bad.status_code == 200 and bad_body["feasible"] is False
    assert bad_body["gains"] is None
    assert any(c["label"] in SPEECH_LABELS for c in bad_body["conflicts"])
