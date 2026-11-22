"use strict";

/* 全局状态 */
const state = {
  file: null,
  objectUrl: null,
  buffer: null,      // 解码后的 AudioBuffer（画波形用）
  duration: 0,
  refs: { A: [], B: [] },   // 每项 {start, end}
  sel: null,              // 当前选区 {start, end}
  result: null,
  stopAt: null,           // 片段复听的停止点
  music: null,            // 配乐 {file, duration, buffer}
  cues: [],               // 提示音区间 {start, end}（不参与压低）
  mix: null,              // 最近一次配乐试听结果
  mixSeq: 0,              // 请求序号：迟到的响应一律丢弃
  analyzeSeq: 0,          // 分析请求序号，同样用于丢弃迟到响应
  musicSeq: 0,            // 配乐解码序号
  mixObjectUrl: null,
};

const LABELS = { A: "甲", B: "乙", review: "待复核", silence: "静音" };

const els = {};
["fileInput", "demoBtn", "fileInfo", "player", "wave", "selInfo",
 "playSelBtn", "clearSelBtn", "addABtn", "addBBtn",
 "countA", "countB", "listA", "listB",
 "analyzeBtn", "status", "legend", "timeline", "segTable",
 "musicInput", "musicInfo", "cueBtn", "cueList", "mixBtn", "backOriginalBtn",
 "mixStatus", "mixResult", "gainChart", "mixStats", "mixPlayer", "conflicts",
].forEach((id) => { els[id] = document.getElementById(id); });

let audioCtx = null;

/* ---------- 工具 ---------- */
function fmt(t) { return t.toFixed(2) + "s"; }

function setStatus(msg, kind) {
  els.status.textContent = msg || "";
  els.status.className = kind || "muted";
}

function b64ToBlob(b64, type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: type || "audio/wav" });
}

/* ---------- 载入音频 ---------- */
async function loadFile(file) {
  const seq = ++state.analyzeSeq;   // 让在途的分析请求作废
  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  els.player.src = state.objectUrl;

  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await file.arrayBuffer();
    const decoded = await audioCtx.decodeAudioData(buf.slice(0));
    if (seq !== state.analyzeSeq) return;   // 期间又换了录音
    state.buffer = decoded;
    state.duration = state.buffer.duration;
    els.fileInfo.textContent =
      `${file.name} · ${fmt(state.duration)} · ${state.buffer.sampleRate} Hz · ${state.buffer.numberOfChannels} 声道`;
  } catch (e) {
    if (seq !== state.analyzeSeq) return;
    state.buffer = null;
    state.duration = 0;
    els.fileInfo.textContent = file.name + "（浏览器无法解码，仍可尝试上传分析）";
  }
  state.sel = null;
  state.result = null;
  // 更换录音：旧的配乐 / 提示音 / 混音都不再对应这条时间轴，全部撤下
  state.music = null;
  state.musicSeq++;
  els.musicInput.value = "";
  state.cues = [];
  invalidateMix("已更换录音，请重新载入配乐并生成试听");
  renderAll();
}

/* ---------- 波形绘制 ---------- */
function drawWave() {
  const c = els.wave;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight || 160;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!state.buffer) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("请先载入音频", w / 2, h / 2);
    return;
  }

  const dur = state.duration;

  // 参考段底色
  for (const who of ["A", "B"]) {
    ctx.fillStyle = who === "A" ? "rgba(59,130,246,0.18)" : "rgba(249,115,22,0.18)";
    for (const r of state.refs[who]) {
      const x0 = (r.start / dur) * w, x1 = (r.end / dur) * w;
      ctx.fillRect(x0, 0, x1 - x0, h);
    }
  }

  // 提示音底色（不参与压低的块）
  ctx.fillStyle = "rgba(5,150,105,0.22)";
  for (const cue of state.cues) {
    const x0 = (cue.start / dur) * w, x1 = (cue.end / dur) * w;
    ctx.fillRect(x0, 0, x1 - x0, h);
  }

  // 波形
  const data = state.buffer.getChannelData(0);
  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / w));
  for (let x = 0; x < w; x++) {
    const start = Math.floor((x / w) * data.length);
    let min = 1, max = -1;
    for (let i = start; i < start + step && i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, ((1 - max) / 2) * h);
    ctx.lineTo(x + 0.5, ((1 - min) / 2) * h + 0.5);
  }
  ctx.stroke();

  // 秒刻度
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  for (let s = 0; s <= Math.floor(dur); s++) {
    const x = (s / dur) * w;
    ctx.fillRect(x, h - 10, 1, 6);
    ctx.fillText(s + "s", x + 2, h - 2);
  }

  // 当前选区
  if (state.sel) {
    const x0 = (state.sel.start / dur) * w, x1 = (state.sel.end / dur) * w;
    ctx.fillStyle = "rgba(17,24,39,0.18)";
    ctx.fillRect(x0, 0, x1 - x0, h);
    ctx.strokeStyle = "#111827";
    ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, h - 1);
  }

  // 播放头
  const t = els.player.currentTime;
  if (t > 0 && dur > 0) {
    const x = (t / dur) * w;
    ctx.strokeStyle = "#dc2626";
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, h);
    ctx.stroke();
  }
}

/* ---------- 波形交互：拖动选区 / 单击跳转 ---------- */
let dragStart = null;

function waveTime(e) {
  const rect = els.wave.getBoundingClientRect();
  const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
  return (x / rect.width) * state.duration;
}

els.wave.addEventListener("pointerdown", (e) => {
  if (!state.buffer) return;
  dragStart = waveTime(e);
  els.wave.setPointerCapture(e.pointerId);
});

els.wave.addEventListener("pointermove", (e) => {
  if (dragStart === null) return;
  const t = waveTime(e);
  state.sel = { start: Math.min(dragStart, t), end: Math.max(dragStart, t) };
  renderSel();
  drawWave();
});

els.wave.addEventListener("pointerup", (e) => {
  if (dragStart === null) return;
  const t = waveTime(e);
  if (Math.abs(t - dragStart) < 0.03) {   // 视为单击：跳转播放
    els.player.currentTime = t;
    els.player.play();
    state.sel = null;
  } else {
    state.sel = { start: Math.min(dragStart, t), end: Math.max(dragStart, t) };
    invalidateMix("选区已更改，配乐试听已撤下，请重新生成");
  }
  dragStart = null;
  renderSel();
  drawWave();
});

/* ---------- 选区与参考段 ---------- */
function renderSel() {
  const has = !!state.sel && state.sel.end - state.sel.start >= 0.05;
  els.selInfo.textContent = has
    ? `选区：${fmt(state.sel.start)} ~ ${fmt(state.sel.end)}（${fmt(state.sel.end - state.sel.start)}）`
    : "未选择片段";
  els.playSelBtn.disabled = !has;
  els.clearSelBtn.disabled = !has;
  els.addABtn.disabled = !has;
  els.addBBtn.disabled = !has;
  els.cueBtn.disabled = !has || !state.music;
}

function overlaps(a, b) { return a.start < b.end - 1e-6 && b.start < a.end - 1e-6; }

function addRef(who) {
  if (!state.sel) return;
  const seg = { start: +state.sel.start.toFixed(3), end: +state.sel.end.toFixed(3) };
  const all = state.refs.A.concat(state.refs.B);
  if (all.some((r) => overlaps(r, seg))) {
    setStatus("该选区与已有参考段重叠，请重新选择", "err");
    return;
  }
  state.refs[who].push(seg);
  state.refs[who].sort((x, y) => x.start - y.start);
  state.sel = null;
  setStatus("");
  // 改参考：旧分析结果已失效，旧的配乐混音随之撤下
  state.result = null;
  invalidateMix("已修改参考段，请重新分析并生成配乐试听");
  renderAll();
}

function delRef(who, i) {
  state.refs[who].splice(i, 1);
  state.result = null;
  invalidateMix("已修改参考段，请重新分析并生成配乐试听");
  renderAll();
}

function playRange(start, end) {
  els.player.currentTime = Math.max(0, start + 1e-3);
  state.stopAt = end;
  els.player.play();
}

function renderRefs() {
  for (const who of ["A", "B"]) {
    const list = who === "A" ? els.listA : els.listB;
    list.innerHTML = "";
    state.refs[who].forEach((r, i) => {
      const li = document.createElement("li");
      const txt = document.createElement("span");
      txt.className = "grow";
      txt.textContent = `第${i + 1}段：${fmt(r.start)} ~ ${fmt(r.end)}`;
      const play = document.createElement("button");
      play.textContent = "试听";
      play.onclick = () => playRange(r.start, r.end);
      const del = document.createElement("button");
      del.textContent = "删除";
      del.onclick = () => delRef(who, i);
      li.append(txt, play, del);
      list.appendChild(li);
    });
    (who === "A" ? els.countA : els.countB).textContent = state.refs[who].length;
  }
}

/* ---------- 校验（提交前本地预检，服务端仍会复核） ---------- */
function validateRefs() {
  if (!state.file) return "请先载入录音文件";
  for (const who of ["A", "B"]) {
    const n = state.refs[who].length;
    if (n < 3 || n > 8) return `${LABELS[who]}的参考段须为 3～8 段，当前 ${n} 段`;
  }
  const all = state.refs.A.concat(state.refs.B).sort((a, b) => a.start - b.start);
  for (const r of all) {
    if (r.start < 0 || r.end > state.duration + 1e-3 || r.end - r.start < 0.3)
      return `参考段 [${fmt(r.start)}, ${fmt(r.end)}] 越界或过短`;
  }
  for (let i = 1; i < all.length; i++) {
    if (all[i].start < all[i - 1].end - 1e-6) return "存在相互重叠的参考段";
  }
  return null;
}

function refreshAnalyzeBtn() {
  els.analyzeBtn.disabled = !state.file ||
    state.refs.A.length < 3 || state.refs.B.length < 3;
}

/* ---------- 分析 ---------- */
async function analyze() {
  const err = validateRefs();
  if (err) { setStatus(err, "err"); return; }

  // 重做分段：先撤下旧混音并作废在途请求，迟到的响应也不能恢复旧试听
  const seq = ++state.analyzeSeq;
  invalidateMix("正在重新分析，旧配乐试听已撤下");

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("references", JSON.stringify({
    A: state.refs.A.map((r) => [r.start, r.end]),
    B: state.refs.B.map((r) => [r.start, r.end]),
  }));

  els.analyzeBtn.disabled = true;
  setStatus("分析中……", "muted");
  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: fd });
    const body = await resp.json();
    if (seq !== state.analyzeSeq) return;   // 已被更新一次分析 / 新录音取代
    if (!resp.ok) {
      setStatus("分析被拒绝：" + (body.detail || resp.statusText), "err");
      return;
    }
    state.result = body;
    setStatus(
      `分析完成：甲 ${body.stats.A} 帧 / 乙 ${body.stats.B} 帧 / 待复核 ${body.stats.review} 帧 / 静音 ${body.stats.silence} 帧`,
      "ok");
    renderResult();
    renderMixControls();
  } catch (e) {
    if (seq === state.analyzeSeq) setStatus("请求失败：" + e.message, "err");
  } finally {
    if (seq === state.analyzeSeq) refreshAnalyzeBtn();
  }
}

/* ---------- 结果展示 ---------- */
function renderResult() {
  const res = state.result;
  if (!res) {
    els.timeline.classList.add("hidden");
    els.segTable.classList.add("hidden");
    els.legend.classList.add("hidden");
    return;
  }
  els.legend.classList.remove("hidden");
  const tl = els.timeline;
  tl.classList.remove("hidden");
  tl.innerHTML = "";
  const dur = res.duration;

  for (const seg of res.segments) {
    const div = document.createElement("div");
    div.className = "seg " + seg.label;
    div.style.left = (seg.start / dur) * 100 + "%";
    div.style.width = Math.max(((seg.end - seg.start) / dur) * 100, 0.15) + "%";
    if (seg.label === "A" || seg.label === "B") {
      div.style.opacity = (0.45 + 0.55 * seg.confidence).toFixed(2);
    }
    const pct = Math.round(seg.confidence * 100) + "%";
    div.title = `${LABELS[seg.label]} · ${fmt(seg.start)}~${fmt(seg.end)} · 置信度 ${pct}`;
    if (seg.end - seg.start > 0.35 && seg.label !== "silence") {
      div.textContent = `${LABELS[seg.label]} ${pct}`;
    }
    div.onclick = () => playRange(seg.start, seg.end);
    tl.appendChild(div);
  }

  // 提示音圈选（不参与压低的时段）；与讲话段重叠处即冲突来源
  for (const cue of state.cues) {
    const mark = document.createElement("div");
    mark.className = "cue-mark";
    mark.style.left = (cue.start / dur) * 100 + "%";
    mark.style.width = Math.max(((cue.end - cue.start) / dur) * 100, 0.2) + "%";
    mark.title = `提示音 ${fmt(cue.start)} ~ ${fmt(cue.end)}（不参与压低）`;
    tl.appendChild(mark);
  }
  // 无解时圈出造成冲突的讲话时段
  if (state.mix && !state.mix.feasible) {
    for (const c of state.mix.conflicts) {
      const mark = document.createElement("div");
      mark.className = "conflict-mark";
      mark.style.left = (c.start / dur) * 100 + "%";
      mark.style.width = Math.max(((c.end - c.start) / dur) * 100, 0.2) + "%";
      mark.title = c.reason || "造成冲突的讲话时段";
      mark.onclick = () => playRange(c.start, c.end);
      tl.appendChild(mark);
    }
  }

  const ph = document.createElement("div");
  ph.className = "playhead";
  ph.id = "tlPlayhead";
  ph.style.left = "0%";
  tl.appendChild(ph);

  // 明细表
  const box = els.segTable;
  box.classList.remove("hidden");
  let html = "<table><thead><tr><th>#</th><th>标签</th><th>开始</th><th>结束</th><th>时长</th><th>置信度</th></tr></thead><tbody>";
  res.segments.forEach((seg, i) => {
    html += `<tr data-i="${i}"><td>${i + 1}</td><td>${LABELS[seg.label]}</td>` +
      `<td>${fmt(seg.start)}</td><td>${fmt(seg.end)}</td>` +
      `<td>${fmt(seg.end - seg.start)}</td><td>${Math.round(seg.confidence * 100)}%</td></tr>`;
  });
  html += "</tbody></table>";
  box.innerHTML = html;
  box.querySelectorAll("tr[data-i]").forEach((tr) => {
    tr.onclick = () => {
      const seg = state.result.segments[+tr.dataset.i];
      playRange(seg.start, seg.end);
    };
  });
}

/* ---------- 配乐试听 ---------- */

/**
 * 撤下当前混音（若正在播放则停止并释放音频），并让所有在途的试听请求作废：
 * 请求序号一递增，迟到的响应在回调里比对序号后会被直接丢弃。
 */
function invalidateMix(msg) {
  state.mixSeq++;
  state.mix = null;
  if (state.mixObjectUrl) {
    URL.revokeObjectURL(state.mixObjectUrl);
    state.mixObjectUrl = null;
  }
  if (els.mixPlayer) {
    els.mixPlayer.pause();
    els.mixPlayer.removeAttribute("src");
    els.mixPlayer.load();
  }
  if (els.mixResult) els.mixResult.classList.add("hidden");
  if (els.mixStatus) els.mixStatus.textContent = msg || "";
  renderResult();          // 清除时间轴上的冲突圈选
  renderMixControls();
}

function renderMixControls() {
  if (!els.mixBtn) return;
  const ready = !!(state.file && state.result && state.music);
  els.mixBtn.disabled = !ready;
  els.mixBtn.textContent = state.mix ? "重新生成配乐试听" : "生成配乐试听";
  els.musicInfo.textContent = state.music
    ? `${state.music.file.name} · ${fmt(state.music.duration)}`
    : "未载入配乐";
  if (!els.mixStatus.textContent) {
    if (state.music && !state.result) {
      els.mixStatus.textContent = "请先完成第 3 步分析，再生成配乐试听";
    } else if (!state.music) {
      els.mixStatus.textContent = "请先载入与录音等长的配乐 WAV";
    }
  }
}

async function loadMusic(file) {
  if (!state.file) {
    setStatus("请先载入录音，再上传等长配乐", "err");
    return;
  }
  const seq = ++state.musicSeq;
  const expected = state.duration;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await audioCtx.decodeAudioData(await file.arrayBuffer());
    if (seq !== state.musicSeq) return;   // 期间已换录音 / 已选别的配乐
    if (Math.abs(buf.duration - expected) > 0.01) {
      throw new Error(`配乐时长 ${fmt(buf.duration)} 与录音 ${fmt(expected)} 不一致`);
    }
    state.music = { file, duration: buf.duration, buffer: buf };
  } catch (e) {
    if (seq !== state.musicSeq) return;
    state.music = null;
    els.musicInput.value = "";
    els.mixStatus.textContent = "配乐载入失败：" + e.message;
    renderMixControls();
    return;
  }
  // 换配乐：旧混音作废
  invalidateMix("已更换配乐，请重新生成试听");
  renderMixControls();
}

function addCue() {
  if (!state.sel || !state.music) return;
  const seg = { start: +state.sel.start.toFixed(3), end: +state.sel.end.toFixed(3) };
  if (state.cues.some((c) => overlaps(c, seg))) {
    state.mixStatus.textContent = "该选区与已有提示音重叠，请重新选择";
    return;
  }
  // 提示音必须落在配乐能覆盖的块里；提示音增益固定为 1，不参与压低
  state.cues.push(seg);
  state.cues.sort((a, b) => a.start - b.start);
  state.sel = null;
  // 改选区 / 改提示音：旧混音作废
  invalidateMix("提示音已更新，请重新生成试听");
  renderAll();
}

function delCue(i) {
  state.cues.splice(i, 1);
  invalidateMix("提示音已更新，请重新生成试听");
  renderAll();
}

function renderCues() {
  els.cueList.innerHTML = "";
  if (!state.cues.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "暂无提示音";
    els.cueList.appendChild(li);
    return;
  }
  state.cues.forEach((c, i) => {
    const li = document.createElement("li");
    const txt = document.createElement("span");
    txt.className = "grow";
    txt.textContent = `提示音 ${i + 1}：${fmt(c.start)} ~ ${fmt(c.end)}`;
    const play = document.createElement("button");
    play.textContent = "试听";
    play.onclick = () => playRange(c.start, c.end);
    const del = document.createElement("button");
    del.textContent = "删除";
    del.onclick = () => delCue(i);
    li.append(txt, play, del);
    els.cueList.appendChild(li);
  });
}

async function makeMix() {
  if (!state.file || !state.result || !state.music) return;
  const seq = ++state.mixSeq;
  els.mixBtn.disabled = true;
  els.mixStatus.textContent = "正在计算配乐增益并混音……";
  els.mixResult.classList.add("hidden");

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("music", state.music.file);
  fd.append("segments", JSON.stringify(state.result.segments.map((s) => ({
    label: s.label, start: s.start, end: s.end,
  }))));
  fd.append("cues", JSON.stringify(state.cues.map((c) => [c.start, c.end])));

  try {
    const resp = await fetch("/api/mix", { method: "POST", body: fd });
    const body = await resp.json();
    if (seq !== state.mixSeq) return;   // 迟到的响应：状态已变化，丢弃
    if (!resp.ok) {
      els.mixStatus.textContent = "生成失败：" + (body.detail || resp.statusText);
      return;
    }
    state.mix = body;
    renderMixResult();
  } catch (e) {
    if (seq === state.mixSeq) els.mixStatus.textContent = "请求失败：" + e.message;
  } finally {
    if (seq === state.mixSeq) renderMixControls();
  }
}

function renderMixResult() {
  const mix = state.mix;
  if (!mix) return;
  els.mixResult.classList.remove("hidden");
  els.conflicts.classList.add("hidden");
  els.conflicts.innerHTML = "";
  els.mixPlayer.classList.add("hidden");
  renderResult();          // 时间轴上同步提示音 / 冲突圈选
  drawGainChart();

  if (!mix.feasible) {
    els.mixStatus.textContent = "";
    els.mixStats.textContent = "";
    els.conflicts.classList.remove("hidden");
    const title = document.createElement("div");
    title.className = "conflict-title";
    title.textContent = mix.message + " 点击时段可回到原声复听：";
    els.conflicts.appendChild(title);
    if (!mix.conflicts.length) {
      const p = document.createElement("div");
      p.className = "muted";
      p.textContent = "冲突来自提示音之间的增益速率限制，未能定位到具体讲话段。";
      els.conflicts.appendChild(p);
    }
    for (const c of mix.conflicts) {
      const btn = document.createElement("button");
      btn.textContent = `${LABELS[c.label] || "冲突"} ${fmt(c.start)} ~ ${fmt(c.end)} 原声复听`;
      btn.title = c.reason || "";
      btn.onclick = () => {
        els.mixPlayer.pause();
        els.mixPlayer.classList.add("hidden");
        els.backOriginalBtn.classList.add("hidden");
        playRange(c.start, c.end);
      };
      els.conflicts.appendChild(btn);
    }
    els.backOriginalBtn.classList.remove("hidden");
    return;
  }

  // 有解：画曲线 + 试听混音
  const st = mix.stats;
  els.mixStatus.textContent = "配乐试听已生成。";
  els.mixStats.textContent =
    `共 ${st.blocks} 块（20 ms）· 讲话块 ${st.speech_blocks} · 提示音块 ${st.cue_blocks} · ` +
    `压低块 ${st.ducked_blocks} · 增益 ${st.min_gain.toFixed(2)}～${st.max_gain.toFixed(2)}` +
    (mix.mix_scale < 1
      ? ` · 混音峰值 ${mix.mix_peak_before_scale.toFixed(2)} 超满幅，两轨等比缩小至 ${(mix.mix_scale * 100).toFixed(0)}%`
      : "");
  setMixAudio(mix.mix_base64);
}

function setMixAudio(b64) {
  if (state.mixObjectUrl) URL.revokeObjectURL(state.mixObjectUrl);
  state.mixObjectUrl = URL.createObjectURL(b64ToBlob(b64));
  els.mixPlayer.src = state.mixObjectUrl;
  els.mixPlayer.classList.remove("hidden");
  els.backOriginalBtn.classList.remove("hidden");
}

function backToOriginal() {
  els.mixPlayer.pause();
  els.mixPlayer.classList.add("hidden");
  els.backOriginalBtn.classList.add("hidden");
  playRange(0, state.duration);
}

function drawGainChart() {
  const mix = state.mix;
  const c = els.gainChart;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight || 150;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!mix || !mix.gains) return;

  const padL = 34, padB = 18, padT = 8;
  const plotW = w - padL - 6, plotH = h - padB - padT;
  const n = mix.gains.length;
  const xOf = (i) => padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  const yOf = (g) => padT + (1 - g) * plotH;

  // 讲话块 / 提示音块底色
  for (let i = 0; i < n; i++) {
    const x0 = padL + (i / n) * plotW;
    const bw = Math.max(plotW / n, 0.5);
    if (mix.speech[i]) {
      ctx.fillStyle = "rgba(59,130,246,0.16)";
      ctx.fillRect(x0, padT, bw, plotH);
    }
    if (mix.cue[i]) {
      ctx.fillStyle = "rgba(5,150,105,0.22)";
      ctx.fillRect(x0, padT, bw, plotH);
    }
  }

  // 网格与刻度
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#6b7280";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  for (const g of [0, 0.5, 1]) {
    const y = yOf(g);
    ctx.beginPath();
    ctx.moveTo(padL, y); ctx.lineTo(w - 6, y);
    ctx.stroke();
    ctx.fillText(g.toFixed(1), padL - 4, y + 3);
  }
  ctx.textAlign = "center";
  const dur = mix.duration;
  for (let s = 0; s <= Math.floor(dur); s++) {
    const x = padL + (s / dur) * plotW;
    ctx.fillRect(x, padT + plotH, 1, 4);
    ctx.fillText(s + "s", x, h - 4);
  }

  // 该块上限（浅色阶梯）
  if (mix.limits) {
    ctx.strokeStyle = "#c4b5fd";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const y = yOf(mix.limits[i]);
      if (i === 0) ctx.moveTo(xOf(i), y);
      else { ctx.lineTo(xOf(i - 1), y); ctx.lineTo(xOf(i), y); }
    }
    ctx.stroke();
  }

  // 增益曲线
  ctx.strokeStyle = "#7c3aed";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i), y = yOf(mix.gains[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function renderMixPanelState() {
  renderMixControls();
}

/* ---------- 播放控制 ---------- */
els.player.addEventListener("timeupdate", () => {
  if (state.stopAt !== null && els.player.currentTime >= state.stopAt) {
    els.player.pause();
    state.stopAt = null;
  }
  drawWave();
  const res = state.result;
  const ph = document.getElementById("tlPlayhead");
  if (res && ph) ph.style.left = (els.player.currentTime / res.duration) * 100 + "%";
});

/* ---------- 演示 ---------- */
async function loadDemo() {
  setStatus("正在生成演示音频……", "muted");
  try {
    const resp = await fetch("/api/demo");
    const body = await resp.json();
    const blob = b64ToBlob(body.audio_base64);
    const file = new File([blob], "demo.wav", { type: "audio/wav" });
    await loadFile(file);
    state.refs.A = body.references.A.map(([s, e]) => ({ start: s, end: e }));
    state.refs.B = body.references.B.map(([s, e]) => ({ start: s, end: e }));
    setStatus("已载入演示音频并预填参考段，可直接点击「开始分析」", "ok");
    renderAll();
  } catch (e) {
    setStatus("演示载入失败：" + e.message, "err");
  }
}

/* ---------- 汇总渲染 ---------- */
function renderAll() {
  renderSel();
  renderRefs();
  renderCues();
  refreshAnalyzeBtn();
  renderResult();
  renderMixPanelState();
  drawWave();
  if (state.mix && state.mix.feasible) drawGainChart();
}

/* ---------- 事件绑定 ---------- */
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) loadFile(e.target.files[0]);
});
els.demoBtn.addEventListener("click", loadDemo);
els.playSelBtn.addEventListener("click", () => state.sel && playRange(state.sel.start, state.sel.end));
els.clearSelBtn.addEventListener("click", () => {
  if (!state.sel) return;
  state.sel = null;
  invalidateMix("选区已清除，配乐试听已撤下，请重新生成");
  renderAll();
});
els.addABtn.addEventListener("click", () => addRef("A"));
els.addBBtn.addEventListener("click", () => addRef("B"));
els.analyzeBtn.addEventListener("click", analyze);
els.musicInput.addEventListener("change", (e) => {
  if (e.target.files.length) loadMusic(e.target.files[0]);
});
els.cueBtn.addEventListener("click", addCue);
els.mixBtn.addEventListener("click", makeMix);
els.backOriginalBtn.addEventListener("click", backToOriginal);
els.mixPlayer.addEventListener("play", () => els.player.pause());
els.player.addEventListener("play", () => els.mixPlayer.pause());
window.addEventListener("resize", () => { drawWave(); if (state.mix) drawGainChart(); });

renderAll();
