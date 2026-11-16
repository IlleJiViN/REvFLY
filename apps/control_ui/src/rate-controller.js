export function startRateControl() {
  const root = document.getElementById("control-ui-root");
  if (root?.dataset.controlBound === "true") return;
  if (root) root.dataset.controlBound = "true";
  "use strict";

  const byId = (id) => document.getElementById(id);
  const ids = ["connection-status", "page-error", "page-error-title", "page-error-text", "retry-connection", "control-form", "control-fields", "goal-scale", "goal-scale-number", "initial-delta", "threshold", "correction-steps", "correction-help", "render-video", "form-error", "run-button", "target-preview", "preview-placeholder", "goal-source", "initial-angle-info", "import-target", "export-target", "goal-file", "job-panel", "job-title", "job-detail", "job-id", "job-steps", "elapsed-time", "retry-job", "empty-results", "results", "results-title", "result-context", "criterion-badge", "correction-warning", "correction-warning-text", "correction-error", "result-notice", "fast-rmse", "selected-rmse", "max-error", "max-error-caption", "selected-stage", "threshold-caption", "correction-caption", "angle-chart", "rate-chart", "result-video", "video-placeholder", "time-cursor", "time-output", "cursor-readout", "protocol-link", "bundle-download", "status-announcement"];
  const ui = Object.fromEntries(ids.map((id) => [id, byId(id)]));
  const state = { ready: false, busy: false, loadingGoal: false, goalReady: false, goal: null, customGoal: false, source: "", initialAngle: null, goalRequest: 0, jobId: null, request: null, result: null, trace: null, cursor: 0, pollTimer: null, clockTimer: null, startedAt: null, stage: null, pollInFlight: false };
  const closedLoopInput = byId("mode-closed_loop");
  const feedbackSettings = byId("feedback-settings");
  const feedbackKp = byId("feedback-kp");
  const feedbackKd = byId("feedback-kd");
  const scopeInput = byId("control-scope");
  const goalFamily = byId("goal-family");
  const goalJoint = byId("goal-joint");
  const resultJoint = byId("result-joint");
  const resultMuscle = byId("result-muscle");
  const bioInput = byId("bio-mode");
  const storageKey = "sensory-translator:pending-rate-job";
  const radiansToDegrees = (value) => value * 180 / Math.PI;
  const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const finiteArray = (value, length) => Array.isArray(value) && value.length === length && value.every(finite);
  const finiteMatrix = (value, rows, cols) => Array.isArray(value) && value.length === rows && value.every((row) => finiteArray(row, cols));
  const bioLabels = { none: "연속 발화율만", deterministic_isi: "일정한 발화 간격", poisson_refractory: "확률적 발화 · 불응기", direct_lif: "LIF 운동뉴런" };
  const baselineFamily = "kinematic_teacher_scaled";
  const familyLabels = { [baselineFamily]: "기준 궤적 배율", swing_scaled: "다리 휘두르기", decoupled: "부위별 크기 조절", rhythmic_double_tap: "두 번 두드리기", reach_and_hold: "뻗고 유지하기" };
  const jointLabels = {
    joint_LFCoxa_yaw: "기절 좌우 회전", joint_LFCoxa_pitch: "기절 앞뒤 회전", joint_LFCoxa_roll: "기절 비틀림",
    joint_LFTrochanter_yaw: "전절 좌우 회전", joint_LFTrochanter_pitch: "전절 앞뒤 회전", joint_LFTrochanter_roll: "전절 비틀림",
    joint_LFTibia_pitch: "경절 굽힘",
  };
  const modeLabels = { fast: "빠른 생성", auto: "필요할 때 보정", refine: "입력 보정 실행", closed_loop: "관절 피드백" };
  const stageLabels = { queued: "실행을 기다리고 있어요.", running: "작업을 실행하고 있어요.", predicting: "발화율 파형을 만들고 있어요.", fast_verification: "생성한 입력으로 움직임을 검증하고 있어요.", fast_verified: "빠른 생성 결과의 검증을 마쳤어요.", correcting: "입력 파형을 보정하고 있어요.", correction_failed: "빠른 생성 결과를 유지하고 있어요.", final_verification: "선택된 결과를 다시 검증하고 있어요.", rendering: "탑뷰 영상을 만들고 있어요.", completed: "결과를 확인할 수 있어요.", failed: "작업을 완료하지 못했어요." };

  function messageOf(value) {
    if (typeof value === "string") return value;
    if (value && typeof value.message === "string") return value.message;
    if (value && typeof value.error === "string") return value.error;
    if (value !== undefined && value !== null) return JSON.stringify(value);
    return "상세 오류가 제공되지 않았습니다.";
  }

  async function requestJSON(path, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { cache: "no-store", ...options, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch (_) { throw new Error(`서버 응답을 읽지 못했습니다. 연결과 서버 실행 상태를 확인해 주세요. (HTTP ${response.status})`); }
      if (!response.ok) {
        const error = new Error(messageOf(data.error ?? data.message ?? data));
        error.httpStatus = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답 시간이 길어졌습니다. 잠시 후 다시 확인해 주세요.");
      if (error instanceof TypeError) throw new Error("서버에 연결하지 못했습니다. 네트워크와 서버 실행 상태를 확인해 주세요.");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function announce(text) { ui["status-announcement"].textContent = text; }
  function setConnection(status, text) {
    ui["connection-status"].dataset.state = status;
    ui["connection-status"].lastElementChild.textContent = text;
  }
  function showPageError(title, text) {
    ui["page-error-title"].textContent = title;
    ui["page-error-text"].textContent = text;
    ui["page-error"].hidden = false;
  }
  function formError(text = "") {
    ui["form-error"].textContent = text;
    ui["form-error"].hidden = !text;
  }
  function selectedMode() { return document.querySelector('input[name="mode"]:checked')?.value || "fast"; }
  function isMulti() { return state.multijoint?.ready === true && scopeInput.value === "lf15"; }
  function selectedFamily() { return goalFamily.value || baselineFamily; }
  function selectedGoalJoint() { return selectedFamily() === baselineFamily ? goalJoint.value : ""; }
  function jointLabel(name) { return jointLabels[name] || name; }
  function setOptions(input, entries) {
    const previous = input.value;
    input.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => value === previous)) input.value = previous;
  }
  function scopeSettings({ resetGains = false } = {}) {
    const multi = isMulti();
    const capabilities = multi ? state.multijoint : state.capabilities;
    const defaults = capabilities?.defaults || {};
    const modes = capabilities?.modes;
    if (Array.isArray(modes)) {
      for (const radio of document.querySelectorAll('input[name="mode"]')) {
        radio.disabled = !modes.includes(radio.value);
        radio.closest("label").hidden = radio.disabled;
      }
      if (!modes.includes(selectedMode())) (byId(`mode-${defaults.mode}`) || byId("mode-fast")).checked = true;
    } else {
      closedLoopInput.disabled = true;
      closedLoopInput.closest("label").hidden = true;
    }
    if (resetGains && finite(defaults.feedback_kp)) feedbackKp.value = defaults.feedback_kp;
    if (resetGains && finite(defaults.feedback_kd)) feedbackKd.value = defaults.feedback_kd;
    if (resetGains && Number.isInteger(defaults.correction_steps)) ui["correction-steps"].value = defaults.correction_steps;
    byId("multijoint-goal-settings").hidden = !multi;
    byId("goal-family-field").hidden = !multi || goalFamily.options.length < 2;
    byId("goal-joint-field").hidden = multi && selectedFamily() !== baselineFamily;
    byId("bio-settings").hidden = !multi || bioInput.options.length < 2;
    byId("scope-description").textContent = multi ? "왼쪽 앞다리 7축 · 근육 15개 · 교사 보조 입력 0개" : "왼쪽 앞다리 경절 1축 · 근육 2개 · 교사 보조 13개";
    byId("preview-scope").textContent = multi ? "TOP / LEFT FRONT LEG" : "TOP / LF TIBIA";
    byId("preview-pools").textContent = multi ? "근육 15개" : "근육 2개";
    const identity = byId("engine-model");
    const model = multi ? capabilities.model_id : capabilities?.model?.model_id;
    identity.hidden = !model;
    identity.textContent = [model, state.capabilities?.backend?.name].filter(Boolean).join(" · ");
  }
  function updateControls() {
    ui["control-fields"].disabled = state.busy;
    ui["run-button"].disabled = state.busy || !state.ready || state.loadingGoal || !state.goalReady || !state.goal;
    (byId("run-button-label") || ui["run-button"].lastElementChild).textContent = state.busy ? "생성 및 검증 중…" : "움직임 생성 및 검증";
    const feedback = selectedMode() === "closed_loop";
    const optimization = ["auto", "refine"].includes(selectedMode());
    ui["correction-steps"].disabled = state.busy || !optimization;
    ui["correction-help"].textContent = feedback ? "시뮬레이션 중 2 ms마다 관절을 보정합니다." : optimization ? "0~20회, 물리 검증을 거쳐 선택합니다." : "빠른 생성에서는 보정하지 않아요.";
    feedbackSettings.hidden = !feedback;
    ui["import-target"].disabled = state.busy;
    ui["goal-file"].disabled = state.busy;
    ui["export-target"].disabled = state.busy || !state.goal;
    document.querySelectorAll("[data-scale]").forEach((button) => {
      button.setAttribute("aria-pressed", String(!state.customGoal && Number(button.dataset.scale) === Number(ui["goal-scale"].value)));
    });
  }

  function checkedTarget(target, times, multi = isMulti()) {
    const valid = multi ? finiteMatrix(target, 211, 7) : finiteArray(target, 211);
    if (!valid) throw new Error(multi ? "목표 각도는 211개 시점마다 7축 값을 담아야 합니다." : "목표 각도는 유한한 숫자 211개여야 합니다.");
    if (!target.flat().map(radiansToDegrees).every(Number.isFinite)) throw new Error("목표 각도의 크기와 단위를 확인해 주세요.");
    const grid = Array.from({ length: 211 }, (_, i) => i * 0.002);
    if (times !== undefined && (!finiteArray(times, 211) || times.some((t, i) => Math.abs(t - grid[i]) > 1e-9))) {
      throw new Error("time_s는 0, 0.002, …, 0.420초의 211개 값이어야 합니다.");
    }
    return { target_rad: target.slice(), time_s: grid };
  }

  function setGoal(goal, source, custom = false) {
    state.goal = checkedTarget(goal.target_rad, goal.time_s);
    state.goalReady = true;
    state.source = source;
    state.customGoal = custom;
    ui["goal-source"].textContent = source;
    ui["preview-placeholder"].hidden = true;
    renderInitialAngle();
    updateControls();
    drawPreview();
  }

  function renderInitialAngle() {
    ui["initial-angle-info"].textContent = finite(state.initialAngle) ? `물리 시작 각도 ${state.initialAngle.toFixed(2)}°` : "";
  }

  async function loadGoal({ preserveCustom = false } = {}) {
    const scale = Number(ui["goal-scale-number"].value);
    const delta = Number(ui["initial-delta"].value);
    if (!Number.isFinite(scale) || scale < 0.8 || scale > 1.2 || !Number.isFinite(delta) || delta < -5 || delta > 5) {
      state.loadingGoal = false; state.goalReady = false;
      formError("움직임 크기는 0.8~1.2배, 시작 각도 변화는 −5°~+5°로 입력해 주세요.");
      updateControls(); return;
    }
    const version = ++state.goalRequest;
    state.loadingGoal = true;
    state.goalReady = false;
    updateControls();
    try {
      const multi = isMulti();
      const query = new URLSearchParams({ scale: String(scale) });
      if (multi) {
        query.set("scope", "lf15");
        if (selectedFamily() !== baselineFamily) query.set("family", selectedFamily());
        if (selectedGoalJoint()) query.set("joints", selectedGoalJoint());
      } else query.set("initial_delta_deg", String(delta));
      const goal = await requestJSON(`/api/rate/goal?${query}`);
      if (version !== state.goalRequest) return;
      state.initialAngle = multi && Array.isArray(goal.initial_joint_deg) ? goal.initial_joint_deg[6] + delta : goal.initial_tibia_deg;
      if (preserveCustom && state.customGoal && state.goal) { state.goalReady = true; renderInitialAngle(); }
      else {
        const preview = multi ? ` · ${jointLabel(selectedGoalJoint() || state.multijoint.joint_names[6])} 미리보기` : "";
        const name = multi ? familyLabels[selectedFamily()] : "기본 동작";
        setGoal(goal, `${name} ${scale.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}×${preview} · 211개 시점`, false);
      }
      formError();
    } catch (error) {
      if (version !== state.goalRequest) return;
      formError(`목표를 불러오지 못했습니다. ${error.message} 연결이 복구되면 프리셋을 다시 선택해 주세요.`);
      if (!state.goal) ui["preview-placeholder"].textContent = "연결을 확인한 뒤 목표를 다시 불러와 주세요.";
    } finally {
      if (version === state.goalRequest) { state.loadingGoal = false; updateControls(); }
    }
  }

  async function bootstrap() {
    ui["retry-connection"].disabled = true;
    setConnection("loading", "연결 확인 중");
    try {
      const capabilities = await requestJSON("/api/rate/capabilities");
      state.ready = capabilities.ready === true;
      state.capabilities = capabilities;
      state.multijoint = capabilities.multijoint;
      const multiReady = state.multijoint?.ready === true;
      byId("scope-settings").hidden = !multiReady;
      if (multiReady) {
        setOptions(goalJoint, [["", "7축 모두"], ...state.multijoint.joint_names.map((name) => [name, jointLabel(name)])]);
        const families = new Set([baselineFamily, ...(state.multijoint.goal_families || [])]);
        const firstSetup = goalFamily.options.length === 0;
        setOptions(goalFamily, [...families].filter((id) => Object.hasOwn(familyLabels, id)).map((id) => [id, familyLabels[id]]));
        if (firstSetup && families.has("swing_scaled")) goalFamily.value = "swing_scaled";
        const bioModes = state.multijoint.bio_modes || ["none"];
        setOptions(bioInput, bioModes.filter((mode) => Object.hasOwn(bioLabels, mode)).map((mode) => [mode, bioLabels[mode]]));
      } else scopeInput.value = "tibia";
      scopeSettings({ resetGains: true });
      const defaults = capabilities.defaults || {};
      if (finite(defaults.goal_scale)) {
        ui["goal-scale"].value = defaults.goal_scale;
        ui["goal-scale-number"].value = defaults.goal_scale;
      }
      if (finite(defaults.threshold_deg)) ui["threshold"].value = defaults.threshold_deg;
      if (finite(defaults.initial_tibia_delta_deg)) ui["initial-delta"].value = defaults.initial_tibia_delta_deg;
      const version = capabilities.schema_version?.startsWith("sensory_translator_v2_") ? " V2" : "";
      setConnection(state.ready ? "ready" : "error", state.ready ? `엔진${version} 준비됨` : "모델 준비 필요");
      ui["page-error"].hidden = state.ready;
      if (!state.ready) showPageError("모델을 준비해야 실행할 수 있어요", "서버에서 저장 모델과 물리 실행 환경을 확인한 뒤 다시 확인해 주세요.");
      await loadGoal();
      restorePendingJob();
    } catch (error) {
      state.ready = false;
      setConnection("error", "서버 연결 확인 필요");
      showPageError("컨트롤러에 연결하지 못했어요", error.message);
    } finally { ui["retry-connection"].disabled = false; updateControls(); }
  }

  let goalDebounce;
  function scheduleGoal(preserveCustom = false) {
    clearTimeout(goalDebounce);
    ++state.goalRequest;
    state.loadingGoal = true;
    state.goalReady = false;
    updateControls();
    goalDebounce = setTimeout(() => loadGoal({ preserveCustom }), 180);
  }

  async function importGoal(file) {
    if (!file || state.busy) return;
    try {
      if (file.size > 1024 * 1024) throw new Error("목표 JSON은 1MB 이하로 저장해 주세요.");
      let data;
      try { data = JSON.parse(await file.text()); }
      catch (_) { throw new Error("JSON 형식을 확인해 주세요. target_rad 또는 target_deg 배열이 필요합니다."); }
      if (!data || Array.isArray(data) || typeof data !== "object") throw new Error("각도 단위를 알 수 있도록 target_rad 또는 target_deg 필드가 있는 JSON을 사용해 주세요.");
      let radians = data.target_rad ?? data.target_tibia_rad;
      const toRad = (value) => value * Math.PI / 180;
      if (radians === undefined && isMulti() && finiteMatrix(data.target_deg, 211, 7)) radians = data.target_deg.map((row) => row.map(toRad));
      if (radians === undefined && finiteArray(data.target_deg, 211)) radians = data.target_deg.map(toRad);
      if (isMulti() && data.joint_names !== undefined && JSON.stringify(data.joint_names) !== JSON.stringify(state.multijoint.joint_names)) {
        throw new Error("목표 JSON의 7축 순서를 내보낸 예시와 맞춰 주세요.");
      }
      const goal = checkedTarget(radians, data.time_s);
      ++state.goalRequest;
      clearTimeout(goalDebounce);
      state.loadingGoal = false;
      setGoal(goal, `사용자 목표 · ${file.name}`, true);
      formError();
      announce("사용자 목표 211개 시점을 불러왔습니다.");
    } catch (error) { formError(error.message); }
    finally { ui["goal-file"].value = ""; }
  }

  function exportGoal() {
    if (!state.goal) return;
    const metadata = isMulti() ? { joint_names: state.multijoint.joint_names } : { joint: "LF tibia" };
    const blob = new Blob([JSON.stringify({ ...metadata, ...state.goal }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = isMulti() ? "lf_7axis_target.json" : "lf_tibia_target.json";
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; }
  function updateClock() {
    const seconds = Math.max(0, Math.floor((Date.now() - state.startedAt) / 1000));
    ui["elapsed-time"].textContent = `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  }
  function setBusy(busy) {
    state.busy = busy;
    clearInterval(state.clockTimer);
    if (busy) { updateClock(); state.clockTimer = setInterval(updateClock, 1000); }
    updateControls();
  }
  function savePendingJob() {
    try { sessionStorage.setItem(storageKey, JSON.stringify({ id: state.jobId, startedAt: state.startedAt, request: state.request, source: state.source, customGoal: state.customGoal, initialAngle: state.initialAngle, scale: ui["goal-scale-number"].value, family: selectedFamily() })); }
    catch (_) { /* Storage may be unavailable; active polling still works. */ }
  }
  function clearPendingJob() { try { sessionStorage.removeItem(storageKey); } catch (_) { /* No persistent job state. */ } }

  function setStage(stage, status = "running", detail = "") {
    const labels = ["predicting", "fast_verification", "correcting", "completed"];
    const aliases = { queued: -1, running: 0, fast_verified: 1, correction_failed: 2, final_verification: 3, rendering: 3 };
    const index = Object.hasOwn(aliases, stage) ? aliases[stage] : labels.indexOf(stage);
    ui["job-panel"].dataset.status = status;
    ui["job-title"].textContent = stageLabels[stage] || stageLabels[status] || "작업을 실행하고 있어요.";
    ui["job-detail"].textContent = detail || (status === "queued" ? "앞선 작업이 끝나면 순서대로 실행합니다." : "검증된 물리 결과를 기다리고 있습니다.");
    ui["job-steps"].querySelectorAll("li").forEach((item, i) => {
      item.classList.toggle("is-current", status !== "completed" && i === index);
      item.classList.toggle("is-done", status === "completed" || i < index);
      if (status !== "completed" && i === index) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
    if (state.stage !== stage) { announce(ui["job-title"].textContent); state.stage = stage; }
  }

  function progressStage(progress, fallback) {
    if (!progress || typeof progress !== "object") return fallback;
    for (const key of ["stage", "phase", "event"]) if (typeof progress[key] === "string") return progress[key];
    return fallback;
  }
  function progressDetail(progress) {
    if (!progress || typeof progress !== "object") return "";
    const parts = [];
    if (Number.isInteger(progress.iteration)) parts.push(`보정 평가 ${progress.iteration}${Number.isInteger(progress.steps) ? ` / ${progress.steps}` : ""}`);
    else if (Number.isInteger(progress.step) && progress.step > 0) parts.push(`보정 단계 ${progress.step}`);
    if (finite(progress.joint_rmse_deg)) parts.push(`현재 오차 ${progress.joint_rmse_deg.toFixed(3)}°`);
    if (finite(progress.best_joint_rmse_deg)) parts.push(`최선 ${progress.best_joint_rmse_deg.toFixed(3)}°`);
    if (parts.length) return parts.join(" · ");
    return typeof progress.message === "string" ? progress.message : "";
  }

  async function submitJob(event) {
    event.preventDefault();
    if (state.busy || !state.goal || !state.goalReady || !state.ready || state.loadingGoal) return;
    formError();
    const threshold = Number(ui["threshold"].value);
    const mode = selectedMode();
    const steps = ["fast", "closed_loop"].includes(mode) ? 0 : Number(ui["correction-steps"].value);
    const delta = Number(ui["initial-delta"].value);
    if (!Number.isFinite(threshold) || threshold <= 0) return formError("오차 기준에 0보다 큰 각도 값을 입력해 주세요.");
    if (!Number.isInteger(steps) || steps < 0 || steps > 20) return formError("보정 횟수는 0~20 사이의 정수로 입력해 주세요.");
    if (!Number.isFinite(delta) || delta < -5 || delta > 5) return formError("시작 각도 변화는 −5°~+5° 사이로 입력해 주세요.");
    const request = { target_rad: state.goal.target_rad, mode, threshold_deg: threshold, correction_steps: steps, render: ui["render-video"].checked };
    if (isMulti()) {
      request.scope = "lf15";
      if (selectedGoalJoint() && !state.customGoal) request.joint_names = [selectedGoalJoint()];
      request.initial_qpos_delta_deg = [0, 0, 0, 0, 0, 0, delta];
      request.bio_mode = bioInput.value || "none";
    } else request.initial_tibia_delta_deg = delta;
    if (mode === "closed_loop") {
      const gains = [feedbackKp, feedbackKd].map((input) => input.valueAsNumber);
      if (gains.some((value) => !finite(value) || value < 0)) return formError("Kp와 Kd에 0 이상인 숫자를 입력해 주세요.");
      [request.feedback_kp, request.feedback_kd] = gains;
    }
    state.request = request;
    state.startedAt = Date.now(); state.stage = null; state.jobId = null;
    ui["results"].hidden = true; ui["empty-results"].hidden = true;
    ui["result-video"].pause(); ui["retry-job"].hidden = true;
    ui["job-panel"].hidden = false; ui["job-id"].textContent = "";
    setStage("queued", "queued", "요청을 등록하고 있습니다.");
    setBusy(true);
    try {
      const job = await requestJSON("/api/rate/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.request) }, 20000);
      if (typeof job.job_id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(job.job_id)) throw new Error("서버가 유효한 작업 ID를 반환하지 않았습니다.");
      state.jobId = job.job_id;
      ui["job-id"].textContent = `작업 ${state.jobId}`;
      savePendingJob();
      setStage("queued", "queued");
      pollJob();
    } catch (error) {
      setBusy(false);
      setStage("failed", "failed", `실행 요청을 등록하지 못했습니다. ${error.message}`);
      formError(error.message);
    }
  }

  async function pollJob() {
    if (!state.jobId || state.pollInFlight) return;
    stopPolling(); state.pollInFlight = true; ui["retry-job"].hidden = true;
    try {
      const job = await requestJSON(`/api/rate/jobs/${encodeURIComponent(state.jobId)}`);
      if (job.status === "completed") {
        if (!job.result || typeof job.result !== "object") throw new Error("완료된 작업의 결과 데이터가 없습니다. 서버의 작업 기록을 확인해 주세요.");
        renderResult(job.result);
        setStage("completed", "completed", "선택된 입력의 물리 검증과 결과 저장을 마쳤습니다.");
        setBusy(false); clearPendingJob();
      } else if (job.status === "failed") {
        setStage("failed", "failed", messageOf(job.error));
        setBusy(false); clearPendingJob();
        ui["empty-results"].hidden = false;
        announce(`작업 실패. ${messageOf(job.error)}`);
      } else if (job.status === "queued" || job.status === "running") {
        const stage = progressStage(job.progress, job.status);
        const displayStage = stage === "nominal_prediction" ? "predicting" : stage === "evaluating" ? "fast_verification" : stage;
        setStage(displayStage, job.status, progressDetail(job.progress));
        state.pollTimer = setTimeout(pollJob, 1100);
      } else throw new Error("작업 상태를 해석하지 못했습니다. 작업 기록을 다시 조회해 주세요.");
    } catch (error) {
      if (error.httpStatus === 404) {
        setBusy(false); clearPendingJob();
        setStage("failed", "failed", "이 작업을 찾을 수 없습니다. 서버가 재시작되었다면 새로 실행해 주세요.");
      } else {
        ui["job-detail"].textContent = `상태 조회가 중단됐습니다. 작업은 서버에서 계속 실행될 수 있어요. ${error.message}`;
        ui["retry-job"].hidden = false;
        announce("작업 상태를 다시 조회해 주세요.");
      }
    } finally { state.pollInFlight = false; }
  }

  function restorePendingJob() {
    if (state.busy) return;
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey) || "null");
      if (!saved || typeof saved.id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(saved.id) || !saved.request) return;
      const request = saved.request;
      const multi = request.scope === "lf15";
      if (multi && state.multijoint?.ready !== true) return clearPendingJob();
      const goal = checkedTarget(request.target_rad, undefined, multi);
      if (!Object.hasOwn(modeLabels, request.mode)) return;
      state.jobId = saved.id; state.request = request; state.startedAt = finite(saved.startedAt) ? saved.startedAt : Date.now();
      scopeInput.value = multi ? "lf15" : "tibia";
      if (multi) {
        if ([...goalFamily.options].some((option) => option.value === saved.family)) goalFamily.value = saved.family;
        goalJoint.value = request.joint_names?.[0] || "";
        if (Object.hasOwn(bioLabels, request.bio_mode)) bioInput.value = request.bio_mode;
      }
      scopeSettings({ resetGains: true });
      const radio = document.querySelector(`input[name="mode"][value="${request.mode}"]`);
      if (radio && !radio.disabled) radio.checked = true;
      ui["threshold"].value = request.threshold_deg;
      if (["auto", "refine"].includes(request.mode)) ui["correction-steps"].value = request.correction_steps;
      if (finite(request.feedback_kp)) feedbackKp.value = request.feedback_kp;
      if (finite(request.feedback_kd)) feedbackKd.value = request.feedback_kd;
      ui["initial-delta"].value = multi ? request.initial_qpos_delta_deg?.[6] ?? 0 : request.initial_tibia_delta_deg;
      ui["render-video"].checked = request.render === true;
      if (Number.isFinite(Number(saved.scale))) { ui["goal-scale"].value = saved.scale; ui["goal-scale-number"].value = saved.scale; }
      if (finite(saved.initialAngle)) state.initialAngle = saved.initialAngle;
      setGoal(goal, saved.source || "복원한 실행 목표", saved.customGoal === true);
      ui["job-panel"].hidden = false; ui["empty-results"].hidden = true;
      ui["job-id"].textContent = `작업 ${state.jobId}`;
      setStage("running", "running", "이전에 실행한 작업의 상태를 불러옵니다.");
      setBusy(true); pollJob();
    } catch (_) { clearPendingJob(); }
  }

  function fileURL(path) {
    const value = typeof path === "string" ? path : path?.path;
    if (!value || value.startsWith("/") || value.includes("\\") || value.split("/").some((part) => part === ".." || part === "." || !part)) return null;
    return `/api/rate/jobs/${encodeURIComponent(state.jobId)}/files/${value.split("/").map(encodeURIComponent).join("/")}`;
  }
  function angleText(value) { return finite(value) ? `${value.toFixed(3)}°` : "—"; }

  function validatedTrace(trace, multi) {
    const jointSeries = (values) => multi ? finiteMatrix(values, 211, 7) : finiteArray(values, 211);
    if (!trace || !finiteArray(trace.time_ms, 211) || !jointSeries(trace.target_deg) || !jointSeries(trace.actual_deg)) throw new Error("결과의 관절 궤적 211개를 확인하지 못했습니다.");
    if (!finiteArray(trace.rate_time_ms, 420) || !finiteMatrix(trace.rates_hz, 420, multi ? 15 : 2)) throw new Error("결과의 발화율 파형을 확인하지 못했습니다.");
    return trace;
  }

  function renderResult(result) {
    const multi = result.schema_version === "sensory_translator_v2_multijoint_run_v1";
    const v2 = multi || result.schema_version === "sensory_translator_v2_run_v1";
    if (v2) {
      const updated = result.correction_steps_taken;
      const attempted = updated > 0 || result.history?.some((step) => step.used_jacobian) === true;
      const invalidGradient = result.history?.some((step) => step.gradient_valid === false) === true;
      result = { ...result,
        joint_rmse_deg: result.best_rmse_deg,
        criterion_met: result.accuracy_threshold_met,
        fast: { joint_rmse_deg: result.nominal_rmse_deg, max_abs_joint_error_deg: result.nominal_max_abs_joint_error_deg },
        selected_stage: (multi ? updated : result.selected_step) > 0 ? "correction" : "fast",
        correction: { status: invalidGradient ? "stopped" : attempted ? "completed" : "skipped", actual_optimizer_updates: updated },
        artifacts: { ...result.artifacts, video: result.artifacts?.video_mp4, video_poster: result.artifacts?.preview_png, protocol_json: multi ? result.artifacts?.manifest : result.artifacts?.motor_rate_protocol },
      };
    }
    const feedback = v2 && result.mode === "closed_loop";
    state.trace = validatedTrace(result.trace, multi); state.result = result; state.cursor = 0;
    state.multiResult = multi;
    byId("multijoint-result-settings").hidden = !multi;
    byId("rate-legend").hidden = multi;
    if (multi) {
      const joints = result.scope.controlled_joints;
      const muscles = result.scope.actuator_names;
      setOptions(resultJoint, joints.map((name, index) => [String(index), jointLabel(name)]));
      setOptions(resultMuscle, muscles.map((name, index) => [String(index), name]));
      resultJoint.value = String(Math.max(0, joints.indexOf(result.selected_joint_names?.[0] ?? joints[6])));
      resultMuscle.value = String(Math.max(0, muscles.indexOf("LFTibia_flex_93434")));
    }
    ui["results"].hidden = false;
    ui["results-title"].textContent = "검증 결과";
    const scope = multi ? "왼쪽 앞다리 7축 · 근육 15개" : "왼쪽 앞다리 경절";
    ui["result-context"].textContent = [modeLabels[state.request?.mode] || "움직임 생성", scope, result.model_id, `작업 ${state.jobId.slice(0, 12)}`].filter(Boolean).join(" · ");
    byId("fast-metric").hidden = feedback && !multi;
    byId("result-explanation").textContent = multi ? "전체 RMSE는 211개 시점 × 7축의 관절각 오차입니다." : "RMSE는 211개 시점의 관절각 오차입니다.";
    byId("rate-chart-title").textContent = feedback ? "모델의 발화율 명령" : "생성한 연속 발화율";
    byId("rate-chart-detail").hidden = !feedback;
    byId("rate-chart-detail").textContent = feedback ? "관절 피드백은 근육 입력에 적용돼요." : "";
    ui["fast-rmse"].textContent = angleText(result.fast?.joint_rmse_deg);
    ui["selected-rmse"].textContent = angleText(result.joint_rmse_deg);
    const maximum = multi ? Math.max(...state.trace.target_deg.flatMap((row, i) => row.map((value, j) => Math.abs(value - state.trace.actual_deg[i][j])))) : result.max_abs_joint_error_deg;
    ui["max-error"].textContent = angleText(maximum);
    const fastMax = feedback ? null : result.fast?.max_abs_joint_error_deg;
    const maxChange = result.max_abs_joint_error_deg - fastMax;
    ui["max-error-caption"].classList.toggle("is-worse", finite(maxChange) && maxChange > 1e-6);
    ui["max-error-caption"].textContent = finite(fastMax) && finite(result.max_abs_joint_error_deg)
      ? `빠른 생성 ${fastMax.toFixed(3)}° · ${Math.abs(maxChange) < 1e-6 ? "변화 없음" : `${Math.abs(maxChange).toFixed(3)}° ${maxChange > 0 ? "증가" : "감소"}`}`
      : "가장 많이 벗어난 시점";
    ui["max-error"].title = finite(fastMax) ? `빠른 생성 ${angleText(fastMax)} → 선택 결과 ${angleText(result.max_abs_joint_error_deg)}` : "선택된 입력의 최대 절대오차";
    ui["criterion-badge"].textContent = result.criterion_met === true ? "오차 기준 충족" : "오차 기준 미달";
    ui["criterion-badge"].setAttribute("aria-label", ui["criterion-badge"].textContent);
    ui["criterion-badge"].dataset.met = String(result.criterion_met === true);
    const threshold = state.request?.threshold_deg;
    ui["threshold-caption"].textContent = finite(threshold) ? `요청 기준 ≤ ${threshold.toFixed(3)}°` : "낮을수록 목표에 가깝습니다.";
    ui["selected-stage"].textContent = feedback ? "관절 피드백 적용" : result.selected_stage === "correction" ? "보정된 입력" : "빠른 생성 입력";
    const correction = result.correction || {};
    const updates = correction.actual_optimizer_updates;
    ui["correction-caption"].textContent = feedback ? `Kp ${result.feedback?.kp ?? state.request.feedback_kp} · Kd ${result.feedback?.kd ?? state.request.feedback_kd}` : correction.status === "failed" ? "보정 실패 · 빠른 결과 유지" : correction.status === "stopped" ? `기울기 검증에서 중단 · 갱신 ${updates}회` : correction.status === "skipped" ? "추가 보정 없음" : Number.isInteger(updates) ? `실제 보정 갱신 ${updates}회` : "물리 오차로 입력 선택";
    ui["correction-warning"].hidden = correction.status !== "failed";
    ui["correction-warning-text"].textContent = `추가 보정을 완료하지 못해, 물리 검증을 마친 빠른 생성 결과를 유지했습니다.${Number.isInteger(updates) ? ` 실제 보정 갱신은 ${updates}회입니다.` : ""}`;
    ui["correction-error"].textContent = messageOf(correction.error);
    ui["result-notice"].hidden = true;
    const bioResult = byId("bio-result");
    const bio = result.bio_interface;
    bioResult.hidden = !finite(bio?.overall_rmse_deg);
    bioResult.textContent = bioResult.hidden ? "" : `${bioLabels[state.request.bio_mode] || "스파이크"} 별도 재생 · 7축 RMSE ${angleText(bio.overall_rmse_deg)}`;
    // VP9/WebM avoids black H264 frames on some embedded Linux browsers.
    // Older results keep their original MP4 and are not rewritten.
    const videoURL = fileURL(result.artifacts?.video_webm || result.artifacts?.video);
    const posterURL = fileURL(result.artifacts?.video_poster);
    ui["result-video"].pause(); ui["result-video"].hidden = !videoURL;
    ui["video-placeholder"].hidden = !!videoURL;
    if (videoURL) {
      if (posterURL) ui["result-video"].poster = posterURL;
      else ui["result-video"].removeAttribute("poster");
      ui["result-video"].src = videoURL;
      ui["result-video"].load();
    } else {
      ui["result-video"].removeAttribute("poster");
      ui["result-video"].removeAttribute("src"); ui["result-video"].load();
      ui["video-placeholder"].textContent = state.request?.render ? "이 결과에 탑뷰 영상이 제공되지 않았습니다. 아래 파일과 수치 결과를 확인해 주세요." : "영상 생성을 선택하지 않았습니다. 그래프에서 시점을 이동할 수 있어요.";
    }
    const protocolURL = fileURL(result.artifacts?.protocol_json);
    ui["protocol-link"].hidden = !protocolURL;
    if (protocolURL) {
      ui["protocol-link"].href = protocolURL;
      ui["protocol-link"].firstElementChild.textContent = multi ? "실행 기록 JSON" : "프로토콜 JSON";
    }
    ui["bundle-download"].href = `/api/rate/jobs/${encodeURIComponent(state.jobId)}/download`;
    ui["bundle-download"].hidden = false;
    ui["time-cursor"].value = "0";
    drawResults(); updateCursorReadout();
    announce(`검증 완료. 선택 결과 오차 ${angleText(result.joint_rmse_deg)}. ${ui["criterion-badge"].textContent}.`);
    requestAnimationFrame(() => {
      drawResults();
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) ui["results"].scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function nearestIndex(xs, value) {
    let index = 0;
    for (let i = 1; i < xs.length; i += 1) if (Math.abs(xs[i] - value) < Math.abs(xs[index] - value)) index = i;
    return index;
  }
  function updateCursorReadout() {
    if (!state.trace) return;
    const trace = state.trace; const index = nearestIndex(trace.time_ms, state.cursor);
    const rateIndex = nearestIndex(trace.rate_time_ms, state.cursor);
    ui["time-output"].textContent = `${Math.round(state.cursor)}ms`;
    const joint = Number(resultJoint.value);
    const target = state.multiResult ? trace.target_deg[index][joint] : trace.target_deg[index];
    const actual = state.multiResult ? trace.actual_deg[index][joint] : trace.actual_deg[index];
    const rates = trace.rates_hz[rateIndex];
    const rateText = state.multiResult ? `선택 근육 ${rates[Number(resultMuscle.value)].toFixed(1)}Hz` : `굴근 ${rates[0].toFixed(1)}Hz · 신근 ${rates[1].toFixed(1)}Hz`;
    ui["cursor-readout"].textContent = `목표 ${target.toFixed(2)}° · 실제 ${actual.toFixed(2)}° · ${rateText}`;
  }
  function setCursor(value, fromVideo = false) {
    state.cursor = clamp(value, 0, 420);
    ui["time-cursor"].value = String(state.cursor);
    if (!fromVideo && finite(ui["result-video"].duration) && ui["result-video"].duration > 0) {
      ui["result-video"].pause();
      ui["result-video"].currentTime = ui["result-video"].duration * state.cursor / 420;
    }
    updateCursorReadout(); drawResults();
  }

  function drawPlot(canvas, series, { cursor = null, unit = "°", fixedY = null } = {}) {
    const box = canvas.getBoundingClientRect();
    if (box.width <= 0 || box.height <= 0) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = box.width, height = box.height;
    canvas.width = Math.round(width * pixelRatio); canvas.height = Math.round(height * pixelRatio);
    const context = canvas.getContext("2d"); context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const values = series.flatMap((line) => line.y).filter(Number.isFinite);
    if (!values.length) return;
    let low = fixedY ? fixedY[0] : Math.min(...values), high = fixedY ? fixedY[1] : Math.max(...values);
    if (!Number.isFinite(high - low)) return;
    if (high === low) { low -= 1; high += 1; }
    else if (!fixedY) { const margin = (high - low) * 0.13; low -= margin; high += margin; }
    const left = 49, right = 23, top = 19, bottom = 31;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const x = (value) => left + value / 420 * plotWidth;
    const y = (value) => top + (high - value) / (high - low) * plotHeight;
    context.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    context.lineWidth = 1; context.fillStyle = getComputedStyle(canvas).color;
    for (let i = 0; i <= 4; i += 1) {
      const yy = top + i / 4 * plotHeight;
      context.strokeStyle = getComputedStyle(canvas).borderTopColor; context.beginPath(); context.moveTo(left, yy); context.lineTo(width - right, yy); context.stroke();
      const value = high - i / 4 * (high - low);
      context.textAlign = "right"; context.fillText(Math.abs(value) >= 1000 ? value.toExponential(1) : value.toFixed(high - low < 5 ? 1 : 0), left - 8, yy + 3);
      const tick = i * 105;
      context.textAlign = i === 4 ? "right" : "center"; context.fillText(`${tick}${i === 4 ? "ms" : ""}`, x(tick), height - 10);
    }
    context.textAlign = "left"; context.fillText(unit, 11, 12);
    context.save(); context.beginPath(); context.rect(left - 1, top - 1, plotWidth + 2, plotHeight + 2); context.clip();
    for (const line of series) {
      context.strokeStyle = line.color; context.lineWidth = line.width || 1.9; context.setLineDash(line.dash || []); context.beginPath();
      line.y.forEach((value, i) => { if (i === 0) context.moveTo(x(line.x[i]), y(value)); else context.lineTo(x(line.x[i]), y(value)); }); context.stroke();
    }
    context.setLineDash([]);
    if (cursor !== null) {
      context.strokeStyle = getComputedStyle(canvas).outlineColor; context.lineWidth = 1; context.setLineDash([3, 3]);
      context.beginPath(); context.moveTo(x(cursor), top); context.lineTo(x(cursor), height - bottom); context.stroke(); context.setLineDash([]);
      for (const line of series) {
        const index = nearestIndex(line.x, cursor);
        context.fillStyle = line.color; context.beginPath(); context.arc(x(line.x[index]), y(line.y[index]), 3.2, 0, 2 * Math.PI); context.fill();
      }
    }
    context.restore();
  }
  function chartColor(series) {
    const legend = document.querySelector(`.legend-${series}`);
    return getComputedStyle(legend || ui["target-preview"]).color;
  }
  function drawPreview() {
    if (!state.goal) return;
    const values = state.goal.target_rad;
    const joint = isMulti() ? Math.max(0, state.multijoint.joint_names.indexOf(selectedGoalJoint() || state.multijoint.joint_names[6])) : null;
    const angles = joint === null ? values : values.map((row) => row[joint]);
    drawPlot(ui["target-preview"], [{ x: state.goal.time_s.map((t) => t * 1000), y: angles.map(radiansToDegrees), color: chartColor("target"), width: 2.2 }]);
  }
  function drawResults() {
    if (!state.trace || ui["results"].hidden) return;
    const trace = state.trace;
    const joint = Number(resultJoint.value);
    const angles = (values) => state.multiResult ? values.map((row) => row[joint]) : values;
    drawPlot(ui["angle-chart"], [{ x: trace.time_ms, y: angles(trace.target_deg), color: chartColor("target"), dash: [5, 4] }, { x: trace.time_ms, y: angles(trace.actual_deg), color: chartColor("actual"), width: 2.1 }], { cursor: state.cursor });
    const channels = state.multiResult ? [Number(resultMuscle.value)] : [0, 1];
    const rateLines = channels.map((channel, index) => ({ x: trace.rate_time_ms, y: trace.rates_hz.map((row) => row[channel]), color: chartColor(index === 0 ? "flexor" : "extensor") }));
    const largestRate = Math.max(200, ...rateLines.flatMap((line) => line.y));
    drawPlot(ui["rate-chart"], rateLines, { cursor: state.cursor, unit: "Hz", fixedY: [0, largestRate] });
    if (state.multiResult) {
      const name = state.result.scope.controlled_joints[joint];
      byId("result-joint-error").textContent = `${jointLabel(name)} RMSE ${angleText(state.result.per_joint_rmse_deg[name])}`;
    }
  }

  ui["control-form"].addEventListener("submit", submitJob);
  ui["goal-scale"].addEventListener("input", () => { ui["goal-scale-number"].value = ui["goal-scale"].value; state.customGoal = false; scheduleGoal(); updateControls(); });
  ui["goal-scale-number"].addEventListener("input", () => { const value = Number(ui["goal-scale-number"].value); if (Number.isFinite(value) && value >= 0.8 && value <= 1.2) { ui["goal-scale"].value = value; state.customGoal = false; scheduleGoal(); updateControls(); } });
  ui["initial-delta"].addEventListener("input", () => scheduleGoal(true));
  scopeInput.addEventListener("change", () => {
    scopeSettings({ resetGains: true });
    state.goal = null;
    state.customGoal = false;
    ui["results"].hidden = true;
    ui["job-panel"].hidden = true;
    ui["empty-results"].hidden = false;
    ui["result-video"].pause();
    ui["preview-placeholder"].hidden = false;
    ui["preview-placeholder"].textContent = "목표 궤적을 불러오는 중입니다.";
    scheduleGoal();
  });
  goalFamily.addEventListener("change", () => { scopeSettings(); state.customGoal = false; scheduleGoal(); });
  goalJoint.addEventListener("change", () => { state.customGoal = false; scheduleGoal(); });
  [resultJoint, resultMuscle].forEach((select) => select.addEventListener("change", () => { drawResults(); updateCursorReadout(); }));
  document.querySelectorAll("[data-scale]").forEach((button) => button.addEventListener("click", () => { ui["goal-scale"].value = button.dataset.scale; ui["goal-scale-number"].value = button.dataset.scale; state.customGoal = false; clearTimeout(goalDebounce); loadGoal(); }));
  document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", updateControls));
  ui["import-target"].addEventListener("click", () => ui["goal-file"].click());
  ui["goal-file"].addEventListener("change", () => importGoal(ui["goal-file"].files[0]));
  ui["export-target"].addEventListener("click", exportGoal);
  ui["retry-connection"].addEventListener("click", bootstrap);
  ui["retry-job"].addEventListener("click", pollJob);
  ui["time-cursor"].addEventListener("input", () => setCursor(Number(ui["time-cursor"].value)));
  ui["result-video"].addEventListener("timeupdate", () => {
    const video = ui["result-video"];
    if (finite(video.duration) && video.duration > 0) setCursor(video.currentTime / video.duration * 420, true);
  });
  ui["result-video"].addEventListener("error", () => {
    if (!ui["result-video"].getAttribute("src")) return;
    ui["result-notice"].textContent = "영상 파일을 재생하지 못했습니다. 결과 묶음을 내려받아 확인하거나 새 작업에서 영상을 다시 생성해 주세요.";
    ui["result-notice"].hidden = false;
  });
  let resizeFrame;
  const redraw = () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => { drawPreview(); drawResults(); }); };
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(redraw);
    [ui["target-preview"], ui["angle-chart"], ui["rate-chart"]].forEach((canvas) => observer.observe(canvas.parentElement));
  } else window.addEventListener("resize", redraw);
  updateControls(); bootstrap();
}
