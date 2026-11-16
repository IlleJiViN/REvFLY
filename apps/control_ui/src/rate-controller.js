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
  const bioLabels = { none: "Continuous rates only", deterministic_isi: "Fixed spike intervals", poisson_refractory: "Poisson spikes with a refractory period", direct_lif: "LIF motor neurons" };
  const baselineFamily = "kinematic_teacher_scaled";
  const familyLabels = { [baselineFamily]: "Scale reference trajectory", swing_scaled: "Leg swing", decoupled: "Scale individual axes", rhythmic_double_tap: "Double tap", reach_and_hold: "Reach and hold" };
  const jointLabels = {
    joint_LFCoxa_yaw: "Coxa yaw", joint_LFCoxa_pitch: "Coxa pitch", joint_LFCoxa_roll: "Coxa roll",
    joint_LFTrochanter_yaw: "Trochanter yaw", joint_LFTrochanter_pitch: "Trochanter pitch", joint_LFTrochanter_roll: "Trochanter roll",
    joint_LFTibia_pitch: "Tibia pitch",
  };
  const modeLabels = { fast: "Fast prediction", auto: "Auto refine", refine: "Refine input", closed_loop: "Joint feedback" };
  const stageLabels = { queued: "Waiting to run.", running: "Running job.", predicting: "Predicting firing rates.", fast_verification: "Verifying predicted motion in simulation.", fast_verified: "Fast prediction verified.", correcting: "Refining input waveforms.", correction_failed: "Using the fast prediction.", final_verification: "Verifying the selected result.", rendering: "Rendering top-view video.", completed: "Results are ready.", failed: "Job failed." };

  function messageOf(value) {
    if (typeof value === "string") return value;
    if (value && typeof value.message === "string") return value.message;
    if (value && typeof value.error === "string") return value.error;
    if (value !== undefined && value !== null) return JSON.stringify(value);
    return "No error details were provided.";
  }

  async function requestJSON(path, options = {}, timeout = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { cache: "no-store", ...options, signal: controller.signal });
      let data;
      try { data = await response.json(); }
      catch (_) { throw new Error(`Could not read the response. Check your connection and make sure the server is running. (HTTP ${response.status})`); }
      if (!response.ok) {
        const error = new Error(messageOf(data.error ?? data.message ?? data));
        error.httpStatus = response.status;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The server took too long to respond. Try again shortly.");
      if (error instanceof TypeError) throw new Error("Could not reach the server. Check your connection and make sure the server is running.");
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
    byId("scope-description").textContent = multi ? "Left front leg · 7 axes · 15 muscles · 0 teacher-driven channels" : "Left front tibia · 1 axis · 2 muscles · 13 teacher-driven channels";
    byId("preview-scope").textContent = multi ? "TOP / LEFT FRONT LEG" : "TOP / LF TIBIA";
    byId("preview-pools").textContent = multi ? "15 muscles" : "2 muscles";
    const identity = byId("engine-model");
    const model = multi ? capabilities.model_id : capabilities?.model?.model_id;
    identity.hidden = !model;
    identity.textContent = [model, state.capabilities?.backend?.name].filter(Boolean).join(" · ");
  }
  function updateControls() {
    ui["control-fields"].disabled = state.busy;
    ui["run-button"].disabled = state.busy || !state.ready || state.loadingGoal || !state.goalReady || !state.goal;
    (byId("run-button-label") || ui["run-button"].lastElementChild).textContent = state.busy ? "Running and verifying…" : "Run and verify";
    const feedback = selectedMode() === "closed_loop";
    const optimization = ["auto", "refine"].includes(selectedMode());
    ui["correction-steps"].disabled = state.busy || !optimization;
    ui["correction-help"].textContent = feedback ? "Adjusts joint motion every 2 ms in simulation." : optimization ? "Run 0–20 steps and select the result by simulation error." : "Fast prediction skips refinement.";
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
    if (!valid) throw new Error(multi ? "Enter 211 target samples with seven joint angles per sample." : "Enter 211 finite target angles.");
    if (!target.flat().map(radiansToDegrees).every(Number.isFinite)) throw new Error("Check the magnitude and units of the target angles.");
    const grid = Array.from({ length: 211 }, (_, i) => i * 0.002);
    if (times !== undefined && (!finiteArray(times, 211) || times.some((t, i) => Math.abs(t - grid[i]) > 1e-9))) {
      throw new Error("time_s must contain 211 values from 0 to 0.420 seconds in 0.002-second steps.");
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
    ui["initial-angle-info"].textContent = finite(state.initialAngle) ? `Initial simulation angle ${state.initialAngle.toFixed(2)}°` : "";
  }

  async function loadGoal({ preserveCustom = false } = {}) {
    const scale = Number(ui["goal-scale-number"].value);
    const delta = Number(ui["initial-delta"].value);
    if (!Number.isFinite(scale) || scale < 0.8 || scale > 1.2 || !Number.isFinite(delta) || delta < -5 || delta > 5) {
      state.loadingGoal = false; state.goalReady = false;
      formError("Enter a motion scale from 0.8 to 1.2 and an initial angle offset from −5° to +5°.");
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
        const preview = multi ? ` · ${jointLabel(selectedGoalJoint() || state.multijoint.joint_names[6])} preview` : "";
        const name = multi ? familyLabels[selectedFamily()] : "Default motion";
        setGoal(goal, `${name} ${scale.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}×${preview} · 211 samples`, false);
      }
      formError();
    } catch (error) {
      if (version !== state.goalRequest) return;
      formError(`Could not load the target. ${error.message} Select the preset again once the connection is restored.`);
      if (!state.goal) ui["preview-placeholder"].textContent = "Check the connection and reload the target.";
    } finally {
      if (version === state.goalRequest) { state.loadingGoal = false; updateControls(); }
    }
  }

  async function bootstrap() {
    ui["retry-connection"].disabled = true;
    setConnection("loading", "Checking connection…");
    try {
      const capabilities = await requestJSON("/api/rate/capabilities");
      state.ready = capabilities.ready === true;
      state.capabilities = capabilities;
      state.multijoint = capabilities.multijoint;
      const multiReady = state.multijoint?.ready === true;
      byId("scope-settings").hidden = !multiReady;
      if (multiReady) {
        setOptions(goalJoint, [["", "All 7 axes"], ...state.multijoint.joint_names.map((name) => [name, jointLabel(name)])]);
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
      setConnection(state.ready ? "ready" : "error", state.ready ? `Engine${version} ready` : "Model setup required");
      ui["page-error"].hidden = state.ready;
      if (!state.ready) showPageError("Set up the model to run jobs", "Check the model files and simulation setup on the server, then reconnect.");
      await loadGoal();
      restorePendingJob();
    } catch (error) {
      state.ready = false;
      setConnection("error", "Connection unavailable");
      showPageError("Could not connect to the controller", error.message);
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
      if (file.size > 1024 * 1024) throw new Error("The target JSON must be 1 MB or smaller.");
      let data;
      try { data = JSON.parse(await file.text()); }
      catch (_) { throw new Error("Check that the JSON contains a target_rad or target_deg array."); }
      if (!data || Array.isArray(data) || typeof data !== "object") throw new Error("Use a JSON object with target_rad or target_deg to specify the angle units.");
      let radians = data.target_rad ?? data.target_tibia_rad;
      const toRad = (value) => value * Math.PI / 180;
      if (radians === undefined && isMulti() && finiteMatrix(data.target_deg, 211, 7)) radians = data.target_deg.map((row) => row.map(toRad));
      if (radians === undefined && finiteArray(data.target_deg, 211)) radians = data.target_deg.map(toRad);
      if (isMulti() && data.joint_names !== undefined && JSON.stringify(data.joint_names) !== JSON.stringify(state.multijoint.joint_names)) {
        throw new Error("Match the seven joint axes in the exported example.");
      }
      const goal = checkedTarget(radians, data.time_s);
      ++state.goalRequest;
      clearTimeout(goalDebounce);
      state.loadingGoal = false;
      setGoal(goal, `Custom target · ${file.name}`, true);
      formError();
      announce("Loaded 211 custom target samples.");
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
    ui["job-title"].textContent = stageLabels[stage] || stageLabels[status] || "Running job.";
    ui["job-detail"].textContent = detail || (status === "queued" ? "This job will run after the jobs ahead of it." : "Waiting for simulation results.");
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
    if (Number.isInteger(progress.iteration)) parts.push(`Refinement evaluation ${progress.iteration}${Number.isInteger(progress.steps) ? ` / ${progress.steps}` : ""}`);
    else if (Number.isInteger(progress.step) && progress.step > 0) parts.push(`Refinement step ${progress.step}`);
    if (finite(progress.joint_rmse_deg)) parts.push(`Current error ${progress.joint_rmse_deg.toFixed(3)}°`);
    if (finite(progress.best_joint_rmse_deg)) parts.push(`Best ${progress.best_joint_rmse_deg.toFixed(3)}°`);
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
    if (!Number.isFinite(threshold) || threshold <= 0) return formError("Enter a joint error threshold greater than 0°.");
    if (!Number.isInteger(steps) || steps < 0 || steps > 20) return formError("Enter a whole number from 0 to 20 for refinement steps.");
    if (!Number.isFinite(delta) || delta < -5 || delta > 5) return formError("Enter an initial angle offset from −5° to +5°.");
    const request = { target_rad: state.goal.target_rad, mode, threshold_deg: threshold, correction_steps: steps, render: ui["render-video"].checked };
    if (isMulti()) {
      request.scope = "lf15";
      if (selectedGoalJoint() && !state.customGoal) request.joint_names = [selectedGoalJoint()];
      request.initial_qpos_delta_deg = [0, 0, 0, 0, 0, 0, delta];
      request.bio_mode = bioInput.value || "none";
    } else request.initial_tibia_delta_deg = delta;
    if (mode === "closed_loop") {
      const gains = [feedbackKp, feedbackKd].map((input) => input.valueAsNumber);
      if (gains.some((value) => !finite(value) || value < 0)) return formError("Enter values of 0 or greater for Kp and Kd.");
      [request.feedback_kp, request.feedback_kd] = gains;
    }
    state.request = request;
    state.startedAt = Date.now(); state.stage = null; state.jobId = null;
    ui["results"].hidden = true; ui["empty-results"].hidden = true;
    ui["result-video"].pause(); ui["retry-job"].hidden = true;
    ui["job-panel"].hidden = false; ui["job-id"].textContent = "";
    setStage("queued", "queued", "Submitting job.");
    setBusy(true);
    try {
      const job = await requestJSON("/api/rate/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.request) }, 20000);
      if (typeof job.job_id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(job.job_id)) throw new Error("The server returned an invalid job ID.");
      state.jobId = job.job_id;
      ui["job-id"].textContent = `Job ${state.jobId}`;
      savePendingJob();
      setStage("queued", "queued");
      pollJob();
    } catch (error) {
      setBusy(false);
      setStage("failed", "failed", `Could not submit the job. ${error.message}`);
      formError(error.message);
    }
  }

  async function pollJob() {
    if (!state.jobId || state.pollInFlight) return;
    stopPolling(); state.pollInFlight = true; ui["retry-job"].hidden = true;
    try {
      const job = await requestJSON(`/api/rate/jobs/${encodeURIComponent(state.jobId)}`);
      if (job.status === "completed") {
        if (!job.result || typeof job.result !== "object") throw new Error("The completed job has no result data. Check the job history on the server.");
        renderResult(job.result);
        setStage("completed", "completed", "The selected input passed simulation verification and the results were saved.");
        setBusy(false); clearPendingJob();
      } else if (job.status === "failed") {
        setStage("failed", "failed", messageOf(job.error));
        setBusy(false); clearPendingJob();
        ui["empty-results"].hidden = false;
        announce(`Job failed. ${messageOf(job.error)}`);
      } else if (job.status === "queued" || job.status === "running") {
        const stage = progressStage(job.progress, job.status);
        const displayStage = stage === "nominal_prediction" ? "predicting" : stage === "evaluating" ? "fast_verification" : stage;
        setStage(displayStage, job.status, progressDetail(job.progress));
        state.pollTimer = setTimeout(pollJob, 1100);
      } else throw new Error("Could not read the job status. Check the job history again.");
    } catch (error) {
      if (error.httpStatus === 404) {
        setBusy(false); clearPendingJob();
        setStage("failed", "failed", "Could not find this job. Start a new run if the server restarted.");
      } else {
        ui["job-detail"].textContent = `Status checks stopped. The job may still be running on the server. ${error.message}`;
        ui["retry-job"].hidden = false;
        announce("Check the job status again.");
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
      setGoal(goal, saved.source || "Restored target", saved.customGoal === true);
      ui["job-panel"].hidden = false; ui["empty-results"].hidden = true;
      ui["job-id"].textContent = `Job ${state.jobId}`;
      setStage("running", "running", "Restoring the previous job status.");
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
    if (!trace || !finiteArray(trace.time_ms, 211) || !jointSeries(trace.target_deg) || !jointSeries(trace.actual_deg)) throw new Error("The joint trajectories must contain 211 valid samples.");
    if (!finiteArray(trace.rate_time_ms, 420) || !finiteMatrix(trace.rates_hz, 420, multi ? 15 : 2)) throw new Error("The result contains invalid firing-rate waveforms.");
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
    ui["results-title"].textContent = "Verification results";
    const scope = multi ? "Left front leg · 7 axes · 15 muscles" : "Left front tibia";
    ui["result-context"].textContent = [modeLabels[state.request?.mode] || "Motion generation", scope, result.model_id, `Job ${state.jobId.slice(0, 12)}`].filter(Boolean).join(" · ");
    byId("fast-metric").hidden = feedback && !multi;
    byId("result-explanation").textContent = multi ? "Overall RMSE measures joint angle error across 211 samples and 7 axes." : "RMSE measures joint angle error over 211 samples.";
    byId("rate-chart-title").textContent = feedback ? "Model firing-rate commands" : "Predicted firing rates";
    byId("rate-chart-detail").hidden = !feedback;
    byId("rate-chart-detail").textContent = feedback ? "Joint feedback acts on the muscle inputs." : "";
    ui["fast-rmse"].textContent = angleText(result.fast?.joint_rmse_deg);
    ui["selected-rmse"].textContent = angleText(result.joint_rmse_deg);
    const maximum = multi ? Math.max(...state.trace.target_deg.flatMap((row, i) => row.map((value, j) => Math.abs(value - state.trace.actual_deg[i][j])))) : result.max_abs_joint_error_deg;
    ui["max-error"].textContent = angleText(maximum);
    const fastMax = feedback ? null : result.fast?.max_abs_joint_error_deg;
    const maxChange = result.max_abs_joint_error_deg - fastMax;
    ui["max-error-caption"].classList.toggle("is-worse", finite(maxChange) && maxChange > 1e-6);
    ui["max-error-caption"].textContent = finite(fastMax) && finite(result.max_abs_joint_error_deg)
      ? `Fast prediction ${fastMax.toFixed(3)}° · ${Math.abs(maxChange) < 1e-6 ? "No change" : `${Math.abs(maxChange).toFixed(3)}° ${maxChange > 0 ? "increase" : "decrease"}`}`
      : "Largest deviation from the target";
    ui["max-error"].title = finite(fastMax) ? `Fast prediction ${angleText(fastMax)} → Selected result ${angleText(result.max_abs_joint_error_deg)}` : "Max absolute error for the selected input";
    ui["criterion-badge"].textContent = result.criterion_met === true ? "Within threshold" : "Above threshold";
    ui["criterion-badge"].setAttribute("aria-label", ui["criterion-badge"].textContent);
    ui["criterion-badge"].dataset.met = String(result.criterion_met === true);
    const threshold = state.request?.threshold_deg;
    ui["threshold-caption"].textContent = finite(threshold) ? `Target threshold ≤ ${threshold.toFixed(3)}°` : "Lower is closer to the target.";
    ui["selected-stage"].textContent = feedback ? "Joint feedback applied" : result.selected_stage === "correction" ? "Refined input" : "Fast prediction input";
    const correction = result.correction || {};
    const updates = correction.actual_optimizer_updates;
    ui["correction-caption"].textContent = feedback ? `Kp ${result.feedback?.kp ?? state.request.feedback_kp} · Kd ${result.feedback?.kd ?? state.request.feedback_kd}` : correction.status === "failed" ? "Refinement failed · using fast prediction" : correction.status === "stopped" ? `Stopped at gradient verification · ${updates} updates` : correction.status === "skipped" ? "No refinement" : Number.isInteger(updates) ? `Refinement applied ${updates} updates` : "Input selected by simulation error";
    ui["correction-warning"].hidden = correction.status !== "failed";
    ui["correction-warning-text"].textContent = `Refinement failed. The verified fast prediction was kept.${Number.isInteger(updates) ? ` Refinement applied ${updates} updates.` : ""}`;
    ui["correction-error"].textContent = messageOf(correction.error);
    ui["result-notice"].hidden = true;
    const bioResult = byId("bio-result");
    const bio = result.bio_interface;
    bioResult.hidden = !finite(bio?.overall_rmse_deg);
    bioResult.textContent = bioResult.hidden ? "" : `${bioLabels[state.request.bio_mode] || "Spike"} replay · RMSE across 7 axes ${angleText(bio.overall_rmse_deg)}`;
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
      ui["video-placeholder"].textContent = state.request?.render ? "A top-view video is unavailable. See the result files and metrics below." : "Video rendering was off. Use the timeline to inspect the charts.";
    }
    const protocolURL = fileURL(result.artifacts?.protocol_json);
    ui["protocol-link"].hidden = !protocolURL;
    if (protocolURL) {
      ui["protocol-link"].href = protocolURL;
      ui["protocol-link"].firstElementChild.textContent = multi ? "Run record JSON" : "Protocol JSON";
    }
    ui["bundle-download"].href = `/api/rate/jobs/${encodeURIComponent(state.jobId)}/download`;
    ui["bundle-download"].hidden = false;
    ui["time-cursor"].value = "0";
    drawResults(); updateCursorReadout();
    announce(`Verification complete. Selected result error ${angleText(result.joint_rmse_deg)}. ${ui["criterion-badge"].textContent}.`);
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
    const rateText = state.multiResult ? `Selected muscle ${rates[Number(resultMuscle.value)].toFixed(1)}Hz` : `Flexor ${rates[0].toFixed(1)}Hz · Extensor ${rates[1].toFixed(1)}Hz`;
    ui["cursor-readout"].textContent = `Target ${target.toFixed(2)}° · Actual ${actual.toFixed(2)}° · ${rateText}`;
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
    ui["preview-placeholder"].textContent = "Loading target trajectory…";
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
    ui["result-notice"].textContent = "Could not play the video. Download the results or render a new video.";
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
