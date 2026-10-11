import {
  OWNER,
  chooseEnemyMove,
  createInitialState,
  getCounts,
  getFleetPosition,
  getPlanet,
  launchFleet,
  updateState
} from "./game-core.js";
import { DqnAgent } from "./dqn-agent.js";
import { actionIntent, chooseDemoDecision } from "./demo-policy.js";
import { formatRewardBreakdown, formatStateSummary, summarizeObservation, winnerLabel } from "./rl-inspector.js";
import { SolarMaxEnv } from "./rl-env.js";

const canvas = document.querySelector("#gameCanvas");
const ctx = canvas.getContext("2d");
const statusText = document.querySelector("#statusText");
const demoButton = document.querySelector("#demoButton");
const restartButton = document.querySelector("#restartButton");
const pauseButton = document.querySelector("#pauseButton");
const playerPlanets = document.querySelector("#playerPlanets");
const enemyPlanets = document.querySelector("#enemyPlanets");
const neutralPlanets = document.querySelector("#neutralPlanets");
const demoMode = document.querySelector("#demoMode");
const demoModel = document.querySelector("#demoModel");
const demoStep = document.querySelector("#demoStep");
const demoDone = document.querySelector("#demoDone");
const demoReward = document.querySelector("#demoReward");
const demoAction = document.querySelector("#demoAction");
const demoRewardBreakdown = document.querySelector("#demoRewardBreakdown");
const demoState = document.querySelector("#demoState");
const demoLog = document.querySelector("#demoLog");

const colors = {
  [OWNER.neutral]: {
    fill: "#8d99ae",
    stroke: "#cbd5e1",
    fleet: "#cbd5e1",
    glow: "rgba(203, 213, 225, 0.28)",
    dark: "#4b5563",
    light: "#eef2ff"
  },
  [OWNER.player]: {
    fill: "#2f80ed",
    stroke: "#9dccff",
    fleet: "#7cc7ff",
    glow: "rgba(47, 128, 237, 0.34)",
    dark: "#0b3a8f",
    light: "#d8ecff"
  },
  [OWNER.enemy]: {
    fill: "#e84855",
    stroke: "#ffadb5",
    fleet: "#ff7b86",
    glow: "rgba(232, 72, 85, 0.34)",
    dark: "#8f1d2c",
    light: "#ffd1d6"
  }
};

let seed = 11;
let state = createInitialState(seed);
let selectedPlanetId = null;
let dragState = null;
let paused = false;
let lastTime = performance.now();
let enemyTimer = 1.2;
let demo = null;
let visualTime = 0;
let effects = [];

demoButton.addEventListener("click", () => {
  if (demo?.active) {
    stopDemo();
  } else {
    startDemo();
  }
});

restartButton.addEventListener("click", restart);
pauseButton.addEventListener("click", () => {
  paused = !paused;
  pauseButton.textContent = paused ? "继续" : "暂停";
  setStatus();
});

canvas.addEventListener("pointerdown", (event) => {
  if (demo?.active) {
    return;
  }
  if (state.winner) {
    restart();
    return;
  }

  const point = getCanvasPoint(event);
  const planet = hitPlanet(point.x, point.y);
  if (planet?.owner === OWNER.player) {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    selectedPlanetId = planet.id;
    dragState = {
      pointerId: event.pointerId,
      sourceId: planet.id,
      startX: point.x,
      startY: point.y,
      currentX: point.x,
      currentY: point.y,
      targetId: null,
      moved: false
    };
    setStatus();
  }
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragState || dragState.pointerId !== event.pointerId) {
    return;
  }
  event.preventDefault();
  const point = getCanvasPoint(event);
  const target = hitPlanet(point.x, point.y);
  dragState.currentX = point.x;
  dragState.currentY = point.y;
  dragState.targetId = target && target.id !== dragState.sourceId ? target.id : null;
  dragState.moved = dragState.moved || Math.hypot(point.x - dragState.startX, point.y - dragState.startY) > 6;
  setStatus();
});

canvas.addEventListener("pointerup", handlePointerEnd);
canvas.addEventListener("pointercancel", cancelDrag);
canvas.addEventListener("lostpointercapture", (event) => {
  if (dragState?.pointerId === event.pointerId) {
    dragState = null;
    setStatus();
  }
});

requestAnimationFrame(loop);

function loop(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000);
  lastTime = now;
  visualTime += dt;

  if (!paused && demo?.active) {
    runDemoTick(dt);
  } else if (!paused) {
    const before = snapshotState(state);
    updateState(state, dt);
    detectVisualEvents(before, state);
    enemyTimer -= dt;
    if (!state.winner && enemyTimer <= 0) {
      const move = chooseEnemyMove(state);
      if (move) {
        const fleet = launchFleet(state, move.sourceId, move.targetId, OWNER.enemy, move.fraction);
        if (fleet) {
          addLaunchEffect(getPlanet(state, move.sourceId), colors[OWNER.enemy].fleet);
        }
      }
      enemyTimer = 1.4 + Math.random() * 1.2;
    }
  }

  updateEffects(dt);
  draw();
  updateMetrics();
  requestAnimationFrame(loop);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground();
  drawFleetPaths();
  drawDemoIntent();
  drawDragPreview();
  drawPlanets();
  drawFleets();
  drawEffects();
}

function drawBackground() {
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#030816");
  gradient.addColorStop(0.42, "#0b1729");
  gradient.addColorStop(1, "#180d24");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.strokeStyle = "rgba(124, 199, 255, 0.045)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= canvas.width; x += 70) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= canvas.height; y += 70) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  ctx.restore();

  for (let i = 0; i < 130; i += 1) {
    const x = (i * 97 + Math.sin(visualTime * 0.08 + i) * 8) % canvas.width;
    const y = (i * 193 + Math.cos(visualTime * 0.06 + i) * 6) % canvas.height;
    const radius = (i % 4) * 0.35 + 0.45;
    ctx.fillStyle = i % 7 === 0 ? "rgba(124, 199, 255, 0.55)" : "rgba(255, 255, 255, 0.45)";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  const nebula = ctx.createRadialGradient(760, 210, 40, 760, 210, 420);
  nebula.addColorStop(0, "rgba(47, 128, 237, 0.11)");
  nebula.addColorStop(1, "rgba(47, 128, 237, 0)");
  ctx.fillStyle = nebula;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawFleetPaths() {
  ctx.save();
  ctx.setLineDash([8, 12]);
  for (const fleet of state.fleets) {
    const source = getPlanet(state, fleet.sourceId);
    const target = getPlanet(state, fleet.targetId);
    if (!source || !target) {
      continue;
    }
    ctx.strokeStyle = fleet.owner === OWNER.player ? "rgba(124, 199, 255, 0.13)" : "rgba(255, 123, 134, 0.13)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawDragPreview() {
  if (!dragState) {
    return;
  }

  const source = getPlanet(state, dragState.sourceId);
  const target = dragState.targetId ? getPlanet(state, dragState.targetId) : null;
  if (!source) {
    return;
  }

  const endX = target?.x ?? dragState.currentX;
  const endY = target?.y ?? dragState.currentY;
  ctx.save();
  drawIntentBeam(source.x, source.y, endX, endY, target ? "rgba(255, 255, 255, 0.92)" : "rgba(124, 199, 255, 0.78)");
  ctx.strokeStyle = target ? "rgba(255, 255, 255, 0.95)" : "rgba(124, 199, 255, 0.82)";
  ctx.lineWidth = target ? 3 : 2;
  ctx.setLineDash(target ? [] : [12, 10]);
  ctx.beginPath();
  ctx.moveTo(source.x, source.y);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  drawArrowHead(source.x, source.y, endX, endY, ctx.strokeStyle);

  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(124, 199, 255, 0.18)";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(source.x, source.y, source.radius + 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (target) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
    ctx.beginPath();
    ctx.arc(target.x, target.y, target.radius + 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(endX, endY, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawDemoIntent() {
  if (!demo?.active || !demo.intent || demo.intentAge > 1.1) {
    return;
  }

  const source = getPlanet(state, demo.intent.sourceId);
  const target = getPlanet(state, demo.intent.targetId);
  if (!source || !target) {
    return;
  }

  const alpha = Math.max(0, 1 - demo.intentAge / 1.1);
  ctx.save();
  drawIntentBeam(source.x, source.y, target.x, target.y, `rgba(255, 255, 255, ${0.3 + alpha * 0.55})`);
  ctx.strokeStyle = `rgba(255, 255, 255, ${0.35 + alpha * 0.55})`;
  ctx.lineWidth = 4;
  ctx.setLineDash([18, 10]);
  ctx.beginPath();
  ctx.moveTo(source.x, source.y);
  ctx.lineTo(target.x, target.y);
  ctx.stroke();
  drawArrowHead(source.x, source.y, target.x, target.y, ctx.strokeStyle);

  ctx.setLineDash([]);
  ctx.fillStyle = `rgba(124, 199, 255, ${0.12 + alpha * 0.16})`;
  ctx.strokeStyle = `rgba(124, 199, 255, ${0.45 + alpha * 0.4})`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(source.x, source.y, source.radius + 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = `rgba(255, 255, 255, ${0.08 + alpha * 0.14})`;
  ctx.strokeStyle = `rgba(255, 255, 255, ${0.4 + alpha * 0.45})`;
  ctx.beginPath();
  ctx.arc(target.x, target.y, target.radius + 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawPlanets() {
  for (const planet of state.planets) {
    const palette = colors[planet.owner];
    ctx.save();
    const pulse = 1 + Math.sin(visualTime * 2.2 + planet.x * 0.01) * 0.035;
    const glow = ctx.createRadialGradient(planet.x, planet.y, planet.radius * 0.4, planet.x, planet.y, planet.radius * 2.0);
    glow.addColorStop(0, palette.glow);
    glow.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(planet.x, planet.y, planet.radius * 2.0 * pulse, 0, Math.PI * 2);
    ctx.fill();

    const body = ctx.createRadialGradient(
      planet.x - planet.radius * 0.35,
      planet.y - planet.radius * 0.4,
      planet.radius * 0.12,
      planet.x,
      planet.y,
      planet.radius
    );
    body.addColorStop(0, palette.light);
    body.addColorStop(0.28, palette.fill);
    body.addColorStop(1, palette.dark);
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = 24;
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(planet.x, planet.y, planet.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    for (let ring = 0; ring < 2; ring += 1) {
      ctx.beginPath();
      ctx.ellipse(
        planet.x,
        planet.y,
        planet.radius * (0.62 + ring * 0.18),
        planet.radius * (0.22 + ring * 0.06),
        -0.45,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    }

    ctx.strokeStyle = planet.id === selectedPlanetId ? "#ffffff" : palette.stroke;
    ctx.lineWidth = planet.id === selectedPlanetId ? 5 : 3;
    ctx.beginPath();
    ctx.arc(planet.x, planet.y, planet.radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.font = "700 20px Inter, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(Math.floor(planet.ships).toString(), planet.x, planet.y - 3);

    ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
    ctx.font = "12px Inter, Segoe UI, sans-serif";
    ctx.fillText(`+${planet.growth.toFixed(1)}`, planet.x, planet.y + 20);
    ctx.restore();
  }
}

function drawFleets() {
  for (const fleet of state.fleets) {
    drawFleetStream(fleet);
  }
}

function drawFleetStream(fleet) {
  const source = getPlanet(state, fleet.sourceId);
  const target = getPlanet(state, fleet.targetId);
  if (!source || !target) {
    return;
  }

  const palette = colors[fleet.owner];
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / distance;
  const ny = dx / distance;
  const count = Math.max(3, Math.min(18, Math.ceil(fleet.ships / 4)));

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < count; i += 1) {
    const stagger = (i / count) * 0.18;
    const wave = Math.sin(visualTime * 7 + i * 1.7) * 5;
    const progress = Math.max(0, Math.min(1, fleet.progress - stagger));
    const x = source.x + dx * progress + nx * wave;
    const y = source.y + dy * progress + ny * wave;
    const tailX = x - (dx / distance) * (10 + (i % 3) * 4);
    const tailY = y - (dy / distance) * (10 + (i % 3) * 4);
    const radius = 2.2 + (i % 3) * 0.6;

    const trail = ctx.createLinearGradient(tailX, tailY, x, y);
    trail.addColorStop(0, "rgba(255, 255, 255, 0)");
    trail.addColorStop(1, palette.fleet);
    ctx.strokeStyle = trail;
    ctx.lineWidth = radius;
    ctx.beginPath();
    ctx.moveTo(tailX, tailY);
    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.fillStyle = palette.fleet;
    ctx.shadowColor = palette.fleet;
    ctx.shadowBlur = 9;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";

  const point = getFleetPosition(state, fleet);
  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.font = "700 11px Inter, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
  ctx.shadowBlur = 4;
  ctx.fillText(fleet.ships.toString(), point.x, point.y - 14);
  ctx.restore();
}

function updateMetrics() {
  const counts = getCounts(state);
  playerPlanets.textContent = counts.player;
  enemyPlanets.textContent = counts.enemy;
  neutralPlanets.textContent = counts.neutral;
  updateDemoPanel();
  setStatus();
}

function setStatus() {
  if (demo?.active) {
    if (state.winner === OWNER.player) {
      statusText.textContent = "演示结束：蓝色胜利。点击 DQN 演示返回手动模式，或点击重新开始再看一局。";
      return;
    }
    if (state.winner === OWNER.enemy) {
      statusText.textContent = "演示结束：红色胜利。当前是未训练 demo policy，可继续调整训练。";
      return;
    }
    if (paused) {
      statusText.textContent = "DQN 演示已暂停。";
      return;
    }
    statusText.textContent = `DQN 演示中：${demo.policyName} 正在按 action mask 自动选择合法动作。`;
    return;
  }
  if (state.winner === OWNER.player) {
    statusText.textContent = "胜利！点击重新开始，或直接点击画布再来一局。";
    return;
  }
  if (state.winner === OWNER.enemy) {
    statusText.textContent = "失败。点击重新开始，调整扩张节奏再试。";
    return;
  }
  if (paused) {
    statusText.textContent = "已暂停。";
    return;
  }
  if (dragState) {
    const source = getPlanet(state, dragState.sourceId);
    if (dragState.targetId) {
      statusText.textContent = `释放即可从 ${Math.floor(source.ships)} 艘舰的星球派出一半舰队。`;
    } else {
      statusText.textContent = "拖到目标星球后释放，派出一半舰队。";
    }
    return;
  }
  if (selectedPlanetId) {
    const planet = getPlanet(state, selectedPlanetId);
    statusText.textContent = `已选择 ${Math.floor(planet.ships)} 艘舰的星球，可拖拽或点击目标派遣一半舰队。`;
    return;
  }
  statusText.textContent = "从蓝色星球拖到目标星球，释放后派出一半舰队。";
}

function restart() {
  if (demo?.active) {
    startDemo();
    return;
  }

  seed += 1;
  state = createInitialState(seed);
  selectedPlanetId = null;
  dragState = null;
  paused = false;
  enemyTimer = 1.2;
  pauseButton.textContent = "暂停";
  setStatus();
}

function startDemo() {
  seed += 1;
  const env = new SolarMaxEnv({
    seed,
    difficulty: "normal",
    opponent: "random",
    maxEpisodeSteps: 220
  });
  const observation = env.reset(seed);
  demo = {
    active: true,
    env,
    agent: new DqnAgent({
      hiddenSize: 32,
      rng: seededRandom(seed + 31)
    }),
    observation,
    timer: 0.25,
    stepDelay: 0.95,
    totalReward: 0,
    lastReward: 0,
    lastActionLabel: "准备开始",
    lastRewardBreakdown: null,
    lastResultInfo: null,
    stateSummary: summarizeObservation(observation),
    stepLog: [],
    intent: null,
    intentAge: 999,
    epsilon: 0.35,
    policyName: "未充分训练 DQN demo policy",
    modelSource: "loading checkpoint..."
  };
  state = env.state;
  selectedPlanetId = null;
  dragState = null;
  paused = false;
  pauseButton.textContent = "暂停";
  demoButton.textContent = "停止演示";
  setStatus();
  updateDemoPanel();
  loadCheckpointForDemo(demo);
}

function stopDemo() {
  demo = null;
  seed += 1;
  state = createInitialState(seed);
  selectedPlanetId = null;
  dragState = null;
  paused = false;
  enemyTimer = 1.2;
  demoButton.textContent = "DQN 演示";
  pauseButton.textContent = "暂停";
  setStatus();
  updateDemoPanel();
}

function runDemoTick(dt) {
  if (!demo || demo.env.done) {
    if (demo) {
      demo.intentAge += dt;
    }
    return;
  }

  demo.timer -= dt;
  demo.intentAge += dt;
  if (demo.timer > 0) {
    return;
  }

  const decision = chooseDemoDecision(demo.agent, demo.observation, demo.epsilon);
  const action = decision.action;
  const planetsBefore = demo.env.state.planets;
  const intent = actionIntent(action, planetsBefore);
  const before = snapshotState(demo.env.state);
  const result = demo.env.step(action ?? -1);
  detectVisualEvents(before, demo.env.state);
  if (intent) {
    addLaunchEffect(getPlanet(demo.env.state, intent.sourceId), colors[OWNER.player].fleet);
  }
  demo.observation = result.observation;
  demo.totalReward += result.reward;
  demo.lastReward = result.reward;
  demo.lastRewardBreakdown = result.info.rewardBreakdown;
  demo.lastResultInfo = result.info;
  demo.stateSummary = summarizeObservation(result.observation);
  demo.lastActionLabel = formatActionLabel(action, intent, result.info, decision);
  demo.intent = intent;
  demo.intentAge = 0;
  demo.timer = demo.stepDelay;
  demo.stepLog.unshift(
    [
      `#${result.info.steps}`,
      demo.lastActionLabel,
      `reward ${formatSigned(result.reward)}`,
      `P/E/N ${demo.stateSummary.playerPlanets}/${demo.stateSummary.enemyPlanets}/${demo.stateSummary.neutralPlanets}`,
      `done ${result.done}`
    ].join(" | ")
  );
  demo.stepLog = demo.stepLog.slice(0, 8);
  state = demo.env.state;
}

function updateDemoPanel() {
  if (!demo?.active) {
    demoMode.textContent = "手动游玩";
    demoModel.textContent = "未加载";
    demoStep.textContent = "0";
    demoDone.textContent = "false / none";
    demoReward.textContent = "0.00";
    demoAction.textContent = "-";
    demoRewardBreakdown.textContent = "-";
    demoState.textContent = "-";
    demoLog.replaceChildren();
    return;
  }

  demoMode.textContent = `${demo.policyName}，epsilon ${demo.epsilon.toFixed(2)}；输仍可能发生`;
  demoModel.textContent = demo.modelSource;
  demoStep.textContent = demo.env.steps.toString();
  demoDone.textContent = `${demo.env.done} / ${winnerLabel(demo.env.state.winner)}`;
  demoReward.textContent = `${demo.totalReward.toFixed(2)} (${demo.lastReward >= 0 ? "+" : ""}${demo.lastReward.toFixed(2)})`;
  demoAction.textContent = demo.lastActionLabel;
  demoRewardBreakdown.textContent = formatRewardBreakdown(demo.lastRewardBreakdown);
  demoState.textContent = formatStateSummary(demo.stateSummary);
  demoLog.replaceChildren(...demo.stepLog.map((entry) => {
    const item = document.createElement("li");
    item.textContent = entry;
    return item;
  }));
}

async function loadCheckpointForDemo(targetDemo) {
  try {
    const response = await fetch("./checkpoints/dqn-latest.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const checkpoint = await response.json();
    if (!targetDemo.active || demo !== targetDemo) {
      return;
    }

    targetDemo.agent = DqnAgent.fromJSON(checkpoint, { rng: seededRandom(seed + 131) });
    targetDemo.epsilon = 0.02;
    targetDemo.policyName = "loaded checkpoint DQN";
    targetDemo.modelSource = `loaded checkpoints/dqn-latest.json (${checkpoint.metadata?.episodes ?? "?"} episodes)`;
    setStatus();
    updateDemoPanel();
  } catch (error) {
    if (!targetDemo.active || demo !== targetDemo) {
      return;
    }

    targetDemo.modelSource = `load failed; using untrained fallback (${error.message})`;
    targetDemo.policyName = "未充分训练 DQN demo policy";
    targetDemo.epsilon = 0.35;
    setStatus();
    updateDemoPanel();
  }
}

function handlePointerEnd(event) {
  if (demo?.active) {
    return;
  }

  const point = getCanvasPoint(event);
  const target = hitPlanet(point.x, point.y);

  if (dragState?.pointerId === event.pointerId) {
    event.preventDefault();
    const sourceId = dragState.sourceId;
    const source = getPlanet(state, sourceId);
    const shouldLaunch = target && target.id !== sourceId;
    if (shouldLaunch) {
      const fleet = launchFleet(state, sourceId, target.id, OWNER.player, 0.5);
      if (fleet) {
        addLaunchEffect(source, colors[OWNER.player].fleet);
      }
      selectedPlanetId = fleet && source.ships >= 4 ? sourceId : null;
    } else if (!dragState.moved) {
      selectedPlanetId = sourceId;
    }
    dragState = null;
    setStatus();
    return;
  }

  if (!target) {
    selectedPlanetId = null;
    setStatus();
    return;
  }

  const selected = getPlanet(state, selectedPlanetId);
  if (selected?.owner === OWNER.player && target.id !== selectedPlanetId) {
    const fleet = launchFleet(state, selectedPlanetId, target.id, OWNER.player, 0.5);
    if (fleet) {
      addLaunchEffect(selected, colors[OWNER.player].fleet);
      selectedPlanetId = selected.ships >= 4 ? selectedPlanetId : null;
    }
  } else if (target.owner === OWNER.player) {
    selectedPlanetId = target.id;
  } else {
    selectedPlanetId = null;
  }
  setStatus();
}

function cancelDrag(event) {
  if (dragState?.pointerId === event.pointerId) {
    dragState = null;
    setStatus();
  }
}

function seededRandom(seedValue) {
  let value = seedValue >>> 0;
  return function next() {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function formatActionLabel(action, intent, info, decision) {
  const actionInfo = info.action;
  const legal = info.legal ? "legal" : "illegal";
  const explored = decision.explored ? "explore" : "greedy";
  if (!intent || !actionInfo) {
    return `index ${action ?? "none"} | 无合法动作 | ${legal} | ${decision.policy} ${explored}`;
  }

  return [
    `index ${action}`,
    `${intent.sourceId} -> ${intent.targetId}`,
    `${Math.round(intent.ratio * 100)}%`,
    legal,
    `${decision.policy} ${explored}`
  ].join(" | ");
}

function formatSigned(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function drawIntentBeam(x1, y1, x2, y2, color) {
  const beam = ctx.createLinearGradient(x1, y1, x2, y2);
  beam.addColorStop(0, "rgba(255, 255, 255, 0.05)");
  beam.addColorStop(0.55, color);
  beam.addColorStop(1, "rgba(255, 255, 255, 0.9)");
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = beam;
  ctx.lineWidth = 9;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawArrowHead(x1, y1, x2, y2, color) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const size = 18;
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - Math.cos(angle - 0.45) * size, y2 - Math.sin(angle - 0.45) * size);
  ctx.lineTo(x2 - Math.cos(angle + 0.45) * size, y2 - Math.sin(angle + 0.45) * size);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function updateEffects(dt) {
  for (const effect of effects) {
    effect.age += dt;
  }
  effects = effects.filter((effect) => effect.age < effect.duration);
}

function drawEffects() {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const effect of effects) {
    const t = effect.age / effect.duration;
    const alpha = Math.max(0, 1 - t);
    if (effect.type === "ring") {
      ctx.strokeStyle = withAlpha(effect.color, alpha * 0.9);
      ctx.lineWidth = 3 + alpha * 5;
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, effect.radius + t * effect.expand, 0, Math.PI * 2);
      ctx.stroke();
    } else if (effect.type === "flash") {
      const gradient = ctx.createRadialGradient(effect.x, effect.y, 0, effect.x, effect.y, effect.radius * (1 + t));
      gradient.addColorStop(0, withAlpha(effect.color, alpha));
      gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, effect.radius * (1 + t), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function addLaunchEffect(planet, color) {
  if (!planet) {
    return;
  }
  effects.push({
    type: "flash",
    x: planet.x,
    y: planet.y,
    radius: planet.radius * 0.9,
    color,
    age: 0,
    duration: 0.42
  });
}

function addImpactEffect(planet, color, captured = false) {
  if (!planet) {
    return;
  }
  effects.push({
    type: "ring",
    x: planet.x,
    y: planet.y,
    radius: planet.radius * 0.8,
    expand: captured ? 72 : 42,
    color,
    age: 0,
    duration: captured ? 0.95 : 0.62
  });
  if (captured) {
    effects.push({
      type: "flash",
      x: planet.x,
      y: planet.y,
      radius: planet.radius * 1.25,
      color,
      age: 0,
      duration: 0.58
    });
  }
}

function snapshotState(currentState) {
  return {
    planets: currentState.planets.map((planet) => ({
      id: planet.id,
      owner: planet.owner,
      x: planet.x,
      y: planet.y,
      radius: planet.radius
    })),
    fleets: currentState.fleets.map((fleet) => ({
      id: fleet.id,
      owner: fleet.owner,
      targetId: fleet.targetId
    }))
  };
}

function detectVisualEvents(before, after) {
  const afterFleetIds = new Set(after.fleets.map((fleet) => fleet.id));
  for (const fleet of before.fleets) {
    if (!afterFleetIds.has(fleet.id)) {
      const target = getPlanet(after, fleet.targetId);
      addImpactEffect(target, colors[fleet.owner]?.fleet ?? "rgba(255,255,255,0.9)");
    }
  }

  for (const beforePlanet of before.planets) {
    const afterPlanet = getPlanet(after, beforePlanet.id);
    if (afterPlanet && afterPlanet.owner !== beforePlanet.owner) {
      addImpactEffect(afterPlanet, colors[afterPlanet.owner].fleet, true);
    }
  }
}

function withAlpha(color, alpha) {
  if (color.startsWith("#")) {
    const value = Number.parseInt(color.slice(1), 16);
    const r = (value >> 16) & 255;
    const g = (value >> 8) & 255;
    const b = value & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  if (color.startsWith("rgba")) {
    return color.replace(/rgba\(([^,]+),([^,]+),([^,]+),[^)]+\)/, `rgba($1,$2,$3,${alpha})`);
  }
  return color;
}

function hitPlanet(x, y) {
  return state.planets.find((planet) => Math.hypot(planet.x - x, planet.y - y) <= planet.radius);
}

function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height
  };
}
