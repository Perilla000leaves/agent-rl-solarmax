import {
  OWNER,
  chooseEnemyMove,
  createInitialState,
  getCounts,
  getPlanet,
  launchFleet,
  updateState
} from "./game-core.js";

export const SEND_RATIOS = Object.freeze([0.25, 0.5, 0.75]);
export const PLANET_COUNT = 8;
export const MAX_FLEETS = 32;
export const ACTION_COUNT = PLANET_COUNT * PLANET_COUNT * SEND_RATIOS.length;
export const OPPONENT_POLICIES = Object.freeze(["aggressive", "expansion", "defensive"]);
export const DIFFICULTIES = Object.freeze({
  easy: {
    enemyIntervalSteps: 4,
    enemySendRatio: 0.42,
    enemyShipMultiplier: 0.92,
    enemyGrowthMultiplier: 0.96,
    randomOpponent: false
  },
  normal: {
    enemyIntervalSteps: 3,
    enemySendRatio: 0.5,
    enemyShipMultiplier: 1,
    enemyGrowthMultiplier: 1,
    randomOpponent: false
  },
  hard: {
    enemyIntervalSteps: 2,
    enemySendRatio: 0.62,
    enemyShipMultiplier: 1.16,
    enemyGrowthMultiplier: 1.08,
    randomOpponent: true
  }
});

const STEP_SECONDS = 0.75;
const ENEMY_INTERVAL_STEPS = 3;
const MAX_EPISODE_STEPS = 600;
const MAX_SHIPS = 220;
const MAP_WIDTH = 1100;
const MAP_HEIGHT = 700;

export class SolarMaxEnv {
  constructor(options = {}) {
    this.difficultyName = options.difficulty ?? "normal";
    this.difficulty = DIFFICULTIES[this.difficultyName] ?? DIFFICULTIES.normal;
    this.opponent = options.opponent ?? "aggressive";
    this.maxEpisodeSteps = options.maxEpisodeSteps ?? MAX_EPISODE_STEPS;
    this.stepSeconds = options.stepSeconds ?? STEP_SECONDS;
    this.enemyIntervalSteps = options.enemyIntervalSteps ?? this.difficulty.enemyIntervalSteps ?? ENEMY_INTERVAL_STEPS;
    this.seed = options.seed ?? 7;
    this.reset(this.seed);
  }

  reset(seed = this.seed) {
    this.seed = seed;
    this.rng = mulberry32(seed * 9973 + 17);
    this.currentOpponent = selectOpponent(this.opponent, this.difficulty, this.rng);
    this.state = createInitialState(seed, { randomize: true });
    applyDifficulty(this.state, this.difficulty);
    this.steps = 0;
    this.done = false;
    this.lastMetrics = getMetrics(this.state);
    return this.observe();
  }

  step(action) {
    if (this.done) {
      return {
        observation: this.observe(),
        reward: 0,
        done: true,
        info: { alreadyDone: true }
      };
    }

    const before = this.lastMetrics;
    const decoded = decodeAction(action);
    const mask = this.legalActions();
    const legalActionCount = countLegalActions(mask);
    const legal = Number.isInteger(action) && action >= 0 && action < ACTION_COUNT && mask[action] === 1;
    let launched = false;
    let illegal = false;

    if (legal) {
      const source = this.state.planets[decoded.sourceIndex];
      const target = this.state.planets[decoded.targetIndex];
      launched = Boolean(launchFleet(this.state, source.id, target.id, OWNER.player, decoded.sendRatio));
    } else {
      illegal = true;
    }

    if (this.steps % this.enemyIntervalSteps === 0) {
      const enemyMove = chooseOpponentMove(this.state, this.currentOpponent, {
        rng: this.rng,
        sendRatio: this.difficulty.enemySendRatio
      });
      if (enemyMove) {
        launchFleet(this.state, enemyMove.sourceId, enemyMove.targetId, OWNER.enemy, enemyMove.fraction);
      }
    }

    updateState(this.state, this.stepSeconds);
    this.steps += 1;
    this.done = Boolean(this.state.winner) || this.steps >= this.maxEpisodeSteps;

    const after = getMetrics(this.state);
    const rewardBreakdown = calculateRewardBreakdown(before, after, this.state.winner, illegal);
    const reward = rewardBreakdown.total;
    this.lastMetrics = after;

    return {
      observation: this.observe(),
      reward,
      done: this.done,
      info: {
        legal,
        illegal,
        launched,
        action: legal ? { index: action, ...decoded } : null,
        actionIndex: Number.isInteger(action) ? action : null,
        legalActionCount,
        opponent: this.currentOpponent,
        difficulty: this.difficultyName,
        rewardBreakdown,
        beforeMetrics: before,
        afterMetrics: after,
        winner: this.state.winner,
        steps: this.steps
      }
    };
  }

  observe() {
    return {
      planets: observePlanets(this.state),
      fleets: observeFleets(this.state),
      global: observeGlobal(this.state, this.steps, this.maxEpisodeSteps),
      actionMask: this.legalActions()
    };
  }

  legalActions() {
    const mask = Array(ACTION_COUNT).fill(0);
    for (let sourceIndex = 0; sourceIndex < PLANET_COUNT; sourceIndex += 1) {
      const source = this.state.planets[sourceIndex];
      if (!source || source.owner !== OWNER.player) {
        continue;
      }

      for (let targetIndex = 0; targetIndex < PLANET_COUNT; targetIndex += 1) {
        if (sourceIndex === targetIndex || !this.state.planets[targetIndex]) {
          continue;
        }

        for (let ratioIndex = 0; ratioIndex < SEND_RATIOS.length; ratioIndex += 1) {
          const ships = Math.floor(source.ships * SEND_RATIOS[ratioIndex]);
          if (ships >= 2) {
            mask[encodeAction(sourceIndex, targetIndex, ratioIndex)] = 1;
          }
        }
      }
    }
    return mask;
  }
}

export function encodeAction(sourceIndex, targetIndex, ratioIndex) {
  return (sourceIndex * PLANET_COUNT + targetIndex) * SEND_RATIOS.length + ratioIndex;
}

export function decodeAction(action) {
  const ratioIndex = action % SEND_RATIOS.length;
  const pairIndex = Math.floor(action / SEND_RATIOS.length);
  const sourceIndex = Math.floor(pairIndex / PLANET_COUNT);
  const targetIndex = pairIndex % PLANET_COUNT;
  return {
    sourceIndex,
    targetIndex,
    ratioIndex,
    sendRatio: SEND_RATIOS[ratioIndex]
  };
}

export function sampleLegalAction(actionMask, rng = Math.random) {
  const legal = [];
  for (let i = 0; i < actionMask.length; i += 1) {
    if (actionMask[i] === 1) {
      legal.push(i);
    }
  }
  if (legal.length === 0) {
    return null;
  }
  return legal[Math.floor(rng() * legal.length)];
}

export function chooseOpponentMove(state, policy = "aggressive", options = {}) {
  const sendRatio = options.sendRatio ?? 0.5;
  const rng = options.rng ?? Math.random;
  const move =
    policy === "expansion"
      ? chooseExpansionMove(state, sendRatio)
      : policy === "defensive"
        ? chooseDefensiveMove(state, sendRatio)
        : chooseAggressiveMove(state, sendRatio);

  if (move) {
    return move;
  }

  const fallback = chooseEnemyMove(state);
  return fallback ? { ...fallback, fraction: sendRatio + jitter(rng, 0.06) } : null;
}

function observePlanets(state) {
  return state.planets.flatMap((planet) => {
    const incomingPlayer = incomingShips(state, planet.id, OWNER.player);
    const incomingEnemy = incomingShips(state, planet.id, OWNER.enemy);
    return [
      planet.owner === OWNER.player ? 1 : 0,
      planet.owner === OWNER.enemy ? 1 : 0,
      planet.owner === OWNER.neutral ? 1 : 0,
      planet.x / MAP_WIDTH,
      planet.y / MAP_HEIGHT,
      planet.radius / 60,
      normalizeShips(planet.ships),
      planet.growth / 5,
      normalizeShips(incomingPlayer),
      normalizeShips(incomingEnemy)
    ];
  });
}

function observeFleets(state) {
  const features = [];
  const fleets = state.fleets.slice(0, MAX_FLEETS);
  for (const fleet of fleets) {
    features.push(
      fleet.owner === OWNER.player ? 1 : 0,
      fleet.owner === OWNER.enemy ? 1 : 0,
      planetIndex(state, fleet.sourceId) / (PLANET_COUNT - 1),
      planetIndex(state, fleet.targetId) / (PLANET_COUNT - 1),
      normalizeShips(fleet.ships),
      clamp01(fleet.progress),
      clamp01(1 - fleet.progress)
    );
  }

  while (features.length < MAX_FLEETS * 7) {
    features.push(0);
  }
  return features;
}

function observeGlobal(state, steps, maxEpisodeSteps) {
  const counts = getCounts(state);
  const metrics = getMetrics(state);
  return [
    steps / maxEpisodeSteps,
    counts.player / PLANET_COUNT,
    counts.enemy / PLANET_COUNT,
    counts.neutral / PLANET_COUNT,
    metrics.playerGrowth / 24,
    metrics.enemyGrowth / 24,
    normalizeShips(metrics.playerShips),
    normalizeShips(metrics.enemyShips),
    state.winner === OWNER.player ? 1 : 0,
    state.winner === OWNER.enemy ? 1 : 0
  ];
}

function calculateRewardBreakdown(before, after, winner, illegal) {
  const reserveBefore = Math.min(1, before.playerPlanetShips / Math.max(1, before.playerPlanets * 26));
  const reserveAfter = Math.min(1, after.playerPlanetShips / Math.max(1, after.playerPlanets * 26));
  const components = {
    planetDelta: (after.playerPlanets - before.playerPlanets) * 4 - (after.enemyPlanets - before.enemyPlanets) * 2.5,
    neutralExpansion: Math.max(0, before.neutralPlanets - after.neutralPlanets) * Math.max(0, after.playerPlanets - before.playerPlanets) * 2,
    growthDelta: (after.playerGrowth - before.playerGrowth) * 0.65 - (after.enemyGrowth - before.enemyGrowth) * 0.35,
    shipDelta: (after.playerShips - before.playerShips) * 0.01 - (after.enemyShips - before.enemyShips) * 0.01,
    reserveDelta: (reserveAfter - reserveBefore) * 1.2,
    lowReserve: reserveAfter < 0.35 ? -0.15 : 0,
    winLoss: winner === OWNER.player ? 25 : winner === OWNER.enemy ? -25 : 0,
    illegal: illegal ? -1 : 0
  };
  const total = Object.values(components).reduce((sum, value) => sum + value, 0);
  return { ...components, total };
}

function chooseAggressiveMove(state, sendRatio) {
  const move = chooseEnemyMove(state);
  return move ? { ...move, fraction: sendRatio } : null;
}

function chooseExpansionMove(state, sendRatio) {
  const sources = enemySources(state, 16);
  for (const source of sources) {
    const target = state.planets
      .filter((planet) => planet.owner === OWNER.neutral)
      .map((planet) => ({
        planet,
        score: planet.ships + distance(source, planet) / 22 - planet.growth * 5
      }))
      .sort((a, b) => a.score - b.score)[0]?.planet;

    if (target && source.ships * sendRatio >= target.ships * 0.75) {
      return { sourceId: source.id, targetId: target.id, fraction: sendRatio };
    }
  }
  return chooseAggressiveMove(state, sendRatio);
}

function chooseDefensiveMove(state, sendRatio) {
  const threatened = state.planets
    .filter((planet) => planet.owner === OWNER.enemy)
    .map((planet) => ({
      planet,
      pressure: incomingShips(state, planet.id, OWNER.player) - incomingShips(state, planet.id, OWNER.enemy)
    }))
    .filter((item) => item.pressure > item.planet.ships * 0.4)
    .sort((a, b) => b.pressure - a.pressure)[0]?.planet;

  if (threatened) {
    const source = enemySources(state, 22).find((planet) => planet.id !== threatened.id);
    if (source) {
      return { sourceId: source.id, targetId: threatened.id, fraction: Math.min(0.45, sendRatio) };
    }
  }

  const source = enemySources(state, 28)[0];
  if (!source) {
    return null;
  }
  const target = state.planets
    .filter((planet) => planet.owner === OWNER.player)
    .map((planet) => ({
      planet,
      score: planet.ships + distance(source, planet) / 16
    }))
    .sort((a, b) => a.score - b.score)[0]?.planet;

  return target && source.ships > target.ships * 1.2
    ? { sourceId: source.id, targetId: target.id, fraction: Math.min(0.55, sendRatio) }
    : null;
}

function enemySources(state, minShips) {
  return state.planets
    .filter((planet) => planet.owner === OWNER.enemy && planet.ships >= minShips)
    .sort((a, b) => b.ships - a.ships);
}

function applyDifficulty(state, difficulty) {
  const enemyHome = getPlanet(state, "e-home");
  if (enemyHome) {
    enemyHome.ships *= difficulty.enemyShipMultiplier;
    enemyHome.growth *= difficulty.enemyGrowthMultiplier;
  }
}

function selectOpponent(opponent, difficulty, rng) {
  if (opponent === "random" || difficulty.randomOpponent) {
    return OPPONENT_POLICIES[Math.floor(rng() * OPPONENT_POLICIES.length)];
  }
  return OPPONENT_POLICIES.includes(opponent) ? opponent : "aggressive";
}

function getMetrics(state) {
  const metrics = {
    playerPlanets: 0,
    enemyPlanets: 0,
    neutralPlanets: 0,
    playerGrowth: 0,
    enemyGrowth: 0,
    playerShips: 0,
    enemyShips: 0,
    playerPlanetShips: 0,
    enemyPlanetShips: 0,
    playerFleets: 0,
    enemyFleets: 0,
    fleetCount: state.fleets.length
  };

  for (const planet of state.planets) {
    if (planet.owner === OWNER.player) {
      metrics.playerPlanets += 1;
      metrics.playerGrowth += planet.growth;
      metrics.playerShips += planet.ships;
      metrics.playerPlanetShips += planet.ships;
    } else if (planet.owner === OWNER.enemy) {
      metrics.enemyPlanets += 1;
      metrics.enemyGrowth += planet.growth;
      metrics.enemyShips += planet.ships;
      metrics.enemyPlanetShips += planet.ships;
    } else {
      metrics.neutralPlanets += 1;
    }
  }

  for (const fleet of state.fleets) {
    if (fleet.owner === OWNER.player) {
      metrics.playerFleets += 1;
      metrics.playerShips += fleet.ships;
    } else if (fleet.owner === OWNER.enemy) {
      metrics.enemyFleets += 1;
      metrics.enemyShips += fleet.ships;
    }
  }

  return metrics;
}

function incomingShips(state, planetId, owner) {
  return state.fleets
    .filter((fleet) => fleet.targetId === planetId && fleet.owner === owner)
    .reduce((sum, fleet) => sum + fleet.ships, 0);
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function planetIndex(state, planetId) {
  return Math.max(0, state.planets.findIndex((planet) => planet.id === planetId));
}

function normalizeShips(ships) {
  return clamp01(ships / MAX_SHIPS);
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function countLegalActions(mask) {
  return mask.reduce((sum, value) => sum + (value === 1 ? 1 : 0), 0);
}

function jitter(rng, amount) {
  return (rng() * 2 - 1) * amount;
}

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
