export const OWNER = Object.freeze({
  neutral: 0,
  player: 1,
  enemy: 2
});

export const OWNER_NAMES = Object.freeze({
  [OWNER.neutral]: "neutral",
  [OWNER.player]: "player",
  [OWNER.enemy]: "enemy"
});

const BASE_SPEED = 88;
const MIN_LAUNCH_SHIPS = 2;

export function createInitialState(seed = 7, options = {}) {
  const rng = mulberry32(seed);
  const randomize = options.randomize ?? true;
  const homeJitter = randomize ? () => jitter(rng, 14) : () => 0;
  const neutralJitter = randomize ? () => jitter(rng, 34) : () => 0;
  const shipJitter = randomize ? (amount) => Math.floor(jitter(rng, amount)) : () => 0;
  const growthJitter = randomize ? (amount) => jitter(rng, amount) : () => 0;
  const planets = [
    planet("p-home", 180 + homeJitter(), 350 + homeJitter(), 52, OWNER.player, 44, 4.6),
    planet("e-home", 920 + homeJitter(), 350 + homeJitter(), 52, OWNER.enemy, 44, 4.6),
    planet("north", 540 + neutralJitter(), 145 + neutralJitter(), 42, OWNER.neutral, 28 + shipJitter(5), 3.2 + growthJitter(0.25)),
    planet("south", 560 + neutralJitter(), 555 + neutralJitter(), 46, OWNER.neutral, 34 + shipJitter(5), 3.6 + growthJitter(0.25)),
    planet("west", 330 + neutralJitter(), 210 + neutralJitter(), 34, OWNER.neutral, 25 + shipJitter(8), 2.6 + growthJitter(0.25)),
    planet("east", 770 + neutralJitter(), 500 + neutralJitter(), 34, OWNER.neutral, 25 + shipJitter(8), 2.6 + growthJitter(0.25)),
    planet("mid-left", 410 + neutralJitter(), 430 + neutralJitter(), 30, OWNER.neutral, 22 + shipJitter(7), 2.2 + growthJitter(0.2)),
    planet("mid-right", 710 + neutralJitter(), 260 + neutralJitter(), 30, OWNER.neutral, 22 + shipJitter(7), 2.2 + growthJitter(0.2))
  ];

  return {
    planets,
    fleets: [],
    time: 0,
    nextFleetId: 1,
    winner: null
  };
}

export function updateState(state, dt) {
  if (state.winner) {
    return state;
  }

  state.time += dt;
  for (const planet of state.planets) {
    if (planet.owner !== OWNER.neutral) {
      planet.ships = Math.min(999, planet.ships + planet.growth * dt);
    }
  }

  for (const fleet of state.fleets) {
    fleet.progress += dt / fleet.duration;
  }

  const arrived = state.fleets.filter((fleet) => fleet.progress >= 1);
  state.fleets = state.fleets.filter((fleet) => fleet.progress < 1);
  for (const fleet of arrived) {
    resolveArrival(state, fleet);
  }

  state.winner = getWinner(state);
  return state;
}

export function launchFleet(state, sourceId, targetId, owner, fraction = 0.5) {
  if (state.winner || sourceId === targetId) {
    return null;
  }

  const source = getPlanet(state, sourceId);
  const target = getPlanet(state, targetId);
  if (!source || !target || source.owner !== owner) {
    return null;
  }

  const ships = Math.floor(source.ships * clamp(fraction, 0, 1));
  if (ships < MIN_LAUNCH_SHIPS) {
    return null;
  }

  source.ships -= ships;
  const distance = Math.hypot(target.x - source.x, target.y - source.y);
  const fleet = {
    id: `f-${state.nextFleetId++}`,
    owner,
    sourceId,
    targetId,
    x: source.x,
    y: source.y,
    ships,
    progress: 0,
    duration: Math.max(0.8, distance / BASE_SPEED)
  };
  state.fleets.push(fleet);
  return fleet;
}

export function chooseEnemyMove(state) {
  const enemyPlanets = state.planets
    .filter((planet) => planet.owner === OWNER.enemy && planet.ships >= 18)
    .sort((a, b) => b.ships - a.ships);

  if (enemyPlanets.length === 0) {
    return null;
  }

  const source = enemyPlanets[0];
  const targets = state.planets
    .filter((planet) => planet.owner !== OWNER.enemy)
    .map((planet) => ({
      planet,
      score: planet.ships + Math.hypot(planet.x - source.x, planet.y - source.y) / 18 - planet.growth * 4
    }))
    .sort((a, b) => a.score - b.score);

  const target = targets[0]?.planet;
  if (!target || source.ships < target.ships * 0.85 + 8) {
    return null;
  }

  return { sourceId: source.id, targetId: target.id, fraction: 0.5 };
}

export function getWinner(state) {
  const playerAlive = hasPresence(state, OWNER.player);
  const enemyAlive = hasPresence(state, OWNER.enemy);
  if (playerAlive && !enemyAlive) {
    return OWNER.player;
  }
  if (enemyAlive && !playerAlive) {
    return OWNER.enemy;
  }
  return null;
}

export function getCounts(state) {
  return state.planets.reduce(
    (counts, planet) => {
      counts[OWNER_NAMES[planet.owner]] += 1;
      return counts;
    },
    { neutral: 0, player: 0, enemy: 0 }
  );
}

export function getPlanet(state, id) {
  return state.planets.find((planet) => planet.id === id);
}

export function getFleetPosition(state, fleet) {
  const source = getPlanet(state, fleet.sourceId);
  const target = getPlanet(state, fleet.targetId);
  if (!source || !target) {
    return { x: fleet.x, y: fleet.y };
  }
  return {
    x: lerp(source.x, target.x, fleet.progress),
    y: lerp(source.y, target.y, fleet.progress)
  };
}

function resolveArrival(state, fleet) {
  const target = getPlanet(state, fleet.targetId);
  if (!target) {
    return;
  }

  if (target.owner === fleet.owner) {
    target.ships += fleet.ships;
    return;
  }

  target.ships -= fleet.ships;
  if (target.ships < 0) {
    target.owner = fleet.owner;
    target.ships = Math.abs(target.ships);
  }
}

function hasPresence(state, owner) {
  return (
    state.planets.some((planet) => planet.owner === owner) ||
    state.fleets.some((fleet) => fleet.owner === owner)
  );
}

function planet(id, x, y, radius, owner, ships, growth) {
  return { id, x, y, radius, owner, ships, growth };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(start, end, amount) {
  return start + (end - start) * amount;
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
