import { DqnAgent } from "../src/dqn-agent.js";
import { SolarMaxEnv } from "../src/rl-env.js";
import { existsSync, readFileSync } from "node:fs";

const checkpointPath = process.argv[2] ?? "checkpoints/dqn-latest.json";
const episodes = Number.parseInt(process.argv[3] ?? "25", 10);
const maxEpisodeSteps = Number.parseInt(process.argv[4] ?? "300", 10);
const difficulty = process.argv[5] ?? "normal";
const opponent = process.argv[6] ?? "random";
const baseSeed = Number.parseInt(process.argv[7] ?? "500", 10);

if (!existsSync(checkpointPath)) {
  console.error(`Checkpoint not found: ${checkpointPath}`);
  console.error("Run: node scripts/train-dqn.js 20 200 normal random 1");
  process.exit(1);
}

const checkpoint = JSON.parse(readFileSync(checkpointPath, "utf8"));
const agent = DqnAgent.fromJSON(checkpoint, { rng: mulberry32(baseSeed + 77) });
const env = new SolarMaxEnv({ seed: baseSeed, maxEpisodeSteps, difficulty, opponent });

let wins = 0;
let losses = 0;
let draws = 0;
let totalReward = 0;
let totalSteps = 0;

for (let episode = 1; episode <= episodes; episode += 1) {
  let observation = env.reset(baseSeed + episode);
  let done = false;
  let reward = 0;
  let lastInfo = null;

  while (!done) {
    const action = agent.act(observation, 0);
    const result = env.step(action ?? -1);
    observation = result.observation;
    reward += result.reward;
    done = result.done;
    lastInfo = result.info;
  }

  totalReward += reward;
  totalSteps += lastInfo?.steps ?? 0;
  if (lastInfo?.winner === 1) {
    wins += 1;
  } else if (lastInfo?.winner === 2) {
    losses += 1;
  } else {
    draws += 1;
  }
}

console.log(
  JSON.stringify(
    {
      checkpointPath,
      checkpointMetadata: checkpoint.metadata ?? {},
      episodes,
      difficulty,
      opponent,
      baseSeed,
      wins,
      losses,
      draws,
      winRate: wins / episodes,
      averageReward: totalReward / episodes,
      averageSteps: totalSteps / episodes
    },
    null,
    2
  )
);

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
