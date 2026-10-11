import { DqnAgent } from "../src/dqn-agent.js";
import { SolarMaxEnv } from "../src/rl-env.js";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const episodes = Number.parseInt(process.argv[2] ?? "50", 10);
const maxEpisodeSteps = Number.parseInt(process.argv[3] ?? "300", 10);
const difficulty = process.argv[4] ?? "normal";
const opponent = process.argv[5] ?? "aggressive";
const baseSeed = Number.parseInt(process.argv[6] ?? "1", 10);
const checkpointPath = process.argv[7] ?? "checkpoints/dqn-latest.json";
const resumePath = process.argv[8] && process.argv[8] !== "none" ? process.argv[8] : null;
const epsilonStart = Number.parseFloat(process.argv[9] ?? "1");
const epsilonMin = Number.parseFloat(process.argv[10] ?? "0.08");
const epsilonDecay = Number.parseFloat(process.argv[11] ?? "0.94");
const learningRate = Number.parseFloat(process.argv[12] ?? "0.0015");
const gamma = Number.parseFloat(process.argv[13] ?? "0.96");
const rng = mulberry32(2026 + baseSeed);
const env = new SolarMaxEnv({ seed: baseSeed, maxEpisodeSteps, difficulty, opponent });
const agent = loadOrCreateAgent();

let epsilon = epsilonStart;
let wins = 0;
let losses = 0;
let draws = 0;
const recentRewards = [];
const recentLosses = [];

console.log(
  [
    `config episodes=${episodes}`,
    `maxEpisodeSteps=${maxEpisodeSteps}`,
    `difficulty=${difficulty}`,
    `opponent=${opponent}`,
    `baseSeed=${baseSeed}`,
    `checkpointPath=${checkpointPath}`,
    `resumePath=${resumePath ?? "none"}`,
    `epsilonStart=${epsilonStart}`,
    `epsilonMin=${epsilonMin}`,
    `epsilonDecay=${epsilonDecay}`,
    `learningRate=${learningRate}`,
    `gamma=${gamma}`
  ].join(" ")
);

for (let episode = 1; episode <= episodes; episode += 1) {
  let observation = env.reset(baseSeed + episode);
  let done = false;
  let episodeReward = 0;
  let lastInfo = null;
  let lossSum = 0;
  let lossCount = 0;

  while (!done) {
    const action = agent.act(observation, epsilon);
    const result = env.step(action ?? -1);
    agent.remember({
      observation,
      action: action ?? -1,
      reward: result.reward,
      nextObservation: result.observation,
      done: result.done
    });

    const loss = agent.trainStep();
    if (loss !== null && Number.isFinite(loss)) {
      lossSum += loss;
      lossCount += 1;
      recentLosses.push(loss);
      if (recentLosses.length > 25) {
        recentLosses.shift();
      }
    }

    observation = result.observation;
    episodeReward += result.reward;
    done = result.done;
    lastInfo = result.info;
  }

  if (lastInfo?.winner === 1) {
    wins += 1;
  } else if (lastInfo?.winner === 2) {
    losses += 1;
  } else {
    draws += 1;
  }

  recentRewards.push(episodeReward);
  if (recentRewards.length > 25) {
    recentRewards.shift();
  }

  epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
  const averageRecentReward = average(recentRewards);
  const averageLoss = lossCount > 0 ? lossSum / lossCount : average(recentLosses);

  console.log(
    [
      `episode=${episode}`,
      `reward=${episodeReward.toFixed(2)}`,
      `recentReward=${averageRecentReward.toFixed(2)}`,
      `steps=${lastInfo?.steps ?? 0}`,
      `epsilon=${epsilon.toFixed(3)}`,
      `loss=${Number.isFinite(averageLoss) ? averageLoss.toFixed(4) : "n/a"}`,
      `opponent=${lastInfo?.opponent ?? "n/a"}`,
      `difficulty=${lastInfo?.difficulty ?? difficulty}`,
      `wins=${wins}`,
      `losses=${losses}`,
      `draws=${draws}`
    ].join(" ")
  );
}

console.log(
  JSON.stringify(
    {
      episodes,
      wins,
      losses,
      draws,
      winRate: wins / episodes,
      finalEpsilon: epsilon,
      difficulty,
      opponent,
      baseSeed
    },
    null,
    2
  )
);

saveCheckpoint(agent, checkpointPath, {
  episodes,
  maxEpisodeSteps,
  difficulty,
  opponent,
  baseSeed,
  resumePath,
  wins,
  losses,
  draws,
  winRate: wins / episodes,
  finalEpsilon: epsilon,
  epsilonStart,
  epsilonMin,
  epsilonDecay,
  learningRate
});
console.log(`saved checkpoint ${checkpointPath}`);

function loadOrCreateAgent() {
  if (resumePath && existsSync(resumePath)) {
    const checkpoint = JSON.parse(readFileSync(resumePath, "utf8"));
    return DqnAgent.fromJSON(checkpoint, {
      rng,
      hyperparameters: { learningRate, gamma }
    });
  }

  if (resumePath) {
    console.warn(`resume checkpoint not found, starting fresh: ${resumePath}`);
  }

  return new DqnAgent({
    rng,
    hiddenSize: 64,
    replayCapacity: 6000,
    minReplaySize: 96,
    batchSize: 32,
    targetUpdateEvery: 100,
    learningRate,
    gamma
  });
}

function average(values) {
  if (values.length === 0) {
    return Number.NaN;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function saveCheckpoint(agent, path, metadata) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(agent.toJSON(metadata), null, 2)}\n`, "utf8");
}
