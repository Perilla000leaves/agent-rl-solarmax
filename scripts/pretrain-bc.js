import { DqnAgent, flattenObservation, maskedArgMax } from "../src/dqn-agent.js";
import { SolarMaxEnv } from "../src/rl-env.js";
import { chooseHeuristicAction } from "../src/heuristic-policy.js";
import { OWNER } from "../src/game-core.js";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const episodesPerDifficulty = Number.parseInt(process.argv[2] ?? "150", 10);
const difficulties = (process.argv[3] ?? "easy,normal")
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);
const opponent = process.argv[4] ?? "random";
const baseSeed = Number.parseInt(process.argv[5] ?? "101", 10);
const checkpointPath = process.argv[6] ?? "checkpoints/dqn-bc.json";
const epochs = Number.parseInt(process.argv[7] ?? "4", 10);
const bcLearningRate = Number.parseFloat(process.argv[8] ?? "0.002");
const hiddenSize = 96;
const rng = mulberry32(2026 + baseSeed);

console.log(
  [
    `episodesPerDifficulty=${episodesPerDifficulty}`,
    `difficulties=${difficulties.join("|")}`,
    `opponent=${opponent}`,
    `baseSeed=${baseSeed}`,
    `checkpointPath=${checkpointPath}`,
    `epochs=${epochs}`,
    `bcLearningRate=${bcLearningRate}`,
    `hiddenSize=${hiddenSize}`
  ].join(" ")
);

// 1) 用启发式策略生成示范数据
const samples = [];
const collectionStats = [];

difficulties.forEach((difficulty, difficultyIndex) => {
  const env = new SolarMaxEnv({ seed: baseSeed, maxEpisodeSteps: 300, difficulty, opponent });
  let wins = 0;
  let losses = 0;
  let draws = 0;

  for (let episode = 1; episode <= episodesPerDifficulty; episode += 1) {
    const seed = baseSeed + difficultyIndex * 1000 + episode;
    env.reset(seed);
    let done = false;
    let result = null;

    while (!done) {
      const observation = env.observe();
      const action = chooseHeuristicAction(env.state);
      if (action >= 0 && observation.actionMask[action] === 1) {
        samples.push({
          state: flattenObservation(observation),
          action,
          mask: observation.actionMask.slice()
        });
      }
      result = env.step(action);
      done = result.done;
    }

    if (result.info.winner === OWNER.player) {
      wins += 1;
    } else if (result.info.winner === OWNER.enemy) {
      losses += 1;
    } else {
      draws += 1;
    }
  }

  collectionStats.push({ difficulty, wins, losses, draws });
  console.log(
    `collected difficulty=${difficulty} episodes=${episodesPerDifficulty} wins=${wins} losses=${losses} draws=${draws}`
  );
});
console.log(`samples=${samples.length}`);

// 2) 行为克隆训练：把 expert 动作推成合法动作里的最高分
const agent = new DqnAgent({
  rng,
  hiddenSize,
  replayCapacity: 6000,
  minReplaySize: 96,
  batchSize: 32,
  targetUpdateEvery: 100,
  learningRate: bcLearningRate,
  gamma: 0.99,
  maxTargetValue: 75
});

for (let epoch = 1; epoch <= epochs; epoch += 1) {
  shuffle(samples, rng);
  let lossSum = 0;
  let lossCount = 0;
  let correct = 0;

  for (const sample of samples) {
    const loss = agent.imitateStep(sample.state, sample.action, sample.mask);
    if (loss !== null && Number.isFinite(loss)) {
      lossSum += loss;
      lossCount += 1;
    }
    const qValues = agent.online.predict(sample.state);
    if (maskedArgMax(qValues, sample.mask) === sample.action) {
      correct += 1;
    }
  }

  console.log(
    `epoch=${epoch} loss=${(lossSum / Math.max(1, lossCount)).toFixed(4)} accuracy=${(correct / samples.length).toFixed(4)}`
  );
}

// 3) 保存 checkpoint
mkdirSync(dirname(checkpointPath), { recursive: true });
writeFileSync(
  checkpointPath,
  `${JSON.stringify(
    agent.toJSON({
      pretrain: true,
      difficulties,
      episodesPerDifficulty,
      opponent,
      baseSeed,
      samples: samples.length,
      epochs,
      bcLearningRate,
      collectionStats
    }),
    null,
    2
  )}\n`,
  "utf8"
);
console.log(`saved checkpoint ${checkpointPath}`);

function shuffle(items, random) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
