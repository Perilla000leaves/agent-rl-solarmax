import { DqnAgent, flattenObservation, ReplayBuffer } from "../src/dqn-agent.js";
import { SolarMaxEnv } from "../src/rl-env.js";
import { chooseHeuristicAction } from "../src/heuristic-policy.js";
import { OWNER } from "../src/game-core.js";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

// 从 BC 热启动的温和微调：
// - 重新注入启发式示范样本，让 trainStep 的 DQfD 正则生效（上次微调崩掉的主因是 demo 丢失）
// - 低学习率 + 低探索，避免 TD 更新把模仿策略冲垮
// - 支持多难度交替训练（如 "easy,normal"），两难度的示范都会注入，避免单难度微调导致另一难度遗忘
// - 周期性 held-out 贪心评估，随时观察是否退化
const episodes = Number.parseInt(process.argv[2] ?? "1000", 10);
const difficulties = (process.argv[3] ?? "normal")
  .split(",")
  .map((name) => name.trim())
  .filter(Boolean);
const opponent = process.argv[4] ?? "random";
const baseSeed = Number.parseInt(process.argv[5] ?? "8001", 10);
const bcPath = process.argv[6] ?? "checkpoints/dqn-bc.json";
const outPath = process.argv[7] ?? "checkpoints/dqn-bc-tuned.json";
const demoEpisodes = Number.parseInt(process.argv[8] ?? "60", 10);
const learningRate = Number.parseFloat(process.argv[9] ?? "0.0002");

const maxEpisodeSteps = 300;
const epsilonStart = 0.08;
const epsilonMin = 0.02;
const epsilonDecay = 0.995;
const evalEvery = 100;
const evalEpisodes = 20;
const evalBaseSeed = 7000;
const demoBaseSeed = 50000;
const rng = mulberry32(2026 + baseSeed);

if (!existsSync(bcPath)) {
  console.error(`BC checkpoint not found: ${bcPath}`);
  process.exit(1);
}

const checkpoint = JSON.parse(readFileSync(bcPath, "utf8"));
const agent = DqnAgent.fromJSON(checkpoint, { rng, hyperparameters: { learningRate } });
// 多难度微调的数据更多，扩大回放缓冲，避免旧数据被过早挤出
agent.replay = new ReplayBuffer(Math.max(agent.replayCapacity, 20000), rng);

console.log(
  [
    `config episodes=${episodes}`,
    `difficulties=${difficulties.join("|")}`,
    `opponent=${opponent}`,
    `baseSeed=${baseSeed}`,
    `bcPath=${bcPath}`,
    `outPath=${outPath}`,
    `demoEpisodes=${demoEpisodes}`,
    `learningRate=${learningRate}`,
    `gamma=${agent.gamma}`,
    `hiddenSize=${agent.hiddenSize}`,
    `epsilonStart=${epsilonStart}`,
    `epsilonMin=${epsilonMin}`,
    `epsilonDecay=${epsilonDecay}`
  ].join(" ")
);

// 1) 重新注入示范数据：每个训练难度都用启发式策略采集
let demoAdded = 0;
let demoWins = 0;
let demoLosses = 0;

difficulties.forEach((demoDifficulty, difficultyIndex) => {
  const demoEnv = new SolarMaxEnv({ seed: demoBaseSeed, maxEpisodeSteps, difficulty: demoDifficulty, opponent });

  for (let episode = 1; episode <= demoEpisodes; episode += 1) {
    demoEnv.reset(demoBaseSeed + difficultyIndex * 1000 + episode);
    let done = false;
    let result = null;

    while (!done) {
      const observation = demoEnv.observe();
      const action = chooseHeuristicAction(demoEnv.state);
      if (action >= 0 && observation.actionMask[action] === 1) {
        if (agent.addDemo(flattenObservation(observation), action, observation.actionMask)) {
          demoAdded += 1;
        }
      }
      result = demoEnv.step(action);
      done = result.done;
    }

    if (result.info.winner === OWNER.player) {
      demoWins += 1;
    } else if (result.info.winner === OWNER.enemy) {
      demoLosses += 1;
    }
  }
});
console.log(`demo injection episodesPerDifficulty=${demoEpisodes} samples=${demoAdded} heuristicWins=${demoWins} heuristicLosses=${demoLosses}`);

// 2) 微调前的贪心基线（held-out 种子，与训练/采数据种子完全错开）
const baseline = {};
for (const evalDifficulty of difficulties) {
  baseline[evalDifficulty] = evaluateGreedy(evalDifficulty, opponent, evalEpisodes, evalBaseSeed);
}
console.log(
  `baseline eval ${difficulties.map((name) => `${name}=${baseline[name].wins}/${baseline[name].episodes}`).join(" ")}`
);

const evalHistory = [
  { episode: 0, results: baseline }
];

// 3) 温和微调：多难度按 episode 交替
const envs = new Map(
  difficulties.map((name) => [name, new SolarMaxEnv({ seed: baseSeed, maxEpisodeSteps, difficulty: name, opponent })])
);
let epsilon = epsilonStart;
let wins = 0;
let losses = 0;
let draws = 0;
const recentRewards = [];

for (let episode = 1; episode <= episodes; episode += 1) {
  const trainDifficulty = difficulties[(episode - 1) % difficulties.length];
  const env = envs.get(trainDifficulty);
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
    }
    observation = result.observation;
    episodeReward += result.reward;
    done = result.done;
    lastInfo = result.info;
  }

  if (lastInfo?.winner === OWNER.player) {
    wins += 1;
  } else if (lastInfo?.winner === OWNER.enemy) {
    losses += 1;
  } else {
    draws += 1;
  }

  recentRewards.push(episodeReward);
  if (recentRewards.length > 50) {
    recentRewards.shift();
  }
  epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);

  if (episode % evalEvery === 0 || episode === episodes) {
    const results = {};
    for (const evalDifficulty of difficulties) {
      results[evalDifficulty] = evaluateGreedy(evalDifficulty, opponent, evalEpisodes, evalBaseSeed);
    }
    evalHistory.push({ episode, results });
    console.log(
      [
        `eval episode=${episode}`,
        ...difficulties.map((name) => `${name}=${results[name].wins}/${results[name].episodes}`),
        `epsilon=${epsilon.toFixed(4)}`,
        `avgRecentReward=${average(recentRewards).toFixed(2)}`,
        `trainWins=${wins}`,
        `trainLoss=${lossSum > 0 ? (lossSum / Math.max(1, lossCount)).toFixed(4) : "n/a"}`
      ].join(" ")
    );
  } else if (episode % 25 === 0) {
    console.log(
      [
        `episode=${episode}`,
        `epsilon=${epsilon.toFixed(4)}`,
        `avgRecentReward=${average(recentRewards).toFixed(2)}`,
        `wins=${wins}`,
        `losses=${losses}`,
        `draws=${draws}`
      ].join(" ")
    );
  }
}

const finalResults = evalHistory[evalHistory.length - 1].results;
console.log(
  JSON.stringify(
    {
      episodes,
      difficulties,
      opponent,
      baseSeed,
      demoEpisodes,
      demoAdded,
      learningRate,
      finalResults,
      trainWins: wins,
      trainLosses: losses,
      trainDraws: draws,
      finalEpsilon: epsilon
    },
    null,
    2
  )
);

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(
  outPath,
  `${JSON.stringify(
    agent.toJSON({
      finetune: true,
      resumePath: bcPath,
      difficulties,
      opponent,
      baseSeed,
      episodes,
      demoEpisodes,
      demoAdded,
      demoWins,
      demoLosses,
      learningRate,
      epsilonStart,
      epsilonMin,
      epsilonDecay,
      trainWins: wins,
      trainLosses: losses,
      trainDraws: draws,
      finalEpsilon: epsilon,
      evalHistory
    }),
    null,
    2
  )}\n`,
  "utf8"
);
console.log(`saved checkpoint ${outPath}`);

function evaluateGreedy(evalDifficulty, evalOpponent, count, evalSeed) {
  const evalEnv = new SolarMaxEnv({
    seed: evalSeed,
    maxEpisodeSteps,
    difficulty: evalDifficulty,
    opponent: evalOpponent
  });
  let evalWins = 0;
  let evalLosses = 0;
  let evalDraws = 0;

  for (let episode = 1; episode <= count; episode += 1) {
    let observation = evalEnv.reset(evalSeed + episode);
    let done = false;
    let lastInfo = null;

    while (!done) {
      const action = agent.act(observation, 0);
      const result = evalEnv.step(action ?? -1);
      observation = result.observation;
      done = result.done;
      lastInfo = result.info;
    }

    if (lastInfo?.winner === OWNER.player) {
      evalWins += 1;
    } else if (lastInfo?.winner === OWNER.enemy) {
      evalLosses += 1;
    } else {
      evalDraws += 1;
    }
  }

  return { wins: evalWins, losses: evalLosses, draws: evalDraws, episodes: count };
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
