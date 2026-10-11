import { SolarMaxEnv, sampleLegalAction } from "../src/rl-env.js";

const episodes = Number.parseInt(process.argv[2] ?? "25", 10);
const maxEpisodeSteps = Number.parseInt(process.argv[3] ?? "600", 10);
const rng = mulberry32(1234);

let wins = 0;
let losses = 0;
let draws = 0;
let totalReward = 0;
let totalSteps = 0;

for (let episode = 0; episode < episodes; episode += 1) {
  const env = new SolarMaxEnv({ seed: 100 + episode, maxEpisodeSteps });
  let observation = env.reset(100 + episode);
  let episodeReward = 0;
  let result = null;

  while (!result?.done) {
    const action = sampleLegalAction(observation.actionMask, rng);
    result = env.step(action ?? -1);
    observation = result.observation;
    episodeReward += result.reward;
  }

  totalReward += episodeReward;
  totalSteps += result.info.steps;
  if (result.info.winner === 1) {
    wins += 1;
  } else if (result.info.winner === 2) {
    losses += 1;
  } else {
    draws += 1;
  }
}

const summary = {
  episodes,
  wins,
  losses,
  draws,
  winRate: wins / episodes,
  averageSteps: totalSteps / episodes,
  averageReward: totalReward / episodes
};

console.log(JSON.stringify(summary, null, 2));

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
