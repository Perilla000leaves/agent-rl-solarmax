import { SolarMaxEnv } from "../src/rl-env.js";
import { OWNER } from "../src/game-core.js";
import { chooseHeuristicAction } from "../src/heuristic-policy.js";

const episodes = Number.parseInt(process.argv[2] ?? "20", 10);
const difficulty = process.argv[3] ?? "easy";
const opponent = process.argv[4] ?? "expansion";
const baseSeed = Number.parseInt(process.argv[5] ?? "500", 10);

let wins = 0;
let losses = 0;
let draws = 0;
let totalReward = 0;
let totalSteps = 0;

for (let episode = 1; episode <= episodes; episode += 1) {
  const env = new SolarMaxEnv({ seed: baseSeed, maxEpisodeSteps: 300, difficulty, opponent });
  let done = false;
  let result = null;
  let reward = 0;

  env.reset(baseSeed + episode);
  while (!done) {
    const action = chooseHeuristicAction(env.state);
    result = env.step(action);
    done = result.done;
    reward += result.reward;
  }

  totalReward += reward;
  totalSteps += result.info.steps;
  if (result.info.winner === OWNER.player) {
    wins += 1;
  } else if (result.info.winner === OWNER.enemy) {
    losses += 1;
  } else {
    draws += 1;
  }
}

console.log(
  JSON.stringify(
    {
      difficulty,
      opponent,
      episodes,
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
