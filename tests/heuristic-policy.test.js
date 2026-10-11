import assert from "node:assert/strict";
import test from "node:test";
import { SolarMaxEnv } from "../src/rl-env.js";
import { chooseHeuristicAction } from "../src/heuristic-policy.js";

test("heuristic policy picks a legal action from the opening", () => {
  const env = new SolarMaxEnv({ seed: 11, maxEpisodeSteps: 300 });
  const observation = env.reset(11);
  const action = chooseHeuristicAction(env.state);
  assert.ok(action >= 0);
  assert.equal(observation.actionMask[action], 1);
});

test("heuristic policy only returns legal actions or -1", () => {
  const env = new SolarMaxEnv({ seed: 12, maxEpisodeSteps: 300 });
  let done = false;
  let result = null;

  env.reset(12);
  while (!done) {
    const observation = env.observe();
    const action = chooseHeuristicAction(env.state);
    assert.ok(action === -1 || observation.actionMask[action] === 1);
    result = env.step(action);
    done = result.done;
  }

  assert.ok(result.info.steps > 0);
});
