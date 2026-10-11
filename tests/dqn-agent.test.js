import assert from "node:assert/strict";
import test from "node:test";
import { DqnAgent, flattenObservation, maskedArgMax, maskedMax } from "../src/dqn-agent.js";
import { ACTION_COUNT, SolarMaxEnv } from "../src/rl-env.js";

test("flattenObservation creates a fixed-length vector", () => {
  const env = new SolarMaxEnv({ seed: 1 });
  const flat = flattenObservation(env.reset(1));
  assert.equal(flat.length, 314);
  assert.ok(flat.every((value) => Number.isFinite(value)));
});

test("masked helpers ignore illegal actions", () => {
  const values = [100, 1, 7, 3];
  const mask = [0, 1, 1, 0];
  assert.equal(maskedArgMax(values, mask), 2);
  assert.equal(maskedMax(values, mask), 7);
  assert.equal(maskedArgMax(values, [0, 0, 0, 0]), null);
  assert.equal(maskedMax(values, [0, 0, 0, 0]), 0);
});

test("DQN agent selects legal actions and can train from replay", () => {
  const rng = fixedRng();
  const env = new SolarMaxEnv({ seed: 3, maxEpisodeSteps: 20 });
  const agent = new DqnAgent({
    rng,
    hiddenSize: 16,
    minReplaySize: 4,
    batchSize: 4,
    targetUpdateEvery: 2
  });

  let observation = env.reset(3);
  for (let i = 0; i < 8; i += 1) {
    const action = agent.act(observation, 0.5);
    assert.ok(action === null || (action >= 0 && action < ACTION_COUNT));
    if (action !== null) {
      assert.equal(observation.actionMask[action], 1);
    }

    const result = env.step(action ?? -1);
    agent.remember({
      observation,
      action: action ?? -1,
      reward: result.reward,
      nextObservation: result.observation,
      done: result.done
    });
    observation = result.observation;
  }

  const loss = agent.trainStep();
  assert.ok(loss === null || Number.isFinite(loss));
});

test("DQN agent does not store sentinel actions in replay", () => {
  const env = new SolarMaxEnv({ seed: 4 });
  const agent = new DqnAgent({ hiddenSize: 8 });
  const observation = env.reset(4);
  const result = env.step(-1);

  const stored = agent.remember({
    observation,
    action: -1,
    reward: result.reward,
    nextObservation: result.observation,
    done: result.done
  });

  assert.equal(stored, false);
  assert.equal(agent.replay.size, 0);
});

test("DQN checkpoint round-trips and can select a legal action", () => {
  const env = new SolarMaxEnv({ seed: 6 });
  const observation = env.reset(6);
  const agent = new DqnAgent({ hiddenSize: 12, rng: fixedRng() });
  const checkpoint = agent.toJSON({ note: "round trip" });
  const loaded = DqnAgent.fromJSON(checkpoint, { rng: fixedRng() });
  const action = loaded.act(observation, 0);

  assert.equal(checkpoint.format, "agent-rl-dqn-checkpoint-v1");
  assert.equal(checkpoint.metadata.note, "round trip");
  assert.ok(action === null || observation.actionMask[action] === 1);
});

test("DQN agent can learn to imitate an expert action", () => {
  const rng = fixedRng();
  const env = new SolarMaxEnv({ seed: 8, maxEpisodeSteps: 30 });
  const observation = env.reset(8);
  const agent = new DqnAgent({ rng, hiddenSize: 32, learningRate: 0.005 });
  const state = flattenObservation(observation);
  const expertAction = observation.actionMask.indexOf(1);
  assert.ok(expertAction >= 0);

  for (let i = 0; i < 80; i += 1) {
    agent.imitateStep(state, expertAction, observation.actionMask);
  }

  const qValues = agent.online.predict(state);
  assert.equal(maskedArgMax(qValues, observation.actionMask), expertAction);
});

function fixedRng() {
  let seed = 99;
  return function next() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}
