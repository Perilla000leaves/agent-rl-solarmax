import assert from "node:assert/strict";
import test from "node:test";
import {
  DIFFICULTIES,
  OPPONENT_POLICIES,
  PLANET_COUNT,
  SolarMaxEnv,
  chooseOpponentMove
} from "../src/rl-env.js";

test("different seeds keep observation shape but change map features", () => {
  const env = new SolarMaxEnv({ difficulty: "normal", opponent: "aggressive" });
  const first = env.reset(101);
  const firstPlanetFeatures = first.planets.slice();
  const second = env.reset(102);

  assert.equal(first.planets.length, second.planets.length);
  assert.equal(first.fleets.length, second.fleets.length);
  assert.equal(first.global.length, second.global.length);
  assert.equal(first.actionMask.length, second.actionMask.length);
  assert.equal(env.state.planets.length, PLANET_COUNT);
  assert.notDeepEqual(firstPlanetFeatures, second.planets);
});

test("opponent policies return legal enemy moves when a move is available", () => {
  for (const policy of OPPONENT_POLICIES) {
    const env = new SolarMaxEnv({ seed: 9, difficulty: "normal", opponent: policy });
    const move = chooseOpponentMove(env.state, policy, { sendRatio: 0.5, rng: () => 0.4 });
    assert.ok(move, `${policy} should return a move`);

    const source = env.state.planets.find((planet) => planet.id === move.sourceId);
    const target = env.state.planets.find((planet) => planet.id === move.targetId);
    assert.equal(source.owner, 2);
    assert.notEqual(target.owner, 2);
    assert.notEqual(source.id, target.id);
    assert.ok(move.fraction > 0 && move.fraction <= 1);
  }
});

test("each difficulty can construct, reset, and step", () => {
  for (const difficulty of Object.keys(DIFFICULTIES)) {
    const env = new SolarMaxEnv({ seed: 12, difficulty, opponent: "random", maxEpisodeSteps: 20 });
    const observation = env.reset(12);
    const action = observation.actionMask.findIndex((value) => value === 1);
    const result = env.step(action);

    assert.equal(result.observation.planets.length, PLANET_COUNT * 10);
    assert.equal(result.info.difficulty, difficulty);
    assert.ok(OPPONENT_POLICIES.includes(result.info.opponent));
  }
});
