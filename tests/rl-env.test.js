import assert from "node:assert/strict";
import test from "node:test";
import {
  ACTION_COUNT,
  PLANET_COUNT,
  SEND_RATIOS,
  SolarMaxEnv,
  decodeAction,
  encodeAction,
  sampleLegalAction
} from "../src/rl-env.js";

test("action encoding round-trips source, target, and ratio", () => {
  const action = encodeAction(3, 6, 2);
  assert.equal(action, (3 * PLANET_COUNT + 6) * SEND_RATIOS.length + 2);
  assert.deepEqual(decodeAction(action), {
    sourceIndex: 3,
    targetIndex: 6,
    ratioIndex: 2,
    sendRatio: 0.75
  });
});

test("reset returns fixed-size normalized observation and mask", () => {
  const env = new SolarMaxEnv({ seed: 5 });
  const observation = env.reset(5);

  assert.equal(observation.planets.length, PLANET_COUNT * 10);
  assert.equal(observation.fleets.length, 32 * 7);
  assert.equal(observation.global.length, 10);
  assert.equal(observation.actionMask.length, ACTION_COUNT);
  assert.ok(observation.planets.every((value) => value >= 0 && value <= 1));
  assert.ok(observation.global.every((value) => value >= 0 && value <= 1));
});

test("action mask only allows player-owned sources and non-self targets", () => {
  const env = new SolarMaxEnv({ seed: 5 });
  const mask = env.legalActions();

  assert.equal(mask[encodeAction(0, 1, 1)], 1);
  assert.equal(mask[encodeAction(0, 0, 1)], 0);
  assert.equal(mask[encodeAction(1, 0, 1)], 0);
});

test("step handles legal and illegal actions", () => {
  const env = new SolarMaxEnv({ seed: 5 });
  const legalAction = sampleLegalAction(env.legalActions(), () => 0);
  const legalResult = env.step(legalAction);

  assert.equal(legalResult.info.legal, true);
  assert.equal(legalResult.info.launched, true);
  assert.equal(legalResult.observation.actionMask.length, ACTION_COUNT);

  const illegalResult = env.step(encodeAction(1, 0, 0));
  assert.equal(illegalResult.info.legal, false);
  assert.ok(illegalResult.reward < legalResult.reward + 1);
});
