import assert from "node:assert/strict";
import test from "node:test";
import { formatRewardBreakdown, formatStateSummary, summarizeObservation, winnerLabel } from "../src/rl-inspector.js";
import { SolarMaxEnv } from "../src/rl-env.js";

test("summarizeObservation returns compact state metrics", () => {
  const env = new SolarMaxEnv({ seed: 1 });
  const observation = env.reset(1);
  const summary = summarizeObservation(observation);

  assert.equal(summary.playerPlanets, 1);
  assert.equal(summary.enemyPlanets, 1);
  assert.equal(summary.neutralPlanets, 6);
  assert.ok(summary.legalActions > 0);
  assert.match(formatStateSummary(summary), /星球 P\/E\/N/);
});

test("step info includes reward breakdown and metrics", () => {
  const env = new SolarMaxEnv({ seed: 2 });
  const observation = env.reset(2);
  const action = observation.actionMask.findIndex((value) => value === 1);
  const result = env.step(action);

  assert.equal(typeof result.info.rewardBreakdown.total, "number");
  assert.equal(result.reward, result.info.rewardBreakdown.total);
  assert.equal(result.info.afterMetrics.fleetCount, env.state.fleets.length);
  assert.ok(result.info.legalActionCount > 0);
  assert.match(formatRewardBreakdown(result.info.rewardBreakdown), /星球/);
  assert.equal(winnerLabel(null), "none");
});
