import assert from "node:assert/strict";
import test from "node:test";
import { actionIntent, chooseDemoDecision, describeAction } from "../src/demo-policy.js";
import { SolarMaxEnv, encodeAction } from "../src/rl-env.js";

test("describeAction and actionIntent explain encoded actions", () => {
  const env = new SolarMaxEnv({ seed: 1 });
  env.reset(1);
  const action = encodeAction(0, 2, 1);

  assert.equal(describeAction(action, env.state.planets), "p-home -> north，派出 50%");
  assert.deepEqual(actionIntent(action, env.state.planets), {
    sourceId: "p-home",
    targetId: "north",
    ratio: 0.5,
    label: "p-home -> north，派出 50%"
  });
});

test("describeAction handles missing actions", () => {
  const env = new SolarMaxEnv({ seed: 1 });
  assert.equal(describeAction(null, env.state.planets), "无合法动作");
  assert.equal(actionIntent(-1, env.state.planets), null);
});

test("chooseDemoDecision reports policy and exploration", () => {
  const env = new SolarMaxEnv({ seed: 1 });
  const decision = chooseDemoDecision(null, env.observe(), 1);
  assert.equal(decision.policy, "random legal baseline");
  assert.equal(decision.explored, true);
  assert.equal(env.observe().actionMask[decision.action], 1);
});
