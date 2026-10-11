import { decodeAction, sampleLegalAction } from "./rl-env.js";

export function chooseDemoAction(agent, observation, epsilon = 0.35) {
  return chooseDemoDecision(agent, observation, epsilon).action;
}

export function chooseDemoDecision(agent, observation, epsilon = 0.35) {
  if (!agent) {
    return {
      action: sampleLegalAction(observation.actionMask),
      explored: true,
      policy: "random legal baseline"
    };
  }

  const rng = agent.rng ?? Math.random;
  const explored = rng() < epsilon;
  return {
    action: explored ? sampleLegalAction(observation.actionMask, rng) : agent.act(observation, 0),
    explored,
    policy: "demo DQN baseline"
  };
}

export function describeAction(action, planets) {
  if (!Number.isInteger(action) || action < 0) {
    return "无合法动作";
  }

  const decoded = decodeAction(action);
  const source = planets[decoded.sourceIndex];
  const target = planets[decoded.targetIndex];
  if (!source || !target) {
    return "动作超出地图范围";
  }

  return `${source.id} -> ${target.id}，派出 ${Math.round(decoded.sendRatio * 100)}%`;
}

export function actionIntent(action, planets) {
  if (!Number.isInteger(action) || action < 0) {
    return null;
  }

  const decoded = decodeAction(action);
  const source = planets[decoded.sourceIndex];
  const target = planets[decoded.targetIndex];
  if (!source || !target) {
    return null;
  }

  return {
    sourceId: source.id,
    targetId: target.id,
    ratio: decoded.sendRatio,
    label: describeAction(action, planets)
  };
}
