import { encodeAction } from "./rl-env.js";
import { OWNER } from "./game-core.js";

// 简单规则策略：优先防守受威胁星球，其次由最强星球攻击最划算的目标。
// 返回索引发作所需的 action index；无法行动时返回 -1。
export function chooseHeuristicAction(state) {
  const index = (planet) => state.planets.indexOf(planet);
  const myPlanets = state.planets.filter((planet) => planet.owner === OWNER.player);
  const enemyPlanets = state.planets.filter((planet) => planet.owner === OWNER.enemy);
  const neutralPlanets = state.planets.filter((planet) => planet.owner === OWNER.neutral);

  const incoming = {};
  for (const fleet of state.fleets) {
    if (fleet.owner === OWNER.enemy) {
      incoming[fleet.targetId] = (incoming[fleet.targetId] ?? 0) + fleet.ships;
    }
  }

  // 1) Defense first: reinforce a threatened planet that cannot hold alone
  for (const planet of myPlanets) {
    const threat = incoming[planet.id] ?? 0;
    if (threat > planet.ships * 0.9) {
      const helper = myPlanets
        .filter((source) => source.id !== planet.id && source.ships >= 10 && (incoming[source.id] ?? 0) < source.ships * 0.5)
        .sort((a, b) => b.ships - a.ships)[0];
      if (helper) {
        const needed = threat + 6 - planet.ships;
        const ratio = helper.ships * 0.5 >= needed ? 1 : 2;
        return encodeAction(index(helper), index(planet), ratio);
      }
    }
  }

  // 2) Attack: strongest planet strikes the best affordable target (enemy first)
  const sortedMine = [...myPlanets].sort((a, b) => b.ships - a.ships);
  const source = sortedMine[0];
  if (source && source.ships >= 16) {
    const candidates = [...enemyPlanets, ...neutralPlanets]
      .map((target) => ({
        target,
        score: target.ships + Math.hypot(target.x - source.x, target.y - source.y) / 18 - target.growth * 4
      }))
      .sort((a, b) => a.score - b.score);
    for (const { target } of candidates) {
      if (source.ships * 0.75 > target.ships * 0.85 + 8) {
        return encodeAction(index(source), index(target), 2);
      }
    }
  }

  // 3) Expand: any planet sends 0.75 to the weakest affordable neutral
  for (const src of sortedMine) {
    if (src.ships < 10) {
      continue;
    }
    const target = [...neutralPlanets].sort((a, b) => a.ships - b.ships)[0];
    if (target && src.ships * 0.75 > target.ships * 0.85 + 8) {
      return encodeAction(index(src), index(target), 2);
    }
  }

  // 4) Fallback: reinforce the second strongest from the strongest
  if (sortedMine.length >= 2 && sortedMine[0].ships >= 8) {
    return encodeAction(index(sortedMine[0]), index(sortedMine[1]), 0);
  }

  // 5) Last resort: strongest attacks the weakest enemy or neutral anyway
  if (source && source.ships >= 3 && enemyPlanets.length + neutralPlanets.length > 0) {
    const target = [...enemyPlanets, ...neutralPlanets].sort((a, b) => a.ships - b.ships)[0];
    return encodeAction(index(source), index(target), 2);
  }

  return -1;
}
