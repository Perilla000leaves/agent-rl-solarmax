export function summarizeObservation(observation) {
  const global = observation.global;
  return {
    playerPlanets: Math.round(global[1] * 8),
    enemyPlanets: Math.round(global[2] * 8),
    neutralPlanets: Math.round(global[3] * 8),
    playerGrowth: global[4] * 24,
    enemyGrowth: global[5] * 24,
    playerShips: global[6] * 220,
    enemyShips: global[7] * 220,
    fleetCount: countFleets(observation.fleets),
    legalActions: observation.actionMask.filter((value) => value === 1).length
  };
}

export function formatRewardBreakdown(breakdown) {
  if (!breakdown) {
    return "reward breakdown unavailable";
  }

  return [
    `星球 ${formatSigned(breakdown.planetDelta)}`,
    `扩张 ${formatSigned(breakdown.neutralExpansion)}`,
    `产能 ${formatSigned(breakdown.growthDelta)}`,
    `兵力 ${formatSigned(breakdown.shipDelta)}`,
    `预备 ${formatSigned(breakdown.reserveDelta)}`,
    `空虚 ${formatSigned(breakdown.lowReserve)}`,
    `胜负 ${formatSigned(breakdown.winLoss)}`,
    `非法 ${formatSigned(breakdown.illegal)}`
  ].join(" / ");
}

export function winnerLabel(winner) {
  if (winner === 1) {
    return "player";
  }
  if (winner === 2) {
    return "enemy";
  }
  return "none";
}

export function formatStateSummary(summary) {
  return [
    `星球 P/E/N ${summary.playerPlanets}/${summary.enemyPlanets}/${summary.neutralPlanets}`,
    `兵力 P/E ${summary.playerShips.toFixed(1)}/${summary.enemyShips.toFixed(1)}`,
    `产能 P/E ${summary.playerGrowth.toFixed(1)}/${summary.enemyGrowth.toFixed(1)}`,
    `舰队 ${summary.fleetCount}`,
    `合法动作 ${summary.legalActions}`
  ].join(" | ");
}

function countFleets(fleetFeatures) {
  let count = 0;
  for (let i = 0; i < fleetFeatures.length; i += 7) {
    if (fleetFeatures[i] === 1 || fleetFeatures[i + 1] === 1) {
      count += 1;
    }
  }
  return count;
}

function formatSigned(value) {
  const safeValue = Number.isFinite(value) ? value : 0;
  return `${safeValue >= 0 ? "+" : ""}${safeValue.toFixed(2)}`;
}
