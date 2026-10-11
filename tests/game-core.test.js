import assert from "node:assert/strict";
import test from "node:test";
import {
  OWNER,
  chooseEnemyMove,
  createInitialState,
  getPlanet,
  launchFleet,
  updateState
} from "../src/game-core.js";

test("owned planets produce ships over time", () => {
  const state = createInitialState();
  const home = getPlanet(state, "p-home");
  const before = home.ships;

  updateState(state, 2);

  assert.equal(Math.floor(home.ships), Math.floor(before + home.growth * 2));
});

test("launching a fleet removes ships from source and captures weak target", () => {
  const state = createInitialState();
  const source = getPlanet(state, "p-home");
  const target = getPlanet(state, "west");
  source.ships = 80;
  target.ships = 10;

  const fleet = launchFleet(state, source.id, target.id, OWNER.player, 0.5);

  assert.ok(fleet);
  assert.equal(source.ships, 40);
  updateState(state, fleet.duration + 0.01);
  assert.equal(target.owner, OWNER.player);
  assert.equal(target.ships, 30);
});

test("enemy move chooses a legal enemy source and non-enemy target", () => {
  const state = createInitialState();
  const enemy = getPlanet(state, "e-home");
  enemy.ships = 120;

  const move = chooseEnemyMove(state);

  assert.ok(move);
  assert.equal(getPlanet(state, move.sourceId).owner, OWNER.enemy);
  assert.notEqual(getPlanet(state, move.targetId).owner, OWNER.enemy);
});
