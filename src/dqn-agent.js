import { ACTION_COUNT, sampleLegalAction } from "./rl-env.js";

export const OBSERVATION_SIZE = 80 + 224 + 10;

export class DqnAgent {
  constructor(options = {}) {
    this.inputSize = options.inputSize ?? OBSERVATION_SIZE;
    this.hiddenSize = options.hiddenSize ?? 64;
    this.actionCount = options.actionCount ?? ACTION_COUNT;
    this.gamma = options.gamma ?? 0.96;
    this.learningRate = options.learningRate ?? 0.0015;
    this.batchSize = options.batchSize ?? 32;
    this.replayCapacity = options.replayCapacity ?? 5000;
    this.minReplaySize = options.minReplaySize ?? 128;
    this.targetUpdateEvery = options.targetUpdateEvery ?? 200;
    this.rng = options.rng ?? Math.random;
    this.online = new TinyQNetwork(this.inputSize, this.hiddenSize, this.actionCount, this.rng);
    this.target = this.online.clone();
    this.replay = new ReplayBuffer(this.replayCapacity, this.rng);
    this.trainSteps = 0;
  }

  act(observation, epsilon = 0.1) {
    if (this.rng() < epsilon) {
      return sampleLegalAction(observation.actionMask, this.rng);
    }

    const state = flattenObservation(observation);
    const qValues = this.online.predict(state);
    return maskedArgMax(qValues, observation.actionMask);
  }

  remember(transition) {
    if (!Number.isInteger(transition.action) || transition.action < 0 || transition.action >= this.actionCount) {
      return false;
    }

    this.replay.push({
      state: flattenObservation(transition.observation),
      action: transition.action,
      reward: transition.reward,
      nextState: flattenObservation(transition.nextObservation),
      nextActionMask: transition.nextObservation.actionMask.slice(),
      done: transition.done
    });
    return true;
  }

  trainStep() {
    if (this.replay.size < this.minReplaySize) {
      return null;
    }

    const batch = this.replay.sample(this.batchSize);
    let totalLoss = 0;

    for (const transition of batch) {
      const nextQValues = this.target.predict(transition.nextState);
      const nextBest = transition.done ? 0 : maskedMax(nextQValues, transition.nextActionMask);
      const targetValue = transition.reward + this.gamma * nextBest;
      const loss = this.online.trainOne(transition.state, transition.action, targetValue, this.learningRate);
      totalLoss += loss;
    }

    this.trainSteps += 1;
    if (this.trainSteps % this.targetUpdateEvery === 0) {
      this.updateTarget();
    }

    return totalLoss / batch.length;
  }

  updateTarget() {
    this.target = this.online.clone();
  }

  toJSON(metadata = {}) {
    return {
      format: "agent-rl-dqn-checkpoint-v1",
      createdAt: new Date().toISOString(),
      metadata,
      hyperparameters: {
        inputSize: this.inputSize,
        hiddenSize: this.hiddenSize,
        actionCount: this.actionCount,
        gamma: this.gamma,
        learningRate: this.learningRate,
        batchSize: this.batchSize,
        replayCapacity: this.replayCapacity,
        minReplaySize: this.minReplaySize,
        targetUpdateEvery: this.targetUpdateEvery,
        trainSteps: this.trainSteps
      },
      online: this.online.toJSON()
    };
  }

  static fromJSON(checkpoint, options = {}) {
    if (checkpoint?.format !== "agent-rl-dqn-checkpoint-v1") {
      throw new Error("Unsupported DQN checkpoint format");
    }

    const hyperparameters = { ...checkpoint.hyperparameters, ...options.hyperparameters };
    const agent = new DqnAgent({
      ...hyperparameters,
      rng: options.rng ?? Math.random
    });
    agent.online = TinyQNetwork.fromJSON(checkpoint.online);
    agent.target = agent.online.clone();
    agent.trainSteps = hyperparameters.trainSteps ?? 0;
    return agent;
  }
}

export function flattenObservation(observation) {
  const features = [
    ...observation.planets,
    ...observation.fleets,
    ...observation.global
  ];

  if (features.length !== OBSERVATION_SIZE) {
    throw new Error(`Unexpected observation size ${features.length}; expected ${OBSERVATION_SIZE}`);
  }
  return features;
}

export function maskedArgMax(values, actionMask) {
  let bestAction = null;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (actionMask[i] !== 1) {
      continue;
    }
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestAction = i;
    }
  }
  return bestAction;
}

export function maskedMax(values, actionMask) {
  const action = maskedArgMax(values, actionMask);
  return action === null ? 0 : values[action];
}

export class ReplayBuffer {
  constructor(capacity, rng = Math.random) {
    this.capacity = capacity;
    this.rng = rng;
    this.items = [];
    this.index = 0;
  }

  get size() {
    return this.items.length;
  }

  push(item) {
    if (this.items.length < this.capacity) {
      this.items.push(item);
    } else {
      this.items[this.index] = item;
    }
    this.index = (this.index + 1) % this.capacity;
  }

  sample(batchSize) {
    const batch = [];
    for (let i = 0; i < batchSize; i += 1) {
      batch.push(this.items[Math.floor(this.rng() * this.items.length)]);
    }
    return batch;
  }
}

class TinyQNetwork {
  constructor(inputSize, hiddenSize, outputSize, rng = Math.random) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.w1 = createMatrix(hiddenSize, inputSize, () => randomNormal(rng) * Math.sqrt(2 / inputSize));
    this.b1 = Array(hiddenSize).fill(0);
    this.w2 = createMatrix(outputSize, hiddenSize, () => randomNormal(rng) * Math.sqrt(2 / hiddenSize));
    this.b2 = Array(outputSize).fill(0);
  }

  predict(input) {
    const hidden = this.hiddenActivations(input);
    const output = Array(this.outputSize).fill(0);
    for (let action = 0; action < this.outputSize; action += 1) {
      let value = this.b2[action];
      const weights = this.w2[action];
      for (let j = 0; j < this.hiddenSize; j += 1) {
        value += weights[j] * hidden[j];
      }
      output[action] = value;
    }
    return output;
  }

  trainOne(input, action, targetValue, learningRate) {
    const hidden = this.hiddenActivations(input);
    let prediction = this.b2[action];
    for (let j = 0; j < this.hiddenSize; j += 1) {
      prediction += this.w2[action][j] * hidden[j];
    }

    const error = prediction - targetValue;
    const clippedError = Math.max(-10, Math.min(10, error));
    const outputWeights = this.w2[action].slice();

    for (let j = 0; j < this.hiddenSize; j += 1) {
      this.w2[action][j] -= learningRate * clippedError * hidden[j];
    }
    this.b2[action] -= learningRate * clippedError;

    for (let j = 0; j < this.hiddenSize; j += 1) {
      if (hidden[j] <= 0) {
        continue;
      }
      const gradHidden = clippedError * outputWeights[j];
      const row = this.w1[j];
      for (let i = 0; i < this.inputSize; i += 1) {
        row[i] -= learningRate * gradHidden * input[i];
      }
      this.b1[j] -= learningRate * gradHidden;
    }

    return 0.5 * error * error;
  }

  hiddenActivations(input) {
    const hidden = Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j += 1) {
      let value = this.b1[j];
      const weights = this.w1[j];
      for (let i = 0; i < this.inputSize; i += 1) {
        value += weights[i] * input[i];
      }
      hidden[j] = Math.max(0, value);
    }
    return hidden;
  }

  clone() {
    const copy = Object.create(TinyQNetwork.prototype);
    copy.inputSize = this.inputSize;
    copy.hiddenSize = this.hiddenSize;
    copy.outputSize = this.outputSize;
    copy.w1 = this.w1.map((row) => row.slice());
    copy.b1 = this.b1.slice();
    copy.w2 = this.w2.map((row) => row.slice());
    copy.b2 = this.b2.slice();
    return copy;
  }

  toJSON() {
    return {
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      w1: this.w1,
      b1: this.b1,
      w2: this.w2,
      b2: this.b2
    };
  }

  static fromJSON(data) {
    if (!data || !Array.isArray(data.w1) || !Array.isArray(data.w2)) {
      throw new Error("Invalid network checkpoint");
    }

    const network = Object.create(TinyQNetwork.prototype);
    network.inputSize = data.inputSize;
    network.hiddenSize = data.hiddenSize;
    network.outputSize = data.outputSize;
    network.w1 = data.w1.map((row) => row.slice());
    network.b1 = data.b1.slice();
    network.w2 = data.w2.map((row) => row.slice());
    network.b2 = data.b2.slice();
    return network;
  }
}

function createMatrix(rows, cols, initializer) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, initializer));
}

function randomNormal(rng) {
  const u1 = Math.max(Number.EPSILON, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
