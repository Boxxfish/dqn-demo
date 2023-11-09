<script lang="ts">
  import { onMount, type ComponentEvents } from "svelte";
  import {
    AGENT_ICON,
    BOX,
    COIN,
    EMPTY,
    GOAL,
    PIT,
    WALL,
    ICONS,
    type Position,
    type GameState,
  } from "./constants";
  import Policies from "./Policies.svelte";
  export let bindings;
  let { DQN } = bindings;

  const ORIG_CELLS = [
    [0, 0, 4, 3],
    [1, 0, 0, 0],
    [1, 2, 4, 0],
    [0, 0, 4, 1],
  ];

  const reset = () => {
    cells = [...ORIG_CELLS.map((r) => [...r])];
    agentPos = [0, 3];
    score = 0;
    running = true;
    transitions = [];
  };

  let cells = [...ORIG_CELLS.map((r) => [...r])];
  let agentPos: Position = [0, 3];
  let score = 0;
  let rowLen = 4;
  const onKeyDown = (e: KeyboardEvent) => {
    if (activeTab === 0) {
      switch (e.code) {
        case "ArrowUp":
          step(2);
          break;
        case "ArrowDown":
          step(3);
          break;
        case "ArrowLeft":
          step(0);
          break;
        case "ArrowRight":
          step(1);
          break;
      }
    }

    if (e.code === "KeyR") {
      reset();
    }
  };
  const isBorder = (x: number, y: number) =>
    x < 0 || x >= rowLen || y < 0 || y >= rowLen;
  const step = (action: number) => {
    if (running) {
      const oldGameState = [...cells.map((r) => [...r])];
      let dx = 0,
        dy = 0;
      switch (action) {
        case 0:
          dx = -1;
          break;
        case 1:
          dx = 1;
          break;
        case 2:
          dy = -1;
          break;
        case 3:
          dy = 1;
          break;
      }

      const newX = agentPos[0] + dx;
      const newY = agentPos[1] + dy;
      let reward = 0;
      let done = false;

      // Moving into the border.
      if (isBorder(newX, newY)) {
        return;
      }

      const cell = cells[newY][newX];

      // Moving into a wall.
      if (cell === WALL) {
        return;
      }

      // Moving a box.
      if (cell === BOX) {
        const boxNewX = newX + dx;
        const boxNewY = newY + dy;
        if (
          isBorder(boxNewX, boxNewY) ||
          cells[boxNewY][boxNewX] == WALL ||
          cells[boxNewY][boxNewX] == BOX
        ) {
          return;
        } else {
          cells[boxNewY][boxNewX] = BOX;
          cells[newY][newX] = EMPTY;
        }
      }

      // Moving into a coin.
      if (cell === COIN) {
        reward += 1;
        cells[newY][newX] = EMPTY;
      }

      // Moving into the goal.
      if (cell === GOAL) {
        reward += 10;
        done = true;
        endEpisode();
      }

      // Moving into a pit.
      if (cell == PIT) {
        reward -= 10;
        done = true;
        endEpisode();
      }

      score += reward;
      const oldAgentPos = agentPos;
      agentPos = [newX, newY];
      transitions = [
        ...transitions,
        [[oldGameState, oldAgentPos], action, reward, done],
      ];
    }
  };

  let running = true;
  const endEpisode = () => {
    running = false;
    if (interval) {
      clearInterval(interval);
      runningDQN = false;
      running = false;
    }
  };

  const toTransition = (e: ComponentEvents<Policies>["toTransition"]) => {
    const { index } = e.detail;
    if (index > 0) {
      cells = [...transitions[index][0][0].map((r) => [...r])];
      agentPos = transitions[index][0][1];
      score = transitions
        .slice(0, index)
        .map((t) => t[2])
        .reduce((prev, curr) => prev + curr);
      running = true;
      transitions = transitions.slice(0, index);
    } else {
      reset();
    }
  };

  const cellClick = (x: number, y: number) => {
    const dY = y - agentPos[1];
    const dX = x - agentPos[0];
    
    if (dX === -1 && dY === 0) {
      step(0);
    }
    if (dX === 1 && dY === 0) {
      step(1);
    }
    if (dX === 0 && dY === -1) {
      step(2);
    }
    if (dX === 0 && dY === 1) {
      step(3);
    }
  };

  let dqn = null;
  const getState = (cells: number[][], agentPos: Position) => {
    let state = Array(6 * 6 * 6);
    const gridSize = 4;
    const gridSizeBorder = 6;
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const cell = cells[y][x];
        switch (cell) {
          case EMPTY:
            break;
          case COIN:
            state[
              0 * gridSizeBorder * gridSizeBorder +
                (y + 1) * gridSizeBorder +
                (x + 1)
            ] = 1;
            break;
          case PIT:
            state[
              1 * gridSizeBorder * gridSizeBorder +
                (y + 1) * gridSizeBorder +
                (x + 1)
            ] = 1;
            break;
          case WALL:
            state[
              2 * gridSizeBorder * gridSizeBorder +
                (y + 1) * gridSizeBorder +
                (x + 1)
            ] = 1;
            break;
          case BOX:
            state[
              3 * gridSizeBorder * gridSizeBorder +
                (y + 1) * gridSizeBorder +
                (x + 1)
            ] = 1;
            break;
          case GOAL:
            state[
              4 * gridSizeBorder * gridSizeBorder +
                (y + 1) * gridSizeBorder +
                (x + 1)
            ] = 1;
            break;
        }
      }
    }
    state[
      5 * gridSizeBorder * gridSizeBorder +
        (agentPos[1] + 1) * gridSizeBorder +
        (agentPos[0] + 1)
    ] = 1;

    // Outer border
    for (let y = 0; y < gridSizeBorder; y++) {
      for (let x = 0; x < gridSizeBorder; x++) {
        if (
          x >= 1 &&
          x < gridSizeBorder - 1 &&
          y >= 1 &&
          y < gridSizeBorder - 1
        ) {
          continue;
        }
        state[2 * gridSizeBorder * gridSizeBorder + y * gridSizeBorder + x] = 1;
      }
    }
    return state;
  };

  const evalState = (cells: number[][], state) => {
    const qVals: Float32Array = dqn.eval_state(state);

    // Masking
    if (agentPos[0] == 0 || cells[agentPos[1]][agentPos[0] - 1] == WALL) {
      qVals[0] = -Infinity;
    }
    if (agentPos[0] == 3 || cells[agentPos[1]][agentPos[0] + 1] == WALL) {
      qVals[1] = -Infinity;
    }
    if (agentPos[1] == 0 || cells[agentPos[1] - 1][agentPos[0]] == WALL) {
      qVals[2] = -Infinity;
    }
    if (agentPos[1] == 3 || cells[agentPos[1] + 1][agentPos[0]] == WALL) {
      qVals[3] = -Infinity;
    }
    return qVals;
  };
  $: state = getState(cells, agentPos);
  $: qVals = dqn ? evalState(cells, state) : [];
  $: action = qVals.length > 0 ? qVals.indexOf(Math.max(...qVals)) : 0;
  const stepDQN = () => {
    if (dqn) {
      step(action);
    }
  };

  let interval;
  onMount(async () => {
    fetch("q_net_grid.safetensors")
      .then((response) => response.blob())
      .then((data) => data.arrayBuffer())
      .then((data) => {
        dqn = DQN.load(new Uint8Array(data));
      })
      .catch((error) => {
        console.log(error);
        return [];
      });
    return () => {
      clearInterval(interval);
    };
  });

  const runDQN = () => {
    interval = setInterval(stepDQN, 1000);
    runningDQN = true;
    running = true;
  };

  const pauseDQN = () => {
    clearInterval(interval);
    runningDQN = false;
  };

  let transitions: [GameState, number, number, boolean][] = [];

  let activeTab = 0;
  let runningDQN = false;
</script>

<main class="color-dark">
  <h1>Deep Q Network Demo</h1>
  <div class="container">
    <div class="game color-dark">
      <div class="cover {running ? '' : 'visible'}">
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <p class="color-light bg-dark" on:click={reset}>
          Press <b>R</b> to Restart<br />Or Click Here
        </p>
      </div>
      {#each cells as row, y}
        {#each row as cell, x}
          <div class="cell bg-primary" on:click={() => cellClick(x, y)}>
            <div class="cell-icon">
              <i
                class="bi {x === agentPos[0] && y === agentPos[1]
                  ? AGENT_ICON
                  : cell > 0
                  ? ICONS[cell - 1]
                  : ''}"
              />
              {#if x === agentPos[0] - 1 && y == agentPos[1] && qVals[0] !== -Infinity}
                <span class="q-value">{(qVals[0] * 10).toFixed(2)}</span>
              {/if}
              {#if x === agentPos[0] + 1 && y == agentPos[1] && qVals[1] !== -Infinity}
                <span class="q-value">{(qVals[1] * 10).toFixed(2)}</span>
              {/if}
              {#if x === agentPos[0] && y == agentPos[1] - 1 && qVals[2] !== -Infinity}
                <span class="q-value">{(qVals[2] * 10).toFixed(2)}</span>
              {/if}
              {#if x === agentPos[0] && y == agentPos[1] + 1 && qVals[3] !== -Infinity}
                <span class="q-value">{(qVals[3] * 10).toFixed(2)}</span>
              {/if}
            </div>
          </div>
        {/each}
      {/each}
    </div>
    <div class="score bg-dark color-light">
      <h1>Score: {score}</h1>
    </div>
  </div>
  <Policies
    on:toTransition={toTransition}
    on:tabChanged={(e) => (activeTab = e.detail.index)}
    on:run={runDQN}
    on:pause={pauseDQN}
    {transitions}
    {activeTab}
    {runningDQN}
  />
</main>

<svelte:window on:keydown={onKeyDown} />

<style>
  main {
    padding: 1em;
  }

  .container {
    display: flex;
  }

  .score {
    width: 12rem;
    text-align: center;
  }

  .cover {
    display: none;
    position: absolute;
    width: 41rem;
    height: 41rem;
    background-color: #ffffff88;
  }

  .cover p {
    text-align: center;
    font-size: 2rem;
    margin-top: 20rem;
    padding: 2rem;
  }

  .visible {
    display: block;
  }

  .game {
    width: 41rem;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
  }

  .cell {
    width: 10rem;
    height: 10rem;
    margin: 0.1rem;
  }

  .cell-icon {
    font-size: 5rem;
    text-align: center;
    align-content: center;
  }

  .q-value {
    position: absolute;
    text-align: center;
    font-size: 1rem;
    font-weight: bold;
  }

  .action {
    padding: 1rem;
    font-size: 2rem;
    display: flex;
    align-items: center;
  }

  .transition {
    display: flex;
    align-items: center;
    font-size: 2rem;
    padding: 1rem;
    margin: 1rem;
    box-shadow: 0 0 0.2rem black;
    cursor: pointer;
  }

  .transition-text {
    display: flex;
    margin: 1rem;
  }
</style>
