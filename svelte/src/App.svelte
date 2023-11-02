<script lang="ts">
  // export let bindings;

  const EMPTY = 0;
  const COIN = 1;
  const PIT = 2;
  const GOAL = 3;
  const WALL = 4;
  const BOX = 5;

  const ORIG_CELLS = [
    [0, 0, 4, 3],
    [1, 5, 0, 0],
    [1, 2, 4, 0],
    [0, 0, 4, 1],
  ];
  const AGENT_ICON = "bi-person-fill";
  const COIN_ICON = "bi-gem";
  const PIT_ICON = "bi-exclamation-octagon-fill color-danger";
  const GOAL_ICON = "bi-flag";
  const WALL_ICON = "bi-square-fill";
  const BOX_ICON = "bi-box2-fill";
  const ICONS = [COIN_ICON, PIT_ICON, GOAL_ICON, WALL_ICON, BOX_ICON];
  const ACTION_ICONS = [
    "bi-arrow-left",
    "bi-arrow-right",
    "bi-arrow-up",
    "bi-arrow-down",
  ];

  const reset = () => {
    cells = [...ORIG_CELLS.map((r) => [...r])];
    agentPos = [0, 3];
    score = 0;
    running = true;
    transitions = [];
  };

  let cells = [...ORIG_CELLS.map((r) => [...r])];
  type Position = [number, number];
  let agentPos: Position = [0, 3];
  let score = 0;
  let rowLen = 4;
  const onKeyDown = (e: KeyboardEvent) => {
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
      case "KeyR":
        reset();
        break;
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
  };

  const toTransition = (i: number) => {
    if (i > 0) {
      cells = [...transitions[i][0][0].map((r) => [...r])];
      agentPos = transitions[i][0][1];
      score = transitions
        .slice(0, i)
        .map((t) => t[2])
        .reduce((prev, curr) => prev + curr);
      running = true;
      transitions = transitions.slice(0, i);
    } else {
      reset();
    }
  };

  type GameState = [number[][], Position];
  let transitions: [GameState, number, number, boolean][] = [];
</script>

<main class="color-dark">
  <h1>Deep Q Network Demo</h1>
  <div class="episode-status bg-dark color-light">
    {running ? "RUNNING..." : "EPISODE END"}
  </div>
  <div class="game color-dark">
    {#each cells as row, y}
      {#each row as cell, x}
        <div class="cell bg-primary">
          <span>{y * rowLen + x}</span>
          <div class="cell-icon">
            <i
              class="bi {x === agentPos[0] && y === agentPos[1]
                ? AGENT_ICON
                : cell > 0
                ? ICONS[cell - 1]
                : ''}"
            />
          </div>
        </div>
      {/each}
    {/each}
  </div>
  <p>Score: {score}</p>
  <div>
    <h1>Manual</h1>
    <p>Use the arrow keys to move. Press <b>R</b> to restart.</p>
    <div class="transitions">
      {#each transitions as transition, i}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <div class="transition" on:click={() => toTransition(i)}>
          <div class="state">
            {#each transition[0][0] as row, y}
              {#each row as cell, x}
                <div class="cell-mini bg-primary">
                  <div class="cell-icon-mini">
                    <i
                      class="bi {x === transition[0][1][0] &&
                      y === transition[0][1][1]
                        ? AGENT_ICON
                        : cell > 0
                        ? ICONS[cell - 1]
                        : ''}"
                    />
                  </div>
                </div>
              {/each}
            {/each}
          </div>
          <div class="action">
            <i class="bi {ACTION_ICONS[transition[1]]}" />
          </div>
          <div class="transition-text">
            {transition[2] > 0 ? "+" : ""}{transition[2]}
          </div>
          {#if transition[3]}
            <div class="transition-text">Done</div>
          {/if}
        </div>
      {/each}
    </div>
  </div>
  <!-- <p>{bindings.add(6, 7)}</p> -->
</main>

<svelte:window on:keydown={onKeyDown} />

<style>
  main {
    padding: 1em;
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

  .transitions {
    display: flex;
    flex-wrap: wrap;
  }

  .state {
    display: flex;
    flex-wrap: wrap;
    width: 8rem;
  }

  .cell-mini {
    width: 2rem;
    height: 2rem;
  }

  .cell-icon-mini {
    font-size: 1rem;
    text-align: center;
    align-content: center;
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

  .episode-status {
    padding: 1rem;
    font-size: 2rem;
    width: 38.5rem;
    display: flex;
  }

  p {
    font-size: 1.6rem;
  }
</style>
