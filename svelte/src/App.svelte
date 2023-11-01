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

  let cells = [...ORIG_CELLS.map((r) => [...r])];
  type Position = [number, number];
  let agentPos: Position = [0, 3];
  let score = 0;
  let rowLen = 4;
  const onKeyDown = (e: KeyboardEvent) => {
    switch (e.code) {
      case "ArrowUp":
        moveTo(0, -1);
        break;
      case "ArrowDown":
        moveTo(0, 1);
        break;
      case "ArrowLeft":
        moveTo(-1, 0);
        break;
      case "ArrowRight":
        moveTo(1, 0);
        break;
      case "KeyR":
        cells = [...ORIG_CELLS.map((r) => [...r])];
        agentPos = [0, 3];
        score = 0;
        running = true;
        break;
    }
  };
  const isBorder = (x: number, y: number) =>
    x < 0 || x >= rowLen || y < 0 || y >= rowLen;
  const moveTo = (dx: number, dy: number) => {
    if (running) {
      const newX = agentPos[0] + dx;
      const newY = agentPos[1] + dy;

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
        score += 1;
        cells[newY][newX] = EMPTY;
      }

      agentPos = [newX, newY];

      // Moving into the goal.
      if (cell === GOAL) {
        score += 10;
        endEpisode();
      }

      // Moving into a pit.
      if (cell == PIT) {
        score -= 10;
        endEpisode();
      }
    }
  };

  let running = true;
  const endEpisode = () => {
    running = false;
  };
</script>

<main>
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
                : cells[y][x] > 0
                ? ICONS[cells[y][x] - 1]
                : ''}"
            />
          </div>
        </div>
      {/each}
    {/each}
  </div>
  <p>Score: {score}</p>
  <!-- <p>{bindings.add(6, 7)}</p> -->
</main>

<svelte:window on:keydown={onKeyDown} />

<style>
  main {
    padding: 1em;
  }

  .game {
    width: 41rem;
    height: 41rem;
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
