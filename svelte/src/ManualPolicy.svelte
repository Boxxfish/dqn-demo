<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { ACTION_ICONS, AGENT_ICON, ICONS, type GameState } from "./constants";
  const dispatch = createEventDispatcher();

  export let transitions: [GameState, number, number, boolean][] = [];
</script>

<div>
  <p>Use the arrow keys to move. Press <b>R</b> to restart.</p>
  <div class="transitions">
    {#each transitions as transition, i}
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <div
        class="transition"
        on:click={() => dispatch("toTransition", { index: i })}
      >
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

<style>
  .transitions {
    display: flex;
    flex-wrap: wrap;
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

  p {
    font-size: 1.6rem;
  }
</style>
