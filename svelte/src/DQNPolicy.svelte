<script lang="ts">
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();
  export let runningDQN;
</script>

<div class="options">
  {#if !runningDQN}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <div class="option" on:click={() => dispatch("run")}>Run</div>
  {:else}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <div class="option" on:click={() => dispatch("pause")}>Pause</div>
  {/if}
</div>

<style>
  .options {
    display: flex;
  }

  .option {
    padding: 1rem;
    cursor: pointer;
  }

  div {
    font-size: 1.6rem;
  }

  .option:hover {
    outline: solid 1px black;
  }
</style>
