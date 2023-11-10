<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import ManualPolicy from "./ManualPolicy.svelte";
  import DqnPolicy from "./DQNPolicy.svelte";
  const dispatch = createEventDispatcher();

  export let runningDQN;
  export let activeTab = 0;

  const items = [
    {
      label: "Manual",
      component: ManualPolicy,
    },
    {
      label: "DQN",
      component: DqnPolicy,
    },
  ];
  $: activeItem = items[activeTab];
</script>

<div>
  <h1>Policies</h1>
  <ul>
    {#each items as item, i}
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <li
        on:click={() => dispatch("tabChanged", { index: i })}
        class={activeTab === i ? "bg-dark color-light" : ""}
      >
        <span>{item.label}</span>
      </li>
    {/each}
  </ul>
  {#each items as item, i}
    {#if activeTab == i}
      <div class="box">
        <svelte:component
          this={activeItem.component}
          {runningDQN}
          on:run={() => dispatch("run")}
          on:pause={() => dispatch("pause")}
        />
      </div>
    {/if}
  {/each}
</div>

<style>

  ul {
    display: flex;
    list-style: none;
  }

  li {
    padding: 1rem;
    font-size: 1.8rem;
    margin: 0.8rem;
    cursor: pointer;
  }

  li:hover {
    outline: solid 1px black;
  }

  .box {
    display: flex;
  }
</style>
