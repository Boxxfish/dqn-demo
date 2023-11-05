// Icons
export const AGENT_ICON = "bi-person-fill";
export const COIN_ICON = "bi-gem";
export const PIT_ICON = "bi-exclamation-octagon-fill color-danger";
export const GOAL_ICON = "bi-flag";
export const WALL_ICON = "bi-square-fill";
export const BOX_ICON = "bi-box2-fill";
export const ICONS = [COIN_ICON, PIT_ICON, GOAL_ICON, WALL_ICON, BOX_ICON];
export const ACTION_ICONS = [
  "bi-arrow-left",
  "bi-arrow-right",
  "bi-arrow-up",
  "bi-arrow-down",
];

// Reference grid
export const EMPTY = 0;
export const COIN = 1;
export const PIT = 2;
export const GOAL = 3;
export const WALL = 4;
export const BOX = 5;

export type Position = [number, number];
export type GameState = [number[][], Position];