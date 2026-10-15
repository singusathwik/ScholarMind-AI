import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import { JSDOM } from "jsdom";
// Node 25 exposes its own localStorage; use the JSDOM window's browser storage.
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: new JSDOM("", { url: "http://localhost" }).window.localStorage,
});
afterEach(cleanup);
HTMLDialogElement.prototype.showModal = function () {
  this.setAttribute("open", "");
};
