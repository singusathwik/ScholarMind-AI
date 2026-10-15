import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
function abortableFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      (_url: string, options: RequestInit) =>
        new Promise((_resolve, reject) => {
          if (options.signal?.aborted) reject(options.signal.reason);
          options.signal?.addEventListener("abort", () =>
            reject(options.signal?.reason),
          );
        }),
    ),
  );
}
it("aborts timed-out requests with an actionable error", async () => {
  vi.useFakeTimers();
  abortableFetch();
  const pending = expect(api("/search", undefined, 50)).rejects.toMatchObject({
    name: "TimeoutError",
  });
  await vi.advanceTimersByTimeAsync(50);
  await pending;
  expect(vi.getTimerCount()).toBe(0);
});
it("forwards caller cancellation and clears the timeout", async () => {
  vi.useFakeTimers();
  abortableFetch();
  const controller = new AbortController();
  const pending = expect(
    api("/search", { signal: controller.signal }),
  ).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await pending;
  expect(vi.getTimerCount()).toBe(0);
});
it("explains connection failures", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
  );
  await expect(api("/health")).rejects.toThrow("Cannot reach the backend");
});
