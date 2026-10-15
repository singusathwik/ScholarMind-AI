import { expect, test } from "@playwright/test";

test("real corpus discovery, details, recommendations, all retrieval modes and dashboard", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByText("Local index ready")).toBeVisible();
  await page.screenshot({
    path: "../docs/screenshots/discover.png",
    fullPage: true,
  });
  await page.getByLabel("Research question").fill("Vitamin D and bone health");
  for (const mode of ["Keyword", "Semantic", "Hybrid", "Hybrid + reranking"]) {
    await page.getByRole("button", { name: mode, exact: true }).click();
    await page.getByRole("button", { name: "Find papers" }).click();
    await expect(page.locator(".paper-card")).toHaveCount(10, {
      timeout: 120000,
    });
    await expect(
      page.getByRole("button", { name: "Find papers" }),
    ).toBeEnabled();
    expect(await page.locator(".paper-card").first().innerText()).toContain(
      "scifact",
    );
  }
  await page.screenshot({
    path: "../docs/screenshots/results.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByText("Page 2 of 2")).toBeVisible();
  await page
    .getByRole("button", { name: "View details", exact: true })
    .first()
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator(".related")).toHaveCount(5);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await expect(page.getByRole("table")).toBeVisible();
  await page.screenshot({
    path: "../docs/screenshots/evaluation.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("mobile layout and search", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Keyword", exact: true }).click();
  await page.getByLabel("Research question").fill("immune cells");
  await page.getByRole("button", { name: "Find papers" }).click();
  await expect(page.locator(".paper-card")).toHaveCount(10);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../docs/screenshots/mobile.png",
    fullPage: true,
  });
});
