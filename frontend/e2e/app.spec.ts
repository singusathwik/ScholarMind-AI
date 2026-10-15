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
  await expect(
    page.getByRole("heading", { name: "How certain is the difference?" }),
  ).toBeVisible();
  await expect(
    page.getByText("Inconclusive", { exact: true }).first(),
  ).toBeVisible();
  await page.screenshot({
    path: "../docs/screenshots/evaluation.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("reading list persists notes and exports honest citation metadata", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Keyword", exact: true }).click();
  await page.getByLabel("Research question").fill("Vitamin D");
  await page.getByRole("button", { name: "Find papers" }).click();
  const card = page.locator(".paper-card").first();
  const title = await card.locator("h3").innerText();
  await card.getByRole("button", { name: "Save paper", exact: true }).click();
  await page.getByRole("button", { name: "Reading list (1)" }).click();
  await page
    .getByLabel(`Notes for ${title}`)
    .fill("Read the methods section next");
  await page.getByLabel(`Status for ${title}`).selectOption("reading");
  await page.reload();
  await page.getByRole("button", { name: "Reading list (1)" }).click();
  await expect(page.getByLabel(`Notes for ${title}`)).toHaveValue(
    "Read the methods section next",
  );
  await expect(page.getByLabel(`Status for ${title}`)).toHaveValue("reading");
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export BibTeX" }).click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toBe("scholarmind-reading-list.bib");
  const stream = await download.createReadStream();
  let bib = "";
  for await (const chunk of stream!) bib += chunk.toString();
  expect(bib).toContain("@misc{");
  expect(bib).not.toContain("year =");
  expect(bib).not.toContain("author =");
  expect(bib).not.toContain("Read the methods section next");
  await page.screenshot({
    path: "../docs/screenshots/reading-list.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Remove paper", exact: true }).click();
  await expect(
    page.getByText("Your reading list starts with a paper"),
  ).toBeVisible();
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
