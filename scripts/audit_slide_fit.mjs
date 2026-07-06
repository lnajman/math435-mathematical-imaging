#!/usr/bin/env node

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch (error) {
  const nodeModules =
    process.env.CODEX_NODE_MODULES ||
    path.join(
      process.env.HOME || "",
      ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules",
    );
  try {
    ({ chromium } = require(path.join(nodeModules, "playwright")));
  } catch {
    console.error(
      "Could not load Playwright. Install it locally or set CODEX_NODE_MODULES.",
    );
    throw error;
  }
}

const baseUrl = process.argv[2] || "http://127.0.0.1:8024";
const outDir = process.argv[3] || "/tmp/math435-slide-fit";
const viewport = { width: 1280, height: 720 };
const tolerance = 8;

fs.mkdirSync(outDir, { recursive: true });

const slideFiles = fs
  .readdirSync("slides")
  .filter((name) => /^week-.*\.qmd$/.test(name))
  .sort()
  .map((name) => name.replace(/\.qmd$/, ".html"));

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

const chromePath = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].find((candidate) => candidate && fs.existsSync(candidate));
const launchOptions = chromePath
  ? { headless: true, executablePath: chromePath }
  : { headless: true };

const browser = await chromium.launch(launchOptions);
const page = await browser.newPage({ viewport });

const fitIssues = [];
const warnings = [];

for (const file of slideFiles) {
  const deckName = file.replace(/\.html$/, "");
  const url = `${baseUrl.replace(/\/$/, "")}/slides/${file}`;
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.Reveal && window.Reveal.isReady());
  await page.evaluate(() => {
    window.Reveal.configure({
      transition: "none",
      backgroundTransition: "none",
    });
  });

  const slides = await page.evaluate(() =>
    window.Reveal.getSlides().map((slide, order) => {
      const indices = window.Reveal.getIndices(slide);
      const heading =
        slide.querySelector("h1,h2,h3")?.textContent?.trim() ||
        slide.textContent.trim().split(/\n/)[0].trim() ||
        "(untitled)";
      return { order, h: indices.h, v: indices.v || 0, heading };
    }),
  );

  for (const slideInfo of slides) {
    await page.evaluate(
      ({ h, v }) => window.Reveal.slide(h, v, 0),
      slideInfo,
    );
    await page.waitForTimeout(250);

    const result = await page.evaluate((tol) => {
      const slide = document.querySelector(".reveal .slides section.present");
      if (!slide) return { error: "No present slide found" };

      const slideRect = slide.getBoundingClientRect();
      const overflowX = slide.scrollWidth - slide.clientWidth;
      const overflowY = slide.scrollHeight - slide.clientHeight;
      const outliers = [];

      const elements = Array.from(slide.querySelectorAll("*"));
      for (const el of elements) {
        const style = window.getComputedStyle(el);
        if (
          style.display === "none" ||
          style.visibility === "hidden" ||
          style.opacity === "0" ||
          el.closest("script,style,noscript")
        ) {
          continue;
        }

        const rect = el.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) continue;

        const outside = {
          left: slideRect.left - rect.left,
          right: rect.right - slideRect.right,
          top: slideRect.top - rect.top,
          bottom: rect.bottom - slideRect.bottom,
        };

        if (
          outside.left > tol ||
          outside.right > tol ||
          outside.top > tol ||
          outside.bottom > tol
        ) {
          outliers.push({
            tag: el.tagName.toLowerCase(),
            className: String(el.className || ""),
            text: (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 120),
            outside,
          });
        }
      }

      return {
        slide: {
          width: Math.round(slideRect.width),
          height: Math.round(slideRect.height),
          scrollWidth: slide.scrollWidth,
          scrollHeight: slide.scrollHeight,
          clientWidth: slide.clientWidth,
          clientHeight: slide.clientHeight,
          overflowX,
          overflowY,
        },
        outliers: outliers.slice(0, 12),
        outlierCount: outliers.length,
        hasScrollOverflow: overflowX > tol || overflowY > tol,
        hasDomOutliers: outliers.length > 0,
      };
    }, tolerance);

    if (result.hasScrollOverflow || result.error) {
      const label = `${deckName}-${String(slideInfo.order + 1).padStart(2, "0")}-${slugify(
        slideInfo.heading,
      )}`;
      const screenshot = path.join(outDir, `${label}.png`);
      await page.screenshot({ path: screenshot, fullPage: false });
      fitIssues.push({ deck: deckName, ...slideInfo, screenshot, ...result });
    } else if (result.hasDomOutliers) {
      warnings.push({ deck: deckName, ...slideInfo, ...result });
    }
  }
}

await browser.close();

const reportPath = path.join(outDir, "slide-fit-report.json");
fs.writeFileSync(
  reportPath,
  `${JSON.stringify({ fitIssues, warnings }, null, 2)}\n`,
);

if (fitIssues.length) {
  console.log(`Found ${fitIssues.length} slide(s) with scroll overflow.`);
  console.log(`Report: ${reportPath}`);
  for (const item of fitIssues) {
    const overflow = item.slide
      ? `overflow x=${item.slide.overflowX}, y=${item.slide.overflowY}, outliers=${item.outlierCount}`
      : item.error;
    console.log(
      `- ${item.deck} slide ${item.order + 1}: ${item.heading} (${overflow})`,
    );
  }
  process.exitCode = 1;
} else {
  console.log("All audited slides fit within the slide frame.");
  if (warnings.length) {
    console.log(
      `DOM outlier warnings ignored as non-scrolling content: ${warnings.length}`,
    );
  }
  console.log(`Report: ${reportPath}`);
}
