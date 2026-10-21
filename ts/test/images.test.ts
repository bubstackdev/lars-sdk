import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it, expect } from "vitest";
import { Lars, ImageInput } from "../src/index.js";

describe("image input contract", () => {
  it("rejects empty/oversized inputs and prevents accidental JSON serialization", () => {
    expect(() => Lars.imageBytes(new Uint8Array())).toThrow("20 MiB");
    expect(() => Lars.imageBytes(new Uint8Array(20 * 1024 * 1024 + 1))).toThrow("20 MiB");
    expect(() => JSON.stringify({ nested: Lars.imageBytes(Buffer.from("png")) })).toThrow("directly");
  });

  it("rejects images on text packs and incompatible state selectors", async () => {
    const dir = mkdtempSync(join(tmpdir(), "lars-image-contract-"));
    writeFileSync(join(dir, "pack.json"), JSON.stringify({ format: 1, name: "text", version: "0", backend: "mock", template: "plain" }));
    const lars = await Lars.open(dir);
    try {
      const image = Lars.imageBytes(Buffer.from("png"));
      const a = lars.noul<ImageInput>("a", "Something is visible");
      await expect(a(image)).rejects.toMatchObject({ type: "invalid_request" });
      const b = lars.noul<ImageInput>("b", "Something else is visible", { state: i => i.withContext("different") });
      await expect(lars.ask(image, { a, b })).rejects.toThrow("same image and context");
      await expect(a.calibrate([{ input: image, label: true }])).rejects.toThrow("directly");
    } finally { lars.close(); }
  });
});

describe.skipIf(!process.env.LARS_TEST_VISION_PACK)("real visual predicates", () => {
  it("returns typed image decisions, snapshots buffers, and recovers after invalid input", async () => {
    const lars = await Lars.open({ pack: process.env.LARS_TEST_VISION_PACK!, threads: 4 });
    try {
      const red = await Lars.imageFile(new URL("../../fixtures/images/red.png", import.meta.url).pathname);
      const blue = await Lars.imageFile(new URL("../../fixtures/images/blue.png", import.meta.url).pathname);
      const predicate = lars.choice("color", "What is the predominant color in the image?", {
        red: "Red", blue: "Blue", other: "Another color",
      }, { state: (i: ImageInput) => i });
      expect(await predicate(red)).toBe("red");
      expect(await predicate(blue)).toBe("blue");
      const decisions = await lars.ask(red, { color: predicate });
      expect(decisions.color).toMatchObject({ value: "red", calibrated: false });
      const source = Buffer.from(await import("node:fs/promises").then(fs => fs.readFile(new URL("../../fixtures/images/red.png", import.meta.url))));
      const snapshot = Lars.imageBytes(source);
      source.fill(0);
      expect(await predicate(snapshot)).toBe("red");
      await expect(predicate(Lars.imageBytes(source))).rejects.toThrow();
      expect(await predicate(red)).toBe("red");
    } finally { lars.close(); }
  });
});
