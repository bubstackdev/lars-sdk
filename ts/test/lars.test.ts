// @moedex/lars on a mock pack (no model: deterministic, keyword-driven answers that test the
// plumbing, not judgment), and on a real pack when LARS_TEST_PACK names one.

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, expectTypeOf, it } from "vitest";

import { Lars, LarsError } from "../src/index.js";

function mockPack(name: string): string {
  const dir = mkdtempSync(join(tmpdir(), `lars-ts-${name}-`));
  writeFileSync(join(dir, "pack.json"),
                JSON.stringify({ format: 1, name, version: "0.0.0", backend: "mock", template: "plain" }));
  return dir;
}

describe("on a mock pack", () => {
  let lars: Lars;
  beforeAll(async () => {
    lars = await Lars.open({ pack: mockPack("ts"), threads: 2 });
  });
  afterAll(() => lars.close());

  it("opens a pack and reports it", () => {
    expect(lars.pack.name).toBe("ts");
    expect(Lars.version()).toMatch(/^\d+\.\d+\.\d+/);
    expect(lars.status()).toMatchObject({ loaded: true, queued: 0 });
  });

  it("nouls resolve to booleans, so `if` reads as it should", async () => {
    const isRefund = lars.noul("isRefund", "The customer asks for a refund");
    const value = await isRefund("Please refund my order");
    expectTypeOf(value).toEqualTypeOf<boolean>();
    expect(typeof value).toBe("boolean");
    const d = await isRefund.decide("Please refund my order");
    expect(d.value).toBe(d.p >= 0.5);
    expect(d.confidence).toBeCloseTo(Math.max(d.p, 1 - d.p), 10);
    expect(d.accepted).toBe(true);
    expect(isRefund.name).toBe("isRefund");
  });

  it("thresholds move the decision", async () => {
    const strict = lars.noul("strictRefund", "The customer asks for a refund", { threshold: 0.999 });
    const d = await strict.decide("Please refund my order");
    expect(d.value).toBe(d.p >= 0.999);
  });

  it("choices resolve to one of their keys, typed as a union", async () => {
    const team = lars.choice("team", "Which team should handle this?", {
      billing: "Payments, charges, refunds",
      technical: "Bugs, outages, errors",
    });
    const value = await team("I was charged twice for my payment");
    expectTypeOf(value).toEqualTypeOf<"billing" | "technical">();
    expect(["billing", "technical"]).toContain(value);
    const d = await team.decide("I was charged twice for my payment");
    expect(Object.keys(d.probabilities).sort()).toEqual(["billing", "technical"]);
    expect(d.confidence).toBe(Math.max(...Object.values(d.probabilities)));
  });

  it("scores and multis", async () => {
    const urgency = lars.score("urgency", "How urgent?", ["routine", "soon", "today", "emergency"]);
    const level = await urgency("The server is down");
    expect(level).toBeGreaterThanOrEqual(0);
    expect(level).toBeLessThanOrEqual(3);
    const d = await urgency.decide("The server is down");
    expect(d.probabilities).toHaveLength(4);
    expect(d.label).toBe(["routine", "soon", "today", "emergency"][d.level]);
    const tags = lars.multi("tags", "Which apply?", { refund: "Wants money back", bug: "Reports a defect" });
    const selected = await tags("refund my money, the app has a bug");
    expectTypeOf(selected).toEqualTypeOf<("refund" | "bug")[]>();
    expect(selected.every((t) => t === "refund" || t === "bug")).toBe(true);
  });

  it("objects are sent as JSON state, and `state` picks the fields", async () => {
    type Ticket = { subject: string; body: string; email: string };
    let seen: unknown;
    const urgent = lars.noul<Ticket>("ticketUrgent", "The ticket is urgent", {
      state: (t) => (seen = { subject: t.subject, body: t.body }),
    });
    await urgent({ subject: "Down", body: "Everything is down", email: "someone@example.com" });
    expect(seen).toEqual({ subject: "Down", body: "Everything is down" });
  });

  it("ask answers several predicates in one request, keyed as given", async () => {
    const angry = lars.noul("angry", "The customer is angry");
    const team = lars.choice("askTeam", "Which team?", { billing: "Charges", technical: "Bugs" });
    const r = await lars.ask("I was charged twice and I am furious", { angry, team });
    expect(typeof r.angry.value).toBe("boolean");
    expect(["billing", "technical"]).toContain(r.team.value);
  });

  it("many concurrent calls all resolve", async () => {
    const isRefund = lars.noul("isRefund", "The customer asks for a refund");
    const values = await Promise.all(Array.from({ length: 300 }, (_, i) => isRefund(`message ${i}: refund please`)));
    expect(values).toHaveLength(300);
  });

  it("calibration fits a per-predicate calibrator and refuses to certify on too few labels", async () => {
    const isRefund = lars.noul("calibrated", "The customer asks for a refund");
    const examples = Array.from({ length: 40 }, (_, i) => ({ input: i % 2 ? `refund order ${i}` : `hello ${i}`, label: i % 2 === 1 }));
    const fit = await isRefund.calibrate(examples, { maxError: 0.05 });
    expect(fit.examples).toBe(40);
    expect(fit.certified?.threshold).toBeNull();
    expect(fit.certified?.reason).toMatch(/cannot certify/);
    expect(lars.status().calibrators).toContain("calibrated");
    const saved = lars.calibrations();
    expect(Object.keys(saved.calibrators)).toContain("calibrated");

    const other = await Lars.open({ pack: lars.status().pack ? (lars.status().pack as { dir: string }).dir : "" });
    await other.loadCalibrations(saved);
    expect(other.status().calibrators).toContain("calibrated");
    other.close();
  });

  it("a certified threshold gates the boolean, and an unaccepted choice falls back", async () => {
    // 200 easy, consistent labels from the mock: certify 5%, then answers below the threshold are not accepted.
    const isRefund = lars.noul("certified", "The customer asks for a refund");
    const inputs = Array.from({ length: 200 }, (_, i) => `refund the order number ${i}`);
    const decisions = await Promise.all(inputs.map((x) => isRefund.decide(x)));
    const fit = await isRefund.calibrate(inputs.map((input, i) => ({ input, label: decisions[i]!.value })), { maxError: 0.05 });
    expect(fit.accuracy).toBeGreaterThan(0.9);
    if (fit.certified?.threshold != null) {
      const d = await isRefund.decide(inputs[0]!);
      expect(d.accepted).toBe(d.confidence >= fit.certified.threshold);
      expect(await isRefund(inputs[0]!)).toBe(d.value && d.accepted);
    }
  });

  it("names are unique per question", () => {
    lars.noul("unique", "One question");
    expect(() => lars.noul("unique", "Another question")).toThrow(LarsError);
  });

  it("errors are LarsErrors with a type", async () => {
    const error = await lars.evaluate({ state: "x", questions: {} }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(LarsError);
    expect((error as LarsError).type).toBe("invalid_request");
    const missing = await Lars.open("./no/such/pack").catch((e: unknown) => e);
    expect((missing as LarsError).type).toBe("pack_error");
  });

  it("calls after close reject", async () => {
    const closing = await Lars.open(mockPack("closing"));
    const p = closing.noul("p", "Anything");
    closing.close();
    await expect(p("x")).rejects.toBeInstanceOf(LarsError);
  });
});

describe.runIf(process.env.LARS_TEST_PACK)("on a real pack", () => {
  let lars: Lars;
  beforeAll(async () => {
    lars = await Lars.open(process.env.LARS_TEST_PACK!);
  }, 60_000);
  afterAll(() => lars?.close());

  it("judges plain cases the way a person would", async () => {
    const isPissed = lars.noul("isPissed", "The customer is angry");
    expect(await isPissed("This is the third time your app ate my order. I am done with you people.")).toBe(true);
    expect(await isPissed("Thanks so much, the new update works great!")).toBe(false);
    const team = lars.choice("team", "Which team should handle this message?", {
      billing: "Payments, charges, invoices, refunds",
      technical: "Bugs, crashes, outages, error messages",
      sales: "Pricing questions, upgrades, new contracts",
    });
    expect(await team("I was charged twice on my credit card this month")).toBe("billing");
    expect(await team("The app crashes every time I open settings")).toBe("technical");
    expect(await team("What would it cost to move our 40 seats to the enterprise plan?")).toBe("sales");
  }, 60_000);

  it("answers several questions about one message in one request", async () => {
    const angry = lars.noul("angry2", "The customer is angry");
    const refund = lars.noul("refund2", "The customer asks for money back");
    const urgency = lars.score("urgency2", "How urgently does this need a reply?", ["whenever", "this week", "today", "right now"]);
    const r = await lars.ask("You charged me twice and I want my money back NOW or I'm calling my bank.", { angry, refund, urgency });
    expect(r.angry.value).toBe(true);
    expect(r.refund.value).toBe(true);
    expect(r.urgency.value).toBeGreaterThan(1.5);
  }, 60_000);
});
