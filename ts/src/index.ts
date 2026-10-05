// @moedex/lars: typed decisions as functions in your code.
//
//   const lars = await Lars.open({ pack: "lars-4b" });
//   const isPissed = lars.noul("isPissed", "The customer is angry");
//   if (await isPissed(message)) escalate(message);
//
// A predicate is a function. Calling it asks the model one typed question about its input and
// resolves to a plain value (a boolean, one of your option keys, a level), never to an object, so
// `if` and `switch` work as they read. `decide()` gives the probabilities behind the value. The
// model runs in this process; requests queue on its worker thread and never block the event loop.

import { readFile, writeFile } from "node:fs/promises";

import { type NativeLars, native } from "./native.js";
import type {
  ChoiceAnswer,
  Certified,
  Constraint,
  Content,
  Evidence,
  MultiAnswer,
  NoulAnswer,
  Question,
  ScoreAnswer,
  SystemOneRequest,
  SystemOneResponse,
} from "./types.js";

export type * from "./types.js";

/** An error from the engine: `type` is invalid_request, pack_error, cancelled, api_error or internal_error. */
export class LarsError extends Error {
  constructor(
    readonly type: string,
    message: string,
  ) {
    super(message);
    this.name = "LarsError";
  }
}

function larsError(error: unknown): LarsError {
  const text = error instanceof Error ? error.message : String(error);
  try {
    const parsed = JSON.parse(text) as { message?: string; error_type?: string };
    if (parsed.error_type) return new LarsError(parsed.error_type, parsed.message ?? text);
  } catch {
    // not one of ours
  }
  return new LarsError("internal_error", text);
}

async function call<T>(f: () => Promise<T>): Promise<T> {
  try {
    return await f();
  } catch (error) {
    throw larsError(error);
  }
}

function sync<T>(f: () => T): T {
  try {
    return f();
  } catch (error) {
    throw larsError(error);
  }
}

export interface OpenOptions {
  /** A pack directory, or `name[@version]` from the pack cache (downloaded on first use). */
  pack: string;
  /** Layers on the GPU (Metal): -1 for all (the default), 0 for CPU only. */
  gpuLayers?: number;
  /** CPU threads; 0 (the default) for the number of cores. */
  threads?: number;
  /** Answer the questions of one request in one batched pass (default true). */
  batchSuffixes?: boolean;
  /** Keep the previous call's prompt prefix for the next one (default true). */
  reusePrefix?: boolean;
  maxRows?: number;
  maxInputTokens?: number;
  /** Free the model after this many idle seconds; it reloads on the next call. 0 keeps it. */
  idleUnloadSeconds?: number;
  /** Check every pack file's sha256 on open, not only its size. */
  verify?: boolean;
}

interface PredicateOptions<I> {
  /** What the model reads for an input: default the string itself, or the object as JSON. Pick
   *  the fields that matter, so whole records (and the personal data in them) stay out of prompts. */
  state?: (input: I) => Content;
  /** Mark an answer abstained when its top two probabilities are within this margin. */
  abstainMargin?: number;
  /** Return the spans of the state that moved the answer most (one extra pass per sentence). */
  explain?: boolean;
}

export interface NoulOptions<I> extends PredicateOptions<I> {
  /** P(yes) at or above which the answer is yes; default 0.5. */
  threshold?: number;
  /** What a yes and a no mean, if the instructions alone are not enough. */
  yes?: Content;
  no?: Content;
}

export interface ChoiceOptions<I, K extends string> extends PredicateOptions<I> {
  /** Returned by the call (not by `decide`) when the answer is not accepted under a certified threshold. */
  fallback?: K;
  /** Extra option orderings to average over, against order bias (each costs a pass). */
  permutations?: number;
}

export interface ScoreOptions<I> extends PredicateOptions<I> {}
export interface MultiOptions<I> extends PredicateOptions<I> {}

interface Decision {
  /** The top probability: how likely the answer is right, once calibrated. Thresholds compare to it. */
  confidence: number;
  /** False when a certified threshold exists for this predicate and the confidence is below it. */
  accepted: boolean;
  abstain?: boolean;
  evidence?: Evidence[];
}

export interface NoulDecision extends Decision {
  value: boolean;
  /** P(yes). */
  p: number;
}

export interface ChoiceDecision<K extends string> extends Decision {
  value: K;
  probabilities: Record<K, number>;
  orderSensitivity?: number;
}

export interface ScoreDecision extends Decision {
  /** The expected level, sum of i * P(level i); may fall between levels. */
  value: number;
  /** The most probable level and its description. */
  level: number;
  label: Content;
  probabilities: number[];
}

export interface MultiDecision<K extends string> {
  value: K[];
  probabilities: Record<K, number>;
  evidence?: Evidence[];
}

export interface CalibrateOptions {
  /** Also certify an acceptance threshold for this error rate (e.g. 0.05), at 95% confidence. */
  maxError?: number;
}

export interface CalibrationResult {
  examples: number;
  /** Which calibrator this predicate now uses: the fit on these labels, or the pack's when the fit was
   *  not significantly better on held-out labels (5-fold cross-validated log loss). */
  kept: "fitted" | "pack";
  /** Accuracy on the labeled examples themselves, with the fitted calibrator. */
  accuracy: number;
  certified: Certified | null;
}

interface Spec<I> {
  name: string;
  question: Question;
  stateOf: (input: I) => Content;
  extensions: (lars: Lars) => SystemOneRequest["lars"];
}

const SPEC = Symbol("lars.predicate");

export interface Predicate<I, V, D> {
  (input: I): Promise<V>;
  /** The value and the probabilities behind it. */
  decide(input: I): Promise<D>;
  readonly name: string;
  readonly question: Question;
  readonly [SPEC]: Spec<I>;
}

export interface Noul<I = string> extends Predicate<I, boolean, NoulDecision> {
  /** Fit this predicate's own calibrator on labeled examples. */
  calibrate(examples: { input: I; label: boolean }[], options?: CalibrateOptions): Promise<CalibrationResult>;
}

export interface Choice<K extends string, I = string> extends Predicate<I, K, ChoiceDecision<K>> {
  calibrate(examples: { input: I; label: K }[], options?: CalibrateOptions): Promise<CalibrationResult>;
}

export interface Score<I = string> extends Predicate<I, number, ScoreDecision> {
  /** Labels are level indices, 0 for the first level. */
  calibrate(examples: { input: I; label: number }[], options?: CalibrateOptions): Promise<CalibrationResult>;
}

export type Multi<K extends string, I = string> = Predicate<I, K[], MultiDecision<K>>;

type AnyPredicate = Predicate<any, unknown, unknown>;
type DecisionOf<P> = P extends Predicate<any, unknown, infer D> ? D : never;
type InputOf<P> = P extends Predicate<infer I, unknown, unknown> ? I : never;

export interface AskOptions {
  /** Structural relations between noul predicates, by predicate name. */
  constraints?: Constraint[];
}

/** `fn` with `props` attached; defineProperty, since a function's own `name` is read-only. */
function callable<F extends (...args: never[]) => unknown, P extends object, Q extends object = object>(
  fn: F,
  props: P,
  more?: Q,
): F & P & Q {
  for (const part of [props, more ?? {}]) {
    const props = part;
    for (const key of Reflect.ownKeys(props)) {
      Object.defineProperty(fn, key, { value: (props as Record<PropertyKey, unknown>)[key], enumerable: true });
    }
  }
  return fn as F & P & Q;
}

function defaultState(input: unknown): Content {
  if (typeof input === "string" || input === null) return input;
  if (typeof input === "object") return input as Content;
  return String(input);
}

function topProbability(probabilities: Record<string, number> | number[]): number {
  return Math.max(...Object.values(probabilities));
}

export interface SavedCalibrations {
  format: 1;
  pack: { name: string; version: string };
  calibrators: Record<string, unknown>;
  certified: Record<string, Certified>;
}

export class Lars {
  readonly #native: NativeLars;
  readonly #names = new Map<string, string>();
  readonly #calibrated = new Set<string>();
  readonly #certified = new Map<string, Certified>();
  /** The pack's manifest (pack.json): name, version, model, evaluation. */
  readonly pack: { name: string; version: string; [key: string]: unknown };

  private constructor(handle: NativeLars) {
    this.#native = handle;
    this.pack = JSON.parse(handle.manifest());
  }

  /** Open a pack and load its model (well under a second to a few seconds). */
  static async open(options: OpenOptions | string): Promise<Lars> {
    const o = typeof options === "string" ? { pack: options } : options;
    const json = JSON.stringify({
      gpu_layers: o.gpuLayers,
      threads: o.threads,
      batch_suffixes: o.batchSuffixes,
      reuse_prefix: o.reusePrefix,
      max_rows: o.maxRows,
      max_input_tokens: o.maxInputTokens,
      idle_unload_s: o.idleUnloadSeconds,
      verify: o.verify,
    });
    return new Lars(await call(() => native().open(o.pack, json)));
  }

  static version(): string {
    return native().version();
  }

  /** A yes/no predicate: resolves to true when the claim holds of the input. */
  noul<I = string>(name: string, instructions: Content, options: NoulOptions<I> = {}): Noul<I> {
    const question: Question = { type: "noul", instructions };
    if (options.yes !== undefined || options.no !== undefined) {
      question.criteria = { ...(options.yes !== undefined && { true: options.yes }), ...(options.no !== undefined && { false: options.no }) };
    }
    const threshold = options.threshold ?? 0.5;
    const decode = (answer: NoulAnswer): NoulDecision => {
      const p = answer.noul;
      return this.#decision(name, { value: p >= threshold, p, confidence: Math.max(p, 1 - p) }, answer);
    };
    const predicate = this.#predicate<I, NoulAnswer, NoulDecision>(name, question, options, decode, {});
    return callable(async (input: I) => {
      const d = await predicate.decide(input);
      return d.value && d.accepted;
    }, predicate, { calibrate: this.#calibrator(name, question, predicate[SPEC], (label: boolean) => (label ? "yes" : "no")) }) as Noul<I>;
  }

  /** Pick one option: resolves to one of the criteria's keys. */
  choice<const K extends string, I = string>(
    name: string,
    instructions: Content,
    criteria: Record<K, Content>,
    options: ChoiceOptions<I, K> = {},
  ): Choice<K, I> {
    const question: Question = { type: "choice", instructions, criteria };
    const decode = (answer: ChoiceAnswer): ChoiceDecision<K> =>
      this.#decision(name, {
        value: answer.choice as K,
        probabilities: answer.probabilities as Record<K, number>,
        confidence: topProbability(answer.probabilities),
        ...(answer.order_sensitivity !== undefined && { orderSensitivity: answer.order_sensitivity }),
      }, answer);
    const extra = options.permutations ? { permutations: options.permutations } : {};
    const predicate = this.#predicate<I, ChoiceAnswer, ChoiceDecision<K>>(name, question, options, decode, extra);
    return callable(async (input: I) => {
      const d = await predicate.decide(input);
      return !d.accepted && options.fallback !== undefined ? options.fallback : d.value;
    }, predicate, { calibrate: this.#calibrator(name, question, predicate[SPEC], (label: K) => label) }) as Choice<K, I>;
  }

  /** Place the input on ordered levels, lowest first: resolves to the expected level. */
  score<I = string>(name: string, instructions: Content, levels: Content[], options: ScoreOptions<I> = {}): Score<I> {
    const question: Question = { type: "score", instructions, criteria: levels };
    const decode = (answer: ScoreAnswer): ScoreDecision => {
      const probabilities = levels.map((_, i) => answer.probabilities[String(i)] ?? 0);
      const level = probabilities.indexOf(Math.max(...probabilities));
      return this.#decision(name, {
        value: answer.score,
        level,
        label: levels[level] ?? null,
        probabilities,
        confidence: Math.max(...probabilities),
      }, answer);
    };
    const predicate = this.#predicate<I, ScoreAnswer, ScoreDecision>(name, question, options, decode, {});
    return callable(async (input: I) => (await predicate.decide(input)).value, predicate, {
      calibrate: this.#calibrator(name, question, predicate[SPEC], (label: number) => String(label)),
    }) as Score<I>;
  }

  /** Every option that applies: resolves to the keys whose probability is at least 0.5. */
  multi<const K extends string, I = string>(
    name: string,
    instructions: Content,
    criteria: Record<K, Content>,
    options: MultiOptions<I> = {},
  ): Multi<K, I> {
    const question: Question = { type: "multi", instructions, criteria };
    const decode = (answer: MultiAnswer): MultiDecision<K> => ({
      value: answer.selected as K[],
      probabilities: answer.probabilities as Record<K, number>,
      ...(answer.evidence && { evidence: answer.evidence }),
    });
    const predicate = this.#predicate<I, MultiAnswer, MultiDecision<K>>(name, question, options, decode, {});
    return callable(async (input: I) => (await predicate.decide(input)).value, predicate) as Multi<K, I>;
  }

  /** Ask several predicates about one input in one request; the model reads the input once. */
  async ask<const P extends Record<string, AnyPredicate>>(
    input: InputOf<P[keyof P]>,
    predicates: P,
    options: AskOptions = {},
  ): Promise<{ [K in keyof P]: DecisionOf<P[K]> }> {
    const entries = Object.entries(predicates);
    const first = entries[0]?.[1];
    if (!first) throw new LarsError("invalid_request", "ask needs at least one predicate");
    const questions: Record<string, Question> = {};
    const calibrators: Record<string, string> = {};
    for (const [, p] of entries) {
      questions[p.name] = p.question;
      if (this.#calibrated.has(p.name)) calibrators[p.name] = p.name;
    }
    const request: SystemOneRequest = {
      state: first[SPEC].stateOf(input),
      questions,
      lars: { calibrators, ...(options.constraints && { constraints: options.constraints }) },
    };
    const response = await this.evaluate(request);
    const out: Record<string, unknown> = {};
    for (const [key, p] of entries) out[key] = this.#decoders.get(p.name)!(response.answers[p.name]);
    return out as { [K in keyof P]: DecisionOf<P[K]> };
  }

  /** Send a raw System One request. */
  async evaluate(request: SystemOneRequest): Promise<SystemOneResponse> {
    return JSON.parse(await call(() => this.#native.evaluate(JSON.stringify(request))));
  }

  /** Every calibrator fitted or loaded on this handle, with its certified thresholds. */
  calibrations(): SavedCalibrations {
    return {
      format: 1,
      pack: { name: this.pack.name, version: this.pack.version },
      calibrators: JSON.parse(sync(() => this.#native.calibrators())),
      certified: Object.fromEntries(this.#certified),
    };
  }

  async saveCalibrations(path: string): Promise<void> {
    await writeFile(path, JSON.stringify(this.calibrations(), null, 2) + "\n");
  }

  /** Load calibrations saved from this pack (an object, or a file path). */
  async loadCalibrations(saved: SavedCalibrations | string): Promise<void> {
    const data: SavedCalibrations = typeof saved === "string" ? JSON.parse(await readFile(saved, "utf8")) : saved;
    for (const [name, calibrator] of Object.entries(data.calibrators)) {
      sync(() => this.#native.setCalibrator(name, JSON.stringify(calibrator)));
      this.#calibrated.add(name);
    }
    for (const [name, certified] of Object.entries(data.certified ?? {})) this.#certified.set(name, certified);
  }

  status(): Record<string, unknown> {
    return JSON.parse(sync(() => this.#native.status()));
  }

  /** Release this handle; queued calls reject with a `cancelled` error. */
  close(): void {
    this.#native.close();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    this.close();
  }

  // ------------------------------------------------------------------ internals

  readonly #decoders = new Map<string, (answer: unknown) => unknown>();

  #decision<D extends { confidence: number }>(name: string, partial: D, answer: { abstain?: boolean; evidence?: Evidence[] }) {
    const certified = this.#certified.get(name);
    const accepted = certified?.threshold == null ? true : partial.confidence >= certified.threshold;
    return {
      ...partial,
      accepted,
      ...(answer.abstain !== undefined && { abstain: answer.abstain }),
      ...(answer.evidence && { evidence: answer.evidence }),
    };
  }

  #predicate<I, A, D>(
    name: string,
    question: Question,
    options: PredicateOptions<I>,
    decode: (answer: A) => D,
    extra: Record<string, unknown>,
  ) {
    const key = JSON.stringify(question);
    const existing = this.#names.get(name);
    if (existing !== undefined && existing !== key) {
      throw new LarsError("invalid_request", `a predicate named ${JSON.stringify(name)} already asks a different question`);
    }
    this.#names.set(name, key);
    this.#decoders.set(name, decode as (answer: unknown) => unknown);
    const stateOf = options.state ?? defaultState;
    const extensions = (lars: Lars) => ({
      ...extra,
      ...(options.abstainMargin !== undefined && { abstain_margin: options.abstainMargin }),
      ...(options.explain && { explain: true }),
      ...(lars.#calibrated.has(name) && { calibrators: { [name]: name } }),
    });
    const spec: Spec<I> = { name, question, stateOf: stateOf as (input: I) => Content, extensions };
    const decide = async (input: I): Promise<D> => {
      const response = await this.evaluate({ state: spec.stateOf(input), questions: { [name]: question }, lars: extensions(this) });
      return decode(response.answers[name] as A);
    };
    return { decide, name, question, [SPEC]: spec };
  }

  #calibrator<I, L>(name: string, question: Question, spec: Spec<I>, label: (l: L) => string) {
    return async (examples: { input: I; label: L }[], options: CalibrateOptions = {}): Promise<CalibrationResult> => {
      const jsonl = examples
        .map((e) => JSON.stringify({ state: spec.stateOf(e.input), question, label: label(e.label) }))
        .join("\n");
      const raw = JSON.parse(await call(() => this.#native.calibrate(name, jsonl, options.maxError ?? null)));
      this.#calibrated.add(name);
      const c = raw.certified;
      const certified: Certified | null = c
        ? {
            maxError: c.max_error,
            threshold: c.threshold,
            ...(c.accepted !== undefined && { accepted: c.accepted, errors: c.errors, coverage: c.coverage, errorUpper95: c.error_upper_95 }),
            ...(c.reason && { reason: c.reason }),
          }
        : null;
      if (certified) this.#certified.set(name, certified);
      else this.#certified.delete(name);
      return { examples: raw.examples, kept: raw.kept, accuracy: raw.accuracy, certified };
    };
  }
}

export default Lars;
