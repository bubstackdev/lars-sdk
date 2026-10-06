// The System One wire shapes the engine speaks, plus the LARS extensions this package uses.

/** Text or structured content, as states, instructions and option descriptions may be. */
export type Content = string | { [key: string]: unknown } | unknown[] | null;

export type Question =
  | { type: "noul"; instructions?: Content; criteria?: { true?: Content; false?: Content } }
  | { type: "choice"; instructions?: Content; criteria: Record<string, Content> }
  | { type: "score"; instructions?: Content; criteria: Content[] }
  | { type: "multi"; instructions?: Content; criteria: Record<string, Content> };

export interface Constraint {
  kind: "complement" | "exclusive";
  questions: string[];
}

export interface LarsExtensions {
  permutations?: number;
  explain?: boolean;
  abstain_margin?: number;
  constraints?: Constraint[];
  features?: Record<string, Record<string, number>>;
  calibrator?: string;
  calibrators?: Record<string, string>;
}

export interface SystemOneRequest {
  state: Content;
  questions: Record<string, Question>;
  model?: string;
  lars?: LarsExtensions;
}

export interface Evidence {
  span: string;
  effect: number;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  abstain?: boolean;
  evidence?: Evidence[];
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  order_sensitivity?: number;
  abstain?: boolean;
  evidence?: Evidence[];
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, Content>;
  probabilities: Record<string, number>;
  confidence: number;
  abstain?: boolean;
  evidence?: Evidence[];
}

export interface MultiAnswer {
  type: "multi";
  probabilities: Record<string, number>;
  selected: string[];
  evidence?: Evidence[];
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer | MultiAnswer;

export interface SystemOneResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** An acceptance threshold certified on labeled examples (Wilson 95% bound on the error rate). */
export interface Certified {
  maxError: number;
  /** Accept answers whose confidence is at least this. Null when the labels cannot certify maxError
   *  (`reason` says why); then no answer is accepted until a calibration that certifies replaces it. */
  threshold: number | null;
  accepted?: number;
  errors?: number;
  /** The share of the labeled examples that would be accepted. */
  coverage?: number;
  errorUpper95?: number;
  reason?: string;
}
