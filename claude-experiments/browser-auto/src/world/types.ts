/** A first-class reference from one fact to another: ref("products", "blue-widget"). */
export interface Ref {
  $ref: [type: string, key: string];
}

/** A single fact's fields. Values are JSON-ish and may contain Refs. */
export type FactRow = Record<string, unknown>;

/** type -> key -> fields */
export type Facts = Record<string, Record<string, FactRow>>;

export interface Seed {
  readonly kind: "bat.seed";
  readonly name: string;
  readonly facts: Facts;
}

export interface Patch {
  type: string;
  key: string;
  field: string;
  value: unknown;
}

/** The merged, patched, validated description of a world. Pure data. */
export interface WorldDescription {
  facts: Facts;
  patches: Patch[];
  /** seed names that contributed, sorted */
  sources: string[];
  fingerprint: string;
}

/** Returned by installers so refs to this type can resolve. */
export type IdMap = Record<string, unknown>;

export interface InstallCtx {
  /** Resolve a ref (or type+key pair) to the id returned by that type's installer. */
  id(ref: Ref): unknown;
  id(type: string, key: string): unknown;
}

export interface EntityDef {
  /** Realize rows (key -> fields, refs still embedded). May return key -> id. */
  install(rows: Record<string, FactRow>, ctx: InstallCtx): Promise<IdMap | void> | IdMap | void;
  /** L1: return null if the row is valid, else an error message. */
  schema?(row: FactRow, key: string): string | null;
  /** L2: read back rows (description-shaped) for the given keys. */
  read?(keys: string[]): Promise<Record<string, FactRow>> | Record<string, FactRow>;
  /** Entity types that must install before this one. */
  needs?: string[];
}

export interface SessionState {
  cookies?: Array<{
    name: string;
    value: string;
    domain?: string;
    path?: string;
    expires?: number;
    httpOnly?: boolean;
    secure?: boolean;
    sameSite?: "Strict" | "Lax" | "None";
  }>;
  localStorage?: Record<string, string>;
}

export interface WorldAdapter {
  readonly kind: "bat.world";
  /** Required: make the world empty. */
  reset(): Promise<void> | void;
  entities: Record<string, EntityDef>;
  /** Mint a real session for a user fact key (for `given user X signed-in`). */
  session?(userKey: string, ctx: InstallCtx): Promise<SessionState> | SessionState;
  /** L3: hash of current world state. */
  fingerprint?(): Promise<string> | string;
  /** L4: snapshot / restore world state. */
  snapshot?(): Promise<string> | string;
  restore?(id: string): Promise<void> | void;
}

export type CapabilityLevel = 0 | 1 | 2 | 3 | 4;

export interface AppliedWorld {
  description: WorldDescription;
  /** ids returned by installers, per type */
  ids: Record<string, IdMap>;
  /** which guarantees were proven vs asserted */
  verification: VerificationReport;
}

export interface VerificationReport {
  level: CapabilityLevel;
  /** human-readable notes: what was proven, what was merely asserted */
  proven: string[];
  asserted: string[];
}
