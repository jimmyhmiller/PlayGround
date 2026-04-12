/* tslint:disable */
/* eslint-disable */

export function start(): void;

export function theme_get_json(): string;

export function theme_preset_names(): string;

export function theme_set_accent(index: number, hex: string): boolean;

export function theme_set_field(key: string, value: string): boolean;

export function theme_set_preset(name: string): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly start: () => void;
    readonly theme_get_json: () => [number, number];
    readonly theme_preset_names: () => [number, number];
    readonly theme_set_accent: (a: number, b: number, c: number) => number;
    readonly theme_set_field: (a: number, b: number, c: number, d: number) => number;
    readonly theme_set_preset: (a: number, b: number) => number;
    readonly wasm_bindgen__closure__destroy__ha3b6bac00ec24833: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h4e9aaed4c664decb: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h5ad3c03a66b84261: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__he4058ea69d132961: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hc2e60cb81d4431bf: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h04f59023cd0a8e37: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h5dd389f26be0fa6d: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h83ec71bf56ee54dc: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hb5ab3969e84ac403: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h0136637d5bedefbf: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h581d386aba956199: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__hf1a207f4ed17542c: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h9788eaa77ce47e49: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h55853532359e6423: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hada2727bf2d81d31: (a: number, b: number, c: number, d: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h84e99b78bd7c1bc4: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb64f419d1d2efa57: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h7792e6f85a748655: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h922a4f930478b022: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h635ced8cf20ac177: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h5712cd81870c9e1f: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h339806f3739deee3: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h6829fabfcb95ffbb: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h36fad0ce39e82eb5: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h398901ffe68d3018: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hbdf87fb2ff6b4580: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h28b4c8a9806237b4: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__haed2ff522d7f00ad: (a: number, b: number) => number;
    readonly wasm_bindgen__convert__closures_____invoke__h0f9703082541f303: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
