//! Native side of `internalBinding('util')` — only the pieces touched by
//! lib/buffer.js and lib/internal/buffer.js.

use rquickjs::function::Func;
use rquickjs::{Ctx, Object, Result};

pub fn make(ctx: Ctx<'_>) -> Result<Object<'_>> {
    let b = Object::new(ctx.clone())?;

    // constants — used by inspect; numeric flags. The exact values don't
    // matter as long as our getOwnNonIndexProperties handles them.
    let constants = Object::new(ctx.clone())?;
    constants.set("ALL_PROPERTIES", 1)?;
    constants.set("ONLY_ENUMERABLE", 2)?;
    b.set("constants", constants)?;

    // getOwnNonIndexProperties(obj, mode) -> string[]
    // We return non-numeric (non-array-index) own keys. The mode flag is
    // ignored — inspect is best-effort here.
    let go_src = r#"
        (function (obj, _mode) {
          const out = [];
          for (const k of Object.getOwnPropertyNames(obj)) {
            // Skip array-index keys.
            const n = +k;
            if (Number.isInteger(n) && n >= 0 && String(n) === k) continue;
            out.push(k);
          }
          return out;
        })
    "#;
    let go_fn: rquickjs::Function<'_> = ctx.eval(go_src)?;
    b.set("getOwnNonIndexProperties", go_fn)?;

    b.set("isInsideNodeModules", Func::from(|| -> bool { false }))?;

    // privateSymbols — Node uses these for transferable detach keys. A plain
    // Symbol works for us since we don't enforce the protections.
    let priv_syms: rquickjs::Object<'_> = ctx.eval(r#"
        ({ untransferable_object_private_symbol: Symbol('untransferable') })
    "#)?;
    b.set("privateSymbols", priv_syms)?;

    Ok(b)
}

pub fn make_config(ctx: Ctx<'_>) -> Result<Object<'_>> {
    let b = Object::new(ctx.clone())?;
    // hasIntl=false skips the ICU code path in buffer.js.
    b.set("hasIntl", false)?;
    Ok(b)
}
