use pyret_attempt2::{Parser, Expr, Program, Loc};
use pyret_attempt2::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, Read};
use num_bigint::BigInt;
use num_traits::{Zero, ToPrimitive, One};

/// Convert a Loc to Pyret's srcloc string representation
fn loc_to_srcloc_string(loc: &Loc) -> String {
    format!(
        "srcloc(\"{}\", {}, {}, {}, {}, {}, {})",
        loc.source,
        loc.start_line,
        loc.start_column,
        loc.start_char,
        loc.end_line,
        loc.end_column,
        loc.end_char
    )
}

/// Convert our AST to Pyret's JSON format (no locations, specific field names)
/// Convert a float to a fraction string representation
/// E.g., 1.5 -> "3/2", 0.25 -> "1/4"
fn decimal_string_to_fraction(s: &str) -> Option<(i64, i64)> {
    // Parse a decimal string like "2.034" to a fraction (2034, 1000)
    // Returns None if it doesn't contain a decimal point or can't be parsed

    let (negative, s) = if s.starts_with('-') {
        (true, &s[1..])
    } else {
        (false, s)
    };

    if !s.contains('.') {
        return None; // Not a decimal
    }

    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return None;
    }

    let integer_part = if parts[0].is_empty() { "0" } else { parts[0] };
    let decimal_part = parts[1];

    // Convert to fraction: e.g., "2.034" -> 2034/1000
    let num_digits = decimal_part.len();

    // Check if the number of digits would cause overflow in i64
    // 10^18 is the largest power of 10 that fits in i64
    if num_digits > 18 {
        return None; // Too many decimal places for i64, caller should use fallback
    }

    let denominator = 10_i64.pow(num_digits as u32);

    // Combine integer and decimal parts
    let numerator_str = format!("{}{}", integer_part, decimal_part);
    let numerator: i64 = numerator_str.parse().ok()?;
    let numerator = if negative { -numerator } else { numerator };

    Some((numerator, denominator))
}

/// Expand scientific notation like "1e300" to exact integer string "1000...000"
/// Returns None if the result wouldn't be an exact integer
fn expand_scientific_notation(s: &str) -> Option<String> {
    // Parse scientific notation: e.g., "1e300", "1.5e10", "-2e5"
    let s = s.trim();
    let (s, negative) = if s.starts_with('-') {
        (&s[1..], true)
    } else {
        (s, false)
    };

    // Split on 'e' or 'E'
    let parts: Vec<&str> = if s.contains('e') {
        s.split('e').collect()
    } else if s.contains('E') {
        s.split('E').collect()
    } else {
        return None;
    };

    if parts.len() != 2 {
        return None;
    }

    let mantissa_str = parts[0];
    let exponent: i32 = parts[1].parse().ok()?;

    // For positive exponents, we can expand to exact integer if there's no fractional part
    // or if the exponent is large enough to eliminate it
    if exponent >= 0 {
        // Parse mantissa
        if mantissa_str.contains('.') {
            let dot_pos = mantissa_str.find('.')?;
            let int_part = &mantissa_str[..dot_pos];
            let frac_part = &mantissa_str[dot_pos + 1..];
            let frac_len = frac_part.len() as i32;

            // If exponent >= frac_len, we can make an exact integer
            if exponent >= frac_len {
                let combined = format!("{}{}", int_part, frac_part);
                let zeros_to_add = exponent - frac_len;
                let mut result = format!("{}{}", combined, "0".repeat(zeros_to_add as usize));
                // Strip leading zeros (e.g., "0001000..." -> "1000...")
                result = result.trim_start_matches('0').to_string();
                // Handle the case where the result is all zeros
                if result.is_empty() {
                    result = "0".to_string();
                }
                return Some(if negative { format!("-{}", result) } else { result });
            } else {
                // Would have a fractional part
                return None;
            }
        } else {
            // No decimal point - simple case
            let result = format!("{}{}", mantissa_str, "0".repeat(exponent as usize));
            return Some(if negative { format!("-{}", result) } else { result });
        }
    } else {
        // Negative exponent - create a fraction
        // For example: 1.5e-300 -> "15/1000...000" (301 zeros)
        // For example: 0.001e-300 -> "1/1000...000" (303 zeros)

        let exponent_abs = (-exponent) as usize;

        // Parse mantissa to get numerator
        let (numerator, decimal_places) = if mantissa_str.contains('.') {
            let dot_pos = mantissa_str.find('.')?;
            let int_part = &mantissa_str[..dot_pos];
            let frac_part = &mantissa_str[dot_pos + 1..];
            let decimal_places = frac_part.len();

            // Combine integer and fractional parts: 1.5 -> 15
            let combined = format!("{}{}", int_part, frac_part);
            // Strip leading zeros
            let numerator = combined.trim_start_matches('0').to_string();
            let numerator = if numerator.is_empty() { "0".to_string() } else { numerator };
            (numerator, decimal_places)
        } else {
            (mantissa_str.to_string(), 0)
        };

        // Denominator is 10^(exponent_abs + decimal_places)
        let total_exponent = exponent_abs + decimal_places;
        let denominator = format!("1{}", "0".repeat(total_exponent));

        // Simplify the fraction using GCD
        use num_bigint::BigInt;
        use std::str::FromStr;

        let num_bigint = BigInt::from_str(&numerator).ok()?;
        let den_bigint = BigInt::from_str(&denominator).ok()?;
        let gcd = gcd_bigint(&num_bigint, &den_bigint);

        let simplified_num = &num_bigint / &gcd;
        let simplified_den = &den_bigint / &gcd;

        // Return as fraction string "numerator/denominator"
        let result = format!("{}/{}", simplified_num, simplified_den);
        Some(if negative { format!("-{}", result) } else { result })
    }
}

fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.clone();
    let mut b = b.clone();

    // Make positive
    if a < BigInt::zero() {
        a = -a;
    }
    if b < BigInt::zero() {
        b = -b;
    }

    while !b.is_zero() {
        let temp = b.clone();
        b = &a % &b;
        a = temp;
    }
    a
}

fn decimal_string_to_fraction_bigint(s: &str) -> Option<(BigInt, BigInt)> {
    // Parse decimal string using BigInt for arbitrary precision

    let (negative, s) = if s.starts_with('-') {
        (true, &s[1..])
    } else {
        (false, s)
    };

    if !s.contains('.') {
        return None;
    }

    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return None;
    }

    let integer_part = if parts[0].is_empty() { "0" } else { parts[0] };
    let decimal_part = parts[1];

    // Convert to fraction using BigInt
    let num_digits = decimal_part.len();
    let ten = BigInt::from(10);
    let denominator = ten.pow(num_digits as u32);

    // Combine integer and decimal parts
    let numerator_str = format!("{}{}", integer_part, decimal_part);
    let numerator: BigInt = numerator_str.parse().ok()?;
    let numerator = if negative { -numerator } else { numerator };

    Some((numerator, denominator))
}

fn decimal_string_to_fraction_with_simplification(s: &str) -> String {
    // Convert decimal string to simplified fraction string
    // Preserves arbitrary precision by working with strings

    // Try i64 first for performance
    if let Some((mut num, mut den)) = decimal_string_to_fraction(s) {
        // Simplify the fraction
        let g = gcd(num, den);
        num /= g;
        den /= g;

        if den == 1 {
            return format!("{}", num);
        } else {
            return format!("{}/{}", num, den);
        }
    }

    // Fall back to BigInt for very large decimals
    if let Some((num, den)) = decimal_string_to_fraction_bigint(s) {
        // Simplify using BigInt GCD
        let g = gcd_bigint(&num, &den);
        let simplified_num = &num / &g;
        let simplified_den = &den / &g;

        if simplified_den == BigInt::from(1) {
            format!("{}", simplified_num)
        } else {
            format!("{}/{}", simplified_num, simplified_den)
        }
    } else {
        // Not a decimal, return as-is
        s.to_string()
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn float_to_fraction_string(f: f64) -> String {
    // If it's an integer, format it carefully to avoid precision loss
    if f.fract() == 0.0 {
        // For very large numbers, f64 loses precision
        // So we need to format them more carefully
        if f.abs() < 9007199254740992.0 {  // 2^53 - max safe integer in f64
            return format!("{}", f as i64);
        } else {
            // For numbers beyond f64's precision, format as float
            // This will show the number as stored (possibly with precision loss)
            return format!("{:.0}", f);
        }
    }

    // Convert to string first to avoid floating point precision errors
    let f_str = format!("{}", f);

    // Try to parse as decimal string first (more accurate)
    if let Some((mut num, mut den)) = decimal_string_to_fraction(&f_str) {
        // Simplify the fraction
        let g = gcd(num, den);
        num /= g;
        den /= g;

        if den == 1 {
            return format!("{}", num);
        } else {
            return format!("{}/{}", num, den);
        }
    }

    // Fallback to old algorithm for non-decimal representations (shouldn't happen)
    let sign = if f < 0.0 { -1.0 } else { 1.0 };
    let f = f.abs();

    let mut num = f;
    let mut den = 1.0;

    while num.fract() != 0.0 && den < 1e10 {
        num *= 10.0;
        den *= 10.0;
    }

    let mut num = num as i64;
    let mut den = den as i64;

    let g = gcd(num, den);
    num /= g;
    den /= g;

    num = (num as f64 * sign) as i64;

    if den == 1 {
        format!("{}", num)
    } else {
        format!("{}/{}", num, den)
    }
}

fn format_scientific_notation(sci_str: &str, _value: f64) -> String {
    // For now, just preserve the original scientific notation with lowercase 'e'
    // Future: could normalize format (e.g., ensure mantissa is between 1 and 10)
    sci_str.replace('E', "e")
}

fn expr_to_pyret_json(expr: &Expr) -> Value {
    match expr {
        Expr::SNum { value, .. } => {
            // Value is stored as a string to support arbitrary precision
            let value_str = {
                // Rough numbers (starting with ~) need normalization
                if value.starts_with('~') {
                    let without_tilde = value.strip_prefix('~').unwrap();

                    // Parse to f64 to truncate to IEEE 754 double precision (17 significant digits)
                    // This ensures we match Pyret's behavior
                    if let Ok(n) = without_tilde.parse::<f64>() {
                        // Format back to string - Rust's default f64 formatting gives us the right precision
                        let normalized = format!("~{}", n);

                        // Check if the normalized form is too long (e.g., ~0.000...005 with 324 zeros)
                        // Convert very long decimals to scientific notation
                        if normalized.len() > 50 {
                            let sci = format!("{:e}", n);
                            // Normalize: add + for positive exponents
                            let sci_normalized = if sci.contains("e") && !sci.contains("e-") {
                                sci.replace("e", "e+")
                            } else {
                                sci
                            };
                            format!("~{}", sci_normalized)
                        } else {
                            normalized
                        }
                    } else {
                        // Parse failed - keep original
                        value.clone()
                    }
                }
                // For decimals (without scientific notation), convert to fraction
                // Use the original string to preserve precision for large decimals
                // For scientific notation, expand to exact integer if possible
                // For integers, use as-is
                else if value.contains('e') || value.contains('E') {
                    // Try to expand scientific notation to exact integer (e.g., 1e300 -> "1000...000")
                    if let Some(expanded) = expand_scientific_notation(value) {
                        expanded
                    } else {
                        // Can't expand to exact integer - parse to f64 and convert to fraction
                        if let Ok(n) = value.parse::<f64>() {
                            float_to_fraction_string(n)
                        } else {
                            value.clone() // Fallback if parse fails
                        }
                    }
                } else if value.contains('.') {
                    // Regular decimal - convert to simplified fraction
                    decimal_string_to_fraction_with_simplification(value)
                } else {
                    // Integer - use as-is (supports arbitrary precision!)
                    value.clone()
                }
            };

            // Normalize negative zero: -0 -> 0, ~-0 -> ~0
            let value_str = if value_str == "-0" {
                "0".to_string()
            } else if value_str == "~-0" {
                "~0".to_string()
            } else {
                value_str
            };

            json!({
                "type": "s-num",
                "value": value_str
            })
        }
        Expr::SStr { s, .. } => {
            json!({
                "type": "s-str",
                "value": s
            })
        }
        Expr::SBool { b, .. } => {
            json!({
                "type": "s-bool",
                "value": b
            })
        }
        Expr::SId { id, .. } => {
            // Names need special handling - for now just extract the string
            json!({
                "type": "s-id",
                "id": name_to_pyret_json(id)
            })
        }
        Expr::SOp { op, left, right, .. } => {
            json!({
                "type": "s-op",
                "op": op,
                "left": expr_to_pyret_json(left),
                "right": expr_to_pyret_json(right)
            })
        }
        Expr::SParen { expr, .. } => {
            json!({
                "type": "s-paren",
                "expr": expr_to_pyret_json(expr)
            })
        }
        Expr::SApp { _fun, args, .. } => {
            json!({
                "type": "s-app",
                "fun": expr_to_pyret_json(_fun),
                "args": args.iter().map(|e| expr_to_pyret_json(e.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::SInstantiate { expr, params, .. } => {
            json!({
                "type": "s-instantiate",
                "expr": expr_to_pyret_json(expr),
                "params": params.iter().map(|a| ann_to_pyret_json(a)).collect::<Vec<_>>()
            })
        }
        Expr::SConstruct { modifier, constructor, values, .. } => {
            json!({
                "type": "s-construct",
                "modifier": modifier_to_pyret_json(modifier),
                "constructor": expr_to_pyret_json(constructor),
                "values": values.iter().map(|e| expr_to_pyret_json(e.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::SDot { obj, field, .. } => {
            json!({
                "type": "s-dot",
                "obj": expr_to_pyret_json(obj),
                "field": field
            })
        }
        Expr::SGetBang { obj, field, .. } => {
            json!({
                "type": "s-get-bang",
                "obj": expr_to_pyret_json(obj),
                "field": field
            })
        }
        Expr::SBracket { obj, field, .. } => {
            json!({
                "type": "s-bracket",
                "obj": expr_to_pyret_json(obj),
                "field": expr_to_pyret_json(field)
            })
        }
        Expr::SCheck { name, body, keyword_check, .. } => {
            json!({
                "type": "s-check",
                "name": name.as_ref(),
                "body": expr_to_pyret_json(body),
                "keyword-check": keyword_check
            })
        }
        Expr::SCheckTest { op, refinement, left, right, cause, .. } => {
            json!({
                "type": "s-check-test",
                "op": check_op_to_pyret_json(op),
                "refinement": refinement.as_ref().map(|e| expr_to_pyret_json(e)),
                "left": expr_to_pyret_json(left),
                "right": right.as_ref().map(|e| expr_to_pyret_json(e)),
                "cause": cause.as_ref().map(|e| expr_to_pyret_json(e))
            })
        }
        Expr::SReactor { fields, .. } => {
            json!({
                "type": "s-reactor",
                "fields": fields.iter().map(|f| member_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::SObj { fields, .. } => {
            json!({
                "type": "s-obj",
                "fields": fields.iter().map(|f| member_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::SFun { name, params, args, ann, doc, body, check, check_loc, blocky, .. } => {
            json!({
                "type": "s-fun",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann),
                "doc": doc,
                "body": expr_to_pyret_json(body),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "blocky": blocky
            })
        }
        Expr::SMethod { name, params, args, ann, doc, body, check, check_loc, blocky, .. } => {
            json!({
                "type": "s-method",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann),
                "doc": doc,
                "body": expr_to_pyret_json(body),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "blocky": blocky
            })
        }
        Expr::SLam { name, params, args, ann, doc, body, check, check_loc, blocky, .. } => {
            json!({
                "type": "s-lam",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann),
                "doc": doc,
                "body": expr_to_pyret_json(body),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "blocky": blocky
            })
        }
        Expr::SBlock { stmts, .. } => {
            json!({
                "type": "s-block",
                "stmts": stmts.iter().map(|s| expr_to_pyret_json(s)).collect::<Vec<_>>()
            })
        }
        Expr::SUserBlock { body, .. } => {
            json!({
                "type": "s-user-block",
                "body": expr_to_pyret_json(body)
            })
        }
        Expr::STuple { fields, .. } => {
            json!({
                "type": "s-tuple",
                "fields": fields.iter().map(|f| expr_to_pyret_json(f.as_ref())).collect::<Vec<_>>()
            })
        }
        Expr::STupleGet { tup, index, .. } => {
            json!({
                "type": "s-tuple-get",
                "tup": expr_to_pyret_json(tup),
                "index": index
            })
        }
        Expr::SIf { branches, blocky, .. } => {
            json!({
                "type": "s-if",
                "branches": branches.iter().map(|b| if_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SIfElse { branches, _else, blocky, .. } => {
            json!({
                "type": "s-if-else",
                "branches": branches.iter().map(|b| if_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else),
                "blocky": blocky
            })
        }
        Expr::SIfPipe { branches, blocky, .. } => {
            json!({
                "type": "s-if-pipe",
                "branches": branches.iter().map(|b| if_pipe_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SIfPipeElse { branches, _else, blocky, .. } => {
            json!({
                "type": "s-if-pipe-else",
                "branches": branches.iter().map(|b| if_pipe_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else),
                "blocky": blocky
            })
        }
        Expr::SFor { iterator, bindings, ann, body, blocky, .. } => {
            json!({
                "type": "s-for",
                "iterator": expr_to_pyret_json(iterator),
                "bindings": bindings.iter().map(|b| for_bind_to_pyret_json(b)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann),
                "body": expr_to_pyret_json(body),
                "blocky": blocky
            })
        }
        Expr::SCases { typ, val, branches, blocky, .. } => {
            json!({
                "type": "s-cases",
                "typ": ann_to_pyret_json(typ),
                "val": expr_to_pyret_json(val),
                "branches": branches.iter().map(|b| cases_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SCasesElse { typ, val, branches, _else, blocky, .. } => {
            json!({
                "type": "s-cases-else",
                "typ": ann_to_pyret_json(typ),
                "val": expr_to_pyret_json(val),
                "branches": branches.iter().map(|b| cases_branch_to_pyret_json(b)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else),
                "blocky": blocky
            })
        }
        Expr::SLetExpr { binds, body, blocky, .. } => {
            json!({
                "type": "s-let-expr",
                "binds": binds.iter().map(|b| let_bind_to_pyret_json(b)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body),
                "blocky": blocky
            })
        }
        Expr::SLet { name, value, keyword_val, .. } => {
            json!({
                "type": "s-let",
                "name": bind_to_pyret_json(name),
                "value": expr_to_pyret_json(value),
                "keyword-val": keyword_val
            })
        }
        Expr::SLetrec { binds, body, blocky, .. } => {
            json!({
                "type": "s-letrec",
                "binds": binds.iter().map(|b| letrec_bind_to_pyret_json(b)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body),
                "blocky": blocky
            })
        }
        Expr::SVar { name, value, .. } => {
            json!({
                "type": "s-var",
                "name": bind_to_pyret_json(name),
                "value": expr_to_pyret_json(value)
            })
        }
        Expr::SRec { name, value, .. } => {
            json!({
                "type": "s-rec",
                "name": bind_to_pyret_json(name),
                "value": expr_to_pyret_json(value)
            })
        }
        Expr::SAssign { id, value, .. } => {
            json!({
                "type": "s-assign",
                "id": name_to_pyret_json(id),
                "value": expr_to_pyret_json(value)
            })
        }
        Expr::SWhen { test, block, blocky, .. } => {
            json!({
                "type": "s-when",
                "test": expr_to_pyret_json(test),
                "block": expr_to_pyret_json(block),
                "blocky": blocky
            })
        }
        Expr::SType { name, params, ann, .. } => {
            json!({
                "type": "s-type",
                "name": name_to_pyret_json(name),
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann)
            })
        }
        Expr::SNewtype { name, namet, .. } => {
            json!({
                "type": "s-newtype",
                "name": name_to_pyret_json(name),
                "namet": name_to_pyret_json(namet)
            })
        }
        Expr::SData { name, params, mixins, variants, shared_members, check_loc, check, .. } => {
            json!({
                "type": "s-data",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "mixins": mixins.iter().map(|m| expr_to_pyret_json(m)).collect::<Vec<_>>(),
                "variants": variants.iter().map(|v| variant_to_pyret_json(v)).collect::<Vec<_>>(),
                "shared-members": shared_members.iter().map(|m| member_to_pyret_json(m)).collect::<Vec<_>>(),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c))
            })
        }
        Expr::SDataExpr { name, params, mixins, variants, shared_members, check_loc, check, .. } => {
            json!({
                "type": "s-data-expr",
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "mixins": mixins.iter().map(|m| expr_to_pyret_json(m)).collect::<Vec<_>>(),
                "variants": variants.iter().map(|v| variant_to_pyret_json(v)).collect::<Vec<_>>(),
                "shared-members": shared_members.iter().map(|m| member_to_pyret_json(m)).collect::<Vec<_>>(),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c))
            })
        }
        Expr::SFrac { num, den, .. } => {
            json!({
                "type": "s-frac",
                "num": num.to_string(),
                "den": den.to_string()
            })
        }
        Expr::SRfrac { num, den, .. } => {
            json!({
                "type": "s-rfrac",
                "num": num.to_string(),
                "den": den.to_string()
            })
        }
        Expr::SExtend { supe, fields, .. } => {
            json!({
                "type": "s-extend",
                "super": expr_to_pyret_json(supe),
                "fields": fields.iter().map(|f| member_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::SUpdate { supe, fields, .. } => {
            json!({
                "type": "s-update",
                "super": expr_to_pyret_json(supe),
                "fields": fields.iter().map(|f| member_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::SSpyBlock { message, contents, .. } => {
            json!({
                "type": "s-spy-block",
                "message": message.as_ref().map(|m| expr_to_pyret_json(m)),
                "contents": contents.iter().map(|f| spy_field_to_pyret_json(f)).collect::<Vec<_>>()
            })
        }
        Expr::STable { headers, rows, .. } => {
            json!({
                "type": "s-table",
                "headers": headers.iter().map(|h| json!({
                    "type": "s-field-name",
                    "name": h.name,
                    "value": ann_to_pyret_json(&h.ann)
                })).collect::<Vec<_>>(),
                "rows": rows.iter().map(|r| json!({
                    "type": "s-table-row",
                    "elems": r.elems.iter().map(|e| expr_to_pyret_json(e)).collect::<Vec<_>>()
                })).collect::<Vec<_>>()
            })
        }
        Expr::SLoadTable { headers, spec, .. } => {
            json!({
                "type": "s-load-table",
                "headers": headers.iter().map(|h| json!({
                    "type": "s-field-name",
                    "name": h.name,
                    "value": ann_to_pyret_json(&h.ann)
                })).collect::<Vec<_>>(),
                "spec": spec.iter().map(|s| load_table_spec_to_pyret_json(s)).collect::<Vec<_>>()
            })
        }
        Expr::STableExtract { column, table, .. } => {
            json!({
                "type": "s-table-extract",
                "column": name_to_pyret_json(column),
                "table": expr_to_pyret_json(table)
            })
        }
        Expr::SContract { name, params, ann, .. } => {
            json!({
                "type": "s-contract",
                "name": name_to_pyret_json(name),
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann)
            })
        }
        Expr::STemplate { .. } => {
            json!({
                "type": "s-template"
            })
        }
        _ => {
            json!({
                "type": "UNSUPPORTED",
                "debug": format!("{:?}", expr)
            })
        }
    }
}

fn member_to_pyret_json(member: &pyret_attempt2::Member) -> Value {
    use pyret_attempt2::Member;
    match member {
        Member::SDataField { name, value, .. } => {
            json!({
                "type": "s-data-field",
                "name": name,
                "value": expr_to_pyret_json(value)
            })
        }
        Member::SMutableField { name, ann, value, .. } => {
            json!({
                "type": "s-mutable-field",
                "name": name,
                "ann": ann_to_pyret_json(ann),
                "value": expr_to_pyret_json(value)
            })
        }
        Member::SMethodField {
            name,
            params,
            args,
            ann,
            doc,
            body,
            check_loc,
            check,
            blocky,
            ..
        } => {
            json!({
                "type": "s-method-field",
                "ann": ann_to_pyret_json(ann),
                "args": args.iter().map(|a| bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "blocky": blocky,
                "body": expr_to_pyret_json(body),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c)),
                "check-loc": check_loc.as_ref().map(|loc| loc_to_srcloc_string(loc)),
                "doc": doc,
                "name": name,
                "params": params.iter().map(|p| name_to_pyret_json(p)).collect::<Vec<_>>()
            })
        }
    }
}

fn spy_field_to_pyret_json(field: &pyret_attempt2::SpyField) -> Value {
    json!({
        "type": "s-spy-expr",
        "name": field.name.as_ref(),
        "value": expr_to_pyret_json(&field.value),
        "implicit-label": field.implicit_label
    })
}

fn variant_to_pyret_json(variant: &pyret_attempt2::Variant) -> Value {
    use pyret_attempt2::Variant;
    match variant {
        Variant::SVariant { name, members, with_members, .. } => {
            json!({
                "type": "s-variant",
                "name": name,
                "members": members.iter().map(|m| variant_member_to_pyret_json(m)).collect::<Vec<_>>(),
                "with-members": with_members.iter().map(|m| member_to_pyret_json(m)).collect::<Vec<_>>()
            })
        }
        Variant::SSingletonVariant { name, with_members, .. } => {
            json!({
                "type": "s-singleton-variant",
                "name": name,
                "with-members": with_members.iter().map(|m| member_to_pyret_json(m)).collect::<Vec<_>>()
            })
        }
    }
}

fn variant_member_to_pyret_json(member: &pyret_attempt2::VariantMember) -> Value {
    json!({
        "type": "s-variant-member",
        "member-type": variant_member_type_to_pyret_json(&member.member_type),
        "bind": bind_to_pyret_json(&member.bind)
    })
}

fn variant_member_type_to_pyret_json(member_type: &pyret_attempt2::VariantMemberType) -> Value {
    use pyret_attempt2::VariantMemberType;
    match member_type {
        VariantMemberType::SNormal => json!("s-normal"),
        VariantMemberType::SMutable => json!("s-mutable"),
    }
}

fn ann_to_pyret_json(ann: &pyret_attempt2::Ann) -> Value {
    use pyret_attempt2::Ann;
    match ann {
        Ann::ABlank => json!({"type": "a-blank"}),
        Ann::AAny { .. } => json!({"type": "a-any"}),
        Ann::AName { id, .. } => json!({
            "type": "a-name",
            "id": name_to_pyret_json(id)
        }),
        Ann::ADot { obj, field, .. } => json!({
            "type": "a-dot",
            "obj": name_to_pyret_json(obj),
            "field": field
        }),
        Ann::AArrow { args, ret, use_parens, .. } => json!({
            "type": "a-arrow",
            "args": args.iter().map(ann_to_pyret_json).collect::<Vec<_>>(),
            "ret": ann_to_pyret_json(ret),
            "use-parens": use_parens
        }),
        Ann::AApp { ann, args, .. } => json!({
            "type": "a-app",
            "ann": ann_to_pyret_json(ann),
            "args": args.iter().map(ann_to_pyret_json).collect::<Vec<_>>()
        }),
        Ann::APred { ann, exp, .. } => json!({
            "type": "a-pred",
            "ann": ann_to_pyret_json(ann),
            "exp": expr_to_pyret_json(exp)
        }),
        Ann::ATuple { fields, .. } => json!({
            "type": "a-tuple",
            "fields": fields.iter().map(ann_to_pyret_json).collect::<Vec<_>>()
        }),
        Ann::ARecord { fields, .. } => json!({
            "type": "a-record",
            "fields": fields.iter().map(|f| json!({
                "type": "a-field",
                "name": f.name,
                "ann": ann_to_pyret_json(&f.ann)
            })).collect::<Vec<_>>()
        }),
        _ => json!({
            "type": "UNSUPPORTED",
            "debug": format!("{:?}", ann)
        })
    }
}

fn check_op_to_pyret_json(op: &pyret_attempt2::CheckOp) -> Value {
    use pyret_attempt2::CheckOp;
    match op {
        CheckOp::SOpIs { .. } => json!({"type": "s-op-is"}),
        CheckOp::SOpIsRoughly { .. } => json!({"type": "s-op-is-roughly"}),
        CheckOp::SOpIsNot { .. } => json!({"type": "s-op-is-not"}),
        CheckOp::SOpIsNotRoughly { .. } => json!({"type": "s-op-is-not-roughly"}),
        CheckOp::SOpIsOp { op, .. } => json!({"type": "s-op-is-op", "op": op}),
        CheckOp::SOpIsNotOp { op, .. } => json!({"type": "s-op-is-not-op", "op": op}),
        CheckOp::SOpSatisfies { .. } => json!({"type": "s-op-satisfies"}),
        CheckOp::SOpSatisfiesNot { .. } => json!({"type": "s-op-satisfies-not"}),
        CheckOp::SOpRaises { .. } => json!({"type": "s-op-raises"}),
        CheckOp::SOpRaisesOther { .. } => json!({"type": "s-op-raises-other"}),
        CheckOp::SOpRaisesNot { .. } => json!({"type": "s-op-raises-not"}),
        CheckOp::SOpRaisesSatisfies { .. } => json!({"type": "s-op-raises-satisfies"}),
        CheckOp::SOpRaisesViolates { .. } => json!({"type": "s-op-raises-violates"}),
    }
}

fn modifier_to_pyret_json(modifier: &pyret_attempt2::ConstructModifier) -> Value {
    use pyret_attempt2::ConstructModifier;
    match modifier {
        ConstructModifier::SConstructNormal => json!("s-construct-normal"),
        ConstructModifier::SConstructLazy => json!("s-construct-lazy"),
    }
}

fn name_to_pyret_json(name: &pyret_attempt2::Name) -> Value {
    use pyret_attempt2::Name;
    match name {
        Name::SUnderscore { .. } => {
            json!({
                "type": "s-underscore"
            })
        }
        Name::SName { s, .. } => {
            json!({
                "type": "s-name",
                "name": s
            })
        }
        Name::SGlobal { s } => {
            json!({
                "type": "s-global",
                "name": s
            })
        }
        Name::SModuleGlobal { s } => {
            json!({
                "type": "s-module-global",
                "name": s
            })
        }
        Name::STypeGlobal { s } => {
            json!({
                "type": "s-type-global",
                "name": s
            })
        }
        Name::SAtom { base, serial } => {
            json!({
                "type": "s-atom",
                "base": base,
                "serial": serial
            })
        }
    }
}

fn bind_to_pyret_json(bind: &pyret_attempt2::Bind) -> Value {
    use pyret_attempt2::Bind;
    match bind {
        Bind::SBind { id, ann, shadows, .. } => {
            json!({
                "type": "s-bind",
                "name": name_to_pyret_json(id),
                "ann": ann_to_pyret_json(ann),
                "shadows": shadows
            })
        }
        Bind::STupleBind { fields, as_name, .. } => {
            json!({
                "type": "s-tuple-bind",
                "fields": fields.iter().map(|f| bind_to_pyret_json(f)).collect::<Vec<_>>(),
                "as-name": as_name.as_ref().map(|n| bind_to_pyret_json(n))
            })
        }
    }
}

fn if_branch_to_pyret_json(branch: &pyret_attempt2::IfBranch) -> Value {
    json!({
        "type": "s-if-branch",
        "test": expr_to_pyret_json(&branch.test),
        "body": expr_to_pyret_json(&branch.body)
    })
}

fn if_pipe_branch_to_pyret_json(branch: &pyret_attempt2::IfPipeBranch) -> Value {
    json!({
        "type": "s-if-pipe-branch",
        "test": expr_to_pyret_json(&branch.test),
        "body": expr_to_pyret_json(&branch.body)
    })
}

fn for_bind_to_pyret_json(for_bind: &pyret_attempt2::ForBind) -> Value {
    json!({
        "type": "s-for-bind",
        "bind": bind_to_pyret_json(&for_bind.bind),
        "value": expr_to_pyret_json(&for_bind.value)
    })
}

fn cases_branch_to_pyret_json(branch: &pyret_attempt2::CasesBranch) -> Value {
    use pyret_attempt2::CasesBranch;
    match branch {
        CasesBranch::SCasesBranch { name, args, body, .. } => {
            json!({
                "type": "s-cases-branch",
                "name": name,
                "args": args.iter().map(|a| cases_bind_to_pyret_json(a)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body)
            })
        }
        CasesBranch::SSingletonCasesBranch { name, body, .. } => {
            json!({
                "type": "s-singleton-cases-branch",
                "name": name,
                "body": expr_to_pyret_json(body)
            })
        }
    }
}

fn cases_bind_to_pyret_json(cases_bind: &pyret_attempt2::CasesBind) -> Value {
    json!({
        "type": "s-cases-bind",
        "field-type": format!("s-cases-bind-{}", match cases_bind.field_type {
            pyret_attempt2::CasesBindType::SNormal => "normal",
            pyret_attempt2::CasesBindType::SMutable => "mutable",
            pyret_attempt2::CasesBindType::SRef => "ref",
        }),
        "bind": bind_to_pyret_json(&cases_bind.bind)
    })
}

fn let_bind_to_pyret_json(let_bind: &pyret_attempt2::LetBind) -> Value {
    use pyret_attempt2::LetBind;
    match let_bind {
        LetBind::SLetBind { b, value, .. } => {
            json!({
                "type": "s-let-bind",
                "bind": bind_to_pyret_json(b),
                "value": expr_to_pyret_json(value)
            })
        }
        LetBind::SVarBind { b, value, .. } => {
            json!({
                "type": "s-var-bind",
                "bind": bind_to_pyret_json(b),
                "value": expr_to_pyret_json(value)
            })
        }
    }
}

fn letrec_bind_to_pyret_json(letrec_bind: &pyret_attempt2::LetrecBind) -> Value {
    json!({
        "type": "s-letrec-bind",
        "bind": bind_to_pyret_json(&letrec_bind.b),
        "value": expr_to_pyret_json(&letrec_bind.value)
    })
}

fn program_to_pyret_json(program: &Program) -> Value {
    let mut obj = json!({
        "type": "s-program",
        "provide": provide_to_pyret_json(&program._provide),
        "provided-types": provide_types_to_pyret_json(&program.provided_types),
        "provides": program.provides.iter().map(|p| provide_block_to_pyret_json(p)).collect::<Vec<_>>(),
        "imports": program.imports.iter().map(|i| import_to_pyret_json(i)).collect::<Vec<_>>(),
        "body": expr_to_pyret_json(&program.body)
    });

    // Add "use" field (set to null when not present, as Pyret does)
    obj["use"] = match &program._use {
        Some(use_stmt) => use_to_pyret_json(use_stmt),
        None => Value::Null,
    };

    obj
}

fn use_to_pyret_json(use_stmt: &pyret_attempt2::Use) -> Value {
    json!({
        "type": "s-use",
        "name": name_to_pyret_json(&use_stmt.name),
        "mod": import_type_to_pyret_json(&use_stmt.module)
    })
}

fn provide_to_pyret_json(provide: &pyret_attempt2::Provide) -> Value {
    use pyret_attempt2::Provide;
    match provide {
        Provide::SProvide { block, .. } => {
            json!({
                "type": "s-provide",
                "block": expr_to_pyret_json(block)
            })
        }
        Provide::SProvideAll { .. } => {
            json!({"type": "s-provide-all"})
        }
        Provide::SProvideNone { .. } => {
            json!({"type": "s-provide-none"})
        }
    }
}

fn a_field_to_pyret_json(field: &pyret_attempt2::AField) -> Value {
    json!({
        "type": "a-field",
        "name": &field.name,
        "ann": ann_to_pyret_json(&field.ann)
    })
}

fn provide_types_to_pyret_json(provide_types: &pyret_attempt2::ProvideTypes) -> Value {
    use pyret_attempt2::ProvideTypes;
    match provide_types {
        ProvideTypes::SProvideTypes { anns, .. } => {
            json!({
                "type": "s-provide-types",
                "anns": anns.iter().map(|a| a_field_to_pyret_json(a)).collect::<Vec<_>>()
            })
        }
        ProvideTypes::SProvideTypesAll { .. } => {
            json!({"type": "s-provide-types-all"})
        }
        ProvideTypes::SProvideTypesNone { .. } => {
            json!({"type": "s-provide-types-none"})
        }
    }
}

fn provide_block_to_pyret_json(provide_block: &pyret_attempt2::ProvideBlock) -> Value {
    json!({
        "type": "s-provide-block",
        "path": provide_block.path.iter().map(|n| name_to_pyret_json(n)).collect::<Vec<_>>(),
        "specs": provide_block.specs.iter().map(|s| provide_spec_to_pyret_json(s)).collect::<Vec<_>>()
    })
}

fn provide_spec_to_pyret_json(spec: &pyret_attempt2::ProvideSpec) -> Value {
    use pyret_attempt2::ProvideSpec;
    match spec {
        ProvideSpec::SProvideName { name, .. } => {
            json!({
                "type": "s-provide-name",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
        ProvideSpec::SProvideType { name, .. } => {
            json!({
                "type": "s-provide-type",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
        ProvideSpec::SProvideData { name, hidden, .. } => {
            json!({
                "type": "s-provide-data",
                "name-spec": name_spec_to_pyret_json(name),
                "hidden": hidden.iter().map(|h| name_to_pyret_json(h)).collect::<Vec<_>>()
            })
        }
        ProvideSpec::SProvideModule { name, .. } => {
            json!({
                "type": "s-provide-module",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
    }
}

fn import_to_pyret_json(import: &pyret_attempt2::Import) -> Value {
    use pyret_attempt2::Import;
    match import {
        Import::SInclude { import, .. } => {
            json!({
                "type": "s-include",
                "import-type": import_type_to_pyret_json(import)
            })
        }
        Import::SIncludeFrom { module_path, names, .. } => {
            json!({
                "type": "s-include-from",
                "mod": module_path.iter().map(|n| name_to_pyret_json(n)).collect::<Vec<_>>(),
                "specs": names.iter().map(|n| include_spec_to_pyret_json(n)).collect::<Vec<_>>()
            })
        }
        Import::SImport { import, name, .. } => {
            json!({
                "type": "s-import",
                "import-type": import_type_to_pyret_json(import),
                "name": name_to_pyret_json(name)
            })
        }
        Import::SImportFields { fields, import, .. } => {
            json!({
                "type": "s-import-fields",
                "fields": fields.iter().map(|f| name_to_pyret_json(f)).collect::<Vec<_>>(),
                "import-type": import_type_to_pyret_json(import)
            })
        }
        Import::SImportTypes { import, types, name, .. } => {
            json!({
                "type": "s-import-types",
                "import-type": import_type_to_pyret_json(import),
                "types": types.iter().map(|t| name_to_pyret_json(t)).collect::<Vec<_>>(),
                "name": name_to_pyret_json(name)
            })
        }
    }
}

fn import_type_to_pyret_json(import_type: &pyret_attempt2::ImportType) -> Value {
    use pyret_attempt2::ImportType;
    match import_type {
        ImportType::SConstImport { module, .. } => {
            json!({
                "type": "s-const-import",
                "mod": module
            })
        }
        ImportType::SSpecialImport { kind, args, .. } => {
            json!({
                "type": "s-special-import",
                "kind": kind,
                "args": args
            })
        }
    }
}

fn include_spec_to_pyret_json(spec: &pyret_attempt2::IncludeSpec) -> Value {
    use pyret_attempt2::IncludeSpec;
    match spec {
        IncludeSpec::SIncludeName { name, .. } => {
            json!({
                "type": "s-include-name",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
        IncludeSpec::SIncludeType { name, .. } => {
            json!({
                "type": "s-include-type",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
        IncludeSpec::SIncludeData { name, hidden, .. } => {
            json!({
                "type": "s-include-data",
                "name-spec": name_spec_to_pyret_json(name),
                "hidden": hidden.iter().map(|h| name_to_pyret_json(h)).collect::<Vec<_>>()
            })
        }
        IncludeSpec::SIncludeModule { name, .. } => {
            json!({
                "type": "s-include-module",
                "name-spec": name_spec_to_pyret_json(name)
            })
        }
    }
}

fn name_spec_to_pyret_json(name_spec: &pyret_attempt2::NameSpec) -> Value {
    use pyret_attempt2::NameSpec;
    match name_spec {
        NameSpec::SStar { hidden, .. } => {
            json!({
                "type": "s-star",
                "hidden": hidden.iter().map(|n| name_to_pyret_json(n)).collect::<Vec<_>>()
            })
        }
        NameSpec::SModuleRef { path, as_name, .. } => {
            json!({
                "type": "s-module-ref",
                "path": path.iter().map(|n| name_to_pyret_json(n)).collect::<Vec<_>>(),
                "as-name": as_name.as_ref().map(|n| name_to_pyret_json(n)).unwrap_or(Value::Null)
            })
        }
        NameSpec::SRemoteRef { uri, name, .. } => {
            json!({
                "type": "s-remote-ref",
                "uri": uri,
                "name": name_to_pyret_json(name)
            })
        }
        NameSpec::SLocalRef { name, .. } => {
            json!({
                "type": "s-local-ref",
                "name": name_to_pyret_json(name)
            })
        }
    }
}

fn load_table_spec_to_pyret_json(spec: &pyret_attempt2::LoadTableSpec) -> Value {
    use pyret_attempt2::LoadTableSpec;
    match spec {
        LoadTableSpec::SSanitize { name, sanitizer, .. } => {
            json!({
                "type": "s-sanitize",
                "name": name_to_pyret_json(name),
                "sanitizer": expr_to_pyret_json(sanitizer)
            })
        }
        LoadTableSpec::STableSrc { src, .. } => {
            json!({
                "type": "s-table-src",
                "src": expr_to_pyret_json(src)
            })
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let input = if args.len() > 1 {
        // Read from file
        fs::read_to_string(&args[1])?
    } else {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    let mut tokenizer = Tokenizer::new(&input);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, "input.arr".to_string());

    // Parse as full program
    let program = parser.parse_program()?;

    let json = program_to_pyret_json(&program);
    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
