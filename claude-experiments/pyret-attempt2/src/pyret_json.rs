use crate::{Expr, FileRegistry, Loc, Program};
use num_bigint::BigInt;
use num_traits::Zero;
use serde_json::{json, Value};

/// Convert a Loc to Pyret's srcloc string representation
pub fn loc_to_srcloc_string(loc: &Loc, registry: &FileRegistry) -> String {
    let filename = registry.get_name(loc.file_id).unwrap_or("unknown");
    format!(
        "srcloc(\"{}\", {}, {}, {}, {}, {}, {})",
        filename,
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
pub fn decimal_string_to_fraction(s: &str) -> Option<(i64, i64)> {
    // Parse a decimal string like "2.034" to a fraction (2034, 1000)
    // Returns None if it doesn't contain a decimal point or can't be parsed

    let (negative, s) = if let Some(stripped) = s.strip_prefix('-') {
        (true, stripped)
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
pub fn expand_scientific_notation(s: &str) -> Option<String> {
    // Parse scientific notation: e.g., "1e300", "1.5e10", "-2e5"
    let s = s.trim();
    let (s, negative) = if let Some(stripped) = s.strip_prefix('-') {
        (stripped, true)
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
                Some(if negative {
                    format!("-{}", result)
                } else {
                    result
                })
            } else {
                // Would have a fractional part
                None
            }
        } else {
            // No decimal point - simple case
            let result = format!("{}{}", mantissa_str, "0".repeat(exponent as usize));
            Some(if negative {
                format!("-{}", result)
            } else {
                result
            })
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
            let numerator = if numerator.is_empty() {
                "0".to_string()
            } else {
                numerator
            };
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
        Some(if negative {
            format!("-{}", result)
        } else {
            result
        })
    }
}

pub fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
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

pub fn decimal_string_to_fraction_bigint(s: &str) -> Option<(BigInt, BigInt)> {
    // Parse decimal string using BigInt for arbitrary precision

    let (negative, s) = if let Some(stripped) = s.strip_prefix('-') {
        (true, stripped)
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

pub fn decimal_string_to_fraction_with_simplification(s: &str) -> String {
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

pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub fn float_to_fraction_string(f: f64) -> String {
    // If it's an integer, format it carefully to avoid precision loss
    if f.fract() == 0.0 {
        // For very large numbers, f64 loses precision
        // So we need to format them more carefully
        if f.abs() < 9007199254740992.0 {
            // 2^53 - max safe integer in f64
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

pub fn normalize_scientific_notation(sci: &str) -> String {
    // Normalize scientific notation: lowercase 'e' and explicit '+' for positive exponents
    // e.g., "1.5e308" -> "1.5e+308", "1.5e-10" -> "1.5e-10"
    if sci.contains("e-") || sci.contains("E-") {
        // Already has a sign (negative)
        sci.replace('E', "e")
    } else if sci.contains('e') || sci.contains('E') {
        // Positive exponent - add explicit '+'
        sci.replace('E', "e").replace("e", "e+")
    } else {
        // No exponent
        sci.to_string()
    }
}

pub fn expr_to_pyret_json(expr: &Expr, registry: &FileRegistry) -> Value {
    match expr {
        Expr::SNum { value, .. } => {
            // Value is stored as a string to support arbitrary precision
            let value_str = {
                // Rough numbers (starting with ~) need normalization
                if value.starts_with('~') {
                    let without_tilde = value.strip_prefix('~').unwrap();

                    // If it contains scientific notation, may need to expand or keep it
                    // Pyret expands ~1e-5 to ~0.00001 but keeps ~1e-7 as ~1e-7
                    // The threshold appears to be exponent >= -6
                    if without_tilde.contains('e') || without_tilde.contains('E') {
                        // Parse to f64 to get the actual value
                        if let Ok(n) = without_tilde.parse::<f64>() {
                            // Determine if we should expand or use scientific notation
                            // Pyret expands for exponents >= -6 (e.g., 1e-5, 1e-6)
                            // but uses scientific notation for exponents < -6 (e.g., 1e-7, 1e-10)

                            // First check the exponent
                            let sci = format!("{:e}", n);
                            if let Some(e_pos) = sci.find('e') {
                                if let Ok(exponent) = sci[e_pos + 1..].parse::<i32>() {
                                    if (-6..0).contains(&exponent) {
                                        // Expand to decimal form
                                        // e.g., 1e-5 -> 0.00001
                                        format!("~{}", n)
                                    } else {
                                        // Use scientific notation with normalized format
                                        format!("~{}", normalize_scientific_notation(&sci))
                                    }
                                } else {
                                    // Can't parse exponent - use scientific notation with normalization
                                    format!("~{}", normalize_scientific_notation(&sci))
                                }
                            } else {
                                // No exponent found - just use the number
                                format!("~{}", n)
                            }
                        } else {
                            // Parse failed - just normalize the 'e' to lowercase
                            let normalized = without_tilde.replace('E', "e");
                            format!("~{}", normalized)
                        }
                    } else {
                        // Not scientific notation - parse to f64 to truncate to IEEE 754 double precision (17 significant digits)
                        // This ensures we match Pyret's behavior
                        if let Ok(n) = without_tilde.parse::<f64>() {
                            // Check the exponent to determine if we should use scientific notation
                            // Pyret uses scientific notation for exponents < -6 (e.g., 1e-7)
                            let sci = format!("{:e}", n);
                            if let Some(e_pos) = sci.find('e') {
                                if let Ok(exponent) = sci[e_pos + 1..].parse::<i32>() {
                                    if exponent < -6 {
                                        // Use scientific notation with normalized format
                                        format!("~{}", normalize_scientific_notation(&sci))
                                    } else {
                                        // Use decimal form
                                        format!("~{}", n)
                                    }
                                } else {
                                    // Can't parse exponent - use decimal form
                                    format!("~{}", n)
                                }
                            } else {
                                // No exponent found - use decimal form
                                format!("~{}", n)
                            }
                        } else {
                            // Parse failed - keep original
                            value.clone()
                        }
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
                "id": name_to_pyret_json(id, registry)
            })
        }
        Expr::SOp {
            op,
            op_l,
            left,
            right,
            ..
        } => {
            json!({
                "type": "s-op",
                "op": op,
                "op-l": loc_to_srcloc_string(op_l, registry),
                "left": expr_to_pyret_json(left, registry),
                "right": expr_to_pyret_json(right, registry)
            })
        }
        Expr::SParen { expr, .. } => {
            json!({
                "type": "s-paren",
                "expr": expr_to_pyret_json(expr, registry)
            })
        }
        Expr::SApp { _fun, args, .. } => {
            json!({
                "type": "s-app",
                "fun": expr_to_pyret_json(_fun, registry),
                "args": args.iter().map(|e| expr_to_pyret_json(e.as_ref(), registry)).collect::<Vec<_>>()
            })
        }
        Expr::SInstantiate { expr, params, .. } => {
            json!({
                "type": "s-instantiate",
                "expr": expr_to_pyret_json(expr, registry),
                "params": params.iter().map(|x| ann_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SConstruct {
            modifier,
            constructor,
            values,
            ..
        } => {
            json!({
                "type": "s-construct",
                "modifier": modifier_to_pyret_json(modifier, registry),
                "constructor": expr_to_pyret_json(constructor, registry),
                "values": values.iter().map(|e| expr_to_pyret_json(e.as_ref(), registry)).collect::<Vec<_>>()
            })
        }
        Expr::SDot { obj, field, .. } => {
            json!({
                "type": "s-dot",
                "obj": expr_to_pyret_json(obj, registry),
                "field": field
            })
        }
        Expr::SGetBang { obj, field, .. } => {
            json!({
                "type": "s-get-bang",
                "obj": expr_to_pyret_json(obj, registry),
                "field": field
            })
        }
        Expr::SBracket { obj, field, .. } => {
            json!({
                "type": "s-bracket",
                "obj": expr_to_pyret_json(obj, registry),
                "field": expr_to_pyret_json(field, registry)
            })
        }
        Expr::SCheck {
            name,
            body,
            keyword_check,
            ..
        } => {
            json!({
                "type": "s-check",
                "name": name.as_ref(),
                "body": expr_to_pyret_json(body, registry),
                "keyword-check": keyword_check
            })
        }
        Expr::SCheckTest {
            op,
            refinement,
            left,
            right,
            cause,
            ..
        } => {
            json!({
                "type": "s-check-test",
                "op": check_op_to_pyret_json(op, registry),
                "refinement": refinement.as_ref().map(|e| expr_to_pyret_json(e, registry)),
                "left": expr_to_pyret_json(left, registry),
                "right": right.as_ref().map(|e| expr_to_pyret_json(e, registry)),
                "cause": cause.as_ref().map(|e| expr_to_pyret_json(e, registry))
            })
        }
        Expr::SReactor { fields, .. } => {
            json!({
                "type": "s-reactor",
                "fields": fields.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SObj { fields, .. } => {
            json!({
                "type": "s-obj",
                "fields": fields.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SFun {
            name,
            params,
            args,
            ann,
            doc,
            body,
            check,
            check_loc,
            blocky,
            ..
        } => {
            json!({
                "type": "s-fun",
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "args": args.iter().map(|x| bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry),
                "doc": doc,
                "body": expr_to_pyret_json(body, registry),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry)),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "blocky": blocky
            })
        }
        Expr::SMethod {
            name,
            params,
            args,
            ann,
            doc,
            body,
            check,
            check_loc,
            blocky,
            ..
        } => {
            json!({
                "type": "s-method",
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "args": args.iter().map(|x| bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry),
                "doc": doc,
                "body": expr_to_pyret_json(body, registry),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry)),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "blocky": blocky
            })
        }
        Expr::SLam {
            name,
            params,
            args,
            ann,
            doc,
            body,
            check,
            check_loc,
            blocky,
            ..
        } => {
            json!({
                "type": "s-lam",
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "args": args.iter().map(|x| bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry),
                "doc": doc,
                "body": expr_to_pyret_json(body, registry),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry)),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "blocky": blocky
            })
        }
        Expr::SBlock { stmts, .. } => {
            json!({
                "type": "s-block",
                "stmts": stmts.iter().map(|s| expr_to_pyret_json(s, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SUserBlock { body, .. } => {
            json!({
                "type": "s-user-block",
                "body": expr_to_pyret_json(body, registry)
            })
        }
        Expr::STuple { fields, .. } => {
            json!({
                "type": "s-tuple",
                "fields": fields.iter().map(|f| expr_to_pyret_json(f.as_ref(), registry)).collect::<Vec<_>>()
            })
        }
        Expr::STupleGet {
            tup,
            index,
            index_loc,
            ..
        } => {
            json!({
                "type": "s-tuple-get",
                "tup": expr_to_pyret_json(tup, registry),
                "index": index,
                "index-loc": loc_to_srcloc_string(index_loc, registry)
            })
        }
        Expr::SIf {
            branches, blocky, ..
        } => {
            json!({
                "type": "s-if",
                "branches": branches.iter().map(|x| if_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SIfElse {
            branches,
            _else,
            blocky,
            ..
        } => {
            json!({
                "type": "s-if-else",
                "branches": branches.iter().map(|x| if_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else, registry),
                "blocky": blocky
            })
        }
        Expr::SIfPipe {
            branches, blocky, ..
        } => {
            json!({
                "type": "s-if-pipe",
                "branches": branches.iter().map(|x| if_pipe_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SIfPipeElse {
            branches,
            _else,
            blocky,
            ..
        } => {
            json!({
                "type": "s-if-pipe-else",
                "branches": branches.iter().map(|x| if_pipe_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else, registry),
                "blocky": blocky
            })
        }
        Expr::SFor {
            iterator,
            bindings,
            ann,
            body,
            blocky,
            ..
        } => {
            json!({
                "type": "s-for",
                "iterator": expr_to_pyret_json(iterator, registry),
                "bindings": bindings.iter().map(|x| for_bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry),
                "body": expr_to_pyret_json(body, registry),
                "blocky": blocky
            })
        }
        Expr::SCases {
            typ,
            val,
            branches,
            blocky,
            ..
        } => {
            json!({
                "type": "s-cases",
                "typ": ann_to_pyret_json(typ, registry),
                "val": expr_to_pyret_json(val, registry),
                "branches": branches.iter().map(|x| cases_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "blocky": blocky
            })
        }
        Expr::SCasesElse {
            typ,
            val,
            branches,
            _else,
            blocky,
            ..
        } => {
            json!({
                "type": "s-cases-else",
                "typ": ann_to_pyret_json(typ, registry),
                "val": expr_to_pyret_json(val, registry),
                "branches": branches.iter().map(|x| cases_branch_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "else": expr_to_pyret_json(_else, registry),
                "blocky": blocky
            })
        }
        Expr::SLetExpr {
            binds,
            body,
            blocky,
            ..
        } => {
            json!({
                "type": "s-let-expr",
                "binds": binds.iter().map(|x| let_bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body, registry),
                "blocky": blocky
            })
        }
        Expr::SLet {
            name,
            value,
            keyword_val,
            ..
        } => {
            json!({
                "type": "s-let",
                "name": bind_to_pyret_json(name, registry),
                "value": expr_to_pyret_json(value, registry),
                "keyword-val": keyword_val
            })
        }
        Expr::SLetrec {
            binds,
            body,
            blocky,
            ..
        } => {
            json!({
                "type": "s-letrec",
                "binds": binds.iter().map(|x| letrec_bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body, registry),
                "blocky": blocky
            })
        }
        Expr::SVar { name, value, .. } => {
            json!({
                "type": "s-var",
                "name": bind_to_pyret_json(name, registry),
                "value": expr_to_pyret_json(value, registry)
            })
        }
        Expr::SRec { name, value, .. } => {
            json!({
                "type": "s-rec",
                "name": bind_to_pyret_json(name, registry),
                "value": expr_to_pyret_json(value, registry)
            })
        }
        Expr::SAssign { id, value, .. } => {
            json!({
                "type": "s-assign",
                "id": name_to_pyret_json(id, registry),
                "value": expr_to_pyret_json(value, registry)
            })
        }
        Expr::SWhen {
            test,
            block,
            blocky,
            ..
        } => {
            json!({
                "type": "s-when",
                "test": expr_to_pyret_json(test, registry),
                "block": expr_to_pyret_json(block, registry),
                "blocky": blocky
            })
        }
        Expr::SType {
            name, params, ann, ..
        } => {
            json!({
                "type": "s-type",
                "name": name_to_pyret_json(name, registry),
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry)
            })
        }
        Expr::SNewtype { name, namet, .. } => {
            json!({
                "type": "s-newtype",
                "name": name_to_pyret_json(name, registry),
                "namet": name_to_pyret_json(namet, registry)
            })
        }
        Expr::SData {
            name,
            params,
            mixins,
            variants,
            shared_members,
            check_loc,
            check,
            ..
        } => {
            json!({
                "type": "s-data",
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "mixins": mixins.iter().map(|m| expr_to_pyret_json(m, registry)).collect::<Vec<_>>(),
                "variants": variants.iter().map(|x| variant_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "shared-members": shared_members.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry))
            })
        }
        Expr::SDataExpr {
            name,
            params,
            mixins,
            variants,
            shared_members,
            check_loc,
            check,
            ..
        } => {
            json!({
                "type": "s-data-expr",
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "mixins": mixins.iter().map(|m| expr_to_pyret_json(m, registry)).collect::<Vec<_>>(),
                "variants": variants.iter().map(|x| variant_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "shared-members": shared_members.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry))
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
                "super": expr_to_pyret_json(supe, registry),
                "fields": fields.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SUpdate { supe, fields, .. } => {
            json!({
                "type": "s-update",
                "super": expr_to_pyret_json(supe, registry),
                "fields": fields.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::SSpyBlock {
            message, contents, ..
        } => {
            json!({
                "type": "s-spy-block",
                "message": message.as_ref().map(|m| expr_to_pyret_json(m, registry)),
                "contents": contents.iter().map(|x| spy_field_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::STable { headers, rows, .. } => {
            json!({
                "type": "s-table",
                "headers": headers.iter().map(|h| json!({
                    "type": "s-field-name",
                    "name": h.name,
                    "value": ann_to_pyret_json(&h.ann, registry)
                })).collect::<Vec<_>>(),
                "rows": rows.iter().map(|r| json!({
                    "type": "s-table-row",
                    "elems": r.elems.iter().map(|e| expr_to_pyret_json(e, registry)).collect::<Vec<_>>()
                })).collect::<Vec<_>>()
            })
        }
        Expr::SLoadTable { headers, spec, .. } => {
            json!({
                "type": "s-load-table",
                "headers": headers.iter().map(|h| json!({
                    "type": "s-field-name",
                    "name": h.name,
                    "value": ann_to_pyret_json(&h.ann, registry)
                })).collect::<Vec<_>>(),
                "spec": spec.iter().map(|x| load_table_spec_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Expr::STableExtract { column, table, .. } => {
            json!({
                "type": "s-table-extract",
                "column": name_to_pyret_json(column, registry),
                "table": expr_to_pyret_json(table, registry)
            })
        }
        Expr::SContract {
            name, params, ann, ..
        } => {
            json!({
                "type": "s-contract",
                "name": name_to_pyret_json(name, registry),
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "ann": ann_to_pyret_json(ann, registry)
            })
        }
        Expr::STemplate { .. } => {
            json!({
                "type": "s-template"
            })
        }
        Expr::STableExtend {
            column_binds,
            extensions,
            ..
        } => {
            json!({
                "type": "s-table-extend",
                "column-binds": column_binds_to_pyret_json(column_binds, registry),
                "extensions": extensions.iter().map(|x| table_extend_field_to_pyret_json(x, registry)).collect::<Vec<_>>()
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

pub fn member_to_pyret_json(member: &crate::Member, registry: &FileRegistry) -> Value {
    use crate::Member;
    match member {
        Member::SDataField { name, value, .. } => {
            json!({
                "type": "s-data-field",
                "name": name,
                "value": expr_to_pyret_json(value, registry)
            })
        }
        Member::SMutableField {
            name, ann, value, ..
        } => {
            json!({
                "type": "s-mutable-field",
                "name": name,
                "ann": ann_to_pyret_json(ann, registry),
                "value": expr_to_pyret_json(value, registry)
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
                "ann": ann_to_pyret_json(ann, registry),
                "args": args.iter().map(|x| bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "blocky": blocky,
                "body": expr_to_pyret_json(body, registry),
                "check": check.as_ref().map(|c| expr_to_pyret_json(c, registry)),
                "check-loc": check_loc.as_ref().map(|l| loc_to_srcloc_string(l, registry)),
                "doc": doc,
                "name": name,
                "params": params.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
    }
}

pub fn spy_field_to_pyret_json(field: &crate::SpyField, registry: &FileRegistry) -> Value {
    json!({
        "type": "s-spy-expr",
        "name": field.name.as_ref(),
        "value": expr_to_pyret_json(&field.value, registry),
        "implicit-label": field.implicit_label
    })
}

pub fn variant_to_pyret_json(variant: &crate::Variant, registry: &FileRegistry) -> Value {
    use crate::Variant;
    match variant {
        Variant::SVariant {
            name,
            constr_loc,
            members,
            with_members,
            ..
        } => {
            json!({
                "type": "s-variant",
                "name": name,
                "constr-loc": loc_to_srcloc_string(constr_loc, registry),
                "members": members.iter().map(|x| variant_member_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "with-members": with_members.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Variant::SSingletonVariant {
            name, with_members, ..
        } => {
            json!({
                "type": "s-singleton-variant",
                "name": name,
                "with-members": with_members.iter().map(|x| member_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
    }
}

pub fn variant_member_to_pyret_json(
    member: &crate::VariantMember,
    registry: &FileRegistry,
) -> Value {
    json!({
        "type": "s-variant-member",
        "member-type": variant_member_type_to_pyret_json(&member.member_type, registry),
        "bind": bind_to_pyret_json(&member.bind, registry)
    })
}

pub fn variant_member_type_to_pyret_json(
    member_type: &crate::VariantMemberType,
    _registry: &FileRegistry,
) -> Value {
    use crate::VariantMemberType;
    match member_type {
        VariantMemberType::SNormal => json!({"type": "s-normal"}),
        VariantMemberType::SMutable => json!({"type": "s-mutable"}),
    }
}

pub fn ann_to_pyret_json(ann: &crate::Ann, registry: &FileRegistry) -> Value {
    use crate::Ann;
    match ann {
        Ann::ABlank => json!({"type": "a-blank"}),
        Ann::AAny { .. } => json!({"type": "a-any"}),
        Ann::AName { id, .. } => json!({
            "type": "a-name",
            "id": name_to_pyret_json(id, registry)
        }),
        Ann::ADot { obj, field, .. } => json!({
            "type": "a-dot",
            "obj": name_to_pyret_json(obj, registry),
            "field": field
        }),
        Ann::AArrow {
            args,
            ret,
            use_parens,
            ..
        } => json!({
            "type": "a-arrow",
            "args": args.iter().map(|x| ann_to_pyret_json(x, registry)).collect::<Vec<_>>(),
            "ret": ann_to_pyret_json(ret, registry),
            "use-parens": use_parens
        }),
        Ann::AApp { ann, args, .. } => json!({
            "type": "a-app",
            "ann": ann_to_pyret_json(ann, registry),
            "args": args.iter().map(|x| ann_to_pyret_json(x, registry)).collect::<Vec<_>>()
        }),
        Ann::APred { ann, exp, .. } => json!({
            "type": "a-pred",
            "ann": ann_to_pyret_json(ann, registry),
            "exp": expr_to_pyret_json(exp, registry)
        }),
        Ann::ATuple { fields, .. } => json!({
            "type": "a-tuple",
            "fields": fields.iter().map(|x| ann_to_pyret_json(x, registry)).collect::<Vec<_>>()
        }),
        Ann::ARecord { fields, .. } => json!({
            "type": "a-record",
            "fields": fields.iter().map(|f| json!({
                "type": "a-field",
                "name": f.name,
                "ann": ann_to_pyret_json(&f.ann, registry)
            })).collect::<Vec<_>>()
        }),
        _ => json!({
            "type": "UNSUPPORTED",
            "debug": format!("{:?}", ann)
        }),
    }
}

pub fn check_op_to_pyret_json(op: &crate::CheckOp, _registry: &FileRegistry) -> Value {
    use crate::CheckOp;
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

pub fn modifier_to_pyret_json(
    modifier: &crate::ConstructModifier,
    _registry: &FileRegistry,
) -> Value {
    use crate::ConstructModifier;
    match modifier {
        ConstructModifier::SConstructNormal => json!({"type": "s-construct-normal"}),
        ConstructModifier::SConstructLazy => json!({"type": "s-construct-lazy"}),
    }
}

pub fn name_to_pyret_json(name: &crate::Name, _registry: &FileRegistry) -> Value {
    use crate::Name;
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

pub fn bind_to_pyret_json(bind: &crate::Bind, registry: &FileRegistry) -> Value {
    use crate::Bind;
    match bind {
        Bind::SBind {
            id, ann, shadows, ..
        } => {
            json!({
                "type": "s-bind",
                "name": name_to_pyret_json(id, registry),
                "ann": ann_to_pyret_json(ann, registry),
                "shadows": shadows
            })
        }
        Bind::STupleBind {
            fields, as_name, ..
        } => {
            json!({
                "type": "s-tuple-bind",
                "fields": fields.iter().map(|x| bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "as-name": as_name.as_ref().map(|n| bind_to_pyret_json(n, registry))
            })
        }
    }
}

pub fn if_branch_to_pyret_json(branch: &crate::IfBranch, registry: &FileRegistry) -> Value {
    json!({
        "type": "s-if-branch",
        "test": expr_to_pyret_json(&branch.test, registry),
        "body": expr_to_pyret_json(&branch.body, registry)
    })
}

pub fn if_pipe_branch_to_pyret_json(
    branch: &crate::IfPipeBranch,
    registry: &FileRegistry,
) -> Value {
    json!({
        "type": "s-if-pipe-branch",
        "test": expr_to_pyret_json(&branch.test, registry),
        "body": expr_to_pyret_json(&branch.body, registry)
    })
}

pub fn for_bind_to_pyret_json(for_bind: &crate::ForBind, registry: &FileRegistry) -> Value {
    json!({
        "type": "s-for-bind",
        "bind": bind_to_pyret_json(&for_bind.bind, registry),
        "value": expr_to_pyret_json(&for_bind.value, registry)
    })
}

pub fn cases_branch_to_pyret_json(branch: &crate::CasesBranch, registry: &FileRegistry) -> Value {
    use crate::CasesBranch;
    match branch {
        CasesBranch::SCasesBranch {
            name,
            pattern_loc,
            args,
            body,
            ..
        } => {
            json!({
                "type": "s-cases-branch",
                "name": name,
                "pat-loc": loc_to_srcloc_string(pattern_loc, registry),
                "args": args.iter().map(|x| cases_bind_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "body": expr_to_pyret_json(body, registry)
            })
        }
        CasesBranch::SSingletonCasesBranch {
            name,
            pattern_loc,
            body,
            ..
        } => {
            json!({
                "type": "s-singleton-cases-branch",
                "name": name,
                "pat-loc": loc_to_srcloc_string(pattern_loc, registry),
                "body": expr_to_pyret_json(body, registry)
            })
        }
    }
}

pub fn cases_bind_to_pyret_json(cases_bind: &crate::CasesBind, registry: &FileRegistry) -> Value {
    json!({
        "type": "s-cases-bind",
        "field-type": format!("s-cases-bind-{}", match cases_bind.field_type {
            crate::CasesBindType::SNormal => "normal",
            crate::CasesBindType::SMutable => "mutable",
            crate::CasesBindType::SRef => "ref",
        }),
        "bind": bind_to_pyret_json(&cases_bind.bind, registry)
    })
}

pub fn let_bind_to_pyret_json(let_bind: &crate::LetBind, registry: &FileRegistry) -> Value {
    use crate::LetBind;
    match let_bind {
        LetBind::SLetBind { b, value, .. } => {
            json!({
                "type": "s-let-bind",
                "bind": bind_to_pyret_json(b, registry),
                "value": expr_to_pyret_json(value, registry)
            })
        }
        LetBind::SVarBind { b, value, .. } => {
            json!({
                "type": "s-var-bind",
                "bind": bind_to_pyret_json(b, registry),
                "value": expr_to_pyret_json(value, registry)
            })
        }
    }
}

pub fn letrec_bind_to_pyret_json(
    letrec_bind: &crate::LetrecBind,
    registry: &FileRegistry,
) -> Value {
    json!({
        "type": "s-letrec-bind",
        "bind": bind_to_pyret_json(&letrec_bind.b, registry),
        "value": expr_to_pyret_json(&letrec_bind.value, registry)
    })
}

pub fn program_to_pyret_json(program: &Program, registry: &FileRegistry) -> Value {
    let mut obj = json!({
        "type": "s-program",
        "provide": provide_to_pyret_json(&program._provide, registry),
        "provided-types": provide_types_to_pyret_json(&program.provided_types, registry),
        "provides": program.provides.iter().map(|x| provide_block_to_pyret_json(x, registry)).collect::<Vec<_>>(),
        "imports": program.imports.iter().map(|x| import_to_pyret_json(x, registry)).collect::<Vec<_>>(),
        "body": expr_to_pyret_json(&program.body, registry)
    });

    // Add "use" field (set to null when not present, as Pyret does)
    obj["use"] = match &program._use {
        Some(use_stmt) => use_to_pyret_json(use_stmt, registry),
        None => Value::Null,
    };

    obj
}

pub fn use_to_pyret_json(use_stmt: &crate::Use, registry: &FileRegistry) -> Value {
    json!({
        "type": "s-use",
        "name": name_to_pyret_json(&use_stmt.name, registry),
        "mod": import_type_to_pyret_json(&use_stmt.module, registry)
    })
}

pub fn provide_to_pyret_json(provide: &crate::Provide, registry: &FileRegistry) -> Value {
    use crate::Provide;
    match provide {
        Provide::SProvide { block, .. } => {
            json!({
                "type": "s-provide",
                "block": expr_to_pyret_json(block, registry)
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

pub fn a_field_to_pyret_json(field: &crate::AField, registry: &FileRegistry) -> Value {
    json!({
        "type": "a-field",
        "name": &field.name,
        "ann": ann_to_pyret_json(&field.ann, registry)
    })
}

pub fn provide_types_to_pyret_json(
    provide_types: &crate::ProvideTypes,
    registry: &FileRegistry,
) -> Value {
    use crate::ProvideTypes;
    match provide_types {
        ProvideTypes::SProvideTypes { anns, .. } => {
            json!({
                "type": "s-provide-types",
                "anns": anns.iter().map(|x| a_field_to_pyret_json(x, registry)).collect::<Vec<_>>()
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

pub fn provide_block_to_pyret_json(
    provide_block: &crate::ProvideBlock,
    registry: &FileRegistry,
) -> Value {
    json!({
        "type": "s-provide-block",
        "path": provide_block.path.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
        "specs": provide_block.specs.iter().map(|x| provide_spec_to_pyret_json(x, registry)).collect::<Vec<_>>()
    })
}

pub fn provide_spec_to_pyret_json(spec: &crate::ProvideSpec, registry: &FileRegistry) -> Value {
    use crate::ProvideSpec;
    match spec {
        ProvideSpec::SProvideName { name, .. } => {
            json!({
                "type": "s-provide-name",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
        ProvideSpec::SProvideType { name, .. } => {
            json!({
                "type": "s-provide-type",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
        ProvideSpec::SProvideData { name, hidden, .. } => {
            json!({
                "type": "s-provide-data",
                "name-spec": name_spec_to_pyret_json(name, registry),
                "hidden": hidden.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        ProvideSpec::SProvideModule { name, .. } => {
            json!({
                "type": "s-provide-module",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
    }
}

pub fn import_to_pyret_json(import: &crate::Import, registry: &FileRegistry) -> Value {
    use crate::Import;
    match import {
        Import::SInclude { import, .. } => {
            json!({
                "type": "s-include",
                "import-type": import_type_to_pyret_json(import, registry)
            })
        }
        Import::SIncludeFrom {
            module_path, names, ..
        } => {
            json!({
                "type": "s-include-from",
                "mod": module_path.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "specs": names.iter().map(|x| include_spec_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        Import::SImport { import, name, .. } => {
            json!({
                "type": "s-import",
                "import-type": import_type_to_pyret_json(import, registry),
                "name": name_to_pyret_json(name, registry)
            })
        }
        Import::SImportFields { fields, import, .. } => {
            json!({
                "type": "s-import-fields",
                "fields": fields.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "import-type": import_type_to_pyret_json(import, registry)
            })
        }
        Import::SImportTypes {
            import,
            types,
            name,
            ..
        } => {
            json!({
                "type": "s-import-types",
                "import-type": import_type_to_pyret_json(import, registry),
                "types": types.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "name": name_to_pyret_json(name, registry)
            })
        }
    }
}

pub fn import_type_to_pyret_json(
    import_type: &crate::ImportType,
    _registry: &FileRegistry,
) -> Value {
    use crate::ImportType;
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

pub fn include_spec_to_pyret_json(spec: &crate::IncludeSpec, registry: &FileRegistry) -> Value {
    use crate::IncludeSpec;
    match spec {
        IncludeSpec::SIncludeName { name, .. } => {
            json!({
                "type": "s-include-name",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
        IncludeSpec::SIncludeType { name, .. } => {
            json!({
                "type": "s-include-type",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
        IncludeSpec::SIncludeData { name, hidden, .. } => {
            json!({
                "type": "s-include-data",
                "name-spec": name_spec_to_pyret_json(name, registry),
                "hidden": hidden.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        IncludeSpec::SIncludeModule { name, .. } => {
            json!({
                "type": "s-include-module",
                "name-spec": name_spec_to_pyret_json(name, registry)
            })
        }
    }
}

pub fn name_spec_to_pyret_json(name_spec: &crate::NameSpec, registry: &FileRegistry) -> Value {
    use crate::NameSpec;
    match name_spec {
        NameSpec::SStar { hidden, .. } => {
            json!({
                "type": "s-star",
                "hidden": hidden.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>()
            })
        }
        NameSpec::SModuleRef { path, as_name, .. } => {
            json!({
                "type": "s-module-ref",
                "path": path.iter().map(|x| name_to_pyret_json(x, registry)).collect::<Vec<_>>(),
                "as-name": as_name.as_ref().map(|x| name_to_pyret_json(x, registry)).unwrap_or(Value::Null)
            })
        }
        NameSpec::SRemoteRef { uri, name, .. } => {
            json!({
                "type": "s-remote-ref",
                "uri": uri,
                "name": name_to_pyret_json(name, registry)
            })
        }
        NameSpec::SLocalRef { name, .. } => {
            json!({
                "type": "s-local-ref",
                "name": name_to_pyret_json(name, registry)
            })
        }
    }
}

pub fn load_table_spec_to_pyret_json(
    spec: &crate::LoadTableSpec,
    registry: &FileRegistry,
) -> Value {
    use crate::LoadTableSpec;
    match spec {
        LoadTableSpec::SSanitize {
            name, sanitizer, ..
        } => {
            json!({
                "type": "s-sanitize",
                "name": name_to_pyret_json(name, registry),
                "sanitizer": expr_to_pyret_json(sanitizer, registry)
            })
        }
        LoadTableSpec::STableSrc { src, .. } => {
            json!({
                "type": "s-table-src",
                "src": expr_to_pyret_json(src, registry)
            })
        }
    }
}

pub fn column_binds_to_pyret_json(
    column_binds: &crate::ColumnBinds,
    registry: &FileRegistry,
) -> Value {
    json!({
        "type": "s-column-binds",
        "binds": column_binds.binds.iter().map(|b| bind_to_pyret_json(b, registry)).collect::<Vec<_>>(),
        "table": expr_to_pyret_json(&column_binds.table, registry)
    })
}

pub fn table_extend_field_to_pyret_json(
    field: &crate::TableExtendField,
    registry: &FileRegistry,
) -> Value {
    use crate::TableExtendField;
    match field {
        TableExtendField::STableExtendField {
            name, value, ann, ..
        } => {
            json!({
                "type": "s-table-extend-field",
                "name": name,
                "value": expr_to_pyret_json(value, registry),
                "ann": ann_to_pyret_json(ann, registry)
            })
        }
        TableExtendField::STableExtendReducer {
            name,
            reducer,
            col,
            ann,
            ..
        } => {
            json!({
                "type": "s-table-extend-reducer",
                "name": name,
                "reducer": expr_to_pyret_json(reducer, registry),
                "col": name_to_pyret_json(col, registry),
                "ann": ann_to_pyret_json(ann, registry)
            })
        }
    }
}
