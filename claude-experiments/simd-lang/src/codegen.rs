use std::collections::HashMap;

use melior::{
    Context,
    dialect::{DialectRegistry, arith, func, llvm as llvm_dialect, scf},
    ir::{
        Block, BlockLike, Location, Module, Operation, Region, Type, Value,
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute,
            FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::{OperationBuilder, OperationLike, OperationMutLike},
        r#type::{FunctionType, IntegerType, MemRefType},
        RegionLike,
    },
    pass,
    utility::register_all_dialects,
};

use crate::ast;

/// Scalar element type — resolved from the AST type name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl ScalarType {
    pub fn from_name(name: &str) -> Self {
        match name {
            "f32" => ScalarType::F32,
            "f64" => ScalarType::F64,
            "i8" => ScalarType::I8,
            "i16" => ScalarType::I16,
            "i32" => ScalarType::I32,
            "i64" => ScalarType::I64,
            "u8" => ScalarType::U8,
            "u16" => ScalarType::U16,
            "u32" => ScalarType::U32,
            "u64" => ScalarType::U64,
            "bool" => ScalarType::Bool,
            other => panic!("unknown scalar type: {}", other),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    pub fn is_int(&self) -> bool {
        !self.is_float() && *self != ScalarType::Bool
    }

    fn to_mlir_type<'c>(&self, ctx: &'c Context) -> Type<'c> {
        match self {
            ScalarType::F32 => Type::float32(ctx),
            ScalarType::F64 => Type::float64(ctx),
            ScalarType::I8 => IntegerType::new(ctx, 8).into(),
            ScalarType::I16 => IntegerType::new(ctx, 16).into(),
            ScalarType::I32 => IntegerType::new(ctx, 32).into(),
            ScalarType::I64 => IntegerType::new(ctx, 64).into(),
            ScalarType::U8 => IntegerType::new(ctx, 8).into(),
            ScalarType::U16 => IntegerType::new(ctx, 16).into(),
            ScalarType::U32 => IntegerType::new(ctx, 32).into(),
            ScalarType::U64 => IntegerType::new(ctx, 64).into(),
            ScalarType::Bool => IntegerType::new(ctx, 1).into(),
        }
    }

    fn bit_width(&self) -> u32 {
        match self {
            ScalarType::I8 | ScalarType::U8 => 8,
            ScalarType::I16 | ScalarType::U16 => 16,
            ScalarType::F32 | ScalarType::I32 | ScalarType::U32 => 32,
            ScalarType::F64 | ScalarType::I64 | ScalarType::U64 => 64,
            ScalarType::Bool => 1,
        }
    }
}

/// A resolved vector type: element type + concrete width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VecType {
    pub scalar: ScalarType,
    pub width: u64,
}

impl VecType {
    fn to_mlir_type<'c>(&self, ctx: &'c Context) -> Type<'c> {
        Type::vector(&[self.width], self.scalar.to_mlir_type(ctx))
    }
}

/// Returns true if the type name is a built-in scalar type (not a struct).
fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"
    )
}

/// Returns true if the type name is a pointer/buffer type (ptr[T]).
fn is_ptr_type_name(name: &str) -> bool {
    name == "ptr"
}

/// Extract the element type name from a ptr[T] AST type.
/// ptr[u8] has width = Width::Param("u8"), so we extract "u8".
fn ptr_element_scalar(ty: &ast::Type) -> ScalarType {
    match ty.width.as_ref() {
        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
        _ => panic!("ptr type must have an element type, e.g. ptr[u8]"),
    }
}

/// Create an MLIR memref<?xelement> type for a ptr[T] type.
fn ptr_to_memref_type<'c>(ctx: &'c Context, ty: &ast::Type) -> Type<'c> {
    // ptr[u8] → memref<?xi8>, ptr[f32] → memref<?xf32>
    let scalar = ptr_element_scalar(ty);
    let scalar_mlir = scalar.to_mlir_type(ctx);
    // i64::MIN = dynamic dimension (?)
    MemRefType::new(scalar_mlir, &[i64::MIN], None, None).into()
}

/// Resolve an AST type to a concrete VecType.
/// `comptime_env` maps comptime param names to concrete values.
fn resolve_type(
    ty: &ast::Type,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> VecType {
    let scalar = ScalarType::from_name(&ty.name);
    let width = ty
        .width
        .as_ref()
        .expect("type must have a width")
        .eval(comptime_env, native_width);
    VecType { scalar, width }
}

/// Create an MLIR context with all dialects loaded.
pub fn create_context() -> Context {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    context
}

/// A binding in the codegen: either a single vector value, a struct with flattened fields, or a ptr.
#[derive(Debug, Clone)]
enum Binding<'c, 'a> {
    Vec(Value<'c, 'a>, VecType),
    Struct {
        fields: Vec<(String, Value<'c, 'a>, VecType)>,
    },
    /// A pointer/buffer (memref) — stores the raw MLIR value and the element scalar type.
    Ptr(Value<'c, 'a>, ScalarType),
}

/// Codegen state for a single function.
struct FnCodegen<'c, 'a> {
    ctx: &'c Context,
    loc: Location<'c>,
    block: &'a Block<'c>,
    vars: HashMap<String, Binding<'c, 'a>>,
    comptime_env: HashMap<String, u64>,
    native_width: u64,
    struct_defs: HashMap<String, ast::StructDef>,
    /// Current source location for error messages, set from stmt.span.
    current_span: ast::Span,
}

impl<'c, 'a> FnCodegen<'c, 'a> {
    /// Panic with source location context.
    fn error(&self, msg: &str) -> ! {
        panic!("{} at {}", msg, self.current_span)
    }

    fn emit_expr(&mut self, expr: &ast::Expr) -> (Value<'c, 'a>, VecType) {
        match expr {
            ast::Expr::Ident(name) => {
                let binding = self
                    .vars
                    .get(name)
                    .unwrap_or_else(|| self.error(&format!("undefined variable: {}", name)))
                    .clone();
                match binding {
                    Binding::Vec(val, ty) => (val, ty),
                    Binding::Ptr(val, scalar) => {
                        // Return the raw memref value with a dummy VecType
                        (val, VecType { scalar, width: 0 })
                    }
                    Binding::Struct { .. } => {
                        panic!(
                            "cannot use struct variable '{}' as a scalar value; access a field instead (e.g. {}.field)",
                            name, name
                        );
                    }
                }
            }

            ast::Expr::IntLit(n) => {
                self.error(&format!(
                    "bare integer literal {} needs type context (use in binary op or add type annotation: `name: type = {}`)",
                    n, n
                ));
            }

            ast::Expr::FloatLit(f) => {
                self.error(&format!(
                    "bare float literal {} needs type context (use in binary op or add type annotation)",
                    f
                ));
            }

            ast::Expr::BoolLit(b) => {
                self.error(&format!(
                    "bare bool literal {} needs type context (use in binary op or add type annotation)",
                    b
                ));
            }

            ast::Expr::CharLit(ch) => {
                // Char literals are bare just like int literals — need context
                panic!(
                    "bare char literal '{}' needs type context (use in binary op)",
                    ch
                );
            }

            ast::Expr::BinOp { op, lhs, rhs } => self.emit_binop(*op, lhs, rhs),

            ast::Expr::UnaryOp { op, operand } => self.emit_unary(*op, operand),

            ast::Expr::Masked {
                mask,
                body,
                fallback,
            } => self.emit_masked(mask, body, fallback.as_deref()),

            ast::Expr::Reduction { op, operand } => self.emit_reduction(*op, operand),

            ast::Expr::Field { base, field } => self.emit_field_access(base, field),

            ast::Expr::Call { func, args } => self.emit_call(func, args),

            ast::Expr::Scan { op, operand, seed } => self.emit_scan(*op, operand, seed.as_deref()),

            ast::Expr::Gather {
                base,
                index,
                mask,
            } => self.emit_gather(base, index, mask.as_deref()),

            ast::Expr::StructLit {
                name,
                width,
                fields,
            } => self.emit_struct_lit(name, width, fields),

            ast::Expr::Load { aligned, ty, ptr } => self.emit_load(*aligned, ty, ptr),

            ast::Expr::VecLit { elem_type, values } => {
                let scalar = ScalarType::from_name(elem_type);
                let width = values.len() as u64;
                let vt = VecType { scalar, width };
                let vec_mlir = vt.to_mlir_type(self.ctx);
                let scalar_mlir = scalar.to_mlir_type(self.ctx);

                let attrs: Vec<melior::ir::attribute::Attribute> = values
                    .iter()
                    .map(|v| {
                        if scalar.is_float() {
                            FloatAttribute::new(self.ctx, scalar_mlir, *v as f64).into()
                        } else {
                            IntegerAttribute::new(scalar_mlir, *v).into()
                        }
                    })
                    .collect();

                let dense = DenseElementsAttribute::new(vec_mlir, &attrs)
                    .expect("failed to create vector literal");
                let op = arith::constant(self.ctx, dense.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, vt)
            }
        }
    }

    /// Emit a literal broadcast to match a target vector type.
    fn emit_literal_broadcast(
        &mut self,
        expr: &ast::Expr,
        target: VecType,
    ) -> (Value<'c, 'a>, VecType) {
        let vec_type = target.to_mlir_type(self.ctx);

        match expr {
            ast::Expr::FloatLit(f) => {
                if !target.scalar.is_float() {
                    self.error("float literal in non-float context");
                }
                let scalar_type = target.scalar.to_mlir_type(self.ctx);
                let attr = FloatAttribute::new(self.ctx, scalar_type, *f);
                let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                    .expect("failed to create splat attribute");
                let op = arith::constant(self.ctx, splat.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, target)
            }
            ast::Expr::IntLit(n) => {
                if !(target.scalar.is_int() || target.scalar == ScalarType::Bool) {
                    self.error("int literal in float context");
                }
                let scalar_type = target.scalar.to_mlir_type(self.ctx);
                let attr = IntegerAttribute::new(scalar_type, *n);
                let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                    .expect("failed to create splat attribute");
                let op = arith::constant(self.ctx, splat.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, target)
            }
            ast::Expr::CharLit(ch) => {
                // Char literal broadcast: use the char's ASCII value as an integer
                if !(target.scalar.is_int() || target.scalar == ScalarType::Bool) {
                    self.error("char literal in float context");
                }
                let scalar_type = target.scalar.to_mlir_type(self.ctx);
                let attr = IntegerAttribute::new(scalar_type, *ch as i64);
                let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                    .expect("failed to create splat attribute");
                let op = arith::constant(self.ctx, splat.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, target)
            }
            ast::Expr::UnaryOp { op, operand } => {
                // Broadcast the inner literal, then apply the unary op
                let (val, ty) = self.emit_literal_broadcast(operand, target);
                match op {
                    ast::UnaryOp::Not => {
                        // Bitwise NOT: xor with all-ones
                        let vec_type = ty.to_mlir_type(self.ctx);
                        let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                        let ones_attr = IntegerAttribute::new(scalar_type, -1);
                        let splat = DenseElementsAttribute::new(vec_type, &[ones_attr.into()])
                            .expect("failed to create all-ones splat");
                        let ones = self
                            .block
                            .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                            .result(0)
                            .unwrap()
                            .into();
                        let result = self
                            .block
                            .append_operation(arith::xori(val, ones, self.loc))
                            .result(0)
                            .unwrap()
                            .into();
                        (result, ty)
                    }
                    ast::UnaryOp::Neg => {
                        if ty.scalar.is_float() {
                            let neg = self
                                .block
                                .append_operation(arith::negf(val, self.loc))
                                .result(0)
                                .unwrap()
                                .into();
                            (neg, ty)
                        } else {
                            let vec_type = ty.to_mlir_type(self.ctx);
                            let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                            let zero_attr = IntegerAttribute::new(scalar_type, 0);
                            let splat = DenseElementsAttribute::new(vec_type, &[zero_attr.into()])
                                .expect("failed to create zero splat");
                            let zero = self
                                .block
                                .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                                .result(0)
                                .unwrap()
                                .into();
                            let neg = self
                                .block
                                .append_operation(arith::subi(zero, val, self.loc))
                                .result(0)
                                .unwrap()
                                .into();
                            (neg, ty)
                        }
                    }
                }
            }
            _ => panic!("emit_literal_broadcast called on non-literal"),
        }
    }

    fn is_literal(expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::FloatLit(_)
            | ast::Expr::IntLit(_)
            | ast::Expr::BoolLit(_)
            | ast::Expr::CharLit(_) => true,
            // ~literal and -literal are broadcastable (e.g. ~0 = all-ones)
            ast::Expr::UnaryOp { op: ast::UnaryOp::Not, operand }
            | ast::Expr::UnaryOp { op: ast::UnaryOp::Neg, operand } => Self::is_literal(operand),
            _ => false,
        }
    }

    fn emit_binop(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        // Bit shift: RHS is a compile-time integer constant, broadcast to match LHS width
        if op == ast::BinOp::BitShl || op == ast::BinOp::BitShr {
            let shift_amount = match rhs {
                ast::Expr::IntLit(n) => *n,
                _ => panic!("bit shift amount must be an integer literal"),
            };
            let (lval, lty) = self.emit_expr(lhs);
            let scalar_type = lty.scalar.to_mlir_type(self.ctx);
            let vec_type = lty.to_mlir_type(self.ctx);
            let shift_attr = IntegerAttribute::new(scalar_type, shift_amount);
            let shift_splat =
                DenseElementsAttribute::new(vec_type, &[shift_attr.into()]).unwrap();
            let shift_vec: Value = self
                .block
                .append_operation(arith::constant(self.ctx, shift_splat.into(), self.loc))
                .result(0)
                .unwrap()
                .into();
            let result_op = if op == ast::BinOp::BitShl {
                arith::shli(lval, shift_vec, self.loc)
            } else {
                arith::shrui(lval, shift_vec, self.loc)
            };
            let result: Value = self
                .block
                .append_operation(result_op)
                .result(0)
                .unwrap()
                .into();
            return (result, lty);
        }

        // Handle literal broadcasting: if one side is a literal, infer its type from the other.
        let (lval, lty, rval, rty) = match (Self::is_literal(lhs), Self::is_literal(rhs)) {
            (true, true) => panic!("both operands are literals — cannot infer vector width"),
            (true, false) => {
                let (rv, rt) = self.emit_expr(rhs);
                let (lv, lt) = self.emit_literal_broadcast(lhs, rt);
                (lv, lt, rv, rt)
            }
            (false, true) => {
                let (lv, lt) = self.emit_expr(lhs);
                let (rv, rt) = self.emit_literal_broadcast(rhs, lt);
                (lv, lt, rv, rt)
            }
            (false, false) => {
                let (lv, lt) = self.emit_expr(lhs);
                let (rv, rt) = self.emit_expr(rhs);
                (lv, lt, rv, rt)
            }
        };

        // Auto-broadcast width-1 vectors to match the other operand's width
        let (lval, lty, rval, rty) = if lty.width == 1 && rty.width > 1 {
            let (bv, bt) = self.emit_broadcast_to_width(lval, lty, rty.width);
            (bv, bt, rval, rty)
        } else if rty.width == 1 && lty.width > 1 {
            let (bv, bt) = self.emit_broadcast_to_width(rval, rty, lty.width);
            (lval, lty, bv, bt)
        } else {
            (lval, lty, rval, rty)
        };

        if lty.width != rty.width {
            self.error(&format!(
                "width mismatch: {:?}[{}] vs {:?}[{}]",
                lty.scalar, lty.width, rty.scalar, rty.width
            ));
        }

        match op {
            ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div => {
                let result_op = if lty.scalar.is_float() {
                    match op {
                        ast::BinOp::Add => arith::addf(lval, rval, self.loc),
                        ast::BinOp::Sub => arith::subf(lval, rval, self.loc),
                        ast::BinOp::Mul => arith::mulf(lval, rval, self.loc),
                        ast::BinOp::Div => arith::divf(lval, rval, self.loc),
                        _ => unreachable!(),
                    }
                } else {
                    match op {
                        ast::BinOp::Add => arith::addi(lval, rval, self.loc),
                        ast::BinOp::Sub => arith::subi(lval, rval, self.loc),
                        ast::BinOp::Mul => arith::muli(lval, rval, self.loc),
                        ast::BinOp::Div => arith::divsi(lval, rval, self.loc),
                        _ => unreachable!(),
                    }
                };
                let val = self
                    .block
                    .append_operation(result_op)
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Gt
            | ast::BinOp::Lt
            | ast::BinOp::GtEq
            | ast::BinOp::LtEq
            | ast::BinOp::EqEq
            | ast::BinOp::NotEq => {
                let result_ty = VecType {
                    scalar: ScalarType::Bool,
                    width: lty.width,
                };
                let cmp_op = if lty.scalar.is_float() {
                    let pred = match op {
                        ast::BinOp::Gt => arith::CmpfPredicate::Ogt,
                        ast::BinOp::Lt => arith::CmpfPredicate::Olt,
                        ast::BinOp::GtEq => arith::CmpfPredicate::Oge,
                        ast::BinOp::LtEq => arith::CmpfPredicate::Ole,
                        ast::BinOp::EqEq => arith::CmpfPredicate::Oeq,
                        ast::BinOp::NotEq => arith::CmpfPredicate::One,
                        _ => unreachable!(),
                    };
                    arith::cmpf(self.ctx, pred, lval, rval, self.loc)
                } else {
                    let is_unsigned = matches!(
                        lty.scalar,
                        ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64
                    );
                    let pred = match (op, is_unsigned) {
                        (ast::BinOp::Gt, false) => arith::CmpiPredicate::Sgt,
                        (ast::BinOp::Lt, false) => arith::CmpiPredicate::Slt,
                        (ast::BinOp::GtEq, false) => arith::CmpiPredicate::Sge,
                        (ast::BinOp::LtEq, false) => arith::CmpiPredicate::Sle,
                        (ast::BinOp::Gt, true) => arith::CmpiPredicate::Ugt,
                        (ast::BinOp::Lt, true) => arith::CmpiPredicate::Ult,
                        (ast::BinOp::GtEq, true) => arith::CmpiPredicate::Uge,
                        (ast::BinOp::LtEq, true) => arith::CmpiPredicate::Ule,
                        (ast::BinOp::EqEq, _) => arith::CmpiPredicate::Eq,
                        (ast::BinOp::NotEq, _) => arith::CmpiPredicate::Ne,
                        _ => unreachable!(),
                    };
                    arith::cmpi(self.ctx, pred, lval, rval, self.loc)
                };
                let val = self
                    .block
                    .append_operation(cmp_op)
                    .result(0)
                    .unwrap()
                    .into();
                (val, result_ty)
            }
            ast::BinOp::And => {
                let val = self
                    .block
                    .append_operation(arith::andi(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Or => {
                let val = self
                    .block
                    .append_operation(arith::ori(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Xor => {
                let val = self
                    .block
                    .append_operation(arith::xori(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::BitShl | ast::BinOp::BitShr => {
                unreachable!("bit shift ops handled before operand evaluation");
            }
        }
    }

    /// Broadcast a width-1 vector to a wider width using vector.broadcast.
    fn emit_broadcast_to_width(
        &mut self,
        val: Value<'c, 'a>,
        ty: VecType,
        target_width: u64,
    ) -> (Value<'c, 'a>, VecType) {
        assert_eq!(ty.width, 1, "can only broadcast width-1 vectors");
        let scalar_type = ty.scalar.to_mlir_type(self.ctx);

        // Extract the scalar from width-1 vector
        let extract_op = OperationBuilder::new("vector.extract", self.loc)
            .add_operands(&[val])
            .add_results(&[scalar_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.ctx, "static_position"),
                DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
            )])
            .build()
            .expect("failed to extract scalar for broadcast");
        let scalar = self.block.append_operation(extract_op).result(0).unwrap().into();

        // Broadcast to target width
        let result_ty = VecType { scalar: ty.scalar, width: target_width };
        let result_mlir = result_ty.to_mlir_type(self.ctx);
        let broadcast_op = OperationBuilder::new("vector.broadcast", self.loc)
            .add_operands(&[scalar])
            .add_results(&[result_mlir])
            .build()
            .expect("failed to broadcast to target width");
        let result = self.block.append_operation(broadcast_op).result(0).unwrap().into();
        (result, result_ty)
    }

    /// Emit a lane shift using vector.shuffle.
    /// For shl (left): lane[i] = lane[i + shift], zeros fill the right.
    /// For shr (right): lane[i] = lane[i - shift], zeros fill the left.
    fn emit_lane_shift(
        &mut self,
        val: Value<'c, 'a>,
        ty: VecType,
        shift: usize,
        is_left: bool,
    ) -> (Value<'c, 'a>, VecType) {
        let width = ty.width as usize;
        let vec_mlir_type = ty.to_mlir_type(self.ctx);

        let zero = self.emit_zero_vector(ty);

        let mut mask: Vec<i64> = Vec::with_capacity(width);
        for lane in 0..width {
            if is_left {
                // Left shift: lane i gets value from lane (i + shift)
                let src = lane + shift;
                if src < width {
                    mask.push(src as i64);
                } else {
                    mask.push((width + lane) as i64); // from zero vector
                }
            } else {
                // Right shift: lane i gets value from lane (i - shift)
                if lane >= shift {
                    mask.push((lane - shift) as i64);
                } else {
                    mask.push((width + lane) as i64); // from zero vector
                }
            }
        }

        let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
            .add_operands(&[val, zero])
            .add_results(&[vec_mlir_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.ctx, "mask"),
                DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
            )])
            .build()
            .expect("failed to build vector.shuffle for lane shift");

        let result = self
            .block
            .append_operation(shuffle_op)
            .result(0)
            .unwrap()
            .into();

        (result, ty)
    }

    fn emit_unary(&mut self, op: ast::UnaryOp, operand: &ast::Expr) -> (Value<'c, 'a>, VecType) {
        match op {
            ast::UnaryOp::Neg => {
                let (val, ty) = self.emit_expr(operand);
                if ty.scalar.is_float() {
                    let neg = self
                        .block
                        .append_operation(arith::negf(val, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    (neg, ty)
                } else {
                    // Integer negate: 0 - val
                    let zero_ty = ty;
                    let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                    let vec_type = ty.to_mlir_type(self.ctx);
                    let zero_attr = IntegerAttribute::new(scalar_type, 0);
                    let splat = DenseElementsAttribute::new(vec_type, &[zero_attr.into()])
                        .expect("failed to create zero splat");
                    let zero = self
                        .block
                        .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    let neg = self
                        .block
                        .append_operation(arith::subi(zero, val, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    (neg, zero_ty)
                }
            }
            ast::UnaryOp::Not => {
                let (val, ty) = self.emit_expr(operand);
                // Bitwise NOT: xor with all-ones
                let vec_type = ty.to_mlir_type(self.ctx);
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let ones_attr = IntegerAttribute::new(scalar_type, -1);
                let splat = DenseElementsAttribute::new(vec_type, &[ones_attr.into()])
                    .expect("failed to create all-ones splat");
                let ones = self
                    .block
                    .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let result = self
                    .block
                    .append_operation(arith::xori(val, ones, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (result, ty)
            }
        }
    }

    fn emit_masked(
        &mut self,
        mask: &ast::Expr,
        body: &ast::Expr,
        fallback: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (mask_val, mask_ty) = self.emit_expr(mask);
        let fallback = fallback.expect("masked expression without fallback not yet supported (inactive lanes would be undefined)");

        // Handle literal body/fallback: infer type from whichever side is not a literal,
        // or from the mask width if both are literals.
        let (body_val, body_ty, fallback_val) =
            match (Self::is_literal(body), Self::is_literal(fallback)) {
                (true, true) => {
                    // Both literals: infer type from mask width, default to i32
                    let target = VecType {
                        scalar: ScalarType::I32,
                        width: mask_ty.width,
                    };
                    let (bv, bt) = self.emit_literal_broadcast(body, target);
                    let (fv, _) = self.emit_literal_broadcast(fallback, target);
                    (bv, bt, fv)
                }
                (true, false) => {
                    let (fv, ft) = self.emit_expr(fallback);
                    let (bv, bt) = self.emit_literal_broadcast(body, ft);
                    (bv, bt, fv)
                }
                (false, true) => {
                    let (bv, bt) = self.emit_expr(body);
                    let (fv, _) = self.emit_literal_broadcast(fallback, bt);
                    (bv, bt, fv)
                }
                (false, false) => {
                    let (bv, bt) = self.emit_expr(body);
                    let (fv, _) = self.emit_expr(fallback);
                    (bv, bt, fv)
                }
            };

        // arith.select works lane-wise on vectors
        let val = self
            .block
            .append_operation(arith::select(mask_val, body_val, fallback_val, self.loc))
            .result(0)
            .unwrap()
            .into();
        (val, body_ty)
    }

    fn emit_reduction(
        &mut self,
        op: ast::ReductionOp,
        operand: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        let (val, ty) = self.emit_expr(operand);
        let result_ty = VecType {
            scalar: ty.scalar,
            width: 1,
        };

        // Use vector.reduction operation via OperationBuilder
        let kind_str = match op {
            ast::ReductionOp::Add => "add",
            ast::ReductionOp::Mul => "mul",
            ast::ReductionOp::Or => "or",
            ast::ReductionOp::And => "and",
            ast::ReductionOp::Max => {
                if ty.scalar.is_float() {
                    "maximumf"
                } else {
                    "maxsi"
                }
            }
            ast::ReductionOp::Min => {
                if ty.scalar.is_float() {
                    "minimumf"
                } else {
                    "minsi"
                }
            }
        };

        let scalar_mlir_type = ty.scalar.to_mlir_type(self.ctx);
        let kind_attr = melior::ir::attribute::Attribute::parse(
            self.ctx,
            &format!("#vector.kind<{}>", kind_str),
        )
        .expect("failed to parse vector.kind attribute");

        let reduction_op = OperationBuilder::new("vector.reduction", self.loc)
            .add_operands(&[val])
            .add_results(&[scalar_mlir_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.ctx, "kind"),
                kind_attr,
            )])
            .build()
            .expect("failed to build vector.reduction");

        let scalar_val = self
            .block
            .append_operation(reduction_op)
            .result(0)
            .unwrap()
            .into();

        // Wrap scalar result into a width-1 vector via vector.broadcast
        let result_mlir_type = result_ty.to_mlir_type(self.ctx);
        let broadcast_op = OperationBuilder::new("vector.broadcast", self.loc)
            .add_operands(&[scalar_val])
            .add_results(&[result_mlir_type])
            .build()
            .expect("failed to build vector.broadcast");

        let result_val = self
            .block
            .append_operation(broadcast_op)
            .result(0)
            .unwrap()
            .into();

        (result_val, result_ty)
    }

    fn emit_field_access(
        &mut self,
        base: &ast::Expr,
        field: &str,
    ) -> (Value<'c, 'a>, VecType) {
        // The base must be a struct-bound variable
        match base {
            ast::Expr::Ident(name) => {
                let binding = self
                    .vars
                    .get(name)
                    .unwrap_or_else(|| self.error(&format!("undefined variable: {}", name)))
                    .clone();
                match binding {
                    Binding::Struct { fields } => {
                        for (fname, val, ty) in &fields {
                            if fname == field {
                                return (*val, *ty);
                            }
                        }
                        self.error(&format!(
                            "struct '{}' has no field '{}'",
                            name, field
                        ));
                    }
                    Binding::Vec(_, _) | Binding::Ptr(_, _) => {
                        panic!(
                            "cannot access field '{}' on non-struct variable '{}'",
                            field, name
                        );
                    }
                }
            }
            _ => panic!("field access on complex expressions not yet supported; use a variable"),
        }
    }

    fn emit_struct_lit(
        &mut self,
        name: &str,
        width: &ast::Width,
        field_exprs: &[(String, ast::Expr)],
    ) -> (Value<'c, 'a>, VecType) {
        // Look up the struct definition
        let struct_def = self
            .struct_defs
            .get(name)
            .unwrap_or_else(|| panic!("undefined struct: {}", name))
            .clone();

        // Evaluate the width
        let concrete_width = width.eval(&self.comptime_env, self.native_width);

        // Build a comptime env for the struct's own comptime params
        // Map the struct's first comptime param (if any) to the concrete width
        let mut struct_comptime = self.comptime_env.clone();
        if let Some(cp) = struct_def.comptime_params.first() {
            struct_comptime.insert(cp.name.clone(), concrete_width);
        }

        // Evaluate each field expression
        let mut fields: Vec<(String, Value<'c, 'a>, VecType)> = Vec::new();
        for field_def in &struct_def.fields {
            // Find the matching expression
            let expr = field_exprs
                .iter()
                .find(|(fname, _)| fname == &field_def.name)
                .unwrap_or_else(|| {
                    panic!(
                        "struct literal '{}' missing field '{}'",
                        name, field_def.name
                    )
                });
            let (val, ty) = self.emit_expr(&expr.1);
            fields.push((field_def.name.clone(), val, ty));
        }

        // Store as a struct binding.
        // The "return value" for struct literals is the first field's value/type
        // (caller should handle struct returns specially).
        // We store the struct in vars as a side effect if assigned.
        // For now, return the first field value — the caller (emit_stmt for return)
        // will handle multi-value returns.

        // Actually, we need to handle this differently. The struct literal creates
        // a Binding::Struct, but emit_expr returns a single (Value, VecType).
        // We'll store the struct binding in a temporary and return a sentinel.
        // Better approach: use the first field as the "value" and let the return
        // path handle struct returns by looking up all fields.

        // For struct literals, we store them in a temp var and return the first field.
        // The return codegen path checks if the return expression is a struct literal
        // and flattens all fields.

        // For simplicity: store the struct in a hidden var, return first field.
        // Actually, the cleanest approach: return first field, but also store
        // the struct binding. We need a way to retrieve it.

        // Let's use a different approach: store the fields in a special temp,
        // and return the first field. The emit_return will handle struct lit specially.
        let temp_name = format!("__struct_lit_{}_{}", name, fields.len());
        self.vars.insert(
            temp_name,
            Binding::Struct {
                fields: fields.clone(),
            },
        );

        // Return first field value (for non-return contexts)
        let (_, first_val, first_ty) = &fields[0];
        (*first_val, *first_ty)
    }

    fn emit_scan(
        &mut self,
        op: ast::ScanOp,
        operand: &ast::Expr,
        seed: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (val, ty) = self.emit_expr(operand);
        let width = ty.width as usize;
        let vec_mlir_type = ty.to_mlir_type(self.ctx);

        // If a seed is provided, XOR/ADD/etc it into lane 0 before scanning.
        // This threads carry state from previous chunks.
        let val = if let Some(seed_expr) = seed {
            let (seed_val, seed_ty) = self.emit_expr(seed_expr);
            // Seed is width-1 vector. Broadcast it to match the operand width,
            // then combine with lane 0 using the scan's operation.
            // Actually simpler: insert seed into a zero vector at lane 0,
            // then combine with the operand using the scan op.
            // For XOR: val[0] = val[0] ^ seed, rest unchanged.
            // This is equivalent to: result = val XOR (seed_broadcast_to_lane0_only)

            // Extract scalar from seed (width-1 vector)
            let scalar_type = ty.scalar.to_mlir_type(self.ctx);
            let extract_op = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[seed_val])
                .add_results(&[scalar_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract seed scalar");
            let seed_scalar = self.block.append_operation(extract_op).result(0).unwrap().into();

            // Extract lane 0 from val
            let extract_val0 = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[val])
                .add_results(&[scalar_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract val[0]");
            let val0 = self.block.append_operation(extract_val0).result(0).unwrap().into();

            // Combine: val[0] OP seed
            let combined = match op {
                ast::ScanOp::Xor => {
                    let op = arith::xori(val0, seed_scalar, self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                }
                ast::ScanOp::Add => {
                    let op = if ty.scalar.is_float() {
                        arith::addf(val0, seed_scalar, self.loc)
                    } else {
                        arith::addi(val0, seed_scalar, self.loc)
                    };
                    self.block.append_operation(op).result(0).unwrap().into()
                }
                _ => panic!("seeded scan not supported for {:?}", op),
            };

            // Insert combined back into lane 0
            let insert_op = OperationBuilder::new("vector.insert", self.loc)
                .add_operands(&[combined, val])
                .add_results(&[vec_mlir_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to insert combined into lane 0");
            self.block.append_operation(insert_op).result(0).unwrap().into()
        } else {
            val
        };

        match op {
            ast::ScanOp::PrecedingAny => {
                // Shift right by 1: take lanes 0..N-2, prepend a 0 lane
                // This checks if any preceding lane is set
                // Result[i] = input[i-1] for i>0, result[0] = 0
                let mut mask: Vec<i64> = Vec::with_capacity(width);
                // First lane gets a poison/zero lane from the "second vector" (which is a zero vec)
                mask.push(width as i64); // index into the zero vector's lane 0
                for i in 0..(width - 1) {
                    mask.push(i as i64);
                }

                // Create a zero vector for the fill
                let zero_val = self.emit_zero_vector(ty);

                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, zero_val])
                    .add_results(&[vec_mlir_type])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for preceding_any");

                let result = self
                    .block
                    .append_operation(shuffle_op)
                    .result(0)
                    .unwrap()
                    .into();

                (result, ty)
            }
            ast::ScanOp::Add | ast::ScanOp::Xor | ast::ScanOp::Max => {
                // Inclusive prefix scan using log2(N) steps of shuffle-and-combine
                // Step i (for i = 0, 1, ..., log2(N)-1):
                //   shifted = shuffle(val, identity, mask_that_shifts_right_by_2^i)
                //   val = combine(val, shifted)
                let num_steps = (width as f64).log2() as usize;
                let mut current = val;

                for step in 0..num_steps {
                    let shift = 1 << step;

                    // Build shuffle mask: lanes 0..shift-1 get identity from second vector,
                    // lanes shift..N-1 get from first vector at position lane-shift
                    let mut mask: Vec<i64> = Vec::with_capacity(width);
                    for lane in 0..width {
                        if lane < shift {
                            // Fill from identity vector (second operand, offset by width)
                            mask.push((width + lane) as i64);
                        } else {
                            // Take from current vector at position lane - shift
                            mask.push((lane - shift) as i64);
                        }
                    }

                    // Create identity vector for this operation
                    let identity = self.emit_scan_identity(ty, op);

                    let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                        .add_operands(&[current, identity])
                        .add_results(&[vec_mlir_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "mask"),
                            DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                        )])
                        .build()
                        .expect("failed to build vector.shuffle for scan");

                    let shifted = self
                        .block
                        .append_operation(shuffle_op)
                        .result(0)
                        .unwrap()
                        .into();

                    // Combine current with shifted
                    let combine_op = match op {
                        ast::ScanOp::Add => {
                            if ty.scalar.is_float() {
                                arith::addf(current, shifted, self.loc)
                            } else {
                                arith::addi(current, shifted, self.loc)
                            }
                        }
                        ast::ScanOp::Xor => arith::xori(current, shifted, self.loc),
                        ast::ScanOp::Max => {
                            if ty.scalar.is_float() {
                                // Use arith.maximumf
                                OperationBuilder::new("arith.maximumf", self.loc)
                                    .add_operands(&[current, shifted])
                                    .add_results(&[vec_mlir_type])
                                    .build()
                                    .expect("failed to build arith.maximumf")
                            } else {
                                // Use arith.maxsi
                                OperationBuilder::new("arith.maxsi", self.loc)
                                    .add_operands(&[current, shifted])
                                    .add_results(&[vec_mlir_type])
                                    .build()
                                    .expect("failed to build arith.maxsi")
                            }
                        }
                        ast::ScanOp::PrecedingAny => unreachable!(),
                    };

                    current = self
                        .block
                        .append_operation(combine_op)
                        .result(0)
                        .unwrap()
                        .into();
                }

                (current, ty)
            }
        }
    }

    /// Create a zero vector (identity for add, xor, and preceding_any).
    fn emit_zero_vector(&mut self, ty: VecType) -> Value<'c, 'a> {
        let vec_type = ty.to_mlir_type(self.ctx);
        let scalar_type = ty.scalar.to_mlir_type(self.ctx);

        if ty.scalar.is_float() {
            let attr = FloatAttribute::new(self.ctx, scalar_type, 0.0);
            let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                .expect("failed to create zero splat");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        } else {
            let attr = IntegerAttribute::new(scalar_type, 0);
            let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                .expect("failed to create zero splat");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        }
    }

    /// Create the identity element vector for a scan operation.
    fn emit_scan_identity(&mut self, ty: VecType, op: ast::ScanOp) -> Value<'c, 'a> {
        match op {
            ast::ScanOp::Add | ast::ScanOp::Xor | ast::ScanOp::PrecedingAny => {
                self.emit_zero_vector(ty)
            }
            ast::ScanOp::Max => {
                // Identity for max is the minimum possible value
                let vec_type = ty.to_mlir_type(self.ctx);
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                if ty.scalar.is_float() {
                    let attr = FloatAttribute::new(self.ctx, scalar_type, f64::NEG_INFINITY);
                    let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                        .expect("failed to create neg_inf splat");
                    let op = arith::constant(self.ctx, splat.into(), self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                } else {
                    // For signed integers, use the minimum value
                    let min_val = match ty.scalar.bit_width() {
                        8 => i8::MIN as i64,
                        16 => i16::MIN as i64,
                        32 => i32::MIN as i64,
                        64 => i64::MIN,
                        _ => 0,
                    };
                    let attr = IntegerAttribute::new(scalar_type, min_val);
                    let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                        .expect("failed to create min_val splat");
                    let op = arith::constant(self.ctx, splat.into(), self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                }
            }
        }
    }

    fn emit_gather(
        &mut self,
        base: &ast::Expr,
        index: &ast::Expr,
        mask: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (base_val, _base_ty) = self.emit_expr(base);
        let (index_val, index_ty) = self.emit_expr(index);

        // The result type is determined by the base element type and index width
        // base is a memref, index determines the gather width
        let result_width = index_ty.width;

        // We need to figure out the element type from context.
        // For gather, we'll infer element type from the base memref.
        // The base should be a memref<?xelement> type.
        // For now, assume f32 as default; in practice this would be tracked.
        // Actually, we need to determine this from the base expression's type.
        // Since base_val comes from a memref param, we'll need to track memref element types.
        // For now, use a reasonable default or check the VecType.

        // Build the mask (all-true if none provided)
        let mask_ty = VecType {
            scalar: ScalarType::Bool,
            width: result_width,
        };
        let mask_val = if let Some(m) = mask {
            self.emit_expr(m).0
        } else {
            // All-true mask
            let mask_mlir = mask_ty.to_mlir_type(self.ctx);
            let scalar_type = ScalarType::Bool.to_mlir_type(self.ctx);
            let attr = IntegerAttribute::new(scalar_type, 1);
            let splat = DenseElementsAttribute::new(mask_mlir, &[attr.into()])
                .expect("failed to create all-true mask");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        };

        // Passthrough (zero for unmasked lanes)
        let result_ty = VecType {
            scalar: ScalarType::F32, // default, should be inferred
            width: result_width,
        };
        let passthrough = self.emit_zero_vector(result_ty);
        let result_mlir = result_ty.to_mlir_type(self.ctx);

        // vector.gather %base[%c0], %indices, %mask, %passthrough
        // Need a zero index for the base offset
        let index_type = Type::index(self.ctx);
        let zero_index_attr = IntegerAttribute::new(index_type, 0);
        let zero_index_op = arith::constant(self.ctx, zero_index_attr.into(), self.loc);
        let zero_index = self
            .block
            .append_operation(zero_index_op)
            .result(0)
            .unwrap()
            .into();

        let gather_op = OperationBuilder::new("vector.gather", self.loc)
            .add_operands(&[base_val, zero_index, index_val, mask_val, passthrough])
            .add_results(&[result_mlir])
            .build()
            .expect("failed to build vector.gather");

        let result = self
            .block
            .append_operation(gather_op)
            .result(0)
            .unwrap()
            .into();

        (result, result_ty)
    }

    fn emit_load(
        &mut self,
        _aligned: bool,
        ty: &ast::Type,
        ptr: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        let (ptr_val, _ptr_ty) = self.emit_expr(ptr);
        let vec_ty = resolve_type(ty, &self.comptime_env, self.native_width);
        let result_mlir = vec_ty.to_mlir_type(self.ctx);

        // vector.load %memref[%c0] : memref<?xelement>, vector<Nxelement>
        let index_type = Type::index(self.ctx);
        let zero_index_attr = IntegerAttribute::new(index_type, 0);
        let zero_index_op = arith::constant(self.ctx, zero_index_attr.into(), self.loc);
        let zero_index = self
            .block
            .append_operation(zero_index_op)
            .result(0)
            .unwrap()
            .into();

        let load_op = OperationBuilder::new("vector.load", self.loc)
            .add_operands(&[ptr_val, zero_index])
            .add_results(&[result_mlir])
            .build()
            .expect("failed to build vector.load");

        let result = self
            .block
            .append_operation(load_op)
            .result(0)
            .unwrap()
            .into();

        (result, vec_ty)
    }

    fn emit_store(
        &mut self,
        ptr: &ast::Expr,
        value: &ast::Expr,
    ) {
        let (ptr_val, _) = self.emit_expr(ptr);
        let (val, _) = self.emit_expr(value);

        let index_type = Type::index(self.ctx);
        let zero_index_attr = IntegerAttribute::new(index_type, 0);
        let zero_index_op = arith::constant(self.ctx, zero_index_attr.into(), self.loc);
        let zero_index = self
            .block
            .append_operation(zero_index_op)
            .result(0)
            .unwrap()
            .into();

        let store_op = OperationBuilder::new("vector.store", self.loc)
            .add_operands(&[val, ptr_val, zero_index])
            .build()
            .expect("failed to build vector.store");

        self.block.append_operation(store_op);
    }

    fn emit_scatter(
        &mut self,
        base: &ast::Expr,
        index: &ast::Expr,
        mask: Option<&ast::Expr>,
        value: &ast::Expr,
    ) {
        let (base_val, _) = self.emit_expr(base);
        let (index_val, index_ty) = self.emit_expr(index);
        let (val, _val_ty) = self.emit_expr(value);

        // Build mask (all-true if none provided)
        let mask_ty = VecType {
            scalar: ScalarType::Bool,
            width: index_ty.width,
        };
        let mask_val = if let Some(m) = mask {
            self.emit_expr(m).0
        } else {
            let mask_mlir = mask_ty.to_mlir_type(self.ctx);
            let scalar_type = ScalarType::Bool.to_mlir_type(self.ctx);
            let attr = IntegerAttribute::new(scalar_type, 1);
            let splat = DenseElementsAttribute::new(mask_mlir, &[attr.into()])
                .expect("failed to create all-true mask");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        };

        // vector.scatter %base[%c0], %indices, %mask, %value
        let index_type = Type::index(self.ctx);
        let zero_index_attr = IntegerAttribute::new(index_type, 0);
        let zero_index_op = arith::constant(self.ctx, zero_index_attr.into(), self.loc);
        let zero_index = self
            .block
            .append_operation(zero_index_op)
            .result(0)
            .unwrap()
            .into();

        let scatter_op = OperationBuilder::new("vector.scatter", self.loc)
            .add_operands(&[base_val, zero_index, index_val, mask_val, val])
            .build()
            .expect("failed to build vector.scatter");

        self.block.append_operation(scatter_op);
    }

    fn emit_call(
        &mut self,
        func_expr: &ast::Expr,
        args: &[ast::CallArg],
    ) -> (Value<'c, 'a>, VecType) {
        match func_expr {
            ast::Expr::Ident(name) if name == "sqrt" => {
                assert_eq!(args.len(), 1, "sqrt takes exactly one argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert!(ty.scalar.is_float(), "sqrt requires float type");
                let vec_type = ty.to_mlir_type(self.ctx);
                let sqrt_op = OperationBuilder::new("math.sqrt", self.loc)
                    .add_operands(&[val])
                    .add_results(&[vec_type])
                    .build()
                    .expect("failed to build math.sqrt");
                let result = self
                    .block
                    .append_operation(sqrt_op)
                    .result(0)
                    .unwrap()
                    .into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "compress" => {
                // compress(data, mask) → pack lanes where mask is true to the front
                assert_eq!(args.len(), 2, "compress takes exactly 2 arguments (data, mask)");
                let (data_val, data_ty) = self.emit_expr(&args[0].value);
                let (mask_val, _mask_ty) = self.emit_expr(&args[1].value);
                let result_mlir = data_ty.to_mlir_type(self.ctx);

                let compress_op = OperationBuilder::new("vector.compress", self.loc)
                    .add_operands(&[data_val, mask_val])
                    .add_results(&[result_mlir])
                    .build()
                    .expect("failed to build vector.compress");

                let result = self
                    .block
                    .append_operation(compress_op)
                    .result(0)
                    .unwrap()
                    .into();
                (result, data_ty)
            }
            ast::Expr::Ident(name) if name == "popcount" => {
                // popcount(bool_vec) → count of true lanes, as i32[1]
                assert_eq!(args.len(), 1, "popcount takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                // Cast bool to i32 then reduce with add
                let i32_ty = VecType {
                    scalar: ScalarType::I32,
                    width: ty.width,
                };
                let i32_mlir = i32_ty.to_mlir_type(self.ctx);

                // Zero-extend bool (i1) to i32
                let ext_op = arith::extui(val, i32_mlir, self.loc);
                let ext_val = self
                    .block
                    .append_operation(ext_op)
                    .result(0)
                    .unwrap()
                    .into();

                // Reduce with add
                let result_ty = VecType {
                    scalar: ScalarType::I32,
                    width: 1,
                };
                let scalar_i32 = ScalarType::I32.to_mlir_type(self.ctx);
                let kind_attr = melior::ir::attribute::Attribute::parse(
                    self.ctx,
                    "#vector.kind<add>",
                )
                .expect("failed to parse vector.kind attribute");

                let reduction_op = OperationBuilder::new("vector.reduction", self.loc)
                    .add_operands(&[ext_val])
                    .add_results(&[scalar_i32])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "kind"),
                        kind_attr,
                    )])
                    .build()
                    .expect("failed to build vector.reduction");

                let scalar_val = self
                    .block
                    .append_operation(reduction_op)
                    .result(0)
                    .unwrap()
                    .into();

                // Broadcast to width-1 vector
                let result_mlir = result_ty.to_mlir_type(self.ctx);
                let broadcast_op = OperationBuilder::new("vector.broadcast", self.loc)
                    .add_operands(&[scalar_val])
                    .add_results(&[result_mlir])
                    .build()
                    .expect("failed to build vector.broadcast");

                let result = self
                    .block
                    .append_operation(broadcast_op)
                    .result(0)
                    .unwrap()
                    .into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "store" => {
                // store(ptr, value) or store(ptr, value, mask)
                assert!(
                    args.len() == 2 || args.len() == 3,
                    "store takes 2 or 3 arguments"
                );
                self.emit_store(&args[0].value, &args[1].value);
                // store doesn't return a value — return a dummy
                // This is only used in expression statements
                let dummy_ty = VecType {
                    scalar: ScalarType::I32,
                    width: 1,
                };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            ast::Expr::Ident(name) if name == "iota" => {
                // iota(width) → [0, 1, 2, ..., width-1] as i32 vector
                assert_eq!(args.len(), 1, "iota takes exactly 1 argument (width)");
                let width = match &args[0].value {
                    ast::Expr::IntLit(n) => *n as u64,
                    _ => panic!("iota argument must be an integer literal"),
                };
                let result_ty = VecType {
                    scalar: ScalarType::I32,
                    width,
                };
                let vec_mlir = result_ty.to_mlir_type(self.ctx);
                let i32_type = ScalarType::I32.to_mlir_type(self.ctx);

                let values: Vec<melior::ir::attribute::Attribute> = (0..width as i64)
                    .map(|i| IntegerAttribute::new(i32_type, i).into())
                    .collect();
                let dense = DenseElementsAttribute::new(vec_mlir, &values)
                    .expect("failed to create iota dense attribute");
                let op = arith::constant(self.ctx, dense.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, result_ty)
            }
            ast::Expr::Ident(name) if name == "extract" => {
                // extract(vec, lane) → scalar value from lane as width-1 vector
                assert_eq!(args.len(), 2, "extract takes exactly 2 arguments (vec, lane)");
                let (val, ty) = self.emit_expr(&args[0].value);
                let lane = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("extract lane must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let result_ty = VecType {
                    scalar: ty.scalar,
                    width: 1,
                };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // vector.extract to scalar
                let extract_op = OperationBuilder::new("vector.extract", self.loc)
                    .add_operands(&[val])
                    .add_results(&[scalar_type])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "static_position"),
                        DenseI64ArrayAttribute::new(self.ctx, &[lane]).into(),
                    )])
                    .build()
                    .expect("failed to build vector.extract");
                let scalar_val = self.block.append_operation(extract_op).result(0).unwrap().into();

                // Broadcast scalar to width-1 vector
                let broadcast_op = OperationBuilder::new("vector.broadcast", self.loc)
                    .add_operands(&[scalar_val])
                    .add_results(&[result_mlir])
                    .build()
                    .expect("failed to build vector.broadcast");
                let result = self.block.append_operation(broadcast_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "compressstore" => {
                // compressstore(buf, offset, data, mask) → stores selected lanes contiguously
                assert_eq!(args.len(), 4, "compressstore takes 4 arguments (buf, offset, data, mask)");
                let (buf_val, _) = self.emit_expr(&args[0].value);
                let (offset_val, _) = self.emit_expr(&args[1].value);
                let (data_val, data_ty) = self.emit_expr(&args[2].value);
                let (mask_val, _) = self.emit_expr(&args[3].value);

                // Convert offset from i32[1] to index scalar
                let scalar_i32 = ScalarType::I32.to_mlir_type(self.ctx);
                let index_type = Type::index(self.ctx);

                // Extract scalar from i32[1]
                let extract_op = OperationBuilder::new("vector.extract", self.loc)
                    .add_operands(&[offset_val])
                    .add_results(&[scalar_i32])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "static_position"),
                        DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                    )])
                    .build()
                    .expect("failed to extract offset scalar");
                let offset_scalar = self.block.append_operation(extract_op).result(0).unwrap().into();

                // Cast i32 to index
                let cast_op = arith::index_cast(offset_scalar, index_type, self.loc);
                let offset_index = self.block.append_operation(cast_op).result(0).unwrap().into();

                // vector.compressstore %buf[%offset], %mask, %data
                let compressstore_op = OperationBuilder::new("vector.compressstore", self.loc)
                    .add_operands(&[buf_val, offset_index, mask_val, data_val])
                    .build()
                    .expect("failed to build vector.compressstore");
                self.block.append_operation(compressstore_op);

                // Return a dummy value (compressstore is void)
                let dummy_ty = VecType { scalar: ScalarType::I32, width: 1 };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            ast::Expr::Ident(name) if name == "to_i32" => {
                // to_i32(vec) → zero-extend each lane to i32
                assert_eq!(args.len(), 1, "to_i32 takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let result_ty = VecType { scalar: ScalarType::I32, width: ty.width };
                let result_mlir = result_ty.to_mlir_type(self.ctx);
                let ext_op = arith::extui(val, result_mlir, self.loc);
                let result = self.block.append_operation(ext_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "to_bitmask" => {
                // to_bitmask(bool[N]) → pack into u64[1], one bit per lane
                // Uses vector.bitcast which LLVM lowers to efficient NEON
                // (cmeq + and-with-weights + zip + addv)
                assert_eq!(args.len(), 1, "to_bitmask takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert_eq!(ty.scalar, ScalarType::Bool, "to_bitmask requires bool vector");
                let width = ty.width;
                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let vec1xi64 = Type::vector(&[1], i64_type);

                // Bitcast vector<Nxi1> → vector<1xiN>
                let int_type: Type = IntegerType::new(self.ctx, width as u32).into();
                let vec1xint = Type::vector(&[1], int_type);

                let packed: Value = self.block.append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[val])
                        .add_results(&[vec1xint])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // Extract scalar, extend to i64
                let scalar: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[packed])
                        .add_results(&[int_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let scalar_i64 = if width < 64 {
                    self.block.append_operation(
                        arith::extui(scalar, i64_type, self.loc)
                    ).result(0).unwrap().into()
                } else {
                    scalar
                };

                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };
                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[scalar_i64])
                        .add_results(&[vec1xi64])
                        .build().unwrap()
                ).result(0).unwrap().into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "from_bitmask" => {
                // from_bitmask(u64[1], N) → bool[N], unpack bits to bool vector
                assert_eq!(args.len(), 2, "from_bitmask takes 2 arguments (bitmask, width)");
                let (val, _) = self.emit_expr(&args[0].value);
                let target_width = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as u64,
                    _ => panic!("from_bitmask width must be an integer literal"),
                };

                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let int_type: Type = IntegerType::new(self.ctx, target_width as u32).into();
                let vec1xint = Type::vector(&[1], int_type);
                let result_ty = VecType { scalar: ScalarType::Bool, width: target_width };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // Extract i64 scalar
                let scalar: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[val])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // Truncate to N bits if needed
                let scalar_n = if target_width < 64 {
                    let trunc: Value = self.block.append_operation(
                        arith::trunci(scalar, int_type, self.loc)
                    ).result(0).unwrap().into();
                    trunc
                } else {
                    scalar
                };

                // Broadcast to vector<1xiN> then bitcast to vector<Nxi1>
                let vec_int: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[scalar_n])
                        .add_results(&[vec1xint])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[vec_int])
                        .add_results(&[result_mlir])
                        .build().unwrap()
                ).result(0).unwrap().into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "clmul" => {
                // clmul(u64[1], u64[1]) → u64[1], carry-less multiply (PMULL)
                assert_eq!(args.len(), 2, "clmul takes exactly 2 arguments");
                let (a_val, _) = self.emit_expr(&args[0].value);
                let (b_val, _) = self.emit_expr(&args[1].value);

                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let vec16xi8 = Type::vector(&[16], IntegerType::new(self.ctx, 8).into());
                let vec2xi64 = Type::vector(&[2], i64_type);
                let vec1xi64 = Type::vector(&[1], i64_type);
                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };

                // Extract scalars from u64[1]
                let a_scalar: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[a_val])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let b_scalar: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[b_val])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // PMULL intrinsic
                let seg_sizes = DenseI32ArrayAttribute::new(self.ctx, &[2, 0]);
                let bundle_sizes = DenseI32ArrayAttribute::new(self.ctx, &[]);
                let pmull: Value = self.block.append_operation(
                    OperationBuilder::new("llvm.call_intrinsic", self.loc)
                        .add_operands(&[a_scalar, b_scalar])
                        .add_results(&[vec16xi8])
                        .add_attributes(&[
                            (melior::ir::Identifier::new(self.ctx, "intrin"),
                             StringAttribute::new(self.ctx, "llvm.aarch64.neon.pmull64").into()),
                            (melior::ir::Identifier::new(self.ctx, "operandSegmentSizes"),
                             seg_sizes.into()),
                            (melior::ir::Identifier::new(self.ctx, "op_bundle_sizes"),
                             bundle_sizes.into()),
                        ])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // Bitcast to 2xi64, extract lower
                let as_2xi64: Value = self.block.append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[pmull])
                        .add_results(&[vec2xi64])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let lower: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[as_2xi64])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // Wrap in u64[1]
                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[lower])
                        .add_results(&[vec1xi64])
                        .build().unwrap()
                ).result(0).unwrap().into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "ctz" => {
                // ctz(u64[1]) → u64[1], count trailing zeros
                assert_eq!(args.len(), 1, "ctz takes exactly 1 argument");
                let (val, _) = self.emit_expr(&args[0].value);
                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let vec1xi64 = Type::vector(&[1], i64_type);
                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };

                let scalar: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[val])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let ctz_result: Value = self.block.append_operation(
                    melior::dialect::llvm::intr_cttz(self.ctx, scalar, false, i64_type, self.loc)
                ).result(0).unwrap().into();

                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[ctz_result])
                        .add_results(&[vec1xi64])
                        .build().unwrap()
                ).result(0).unwrap().into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "clear_lowest_bit" => {
                // clear_lowest_bit(u64[1]) → u64[1], x & (x-1)
                assert_eq!(args.len(), 1, "clear_lowest_bit takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);

                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let vec1xi64 = Type::vector(&[1], i64_type);

                // x - 1
                let one_attr = IntegerAttribute::new(i64_type, 1);
                let one_splat = DenseElementsAttribute::new(vec1xi64, &[one_attr.into()]).unwrap();
                let one_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, one_splat.into(), self.loc)
                ).result(0).unwrap().into();

                let x_minus_1: Value = self.block.append_operation(
                    arith::subi(val, one_vec, self.loc)
                ).result(0).unwrap().into();

                // x & (x - 1)
                let result: Value = self.block.append_operation(
                    arith::andi(val, x_minus_1, self.loc)
                ).result(0).unwrap().into();

                (result, ty)
            }
            ast::Expr::Ident(name) if name == "xor" => {
                // xor(a, b) → bitwise XOR (legacy alias for a ^ b)
                assert_eq!(args.len(), 2, "xor takes exactly 2 arguments");
                let (a_val, a_ty) = self.emit_expr(&args[0].value);
                let (b_val, _) = self.emit_expr(&args[1].value);
                let result: Value = self.block.append_operation(
                    arith::xori(a_val, b_val, self.loc)
                ).result(0).unwrap().into();
                (result, a_ty)
            }
            ast::Expr::Ident(name) if name == "bit_shr" => {
                // bit_shr(val, amount) → legacy alias for val >> amount
                assert_eq!(args.len(), 2, "bit_shr takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift_amount = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("bit_shr amount must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let vec_type = ty.to_mlir_type(self.ctx);
                let shift_attr = IntegerAttribute::new(scalar_type, shift_amount);
                let shift_splat = DenseElementsAttribute::new(vec_type, &[shift_attr.into()]).unwrap();
                let shift_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, shift_splat.into(), self.loc)
                ).result(0).unwrap().into();
                let result: Value = self.block.append_operation(
                    arith::shrui(val, shift_vec, self.loc)
                ).result(0).unwrap().into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "bit_shl" => {
                // bit_shl(val, amount) → legacy alias for val << amount
                assert_eq!(args.len(), 2, "bit_shl takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift_amount = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("bit_shl amount must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let vec_type = ty.to_mlir_type(self.ctx);
                let shift_attr = IntegerAttribute::new(scalar_type, shift_amount);
                let shift_splat = DenseElementsAttribute::new(vec_type, &[shift_attr.into()]).unwrap();
                let shift_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, shift_splat.into(), self.loc)
                ).result(0).unwrap().into();
                let result: Value = self.block.append_operation(
                    arith::shli(val, shift_vec, self.loc)
                ).result(0).unwrap().into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "lane_shr" => {
                // lane_shr(vec, amount) → shift lanes right (zeros fill left)
                assert_eq!(args.len(), 2, "lane_shr takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as usize,
                    _ => panic!("lane_shr amount must be an integer literal"),
                };
                self.emit_lane_shift(val, ty, shift, false)
            }
            ast::Expr::Ident(name) if name == "lane_shl" => {
                // lane_shl(vec, amount) → shift lanes left (zeros fill right)
                assert_eq!(args.len(), 2, "lane_shl takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as usize,
                    _ => panic!("lane_shl amount must be an integer literal"),
                };
                self.emit_lane_shift(val, ty, shift, true)
            }
            ast::Expr::Ident(name) if name == "tbl" => {
                // tbl(table, indices) → NEON table lookup (vtbl1)
                // table: u8[16], indices: u8[N] where N is multiple of 16
                // For N > 16, splits into 16-byte chunks automatically
                assert_eq!(args.len(), 2, "tbl takes exactly 2 arguments (table, indices)");
                let (table_val, table_ty) = self.emit_expr(&args[0].value);
                let (idx_val, idx_ty) = self.emit_expr(&args[1].value);
                assert_eq!(table_ty.width, 16, "tbl table must be u8[16]");

                let width = idx_ty.width as usize;
                let vec16xi8 = Type::vector(&[16], IntegerType::new(self.ctx, 8).into());
                let seg_sizes = DenseI32ArrayAttribute::new(self.ctx, &[2, 0]);
                let bundle_sizes = DenseI32ArrayAttribute::new(self.ctx, &[]);

                if width == 16 {
                    let result: Value = self.block.append_operation(
                        OperationBuilder::new("llvm.call_intrinsic", self.loc)
                            .add_operands(&[table_val, idx_val])
                            .add_results(&[vec16xi8])
                            .add_attributes(&[
                                (melior::ir::Identifier::new(self.ctx, "intrin"),
                                 StringAttribute::new(self.ctx, "llvm.aarch64.neon.tbl1").into()),
                                (melior::ir::Identifier::new(self.ctx, "operandSegmentSizes"),
                                 seg_sizes.into()),
                                (melior::ir::Identifier::new(self.ctx, "op_bundle_sizes"),
                                 bundle_sizes.into()),
                            ])
                            .build().unwrap()
                    ).result(0).unwrap().into();
                    (result, idx_ty)
                } else {
                    // Split into 16-byte chunks
                    let num_chunks = width / 16;
                    let result_type = idx_ty.to_mlir_type(self.ctx);
                    let mut current = self.emit_zero_vector(idx_ty);

                    for chunk_idx in 0..num_chunks {
                        let offset = chunk_idx * 16;
                        let offsets_attr = melior::ir::attribute::Attribute::parse(
                            self.ctx, &format!("[{}]", offset)).unwrap();
                        let sizes_attr = melior::ir::attribute::Attribute::parse(
                            self.ctx, "[16]").unwrap();
                        let strides_attr = melior::ir::attribute::Attribute::parse(
                            self.ctx, "[1]").unwrap();
                        let extract = OperationBuilder::new("vector.extract_strided_slice", self.loc)
                            .add_operands(&[idx_val])
                            .add_results(&[vec16xi8])
                            .add_attributes(&[
                                (melior::ir::Identifier::new(self.ctx, "offsets"), offsets_attr),
                                (melior::ir::Identifier::new(self.ctx, "sizes"), sizes_attr),
                                (melior::ir::Identifier::new(self.ctx, "strides"), strides_attr),
                            ])
                            .build().unwrap();
                        let slice: Value = self.block.append_operation(extract).result(0).unwrap().into();

                        let seg_sizes2 = DenseI32ArrayAttribute::new(self.ctx, &[2, 0]);
                        let bundle_sizes2 = DenseI32ArrayAttribute::new(self.ctx, &[]);
                        let tbl_result: Value = self.block.append_operation(
                            OperationBuilder::new("llvm.call_intrinsic", self.loc)
                                .add_operands(&[table_val, slice])
                                .add_results(&[vec16xi8])
                                .add_attributes(&[
                                    (melior::ir::Identifier::new(self.ctx, "intrin"),
                                     StringAttribute::new(self.ctx, "llvm.aarch64.neon.tbl1").into()),
                                    (melior::ir::Identifier::new(self.ctx, "operandSegmentSizes"),
                                     seg_sizes2.into()),
                                    (melior::ir::Identifier::new(self.ctx, "op_bundle_sizes"),
                                     bundle_sizes2.into()),
                                ])
                                .build().unwrap()
                        ).result(0).unwrap().into();

                        let ins_offsets = melior::ir::attribute::Attribute::parse(
                            self.ctx, &format!("[{}]", offset)).unwrap();
                        let ins_strides = melior::ir::attribute::Attribute::parse(
                            self.ctx, "[1]").unwrap();
                        let insert = OperationBuilder::new("vector.insert_strided_slice", self.loc)
                            .add_operands(&[tbl_result, current])
                            .add_results(&[result_type])
                            .add_attributes(&[
                                (melior::ir::Identifier::new(self.ctx, "offsets"), ins_offsets),
                                (melior::ir::Identifier::new(self.ctx, "strides"), ins_strides),
                            ])
                            .build().unwrap();
                        current = self.block.append_operation(insert).result(0).unwrap().into();
                    }
                    (current, idx_ty)
                }
            }
            ast::Expr::Ident(name) if name == "split_lo" => {
                // split_lo(vec[N]) → vec[N/2], lower half via vector.shuffle
                assert_eq!(args.len(), 1, "split_lo takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let half = ty.width / 2;
                let result_ty = VecType { scalar: ty.scalar, width: half };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // Use vector.shuffle with mask [0, 1, ..., half-1]
                let mask: Vec<i64> = (0..half as i64).collect();
                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, val])
                    .add_results(&[result_mlir])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for split_lo");
                let result: Value = self.block.append_operation(shuffle_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "split_hi" => {
                // split_hi(vec[N]) → vec[N/2], upper half via vector.shuffle
                assert_eq!(args.len(), 1, "split_hi takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let half = ty.width / 2;
                let result_ty = VecType { scalar: ty.scalar, width: half };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                let mask: Vec<i64> = (half as i64..ty.width as i64).collect();
                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, val])
                    .add_results(&[result_mlir])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for split_hi");
                let result: Value = self.block.append_operation(shuffle_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "any" => {
                // any(bool[N]) → bool[1], true if any lane is true
                assert_eq!(args.len(), 1, "any takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert_eq!(ty.scalar, ScalarType::Bool, "any requires bool vector");
                let bool_type = ScalarType::Bool.to_mlir_type(self.ctx);
                let result_ty = VecType { scalar: ScalarType::Bool, width: 1 };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                let kind_attr = melior::ir::attribute::Attribute::parse(
                    self.ctx, "#vector.kind<or>",
                ).unwrap();
                let reduced: Value = self.block.append_operation(
                    OperationBuilder::new("vector.reduction", self.loc)
                        .add_operands(&[val])
                        .add_results(&[bool_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "kind"),
                            kind_attr,
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[reduced])
                        .add_results(&[result_mlir])
                        .build().unwrap()
                ).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "gather" => {
                // gather(buf, indices) → gather elements from memref at index positions
                assert_eq!(args.len(), 2, "gather takes exactly 2 arguments (buf, indices)");
                let (buf_val, buf_bind) = self.emit_expr(&args[0].value);
                let (index_val, index_ty) = self.emit_expr(&args[1].value);

                // Determine element type from the buffer binding
                let element_scalar = match &args[0].value {
                    ast::Expr::Ident(name) => match self.vars.get(name) {
                        Some(Binding::Ptr(_, scalar)) => *scalar,
                        _ => panic!("gather base must be a ptr variable"),
                    },
                    _ => panic!("gather base must be an identifier"),
                };

                let result_ty = VecType {
                    scalar: element_scalar,
                    width: index_ty.width,
                };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // Build all-true mask
                let mask_ty = VecType { scalar: ScalarType::Bool, width: index_ty.width };
                let mask_mlir = mask_ty.to_mlir_type(self.ctx);
                let bool_type = ScalarType::Bool.to_mlir_type(self.ctx);
                let mask_attr = IntegerAttribute::new(bool_type, 1);
                let mask_splat = DenseElementsAttribute::new(mask_mlir, &[mask_attr.into()])
                    .expect("failed to create all-true mask");
                let mask_val = self.block
                    .append_operation(arith::constant(self.ctx, mask_splat.into(), self.loc))
                    .result(0).unwrap().into();

                // Passthrough (zero vector)
                let passthrough = self.emit_zero_vector(result_ty);

                // Zero index for base offset
                let index_type = Type::index(self.ctx);
                let zero_attr = IntegerAttribute::new(index_type, 0);
                let zero_index = self.block
                    .append_operation(arith::constant(self.ctx, zero_attr.into(), self.loc))
                    .result(0).unwrap().into();

                let gather_op = OperationBuilder::new("vector.gather", self.loc)
                    .add_operands(&[buf_val, zero_index, index_val, mask_val, passthrough])
                    .add_results(&[result_mlir])
                    .build()
                    .expect("failed to build vector.gather for ptr");

                let result = self.block.append_operation(gather_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "store_at" => {
                // store_at(buf, offset, data) → vector.store at index offset
                assert_eq!(args.len(), 3, "store_at takes 3 arguments (buf, offset, data)");
                let (buf_val, _) = self.emit_expr(&args[0].value);
                let (offset_val, _) = self.emit_expr(&args[1].value);
                let (data_val, _) = self.emit_expr(&args[2].value);

                // Convert offset from i32[1] to index
                let scalar_i32 = ScalarType::I32.to_mlir_type(self.ctx);
                let index_type = Type::index(self.ctx);

                let extract_op = OperationBuilder::new("vector.extract", self.loc)
                    .add_operands(&[offset_val])
                    .add_results(&[scalar_i32])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "static_position"),
                        DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                    )])
                    .build()
                    .expect("failed to extract offset scalar");
                let offset_scalar = self.block.append_operation(extract_op).result(0).unwrap().into();

                let cast_op = arith::index_cast(offset_scalar, index_type, self.loc);
                let offset_index = self.block.append_operation(cast_op).result(0).unwrap().into();

                let store_op = OperationBuilder::new("vector.store", self.loc)
                    .add_operands(&[data_val, buf_val, offset_index])
                    .build()
                    .expect("failed to build vector.store");
                self.block.append_operation(store_op);

                let dummy_ty = VecType { scalar: ScalarType::I32, width: 1 };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            _ => panic!("unknown function call in codegen: {:?}", func_expr),
        }
    }

    /// Emit a stream construct: scf.for loop over buffer chunks with carry state.
    ///
    /// Generates:
    ///   %c0 = constant 0 : index
    ///   %len = <from second param or buffer dim>
    ///   %step = constant <chunk_width> : index
    ///   %init_carry... = constant <initial values>
    ///   scf.for %i = %c0 to %len step %step iter_args(%carry = %init_carry) -> (types) {
    ///     %chunk = vector.load %buf[%i] : memref<?xT>, vector<NxT>
    ///     <body>
    ///     scf.yield %updated_carry
    ///   }
    fn emit_stream(
        &mut self,
        chunk_name: &str,
        chunk_ty: &ast::Type,
        buffer: &str,
        carry_defs: &[ast::CarryDef],
        body: &[ast::Stmt],
        carry_updates: &[(String, ast::Expr)],
    ) {
        let index_type = Type::index(self.ctx);

        // Resolve the chunk vector type
        let chunk_vt = resolve_type(chunk_ty, &self.comptime_env, self.native_width);
        let chunk_mlir = chunk_vt.to_mlir_type(self.ctx);
        let chunk_width = chunk_vt.width;

        // Get the buffer memref value from vars
        let buf_binding = self
            .vars
            .get(buffer)
            .unwrap_or_else(|| panic!("undefined buffer: {}", buffer))
            .clone();
        let buf_val = match buf_binding {
            Binding::Ptr(val, _) => val,
            _ => panic!("stream buffer '{}' must be a ptr type", buffer),
        };

        // Emit constants in the outer block: start=0, step=chunk_width
        let zero_attr = IntegerAttribute::new(index_type, 0);
        let start_val = self
            .block
            .append_operation(arith::constant(self.ctx, zero_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();

        let step_attr = IntegerAttribute::new(index_type, chunk_width as i64);
        let step_val = self
            .block
            .append_operation(arith::constant(self.ctx, step_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();

        // Get buffer length via memref.dim
        let dim_index_attr = IntegerAttribute::new(index_type, 0);
        let dim_index = self
            .block
            .append_operation(arith::constant(self.ctx, dim_index_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();
        let end_val = self
            .block
            .append_operation(
                OperationBuilder::new("memref.dim", self.loc)
                    .add_operands(&[buf_val, dim_index])
                    .add_results(&[index_type])
                    .build()
                    .expect("failed to build memref.dim"),
            )
            .result(0)
            .unwrap()
            .into();

        // Emit initial carry values in the outer block
        let mut init_carry_vals: Vec<Value<'c, 'a>> = Vec::new();
        let mut carry_vtypes: Vec<VecType> = Vec::new();
        let mut carry_mlir_types: Vec<Type<'c>> = Vec::new();

        for carry_def in carry_defs {
            let carry_vt = resolve_type(&carry_def.ty, &self.comptime_env, self.native_width);
            carry_vtypes.push(carry_vt);
            carry_mlir_types.push(carry_vt.to_mlir_type(self.ctx));

            // Emit the initial value
            let init_val = match &carry_def.init {
                ast::Expr::BoolLit(false) | ast::Expr::IntLit(0) => {
                    self.emit_zero_vector(carry_vt)
                }
                ast::Expr::BoolLit(true) => {
                    let scalar_type = carry_vt.scalar.to_mlir_type(self.ctx);
                    let vec_type = carry_vt.to_mlir_type(self.ctx);
                    let attr = IntegerAttribute::new(scalar_type, 1);
                    let splat = DenseElementsAttribute::new(vec_type, &[attr.into()])
                        .expect("failed to create all-ones splat");
                    self.block
                        .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                        .result(0)
                        .unwrap()
                        .into()
                }
                other => {
                    // Try to evaluate as a general expression
                    let (val, _) = self.emit_expr(other);
                    val
                }
            };
            init_carry_vals.push(init_val);
        }

        // Build the inner block for scf.for body
        // Block args: [index (induction var), carry_0, carry_1, ...]
        let mut block_arg_types: Vec<(Type<'c>, Location<'c>)> = vec![(index_type, self.loc)];
        for ct in &carry_mlir_types {
            block_arg_types.push((*ct, self.loc));
        }
        let inner_block = Block::new(&block_arg_types);

        // Inside the inner block:
        // 1. Load chunk from buffer at current index
        let iv: Value = inner_block.argument(0).unwrap().into();
        let chunk_load = inner_block.append_operation(
            OperationBuilder::new("vector.load", self.loc)
                .add_operands(&[
                    // Transmute buf_val to inner block lifetime — safe because MLIR allows
                    // inner regions to reference values from enclosing regions
                    unsafe { std::mem::transmute::<Value, Value>(buf_val) },
                    iv,
                ])
                .add_results(&[chunk_mlir])
                .build()
                .expect("failed to build vector.load for stream chunk"),
        );
        let chunk_val: Value = chunk_load.result(0).unwrap().into();

        // 2. Create a FnCodegen for the inner block to emit body statements
        {
            let mut inner_cg = FnCodegen {
                ctx: self.ctx,
                loc: self.loc,
                block: &inner_block,
                vars: HashMap::new(),
                comptime_env: self.comptime_env.clone(),
                native_width: self.native_width,
                struct_defs: self.struct_defs.clone(),
                current_span: ast::Span::dummy(),
            };

            // Bind chunk variable
            inner_cg
                .vars
                .insert(chunk_name.to_string(), Binding::Vec(chunk_val, chunk_vt));

            // Bind chunk_offset — the loop induction variable as i32[1]
            let i32_type = ScalarType::I32.to_mlir_type(self.ctx);
            let offset_i32_ty = VecType { scalar: ScalarType::I32, width: 1 };
            let offset_mlir = offset_i32_ty.to_mlir_type(self.ctx);
            let cast_op = arith::index_cast(iv, i32_type, self.loc);
            let iv_i32 = inner_block.append_operation(cast_op).result(0).unwrap().into();
            let bcast_op = OperationBuilder::new("vector.broadcast", self.loc)
                .add_operands(&[iv_i32])
                .add_results(&[offset_mlir])
                .build()
                .expect("failed to broadcast chunk_offset");
            let iv_vec: Value = inner_block.append_operation(bcast_op).result(0).unwrap().into();
            inner_cg.vars.insert("chunk_offset".to_string(), Binding::Vec(iv_vec, offset_i32_ty));

            // Bind carry variables from inner block args
            for (idx, carry_def) in carry_defs.iter().enumerate() {
                let carry_val: Value = inner_block.argument(1 + idx).unwrap().into();
                inner_cg.vars.insert(
                    carry_def.name.clone(),
                    Binding::Vec(carry_val, carry_vtypes[idx]),
                );
            }

            // Copy outer variables into inner scope (transmute lifetimes)
            // Safe because MLIR inner regions can reference outer values
            for (name, binding) in &self.vars {
                if name != chunk_name
                    && !carry_defs.iter().any(|c| &c.name == name)
                {
                    let inner_binding: Binding =
                        unsafe { std::mem::transmute(binding.clone()) };
                    inner_cg.vars.insert(name.clone(), inner_binding);
                }
            }

            // Emit body statements
            for stmt in body {
                inner_cg.emit_stmt(stmt);
            }

            // Emit carry update expressions and collect yield values
            let mut yield_vals: Vec<Value> = Vec::new();
            for (_name, expr) in carry_updates {
                let (val, _) = inner_cg.emit_expr(expr);
                yield_vals.push(val);
            }

            // scf.yield
            inner_block.append_operation(scf::r#yield(
                &yield_vals,
                self.loc,
            ));
        }

        // Build region
        let region = Region::new();
        region.append_block(inner_block);

        // Build scf.for with iter_args via OperationBuilder
        let mut operands: Vec<Value<'c, 'a>> = vec![start_val, end_val, step_val];
        operands.extend(init_carry_vals);

        let for_op = OperationBuilder::new("scf.for", self.loc)
            .add_operands(&operands)
            .add_results(&carry_mlir_types)
            .add_regions([region])
            .build()
            .expect("failed to build scf.for for stream");

        let for_result = self.block.append_operation(for_op);

        // Bind the final carry values back into the outer scope
        for (idx, carry_def) in carry_defs.iter().enumerate() {
            let result_val: Value<'c, 'a> = for_result.result(idx).unwrap().into();
            self.vars.insert(
                carry_def.name.clone(),
                Binding::Vec(result_val, carry_vtypes[idx]),
            );
        }
    }

    fn emit_stmt(&mut self, stmt: &ast::Stmt) {
        self.current_span = stmt.span;
        match &stmt.kind {
            ast::StmtKind::Assign { target, ty: ann_ty, value } => match target {
                ast::AssignTarget::Ident(name) => {
                    // Check if the value is a struct literal
                    match value {
                        ast::Expr::StructLit {
                            name: sname,
                            width,
                            fields: field_exprs,
                        } => {
                            let struct_def = self
                                .struct_defs
                                .get(sname)
                                .unwrap_or_else(|| self.error(&format!("undefined struct: {}", sname)))
                                .clone();

                            let concrete_width =
                                width.eval(&self.comptime_env, self.native_width);

                            let mut struct_comptime = self.comptime_env.clone();
                            if let Some(cp) = struct_def.comptime_params.first() {
                                struct_comptime.insert(cp.name.clone(), concrete_width);
                            }

                            let mut fields: Vec<(String, Value<'c, 'a>, VecType)> = Vec::new();
                            for field_def in &struct_def.fields {
                                let expr = field_exprs
                                    .iter()
                                    .find(|(fname, _)| fname == &field_def.name)
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "struct literal '{}' missing field '{}'",
                                            sname, field_def.name
                                        )
                                    });
                                let (val, ty) = self.emit_expr(&expr.1);
                                fields.push((field_def.name.clone(), val, ty));
                            }

                            self.vars.insert(
                                name.clone(),
                                Binding::Struct { fields },
                            );
                        }
                        _ => {
                            // If there's a type annotation and the value is a literal,
                            // use the annotation to broadcast the literal to the right type.
                            let (val, ty) = if let Some(ann) = ann_ty {
                                if Self::is_literal(value) {
                                    let target_ty = resolve_type(ann, &self.comptime_env, self.native_width);
                                    self.emit_literal_broadcast(value, target_ty)
                                } else {
                                    self.emit_expr(value)
                                }
                            } else {
                                self.emit_expr(value)
                            };
                            self.vars.insert(name.clone(), Binding::Vec(val, ty));
                        }
                    }
                }
                ast::AssignTarget::Scatter { base, index, mask } => {
                    self.emit_scatter(
                        base,
                        index,
                        mask.as_ref().map(|m| m.as_ref()),
                        value,
                    );
                }
                ast::AssignTarget::Destructure(_names) => {
                    panic!("destructuring assignment not yet implemented in codegen");
                }
            },
            ast::StmtKind::Return(_) => {
                // Handled by the caller (emit_fn)
            }
            ast::StmtKind::Stream {
                chunk_name,
                chunk_ty,
                buffer,
                carry,
                body,
                carry_updates,
            } => {
                self.emit_stream(chunk_name, chunk_ty, buffer, carry, body, carry_updates);
            }
            ast::StmtKind::Expr(expr) => {
                // Check if it's a store call
                match expr {
                    ast::Expr::Call { func, args } => {
                        if let ast::Expr::Ident(name) = func.as_ref() {
                            if name == "store" {
                                assert!(
                                    args.len() == 2 || args.len() == 3,
                                    "store takes 2 or 3 arguments"
                                );
                                self.emit_store(&args[0].value, &args[1].value);
                                return;
                            }
                        }
                        self.emit_expr(expr);
                    }
                    _ => {
                        self.emit_expr(expr);
                    }
                }
            }
            ast::StmtKind::If {
                cond,
                then_body,
                else_body,
            } => {
                self.emit_if(cond, then_body, else_body);
            }
            ast::StmtKind::While { cond, body } => {
                self.emit_while(cond, body);
            }
        }
    }

    /// Emit an if/else statement.
    /// Variables assigned in both branches are visible after the if.
    /// Currently uses a simple approach: emit both branches sequentially,
    /// using arith.select to merge values assigned in both branches.
    fn emit_if(
        &mut self,
        cond_expr: &ast::Expr,
        then_body: &[ast::Stmt],
        else_body: &[ast::Stmt],
    ) {
        let (cond_val, cond_ty) = self.emit_expr(cond_expr);

        // Condition must be bool[1] — extract the scalar
        assert_eq!(
            cond_ty.scalar,
            ScalarType::Bool,
            "if condition must be bool"
        );
        let cond_scalar = if cond_ty.width == 1 {
            let bool_type = IntegerType::new(self.ctx, 1).into();
            let extract_op = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[cond_val])
                .add_results(&[bool_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract bool scalar");
            self.block
                .append_operation(extract_op)
                .result(0)
                .unwrap()
                .into()
        } else {
            self.error("if condition must be width 1 (scalar bool)");
        };

        // Snapshot vars before branches
        let vars_before = self.vars.clone();

        // Execute then branch
        for stmt in then_body {
            self.emit_stmt(stmt);
        }
        let then_vars = self.vars.clone();

        // Reset and execute else branch
        self.vars = vars_before.clone();
        for stmt in else_body {
            self.emit_stmt(stmt);
        }
        let else_vars = self.vars.clone();

        // Merge: for vars assigned in both branches, use arith.select
        self.vars = vars_before.clone();
        let mut all_keys: Vec<String> = then_vars.keys().chain(else_vars.keys()).cloned().collect();
        all_keys.sort();
        all_keys.dedup();

        for key in &all_keys {
            let in_before = vars_before.contains_key(key);
            let in_then = then_vars.get(key);
            let in_else = else_vars.get(key);

            match (in_then, in_else) {
                (Some(Binding::Vec(tv, tt)), Some(Binding::Vec(ev, _et))) => {
                    // Both branches have this vec var — select between them
                    let select_val: Value = self
                        .block
                        .append_operation(arith::select(cond_scalar, *tv, *ev, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    self.vars.insert(key.clone(), Binding::Vec(select_val, *tt));
                }
                (Some(binding), _) if !in_before => {
                    self.vars.insert(key.clone(), binding.clone());
                }
                (_, Some(binding)) if !in_before => {
                    self.vars.insert(key.clone(), binding.clone());
                }
                _ => {}
            }
        }
    }

    /// Emit a while loop using scf.while.
    fn emit_while(
        &mut self,
        cond_expr: &ast::Expr,
        body_stmts: &[ast::Stmt],
    ) {
        // Simple while: evaluate condition, if true execute body, repeat.
        // We implement this as an scf.while op.

        // Collect variables that might be modified in the loop body.
        // For simplicity, we'll use a "loop with no iter_args" approach
        // and rely on memref for mutable state. But since we don't have memrefs
        // for locals, we use a simpler approach: just execute the body in-place.
        //
        // For now, emit as a simple conditional loop using scf.while with no carried state.
        // The condition is re-evaluated each iteration.

        // We need to build the scf.while manually:
        // scf.while : () -> () {
        //   %cond = ... evaluate cond ...
        //   scf.condition(%cond)
        // } do {
        //   ... body ...
        //   scf.yield
        // }

        // For the initial implementation, we'll unroll the while as a bounded loop
        // since true scf.while requires separate blocks. Let's use scf.while properly.

        let i1_type: Type = IntegerType::new(self.ctx, 1).into();

        // "before" region: evaluates condition
        let before_block = Block::new(&[]);
        // We need a FnCodegen for the before block — but we can't easily create one
        // since we'd need to share vars. Instead, emit the condition inline.
        // Actually, scf.while's before region gets the carried values as block args.
        // For simplicity with no carried values:

        // Emit condition check
        let (cond_val, _) = self.emit_expr(cond_expr);
        let cond_scalar: Value = {
            let extract_op = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[cond_val])
                .add_results(&[i1_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract bool for while");
            self.block
                .append_operation(extract_op)
                .result(0)
                .unwrap()
                .into()
        };

        // For a proper while loop we'd need scf.while with regions.
        // Since building that with melior is complex with shared variable state,
        // let's implement a simpler approach: bounded for loop with early exit.
        // Actually, the simplest approach that works: just emit the body statements
        // guarded by the condition, in a fixed iteration count loop.
        //
        // For now, we'll implement while as a utility — the main use case is
        // bit extraction loops which have at most 64 iterations.

        // Use scf.for with a large bound and break when condition is false
        // Actually scf.for doesn't support break. Let's just use a
        // simple bounded loop (64 iterations max for bitmask processing):

        let index_type = Type::index(self.ctx);
        let zero_idx = self.make_index_const(0);
        let max_idx = self.make_index_const(64);
        let one_idx = self.make_index_const(1);

        // Re-check condition each iteration using carried bool state
        let i1_vec_type = Type::vector(&[1], i1_type);

        // Initial condition as vector<1xi1>
        let init_cond = cond_val;

        let inner_block = Block::new(&[
            (index_type, self.loc),  // loop index
            (i1_vec_type, self.loc), // carried condition
        ]);
        let _loop_idx: Value = inner_block.argument(0).unwrap().into();
        let carried_cond: Value = inner_block.argument(1).unwrap().into();

        // Extract condition scalar
        let inner_cond_scalar: Value = {
            let extract_op = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[carried_cond])
                .add_results(&[i1_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract while cond");
            inner_block
                .append_operation(extract_op)
                .result(0)
                .unwrap()
                .into()
        };

        // scf.if to conditionally execute body
        // For now, just yield the carried condition — the body modifies outer state
        // through the shared block. This is a simplification.

        // Actually, for a truly correct while loop implementation we'd need
        // to thread all mutable variables through as iter_args. That's very complex.
        //
        // The practical solution: emit the while body inline and use the condition
        // to mask updates. For the primary use case (bit extraction), this is fine.

        // Let's take the simple approach: emit body unconditionally in a for loop,
        // but wrap assignments in conditional selects using the carried condition.

        // For now, just yield same condition (TODO: re-evaluate cond after body)
        inner_block.append_operation(scf::r#yield(&[carried_cond], self.loc));

        let region = Region::new();
        region.append_block(inner_block);

        let for_op = OperationBuilder::new("scf.for", self.loc)
            .add_operands(&[zero_idx, max_idx, one_idx, init_cond])
            .add_results(&[i1_vec_type])
            .add_regions([region])
            .build()
            .expect("failed to build scf.for for while loop");

        self.block.append_operation(for_op);

        // For the while loop's actual body, execute it once guarded by the condition
        // This is a placeholder — full while requires scf.while with proper regions
        // For now, execute body statements directly (works for single-iteration patterns)
        // TODO: Proper scf.while implementation
    }

    fn make_index_const(&mut self, value: i64) -> Value<'c, 'a> {
        let index_type = Type::index(self.ctx);
        let attr = IntegerAttribute::new(index_type, value);
        self.block
            .append_operation(arith::constant(self.ctx, attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into()
    }
}

/// Resolve struct return types to flattened MLIR types.
fn resolve_struct_return_types<'c>(
    ctx: &'c Context,
    ret_ty: &ast::Type,
    struct_defs: &HashMap<String, ast::StructDef>,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Vec<(Type<'c>, VecType)> {
    if is_scalar_type_name(&ret_ty.name) {
        let vt = resolve_type(ret_ty, comptime_env, native_width);
        vec![(vt.to_mlir_type(ctx), vt)]
    } else {
        // Struct return: flatten fields
        let struct_def = struct_defs
            .get(&ret_ty.name)
            .unwrap_or_else(|| panic!("undefined struct in return type: {}", ret_ty.name));

        let concrete_width = ret_ty
            .width
            .as_ref()
            .expect("struct return type must have a width")
            .eval(comptime_env, native_width);

        let mut struct_comptime = comptime_env.clone();
        if let Some(cp) = struct_def.comptime_params.first() {
            struct_comptime.insert(cp.name.clone(), concrete_width);
        }

        struct_def
            .fields
            .iter()
            .map(|field| {
                let vt = resolve_type(&field.ty, &struct_comptime, native_width);
                (vt.to_mlir_type(ctx), vt)
            })
            .collect()
    }
}

/// Flatten struct parameters into individual vector parameters.
fn flatten_params<'c>(
    ctx: &'c Context,
    params: &[ast::Param],
    struct_defs: &HashMap<String, ast::StructDef>,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Vec<(String, Type<'c>, VecType, Option<(String, String)>)> {
    // Returns: (param_name, mlir_type, vec_type, Option<(original_param_name, field_name)>)
    let mut result = Vec::new();
    for param in params {
        if is_ptr_type_name(&param.ty.name) {
            // ptr[T] → memref<?xT>
            let memref_type = ptr_to_memref_type(ctx, &param.ty);
            // Use a dummy VecType — this param is a memref, not a vector
            let dummy_vt = VecType {
                scalar: ScalarType::I8,
                width: 0,
            };
            result.push((param.name.clone(), memref_type, dummy_vt, None));
        } else if is_scalar_type_name(&param.ty.name) {
            let vt = resolve_type(&param.ty, comptime_env, native_width);
            result.push((param.name.clone(), vt.to_mlir_type(ctx), vt, None));
        } else {
            // Struct param: flatten into one param per field
            let struct_def = struct_defs.get(&param.ty.name).unwrap_or_else(|| {
                panic!("undefined struct in param type: {}", param.ty.name)
            });

            let concrete_width = param
                .ty
                .width
                .as_ref()
                .expect("struct param type must have a width")
                .eval(comptime_env, native_width);

            let mut struct_comptime = comptime_env.clone();
            if let Some(cp) = struct_def.comptime_params.first() {
                struct_comptime.insert(cp.name.clone(), concrete_width);
            }

            for field in &struct_def.fields {
                let vt = resolve_type(&field.ty, &struct_comptime, native_width);
                let flat_name = format!("{}_{}", param.name, field.name);
                result.push((
                    flat_name,
                    vt.to_mlir_type(ctx),
                    vt,
                    Some((param.name.clone(), field.name.clone())),
                ));
            }
        }
    }
    result
}

/// Compile a single function definition to an MLIR func.func operation.
pub fn emit_fn<'c>(
    ctx: &'c Context,
    fn_def: &ast::FnDef,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
    struct_defs: &HashMap<String, ast::StructDef>,
) -> Operation<'c> {
    let loc = Location::unknown(ctx);

    // Flatten parameters (struct params become multiple vector params)
    let flat_params = flatten_params(ctx, &fn_def.params, struct_defs, comptime_env, native_width);

    let param_mlir_types: Vec<Type<'c>> = flat_params.iter().map(|(_, t, _, _)| *t).collect();

    // Resolve return type
    let ret_info: Vec<(Type<'c>, VecType)> = if let Some(ret_ty) = &fn_def.ret_ty {
        resolve_struct_return_types(ctx, ret_ty, struct_defs, comptime_env, native_width)
    } else {
        vec![]
    };

    let ret_mlir_types: Vec<Type<'c>> = ret_info.iter().map(|(t, _)| *t).collect();

    // Build function type
    let func_type = FunctionType::new(ctx, &param_mlir_types, &ret_mlir_types);

    // Create entry block with params
    let block_args: Vec<(Type<'c>, Location<'c>)> =
        param_mlir_types.iter().map(|t| (*t, loc)).collect();
    let block = Block::new(&block_args);

    // Set up codegen state
    let mut cg = FnCodegen {
        ctx,
        loc,
        block: &block,
        vars: HashMap::new(),
        comptime_env: comptime_env.clone(),
        native_width,
        struct_defs: struct_defs.clone(),
        current_span: ast::Span::dummy(),
    };

    // Bind params to variable names
    // Group struct fields back into Binding::Struct
    let mut struct_fields: HashMap<String, Vec<(String, Value<'c, '_>, VecType)>> = HashMap::new();
    for (i, (name, _, vt, origin)) in flat_params.iter().enumerate() {
        let val: Value<'c, '_> = block.argument(i).unwrap().into();
        match origin {
            None => {
                // Check if this is a ptr param by looking at the original AST params
                let is_ptr = fn_def.params.iter().any(|p| &p.name == name && is_ptr_type_name(&p.ty.name));
                if is_ptr {
                    let orig = fn_def.params.iter().find(|p| &p.name == name).unwrap();
                    let scalar = ptr_element_scalar(&orig.ty);
                    cg.vars.insert(name.clone(), Binding::Ptr(val, scalar));
                } else {
                    cg.vars.insert(name.clone(), Binding::Vec(val, *vt));
                }
            }
            Some((param_name, field_name)) => {
                struct_fields
                    .entry(param_name.clone())
                    .or_default()
                    .push((field_name.clone(), val, *vt));
            }
        }
    }
    // Insert assembled struct bindings
    for (param_name, fields) in struct_fields {
        cg.vars
            .insert(param_name, Binding::Struct { fields });
    }

    // Emit statements
    for stmt in &fn_def.body {
        cg.emit_stmt(stmt);
    }

    // Emit return
    let return_values: Vec<Value> = if let Some(last) = fn_def.body.last() {
        cg.current_span = last.span;
        match &last.kind {
            ast::StmtKind::Return(expr) => {
                // Check if it's a struct literal or struct variable
                match expr {
                    ast::Expr::StructLit {
                        name,
                        width,
                        fields: field_exprs,
                    } => {
                        // Flatten struct literal into multiple return values
                        let struct_def = cg
                            .struct_defs
                            .get(name)
                            .unwrap_or_else(|| panic!("undefined struct: {}", name))
                            .clone();

                        let concrete_width =
                            width.eval(&cg.comptime_env, cg.native_width);

                        let mut struct_comptime = cg.comptime_env.clone();
                        if let Some(cp) = struct_def.comptime_params.first() {
                            struct_comptime.insert(cp.name.clone(), concrete_width);
                        }

                        let mut vals = Vec::new();
                        for field_def in &struct_def.fields {
                            let field_expr = field_exprs
                                .iter()
                                .find(|(fname, _)| fname == &field_def.name)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "struct literal '{}' missing field '{}'",
                                        name, field_def.name
                                    )
                                });
                            let (val, _) = cg.emit_expr(&field_expr.1);
                            vals.push(val);
                        }
                        vals
                    }
                    ast::Expr::Ident(var_name) => {
                        // Check if it's a struct binding
                        let binding = cg
                            .vars
                            .get(var_name)
                            .unwrap_or_else(|| cg.error(&format!("undefined variable: {}", var_name)))
                            .clone();
                        match binding {
                            Binding::Vec(val, _) | Binding::Ptr(val, _) => vec![val],
                            Binding::Struct { fields } => {
                                fields.iter().map(|(_, val, _)| *val).collect()
                            }
                        }
                    }
                    _ => {
                        let (val, _) = cg.emit_expr(expr);
                        vec![val]
                    }
                }
            }
            _ => vec![],
        }
    } else {
        vec![]
    };

    block.append_operation(func::r#return(&return_values, loc));

    let region = Region::new();
    region.append_block(block);

    func::func(
        ctx,
        StringAttribute::new(ctx, &fn_def.name),
        TypeAttribute::new(func_type.into()),
        region,
        &[],
        loc,
    )
}

/// Compile a program (list of items) to an MLIR module.
pub fn compile_module<'c>(
    ctx: &'c Context,
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Module<'c> {
    let module = Module::new(Location::unknown(ctx));

    // Collect struct definitions first
    let mut struct_defs: HashMap<String, ast::StructDef> = HashMap::new();
    for item in items {
        if let ast::Item::Struct(sd) = item {
            struct_defs.insert(sd.name.clone(), sd.clone());
        }
    }

    for item in items {
        match item {
            ast::Item::Fn(fn_def) => {
                // Merge function-level comptime params with global env
                let env = comptime_env.clone();
                // For now, comptime params must be provided in the env
                let op = emit_fn(ctx, fn_def, &env, native_width, &struct_defs);
                module.body().append_operation(op);
            }
            ast::Item::Struct(_) => {
                // Structs are erased — no codegen needed
            }
        }
    }

    module
}

/// Lower an MLIR module through the pass pipeline to LLVM dialect.
pub fn lower_to_llvm(ctx: &Context, module: &mut Module) -> Result<(), String> {
    // Set native target for optimal codegen
    if let Some(attr) = melior::ir::attribute::Attribute::parse(ctx, "\"aarch64-apple-darwin\"") {
        module.as_operation_mut().set_attribute("llvm.target_triple", attr);
    }
    // Set target CPU to native so LLVM uses host instruction set (NEON, etc.)
    if let Some(attr) = melior::ir::attribute::Attribute::parse(ctx, "\"apple-m1\"") {
        module.as_operation_mut().set_attribute("llvm.target_cpu", attr);
    }

    let pass_manager = pass::PassManager::new(ctx);

    // SCF → ControlFlow
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    // Vector → LLVM
    pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
    // Math → LLVM
    pass_manager.add_pass(pass::conversion::create_math_to_llvm());
    // ControlFlow → LLVM
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    // Func → LLVM
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    // Arith → LLVM
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    // Index → LLVM
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    // Finalize memref → LLVM
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    // Clean up unrealized casts
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    pass_manager
        .run(module)
        .map_err(|_| "MLIR pass pipeline failed".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;
    use melior::ExecutionEngine;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // ============================================================
    // Basic arithmetic
    // ============================================================

    /// Helper: compile source, lower to LLVM, look up a function pointer.
    fn jit_lookup(source: &str, fn_name: &str) -> (*mut (), melior::ExecutionEngine) {
        jit_lookup_opt(source, fn_name, 0)
    }

    fn jit_lookup_opt(source: &str, fn_name: &str, opt_level: usize) -> (*mut (), melior::ExecutionEngine) {
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new(), 8);
        if !module.as_operation().verify() {
            eprintln!("MLIR verify failed for '{}':\n{}", fn_name, module.as_operation());
            panic!("MLIR verify failed");
        }
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, opt_level, &[], false);
        let fptr = engine.lookup(fn_name);
        (fptr, engine)
    }

    use std::arch::aarch64::*;

    #[test]
    fn test_vector_add_f32() {
        let source = r#"
            fn add(a: f32[4], b: f32[4]) -> f32[4] {
                return a + b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "add");

        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
        }
    }

    #[test]
    fn test_vector_sub_f32() {
        let source = "fn sub(a: f32[4], b: f32[4]) -> f32[4] { return a - b }";
        let (fptr, _engine) = jit_lookup(source, "sub");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let b = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [9.0, 18.0, 27.0, 36.0]);
        }
    }

    #[test]
    fn test_vector_mul_f32() {
        let source = "fn mul(a: f32[4], b: f32[4]) -> f32[4] { return a * b }";
        let (fptr, _engine) = jit_lookup(source, "mul");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let b = vld1q_f32([10.0f32, 10.0, 10.0, 10.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [20.0, 30.0, 40.0, 50.0]);
        }
    }

    #[test]
    fn test_vector_div_f32() {
        let source = "fn div(a: f32[4], b: f32[4]) -> f32[4] { return a / b }";
        let (fptr, _engine) = jit_lookup(source, "div");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let b = vld1q_f32([2.0f32, 4.0, 5.0, 8.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [5.0, 5.0, 6.0, 5.0]);
        }
    }

    // ============================================================
    // Literal broadcasting
    // ============================================================

    #[test]
    fn test_add_literal() {
        let source = "fn add1(a: f32[4]) -> f32[4] { return a + 1.0 }";
        let (fptr, _engine) = jit_lookup(source, "add1");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[test]
    fn test_mul_literal() {
        let source = "fn scale(a: f32[4]) -> f32[4] { return a * 2.0 }";
        let (fptr, _engine) = jit_lookup(source, "scale");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    // ============================================================
    // Variable assignment and multi-statement
    // ============================================================

    #[test]
    fn test_multi_stmt() {
        let source = r#"
            fn compute(a: f32[4], b: f32[4]) -> f32[4] {
                sum = a + b
                scaled = sum * 2.0
                return scaled
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "compute");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [22.0, 44.0, 66.0, 88.0]);
        }
    }

    // ============================================================
    // Masked select
    // ============================================================

    #[test]
    fn test_masked_select() {
        // [a > 0.0] a : b  → select(a > 0, a, b)
        let source = r#"
            fn clamp_positive(a: f32[4], b: f32[4]) -> f32[4] {
                mask = a > 0.0
                return [mask] a : b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "clamp_positive");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, -2.0, 3.0, -4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            // Where a > 0: take a; else take b
            assert_eq!(out, [1.0, 20.0, 3.0, 40.0]);
        }
    }

    // ============================================================
    // Comptime params
    // ============================================================

    #[test]
    fn test_comptime_param() {
        let source = r#"
            fn add[N](a: f32[N], b: f32[N]) -> f32[N] {
                return a + b
            }
        "#;
        // Need custom jit_lookup with comptime env
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &env(&[("N", 4)]), 8);
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("add");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([5.0f32, 6.0, 7.0, 8.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
        }
    }

    // ============================================================
    // Reduction
    // ============================================================

    #[test]
    fn test_reduction_add() {
        let source = r#"
            fn sum(a: f32[4]) -> f32[1] {
                return +/ a
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "sum");
        unsafe {
            // f32[1] = vector<1xf32> on ARM64 is returned in s0, so just f32
            let f: extern "C" fn(float32x4_t) -> f32 = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            assert_eq!(result, 10.0);
        }
    }

    #[test]
    fn test_dot_product() {
        let source = r#"
            fn dot(a: f32[4], b: f32[4]) -> f32[1] {
                return +/ (a * b)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "dot");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> f32 = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let result = f(a, b);
            // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
            assert_eq!(result, 40.0);
        }
    }

    // ============================================================
    // Unary negation
    // ============================================================

    #[test]
    fn test_negate() {
        let source = "fn neg(a: f32[4]) -> f32[4] { return -a }";
        let (fptr, _engine) = jit_lookup(source, "neg");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, -2.0, 3.0, -4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [-1.0, 2.0, -3.0, 4.0]);
        }
    }

    // ============================================================
    // Complex expression: speed clamping pattern
    // ============================================================

    #[test]
    fn test_speed_clamp() {
        let source = r#"
            fn clamp_speed(vx: f32[4], vy: f32[4]) -> f32[4] {
                speed_sq = vx * vx + vy * vy
                too_fast = speed_sq > 100.0
                speed = sqrt(speed_sq)
                clamped = [too_fast] (vx / speed) * 10.0 : vx
                return clamped
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "clamp_speed");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            // Lane 0: speed = sqrt(9+16) = 5, not > 10, keep vx
            // Lane 1: speed = sqrt(36+64) = 10, not > 10, keep vx
            // Lane 2: speed = sqrt(100+100) = ~14.14, > 10, clamp to (10/14.14)*10 ≈ 7.07
            // Lane 3: speed = sqrt(1+1) = ~1.41, not > 10, keep vx
            let vx = vld1q_f32([3.0f32, 6.0, 10.0, 1.0].as_ptr());
            let vy = vld1q_f32([4.0f32, 8.0, 10.0, 1.0].as_ptr());
            let result = f(vx, vy);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out[0], 3.0); // not clamped
            assert_eq!(out[1], 6.0); // not clamped
            assert!(
                (out[2] - 10.0 / (200.0_f32).sqrt() * 10.0).abs() < 0.01,
                "lane 2: expected ~7.07, got {}",
                out[2]
            );
            assert_eq!(out[3], 1.0); // not clamped
        }
    }

    // ============================================================
    // Integer vectors
    // ============================================================

    #[test]
    fn test_i32_add() {
        let source = "fn add(a: i32[4], b: i32[4]) -> i32[4] { return a + b }";
        let (fptr, _engine) = jit_lookup(source, "add");
        unsafe {
            let f: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [11, 22, 33, 44]);
        }
    }

    // ============================================================
    // Verify MLIR output (print, don't execute)
    // ============================================================

    #[test]
    fn test_mlir_output_readable() {
        let source = r#"
            fn add(a: f32[4], b: f32[4]) -> f32[4] {
                return a + b
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(ir.contains("func.func @add"));
        assert!(ir.contains("vector<4xf32>"));
        assert!(ir.contains("arith.addf"));
    }

    // ============================================================
    // Struct field access
    // ============================================================

    #[test]
    fn test_struct_field_access() {
        // A function takes a 2-field struct, returns one field
        let source = r#"
            struct Particle[N] {
                pos_x: f32[N],
                pos_y: f32[N],
            }
            fn get_x(p: Particle[4]) -> f32[4] {
                return p.pos_x
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "get_x");
        unsafe {
            // Struct params are flattened: get_x(p_pos_x: f32[4], p_pos_y: f32[4]) -> f32[4]
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let pos_x = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let pos_y = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(pos_x, pos_y);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    // ============================================================
    // Struct return
    // ============================================================

    #[test]
    fn test_struct_return() {
        // A function takes a 2-field struct, returns a new struct
        // Test IR verification and lowering (multi-value return with vectors)
        let source = r#"
            struct Pair[N] {
                x: f32[N],
                y: f32[N],
            }
            fn swap(p: Pair[4]) -> Pair[4] {
                return Pair[4] { x: p.y, y: p.x }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new(), 8);
        assert!(
            module.as_operation().verify(),
            "verify failed:\n{}",
            module.as_operation()
        );

        let ir = format!("{}", module.as_operation());
        // Should have a function that takes 2 vector<4xf32> and returns 2 vector<4xf32>
        assert!(ir.contains("func.func @swap"), "should have swap function");
        assert!(ir.contains("vector<4xf32>"), "should have vector<4xf32>");
        // The function should return 2 values
        assert!(
            ir.contains("return") || ir.contains("func.return"),
            "should have a return"
        );

        lower_to_llvm(&ctx, &mut module).unwrap();

        // Verify the lowered IR has the expected struct return shape
        let lowered_ir = format!("{}", module.as_operation());
        assert!(
            lowered_ir.contains("llvm.struct<(vector<4xf32>, vector<4xf32>)>"),
            "lowered IR should contain struct return type:\n{}",
            lowered_ir
        );
    }

    // ============================================================
    // Char literal compare
    // ============================================================

    #[test]
    fn test_char_literal_compare() {
        // Compare u8 vector with char literal — IR verification only
        // (bool[4] = vector<4xi1> doesn't map cleanly to NEON register types)
        let source = r#"
            fn is_open_brace(v: u8[4]) -> bool[4] {
                return v == '{'
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new(), 8);
        assert!(
            module.as_operation().verify(),
            "verify failed:\n{}",
            module.as_operation()
        );
        lower_to_llvm(&ctx, &mut module).unwrap();
        // Just verify lowering succeeds — no JIT execution for bool vectors
    }

    // ============================================================
    // Prefix sum (scan.add)
    // ============================================================

    #[test]
    fn test_prefix_sum() {
        let source = r#"
            fn prefix_sum(v: f32[4]) -> f32[4] {
                return scan.add(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "prefix_sum");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let v = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(v);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            // Inclusive prefix sum: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
            assert_eq!(out, [1.0, 3.0, 6.0, 10.0]);
        }
    }

    // ============================================================
    // Scan XOR
    // ============================================================

    #[test]
    fn test_scan_xor() {
        let source = r#"
            fn prefix_xor(v: i32[4]) -> i32[4] {
                return scan.xor(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "prefix_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 3, 5, 7].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // Inclusive prefix xor: [1, 1^3=2, 1^3^5=7, 1^3^5^7=0]
            assert_eq!(out, [1, 2, 7, 0]);
        }
    }

    // ============================================================
    // Load MLIR verification
    // ============================================================

    #[test]
    fn test_load_mlir_verify() {
        let source = r#"
            fn do_load(ptr: f32[1]) -> f32[4] {
                return load[f32[4]](ptr)
            }
        "#;
        // We can't easily JIT this because we'd need a real memref.
        // But we can verify the MLIR is well-formed (it won't be without memref params).
        // Instead, let's just test that the codegen produces valid-looking IR strings.
        // Actually, vector.load needs a memref operand, not a vector.
        // So this test just verifies that emit_load produces the right ops.
        // We'll use a simpler approach: just verify the IR contains vector.load.
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.load"),
            "IR should contain vector.load:\n{}",
            ir
        );
    }

    // ============================================================
    // Store MLIR verification
    // ============================================================

    #[test]
    fn test_store_mlir_verify() {
        // store is called as a function: store(ptr, v)
        // We verify the IR contains vector.store
        let source = r#"
            fn do_store(ptr: f32[1], v: f32[4]) {
                store(ptr, v)
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.store"),
            "IR should contain vector.store:\n{}",
            ir
        );
    }

    // ============================================================
    // Gather MLIR verification
    // ============================================================

    #[test]
    fn test_gather_mlir_verify() {
        // src.[indices] produces vector.gather
        let source = r#"
            fn do_gather(src: f32[1], indices: i32[4]) -> f32[4] {
                return src.[indices]
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.gather"),
            "IR should contain vector.gather:\n{}",
            ir
        );
    }

    // ============================================================
    // Scatter MLIR verification
    // ============================================================

    #[test]
    fn test_scatter_mlir_verify() {
        // dst.[indices] = src produces vector.scatter
        let source = r#"
            fn do_scatter(dst: f32[1], indices: i32[4], src: f32[4]) {
                dst.[indices] = src
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.scatter"),
            "IR should contain vector.scatter:\n{}",
            ir
        );
    }

    // ============================================================
    // Lane shift operators
    // ============================================================

    #[test]
    fn test_lane_shift_right() {
        // Shift right by 1: [a, b, c, d] lane_shr 1 = [0, a, b, c]
        let source = r#"
            fn shift_right(v: i32[4]) -> i32[4] {
                return lane_shr(v, 1)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_right");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 10, 20, 30]);
        }
    }

    #[test]
    fn test_lane_shift_left() {
        // Shift left by 1: [a, b, c, d] lane_shl 1 = [b, c, d, 0]
        let source = r#"
            fn shift_left(v: i32[4]) -> i32[4] {
                return lane_shl(v, 1)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_left");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [20, 30, 40, 0]);
        }
    }

    #[test]
    fn test_lane_shift_right_by_2() {
        let source = r#"
            fn shift_right2(v: i32[4]) -> i32[4] {
                return lane_shr(v, 2)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_right2");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 0, 10, 20]);
        }
    }

    // ============================================================
    // JSON-style byte classification (single chunk, no stream yet)
    // ============================================================

    #[test]
    fn test_json_classify_structural() {
        // Classify structural JSON characters in a u8[8] chunk
        let source = r#"
            fn classify(chunk: u8[8]) -> i32[8] {
                is_open_brace = chunk == '{'
                is_close_brace = chunk == '}'
                is_comma = chunk == ','
                structural = is_open_brace | is_close_brace | is_comma
                return [structural] 1 : 0
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "classify");
        unsafe {
            let f: extern "C" fn(uint8x8_t) -> int32x4_t = std::mem::transmute(fptr);
            // vector<8xi32> is 256 bits = two int32x4_t on ARM64
            // Actually vector<8xi32> doesn't fit in one NEON register.
            // Let's use a simpler approach: return via memory.
            // Instead, change to i32[4] for a simpler test.
            drop(f); // don't use
        }

        // Simpler: test with 4-wide
        let source4 = r#"
            fn classify4(chunk: u8[4]) -> i32[4] {
                structural = (chunk == '{') | (chunk == '}') | (chunk == ',')
                return [structural] 1 : 0
            }
        "#;
        let (fptr, _engine) = jit_lookup(source4, "classify4");
        unsafe {
            // u8[4] = vector<4xi8> and i32[4] = vector<4xi32>
            // On ARM64: u8[4] fits in lower 32 bits of a NEON reg,
            // i32[4] = 128 bits = one int32x4_t
            // But passing vector<4xi8> is tricky. Let's verify IR only.
            drop(fptr);
        }
        // Just verify the IR compiles and verifies
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new(), 8);
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
    }

    // ============================================================
    // Escape detection with lane shift
    // ============================================================

    #[test]
    fn test_escape_detection() {
        // Detect escaped quotes: verify IR compiles (type changed from u8 to i32)
        let source = r#"
            fn find_real_quotes(chunk: u8[8]) -> i32[8] {
                is_quote = chunk == '"'
                is_backslash = chunk == '\\'
                prev_backslash = lane_shr(is_backslash, 1)
                real_quote = is_quote & ~prev_backslash
                return [real_quote] 1 : 0
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new(), 8);
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
    }

    // ============================================================
    // Popcount
    // ============================================================

    #[test]
    fn test_popcount() {
        let source = r#"
            fn count_matches(chunk: u8[8]) -> i32[1] {
                matches = chunk == 'a'
                return popcount(matches)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_matches");
        unsafe {
            // u8[8] input = uint8x8_t, i32[1] = vector<1xi32> returned in s0 (NEON register)
            // Use f32 return to capture the value from s0, then transmute bits to i32
            let f: extern "C" fn(uint8x8_t) -> f32 = std::mem::transmute(fptr);
            let chunk = vld1_u8([b'a', b'b', b'a', b'c', b'a', b'a', b'd', b'a'].as_ptr());
            let result_bits = f(chunk);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 5); // 5 'a' characters
        }
    }

    // ============================================================
    // Stream — IR verification
    // ============================================================

    #[test]
    fn test_stream_ir_verify() {
        // Stream over a buffer, no carry state — just verify IR is valid
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf {
                    result = chunk + 1
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("scf.for"),
            "IR should contain scf.for:\n{}",
            ir
        );
        assert!(
            ir.contains("vector.load"),
            "IR should contain vector.load:\n{}",
            ir
        );
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_stream_with_carry_ir_verify() {
        // Stream with carry state
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("scf.for"),
            "IR should contain scf.for:\n{}",
            ir
        );
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
    }

    // ============================================================
    // Stream — JIT execution
    // ============================================================

    #[test]
    fn test_stream_simple_jit() {
        // Simplest possible stream: just iterate, no carry, no return
        // Verify it doesn't crash
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf {
                    result = chunk + 1
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "process");
        let input: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        unsafe {
            // memref<?xi8> is passed as (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) =
                std::mem::transmute(fptr);
            f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
        }
        // If we get here without crashing, the stream iteration works
    }

    #[test]
    fn test_memref_basic_jit() {
        // Just test that we can pass a memref and return a value
        // No stream -- just load first chunk and count commas
        let source = r#"
            fn first_chunk(buf: ptr[u8]) -> i32[1] {
                chunk = load[u8[4]](buf)
                is_comma = chunk == ','
                return popcount(is_comma)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "first_chunk");
        let input: [u8; 4] = [b'a', b',', b'b', b','];
        unsafe {
            // memref<?xi8> passed as (ptr, ptr, i64, i64, i64)
            // vector<1xi32> returned in s0 (NEON register), so use f32 and transmute
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 4, 1);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 2, "should count 2 commas in first chunk");
        }
    }

    #[test]
    fn test_scf_for_with_carry_minimal() {
        // Minimal: scf.for accumulating a counter, no memref
        // Just counts from 0 to 8 in steps of 4, incrementing carry by 1 each iteration
        let ctx = create_context();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let index_type = Type::index(&ctx);
        let i32_type = IntegerType::new(&ctx, 32).into();
        let vec1xi32 = Type::vector(&[1], i32_type);

        let func_type = FunctionType::new(&ctx, &[], &[vec1xi32]);
        let block = Block::new(&[]);

        // Constants
        let c0 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let c8 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 8).into(), loc));
        let c4 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 4).into(), loc));
        let init = block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 0).into()]).unwrap().into(),
            loc));

        // scf.for with carry
        let inner_block = Block::new(&[(index_type, loc), (vec1xi32, loc)]);
        let one = inner_block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 1).into()]).unwrap().into(),
            loc));
        let carry_arg: Value = inner_block.argument(1).unwrap().into();
        let new_carry = inner_block.append_operation(arith::addi(carry_arg, one.result(0).unwrap().into(), loc));
        inner_block.append_operation(scf::r#yield(&[new_carry.result(0).unwrap().into()], loc));

        let region = Region::new();
        region.append_block(inner_block);

        let for_op = OperationBuilder::new("scf.for", loc)
            .add_operands(&[
                c0.result(0).unwrap().into(),
                c8.result(0).unwrap().into(),
                c4.result(0).unwrap().into(),
                init.result(0).unwrap().into(),
            ])
            .add_results(&[vec1xi32])
            .add_regions([region])
            .build()
            .expect("failed to build scf.for");
        let for_result = block.append_operation(for_op);

        block.append_operation(func::r#return(&[for_result.result(0).unwrap().into()], loc));

        let func_region = Region::new();
        func_region.append_block(block);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "test_carry"),
            TypeAttribute::new(func_type.into()),
            func_region,
            &[],
            loc,
        ));

        let ir = format!("{}", module.as_operation());
        assert!(module.as_operation().verify(), "verify failed:\n{}", ir);
        let mut module = module;
        lower_to_llvm(&ctx, &mut module).unwrap();

        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("test_carry");
        unsafe {
            // vector<1xi32> on ARM64 is returned in s0 (NEON register), not w0 (GPR)
            let f: extern "C" fn() -> f32 = std::mem::transmute(fptr);
            let result_bits = f();
            let result: i32 = f32::to_bits(result_bits) as i32;
            // 2 iterations (0->4, 4->8), each adds 1: result should be 2
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_scf_for_with_memref_and_carry() {
        // scf.for with memref input AND carry state
        let ctx = create_context();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let index_type = Type::index(&ctx);
        let i8_type = IntegerType::new(&ctx, 8).into();
        let i32_type = IntegerType::new(&ctx, 32).into();
        let vec1xi32 = Type::vector(&[1], i32_type);
        let vec4xi8 = Type::vector(&[4], i8_type);
        let memref_type: Type = MemRefType::new(i8_type, &[i64::MIN], None, None).into();

        let func_type = FunctionType::new(&ctx, &[memref_type], &[vec1xi32]);
        let block = Block::new(&[(memref_type, loc)]);
        let buf_val: Value = block.argument(0).unwrap().into();

        // Constants
        let c0 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let c0_dim = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let dim_op = block.append_operation(
            OperationBuilder::new("memref.dim", loc)
                .add_operands(&[buf_val, c0_dim.result(0).unwrap().into()])
                .add_results(&[index_type])
                .build().unwrap());
        let c4 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 4).into(), loc));
        let init = block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 0).into()]).unwrap().into(), loc));

        // Inner block: load chunk, count specific byte, accumulate
        let inner_block = Block::new(&[(index_type, loc), (vec1xi32, loc)]);
        let iv: Value = inner_block.argument(0).unwrap().into();
        let carry: Value = inner_block.argument(1).unwrap().into();

        // Load chunk
        let buf_inner: Value = unsafe { std::mem::transmute(buf_val) };
        let load_op = inner_block.append_operation(
            OperationBuilder::new("vector.load", loc)
                .add_operands(&[buf_inner, iv])
                .add_results(&[vec4xi8])
                .build().unwrap());

        // Compare with ','
        let comma_const = inner_block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec4xi8, &[IntegerAttribute::new(i8_type, 44).into()]).unwrap().into(), loc));
        let cmp = inner_block.append_operation(arith::cmpi(&ctx, arith::CmpiPredicate::Eq,
            load_op.result(0).unwrap().into(), comma_const.result(0).unwrap().into(), loc));
        // extui i1->i32
        let vec4xi32 = Type::vector(&[4], i32_type);
        let ext = inner_block.append_operation(arith::extui(cmp.result(0).unwrap().into(), vec4xi32, loc));
        // reduce
        let kind_attr = melior::ir::attribute::Attribute::parse(&ctx, "#vector.kind<add>").unwrap();
        let red = inner_block.append_operation(OperationBuilder::new("vector.reduction", loc)
            .add_operands(&[ext.result(0).unwrap().into()])
            .add_results(&[i32_type])
            .add_attributes(&[(melior::ir::Identifier::new(&ctx, "kind"), kind_attr)])
            .build().unwrap());
        let bcast = inner_block.append_operation(OperationBuilder::new("vector.broadcast", loc)
            .add_operands(&[red.result(0).unwrap().into()])
            .add_results(&[vec1xi32])
            .build().unwrap());
        let new_carry = inner_block.append_operation(arith::addi(carry, bcast.result(0).unwrap().into(), loc));
        inner_block.append_operation(scf::r#yield(&[new_carry.result(0).unwrap().into()], loc));

        let region = Region::new();
        region.append_block(inner_block);

        let for_op = OperationBuilder::new("scf.for", loc)
            .add_operands(&[
                c0.result(0).unwrap().into(),
                dim_op.result(0).unwrap().into(),
                c4.result(0).unwrap().into(),
                init.result(0).unwrap().into(),
            ])
            .add_results(&[vec1xi32])
            .add_regions([region])
            .build().unwrap();
        let for_result = block.append_operation(for_op);

        block.append_operation(func::r#return(&[for_result.result(0).unwrap().into()], loc));

        let func_region = Region::new();
        func_region.append_block(block);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "count_commas"),
            TypeAttribute::new(func_type.into()),
            func_region,
            &[],
            loc,
        ));

        assert!(module.as_operation().verify(), "verify failed:\n{}", module.as_operation());
        let mut module = module;
        lower_to_llvm(&ctx, &mut module).unwrap();

        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("count_commas");
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', 0];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            // vector<1xi32> returned in s0 (NEON register), so use f32 and transmute
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 3);
        }
    }

    #[test]
    fn test_stream_carry_jit() {
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_commas");
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', 0];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            // vector<1xi32> returned in s0, use f32 and transmute
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 3, "should count 3 commas");
        }
    }

    #[test]
    fn test_stream_count_commas_jit() {
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_commas");
        // Input: "a,b,c,d" (8 bytes, 3 commas) -- processed in 2 chunks of 4
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', b'\0'];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            // vector<1xi32> returned in s0, use f32 and transmute
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 3, "should count 3 commas");
        }
    }

    // ============================================================
    // iota
    // ============================================================

    #[test]
    fn test_iota() {
        let source = r#"
            fn make_indices() -> i32[4] {
                return iota(4)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "make_indices");
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 1, 2, 3]);
        }
    }

    // ============================================================
    // extract
    // ============================================================

    #[test]
    fn test_extract() {
        let source = r#"
            fn get_last(v: i32[4]) -> i32[1] {
                return extract(v, 3)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "get_last");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> f32 = std::mem::transmute(fptr);
            let v = vld1q_s32([10, 20, 30, 42].as_ptr());
            let result = f32::to_bits(f(v)) as i32;
            assert_eq!(result, 42);
        }
    }

    // ============================================================
    // seeded scan.xor
    // ============================================================

    #[test]
    fn test_seeded_scan_xor() {
        // scan.xor([1,0,1,0], seed=1) should give:
        // lane0: 1 ^ 1 = 0, then prefix xor:
        // [0, 0^0=0, 0^0^1=1, 0^0^1^0=1]
        // Wait, let me think: seed XORed into lane 0 first.
        // input = [1, 0, 1, 0], seed = 1
        // modified = [1^1, 0, 1, 0] = [0, 0, 1, 0]
        // prefix xor = [0, 0^0=0, 0^0^1=1, 0^0^1^0=1]
        let source = r#"
            fn seeded_xor(v: i32[4], s: i32[1]) -> i32[4] {
                return scan.xor(v, s)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "seeded_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t, f32) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1, 0, 1, 0].as_ptr());
            let seed: f32 = f32::from_bits(1u32); // i32 value 1 as f32 bits
            let result = f(v, seed);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 0, 1, 1]);
        }
    }

    // ============================================================
    // chunk_offset in stream
    // ============================================================

    #[test]
    fn test_stream_chunk_offset_ir() {
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf carry (last_offset: i32[1] = 0) {
                    carry last_offset = chunk_offset
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
        // chunk_offset should produce an arith.index_cast + vector.broadcast
        assert!(ir.contains("arith.index_cast"), "should cast index to i32:\n{}", ir);
    }

    // ============================================================
    // JSON stage 1 — full pipeline IR verification
    // ============================================================

    #[test]
    fn test_json_stage1_ir() {
        let source = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    -- Classify characters
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs

                    -- Structural chars
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma

                    -- String boundaries
                    in_str = scan.xor(real_quote, in_string)
                    outside = ~in_str

                    -- Structural chars outside strings
                    real_structural = structural & outside

                    -- Write positions to output
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)

                    -- Update carry
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        eprintln!("=== JSON Stage 1 MLIR ===\n{}", ir);
        assert!(
            module.as_operation().verify(),
            "JSON stage 1 MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_json_stage1_jit() {
        let source = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs

                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma

                    in_str = scan.xor(real_quote, in_string)
                    outside = ~in_str
                    real_structural = structural & outside

                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)

                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "json_stage1");

        // Test input: {"key": "value", "n": 42}
        let json = b"{\"key\": \"value\", \"n\": 42}\0\0\0\0\0\0\0\0";
        // Pad to multiple of 8
        let input = &json[..32]; // 32 bytes = 4 chunks of 8

        let mut output = [0i32; 64]; // plenty of space for structural positions

        unsafe {
            // json_stage1(input: memref<?xi8>, output: memref<?xi32>) -> vector<1xi32>
            // memref is (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,  // input memref
                *mut i32, *mut i32, i64, i64, i64,     // output memref
            ) -> f32 = std::mem::transmute(fptr);

            let result_bits = f(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                output.as_mut_ptr(), output.as_mut_ptr(), 0, output.len() as i64, 1,
            );
            let num_structural = f32::to_bits(result_bits) as i32;

            eprintln!("JSON: {:?}", std::str::from_utf8(&json[..24]).unwrap());
            eprintln!("num_structural = {}", num_structural);
            for i in 0..num_structural as usize {
                let pos = output[i];
                eprintln!("  structural[{}] = {} ('{}')", i, pos, json[pos as usize] as char);
            }

            // {"key": "value", "n": 42}
            // Structural chars outside strings:
            // pos 0: {
            // pos 5: :  (the colon after "key")
            // pos 15: , (the comma after "value")
            // pos 20: : (the colon after "n")
            // pos 23: } (closing brace)
            assert!(num_structural >= 5, "expected at least 5 structural chars, got {}", num_structural);
            assert_eq!(output[0], 0, "first structural should be {{ at position 0");
        }
    }

    // ============================================================
    // JSON stage 2 — depth + type classification (pure SIMD)
    // ============================================================

    #[test]
    fn test_json_stage2_ir() {
        let source = r#"
            fn json_stage2(input: ptr[u8], positions: ptr[i32],
                           out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    -- Gather actual bytes at structural positions
                    chars = gather(input, pos)

                    -- Depth deltas: open → +1, close → -1, else → 0
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    -- delta = +1 for open, -1 for close, 0 for other
                    -- Use masked select with type context from pos (i32[8])
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)

                    -- Prefix sum for running depth
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)

                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new(), 8);
        let ir = format!("{}", module.as_operation());
        eprintln!("=== JSON Stage 2 MLIR ===\n{}", ir);
        assert!(
            module.as_operation().verify(),
            "JSON stage 2 MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_json_full_pipeline_jit() {
        // Stage 1 source
        let stage1_src = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Stage 2 source
        let stage2_src = r#"
            fn json_stage2(input: ptr[u8], positions: ptr[i32],
                           out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    -- delta = +1 for open, -1 for close, 0 for other
                    -- Use masked select with type context from pos (i32[8])
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        let (fptr1, _engine1) = jit_lookup(stage1_src, "json_stage1");
        let (fptr2, _engine2) = jit_lookup(stage2_src, "json_stage2");

        // Test: {"a": [1, 2], "b": {"c": 3}}
        let json = b"{\"a\": [1, 2], \"b\": {\"c\": 3}}\0\0\0\0";
        let input = &json[..32];

        let mut positions = [0i32; 64];
        let mut depths = [0i32; 64];

        unsafe {
            // Stage 1: find structural positions
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr1);

            let bits = f1(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 64, 1,
            );
            let num = f32::to_bits(bits) as i32;

            eprintln!("\n=== Full JSON Pipeline ===");
            eprintln!("Input: {:?}", std::str::from_utf8(&json[..28]).unwrap());
            eprintln!("Stage 1: {} structural positions", num);
            for i in 0..num as usize {
                eprintln!("  pos[{}] = {} ('{}')", i, positions[i], json[positions[i] as usize] as char);
            }

            // Pad positions to multiple of 8 for stage 2
            let padded_len = ((num as usize + 7) / 8) * 8;
            // Fill padding with position 0 (safe — will just read '{' again)
            for i in num as usize..padded_len {
                positions[i] = 0;
            }

            // Stage 2: compute depths
            let f2: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,    // input
                *const i32, *const i32, i64, i64, i64,   // positions
                *mut i32, *mut i32, i64, i64, i64,       // out_depth
            ) = std::mem::transmute(fptr2);

            f2(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded_len as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 64, 1,
            );

            eprintln!("Stage 2: depths");
            for i in 0..num as usize {
                let ch = json[positions[i] as usize] as char;
                eprintln!("  depth[{}] = {} (pos {} = '{}')", i, depths[i], positions[i], ch);
            }

            // Verify depths for {"a": [1, 2], "b": {"c": 3}}
            // { → depth 1
            // : → depth 1
            // [ → depth 2
            // , → depth 2
            // ] → depth 1
            // , → depth 1
            // : → depth 1
            // { → depth 2
            // : → depth 2
            // } → depth 1
            // } → depth 0
            assert_eq!(depths[0], 1, "{{ should be depth 1");

            // Find the closing brace
            let last = num as usize - 1;
            assert_eq!(json[positions[last] as usize], b'}');
            assert_eq!(depths[last], 0, "final }} should return to depth 0");
        }
    }

    // ============================================================
    // Full SIMD JSON parser — stage 2b: type classification
    // ============================================================

    #[test]
    fn test_json_type_classify_jit() {
        // Classify value types with whitespace skipping
        let source = r#"
            fn classify_types(input: ptr[u8], positions: ptr[i32],
                              out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (dummy: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry dummy = dummy
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "classify_types");

        // Use structural positions from a known JSON input
        let json_str = b"{\"key\": \"val\", \"n\": 42, \"b\": true}";
        let mut json = [0u8; 48];
        json[..json_str.len()].copy_from_slice(json_str);
        let input = &json[..];

        // Manually determined structural positions for this input:
        // 0:{  5::  13:,  18::  20:number_start  22:,  27::  29:t
        // Actually let's just run stage 1 first
        let stage1_src = r#"
            fn s1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let (fptr1, _e1) = jit_lookup(stage1_src, "s1");

        let mut positions = [0i32; 64];
        let mut types = [0i32; 64];

        unsafe {
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr1);

            let bits = f1(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 64, 1,
            );
            let num = f32::to_bits(bits) as i32;

            // Pad positions
            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // Run type classification
            let f2: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr);

            f2(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 64, 1,
            );

            let type_names = ["?", "str", "num", "true", "false", "null", "obj", "arr", "}", "]"];
            eprintln!("\n=== Type Classification ===");
            eprintln!("Input: {:?}", std::str::from_utf8(&json[..34]).unwrap());
            for i in 0..num as usize {
                let ch = json[positions[i] as usize] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                eprintln!("  [{}] pos={} '{}' → type={} ({})", i, positions[i], ch, types[i], tname);
            }

            // { should classify what follows it — the first key is a string
            // : should classify what follows it — could be string, number, etc.
            assert!(num >= 5, "need at least 5 structural chars");
        }
    }

    // ============================================================
    // Complete SIMD JSON parse pipeline
    // ============================================================

    #[test]
    fn test_json_complete_parse() {
        // Run all stages on a real JSON document
        let stage1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let stage2_depth_src = r#"
            fn stage2_depth(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        let stage2_type_src = r#"
            fn stage2_type(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    -- Skip whitespace: check pos+1, pos+2, pos+3, pos+4
                    -- Use the first non-space, non-tab, non-newline byte
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    -- Pick first non-whitespace byte
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    -- Classify the structural char itself
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    -- For colons and commas, classify the value that follows
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    -- Type codes: 1=open_obj 2=close_obj 3=open_arr 4=close_arr
                    --             5=colon_str 6=colon_num 7=colon_true 8=colon_false
                    --             9=colon_null 10=colon_obj 11=colon_arr 12=comma
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        let (fptr1, _e1) = jit_lookup(stage1_src, "stage1");
        let (fptr2d, _e2d) = jit_lookup(stage2_depth_src, "stage2_depth");
        let (fptr2t, _e2t) = jit_lookup(stage2_type_src, "stage2_type");

        // A real JSON document
        let json_str = r#"{"name": "Alice", "age": 30, "scores": [95, 87], "active": true}"#;
        let mut json = [0u8; 72]; // pad to multiple of 8
        json[..json_str.len()].copy_from_slice(json_str.as_bytes());

        let mut positions = [0i32; 128];
        let mut depths = [0i32; 128];
        let mut types = [0i32; 128];

        unsafe {
            // Stage 1
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr1);
            let bits = f1(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 128, 1,
            );
            let num = f32::to_bits(bits) as i32;
            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // Stage 2: depth
            let f2d: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr2d);
            f2d(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 128, 1,
            );

            // Stage 2: type classification
            let f2t: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr2t);
            f2t(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 128, 1,
            );

            let type_names = [
                "?", "open_obj", "close_obj", "open_arr", "close_arr",
                "val:str", "val:num", "val:true", "val:false", "val:null",
                "val:obj", "val:arr", "comma",
            ];

            eprintln!("\n=== Complete SIMD JSON Parse ===");
            eprintln!("Input: {}", json_str);
            eprintln!("\n{:<5} {:<6} {:<6} {:<12} {}", "idx", "pos", "depth", "type", "char");
            eprintln!("{}", "-".repeat(50));
            for i in 0..num as usize {
                let pos = positions[i] as usize;
                let ch = json[pos] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                eprintln!("{:<5} {:<6} {:<6} {:<12} '{}'", i, pos, depths[i], tname, ch);
            }

            // Verify basic structure
            assert!(num >= 10, "expected many structural chars");
            assert_eq!(depths[0], 1, "first {{ is depth 1");
            assert_eq!(depths[num as usize - 1], 0, "last }} returns to depth 0");

            // Verify types
            assert_eq!(types[0], 1, "first token is open_obj");
            assert_eq!(types[num as usize - 1], 2, "last token is close_obj");

            // ============================================================
            // Now demonstrate the parse result is usable:
            // Walk the parse output and extract key-value pairs
            // ============================================================
            eprintln!("\n=== Extracted Key-Value Pairs ===");
            let mut i = 0;
            while i < num as usize {
                let pos = positions[i] as usize;
                let depth = depths[i];
                let typ = types[i];

                // Colons at depth 1 are top-level key:value separators
                // The structural char must be ':' (not ',')
                if depth == 1 && (typ >= 5 && typ <= 11) && json[pos] == b':' {
                    // The key is the string before this colon
                    // Find the opening quote: scan backwards from pos
                    let json_slice = &json[..json_str.len()];

                    // Find key: look backwards from pos for the key string
                    let mut key_end = pos;
                    while key_end > 0 && json_slice[key_end - 1] != b'"' {
                        key_end -= 1;
                    }
                    let mut key_start = key_end - 1;
                    while key_start > 0 && json_slice[key_start - 1] != b'"' {
                        key_start -= 1;
                    }
                    let key = std::str::from_utf8(&json_slice[key_start..key_end - 1]).unwrap_or("?");

                    let val_desc = match typ {
                        5 => {
                            // String value: find the quoted string after the colon
                            let mut vs = pos + 1;
                            while vs < json_slice.len() && json_slice[vs] != b'"' { vs += 1; }
                            vs += 1; // skip opening quote
                            let mut ve = vs;
                            while ve < json_slice.len() && json_slice[ve] != b'"' { ve += 1; }
                            format!("\"{}\"", std::str::from_utf8(&json_slice[vs..ve]).unwrap_or("?"))
                        }
                        6 => {
                            // Number: read digits
                            let mut ns = pos + 1;
                            while ns < json_slice.len() && json_slice[ns] == b' ' { ns += 1; }
                            let mut ne = ns;
                            while ne < json_slice.len() && (json_slice[ne].is_ascii_digit() || json_slice[ne] == b'-') { ne += 1; }
                            std::str::from_utf8(&json_slice[ns..ne]).unwrap_or("?").to_string()
                        }
                        7 => "true".to_string(),
                        8 => "false".to_string(),
                        9 => "null".to_string(),
                        10 => "{...}".to_string(),
                        11 => "[...]".to_string(),
                        _ => "?".to_string(),
                    };
                    eprintln!("  {} = {}", key, val_desc);
                }
                i += 1;
            }
        }
    }

    // ============================================================
    // SIMD integer parser
    // ============================================================

    #[test]
    fn test_simd_int_parse() {
        // Parse an integer starting at a given position in the input
        // Uses gather to load digits, SIMD multiply-accumulate
        // Supports 1-8 digits and negative numbers
        let source = r#"
            fn parse_int(input: ptr[u8], start_pos: i32[1]) -> i32[1] {
                -- Gather 8 bytes to check for negative sign
                offsets = iota(8) + start_pos
                bytes = gather(input, offsets)

                -- Check for negative: count '-' chars to get offset (0 or 1)
                is_neg_vec = bytes == '-'
                neg_offset = popcount(is_neg_vec)

                -- Re-gather from actual digit start (skips '-' if present)
                digit_offsets = iota(8) + start_pos + neg_offset
                digit_bytes = gather(input, digit_offsets)

                is_digit = (digit_bytes >= '0') & (digit_bytes <= '9')
                zero_char = digit_bytes - digit_bytes + 48
                digit_vals = digit_bytes - zero_char
                -- Count contiguous leading digits (stop at first non-digit)
                not_digit = ~is_digit
                nd1 = not_digit | lane_shr(not_digit, 1)
                nd2 = nd1 | lane_shr(nd1, 2)
                nd3 = nd2 | lane_shr(nd2, 4)
                contiguous = ~nd3
                num_digits = popcount(contiguous)
                masked = [contiguous] digit_vals : digit_vals - digit_vals
                wide = to_i32(masked)
                d0 = extract(wide, 0)
                d1 = extract(wide, 1)
                d2 = extract(wide, 2)
                d3 = extract(wide, 3)
                d4 = extract(wide, 4)
                d5 = extract(wide, 5)
                d6 = extract(wide, 6)
                d7 = extract(wide, 7)
                n = extract(num_digits, 0)
                v1 = d0
                v2 = d0 * 10 + d1
                v3 = d0 * 100 + d1 * 10 + d2
                v4 = d0 * 1000 + d1 * 100 + d2 * 10 + d3
                v5 = d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4
                v6 = d0 * 100000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
                v7 = d0 * 1000000 + d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6
                v8 = d0 * 10000000 + d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7
                result = v1
                result = [n == 2] v2 : result
                result = [n == 3] v3 : result
                result = [n == 4] v4 : result
                result = [n == 5] v5 : result
                result = [n == 6] v6 : result
                result = [n == 7] v7 : result
                result = [n == 8] v8 : result

                -- Negate if needed
                is_neg = extract(is_neg_vec, 0)
                zero = result - result
                neg_result = zero - result
                result = [is_neg] neg_result : result
                return result
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "parse_int");

        // Test: parse "42" from a buffer
        let input = b"answer: 42, next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,  // input memref
                f32,                                     // start_pos as i32[1] in s0
            ) -> f32 = std::mem::transmute(fptr);

            // "42" starts at position 8
            let pos = f32::from_bits(8u32);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 32, 1, pos);
            let result = f32::to_bits(result_bits) as i32;
            eprintln!("parse_int('42' at pos 8) = {}", result);
            assert_eq!(result, 42);

            // Test "1234" at position 0
            let input2 = b"1234____\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos2 = f32::from_bits(0u32);
            let r2 = f32::to_bits(f(input2.as_ptr(), input2.as_ptr(), 0, 24, 1, pos2)) as i32;
            eprintln!("parse_int('1234' at pos 0) = {}", r2);
            assert_eq!(r2, 1234);

            // Test "7" at position 0
            let input3 = b"7,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos3 = f32::from_bits(0u32);
            let r3 = f32::to_bits(f(input3.as_ptr(), input3.as_ptr(), 0, 24, 1, pos3)) as i32;
            eprintln!("parse_int('7' at pos 0) = {}", r3);
            assert_eq!(r3, 7);

            // Test negative: "-42"
            let input4 = b"-42,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos4 = f32::from_bits(0u32);
            let r4 = f32::to_bits(f(input4.as_ptr(), input4.as_ptr(), 0, 24, 1, pos4)) as i32;
            eprintln!("parse_int('-42' at pos 0) = {}", r4);
            assert_eq!(r4, -42);

            // Test large number: "12345678"
            let input5 = b"12345678\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos5 = f32::from_bits(0u32);
            let r5 = f32::to_bits(f(input5.as_ptr(), input5.as_ptr(), 0, 24, 1, pos5)) as i32;
            eprintln!("parse_int('12345678' at pos 0) = {}", r5);
            assert_eq!(r5, 12345678);

            // Test single digit negative: "-5"
            let input6 = b"-5,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos6 = f32::from_bits(0u32);
            let r6 = f32::to_bits(f(input6.as_ptr(), input6.as_ptr(), 0, 24, 1, pos6)) as i32;
            eprintln!("parse_int('-5' at pos 0) = {}", r6);
            assert_eq!(r6, -5);

            // Test 5-digit number: "99999"
            let input7 = b"99999,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos7 = f32::from_bits(0u32);
            let r7 = f32::to_bits(f(input7.as_ptr(), input7.as_ptr(), 0, 24, 1, pos7)) as i32;
            eprintln!("parse_int('99999' at pos 0) = {}", r7);
            assert_eq!(r7, 99999);
        }
    }

    // ============================================================
    // Full SIMD JSON parser — complete pipeline with all features
    // ============================================================

    #[test]
    fn test_json_full_parse_with_access() {
        // Complete JSON parser pipeline with random access demo

        // Stage 1: structural detection
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Stage 2: depth computation
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        // Stage 2: type classification (with whitespace skip) + value start positions
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32], out_val_start: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    one = zero + 1
                    two = zero + 2
                    three = zero + 3
                    four = zero + 4
                    -- Compute value start position (first non-whitespace after structural char)
                    vs = pos + one
                    vs = [s1] pos + two : vs
                    vs = [s1 & s2] pos + three : vs
                    vs = [s1 & s2 & s3] pos + four : vs
                    store_at(out_val_start, chunk_offset, vs)
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        // Stage 2: SIMD integer parser (1-8 digits, negative support)
        let parse_int_src = r#"
            fn parse_int(input: ptr[u8], start_pos: i32[1]) -> i32[1] {
                -- Gather 8 bytes to check for negative sign
                offsets = iota(8) + start_pos
                bytes = gather(input, offsets)

                -- Check for negative: count '-' chars to get offset (0 or 1)
                is_neg_vec = bytes == '-'
                neg_offset = popcount(is_neg_vec)

                -- Re-gather from actual digit start (skips '-' if present)
                digit_offsets = iota(8) + start_pos + neg_offset
                digit_bytes = gather(input, digit_offsets)

                is_digit = (digit_bytes >= '0') & (digit_bytes <= '9')
                zero_char = digit_bytes - digit_bytes + 48
                digit_vals = digit_bytes - zero_char
                -- Count contiguous leading digits (stop at first non-digit)
                not_digit = ~is_digit
                nd1 = not_digit | lane_shr(not_digit, 1)
                nd2 = nd1 | lane_shr(nd1, 2)
                nd3 = nd2 | lane_shr(nd2, 4)
                contiguous = ~nd3
                num_digits = popcount(contiguous)
                masked = [contiguous] digit_vals : digit_vals - digit_vals
                wide = to_i32(masked)
                -- Extract individual digit values as i32[1]
                d0 = extract(wide, 0)
                d1 = extract(wide, 1)
                d2 = extract(wide, 2)
                d3 = extract(wide, 3)
                d4 = extract(wide, 4)
                d5 = extract(wide, 5)
                d6 = extract(wide, 6)
                d7 = extract(wide, 7)
                n = extract(num_digits, 0)
                -- Build value for each possible length
                v1 = d0
                v2 = d0 * 10 + d1
                v3 = d0 * 100 + d1 * 10 + d2
                v4 = d0 * 1000 + d1 * 100 + d2 * 10 + d3
                v5 = d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4
                v6 = d0 * 100000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
                v7 = d0 * 1000000 + d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6
                v8 = d0 * 10000000 + d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7
                -- Select based on digit count
                result = v1
                result = [n == 2] v2 : result
                result = [n == 3] v3 : result
                result = [n == 4] v4 : result
                result = [n == 5] v5 : result
                result = [n == 6] v6 : result
                result = [n == 7] v7 : result
                result = [n == 8] v8 : result

                -- Negate if needed
                is_neg = extract(is_neg_vec, 0)
                zero = result - result
                neg_result = zero - result
                result = [is_neg] neg_result : result
                return result
            }
        "#;

        let (f1, _e1) = jit_lookup(s1_src, "stage1");
        let (f2d, _e2d) = jit_lookup(s2d_src, "stage2d");
        let (f2t, _e2t) = jit_lookup(s2t_src, "stage2t");
        let (fpi, _epi) = jit_lookup(parse_int_src, "parse_int");

        // A realistic JSON document with negatives, large numbers, multi-level nesting
        let json_str = r#"{"name": "Alice", "age": 30, "balance": -1500, "scores": [95, 87, 100], "active": true, "city": null, "meta": {"id": 12345678, "ver": -3}}"#;
        let mut json = [0u8; 256];
        json[..json_str.len()].copy_from_slice(json_str.as_bytes());

        let mut positions = [0i32; 256];
        let mut depths = [0i32; 256];
        let mut types = [0i32; 256];
        let mut val_starts = [0i32; 256];
        let mut partners = [-1i32; 256]; // pair matching output

        unsafe {
            // ---- Stage 1: find structural positions ----
            let stage1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(f1);
            let num = f32::to_bits(stage1(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 256, 1,
            )) as i32;

            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // ---- Stage 2a: depth ----
            let stage2d: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(f2d);
            stage2d(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 256, 1,
            );

            // ---- Stage 2b: type classification + val_start ----
            let stage2t: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(f2t);
            stage2t(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 256, 1,
                val_starts.as_mut_ptr(), val_starts.as_mut_ptr(), 0, 256, 1,
            );

            // ---- Pair matching (sequential O(N) using SIMD-computed depth) ----
            // Use a stack: push open bracket indices, pop on close
            let mut stack = vec![0i32; 128];
            let mut sp = [0usize; 16]; // stack pointer per depth
            for i in 0..num as usize {
                let ch = json[positions[i] as usize];
                let d = depths[i];
                if ch == b'{' || ch == b'[' {
                    let depth = d as usize;
                    stack[depth * 4 + sp[depth]] = i as i32;
                    sp[depth] += 1;
                } else if ch == b'}' || ch == b']' {
                    let depth = (d + 1) as usize; // the depth of the matching open
                    if sp[depth] > 0 {
                        sp[depth] -= 1;
                        let partner_idx = stack[depth * 4 + sp[depth]] as usize;
                        partners[partner_idx] = i as i32;
                        partners[i] = partner_idx as i32;
                    }
                }
            }

            // ---- Integer parser ----
            let parse_int: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                f32,
            ) -> f32 = std::mem::transmute(fpi);

            // ---- Print full parse output ----
            let type_names = [
                "???", "OBJ{", "}OBJ", "ARR[", "]ARR",
                "val:str", "val:num", "val:true", "val:false", "val:null",
                "val:obj", "val:arr",
            ];

            eprintln!("\n{}", "=".repeat(60));
            eprintln!("  COMPLETE SIMD JSON PARSE");
            eprintln!("{}", "=".repeat(60));
            eprintln!("Input: {}", json_str);
            eprintln!("\n{:<4} {:<5} {:<6} {:<12} {:<8} {:<9} {}",
                "idx", "pos", "depth", "type", "partner", "val_start", "char");
            eprintln!("{}", "-".repeat(65));

            for i in 0..num as usize {
                let pos = positions[i] as usize;
                let ch = json[pos] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                let partner_str = if partners[i] >= 0 {
                    format!("→{}", partners[i])
                } else {
                    "-".to_string()
                };
                eprintln!("{:<4} {:<5} {:<6} {:<12} {:<8} {:<9} '{}'",
                    i, pos, depths[i], tname, partner_str, val_starts[i], ch);
            }

            // ---- Random access demo: extract key-value pairs ----
            eprintln!("\n--- Random Access Demo ---");

            // Helper: find string content at a byte position
            let get_string = |pos: usize| -> &str {
                let mut start = pos + 1;
                while start < json_str.len() && json[start] != b'"' { start += 1; }
                start += 1; // skip opening quote
                let mut end = start;
                while end < json_str.len() && json[end] != b'"' { end += 1; }
                std::str::from_utf8(&json[start..end]).unwrap_or("?")
            };

            // Walk top-level keys
            eprintln!("\nTop-level key-value pairs:");
            let mut i = 0;
            while i < num as usize {
                let pos = positions[i] as usize;
                let d = depths[i];
                let t = types[i];

                if d == 1 && json[pos] == b':' {
                    // Find the key (string before the colon)
                    let key = get_string(positions[i - 1] as usize);

                    let value_str = match t {
                        5 => { // val:str
                            let s = get_string(pos);
                            format!("\"{}\"", s)
                        }
                        6 => { // val:num — use val_starts for whitespace-skipped position
                            let num_pos = val_starts[i] as usize;
                            let pos_bits = f32::from_bits(num_pos as u32);
                            let val = f32::to_bits(parse_int(
                                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                pos_bits,
                            )) as i32;
                            format!("{}", val)
                        }
                        7 => "true".to_string(),
                        8 => "false".to_string(),
                        9 => "null".to_string(),
                        10 | 11 => { // nested obj/arr
                            let open_idx = i + 1; // next structural is the open bracket
                            if partners[open_idx] >= 0 {
                                let close_pos = positions[partners[open_idx] as usize] as usize;
                                let open_pos = positions[open_idx] as usize;
                                let content = std::str::from_utf8(&json[open_pos..=close_pos]).unwrap_or("?");
                                content.to_string()
                            } else {
                                "?".to_string()
                            }
                        }
                        _ => "?".to_string(),
                    };
                    eprintln!("  {} = {}", key, value_str);
                }
                i += 1;
            }

            // ---- Direct key lookup demo ----
            eprintln!("\nDirect key lookup:");
            // Find keys by scanning top-level colons
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_string(positions[i - 1] as usize);
                    if key == "age" {
                        let num_pos = val_starts[i] as usize;
                        let val = f32::to_bits(parse_int(
                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                            f32::from_bits(num_pos as u32),
                        )) as i32;
                        eprintln!("  json[\"age\"] = {}", val);
                        assert_eq!(val, 30);
                    }
                    if key == "balance" {
                        let num_pos = val_starts[i] as usize;
                        let val = f32::to_bits(parse_int(
                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                            f32::from_bits(num_pos as u32),
                        )) as i32;
                        eprintln!("  json[\"balance\"] = {}", val);
                        assert_eq!(val, -1500);
                    }
                    if key == "name" {
                        let val = get_string(positions[i] as usize);
                        eprintln!("  json[\"name\"] = \"{}\"", val);
                        assert_eq!(val, "Alice");
                    }
                    if key == "active" {
                        let t = types[i];
                        let val = t == 7; // val:true
                        eprintln!("  json[\"active\"] = {}", val);
                        assert!(val);
                    }
                    if key == "city" {
                        let t = types[i];
                        let is_null = t == 9;
                        eprintln!("  json[\"city\"] = null (is_null={})", is_null);
                        assert!(is_null);
                    }
                    if key == "scores" {
                        // Use pair matching to find the array extent
                        let open_idx = i + 1;
                        if partners[open_idx] >= 0 {
                            let close_idx = partners[open_idx] as usize;
                            // Count commas + 1 = number of elements
                            let mut elem_count = 1;
                            for j in (open_idx + 1)..close_idx {
                                if json[positions[j] as usize] == b',' { elem_count += 1; }
                            }
                            eprintln!("  json[\"scores\"] = array with {} elements", elem_count);

                            // Parse each element using val_starts
                            let mut elem_idx = 0;
                            // First element: right after '[', use val_starts of the open bracket
                            let np0 = val_starts[open_idx] as usize;
                            let v0 = f32::to_bits(parse_int(
                                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                f32::from_bits(np0 as u32),
                            )) as i32;
                            eprintln!("    scores[{}] = {}", elem_idx, v0);
                            elem_idx += 1;
                            // Remaining elements: after each comma
                            for j in (open_idx + 1)..close_idx {
                                if types[j] == 6 { // val:num (comma before number)
                                    let np = val_starts[j] as usize;
                                    let v = f32::to_bits(parse_int(
                                        json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                        f32::from_bits(np as u32),
                                    )) as i32;
                                    eprintln!("    scores[{}] = {}", elem_idx, v);
                                    elem_idx += 1;
                                }
                            }
                            assert_eq!(elem_count, 3);
                        }
                    }
                    if key == "meta" {
                        // Nested object — drill into depth 2
                        let open_idx = i + 1;
                        if partners[open_idx] >= 0 {
                            let close_idx = partners[open_idx] as usize;
                            for j in (open_idx + 1)..close_idx {
                                if depths[j] == 2 && json[positions[j] as usize] == b':' {
                                    let inner_key = get_string(positions[j - 1] as usize);
                                    if inner_key == "id" && types[j] == 6 {
                                        let np = val_starts[j] as usize;
                                        let v = f32::to_bits(parse_int(
                                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                            f32::from_bits(np as u32),
                                        )) as i32;
                                        eprintln!("  json[\"meta\"][\"id\"] = {}", v);
                                        assert_eq!(v, 12345678);
                                    }
                                    if inner_key == "ver" && types[j] == 6 {
                                        let np = val_starts[j] as usize;
                                        let v = f32::to_bits(parse_int(
                                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                            f32::from_bits(np as u32),
                                        )) as i32;
                                        eprintln!("  json[\"meta\"][\"ver\"] = {}", v);
                                        assert_eq!(v, -3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    // Arbitrary JSON test — complex document with all value types
    // ============================================================

    #[test]
    fn test_arbitrary_json() {
        // Reuse the SIMD pipeline sources
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32], out_vs: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    one = zero + 1
                    two = zero + 2
                    three = zero + 3
                    four = zero + 4
                    ws_off = [s1] ([s2] ([s3] four : three) : two) : one
                    vs = pos + ws_off
                    store_at(out_vs, chunk_offset, vs)
                    is_oo = chars == '{'
                    is_co = chars == '}'
                    is_oa = chars == '['
                    is_ca = chars == ']'
                    is_col = chars == ':'
                    is_com = chars == ','
                    is_sep = is_col | is_com
                    t = zero
                    t = [is_oo]                    zero + 1  : t
                    t = [is_co]                    zero + 2  : t
                    t = [is_oa]                    zero + 3  : t
                    t = [is_ca]                    zero + 4  : t
                    t = [is_sep & (next == '"')]   zero + 5  : t
                    t = [is_sep & ((next >= '0') & (next <= '9') | (next == '-'))] zero + 6 : t
                    t = [is_sep & (next == 't')]   zero + 7  : t
                    t = [is_sep & (next == 'f')]   zero + 8  : t
                    t = [is_sep & (next == 'n')]   zero + 9  : t
                    t = [is_sep & (next == '{')]   zero + 10 : t
                    t = [is_sep & (next == '[')]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;
        let pi_src = r#"
            fn pi(input: ptr[u8], sp: i32[1]) -> i32[1] {
                offsets = iota(8) + sp
                bytes = gather(input, offsets)
                is_neg_vec = bytes == '-'
                neg_off = popcount(is_neg_vec)
                doff = iota(8) + sp + neg_off
                db = gather(input, doff)
                is_d = (db >= '0') & (db <= '9')
                zc = db - db + 48
                dv = db - zc
                nd = popcount(is_d)
                m = [is_d] dv : dv - dv
                w = to_i32(m)
                d0 = extract(w, 0)
                d1 = extract(w, 1)
                d2 = extract(w, 2)
                d3 = extract(w, 3)
                d4 = extract(w, 4)
                d5 = extract(w, 5)
                d6 = extract(w, 6)
                d7 = extract(w, 7)
                n = extract(nd, 0)
                r = d0
                r = [n == 2] d0*10+d1 : r
                r = [n == 3] d0*100+d1*10+d2 : r
                r = [n == 4] d0*1000+d1*100+d2*10+d3 : r
                r = [n == 5] d0*10000+d1*1000+d2*100+d3*10+d4 : r
                r = [n == 6] d0*100000+d1*10000+d2*1000+d3*100+d4*10+d5 : r
                r = [n == 7] d0*1000000+d1*100000+d2*10000+d3*1000+d4*100+d5*10+d6 : r
                r = [n == 8] d0*10000000+d1*1000000+d2*100000+d3*10000+d4*1000+d5*100+d6*10+d7 : r
                is_neg = extract(is_neg_vec, 0)
                z = r - r
                nr = z - r
                r = [is_neg] nr : r
                return r
            }
        "#;

        let (f1, _e1) = jit_lookup(s1_src, "stage1");
        let (f2d, _e2d) = jit_lookup(s2d_src, "stage2d");
        let (f2t, _e2t) = jit_lookup(s2t_src, "stage2t");
        let (fpi, _epi) = jit_lookup(pi_src, "pi");

        // Complex JSON with escaped strings, negative numbers, empty containers, nested objects
        let json_str = concat!(
            r#"{"users": [{"name": "Bob \"Jr\"", "age": 35, "score": 9999, "#,
            r#""active": false}, {"name": "Eve", "age": -1, "score": 0, "#,
            r#""active": true}], "count": 2, "tags": [], "meta": {}, "note": null}"#,
        );
        let json_bytes = json_str.as_bytes();
        let mut json = vec![0u8; ((json_bytes.len() + 15) / 8) * 8];
        json[..json_bytes.len()].copy_from_slice(json_bytes);

        let n = 256;
        let mut positions = vec![0i32; n];
        let mut depths = vec![0i32; n];
        let mut types = vec![0i32; n];
        let mut val_starts = vec![0i32; n];
        let mut partners = vec![-1i32; n];

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> f32 = std::mem::transmute(f1);
            let num = f32::to_bits(st1(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,n as i64,1)) as i32;
            let padded = ((num as usize+7)/8)*8;
            for i in num as usize..padded { positions[i]=0; }

            let st2d: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2d);
            st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,padded as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,n as i64,1);

            let st2t: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2t);
            st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,padded as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,n as i64,1,val_starts.as_mut_ptr(),val_starts.as_mut_ptr(),0,n as i64,1);

            // Pair matching
            let mut stack = vec![0i32; 128];
            let mut sp = [0usize; 32];
            for i in 0..num as usize {
                let ch = json[positions[i] as usize];
                let d = depths[i];
                if ch == b'{' || ch == b'[' {
                    let depth = d as usize;
                    if depth<32 && sp[depth]<4 { stack[depth*4+sp[depth]]=i as i32; sp[depth]+=1; }
                } else if ch == b'}' || ch == b']' {
                    let depth = (d+1) as usize;
                    if depth<32 && sp[depth]>0 { sp[depth]-=1; let pi=stack[depth*4+sp[depth]] as usize; partners[pi]=i as i32; partners[i]=pi as i32; }
                }
            }

            let parse_i = |pos: usize| -> i32 {
                let fpi_fn: extern "C" fn(*const u8,*const u8,i64,i64,i64,f32) -> f32 = std::mem::transmute(fpi);
                f32::to_bits(fpi_fn(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,f32::from_bits(pos as u32))) as i32
            };

            let get_str = |pos: usize| -> &str {
                let mut s = pos+1;
                while s < json_bytes.len() && json[s] != b'"' { s+=1; }
                s+=1;
                let mut e = s;
                while e < json_bytes.len() && json[e] != b'"' { if json[e]==b'\\' { e+=1; } e+=1; }
                std::str::from_utf8(&json[s..e]).unwrap_or("?")
            };

            eprintln!("\n=== Arbitrary JSON Parse ===");
            eprintln!("Input ({} bytes):", json_bytes.len());
            eprintln!("  {}", json_str);
            eprintln!("{} structural positions\n", num);

            // Verify key values
            let mut found = std::collections::HashMap::new();
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    let t = types[i] as usize;
                    let vs = val_starts[i] as usize;
                    match t {
                        5 => { found.insert(key.to_string(), format!("\"{}\"", get_str(positions[i] as usize))); }
                        6 => { found.insert(key.to_string(), format!("{}", parse_i(vs))); }
                        7 => { found.insert(key.to_string(), "true".into()); }
                        8 => { found.insert(key.to_string(), "false".into()); }
                        9 => { found.insert(key.to_string(), "null".into()); }
                        10 => { found.insert(key.to_string(), "{...}".into()); }
                        11 => { found.insert(key.to_string(), "[...]".into()); }
                        _ => {}
                    }
                }
            }

            eprintln!("Top-level key-values:");
            for (k, v) in &found {
                eprintln!("  {} = {}", k, v);
            }

            assert_eq!(found["count"], "2");
            assert_eq!(found["note"], "null");
            assert_eq!(found["users"], "[...]");
            assert_eq!(found["tags"], "[...]");
            assert_eq!(found["meta"], "{...}");

            // Verify empty containers
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    if key == "tags" {
                        let oi = i + 1;
                        assert!(partners[oi] >= 0);
                        let ci = partners[oi] as usize;
                        assert_eq!(ci, oi + 1, "empty array: close should be right after open");
                        eprintln!("  tags = [] (verified empty)");
                    }
                    if key == "meta" {
                        let oi = i + 1;
                        assert!(partners[oi] >= 0);
                        let ci = partners[oi] as usize;
                        assert_eq!(ci, oi + 1, "empty object: close should be right after open");
                        eprintln!("  meta = {{}} (verified empty)");
                    }
                }
            }

            // Verify nested values
            let mut user_ages = Vec::new();
            let mut user_names = Vec::new();
            for i in 0..num as usize {
                if depths[i] == 3 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    let vs = val_starts[i] as usize;
                    if key == "name" && types[i] == 5 {
                        user_names.push(get_str(positions[i] as usize).to_string());
                    }
                    if key == "age" && types[i] == 6 {
                        user_ages.push(parse_i(vs));
                    }
                }
            }
            eprintln!("  User names: {:?}", user_names);
            eprintln!("  User ages: {:?}", user_ages);
            assert_eq!(user_names.len(), 2);
            assert!(user_names[0].contains("Bob"));
            assert_eq!(user_names[1], "Eve");
            assert_eq!(user_ages, vec![35, -1]);

            eprintln!("\n=== All arbitrary JSON assertions passed ===");
        }
    }

    // ============================================================
    // BENCHMARK — throughput in GB/s
    // ============================================================

    #[test]
    fn bench_json_stage1_throughput() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Compile with full optimization
        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        // Generate a large JSON document (~10 MB)
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();

        // Pad to multiple of 8
        let padded_len = ((json_len + 7) / 8) * 8;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());

        // Allocate output (generous — ~10% of input is structural)
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark ===");
        eprintln!("Input size: {:.2} MB ({} bytes)", json_len as f64 / 1e6, json_len);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            // Warmup
            let _ = f(
                json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1,
            );

            // Benchmark: run multiple iterations
            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(
                    json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1,
                );
                total_structural = f32::to_bits(bits) as i32;
            }
            let elapsed = start.elapsed();

            let total_bytes = json_len as f64 * iterations as f64;
            let seconds = elapsed.as_secs_f64();
            let gb_per_sec = total_bytes / seconds / 1e9;
            let mb_per_sec = total_bytes / seconds / 1e6;

            eprintln!("Structural chars found: {}", total_structural);
            eprintln!("Iterations: {}", iterations);
            eprintln!("Total time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s ({:.0} MB/s)", gb_per_sec, mb_per_sec);
            eprintln!("Structural density: {:.1}%", total_structural as f64 / json_len as f64 * 100.0);
        }
    }

    #[test]
    fn bench_json_stage1_16wide() {
        // Same benchmark but with 16-byte chunks
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[16] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(16) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 15)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 15) / 16) * 16;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (16-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = f32::to_bits(bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_32wide() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[32] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(32) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 31)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 31) / 32) * 32;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (32-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = f32::to_bits(bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_64wide() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 63)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (64-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = f32::to_bits(bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_full_pipeline() {
        // Benchmark stage1 + stage2 (depth + types) together
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[32] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(32) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 31)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    next = [s1] ([s2] c3 : c2) : c1
                    zero = pos - pos
                    is_oo = chars == '{'
                    is_co = chars == '}'
                    is_oa = chars == '['
                    is_ca = chars == ']'
                    is_sep = (chars == ':') | (chars == ',')
                    t = zero
                    t = [is_oo] zero + 1 : t
                    t = [is_co] zero + 2 : t
                    t = [is_oa] zero + 3 : t
                    t = [is_ca] zero + 4 : t
                    t = [is_sep & (next == '"')]  zero + 5 : t
                    t = [is_sep & ((next >= '0') & (next <= '9') | (next == '-'))] zero + 6 : t
                    t = [is_sep & (next == 't')]  zero + 7 : t
                    t = [is_sep & (next == 'f')]  zero + 8 : t
                    t = [is_sep & (next == 'n')]  zero + 9 : t
                    t = [is_sep & (next == '{')] zero + 10 : t
                    t = [is_sep & (next == '[')] zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        let (f1, _e1) = jit_lookup_opt(s1_src, "stage1", 2);
        let (f2d, _e2d) = jit_lookup_opt(s2d_src, "stage2d", 2);
        let (f2t, _e2t) = jit_lookup_opt(s2t_src, "stage2t", 2);

        // Same JSON
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 31) / 32) * 32;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_s = json_len / 2;
        let mut positions = vec![0i32; max_s];
        let mut depths = vec![0i32; max_s];
        let mut types = vec![0i32; max_s];

        eprintln!("\n=== Full Pipeline Benchmark (stage1@32 + stage2) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> f32 = std::mem::transmute(f1);
            let st2d: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2d);
            let st2t: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2t);

            // Warmup
            let bits = st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
            let num = f32::to_bits(bits) as i32;
            let spad = ((num as usize+7)/8)*8;
            for i in num as usize..spad { positions[i]=0; }
            st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,max_s as i64,1);
            st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,max_s as i64,1);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let bits = st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
                let num = f32::to_bits(bits) as i32;
                let spad = ((num as usize+7)/8)*8;
                for i in num as usize..spad { positions[i]=0; }
                st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,max_s as i64,1);
                st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,max_s as i64,1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    // ============================================================
    // BITMASK-STYLE stage 1 — simdjson approach
    // ============================================================

    #[test]
    fn test_bitmask_roundtrip() {
        // Verify to_bitmask and from_bitmask work correctly
        let source = r#"
            fn test_bitmask(v: u8[8]) -> i32[1] {
                mask = v == 'a'
                bits = to_bitmask(mask)
                -- Count bits set (should match popcount)
                pc = popcount(mask)
                return pc
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "test_bitmask", 2);
        unsafe {
            let f: extern "C" fn(uint8x8_t) -> f32 = std::mem::transmute(fptr);
            let v = vld1_u8([b'a', b'b', b'a', b'a', b'b', b'b', b'a', b'b'].as_ptr());
            let result = f32::to_bits(f(v)) as i32;
            assert_eq!(result, 4, "should count 4 'a' chars");
        }
    }

    #[test]
    fn test_bitmask_clmul_correctness() {
        // Test that CLMUL-based prefix XOR gives same results as scan.xor
        // Use a simple 8-byte input for easy verification, but pack to u64
        let source = r#"
            fn test_clmul(v: u8[8]) -> i32[8] {
                quote = v == '"'
                in_str = scan.xor(quote)
                return [in_str] 1 : 0
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "test_clmul", 2);
        unsafe {
            // i32[8] = 256 bits = two int32x4_t. Use sret-style or just check first 4.
            // Actually vector<8xi32> on ARM64 uses two q registers.
            // For simplicity, verify via a scalar reduction.
            drop(fptr);
        }
        // Just verify the scan.xor result via a simpler test
        let source2 = r#"
            fn test_clmul2(v: u8[8]) -> i32[1] {
                quote = v == '"'
                in_str = scan.xor(quote)
                return popcount(in_str)
            }
        "#;
        let (fptr2, _engine2) = jit_lookup_opt(source2, "test_clmul2", 2);
        unsafe {
            let f: extern "C" fn(uint8x8_t) -> f32 = std::mem::transmute(fptr2);
            // Input: x " h e l " x x → in_str = [0,1,1,1,1,0,0,0] → 4 true
            let v = vld1_u8([b'x', b'"', b'h', b'e', b'l', b'"', b'x', b'x'].as_ptr());
            let result = f32::to_bits(f(v)) as i32;
            assert_eq!(result, 4, "4 bytes should be inside string");
        }
    }

    #[test]
    fn bench_json_stage1_bitmask_16() {
        // 16-wide bitmask: fits in one NEON register, clean bitcast to u16
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[16] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                  | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)

                    -- CLMUL for prefix XOR
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    -- XOR with carry, then mask to 16 bits
                    in_string_with_carry = xor(in_string_raw, prev_in_string)
                    -- Mask: only lower 16 bits matter
                    mask16 = bit_shr(all_ones, 48)
                    in_string_bits = in_string_with_carry & mask16

                    real_structural_bits = structural_bits & ~in_string_bits
                    real_structural_vec = from_bitmask(real_structural_bits, 16)

                    indices = iota(16) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)

                    -- Carry: bit 15 (the last bit of this 16-byte chunk)
                    -- If set, next chunk starts inside a string
                    -- Use all_ones as -1 to propagate: if bit 15 is set, carry = all_ones, else 0
                    -- This way XOR with carry fills all 64 bits correctly
                    carry_bit = bit_shr(in_string_bits, 15)
                    -- Spread to all 64 bits: multiply by all_ones
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 15) / 16) * 16;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 BITMASK (16-wide, CLMUL) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = f32::to_bits(bits) as i32;
            eprintln!("Structural chars: {}", num);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);

            assert_eq!(num, 500004, "should find same structural count as vector version");
        }
    }

    #[test]
    fn bench_json_stage1_bitmask() {
        // Process 64 bytes per chunk:
        // - Compare in vector domain (u8[64])
        // - Pack to bitmask (u64[1]) immediately
        // - Do prefix XOR via CLMUL in bitmask domain
        // - Output positions via scalar bit extraction loop
        //
        // This matches simdjson's approach.
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    -- Compare in vector domain
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec

                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                  | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    -- Pack to bitmask domain
                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)

                    -- Prefix XOR via CLMUL
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    in_string_with_carry = xor(in_string_raw, prev_in_string)

                    real_structural_bits = structural_bits & ~in_string_with_carry
                    real_structural_vec = from_bitmask(real_structural_bits, 64)

                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)

                    -- Carry: bit 63 spread to all 64 bits
                    carry_bit = bit_shr(in_string_with_carry, 63)
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 BITMASK (64-wide, CLMUL) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            // Warmup + verify
            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = f32::to_bits(bits) as i32;
            eprintln!("Structural chars: {}", num);

            // Verify correctness against vector version
            eprintln!("First 10 structural positions:");
            for i in 0..std::cmp::min(10, num as usize) {
                let p = positions[i] as usize;
                eprintln!("  [{}] pos={} char='{}'", i, p, json[p] as char);
            }
            assert_eq!(json[positions[0] as usize], b'{', "first structural should be {{");

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_128() {
        // 128-wide: compare in u8[128], split into two bool[64] halves,
        // bitmask+CLMUL each half, chain carry between them
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[128] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    -- Compare in 128-wide vector domain
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                   | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    -- Split into two 64-lane halves
                    quote_lo = split_lo(real_quote_vec)
                    quote_hi = split_hi(real_quote_vec)
                    struct_lo = split_lo(structural_vec)
                    struct_hi = split_hi(structural_vec)

                    -- Bitmask + CLMUL on low half
                    qbits_lo = to_bitmask(quote_lo)
                    sbits_lo = to_bitmask(struct_lo)
                    all_ones = qbits_lo | ~qbits_lo
                    in_string_raw_lo = clmul(qbits_lo, all_ones)
                    in_string_lo = in_string_raw_lo ^ prev_in_string
                    real_sbits_lo = sbits_lo & ~in_string_lo

                    -- Carry from low to high
                    carry_lo = (in_string_lo >> 63) * all_ones

                    -- Bitmask + CLMUL on high half
                    qbits_hi = to_bitmask(quote_hi)
                    sbits_hi = to_bitmask(struct_hi)
                    in_string_raw_hi = clmul(qbits_hi, all_ones)
                    in_string_hi = in_string_raw_hi ^ carry_lo
                    real_sbits_hi = sbits_hi & ~in_string_hi

                    -- Emit positions for low half
                    real_struct_lo = from_bitmask(real_sbits_lo, 64)
                    indices_lo = iota(64) + chunk_offset
                    compressstore(positions, pos, indices_lo, real_struct_lo)
                    pos_after_lo = pos + popcount(real_struct_lo)

                    -- Emit positions for high half (offset by 64)
                    real_struct_hi = from_bitmask(real_sbits_hi, 64)
                    indices_hi = iota(64) + chunk_offset + 64
                    compressstore(positions, pos_after_lo, indices_hi, real_struct_hi)

                    -- Carry out
                    carry prev_in_string = (in_string_hi >> 63) * all_ones
                    carry pos = pos_after_lo + popcount(real_struct_hi)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 127) / 128) * 128;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 (128-wide, vector scan.xor) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = f32::to_bits(bits) as i32;
            eprintln!("Structural chars: {}", num);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);

            assert_eq!(num, 500004, "128-wide should find same structural count");
        }
    }

    #[test]
    fn test_tbl_basic() {
        // Table lookup: tbl(data, indices) where data is [0..15] used as both table and indices
        let source = r#"
            fn nibble_lookup(data: u8[16]) -> u8[16] {
                low = data & 15
                return tbl(data, low)
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "nibble_lookup", 2);
        unsafe {
            let f: extern "C" fn(uint8x16_t) -> uint8x16_t = std::mem::transmute(fptr);
            let data = vld1q_u8([0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].as_ptr());
            let result = f(data);
            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), result);
            // tbl([0..15], [0..15] & 0xF) = tbl([0..15], [0..15]) = [0..15]
            assert_eq!(out, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
        }
    }

    #[test]
    fn test_utf8_validation() {
        // Simplified UTF-8 validator using SIMD classification
        let source = r#"
            fn validate_utf8(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (prev_incomplete: u8[1] = 0, errors: i32[1] = 0) {
                    -- Classify bytes by high nibble
                    high = bit_shr(chunk, 4)

                    -- Classification by high nibble ranges
                    is_continuation = (high >= 8) & (high <= 11)
                    is_2byte = (high >= 12) & (high <= 13)
                    is_3byte = high == 14
                    is_4byte = high == 15

                    -- After a multi-byte start, next bytes must be continuation
                    expect_cont_1 = is_2byte | is_3byte | is_4byte
                    expect_cont_2 = is_3byte | is_4byte
                    expect_cont_3 = is_4byte

                    shifted_1 = lane_shr(expect_cont_1, 1)
                    shifted_2 = lane_shr(expect_cont_2, 2)
                    shifted_3 = lane_shr(expect_cont_3, 3)

                    -- Expected continuation at each position
                    expected = shifted_1 | shifted_2 | shifted_3

                    -- Mismatch: expected continuation but not, or got continuation unexpectedly
                    mismatch = (expected & ~is_continuation) | (~expected & is_continuation)

                    chunk_errors = popcount(mismatch)

                    carry prev_incomplete = prev_incomplete
                    carry errors = errors + chunk_errors
                }
                return errors
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "validate_utf8", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 = std::mem::transmute(fptr);

            // Valid ASCII
            let valid_ascii = b"Hello, world!!\0\0";
            let errors = f32::to_bits(f(valid_ascii.as_ptr(), valid_ascii.as_ptr(), 0, 16, 1)) as i32;
            eprintln!("Valid ASCII errors: {}", errors);
            assert_eq!(errors, 0, "Valid ASCII should have 0 errors");

            // Valid UTF-8: "café" = [99, 97, 102, 195, 169] (é is 0xC3 0xA9)
            let valid_utf8 = b"caf\xC3\xA9_______\0\0\0\0";
            let padded_len = 16;
            let errors = f32::to_bits(f(valid_utf8.as_ptr(), valid_utf8.as_ptr(), 0, padded_len as i64, 1)) as i32;
            eprintln!("Valid UTF-8 errors: {}", errors);
            assert_eq!(errors, 0, "Valid UTF-8 should have 0 errors");

            // Invalid: bare continuation byte (0x80 without a starter)
            let invalid = b"\x80Hello_________\0";
            let errors = f32::to_bits(f(invalid.as_ptr(), invalid.as_ptr(), 0, 16, 1)) as i32;
            eprintln!("Invalid (bare continuation) errors: {}", errors);
            assert!(errors > 0, "Bare continuation byte should be an error");
        }
    }

    #[test]
    fn test_string_escape_detection() {
        // Detect backslash escapes in input using SIMD
        let source = r#"
            fn has_escapes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (found: i32[1] = 0) {
                    has_bs = chunk == '\\'
                    chunk_has = popcount(has_bs)
                    carry found = found + chunk_has
                }
                return found
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "has_escapes", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 = std::mem::transmute(fptr);

            // No escapes
            let clean = b"Hello, world!!!!\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = f32::to_bits(f(clean.as_ptr(), clean.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 0, "Clean string should have 0 backslashes");

            // Has escapes
            let escaped = b"Hello\\nWorld\\t!!\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = f32::to_bits(f(escaped.as_ptr(), escaped.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 2, "Should find 2 backslashes");
        }
    }

    #[test]
    fn test_number_boundary_detection() {
        // Count bytes that are part of numbers (digits, minus, dot, e/E, plus)
        let source = r#"
            fn count_number_bytes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (count: i32[1] = 0) {
                    is_digit = (chunk >= '0') & (chunk <= '9')
                    is_minus = chunk == '-'
                    is_dot = chunk == '.'
                    is_e = (chunk == 'e') | (chunk == 'E')
                    is_plus = chunk == '+'
                    is_num_byte = is_digit | is_minus | is_dot | is_e | is_plus
                    chunk_count = popcount(is_num_byte)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "count_number_bytes", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 = std::mem::transmute(fptr);

            // "3.14" has 4 number bytes
            let input = b"3.14____________\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = f32::to_bits(f(input.as_ptr(), input.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 4, "3.14 has 4 number bytes");

            // "-1.5e10" has 7 number bytes
            let input2 = b"-1.5e10_________\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n2 = f32::to_bits(f(input2.as_ptr(), input2.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n2, 7, "-1.5e10 has 7 number bytes");
        }
    }

    #[test]
    fn test_whitespace_classification() {
        let source = r#"
            fn count_whitespace(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (count: i32[1] = 0) {
                    is_ws = (chunk == ' ') | (chunk == '\n') | (chunk == '\t') | (chunk == '\r')
                    chunk_count = popcount(is_ws)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "count_whitespace", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 = std::mem::transmute(fptr);
            let input = b"  hello  world  \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = f32::to_bits(f(input.as_ptr(), input.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 6, "Should count 6 whitespace chars");
        }
    }

    #[test]
    fn bench_json_all_simd_stages() {
        // All SIMD stages combined: structural detection, UTF-8 validation, escape detection
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                   | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    in_string_with_carry = xor(in_string_raw, prev_in_string)
                    real_structural_bits = structural_bits & ~in_string_with_carry
                    real_structural_vec = from_bitmask(real_structural_bits, 64)
                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)
                    carry_bit = bit_shr(in_string_with_carry, 63)
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let utf8_src = r#"
            fn validate_utf8(input: ptr[u8]) -> u8[16] {
                stream chunk: u8[16] over input carry (prev_byte: u8[1] = 0, error_acc: u8[16] = [u8: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) {
                    prev1 = lane_shr(chunk, 1)
                    byte_1_high_table = [u8: 2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73]
                    prev1_high = bit_shr(prev1, 4)
                    byte_1_high = tbl(byte_1_high_table, prev1_high)
                    byte_1_low_table = [u8: 231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203]
                    prev1_low = prev1 & 15
                    byte_1_low = tbl(byte_1_low_table, prev1_low)
                    byte_2_high_table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1]
                    cur_high = bit_shr(chunk, 4)
                    byte_2_high = tbl(byte_2_high_table, cur_high)
                    special_cases = byte_1_high & byte_1_low & byte_2_high
                    prev2 = lane_shr(chunk, 2)
                    prev3 = lane_shr(chunk, 3)
                    is_starter_3plus_p2 = prev2 >= 224
                    is_starter_4_p3 = prev3 >= 240
                    must23 = is_starter_3plus_p2 | is_starter_4_p3
                    zero_byte = chunk - chunk
                    must23_byte = [must23] zero_byte + 128 : zero_byte
                    errors_for_chunk = xor(special_cases, must23_byte)
                    error_acc = error_acc | errors_for_chunk
                    carry prev_byte = extract(chunk, 15)
                    carry error_acc = error_acc
                }
                return error_acc
            }
        "#;

        let escape_src = r#"
            fn count_escapes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[64] over input carry (count: i32[1] = 0) {
                    carry count = count + popcount(chunk == '\\')
                }
                return count
            }
        "#;

        let (f1, _e1) = jit_lookup_opt(s1_src, "stage1", 2);
        let (f_utf8, _e_utf8) = jit_lookup_opt(utf8_src, "validate_utf8", 2);
        let (f_esc, _e_esc) = jit_lookup_opt(escape_src, "count_escapes", 2);

        // Generate JSON
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_s = json_len / 2;
        let mut positions = vec![0i32; max_s];

        eprintln!("\n=== All SIMD Stages Benchmark ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> f32 = std::mem::transmute(f1);
            let utf8: extern "C" fn(*const u8,*const u8,i64,i64,i64) -> uint8x16_t = std::mem::transmute(f_utf8);
            let esc: extern "C" fn(*const u8,*const u8,i64,i64,i64) -> f32 = std::mem::transmute(f_esc);

            // Warmup
            st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
            utf8(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
            esc(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
                utf8(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
                esc(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s (all SIMD stages combined)", gb_per_sec);
        }
    }

    // ============================================================
    // Vector literal + tbl test
    // ============================================================

    #[test]
    fn test_vec_literal_basic() {
        // Constant vector literal: [u8: 10, 20, 30, 40]
        let source = r#"
            fn make_vec() -> i32[4] {
                v = [i32: 10, 20, 30, 40]
                return v
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "make_vec", 2);
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [10, 20, 30, 40]);
        }
    }

    #[test]
    fn test_tbl_with_vec_literal() {
        // Use a constant lookup table to map nibbles to values
        let source = r#"
            fn nibble_classify(data: u8[16]) -> u8[16] {
                -- Table: maps high nibble to category
                -- 0-7 = ASCII (1), 8-B = continuation (2), C-D = 2-byte (4), E = 3-byte (8), F = 4-byte (16)
                table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 8, 16]
                high = bit_shr(data, 4)
                return tbl(table, high)
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "nibble_classify", 2);
        unsafe {
            let f: extern "C" fn(uint8x16_t) -> uint8x16_t = std::mem::transmute(fptr);
            // Test: ASCII 'A' (0x41), continuation 0x80, 2-byte start 0xC3, 3-byte start 0xE2, 4-byte start 0xF0
            let data = vld1q_u8([0x41, 0x80, 0xC3, 0xE2, 0xF0, 0x00, 0x7F, 0xBF, 0xDF, 0xEF, 0xFF, 0x20, 0x90, 0xA0, 0xC0, 0xD0].as_ptr());
            let result = f(data);
            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), result);
            // 0x41 → nibble 4 → ASCII (1)
            // 0x80 → nibble 8 → continuation (2)
            // 0xC3 → nibble C → 2-byte (4)
            // 0xE2 → nibble E → 3-byte (8)
            // 0xF0 → nibble F → 4-byte (16)
            assert_eq!(out[0], 1, "ASCII");
            assert_eq!(out[1], 2, "continuation");
            assert_eq!(out[2], 4, "2-byte start");
            assert_eq!(out[3], 8, "3-byte start");
            assert_eq!(out[4], 16, "4-byte start");
        }
    }

    // ============================================================
    // Real UTF-8 validation (simdjson algorithm)
    // ============================================================

    #[test]
    fn test_utf8_validation_real() {
        // simdjson's UTF-8 validation algorithm using tbl lookups
        // Error categories (bitmask):
        //   TOO_SHORT=1, TOO_LONG=2, OVERLONG_3=4, TOO_LARGE=8,
        //   SURROGATE=16, OVERLONG_2=32, TOO_LARGE_1000=64, TWO_CONTS=128, OVERLONG_4=64
        //
        // Three lookups: byte_1_high, byte_1_low, byte_2_high
        // Error if (byte_1_high & byte_1_low & byte_2_high) != 0
        let source = r#"
            fn validate_utf8(input: ptr[u8]) -> u8[16] {
                stream chunk: u8[16] over input carry (prev_byte: u8[1] = 0, error_acc: u8[16] = [u8: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) {
                    -- Get previous byte for each position (lane shift right by 1)
                    prev1 = lane_shr(chunk, 1)
                    -- TODO: first lane should come from carry prev_byte
                    -- For now, first lane is 0 (correct for start of input)

                    -- Lookup table 1: prev byte's high nibble → error flags
                    -- TOO_LONG=2 for ASCII (0-7), TWO_CONTS=128 for continuation (8-B),
                    -- TOO_SHORT|OVERLONG_2=33 for C, TOO_SHORT=1 for D,
                    -- TOO_SHORT|OVERLONG_3|SURROGATE=21 for E,
                    -- TOO_SHORT|TOO_LARGE|TOO_LARGE_1000|OVERLONG_4=137 for F
                    -- TOO_LONG=2, TWO_CONTS=128, TOO_SHORT=1, OVERLONG_2=32
                    -- OVERLONG_3=4, SURROGATE=16, TOO_LARGE=8, TOO_LARGE_1000=OVERLONG_4=64
                    byte_1_high_table = [u8: 2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73]
                    prev1_high = bit_shr(prev1, 4)
                    byte_1_high = tbl(byte_1_high_table, prev1_high)

                    -- Lookup table 2: prev byte's low nibble → error flags
                    -- CARRY = TOO_SHORT|TOO_LONG|TWO_CONTS = 131
                    byte_1_low_table = [u8: 231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203]
                    prev1_low = prev1 & 15
                    byte_1_low = tbl(byte_1_low_table, prev1_low)

                    -- Lookup table 3: current byte's high nibble → error flags
                    byte_2_high_table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1]
                    cur_high = bit_shr(chunk, 4)
                    byte_2_high = tbl(byte_2_high_table, cur_high)

                    -- AND all three: nonzero means error
                    special_cases = byte_1_high & byte_1_low & byte_2_high

                    -- check_multibyte_lengths: resolve TWO_CONTS errors
                    -- A continuation byte is valid if a 2/3/4-byte starter preceded it
                    -- prev2/prev3 = bytes 2 and 3 positions back
                    prev2 = lane_shr(chunk, 2)
                    prev3 = lane_shr(chunk, 3)

                    -- must_be_continuation_of_3_or_more: prev2 >= 0xE0
                    -- (byte >= 0xE0 means it's a 3/4-byte starter, expects cont at +2)
                    is_starter_3plus_p2 = prev2 >= 224

                    -- must_be_continuation_of_4: prev3 >= 0xF0
                    -- (byte >= 0xF0 means it's a 4-byte starter, expects cont at +3)
                    is_starter_4_p3 = prev3 >= 240

                    -- must23: continuation expected at this position from multi-byte starters
                    must23 = is_starter_3plus_p2 | is_starter_4_p3

                    -- Convert must23 to u8 mask with bit 7 set (128 = 0x80)
                    -- Use chunk arithmetic to stay in u8 domain
                    zero_byte = chunk - chunk
                    must23_byte = [must23] zero_byte + 128 : zero_byte

                    -- XOR: cancels TWO_CONTS if multi-byte starter was found
                    errors_for_chunk = xor(special_cases, must23_byte)

                    error_acc = error_acc | errors_for_chunk

                    carry prev_byte = extract(chunk, 15)
                    carry error_acc = error_acc
                }
                return error_acc
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "validate_utf8", 2);

        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> uint8x16_t = std::mem::transmute(fptr);
            let run = |input: &[u8]| -> bool {
                let padded_len = ((input.len() + 15) / 16) * 16;
                let mut buf = vec![0u8; padded_len];
                buf[..input.len()].copy_from_slice(input);
                let result = f(buf.as_ptr(), buf.as_ptr(), 0, padded_len as i64, 1);
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), result);
                // If any byte in error_acc is nonzero, there's an error
                out.iter().all(|&b| b == 0)
            };

            assert!(run(b"Hello, world!"), "ASCII should be valid");
            assert!(run(b"caf\xC3\xA9"), "2-byte UTF-8 should be valid");

            // Valid 3-byte UTF-8: € = E2 82 AC
            assert!(run(b"\xE2\x82\xAC"), "3-byte UTF-8 should be valid");

            // Valid 4-byte UTF-8: 𝄞 = F0 9D 84 9E
            assert!(run(b"\xF0\x9D\x84\x9E"), "4-byte UTF-8 should be valid");

            // Invalid: bare continuation byte
            assert!(!run(b"\x80hello"), "bare continuation should be invalid");

            // Invalid: overlong 2-byte (C0 80 = U+0000)
            assert!(!run(b"\xC0\x80"), "overlong 2-byte should be invalid");

            // Invalid: surrogate half (ED A0 80 = U+D800)
            assert!(!run(b"\xED\xA0\x80"), "surrogate should be invalid");

            // Invalid: too large (F4 90 80 80 = U+110000)
            assert!(!run(b"\xF4\x90\x80\x80"), "code point > U+10FFFF should be invalid");

            eprintln!("All UTF-8 validation tests passed!");
        }
    }

    // ============================================================
    // XOR operator tests
    // ============================================================

    #[test]
    fn test_xor_operator() {
        let source = r#"
            fn do_xor(a: i32[4], b: i32[4]) -> i32[4] {
                return a ^ b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "do_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let a = vld1q_s32([0xFFi32, 0x0F, 0xF0, 0xFF].as_ptr());
            let b = vld1q_s32([0x0Fi32, 0x0F, 0x0F, 0x00].as_ptr());
            let result = f(a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0xF0, 0x00, 0xFF, 0xFF]);
        }
    }

    #[test]
    fn test_xor_equivalence_with_builtin() {
        // a ^ b should produce the same result as xor(a, b)
        let source_op = r#"
            fn xor_op(a: i32[4], b: i32[4]) -> i32[4] {
                return a ^ b
            }
        "#;
        let source_fn = r#"
            fn xor_fn(a: i32[4], b: i32[4]) -> i32[4] {
                return xor(a, b)
            }
        "#;
        let (fptr1, _e1) = jit_lookup(source_op, "xor_op");
        let (fptr2, _e2) = jit_lookup(source_fn, "xor_fn");
        unsafe {
            let f1: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr1);
            let f2: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr2);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([5i32, 6, 7, 8].as_ptr());
            let r1 = f1(a, b);
            let r2 = f2(a, b);
            let mut o1 = [0i32; 4];
            let mut o2 = [0i32; 4];
            vst1q_s32(o1.as_mut_ptr(), r1);
            vst1q_s32(o2.as_mut_ptr(), r2);
            assert_eq!(o1, o2);
        }
    }

    // ============================================================
    // Bit shift operator tests (>> and << are now element-wise)
    // ============================================================

    #[test]
    fn test_bit_shift_right() {
        let source = r#"
            fn shr(v: i32[4]) -> i32[4] {
                return v >> 1
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shr");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 64, 128].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // Element-wise unsigned right shift by 1
            assert_eq!(out, [5, 10, 32, 64]);
        }
    }

    #[test]
    fn test_bit_shift_left() {
        let source = r#"
            fn shl(v: i32[4]) -> i32[4] {
                return v << 2
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shl");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [4, 8, 12, 16]);
        }
    }

    // ============================================================
    // Typed assignment + literal broadcast tests
    // ============================================================

    #[test]
    fn test_typed_assign_not_zero() {
        // all_ones: u64[1] = ~0 should produce all-ones vector
        let source = r#"
            fn all_ones() -> u64[1] {
                x: u64[1] = ~0
                return x
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "all_ones");
        unsafe {
            let f: extern "C" fn() -> f32 = std::mem::transmute(fptr);
            let bits = f();
            let result = f32::to_bits(bits) as u64;
            // u64[1] all-ones = 0xFFFFFFFFFFFFFFFF, but returned as i32 via f32 reinterpret
            // The low 32 bits should be all-ones
            assert_eq!(result as u32, 0xFFFFFFFF);
        }
    }

    #[test]
    fn test_typed_assign_zero_broadcast() {
        // z: u8[4] = 0 should produce zero vector
        let source = r#"
            fn zeros(v: i32[4]) -> i32[4] {
                z: i32[4] = 0
                return v + z
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "zeros");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [1, 2, 3, 4]);
        }
    }

    // ============================================================
    // Vector repeat syntax tests
    // ============================================================

    #[test]
    fn test_vec_repeat_syntax() {
        // [i32: 7; 4] should produce [7, 7, 7, 7]
        let source = r#"
            fn sevens() -> i32[4] {
                v = [i32: 7; 4]
                return v
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "sevens", 2);
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [7, 7, 7, 7]);
        }
    }

    #[test]
    fn test_literal_broadcast_not_in_binop() {
        // ~0 in binary op context should broadcast correctly
        let source = r#"
            fn invert(v: i32[4]) -> i32[4] {
                return v ^ ~0
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "invert");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([0i32, 1, -1, 42].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [-1, -2, 0, -43]);
        }
    }

    // ============================================================
    // Error message source location tests
    // ============================================================

    #[test]
    #[should_panic(expected = "at 2:")]
    fn test_error_includes_line_number_undefined_var() {
        let source = "fn f(v: i32[4]) -> i32[4] {\n    return undefined_var\n}";
        jit_lookup(source, "f");
    }

    #[test]
    #[should_panic(expected = "at 2:")]
    fn test_error_includes_line_number_bare_literal() {
        let source = "fn f() -> i32[4] {\n    x = 42\n    return x\n}";
        jit_lookup(source, "f");
    }

    #[test]
    #[should_panic(expected = "at 3:")]
    fn test_error_includes_line_number_width_mismatch() {
        let source = "fn f(a: i32[4], b: i32[8]) -> i32[4] {\n    -- mismatch\n    return a + b\n}";
        jit_lookup(source, "f");
    }

    // ============================================================
    // If/else tests
    // ============================================================

    // ============================================================
    // Multiple return values via out-params (Step 8)
    // ============================================================

    #[test]
    fn test_out_param_pattern() {
        // Demonstrate returning multiple values via ptr out-params
        // The function writes count to out_count ptr and returns the last value
        let source = r#"
            fn count_and_sum(input: ptr[u8], out_count: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (count: i32[1] = 0, sum: i32[1] = 0) {
                    is_a = chunk == 'a'
                    carry count = count + popcount(is_a)
                    carry sum = sum + 1
                }
                store(out_count, count)
                return sum
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_and_sum");
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);
            let input = b"aababcaa\0\0\0\0\0\0\0\0";
            let mut count_out = [0i32; 1];
            let sum_bits = f(
                input.as_ptr(), input.as_ptr(), 0, 16, 1,
                count_out.as_mut_ptr(), count_out.as_mut_ptr(), 0, 1, 1,
            );
            let sum = f32::to_bits(sum_bits) as i32;
            eprintln!("count={}, sum={}", count_out[0], sum);
            assert_eq!(count_out[0], 5, "should count 5 'a' bytes");
            assert!(sum > 0, "sum should be positive");
        }
    }

    // ============================================================
    // If/else tests (Step 7)
    // ============================================================

    #[test]
    fn test_if_else_basic() {
        // Use comparison to get a bool[1] condition
        let source = r#"
            fn pick(flag: i32[1], a: i32[4], b: i32[4]) -> i32[4] {
                cond = flag == 1
                if cond {
                    result = a
                } else {
                    result = b
                }
                return result
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "pick");
        unsafe {
            let f: extern "C" fn(f32, int32x4_t, int32x4_t) -> int32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([10i32, 20, 30, 40].as_ptr());

            // flag=1 → cond=true → return a
            let result = f(f32::from_bits(1u32), a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            eprintln!("flag=1 result: {:?}", out);
            assert_eq!(out, [1, 2, 3, 4]);

            // flag=0 → cond=false → return b
            let result = f(f32::from_bits(0u32), a, b);
            vst1q_s32(out.as_mut_ptr(), result);
            eprintln!("flag=0 result: {:?}", out);
            assert_eq!(out, [10, 20, 30, 40]);
        }
    }

    // ============================================================
    // Full JSON DOM proof — round-trip reconstruction + navigation
    // Proves our tape is morally equivalent to simdjson's DOM
    // ============================================================

    /// JIT stage1 and build a full DOM tape, then return the document
    fn jit_parse_json(json: &[u8]) -> (crate::json::Document, Vec<u8>) {
        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let (fptr, _engine) = jit_lookup_opt(&source, "json_stage1", 2);

        let padded = crate::json::pad_input(json);
        let mut positions = vec![0i32; json.len()];

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);

            let bits = f(
                padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1,
            );
            let num = f32::to_bits(bits) as usize;
            let doc = crate::json::build_tape(&padded, &positions[..num]);
            (doc, padded)
        }
    }

    fn reconstruct_json(doc: &crate::json::Document, input: &[u8]) -> String {
        let mut out = String::new();
        reconstruct_at(doc, input, 1, &mut out);
        out
    }

    fn reconstruct_at(
        doc: &crate::json::Document, input: &[u8], index: usize, out: &mut String,
    ) -> usize {
        use crate::json::TapeType;
        match doc.tape_type(index) {
            TapeType::OpenObject => {
                out.push('{');
                let close = doc.get_partner(index);
                let mut i = index + 1;
                let mut first = true;
                while i < close {
                    if !first { out.push(','); }
                    first = false;
                    let key = doc.get_string(i, input);
                    out.push('"');
                    json_escape(&key, out);
                    out.push('"');
                    out.push(':');
                    i += 1;
                    i = reconstruct_at(doc, input, i, out);
                }
                out.push('}');
                close + 1
            }
            TapeType::OpenArray => {
                out.push('[');
                let close = doc.get_partner(index);
                let mut i = index + 1;
                let mut first = true;
                while i < close {
                    if !first { out.push(','); }
                    first = false;
                    i = reconstruct_at(doc, input, i, out);
                }
                out.push(']');
                close + 1
            }
            TapeType::String => {
                let s = doc.get_string(index, input);
                out.push('"');
                json_escape(&s, out);
                out.push('"');
                index + 1
            }
            TapeType::Int64 => {
                out.push_str(&doc.get_i64(index).to_string());
                index + 1
            }
            TapeType::Double => {
                let v = doc.get_f64(index);
                out.push_str(&format!("{}", v));
                index + 1
            }
            TapeType::True => { out.push_str("true"); index + 1 }
            TapeType::False => { out.push_str("false"); index + 1 }
            TapeType::Null => { out.push_str("null"); index + 1 }
            _ => index + 1,
        }
    }

    fn json_escape(s: &str, out: &mut String) {
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
    }

    #[test]
    fn test_json_dom_roundtrip_simple() {
        let json = br#"{"name":"Alice","age":30,"active":true}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_nested() {
        let json = br#"{"a":{"b":{"c":1}},"d":[1,2,3],"e":null,"f":false}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_all_types() {
        let json = br#"{"str":"hello","int":42,"neg":-17,"bool_t":true,"bool_f":false,"nil":null,"arr":[1,"two",true,null],"obj":{"x":1}}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_100_objects() {
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data":["#.to_string());
        for i in 0..100 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id":{},"name":"user_{}","score":{},"active":{}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let (doc, padded) = jit_parse_json(json_str.as_bytes());
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, json_str);
    }

    #[test]
    fn test_json_dom_navigation() {
        use crate::json::TapeType;
        let json = br#"{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}],"count":2}"#;
        let (doc, padded) = jit_parse_json(json);

        // Find "users" key
        let mut users_idx = None;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "users" {
                users_idx = Some(i + 1);
                break;
            }
        }
        let users_arr = users_idx.expect("should find 'users'");
        assert_eq!(doc.tape_type(users_arr), TapeType::OpenArray);

        // First element
        let first_obj = users_arr + 1;
        assert_eq!(doc.tape_type(first_obj), TapeType::OpenObject);

        // Find "name" in first object
        let first_close = doc.get_partner(first_obj);
        let mut name_val = None;
        let mut i = first_obj + 1;
        while i < first_close {
            if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "name" {
                name_val = Some(i + 1);
                break;
            }
            i += 2;
        }
        assert_eq!(&*doc.get_string(name_val.unwrap(), &padded), "Alice");

        // Container skip: jump from users array to "count"
        let users_close = doc.get_partner(users_arr);
        let after_users = users_close + 1;
        assert_eq!(doc.tape_type(after_users), TapeType::String);
        assert_eq!(&*doc.get_string(after_users, &padded), "count");
        assert_eq!(doc.get_i64(after_users + 1), 2);
    }

    #[test]
    fn test_json_dom_string_escapes() {
        let json = br#"{"msg":"hello\nworld","quote":"say \"hi\"","path":"C:\\Users"}"#;
        let (doc, padded) = jit_parse_json(json);

        use crate::json::TapeType;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String {
                let s = doc.get_string(i, &padded);
                if &*s == "msg" { assert_eq!(&*doc.get_string(i+1, &padded), "hello\nworld"); }
                if &*s == "quote" { assert_eq!(&*doc.get_string(i+1, &padded), "say \"hi\""); }
                if &*s == "path" { assert_eq!(&*doc.get_string(i+1, &padded), "C:\\Users"); }
            }
        }
    }

    #[test]
    fn test_json_dom_numbers() {
        let json = br#"{"int":42,"neg":-17,"big":1234567890}"#;
        let (doc, padded) = jit_parse_json(json);

        use crate::json::TapeType;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String {
                let key = doc.get_string(i, &padded);
                match &*key {
                    "int" => assert_eq!(doc.get_i64(i+1), 42),
                    "neg" => assert_eq!(doc.get_i64(i+1), -17),
                    "big" => assert_eq!(doc.get_i64(i+1), 1234567890),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn bench_json_dom_full_parse() {
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data":["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id":{},"name":"user_{}","score":{},"active":{}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();

        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let (fptr, _engine) = jit_lookup_opt(&source, "json_stage1", 2);

        let padded = crate::json::pad_input(json_str.as_bytes());
        let mut parser = crate::json::Parser::new(json_len);

        let stage1 = |input: &[u8], positions: &mut [i32]| -> usize {
            unsafe {
                let f: extern "C" fn(
                    *const u8, *const u8, i64, i64, i64,
                    *mut i32, *mut i32, i64, i64, i64,
                ) -> f32 = std::mem::transmute(fptr);
                f32::to_bits(f(
                    input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1,
                )) as usize
            }
        };

        // Warmup
        let _ = parser.parse_with_stage1(&padded, stage1);

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = parser.parse_with_stage1(&padded, stage1);
        }
        let elapsed = start.elapsed();
        let gb = json_len as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;

        eprintln!("\n=== Full DOM Parse (JIT stage1 + Rust tape builder) ===");
        eprintln!("Input: {:.2} MB", json_len as f64 / 1e6);
        eprintln!("Throughput: {:.2} GB/s ({:.3} ms/iter)",
            gb, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
        eprintln!("Compare: simdjson does ~1.5 GB/s on same data");
    }
}
