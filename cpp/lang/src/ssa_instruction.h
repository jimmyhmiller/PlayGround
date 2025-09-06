#ifndef SSA_INSTRUCTION_H
#define SSA_INSTRUCTION_H

#include <memory>
#include <string>
#include <vector>

// Forward declarations
struct SSAValue;
struct SSABlock;
struct SSAPhi;

enum class BinaryOperator {
  Add,
  Subtract,
  Multiply,
  Divide,
  Equal,
  NotEqual,
  LessThan,
  LessThanOrEqual,
  GreaterThan,
  GreaterThanOrEqual,
  LogicalOr,
  LogicalAnd
};

enum class UnaryOperator {
  Negate,
  Not
};

struct Variable {
  std::string name;
  
  Variable() : name("") {}
  Variable(const std::string& n) : name(n) {}
  
  static Variable temp(size_t id) {
    return Variable("v" + std::to_string(id));
  }
};

struct PhiId {
  size_t id;
  
  PhiId() : id(0) {}
  PhiId(size_t i) : id(i) {}
  
  bool operator==(const PhiId& other) const {
    return id == other.id;
  }
  
  bool operator<(const PhiId& other) const {
    return id < other.id;
  }
};

struct BlockId {
  size_t id;
  
  BlockId() : id(0) {}
  BlockId(size_t i) : id(i) {}
  
  bool operator==(const BlockId& other) const {
    return id == other.id;
  }
  
  bool operator<(const BlockId& other) const {
    return id < other.id;
  }
};

enum class SSAValueType {
  Literal,
  Var,
  Phi,
  Undefined
};

struct SSAValue {
  SSAValueType type;
  int literal_value;
  Variable variable;
  PhiId phi_id;
  
  SSAValue() : type(SSAValueType::Undefined), literal_value(0), variable(""), phi_id(0) {}
  
  static SSAValue literal(int value) {
    SSAValue v;
    v.type = SSAValueType::Literal;
    v.literal_value = value;
    return v;
  }
  
  static SSAValue var(const Variable& variable) {
    SSAValue v;
    v.type = SSAValueType::Var;
    v.variable = variable;
    return v;
  }
  
  static SSAValue phi(const PhiId& phi_id) {
    SSAValue v;
    v.type = SSAValueType::Phi;
    v.phi_id = phi_id;
    return v;
  }
  
  static SSAValue undefined() {
    SSAValue v;
    v.type = SSAValueType::Undefined;
    return v;
  }
  
  bool is_phi() const {
    return type == SSAValueType::Phi;
  }
  
  bool is_same_phi(const PhiId& id) const {
    return type == SSAValueType::Phi && phi_id.id == id.id;
  }
};

enum class PhiReferenceType {
  Instruction,
  Phi
};

struct PhiReference {
  PhiReferenceType type;
  BlockId block_id;
  size_t instruction_offset;
  PhiId phi_id;
  
  static PhiReference instruction(BlockId block_id, size_t offset) {
    PhiReference ref;
    ref.type = PhiReferenceType::Instruction;
    ref.block_id = block_id;
    ref.instruction_offset = offset;
    return ref;
  }
  
  static PhiReference phi(PhiId phi_id) {
    PhiReference ref;
    ref.type = PhiReferenceType::Phi;
    ref.phi_id = phi_id;
    return ref;
  }
};

struct SSAPhi {
  PhiId id;
  BlockId block_id;
  std::vector<SSAValue> operands;
  std::vector<PhiReference> uses;
  
  SSAPhi() : id(), block_id() {}
  SSAPhi(PhiId id, BlockId block_id) : id(id), block_id(block_id) {}
};

enum class SSAInstructionType {
  Assign,
  BinaryOp,
  UnaryOp,
  Jump,
  ConditionalJump,
  Print,
  Phi,
  Return
};

struct SSAInstruction {
  SSAInstructionType type;
  Variable dest;
  SSAValue value;
  SSAValue left;
  BinaryOperator binary_operator;
  UnaryOperator unary_operator;
  SSAValue right;
  SSAValue operand;
  BlockId target;
  SSAValue condition;
  BlockId true_target;
  BlockId false_target;
  PhiId phi_id;
  
  SSAInstruction() : type(SSAInstructionType::Assign) {}
  
  static SSAInstruction assign(const Variable& dest, const SSAValue& value) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::Assign;
    instr.dest = dest;
    instr.value = value;
    return instr;
  }
  
  static SSAInstruction binary_op(const Variable& dest, const SSAValue& left,
                                  BinaryOperator op, const SSAValue& right) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::BinaryOp;
    instr.dest = dest;
    instr.left = left;
    instr.binary_operator = op;
    instr.right = right;
    return instr;
  }
  
  static SSAInstruction unary_op(const Variable& dest, UnaryOperator op,
                                 const SSAValue& operand) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::UnaryOp;
    instr.dest = dest;
    instr.unary_operator = op;
    instr.operand = operand;
    return instr;
  }
  
  static SSAInstruction jump(BlockId target) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::Jump;
    instr.target = target;
    return instr;
  }
  
  static SSAInstruction conditional_jump(const SSAValue& condition,
                                         BlockId true_target,
                                         BlockId false_target) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::ConditionalJump;
    instr.condition = condition;
    instr.true_target = true_target;
    instr.false_target = false_target;
    return instr;
  }
  
  static SSAInstruction print(const SSAValue& value) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::Print;
    instr.value = value;
    return instr;
  }
  
  static SSAInstruction phi(const Variable& dest, PhiId phi_id) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::Phi;
    instr.dest = dest;
    instr.phi_id = phi_id;
    return instr;
  }
  
  static SSAInstruction ret(const SSAValue& value) {
    SSAInstruction instr;
    instr.type = SSAInstructionType::Return;
    instr.value = value;
    return instr;
  }
};

struct SSABlock {
  BlockId id;
  std::vector<SSAInstruction> instructions;
  std::vector<BlockId> predecessors;
  bool sealed;
  
  SSABlock(BlockId id) : id(id), sealed(false) {}
  
  void add_instruction(const SSAInstruction& instruction) {
    instructions.push_back(instruction);
  }
  
  void add_predecessor(BlockId predecessor) {
    if (sealed) {
      throw std::runtime_error("Cannot add predecessor to sealed block");
    }
    predecessors.push_back(predecessor);
  }
  
  void seal() {
    sealed = true;
  }
};

#endif