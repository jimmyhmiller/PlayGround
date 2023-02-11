#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

using ObjType = uint32_t;

struct Obj {
  ObjType type_;
  bool isMarked;
  Obj *next;
};

using Value = uint64_t;

struct ValueArray {
  int capacity;
  int count;
  Value *values;
};

struct Chunk {
  int count;
  int capacity;
  uint8_t *code;
  int *lines;
  ValueArray constants;
};

struct ObjString {
  Obj obj;
  int length;
  char *chars;
  uint32_t hash;
};

struct ObjFunction {
  Obj obj;
  int arity;
  int upvalueCount;
  Chunk chunk;
  ObjString *name;
};

struct ObjUpvalue {
  Obj obj;
  Value *location;
  Value closed;
  ObjUpvalue *next;
};

struct ObjClosure {
  Obj obj;
  ObjFunction *function;
  ObjUpvalue **upvalues;
  int upvalueCount;
};

struct CallFrame {
  ObjClosure *closure;
  uint8_t *ip;
  Value *slots;
};

struct Entry {
  ObjString *key;
  Value value;
};

struct Table {
  int count;
  int capacity;
  Entry *entries;
};

struct VM {
  CallFrame frames[64];
  int frameCount;
  Value stack[16384];
  Value *stackTop;
  Table globals;
  Table strings;
  ObjString *initString;
  ObjUpvalue *openUpvalues;
  size_t bytesAllocated;
  size_t nextGC;
  Obj *objects;
  int grayCount;
  int grayCapacity;
  Obj **grayStack;
};

using OpCode = uint32_t;

using InterpretResult = uint32_t;

constexpr static const OpCode OpCode_OP_CONSTANT = 0;

constexpr static const OpCode OpCode_OP_NIL = 1;

constexpr static const OpCode OpCode_OP_TRUE = 2;

constexpr static const OpCode OpCode_OP_FALSE = 3;

constexpr static const OpCode OpCode_OP_POP = 4;

constexpr static const OpCode OpCode_OP_GET_LOCAL = 5;

constexpr static const OpCode OpCode_OP_SET_LOCAL = 6;

constexpr static const OpCode OpCode_OP_GET_GLOBAL = 7;

constexpr static const OpCode OpCode_OP_DEFINE_GLOBAL = 8;

constexpr static const OpCode OpCode_OP_SET_GLOBAL = 9;

constexpr static const OpCode OpCode_OP_GET_UPVALUE = 10;

constexpr static const OpCode OpCode_OP_SET_UPVALUE = 11;

constexpr static const OpCode OpCode_OP_GET_PROPERTY = 12;

constexpr static const OpCode OpCode_OP_SET_PROPERTY = 13;

constexpr static const OpCode OpCode_OP_GET_SUPER = 14;

constexpr static const OpCode OpCode_OP_EQUAL = 15;

constexpr static const OpCode OpCode_OP_GREATER = 16;

constexpr static const OpCode OpCode_OP_LESS = 17;

constexpr static const OpCode OpCode_OP_ADD = 18;

constexpr static const OpCode OpCode_OP_SUBTRACT = 19;

constexpr static const OpCode OpCode_OP_MULTIPLY = 20;

constexpr static const OpCode OpCode_OP_DIVIDE = 21;

constexpr static const OpCode OpCode_OP_NOT = 22;

constexpr static const OpCode OpCode_OP_NEGATE = 23;

constexpr static const OpCode OpCode_OP_PRINT = 24;

constexpr static const OpCode OpCode_OP_JUMP = 25;

constexpr static const OpCode OpCode_OP_JUMP_IF_FALSE = 26;

constexpr static const OpCode OpCode_OP_LOOP = 27;

constexpr static const OpCode OpCode_OP_CALL = 28;

constexpr static const OpCode OpCode_OP_INVOKE = 29;

constexpr static const OpCode OpCode_OP_SUPER_INVOKE = 30;

constexpr static const OpCode OpCode_OP_CLOSURE = 31;

constexpr static const OpCode OpCode_OP_CLOSE_UPVALUE = 32;

constexpr static const OpCode OpCode_OP_RETURN = 33;

constexpr static const OpCode OpCode_OP_CLASS = 34;

constexpr static const OpCode OpCode_OP_INHERIT = 35;

constexpr static const OpCode OpCode_OP_METHOD = 36;

constexpr static const ObjType ObjType_OBJ_BOUND_METHOD = 0;

constexpr static const ObjType ObjType_OBJ_CLASS = 1;

constexpr static const ObjType ObjType_OBJ_CLOSURE = 2;

constexpr static const ObjType ObjType_OBJ_FUNCTION = 3;

constexpr static const ObjType ObjType_OBJ_INSTANCE = 4;

constexpr static const ObjType ObjType_OBJ_NATIVE = 5;

constexpr static const ObjType ObjType_OBJ_STRING = 6;

constexpr static const ObjType ObjType_OBJ_UPVALUE = 7;

constexpr static const InterpretResult InterpretResult_INTERPRET_OK = 0;

constexpr static const InterpretResult InterpretResult_INTERPRET_COMPILE_ERROR = 1;

constexpr static const InterpretResult InterpretResult_INTERPRET_RUNTIME_ERROR = 2;

extern "C" {

void on_closure_call(VM *vm, ObjClosure *obj_closure);

} // extern "C"
