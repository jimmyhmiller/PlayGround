#ifndef LLVM_CODEGEN_H
#define LLVM_CODEGEN_H

#include "../src/ast.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <memory>
#include <string>

class LLVMCodeGenerator {
public:
  LLVMCodeGenerator();

  // Generate LLVM IR for the entire AST
  void generateCode(const ASTNode *ast);

  // Print the generated IR to stdout
  void printIR();

  // Print the full IR including execution wrapper
  void printFullIR();

  // Execute the generated code and return result
  double executeExpression();

  // Get the LLVM module (for further processing)
  llvm::Module *getModule() { return module.get(); }

private:
  // LLVM context and module
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::IRBuilder<>> builder;
  std::unique_ptr<llvm::ExecutionEngine> executionEngine;

  // Symbol table for named values
  std::map<std::string, llvm::Value *> namedValues;

  // Type tables
  std::map<std::string, llvm::StructType *> structTypes;
  // Map struct name to field names in order
  std::map<std::string, std::vector<std::string>> structFieldNames;

  // Store the last generated expression value for execution
  llvm::Value *lastExpressionValue;

  // Store the last expression AST node for re-generation in wrapper function
  const ASTNode *lastExpressionNode;

  // Store the module with wrapper function for full IR display
  std::unique_ptr<llvm::Module> fullModule;

  // Code generation methods for different AST node types
  llvm::Value *codegen(const ASTNode *node);
  llvm::Value *codegenIntegerLiteral(const ASTNode *node);
  llvm::Value *codegenNumberLiteral(const ASTNode *node);
  llvm::Value *codegenBoolLiteral(const ASTNode *node);
  llvm::Value *codegenIdentifier(const ASTNode *node);
  llvm::Value *codegenBinaryExpression(const ASTNode *node);
  llvm::Value *codegenFunctionCall(const ASTNode *node);
  llvm::Function *codegenFunctionDeclaration(const ASTNode *node);
  llvm::Value *codegenIfStatement(const ASTNode *node);
  llvm::StructType *codegenStructDeclaration(const ASTNode *node);
  llvm::Value *codegenStructLiteral(const ASTNode *node);
  llvm::Value *codegenFieldAccess(const ASTNode *node);
  llvm::Value *codegenLetStatement(const ASTNode *node);

  // Helper methods
  llvm::Function *createFunction(const std::string &name,
                                 const std::vector<std::string> &args);
  void logError(const std::string &message);

  // Binary operation helpers
  llvm::Value *codegenArithmeticOp(llvm::Value *L, llvm::Value *R,
                                   const std::string &op, bool leftIsInt,
                                   bool rightIsInt);
  llvm::Value *codegenComparisonOp(llvm::Value *L, llvm::Value *R,
                                   const std::string &op, bool leftIsInt,
                                   bool rightIsInt);
  void promoteToFloat(llvm::Value *&L, llvm::Value *&R, bool leftIsInt,
                      bool rightIsInt);
};

#endif