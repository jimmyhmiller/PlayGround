#include "llvm_codegen.h"
#include <iostream>
#include <algorithm>

LLVMCodeGenerator::LLVMCodeGenerator()
    : lastExpressionValue(nullptr), lastExpressionNode(nullptr),
      fullModule(nullptr) {
  // Initialize LLVM native target for JIT execution
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Initialize LLVM context, module, and builder
  context = std::make_unique<llvm::LLVMContext>();
  module = std::make_unique<llvm::Module>("my cool jit", *context);
  builder = std::make_unique<llvm::IRBuilder<>>(*context);
}

void LLVMCodeGenerator::generateCode(const ASTNode *ast) {
  if (!ast) {
    logError("Null AST node");
    return;
  }

  // Create a temporary main function context for top-level code generation
  std::vector<llvm::Type *> argTypes;
  llvm::FunctionType *mainType = llvm::FunctionType::get(
      llvm::Type::getDoubleTy(*context), argTypes, false);

  llvm::Function *tempMainFunc = llvm::Function::Create(
      mainType, llvm::Function::InternalLinkage, "__temp_main", module.get());

  llvm::BasicBlock *mainBB =
      llvm::BasicBlock::Create(*context, "entry", tempMainFunc);
  builder->SetInsertPoint(mainBB);

  // Handle the case where the AST is a Block (do block) containing multiple
  // statements
  if (ast->type == ASTNodeType::Block) {
    llvm::Value *lastValue = nullptr;
    for (size_t i = 0; i < ast->child_count(); ++i) {
      lastValue = codegen(ast->child(i));
      // Keep track of the last non-null value as the expression result
      if (lastValue) {
        lastExpressionValue = lastValue;
      }
    }
  } else {
    lastExpressionValue = codegen(ast);
  }

  // Remove the temporary main function after generation
  tempMainFunc->eraseFromParent();
  builder->ClearInsertionPoint();
}

void LLVMCodeGenerator::printIR() { module->print(llvm::outs(), nullptr); }

void LLVMCodeGenerator::printFullIR() {
  if (fullModule) {
    fullModule->print(llvm::outs(), nullptr);
  } else {
    std::cerr
        << "No full module available (executeExpression() must be called first)"
        << std::endl;
  }
}

llvm::Value *LLVMCodeGenerator::codegen(const ASTNode *node) {
  if (!node) {
    return nullptr;
  }

  switch (node->type) {
  case ASTNodeType::IntegerLiteral:
    return codegenIntegerLiteral(node);

  case ASTNodeType::NumberLiteral:
    return codegenNumberLiteral(node);

  case ASTNodeType::BoolLiteral:
    return codegenBoolLiteral(node);

  case ASTNodeType::Identifier:
    return codegenIdentifier(node);

  case ASTNodeType::BinaryExpression:
    return codegenBinaryExpression(node);

  case ASTNodeType::FunctionCall:
    return codegenFunctionCall(node);

  case ASTNodeType::FunctionDeclaration:
    codegenFunctionDeclaration(node);
    return nullptr;

  case ASTNodeType::Program: {
    // Handle program node by processing all children
    llvm::Value *lastValue = nullptr;
    for (size_t i = 0; i < node->child_count(); ++i) {
      const ASTNode *child = node->child(i);
      llvm::Value *value = codegen(child);
      // Only update lastValue for expression statements (not function
      // declarations)
      if (value != nullptr) {
        lastValue = value;
        lastExpressionNode = child; // Store the AST node for later regeneration
      }
    }
    return lastValue;
  }

  case ASTNodeType::ExpressionStatement:
    // Expression statements wrap an expression, just evaluate the inner
    // expression
    if (node->child_count() > 0) {
      return codegen(node->child(0));
    }
    return nullptr;

  case ASTNodeType::Block: {
    // Handle block by processing all children, returning the last value
    llvm::Value *lastValue = nullptr;
    for (size_t i = 0; i < node->child_count(); ++i) {
      lastValue = codegen(node->child(i));
    }
    return lastValue;
  }

  case ASTNodeType::IfStatement:
    return codegenIfStatement(node);

  case ASTNodeType::StructDeclaration:
    codegenStructDeclaration(node);
    return nullptr;

  case ASTNodeType::StructLiteral:
    return codegenStructLiteral(node);

  case ASTNodeType::LetStatement:
    return codegenLetStatement(node);

  default:
    logError("Unknown AST node type: " +
             std::to_string(static_cast<int>(node->type)));
    return nullptr;
  }
}

llvm::Value *LLVMCodeGenerator::codegenIntegerLiteral(const ASTNode *node) {
  if (!node->value.empty()) {
    // Parse the integer - use 64-bit signed integer (equivalent to usize on
    // 64-bit systems)
    int64_t val = std::stoll(node->value);
    return llvm::ConstantInt::get(*context, llvm::APInt(64, val, true));
  }

  logError("Empty integer literal");
  return nullptr;
}

llvm::Value *LLVMCodeGenerator::codegenNumberLiteral(const ASTNode *node) {
  if (!node->value.empty()) {
    // Parse the number - for now, assume it's a double
    double val = std::stod(node->value);
    return llvm::ConstantFP::get(*context, llvm::APFloat(val));
  }

  logError("Empty number literal");
  return nullptr;
}

llvm::Value *LLVMCodeGenerator::codegenBoolLiteral(const ASTNode *node) {
  if (!node->value.empty()) {
    bool val = (node->value == "true");
    return llvm::ConstantInt::get(*context, llvm::APInt(1, val));
  }

  logError("Empty boolean literal");
  return nullptr;
}

llvm::Value *LLVMCodeGenerator::codegenIdentifier(const ASTNode *node) {
  // Look up the value in the symbol table
  llvm::Value *V = namedValues[node->value];
  if (!V) {
    logError("Unknown variable name: " + node->value);
  }
  return V;
}

llvm::Value *LLVMCodeGenerator::codegenBinaryExpression(const ASTNode *node) {
  if (node->child_count() < 2) {
    logError("Binary expression needs two operands");
    return nullptr;
  }

  llvm::Value *L = codegen(node->child(0));
  llvm::Value *R = codegen(node->child(1));

  if (!L || !R) {
    return nullptr;
  }

  // Check if operands are integers or floats
  bool leftIsInt = L->getType()->isIntegerTy();
  bool rightIsInt = R->getType()->isIntegerTy();

  // Get the operator from the node's value field
  std::string op = node->value;

  // Handle field access (.) operator
  if (op == ".") {
    return codegenFieldAccess(node);
  }

  // Handle operations using helper functions
  if (op == "+" || op == "-" || op == "*" || op == "/") {
    return codegenArithmeticOp(L, R, op, leftIsInt, rightIsInt);
  } else if (op == "<" || op == ">" || op == "==" || op == "!=" || op == "<=" ||
             op == ">=") {
    return codegenComparisonOp(L, R, op, leftIsInt, rightIsInt);
  } else {
    logError("Unknown binary operator: " + op);
    return nullptr;
  }
}

llvm::Value *LLVMCodeGenerator::codegenFunctionCall(const ASTNode *node) {
  if (node->child_count() == 0) {
    logError("Function call needs at least function name");
    return nullptr;
  }

  // First child should be the function name
  const ASTNode *nameNode = node->child(0);
  std::string callee = nameNode->value;

  // Look up the function in the module
  llvm::Function *calleeF = module->getFunction(callee);
  if (!calleeF) {
    logError("Unknown function referenced: " + callee);
    return nullptr;
  }

  // Check argument count
  if (calleeF->arg_size() != node->child_count() - 1) {
    logError("Incorrect # arguments passed");
    return nullptr;
  }

  // Generate code for arguments
  std::vector<llvm::Value *> argsV;
  for (size_t i = 1; i < node->child_count(); ++i) {
    argsV.push_back(codegen(node->child(i)));
    if (!argsV.back()) {
      return nullptr;
    }
  }

  return builder->CreateCall(calleeF, argsV, "calltmp");
}

llvm::Function *
LLVMCodeGenerator::codegenFunctionDeclaration(const ASTNode *node) {
  // Extract function name from the node's name field
  std::string functionName = node->name;
  if (functionName.empty()) {
    logError("Function declaration missing name");
    return nullptr;
  }

  // Extract parameter names from function_type
  std::vector<std::string> argNames;
  if (node->function_type) {
    // The first child of function_type contains the parameters
    for (size_t i = 0; i < node->function_type->child_count(); ++i) {
      const ASTNode *child = node->function_type->child(i);
      if (child->type == ASTNodeType::Parameter) {
        // Look for nested parameter with a name
        if (!child->name.empty()) {
          argNames.push_back(child->name);
        } else if (child->child_count() > 0) {
          // Check child parameters
          for (size_t j = 0; j < child->child_count(); ++j) {
            const ASTNode *param = child->child(j);
            if (param->type == ASTNodeType::Parameter && !param->name.empty()) {
              argNames.push_back(param->name);
            }
          }
        }
      }
    }
  }

  // Save the current insertion point and named values
  llvm::BasicBlock *savedBB = builder->GetInsertBlock();
  std::map<std::string, llvm::Value *> savedNamedValues = namedValues;

  // Create the function
  llvm::Function *function = createFunction(functionName, argNames);
  if (!function) {
    return nullptr;
  }

  // Create a new basic block to start insertion into
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(*context, "entry", function);
  builder->SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map
  namedValues.clear();
  size_t idx = 0;
  for (auto &arg : function->args()) {
    if (idx < argNames.size()) {
      namedValues[argNames[idx]] = &arg;
    }
    idx++;
  }

  // Generate code for the function body
  if (node->body) {
    if (llvm::Value *retVal = codegen(node->body.get())) {
      // Finish off the function
      builder->CreateRet(retVal);

      // Validate the generated code
      llvm::verifyFunction(*function);

      // Restore the previous insertion point and named values
      if (savedBB) {
        builder->SetInsertPoint(savedBB);
      }
      namedValues = savedNamedValues;

      return function;
    }
  }

  // Error reading body, remove function and restore context
  function->eraseFromParent();
  if (savedBB) {
    builder->SetInsertPoint(savedBB);
  }
  namedValues = savedNamedValues;
  return nullptr;
}

llvm::Function *
LLVMCodeGenerator::createFunction(const std::string &name,
                                  const std::vector<std::string> &args) {
  // Make the function type: i64(i64,i64) etc. for integer functions
  // For now, default to integer types for consistency with literals
  std::vector<llvm::Type *> argTypes(args.size(),
                                     llvm::Type::getInt64Ty(*context));

  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(*context), argTypes, false);

  llvm::Function *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, name, module.get());

  // Set names for all arguments
  unsigned idx = 0;
  for (auto &arg : F->args()) {
    arg.setName(args[idx++]);
  }

  return F;
}

double LLVMCodeGenerator::executeExpression() {
  if (!lastExpressionValue) {
    logError("No expression value to execute");
    return 0.0;
  }

  // Clear the current insertion point to avoid issues with existing functions
  builder->ClearInsertionPoint();

  // Create a wrapper function that returns our expression result
  std::vector<llvm::Type *> argTypes;
  llvm::FunctionType *wrapperType = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(*context), argTypes, false);

  llvm::Function *wrapperFunc =
      llvm::Function::Create(wrapperType, llvm::Function::ExternalLinkage,
                             "expr_wrapper", module.get());

  // Create basic block for the wrapper function
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(*context, "entry", wrapperFunc);
  builder->SetInsertPoint(BB);

  // Clear namedValues for wrapper function - we should only reference other
  // functions
  std::map<std::string, llvm::Value *> savedNamedValues = namedValues;
  namedValues.clear();

  // Re-generate the expression from the AST in the wrapper function context
  llvm::Value *exprValue = nullptr;
  if (lastExpressionNode) {
    exprValue = codegen(lastExpressionNode);
  }

  if (!exprValue) {
    logError("Failed to regenerate expression in wrapper function");
    return 0.0;
  }

  // Convert double to integer if necessary for wrapper function
  if (exprValue->getType()->isDoubleTy()) {
    exprValue = builder->CreateFPToSI(
        exprValue, llvm::Type::getInt64Ty(*context), "fptoint");
  }

  builder->CreateRet(exprValue);

  // Restore namedValues
  namedValues = savedNamedValues;

  // Verify the function
  llvm::verifyFunction(*wrapperFunc);

  // Create execution engine - we need to clone the module since EngineBuilder
  // takes ownership
  fullModule =
      llvm::CloneModule(*module); // Store the full module for IR display
  std::string errorStr;
  llvm::EngineBuilder engineBuilder(llvm::CloneModule(*fullModule));
  engineBuilder.setErrorStr(&errorStr);
  engineBuilder.setEngineKind(llvm::EngineKind::JIT);
  std::unique_ptr<llvm::ExecutionEngine> tempEngine(engineBuilder.create());

  if (!tempEngine) {
    logError("Failed to create execution engine: " + errorStr);
    return 0.0;
  }

  // Finalize object (needed for MCJIT)
  tempEngine->finalizeObject();

  // Get pointer to the function and execute it
  void *funcPtr = tempEngine->getPointerToFunction(wrapperFunc);
  if (!funcPtr) {
    // Try alternative method - get function address by name
    uint64_t funcAddr = tempEngine->getFunctionAddress("expr_wrapper");
    if (funcAddr == 0) {
      logError("Failed to get function pointer or address");
      return 0.0;
    }
    funcPtr = reinterpret_cast<void *>(funcAddr);
  }

  int64_t (*fp)() = (int64_t (*)())funcPtr;
  return static_cast<double>(fp());
}

llvm::Value *LLVMCodeGenerator::codegenIfStatement(const ASTNode *node) {
  if (node->child_count() < 2) {
    logError("If statement needs at least condition and then branch");
    return nullptr;
  }

  // Get current function to create basic blocks
  llvm::Function *function = builder->GetInsertBlock()->getParent();

  // Create basic blocks for then, else (if present), and merge
  llvm::BasicBlock *thenBB =
      llvm::BasicBlock::Create(*context, "then", function);
  llvm::BasicBlock *elseBB = nullptr;
  llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(*context, "ifcont");

  bool hasElse = (node->child_count() > 2);
  if (hasElse) {
    elseBB = llvm::BasicBlock::Create(*context, "else");
  }

  // Generate condition
  llvm::Value *condValue = codegen(node->child(0));
  if (!condValue) {
    return nullptr;
  }

  // Convert condition to boolean if needed
  if (condValue->getType()->isIntegerTy()) {
    // For integers, compare with 0 (false if 0, true otherwise)
    condValue = builder->CreateICmpNE(
        condValue, llvm::ConstantInt::get(condValue->getType(), 0), "ifcond");
  } else if (condValue->getType()->isDoubleTy()) {
    // For floats, compare with 0.0
    condValue = builder->CreateFCmpONE(
        condValue, llvm::ConstantFP::get(*context, llvm::APFloat(0.0)),
        "ifcond");
  }

  // Create conditional branch
  if (hasElse) {
    builder->CreateCondBr(condValue, thenBB, elseBB);
  } else {
    builder->CreateCondBr(condValue, thenBB, mergeBB);
  }

  // Generate then block
  builder->SetInsertPoint(thenBB);
  llvm::Value *thenValue = codegen(node->child(1));
  if (!thenValue) {
    return nullptr;
  }
  builder->CreateBr(mergeBB);
  thenBB = builder->GetInsertBlock(); // Update in case block was split

  // Generate else block if present
  llvm::Value *elseValue = nullptr;
  if (hasElse) {
    elseBB->insertInto(function);
    builder->SetInsertPoint(elseBB);
    elseValue = codegen(node->child(2));
    if (!elseValue) {
      return nullptr;
    }
    builder->CreateBr(mergeBB);
    elseBB = builder->GetInsertBlock(); // Update in case block was split
  }

  // Generate merge block
  mergeBB->insertInto(function);
  builder->SetInsertPoint(mergeBB);

  // If there's no else branch, create a phi node with appropriate default value
  if (!hasElse) {
    // Return then value directly if no else branch
    return thenValue;
  }

  // Create phi node to merge then and else values
  // First, ensure both values have the same type
  if (thenValue->getType() != elseValue->getType()) {
    // Promote integers to float if needed
    if (thenValue->getType()->isIntegerTy() &&
        elseValue->getType()->isDoubleTy()) {
      thenValue = builder->CreateSIToFP(
          thenValue, llvm::Type::getDoubleTy(*context), "inttofp");
    } else if (elseValue->getType()->isIntegerTy() &&
               thenValue->getType()->isDoubleTy()) {
      elseValue = builder->CreateSIToFP(
          elseValue, llvm::Type::getDoubleTy(*context), "inttofp");
    }
  }

  llvm::PHINode *phi = builder->CreatePHI(thenValue->getType(), 2, "iftmp");
  phi->addIncoming(thenValue, thenBB);
  phi->addIncoming(elseValue, elseBB);

  return phi;
}

llvm::StructType *
LLVMCodeGenerator::codegenStructDeclaration(const ASTNode *node) {
  if (node->name.empty()) {
    logError("Struct declaration missing name");
    return nullptr;
  }

  std::string structName = node->name;

  // Check if struct already exists - return existing one if found
  auto existingStruct = structTypes.find(structName);
  if (existingStruct != structTypes.end()) {
    return existingStruct->second;
  }

  // Collect field types and names
  std::vector<llvm::Type *> fieldTypes;
  std::vector<std::string> fieldNames;

  for (size_t i = 0; i < node->child_count(); ++i) {
    const ASTNode *field = node->child(i);
    if (field->type != ASTNodeType::FieldDeclaration) {
      logError("Expected field declaration in struct " + structName);
      continue;
    }
    
    // Store field name
    fieldNames.push_back(field->name);

    // Get the field type from the first child (should be TypeIdentifier)
    if (field->child_count() > 0) {
      const ASTNode *typeNode = field->child(0);
      llvm::Type *fieldType = nullptr;

      if (typeNode->type == ASTNodeType::TypeIdentifier) {
        std::string typeName = typeNode->value;
        if (typeName == "int") {
          fieldType = llvm::Type::getInt64Ty(*context);
        } else if (typeName == "float" || typeName == "double") {
          fieldType = llvm::Type::getDoubleTy(*context);
        } else if (typeName == "bool") {
          fieldType = llvm::Type::getInt1Ty(*context);
        } else {
          // Check if it's a user-defined struct type
          auto structIt = structTypes.find(typeName);
          if (structIt != structTypes.end()) {
            fieldType = structIt->second;
          } else {
            logError("Unknown type: " + typeName + " in struct " + structName);
            fieldType = llvm::Type::getInt64Ty(*context); // fallback
          }
        }
      } else {
        logError("Expected type identifier for field in struct " + structName);
        fieldType = llvm::Type::getInt64Ty(*context); // fallback
      }

      fieldTypes.push_back(fieldType);
    }
  }

  // Create the struct type
  llvm::StructType *structType =
      llvm::StructType::create(*context, fieldTypes, structName);

  // Store in the struct types and field names maps
  structTypes[structName] = structType;
  structFieldNames[structName] = fieldNames;

  return structType;
}

llvm::Value *LLVMCodeGenerator::codegenStructLiteral(const ASTNode *node) {
  if (node->value.empty()) {
    logError("Struct literal missing type name");
    return nullptr;
  }

  std::string structName = node->value;

  // Look up the struct type
  auto structIt = structTypes.find(structName);
  if (structIt == structTypes.end()) {
    logError("Unknown struct type: " + structName);
    return nullptr;
  }

  llvm::StructType *structType = structIt->second;
  
  // Look up the field names for this struct
  auto fieldNamesIt = structFieldNames.find(structName);
  if (fieldNamesIt == structFieldNames.end()) {
    logError("No field names found for struct type: " + structName);
    return nullptr;
  }
  
  const std::vector<std::string> &fieldNames = fieldNamesIt->second;

  // Create an alloca for the struct instance
  llvm::Value *structPtr =
      builder->CreateAlloca(structType, nullptr, structName + "_instance");

  // Initialize fields from the field assignments
  for (size_t i = 0; i < node->child_count(); ++i) {
    const ASTNode *field = node->child(i);
    if (field->type == ASTNodeType::FieldDeclaration && !field->name.empty()) {

      // Find the field index by name
      auto fieldIt = std::find(fieldNames.begin(), fieldNames.end(), field->name);
      if (fieldIt == fieldNames.end()) {
        logError("Field '" + field->name + "' not found in struct " + structName);
        continue;
      }
      
      size_t fieldIndex = static_cast<size_t>(std::distance(fieldNames.begin(), fieldIt));

      if (field->child_count() > 0) {
        // Generate code for the field value
        llvm::Value *fieldValue = codegen(field->child(0));
        if (!fieldValue) {
          logError("Failed to generate code for field value: " + field->name);
          continue;
        }

        // Create GEP to access the field using the correct index
        llvm::Value *fieldPtr = builder->CreateStructGEP(
            structType, structPtr, fieldIndex, field->name + "_ptr");

        // Store the value in the field
        builder->CreateStore(fieldValue, fieldPtr);
      }
    }
  }

  // Load the struct value
  return builder->CreateLoad(structType, structPtr, structName + "_value");
}

void LLVMCodeGenerator::logError(const std::string &message) {
  std::cerr << "LLVM Codegen Error: " << message << std::endl;
}

void LLVMCodeGenerator::promoteToFloat(llvm::Value *&L, llvm::Value *&R,
                                       bool leftIsInt, bool rightIsInt) {
  if (leftIsInt)
    L = builder->CreateSIToFP(L, llvm::Type::getDoubleTy(*context), "inttofp");
  if (rightIsInt)
    R = builder->CreateSIToFP(R, llvm::Type::getDoubleTy(*context), "inttofp");
}

llvm::Value *LLVMCodeGenerator::codegenArithmeticOp(llvm::Value *L,
                                                    llvm::Value *R,
                                                    const std::string &op,
                                                    bool leftIsInt,
                                                    bool rightIsInt) {
  if (op == "+") {
    if (leftIsInt && rightIsInt) {
      return builder->CreateAdd(L, R, "addtmp");
    } else {
      promoteToFloat(L, R, leftIsInt, rightIsInt);
      return builder->CreateFAdd(L, R, "addtmp");
    }
  } else if (op == "-") {
    if (leftIsInt && rightIsInt) {
      return builder->CreateSub(L, R, "subtmp");
    } else {
      promoteToFloat(L, R, leftIsInt, rightIsInt);
      return builder->CreateFSub(L, R, "subtmp");
    }
  } else if (op == "*") {
    if (leftIsInt && rightIsInt) {
      return builder->CreateMul(L, R, "multmp");
    } else {
      promoteToFloat(L, R, leftIsInt, rightIsInt);
      return builder->CreateFMul(L, R, "multmp");
    }
  } else if (op == "/") {
    // Division always promotes to float
    promoteToFloat(L, R, leftIsInt, rightIsInt);
    return builder->CreateFDiv(L, R, "divtmp");
  }

  logError("Unknown arithmetic operator: " + op);
  return nullptr;
}

llvm::Value *LLVMCodeGenerator::codegenComparisonOp(llvm::Value *L,
                                                    llvm::Value *R,
                                                    const std::string &op,
                                                    bool leftIsInt,
                                                    bool rightIsInt) {
  llvm::Value *result = nullptr;

  if (leftIsInt && rightIsInt) {
    // Integer comparison
    if (op == "<") {
      result = builder->CreateICmpSLT(L, R, "cmptmp");
    } else if (op == ">") {
      result = builder->CreateICmpSGT(L, R, "cmptmp");
    } else if (op == "==") {
      result = builder->CreateICmpEQ(L, R, "cmptmp");
    } else if (op == "!=") {
      result = builder->CreateICmpNE(L, R, "cmptmp");
    } else if (op == "<=") {
      result = builder->CreateICmpSLE(L, R, "cmptmp");
    } else if (op == ">=") {
      result = builder->CreateICmpSGE(L, R, "cmptmp");
    }
  } else {
    // Float comparison (promote integers first)
    promoteToFloat(L, R, leftIsInt, rightIsInt);
    if (op == "<") {
      result = builder->CreateFCmpULT(L, R, "cmptmp");
    } else if (op == ">") {
      result = builder->CreateFCmpUGT(L, R, "cmptmp");
    } else if (op == "==") {
      result = builder->CreateFCmpOEQ(L, R, "cmptmp");
    } else if (op == "!=") {
      result = builder->CreateFCmpONE(L, R, "cmptmp");
    } else if (op == "<=") {
      result = builder->CreateFCmpOLE(L, R, "cmptmp");
    } else if (op == ">=") {
      result = builder->CreateFCmpOGE(L, R, "cmptmp");
    }
  }

  if (result) {
    // Convert bool to integer (0 or 1) for consistency
    return builder->CreateZExt(result, llvm::Type::getInt64Ty(*context),
                               "booltmp");
  }

  logError("Unknown comparison operator: " + op);
  return nullptr;
}

llvm::Value *LLVMCodeGenerator::codegenFieldAccess(const ASTNode *node) {
  if (node->child_count() < 2) {
    logError("Field access needs struct and field name");
    return nullptr;
  }

  // Left side should be the struct value or identifier
  const ASTNode *structNode = node->child(0);
  const ASTNode *fieldNode = node->child(1);

  if (fieldNode->type != ASTNodeType::Identifier) {
    logError("Field access requires field name as identifier");
    return nullptr;
  }

  std::string fieldName = fieldNode->value;

  // Generate code for the struct (this could be a struct literal, identifier, etc.)
  llvm::Value *structValue = codegen(structNode);
  if (!structValue) {
    logError("Failed to generate code for struct in field access");
    return nullptr;
  }

  // Determine the struct type
  llvm::Type *structType = structValue->getType();
  
  // If the struct value is a pointer, we know it's pointing to our struct type
  // In this case we need to handle it differently in the GEP creation
  
  // Find the struct type name and field index
  std::string structTypeName;
  llvm::StructType *structTypePtr = nullptr;
  
  // Look through our struct types to find the matching one
  // We need to handle both direct struct types and pointer-to-struct types
  for (const auto &pair : structTypes) {
    if (pair.second == structType) {
      // Direct struct type match
      structTypeName = pair.first;
      structTypePtr = pair.second;
      break;
    } else if (structType->isPointerTy()) {
      // For pointer types, we'll need to compare by name since we can't easily get element type
      // For now, let's assume it's the struct type we're looking for based on context
      structTypeName = pair.first;
      structTypePtr = pair.second;
      break; // We'll take the first one for now - this is a simplification
    }
  }

  if (structTypeName.empty() || !structTypePtr) {
    logError("Unknown struct type in field access");
    return nullptr;
  }

  // Find field index by name
  auto fieldNamesIt = structFieldNames.find(structTypeName);
  if (fieldNamesIt == structFieldNames.end()) {
    logError("No field names found for struct type: " + structTypeName);
    return nullptr;
  }

  const std::vector<std::string> &fieldNames = fieldNamesIt->second;
  auto fieldIt = std::find(fieldNames.begin(), fieldNames.end(), fieldName);
  if (fieldIt == fieldNames.end()) {
    logError("Field '" + fieldName + "' not found in struct " + structTypeName);
    return nullptr;
  }

  size_t fieldIndex = static_cast<size_t>(std::distance(fieldNames.begin(), fieldIt));

  // If structValue is already a struct value (not a pointer), we need to allocate and store it
  if (!structValue->getType()->isPointerTy()) {
    // Create temporary allocation for the struct
    llvm::Value *structPtr = builder->CreateAlloca(structTypePtr, nullptr, "temp_struct");
    builder->CreateStore(structValue, structPtr);
    structValue = structPtr;
  }

  // Create GEP to access the field
  llvm::Value *fieldPtr = builder->CreateStructGEP(
      structTypePtr, structValue, fieldIndex, fieldName + "_access");

  // Load and return the field value
  llvm::Type *fieldType = structTypePtr->getElementType(fieldIndex);
  return builder->CreateLoad(fieldType, fieldPtr, fieldName + "_value");
}

llvm::Value *LLVMCodeGenerator::codegenLetStatement(const ASTNode *node) {
  if (node->child_count() < 2) {
    logError("Let statement needs variable name and value");
    return nullptr;
  }

  // First child should be the variable identifier
  const ASTNode *varNode = node->child(0);
  if (varNode->type != ASTNodeType::Identifier) {
    logError("Let statement requires variable name as identifier");
    return nullptr;
  }

  std::string varName = varNode->value;

  // Second child should be the value expression
  const ASTNode *valueNode = node->child(1);
  llvm::Value *initValue = codegen(valueNode);
  if (!initValue) {
    logError("Failed to generate code for let statement value");
    return nullptr;
  }

  // Store the value in the named values map for later reference
  namedValues[varName] = initValue;

  // Return the initialized value (let statements can be used as expressions)
  return initValue;
}