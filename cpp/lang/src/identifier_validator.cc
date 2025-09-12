#include "identifier_validator.h"
#include <cctype>

bool IdentifierValidator::is_valid_identifier(const std::string &identifier) {
  if (identifier.empty()) {
    return false;
  }

  // Must start with letter or underscore
  if (!is_valid_identifier_start(identifier[0])) {
    return false;
  }

  // Rest must be letters, digits, or underscores
  for (size_t i = 1; i < identifier.length(); ++i) {
    if (!is_valid_identifier_char(identifier[i])) {
      return false;
    }
  }

  return true;
}

bool IdentifierValidator::is_valid_identifier_start(char c) {
  return std::isalpha(c) || c == '_';
}

bool IdentifierValidator::is_valid_identifier_char(char c) {
  return std::isalnum(c) || c == '_';
}

std::string
IdentifierValidator::get_validation_error(const std::string &identifier) {
  if (identifier.empty()) {
    return "Identifier cannot be empty";
  }

  if (!is_valid_identifier_start(identifier[0])) {
    return "Identifier '" + identifier +
           "' must start with a letter or underscore, not '" + identifier[0] +
           "'";
  }

  for (size_t i = 1; i < identifier.length(); ++i) {
    if (!is_valid_identifier_char(identifier[i])) {
      return "Identifier '" + identifier + "' contains invalid character '" +
             identifier[i] + "' at position " + std::to_string(i);
    }
  }

  return ""; // No error
}