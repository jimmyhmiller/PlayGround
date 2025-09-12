#ifndef IDENTIFIER_VALIDATOR_H
#define IDENTIFIER_VALIDATOR_H

#include <string>

class IdentifierValidator {
public:
  static bool is_valid_identifier(const std::string &identifier);
  static bool is_valid_identifier_start(char c);
  static bool is_valid_identifier_char(char c);
  static std::string get_validation_error(const std::string &identifier);
};

#endif