#include "tokenizer.h"
#include <cassert>
#include <cctype>

Token Tokenizer::handle_whitespace(const std::string_view input) {
  int start = pos;
  int start_line = line;
  int start_column = column;
  while (!at_end(input) && is_whitespace(input[pos])) {
    consume(input);
  }
  return Token{TokenType::Whitespace, input.substr(start, pos - start),
               start_line, start_column};
}

Token Tokenizer::next_token(const std::string_view input) {
  while (!at_end(input)) {
    char current_char = input[pos];
    if (is_whitespace(current_char)) {
      return handle_whitespace(input);
    } else if (std::isalpha(current_char) || current_char == '_') {
      int start = pos;
      int start_line = line;
      int start_column = column;
      while (!at_end(input) && (std::isalnum(input[pos]) || input[pos] == '_' ||
                                input[pos] == '!')) {
        consume(input);
      }
      std::string_view token_value = input.substr(start, pos - start);
      
      // Check for boolean literals
      if (token_value == "true" || token_value == "false") {
        return Token{TokenType::Boolean, token_value, start_line, start_column};
      }
      
      return Token{TokenType::Identifier, token_value, start_line, start_column};
    } else if (std::isdigit(current_char)) {
      int start = pos;
      int start_line = line;
      int start_column = column;
      while (!at_end(input) &&
             (std::isdigit(input[pos]) || input[pos] == '_')) {
        consume(input);
      }
      if (!at_end(input) && input[pos] == '.' &&
          static_cast<size_t>(pos + 1) < input.size() &&
          std::isdigit(input[pos + 1])) {
        consume(input); // consume the '.'
        while (!at_end(input) &&
               (std::isdigit(input[pos]) || input[pos] == '_')) {
          consume(input);
        }
      }
      return Token{TokenType::NUMBER, input.substr(start, pos - start),
                   start_line, start_column};
    } else if (current_char == '"') {
      int start = pos;
      int start_line = line;
      int start_column = column;
      consume(input); // Skip opening quote

      while (!at_end(input) && input[pos] != '"') {
        if (input[pos] == '\\' && static_cast<size_t>(pos + 1) < input.size()) {
          consume(input, 2); // Skip escape sequence
        } else {
          consume(input);
        }
      }

      if (!at_end(input))
        consume(input); // Skip the closing quote

      return Token{TokenType::String, input.substr(start, pos - start),
                   start_line, start_column};
    } else if (is_delimiter(current_char)) {
      int start_line = line;
      int start_column = column;
      Token token{TokenType::Delimiter, std::string_view(&input[pos], 1),
                  start_line, start_column};
      consume(input);
      return token;
    } else if (is_separator(current_char)) {
      int start_line = line;
      int start_column = column;
      Token token{TokenType::Separator, std::string_view(&input[pos], 1),
                  start_line, start_column};
      consume(input);
      return token;
    } else if (is_operator(current_char)) {
      int start = pos;
      int start_line = line;
      int start_column = column;

      if (current_char == '=' && static_cast<size_t>(pos + 1) < input.size() &&
          input[pos + 1] == '=') {
        consume(input, 2); // consume "=="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '!' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume "!="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '<' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume "<="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '>' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume ">="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '-' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '>') {
        consume(input, 2); // consume "->"
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '=' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '>') {
        consume(input, 2); // consume "=>"
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '|' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '|') {
        consume(input, 2); // consume "||"
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else {
        Token token{TokenType::Operator, std::string_view(&input[pos], 1),
                    start_line, start_column};
        consume(input);
        return token;
      }
    } else {
      assert(false && "Unexpected character in input");
    }
  }
  return Token{TokenType::End, "", line, column};
}