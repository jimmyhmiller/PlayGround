#include <iostream>
#define CABIN_TEST
#include "src/main.cc"
int main() { Tokenizer t; std::string input = "{"; Token tok = t.next_token(input); std::cout << "Type: " << static_cast<int>(tok.type) << ", Value: '" << tok.value << "'" << std::endl; return 0; }
