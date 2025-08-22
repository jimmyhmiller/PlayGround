# TODO

## Parser Issues
- [ ] Fix if statement parsing inside function bodies
  - Currently if statements are only parsed at the top level in `build_from_reader_root`
  - Need to handle if statements within blocks/function bodies
  - Example that fails: `fn factorial(n) { if n <= 1 { 1 } { n * factorial(n - 1) } }`

## Completed
- [x] Add function definition support to AST: fn(x, y) { body }
- [x] Add named function support: fn name(x, y) { body }
- [x] Move JSON implementation from lang.cc to separate ast_json.cc file
- [x] Create ast-to-json command line utility