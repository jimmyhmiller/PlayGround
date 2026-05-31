// SOURCE:      export default arbitrary_expression;
// SOURCE-NEXT: export { identifier_1, "string_literal" } from "foo";
// SOURCE-NEXT: export let identifier_2;
// SOURCE-NEXT: export * from "foo";
// SOURCE-EMPTY:
// SOURCE-NEXT: // TODO(b/182441574): Fix AST error.
// SOURCE-NEXT: // export * as identifier_3 from "foo";
