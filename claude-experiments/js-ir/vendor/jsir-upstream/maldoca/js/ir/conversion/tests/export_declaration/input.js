export default arbitrary_expression;

export {identifier_1, "string_literal"} from "foo";

export let identifier_2;

export * from "foo";

// TODO(b/182441574): Fix AST error.
// export * as identifier_3 from "foo";
