# Parse Error Categories

**Total Failures:** 1,119
**Unique Categories:** 65

---

## 1. UnexpectedToken|COMMA (310 failures, 27.70%)

**Examples:**
- `compare-array-sparse.js`: Unexpected token: Token[type=COMMA, lexeme=,, literal=null, line=11, column=21, position=285, endPosition=286]
- `extensibility-02.js`: Unexpected token: Token[type=COMMA, lexeme=,, literal=null, line=19, column=37, position=466, endPosition=467]
- `extensibility-01.js`: Unexpected token: Token[type=COMMA, lexeme=,, literal=null, line=48, column=37, position=1084, endPosition=1085]

---

## 2. UnexpectedToken|DOT_DOT_DOT (243 failures, 21.72%)

**Examples:**
- `compare-array-arguments.js`: Unexpected token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=38, column=2, position=1087, endPosition=1090]
- `iterator-next-non-object.js`: Unexpected token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=39, column=37, position=817, endPosition=820]
- `slice-species.js`: Unexpected token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=41, column=35, position=1147, endPosition=1150]

---

## 3. UnexpectedToken|STAR (107 failures, 9.56%)

**Examples:**
- `sort_stable.js`: Unexpected token: Token[type=STAR, lexeme=*, literal=null, line=12, column=20, position=334, endPosition=335]
- `constructor-byteoffsets-bounds.js`: Unexpected token: Token[type=STAR, lexeme=*, literal=null, line=18, column=49, position=555, endPosition=556]
- `call-next-on-iterator-while-iterating.js`: Unexpected token: Token[type=STAR, lexeme=*, literal=null, line=15, column=40, position=360, endPosition=361]

---

## 4. UnexpectedToken|RBRACKET (97 failures, 8.67%)

**Examples:**
- `syntax.js`: Unexpected token: Token[type=RBRACKET, lexeme=], literal=null, line=39, column=22, position=941, endPosition=942]
- `yield-spread-arr-single.js`: Unexpected token: Token[type=RBRACKET, lexeme=], literal=null, line=27, column=17, position=701, endPosition=702]
- `yield-spread-arr-multiple.js`: Unexpected token: Token[type=RBRACKET, lexeme=], literal=null, line=30, column=23, position=748, endPosition=749]

---

## 5. ExpectedToken|IDENTIFIER|Expected '{' before class body (61 failures, 5.45%)

**Examples:**
- `newer-type-functions-caller-arguments.js`: Expected Expected '{' before class body at token: Token[type=IDENTIFIER, lexeme=Object, literal=null, line=48, column=18, position=1454, endPosition=1460]
- `superPropChains.js`: Expected Expected '{' before class body at token: Token[type=IDENTIFIER, lexeme=bootlegMiddle, literal=null, line=40, column=18, position=891, endPosition=904]
- `superCallBaseInvoked.js`: Expected Expected '{' before class body at token: Token[type=IDENTIFIER, lexeme=base, literal=null, line=39, column=23, position=1045, endPosition=1049]

---

## 6. ExpectedToken|RPAREN|Expected ';' after for loop initializer (41 failures, 3.66%)

**Examples:**
- `for-inof-name-iteration-expression-contains-index-string.js`: Expected Expected ';' after for loop initializer at token: Token[type=RPAREN, lexeme=), literal=null, line=16, column=15, position=388, endPosition=389]
- `destructuring-for-inof-__proto__.js`: Expected Expected ';' after for loop initializer at token: Token[type=RPAREN, lexeme=), literal=null, line=47, column=47, position=1251, endPosition=1252]
- `superPropFor.js`: Expected Expected ';' after for loop initializer at token: Token[type=RPAREN, lexeme=), literal=null, line=12, column=49, position=296, endPosition=297]

---

## 7. UnexpectedToken|QUESTION (30 failures, 2.68%)

**Examples:**
- `optional-chain.js`: Unexpected token: Token[type=QUESTION, lexeme=?, literal=null, line=122, column=58, position=4833, endPosition=4834]
- `nullish-coalescing.js`: Unexpected token: Token[type=QUESTION, lexeme=?, literal=null, line=32, column=22, position=786, endPosition=787]
- `sort_modifications.js`: Unexpected token: Token[type=QUESTION, lexeme=?, literal=null, line=21, column=28, position=413, endPosition=414]

---

## 8. ExpectedToken|DOT_DOT_DOT|identifier in variable declaration (26 failures, 2.32%)

**Examples:**
- `rest-parameter-names.js`: Expected identifier in variable declaration at token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=15, column=10, position=326, endPosition=329]
- `check-fn-after-getting-iterator.js`: Expected identifier in variable declaration at token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=16, column=35, position=384, endPosition=387]
- `check-fn-after-getting-iterator.js`: Expected identifier in variable declaration at token: Token[type=DOT_DOT_DOT, lexeme=..., literal=null, line=16, column=35, position=384, endPosition=387]

---

## 9. ExpectedToken|TEMPLATE_LITERAL|Expected ';' after expression (22 failures, 1.97%)

**Examples:**
- `callexpression-templateliteral.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_LITERAL, lexeme=, literal=, line=18, column=32, position=500, endPosition=500]
- `tv-null-character-escape-sequence.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_LITERAL, lexeme=\0, literal= , line=18, column=2, position=505, endPosition=507]
- `tv-zwnbsp.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_LITERAL, lexeme=\uFEFFtest, literal=﻿test, line=22, column=2, position=601, endPosition=611]

---

## 10. ExpectedToken|IDENTIFIER|Expected '(' after 'for' (20 failures, 1.79%)

**Examples:**
- `absent-value-not-passed.js`: Expected Expected '(' after 'for' at token: Token[type=IDENTIFIER, lexeme=await, literal=null, line=37, column=6, position=771, endPosition=776]
- `return-null.js`: Expected Expected '(' after 'for' at token: Token[type=IDENTIFIER, lexeme=await, literal=null, line=47, column=6, position=1124, endPosition=1129]
- `for-await-next-rejected-promise-close.js`: Expected Expected '(' after 'for' at token: Token[type=IDENTIFIER, lexeme=await, literal=null, line=62, column=8, position=2291, endPosition=2296]

---

## 11. ExpectedToken|TEMPLATE_HEAD|Expected ';' after expression (16 failures, 1.43%)

**Examples:**
- `tv-template-head.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_HEAD, lexeme=, literal=, line=21, column=2, position=663, endPosition=663]
- `tv-template-middle.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_HEAD, lexeme=, literal=, line=19, column=2, position=578, endPosition=578]
- `tv-template-tail.js`: Expected Expected ';' after expression at token: Token[type=TEMPLATE_HEAD, lexeme=, literal=, line=21, column=2, position=660, endPosition=660]

---

## 12. ExpectedToken|SEMICOLON|property name after '.' (15 failures, 1.34%)

**Examples:**
- `15.4.4.16-7-b-1.js`: Expected property name after '.' at token: Token[type=SEMICOLON, lexeme=;, literal=null, line=11, column=16, position=305, endPosition=306]
- `obj-rest-non-string-computed-property-1e0.js`: Expected property name after '.' at token: Token[type=SEMICOLON, lexeme=;, literal=null, line=28, column=10, position=1242, endPosition=1243]
- `obj-rest-non-string-computed-property-1dot.js`: Expected property name after '.' at token: Token[type=SEMICOLON, lexeme=;, literal=null, line=28, column=10, position=1260, endPosition=1261]

---

## 13. UnexpectedToken|COLON (8 failures, 0.71%)

**Examples:**
- `value-yield-non-strict-escaped.js`: Unexpected token: Token[type=COLON, lexeme=:, literal=null, line=11, column=10, position=329, endPosition=330]
- `value-await-module.js`: Unexpected token: Token[type=COLON, lexeme=:, literal=null, line=20, column=5, position=490, endPosition=491]
- `value-await-module-escaped.js`: Unexpected token: Token[type=COLON, lexeme=:, literal=null, line=21, column=10, position=571, endPosition=572]

---

## 14. UnexpectedToken|ASSIGN (7 failures, 0.63%)

**Examples:**
- `yield-as-identifier-in-nested-function.js`: Unexpected token: Token[type=ASSIGN, lexeme==, literal=null, line=16, column=10, position=392, endPosition=393]
- `yield-as-identifier-in-nested-function.js`: Unexpected token: Token[type=ASSIGN, lexeme==, literal=null, line=16, column=10, position=398, endPosition=399]
- `yield-as-identifier-in-nested-function.js`: Unexpected token: Token[type=ASSIGN, lexeme==, literal=null, line=17, column=12, position=401, endPosition=402]

---

## 15. ExpectedToken|RBRACE|identifier in import specifier (7 failures, 0.63%)

**Examples:**
- `instn-once.js`: Expected identifier in import specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=27, column=8, position=917, endPosition=918]
- `instn-named-bndng-trlng-comma.js`: Expected identifier in import specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=53, column=18, position=1888, endPosition=1889]
- `eval-rqstd-order.js`: Expected identifier in import specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=24, column=8, position=785, endPosition=786]

---

## 16. UnexpectedToken|GT (7 failures, 0.63%)

**Examples:**
- `comment-single-line-html-close.js`: Unexpected token: Token[type=GT, lexeme=>, literal=null, line=16, column=2, position=366, endPosition=367]
- `comment-multi-line-html-close.js`: Unexpected token: Token[type=GT, lexeme=>, literal=null, line=17, column=5, position=370, endPosition=371]
- `single-line-html-close.js`: Unexpected token: Token[type=GT, lexeme=>, literal=null, line=25, column=2, position=663, endPosition=664]

---

## 17. ExpectedToken|TEMPLATE_LITERAL|Expected ')' after arguments (6 failures, 0.54%)

**Examples:**
- `tagged-template-constant-folding.js`: Expected Expected ')' after arguments at token: Token[type=TEMPLATE_LITERAL, lexeme=, literal=, line=19, column=22, position=464, endPosition=464]
- `illegal-in-identifier-context.js`: Expected Expected ')' after arguments at token: Token[type=TEMPLATE_LITERAL, lexeme=var #\u0061;, literal=var #a;, line=20, column=48, position=694, endPosition=706]
- `zero-literal-segments.js`: Expected Expected ')' after arguments at token: Token[type=TEMPLATE_LITERAL, lexeme=, literal=, line=9, column=27, position=262, endPosition=262]

---

## 18. ExpectedToken|CLASS|Expected '{' before class body (6 failures, 0.54%)

**Examples:**
- `derivedConstructorArrowEvalSuperCall.js`: Expected Expected '{' before class body at token: Token[type=CLASS, lexeme=class, literal=null, line=9, column=18, position=205, endPosition=210]
- `derivedConstructorArrowEvalGetThis.js`: Expected Expected '{' before class body at token: Token[type=CLASS, lexeme=class, literal=null, line=9, column=18, position=205, endPosition=210]
- `derivedConstructorArrowEvalNestedSuperCall.js`: Expected Expected '{' before class body at token: Token[type=CLASS, lexeme=class, literal=null, line=9, column=18, position=205, endPosition=210]

---

## 19. ExpectedToken|IDENTIFIER|Expected ';' after expression (6 failures, 0.54%)

**Examples:**
- `S7.9_A5.2_T1.js`: Expected Expected ';' after expression at token: Token[type=IDENTIFIER, lexeme=y, literal=null, line=14, column=2, position=332, endPosition=333]
- `S7.9_A5.6_T1.js`: Expected Expected ';' after expression at token: Token[type=IDENTIFIER, lexeme=y, literal=null, line=27, column=2, position=812, endPosition=813]
- `S7.9.2_A1_T5.js`: Expected Expected ';' after expression at token: Token[type=IDENTIFIER, lexeme=c, literal=null, line=15, column=2, position=401, endPosition=402]

---

## 20. UnexpectedToken|OF (5 failures, 0.45%)

**Examples:**
- `invoked-as-func.js`: Unexpected token: Token[type=OF, lexeme=of, literal=null, line=21, column=2, position=494, endPosition=496]
- `invoked-as-func.js`: Unexpected token: Token[type=OF, lexeme=of, literal=null, line=22, column=4, position=548, endPosition=550]
- `invoked-as-func.js`: Unexpected token: Token[type=OF, lexeme=of, literal=null, line=22, column=4, position=568, endPosition=570]

---

## 21. ExpectedToken|DOT|property name in class body (4 failures, 0.36%)

**Examples:**
- `literal-numeric-leading-decimal.js`: Expected property name in class body at token: Token[type=DOT, lexeme=., literal=null, line=31, column=6, position=975, endPosition=976]
- `literal-numeric-leading-decimal.js`: Expected property name in class body at token: Token[type=DOT, lexeme=., literal=null, line=33, column=13, position=1012, endPosition=1013]
- `literal-numeric-leading-decimal.js`: Expected property name in class body at token: Token[type=DOT, lexeme=., literal=null, line=31, column=6, position=980, endPosition=981]

---

## 22. ExpectedToken|DOT|'from' after import specifiers (4 failures, 0.36%)

**Examples:**
- `goal-script.js`: Expected 'from' after import specifiers at token: Token[type=DOT, lexeme=., literal=null, line=18, column=6, position=477, endPosition=478]
- `goal-module.js`: Expected 'from' after import specifiers at token: Token[type=DOT, lexeme=., literal=null, line=14, column=6, position=429, endPosition=430]
- `goal-module-nested-function.js`: Expected 'from' after import specifiers at token: Token[type=DOT, lexeme=., literal=null, line=15, column=8, position=461, endPosition=462]

---

## 23. ExpectedToken|LPAREN|Expected '{' before class body (3 failures, 0.27%)

**Examples:**
- `strictExecution.js`: Expected Expected '{' before class body at token: Token[type=LPAREN, lexeme=(, literal=null, line=38, column=26, position=1076, endPosition=1077]
- `heritage-async-arrow-function.js`: Expected Expected '{' before class body at token: Token[type=LPAREN, lexeme=(, literal=null, line=29, column=24, position=967, endPosition=968]
- `heritage-arrow-function.js`: Expected Expected '{' before class body at token: Token[type=LPAREN, lexeme=(, literal=null, line=29, column=24, position=961, endPosition=962]

---

## 24. ExpectedToken|YIELD|Expected '(' after function (3 failures, 0.27%)

**Examples:**
- `yield-as-function-expression-binding-identifier.js`: Expected Expected '(' after function at token: Token[type=YIELD, lexeme=yield, literal=null, line=15, column=12, position=364, endPosition=369]
- `yield-as-function-expression-binding-identifier.js`: Expected Expected '(' after function at token: Token[type=YIELD, lexeme=yield, literal=null, line=15, column=12, position=370, endPosition=375]
- `yield-as-function-expression-binding-identifier.js`: Expected Expected '(' after function at token: Token[type=YIELD, lexeme=yield, literal=null, line=16, column=14, position=371, endPosition=376]

---

## 25. ExpectedToken|ASSIGN|identifier in variable declaration (3 failures, 0.27%)

**Examples:**
- `using-declaring-let-split-across-two-lines.js`: Expected identifier in variable declaration at token: Token[type=ASSIGN, lexeme==, literal=null, line=16, column=6, position=432, endPosition=433]
- `await-using-declaring-let-split-across-two-lines.js`: Expected identifier in variable declaration at token: Token[type=ASSIGN, lexeme==, literal=null, line=17, column=6, position=513, endPosition=514]
- `head-lhs-let.js`: Expected identifier in variable declaration at token: Token[type=ASSIGN, lexeme==, literal=null, line=25, column=4, position=682, endPosition=683]

---

## 26. UnexpectedToken|ARROW (3 failures, 0.27%)

**Examples:**
- `head-init-async-of.js`: Unexpected token: Token[type=ARROW, lexeme==>, literal=null, line=13, column=14, position=351, endPosition=353]
- `arrowparameters-bindingidentifier-no-yield.js`: Unexpected token: Token[type=ARROW, lexeme==>, literal=null, line=20, column=15, position=383, endPosition=385]
- `arrowparameters-bindingidentifier-yield.js`: Unexpected token: Token[type=ARROW, lexeme==>, literal=null, line=14, column=15, position=329, endPosition=331]

---

## 27. ExpectedToken|FALSE|property name after '.' (3 failures, 0.27%)

**Examples:**
- `accessor-name-computed-in.js`: Expected property name after '.' at token: Token[type=FALSE, lexeme=false, literal=false, line=24, column=14, position=707, endPosition=712]
- `accessor-name-inst-computed-in.js`: Expected property name after '.' at token: Token[type=FALSE, lexeme=false, literal=false, line=26, column=22, position=893, endPosition=898]
- `accessor-name-static-computed-in.js`: Expected property name after '.' at token: Token[type=FALSE, lexeme=false, literal=false, line=28, column=12, position=916, endPosition=921]

---

## 28. ExpectedToken|STRING|Expected ';' after expression (3 failures, 0.27%)

**Examples:**
- `multi-line-asi-carriage-return.js`: Expected Expected ';' after expression at token: Token[type=STRING, lexeme='', literal=, line=17, column=7, position=539, endPosition=541]
- `multi-line-asi-paragraph-separator.js`: Expected Expected ';' after expression at token: Token[type=STRING, lexeme='', literal=, line=17, column=7, position=542, endPosition=544]
- `multi-line-asi-line-separator.js`: Expected Expected ';' after expression at token: Token[type=STRING, lexeme='', literal=, line=17, column=7, position=538, endPosition=540]

---

## 29. ExpectedToken|EOF|Expected ';' after do-while statement (3 failures, 0.27%)

**Examples:**
- `S7.9_A9_T9.js`: Expected Expected ';' after do-while statement at token: Token[type=EOF, lexeme=, literal=null, line=13, column=0, position=302, endPosition=302]
- `S7.9_A9_T1.js`: Expected Expected ';' after do-while statement at token: Token[type=EOF, lexeme=, literal=null, line=13, column=0, position=304, endPosition=304]
- `S7.9_A9_T5.js`: Expected Expected ';' after do-while statement at token: Token[type=EOF, lexeme=, literal=null, line=15, column=0, position=321, endPosition=321]

---

## 30. ExpectedToken|RBRACE|identifier in export specifier (3 failures, 0.27%)

**Examples:**
- `parse-export-empty.js`: Expected identifier in export specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=27, column=7, position=787, endPosition=788]
- `instn-iee-trlng-comma.js`: Expected identifier in export specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=15, column=13, position=490, endPosition=491]
- `instn-resolve-empty-export.js`: Expected identifier in export specifier at token: Token[type=RBRACE, lexeme=}, literal=null, line=40, column=8, position=1107, endPosition=1108]

---

## 31. ExpectedToken|TRUE|property key (2 failures, 0.18%)

**Examples:**
- `stringify-replacer-array-skipped-element.js`: Expected property key at token: Token[type=TRUE, lexeme=true, literal=true, line=17, column=40, position=474, endPosition=478]
- `S11.1.5_A4.1.js`: Expected property key at token: Token[type=TRUE, lexeme=true, literal=true, line=11, column=14, position=325, endPosition=329]

---

## 32. ExpectedToken|NULL|Expected '{' before class body (2 failures, 0.18%)

**Examples:**
- `superCallInvalidBase.js`: Expected Expected '{' before class body at token: Token[type=NULL, lexeme=null, literal=null, line=14, column=49, position=348, endPosition=352]
- `classHeritage.js`: Expected Expected '{' before class body at token: Token[type=NULL, lexeme=null, literal=null, line=25, column=36, position=850, endPosition=854]

---

## 33. ExpectedToken|EOF|Expected ';' after import (2 failures, 0.18%)

**Examples:**
- `parent-tla_FIXTURE.js`: Expected Expected ';' after import at token: Token[type=EOF, lexeme=, literal=null, line=5, column=0, position=165, endPosition=165]
- `module-graphs-parent-tla_FIXTURE.js`: Expected Expected ';' after import at token: Token[type=EOF, lexeme=, literal=null, line=5, column=0, position=165, endPosition=165]

---

## 34. ExpectedToken|TEMPLATE_HEAD|Expected ')' after arguments (2 failures, 0.18%)

**Examples:**
- `template-substitutions-are-appended-on-same-index.js`: Expected Expected ')' after arguments at token: Token[type=TEMPLATE_HEAD, lexeme=1, literal=1, line=32, column=27, position=1228, endPosition=1229]
- `special-characters.js`: Expected Expected ')' after arguments at token: Token[type=TEMPLATE_HEAD, lexeme=\u0065\`\r\r\n\n, literal=e`

, line=11, column=12, position=339, endPosition=355]

---

## 35. UnexpectedToken|RPAREN (2 failures, 0.18%)

**Examples:**
- `constructor.js`: Unexpected token: Token[type=RPAREN, lexeme=), literal=null, line=35, column=0, position=655, endPosition=656]
- `non-iterable-input-does-not-use-array-prototype.js`: Unexpected token: Token[type=RPAREN, lexeme=), literal=null, line=27, column=2, position=949, endPosition=950]

---

## 36. ExpectedToken|NE_STRICT|property name after '.' (2 failures, 0.18%)

**Examples:**
- `S7.8.3_A3.1_T2.js`: Expected property name after '.' at token: Token[type=NE_STRICT, lexeme=!==, literal=null, line=11, column=8, position=302, endPosition=305]
- `S7.8.3_A3.1_T1.js`: Expected property name after '.' at token: Token[type=NE_STRICT, lexeme=!==, literal=null, line=11, column=7, position=289, endPosition=292]

---

## 37. UnexpectedToken|LET (2 failures, 0.18%)

**Examples:**
- `head-var-bound-names-let.js`: Unexpected token: Token[type=LET, lexeme=let, literal=null, line=14, column=19, position=370, endPosition=373]
- `head-var-bound-names-let.js`: Unexpected token: Token[type=LET, lexeme=let, literal=null, line=14, column=19, position=380, endPosition=383]

---

## 38. ExpectedToken|IDENTIFIER|Expected ';' after for loop initializer (2 failures, 0.18%)

**Examples:**
- `identifier-let-allowed-as-lefthandside-expression-strict.js`: Expected Expected ';' after for loop initializer at token: Token[type=IDENTIFIER, lexeme=o, literal=null, line=16, column=12, position=390, endPosition=391]
- `head-lhs-let.js`: Expected Expected ';' after for loop initializer at token: Token[type=IDENTIFIER, lexeme=obj, literal=null, line=27, column=13, position=673, endPosition=676]

---

## 39. ExpectedToken|NULL|property key (2 failures, 0.18%)

**Examples:**
- `ident-name-reserved-word-literal-prop-name.js`: Expected property key at token: Token[type=NULL, lexeme=null, literal=null, line=11, column=4, position=273, endPosition=277]
- `S11.1.5_A4.2.js`: Expected property key at token: Token[type=NULL, lexeme=null, literal=null, line=11, column=14, position=325, endPosition=329]

---

## 40. ExpectedToken|TEMPLATE_HEAD|Expected ';' after return statement (2 failures, 0.18%)

**Examples:**
- `tco-call.js`: Expected Expected ';' after return statement at token: Token[type=TEMPLATE_HEAD, lexeme=, literal=, line=21, column=17, position=523, endPosition=523]
- `tco-member.js`: Expected Expected ';' after return statement at token: Token[type=TEMPLATE_HEAD, lexeme=, literal=, line=18, column=12, position=480, endPosition=480]

---

## 41. ExpectedToken|IDENTIFIER|Expected ';' after export default (2 failures, 0.18%)

**Examples:**
- `export-default-asyncfunction-declaration-binding.js`: Expected Expected ';' after export default at token: Token[type=IDENTIFIER, lexeme=A, literal=null, line=18, column=0, position=496, endPosition=497]
- `export-default-asyncgenerator-declaration-binding.js`: Expected Expected ';' after export default at token: Token[type=IDENTIFIER, lexeme=AG, literal=null, line=18, column=0, position=501, endPosition=503]

---

## 42. UnexpectedToken|LT (2 failures, 0.18%)

**Examples:**
- `comment-single-line-html-open.js`: Unexpected token: Token[type=LT, lexeme=<, literal=null, line=16, column=0, position=363, endPosition=364]
- `single-line-html-open.js`: Unexpected token: Token[type=LT, lexeme=<, literal=null, line=22, column=0, position=522, endPosition=523]

---

## 43. ExpectedToken|DOT|':' (1 failures, 0.09%)

**Examples:**
- `object-literal-accessor-property-name.js`: Expected ':' at token: Token[type=DOT, lexeme=., literal=null, line=16, column=8, position=349, endPosition=350]

---

## 44. UnexpectedToken|RBRACE (1 failures, 0.09%)

**Examples:**
- `derivedConstructorTDZReturnAliasedTry.js`: Unexpected token: Token[type=RBRACE, lexeme=}, literal=null, line=16, column=4, position=331, endPosition=332]

---

## 45. ExpectedToken|YIELD|function name (1 failures, 0.09%)

**Examples:**
- `yield-as-generator-declaration-binding-identifier.js`: Expected function name at token: Token[type=YIELD, lexeme=yield, literal=null, line=14, column=10, position=345, endPosition=350]

---

## 46. ExpectedToken|RBRACE|Expected ';' after do-while statement (1 failures, 0.09%)

**Examples:**
- `tco-body.js`: Expected Expected ';' after do-while statement at token: Token[type=RBRACE, lexeme=}, literal=null, line=20, column=0, position=497, endPosition=498]

---

## 47. ExpectedToken|LBRACE|Expected ';' after for loop initializer (1 failures, 0.09%)

**Examples:**
- `identifier-let-allowed-as-lefthandside-expression-not-strict.js`: Expected Expected ';' after for loop initializer at token: Token[type=LBRACE, lexeme={, literal=null, line=12, column=12, position=441, endPosition=442]

---

## 48. ExpectedToken|FUNCTION|Expected '{' before class body (1 failures, 0.09%)

**Examples:**
- `arguments-callee.js`: Expected Expected '{' before class body at token: Token[type=FUNCTION, lexeme=function, literal=null, line=8, column=22, position=221, endPosition=229]

---

## 49. ExpectedToken|NULL|property name after '.' (1 failures, 0.09%)

**Examples:**
- `ident-name-reserved-word-literal-memberexpr.js`: Expected property name after '.' at token: Token[type=NULL, lexeme=null, literal=null, line=12, column=11, position=285, endPosition=289]

---

## 50. ExpectedToken|NULL|':' (1 failures, 0.09%)

**Examples:**
- `ident-name-reserved-word-literal-accessor.js`: Expected ':' at token: Token[type=NULL, lexeme=null, literal=null, line=13, column=8, position=301, endPosition=305]

---

## 51. ExpectedToken|TRUE|property name after '.' (1 failures, 0.09%)

**Examples:**
- `optional-chain-prod-expression.js`: Expected property name after '.' at token: Token[type=TRUE, lexeme=true, literal=true, line=16, column=4, position=341, endPosition=345]

---

## 52. ExpectedToken|NUMBER|property name after '?.' (1 failures, 0.09%)

**Examples:**
- `punctuator-decimal-lookahead.js`: Expected property name after '?.' at token: Token[type=NUMBER, lexeme=30, literal=30, line=14, column=21, position=393, endPosition=395]

---

## 53. UnexpectedToken|STAR_ASSIGN (1 failures, 0.09%)

**Examples:**
- `exp-assignment-operator.js`: Unexpected token: Token[type=STAR_ASSIGN, lexeme=*=, literal=null, line=25, column=23, position=871, endPosition=873]

---

## 54. ExpectedToken|WHILE|Expected ';' after export (1 failures, 0.09%)

**Examples:**
- `await-import-evaluation_FIXTURE.js`: Expected Expected ';' after export at token: Token[type=WHILE, lexeme=while, literal=null, line=9, column=0, position=197, endPosition=202]

---

## 55. ExpectedToken|TEMPLATE_LITERAL|Expected ')' after import source (1 failures, 0.09%)

**Examples:**
- `tagged-function-call.js`: Expected Expected ')' after import source at token: Token[type=TEMPLATE_LITERAL, lexeme=./module-code-other_FIXTURE.js, literal=./module-code-other_FIXTURE.js, line=29, column=31, position=984, endPosition=1014]

---

## 56. ExpectedToken|LBRACKET|property name after '.' (1 failures, 0.09%)

**Examples:**
- `S11.2.1_A3_T2.js`: Expected property name after '.' at token: Token[type=LBRACKET, lexeme=[, literal=null, line=29, column=6, position=779, endPosition=780]

---

## 57. UnexpectedToken|TEMPLATE_MIDDLE (1 failures, 0.09%)

**Examples:**
- `rhs-template-middle.js`: Unexpected token: Token[type=TEMPLATE_MIDDLE, lexeme=3, literal=3, line=16, column=19, position=515, endPosition=516]

---

## 58. RuntimeException (1 failures, 0.09%)

**Examples:**
- `invalid-escape-sequences.js`: Unexpected character: \ (U+005C)

---

## 59. ExpectedToken|TEMPLATE_LITERAL|Expected ';' after variable declaration (1 failures, 0.09%)

**Examples:**
- `chained-application.js`: Expected Expected ';' after variable declaration at token: Token[type=TEMPLATE_LITERAL, lexeme=x, literal=x, line=17, column=16, position=454, endPosition=455]

---

## 60. ExpectedToken|TRUE|Expected ';' after do-while statement (1 failures, 0.09%)

**Examples:**
- `S7.9_A9_T2.js`: Expected Expected ';' after do-while statement at token: Token[type=TRUE, lexeme=true, literal=true, line=12, column=0, position=303, endPosition=307]

---

## 61. ExpectedToken|IDENTIFIER|Expected ';' after do-while statement (1 failures, 0.09%)

**Examples:**
- `do-while-same-line.js`: Expected Expected ';' after do-while statement at token: Token[type=IDENTIFIER, lexeme=x, literal=null, line=19, column=21, position=751, endPosition=752]

---

## 62. ExpectedToken|DEFAULT|identifier after 'as' (1 failures, 0.09%)

**Examples:**
- `export-star-as-dflt.js`: Expected identifier after 'as' at token: Token[type=DEFAULT, lexeme=default, literal=null, line=21, column=12, position=734, endPosition=741]

---

## 63. ExpectedToken|VAR|Expected ';' after import (1 failures, 0.09%)

**Examples:**
- `instn-uniq-env-rec.js`: Expected Expected ';' after import at token: Token[type=VAR, lexeme=var, literal=null, line=21, column=0, position=558, endPosition=561]

---

## 64. UnexpectedToken|SEMICOLON (1 failures, 0.09%)

**Examples:**
- `no-operand.js`: Unexpected token: Token[type=SEMICOLON, lexeme=;, literal=null, line=17, column=5, position=320, endPosition=321]

---

## 65. ExpectedToken|IDENTIFIER|string literal after 'from' (1 failures, 0.09%)

**Examples:**
- `import-source-binding-name_FIXTURE.js`: Expected string literal after 'from' at token: Token[type=IDENTIFIER, lexeme=from, literal=null, line=18, column=12, position=704, endPosition=708]

---
