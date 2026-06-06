// A small parser for a reasonable JS subset, producing the AST `jsmix` consumes.
// Front-end only (plain Node) -- it turns a JS interpreter's SOURCE TEXT into the
// AST that the partial evaluator works on. Supports: function declarations;
// statements return/if-else/var/block/expr; expressions with ?:, comparison,
// + - *, calls, member a[b]/a.b, array literals, number/string/bool literals,
// identifiers, and function/arrow expressions (for reducers).
//
// AST node shapes (arrays, tagged at [0]):
//   ["lit", value]                          number/string/bool literal
//   ["var", name]
//   ["arr", [elems]]                        array literal
//   ["bin", op, a, b]                        op in + - * === !== < > <= >=
//   ["idx", a, b]                            a[b]
//   ["dot", a, name]                         a.name
//   ["call", callee, [args]]                 callee(args)   (callee is an AST)
//   ["if", c, t, e]                          e may be null (expression form / stmt)
//   ["cond", c, t, e]                        ternary
//   ["fun", [params], body]                  function/arrow expression; body = stmts or single expr
//   ["return", e]
//   ["var", name, e]   (statement)           NOTE: distinguish from ["var",name] ref by length
//   ["block", [stmts]]
// A function declaration -> ["fundecl", name, [params], [stmts]]
// A program -> list of fundecls.

function tokenize(src) {
  var toks = [];
  var i = 0;
  var n = src.length;
  function isIdStart(c){ return (c>="a"&&c<="z")||(c>="A"&&c<="Z")||c==="_"||c==="$"; }
  function isId(c){ return isIdStart(c)||(c>="0"&&c<="9"); }
  function isDig(c){ return c>="0"&&c<="9"; }
  while (i < n) {
    var c = src[i];
    if (c===" "||c==="\t"||c==="\n"||c==="\r"){ i++; continue; }
    if (c==="/"&&src[i+1]==="/"){ while(i<n&&src[i]!=="\n")i++; continue; }
    if (isIdStart(c)){ var j=i; while(j<n&&isId(src[j]))j++; toks.push(["id",src.slice(i,j)]); i=j; continue; }
    if (isDig(c)||(c==="-"&&isDig(src[i+1])&&false)){ var j2=i; while(j2<n&&isDig(src[j2]))j2++; toks.push(["num",parseInt(src.slice(i,j2),10)]); i=j2; continue; }
    if (c==='"'||c==="'"){ var q=c; var j3=i+1; var s=""; while(j3<n&&src[j3]!==q){ s+=src[j3]; j3++; } toks.push(["str",s]); i=j3+1; continue; }
    // multi-char operators
    var three=src.slice(i,i+3), two=src.slice(i,i+2);
    if (three==="===" || three==="!=="){ toks.push(["op",three]); i+=3; continue; }
    if (two==="=>"||two==="<="||two===">="||two==="=="||two==="!="){ toks.push(["op",two]); i+=2; continue; }
    toks.push(["op",c]); i++;
  }
  toks.push(["eof",""]);
  return toks;
}

function parseProgram(src) {
  var toks = tokenize(src);
  var p = 0;
  function peek(){ return toks[p]; }
  function next(){ return toks[p++]; }
  function isOp(v){ return toks[p][0]==="op"&&toks[p][1]===v; }
  function isId(v){ return toks[p][0]==="id"&&toks[p][1]===v; }
  function eatOp(v){ if(!isOp(v)) throw new Error("expected '"+v+"' got "+JSON.stringify(toks[p])+" at "+p); p++; }
  function eatId(v){ if(!isId(v)) throw new Error("expected id "+v); p++; }

  function parseParams(){ // assumes '(' consumed; returns names, consumes ')'
    var ps=[];
    if(!isOp(")")){ ps.push(next()[1]); while(isOp(",")){ next(); ps.push(next()[1]); } }
    eatOp(")");
    return ps;
  }
  function parsePrimary(){
    var t=peek();
    if(t[0]==="num"){ next(); return ["lit", t[1]]; }
    if(t[0]==="str"){ next(); return ["lit", t[1]]; }
    if(t[0]==="id"){
      if(t[1]==="true"){ next(); return ["lit", true]; }
      if(t[1]==="false"){ next(); return ["lit", false]; }
      if(t[1]==="function"){ next(); eatOp("("); var ps=parseParams(); var body=parseBlock(); return ["fun", ps, body]; }
      next(); return ["var", t[1]];
    }
    if(isOp("(")){
      // could be arrow params or parenthesized expr
      var save=p; next();
      // try arrow: ( params ) =>
      if(tryArrowParams()){ var ps2=parseParamsAfterParen(save); var b2=parseArrowBody(); return ["fun", ps2, b2]; }
      p=save; next(); var e=parseExpr(); eatOp(")"); return e;
    }
    if(isOp("[")){ next(); var els=[]; if(!isOp("]")){ els.push(parseExpr()); while(isOp(",")){ next(); els.push(parseExpr()); } } eatOp("]"); return ["arr", els]; }
    throw new Error("unexpected token "+JSON.stringify(t)+" at "+p);
  }
  function tryArrowParams(){
    // p is just after '('; scan to matching ')' then check '=>'
    var depth=1, q=p;
    while(q<toks.length&&depth>0){ if(toks[q][0]==="op"&&toks[q][1]==="(")depth++; else if(toks[q][0]==="op"&&toks[q][1]===")")depth--; q++; }
    return toks[q] && toks[q][0]==="op" && toks[q][1]==="=>";
  }
  function parseParamsAfterParen(save){ p=save; next(); return parseParams(); }
  function parseArrowBody(){ eatOp("=>"); if(isOp("{")){ return parseBlock(); } return [["return", parseExpr()]]; }

  function parsePostfix(){
    var e=parsePrimary();
    for(;;){
      if(isOp(".")){ next(); var name=next()[1]; e=["dot", e, name]; }
      else if(isOp("[")){ next(); var ix=parseExpr(); eatOp("]"); e=["idx", e, ix]; }
      else if(isOp("(")){ next(); var args=[]; if(!isOp(")")){ args.push(parseExpr()); while(isOp(",")){ next(); args.push(parseExpr()); } } eatOp(")"); e=["call", e, args]; }
      else break;
    }
    return e;
  }
  var MUL=["*"], ADD=["+","-"], CMP=["<",">","<=",">=","===","!==","==","!="];
  function parseMul(){ var e=parsePostfix(); while(peek()[0]==="op"&&MUL.indexOf(peek()[1])>=0){ var o=next()[1]; e=["bin",o,e,parsePostfix()]; } return e; }
  function parseAdd(){ var e=parseMul(); while(peek()[0]==="op"&&ADD.indexOf(peek()[1])>=0){ var o=next()[1]; e=["bin",o,e,parseMul()]; } return e; }
  function parseCmp(){ var e=parseAdd(); while(peek()[0]==="op"&&CMP.indexOf(peek()[1])>=0){ var o=next()[1]; e=["bin",o,e,parseAdd()]; } return e; }
  function parseCond(){ var e=parseCmp(); if(isOp("?")){ next(); var t=parseExpr(); eatOp(":"); var el=parseExpr(); return ["cond",e,t,el]; } return e; }
  function parseAssign(){ var e=parseCond(); if(isOp("=")){ next(); var rhs=parseAssign(); return ["assign", e, rhs]; } return e; }
  function parseExpr(){ return parseAssign(); }

  function parseStmt(){
    if(isId("return")){ next(); var e=parseExpr(); if(isOp(";"))next(); return ["return", e]; }
    if(isId("var")){ next(); var nm=next()[1]; eatOp("="); var ve=parseExpr(); if(isOp(";"))next(); return ["var", nm, ve]; }
    if(isId("if")){ next(); eatOp("("); var c=parseExpr(); eatOp(")"); var th=parseStmt(); var el=null; if(isId("else")){ next(); el=parseStmt(); } return ["if", c, th, el]; }
    if(isId("while")){ next(); eatOp("("); var wc=parseExpr(); eatOp(")"); var wb=parseStmt(); return ["while", wc, wb]; }
    if(isOp("{")){ return ["block", parseBlock()]; }
    var ex=parseExpr(); if(isOp(";"))next(); return ["expr", ex];
  }
  function parseBlock(){ eatOp("{"); var ss=[]; while(!isOp("}")){ ss.push(parseStmt()); } eatOp("}"); return ss; }

  var funcs=[];
  while(peek()[0]!=="eof"){
    eatId("function"); var name=next()[1]; eatOp("("); var ps=parseParams(); var body=parseBlock();
    funcs.push(["fundecl", name, ps, body]);
  }
  return funcs;
}

if (typeof module !== "undefined") module.exports = { parseProgram: parseProgram, tokenize: tokenize };
