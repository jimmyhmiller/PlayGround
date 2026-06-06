// Compiler derived from int.js by specializing jsmix (RustPE, 40 blocks).
// Use:  node mycompiler.js '<program-json>' [input]
function compile(v0) {
  let v1067, v1068, v1385, v1395, v1397, v1398, v1399, v1401, v100046, v100047, v100048, v100049, v100050, v100051, v100052, v100053, v100054, v100055, v100056, v100057, v100058, v100059, v500568, v700259, v700260, v701551, v701553, v1000096514, v1000143874;
  let __pc = 0;
  for (;;) {
    switch (__pc) {
      case 0: {
        v100046 = ["var", "prog"];
        v100047 = ["dot", v100046, "reduce"];
        v100048 = ["acc", "instr"];
        v100049 = ["var", "step"];
        v100050 = ["var", "instr"];
        v100051 = ["var", "acc"];
        v100052 = ["var", "input"];
        v100053 = [v100050, v100051, v100052];
        v100054 = ["call", v100049, v100053];
        v100055 = ["return", v100054];
        v100056 = [v100055];
        v100057 = ["fun", v100048, v100056];
        v100058 = ["lit", 0];
        v100059 = [v100057, v100058];
        v1385 = ["call", v100047, v100059];
        v1397 = v100047;
        v1398 = v100059;
        v701551 = v0;
        v701553 = "0";
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 1: {
        if (!((v1399.length > 0))) { __pc = 2; } else { __pc = 3; } break;
      }
      case 2: {
        if (!(((typeof v1401) === "number"))) { __pc = 38; } else { __pc = 39; } break;
      }
      case 3: {
        v1000143874 = v1399[0];
        __pc = 4; break;
      }
      case 4: {
        v1395 = v1000143874[0];
        if (!((v1395 === "addlit"))) { __pc = 14; } else { __pc = 5; } break;
      }
      case 5: {
        v1000096514 = v1000143874[1];
        if (!((((typeof v1401) === "number") && ((typeof v1000096514) === "number")))) { __pc = 6; } else { __pc = 7; } break;
      }
      case 6: {
        if (!(((typeof v1401) === "number"))) { __pc = 8; } else { __pc = 9; } break;
      }
      case 7: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (v1401 + v1000096514);
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 8: {
        if (!(((typeof v1000096514) === "number"))) { __pc = 12; } else { __pc = 13; } break;
      }
      case 9: {
        if (!(((typeof v1000096514) === "number"))) { __pc = 10; } else { __pc = 11; } break;
      }
      case 10: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " + ") + v1000096514) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 11: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " + ") + ("" + v1000096514)) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 12: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1401) + " + ") + v1000096514) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 13: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1401) + " + ") + ("" + v1000096514)) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 14: {
        v1395 = v1000143874[0];
        if (!((v1395 === "mullit"))) { __pc = 25; } else { __pc = 15; } break;
      }
      case 15: {
        v1000096514 = v1000143874[1];
        if (!((((typeof v1401) === "number") && ((typeof v1000096514) === "number")))) { __pc = 16; } else { __pc = 17; } break;
      }
      case 16: {
        if (!(((typeof v1401) === "number"))) { __pc = 18; } else { __pc = 19; } break;
      }
      case 17: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (v1401 * v1000096514);
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 18: {
        if (!(((typeof v1000096514) === "number"))) { __pc = 24; } else { __pc = 22; } break;
      }
      case 19: {
        if (!(((typeof v1000096514) === "number"))) { __pc = 21; } else { __pc = 20; } break;
      }
      case 20: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " * ") + ("" + v1000096514)) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 21: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " * ") + v1000096514) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 22: {
        v700259 = v1401;
        v700260 = ("" + v1000096514);
        v1067 = v700259;
        v1068 = v700260;
        __pc = 23; break;
      }
      case 23: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1067) + " * ") + v1068) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 24: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1401) + " * ") + v1000096514) + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 25: {
        v1395 = v1000143874[0];
        if (!((v1395 === "addin"))) { __pc = 31; } else { __pc = 26; } break;
      }
      case 26: {
        if (!((((typeof v1401) === "number") && false))) { __pc = 27; } else { __pc = 28; } break;
      }
      case 27: {
        if (!(((typeof v1401) === "number"))) { __pc = 30; } else { __pc = 29; } break;
      }
      case 28: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (v1401 + "x");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 29: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " + ") + "x") + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 30: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1401) + " + ") + "x") + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 31: {
        v1395 = v1000143874[0];
        if (!((v1395 === "subin"))) { __pc = 37; } else { __pc = 32; } break;
      }
      case 32: {
        if (!((((typeof v1401) === "number") && false))) { __pc = 33; } else { __pc = 34; } break;
      }
      case 33: {
        if (!(((typeof v1401) === "number"))) { __pc = 36; } else { __pc = 35; } break;
      }
      case 34: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (v1401 - "x");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 35: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + ("" + v1401)) + " - ") + "x") + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 36: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v701553 = (((("(" + v1401) + " - ") + "x") + ")");
        v1399 = v701551;
        v1401 = v701553;
        __pc = 1; break;
      }
      case 37: {
        v500568 = v1399.slice(1);
        v701551 = v500568;
        v1399 = v701551;
        __pc = 1; break;
      }
      case 38: {
        return v1401;
      }
      case 39: {
        return ("" + v1401);
      }
    }
  }
}

// compile(prog) returns a JS SOURCE STRING. To run it, make a native function.
module.exports = { compile: compile,
                   run: function(p,x){ return Function("x","return "+compile(p)+";")(x); } };
if (require.main === module) {
  var prog = JSON.parse(process.argv[2] || "[]");
  var src = compile(prog);
  console.log("compiled JS source:  function (x) { return " + src + "; }");
  if (process.argv[3] !== undefined)
    console.log("run(" + process.argv[3] + ") = " + Function("x","return "+src+";")(Number(process.argv[3])));
}
