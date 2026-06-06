function main(v0) {
  let v1002, v1003, v1004, v1005, v1006, v1007, v1008, v1009, v1010, v1011, v100004, v100008, v500081, v500085, v500102, v700003, v700004, v700005, v700006, v700007, v700008, v700009, v700010, v700011, v1000015873, v3000000057, v3000000070, v3000000088;
  let __pc = 0;
  for (;;) {
    switch (__pc) {
      case 0: {
        v100004 = ["sub", "binary", "sub"];
        v1002 = [];
        v700003 = 0;
        v700004 = undefined;
        v700005 = undefined;
        v700006 = undefined;
        v700007 = undefined;
        v700008 = undefined;
        v700009 = undefined;
        v700010 = undefined;
        v700011 = undefined;
        v1003 = v700003;
        v1004 = v700004;
        v1005 = v700005;
        v1006 = v700006;
        v1007 = v700007;
        v1008 = v700008;
        v1009 = v700009;
        v1010 = v700010;
        v1011 = v700011;
        __pc = 1; break;
      }
      case 1: {
        if (!((v1003 < v0.length))) { __pc = 2; } else { __pc = 3; } break;
      }
      case 2: {
        v500102 = v1002.pop();
        return v500102;
      }
      case 3: {
        v1004 = v0[v1003];
        v1005 = v1004[0];
        __pc = 4; break;
      }
      case 4: {
        if (!(("pushlit" === v1005))) { __pc = 5; } else { __pc = 6; } break;
      }
      case 5: {
        v700006 = "?";
        v700007 = "?";
        v1006 = v700006;
        v1007 = v700007;
        __pc = 7; break;
      }
      case 6: {
        v700006 = "leaflit";
        v700007 = "lit";
        v1006 = v700006;
        v1007 = v700007;
        __pc = 7; break;
      }
      case 7: {
        if (!(("pushin" === v1005))) { __pc = 9; } else { __pc = 8; } break;
      }
      case 8: {
        v700006 = "leafin";
        v700007 = "in";
        v1006 = v700006;
        v1007 = v700007;
        __pc = 9; break;
      }
      case 9: {
        if (!(("add" === v1005))) { __pc = 11; } else { __pc = 10; } break;
      }
      case 10: {
        v700006 = "binary";
        v700007 = "add";
        v1006 = v700006;
        v1007 = v700007;
        __pc = 11; break;
      }
      case 11: {
        if (!(("mul" === v1005))) { __pc = 13; } else { __pc = 12; } break;
      }
      case 12: {
        v700006 = "binary";
        v700007 = "mul";
        v1006 = v700006;
        v1007 = v700007;
        __pc = 13; break;
      }
      case 13: {
        if (!((v100004[0] === v1005))) { __pc = 15; } else { __pc = 14; } break;
      }
      case 14: {
        v1006 = v100004[1];
        v1007 = v100004[2];
        __pc = 15; break;
      }
      case 15: {
        if (!((v1006 === "leaflit"))) { __pc = 16; } else { __pc = 17; } break;
      }
      case 16: {
        if (!((v1006 === "leafin"))) { __pc = 19; } else { __pc = 20; } break;
      }
      case 17: {
        v3000000057 = v1002.push;
        v1000015873 = v1004[1];
        v100008 = [v1007, v1000015873];
        v1002.push(v100008);
        __pc = 18; break;
      }
      case 18: {
        v700003 = (v1003 + 1);
        v700008 = 5;
        v700009 = v100004;
        v1003 = v700003;
        v1008 = v700008;
        v1009 = v700009;
        __pc = 1; break;
      }
      case 19: {
        if (!((v1006 === "binary"))) { __pc = 21; } else { __pc = 22; } break;
      }
      case 20: {
        v3000000070 = v1002.push;
        v100008 = [v1007];
        v1002.push(v100008);
        __pc = 18; break;
      }
      case 21: {
        v700003 = (v1003 + 1);
        v700008 = 5;
        v700009 = v100004;
        v1003 = v700003;
        v1008 = v700008;
        v1009 = v700009;
        __pc = 1; break;
      }
      case 22: {
        v500081 = v1002.pop();
        v500085 = v1002.pop();
        v3000000088 = v1002.push;
        v100008 = [v1007, v500085, v500081];
        v1002.push(v100008);
        v700010 = v500081;
        v700011 = v500085;
        v1010 = v700010;
        v1011 = v700011;
        __pc = 18; break;
      }
    }
  }
}
