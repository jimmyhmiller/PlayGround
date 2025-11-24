// Test cases for number representation
// Small integers
var a = 0;
var b = 1;
var c = -1;
var d = 42;
var e = -42;
var f = 123456;

// Large integers within safe range
var g = 9007199254740991;     // 2^53 - 1 (MAX_SAFE_INTEGER)
var h = -9007199254740991;    // -(2^53 - 1) (MIN_SAFE_INTEGER)
var i = 9007199254740990;     // Just under max safe
var j = 1234567890123456;     // Large but safe

// Large integers beyond safe range (will lose precision)
var k = 9007199254740992;     // 2^53 (first unsafe)
var l = 9007199254740993;     // Should round
var m = 18446744073709551616; // 2^64
var n = 72057594037927936;    // 2^56
var o = 9007199254740990976;  // Large number from failing test

// Even larger numbers
var p = 295147905179352825856;
var q = 4722366482869645213696;
var r = 75557863725914323419136;

// Decimal numbers
var s = 0.5;
var t = 3.14159;
var u = -2.71828;
var v = 0.000001;
var w = 1.23e10;
var x = 1.23e-10;

// Special edge cases
var y = 16;
var z = 256;
var aa = 65536;
var bb = 16777216;
var cc = 4294967296;
