var cryptopp = require("cryptopp")
var keyPair = cryptopp.ecies.prime.generateKeyPair("secp256r1");
var cipher = cryptopp.ecies.prime.encrypt("Testing ECIES", keyPair.publicKey, keyPair.curveName);
var plainText = cryptopp.ecies.prime.decrypt(cipher, keyPair.privateKey, keyPair.curveName);
console.log(plainText)