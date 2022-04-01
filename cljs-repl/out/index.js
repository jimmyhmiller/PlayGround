import {npmDeps} from "./npm_deps.js";
var CLOSURE_UNCOMPILED_DEFINES = {"cljs.core._STAR_global_STAR_":"window","cljs.core._STAR_target_STAR_":"bundle"};
var CLOSURE_NO_DEPS = true;
if(typeof goog == "undefined") document.write('<script src="out/goog/base.js"></script>');
document.write('<script src="out/goog/deps.js"></script>');
document.write('<script src="out/cljs_deps.js"></script>');
document.write('<script>if (typeof goog == "undefined") console.warn("ClojureScript could not load :main, did you forget to specify :asset-path?");</script>');
document.write('<script>goog.require("process.env");</script>');
document.write('<script>goog.require("clojure.browser.repl.preload");</script>');
document.write('<script>goog.require("app.main");</script>');
window.require = function(lib) {
   return npmDeps[lib];
}
