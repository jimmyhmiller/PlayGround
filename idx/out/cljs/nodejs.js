goog.provide('cljs.nodejs');
goog.require('cljs.core');
goog.require('cljs.core.constants');
cljs.nodejs.require = require;
cljs.nodejs.process = process;
cljs.nodejs.enable_util_print_BANG_ = (function cljs$nodejs$enable_util_print_BANG_(){
cljs.core._STAR_print_newline_STAR_ = false;

cljs.core._STAR_print_fn_STAR_ = (function() { 
var G__24__delegate = function (args){
return console.log.apply(console,cljs.core.into_array.cljs$core$IFn$_invoke$arity$1(args));
};
var G__24 = function (var_args){
var args = null;
if (arguments.length > 0) {
var G__25__i = 0, G__25__a = new Array(arguments.length -  0);
while (G__25__i < G__25__a.length) {G__25__a[G__25__i] = arguments[G__25__i + 0]; ++G__25__i;}
  args = new cljs.core.IndexedSeq(G__25__a,0);
} 
return G__24__delegate.call(this,args);};
G__24.cljs$lang$maxFixedArity = 0;
G__24.cljs$lang$applyTo = (function (arglist__26){
var args = cljs.core.seq(arglist__26);
return G__24__delegate(args);
});
G__24.cljs$core$IFn$_invoke$arity$variadic = G__24__delegate;
return G__24;
})()
;

cljs.core._STAR_print_err_fn_STAR_ = (function() { 
var G__27__delegate = function (args){
return console.error.apply(console,cljs.core.into_array.cljs$core$IFn$_invoke$arity$1(args));
};
var G__27 = function (var_args){
var args = null;
if (arguments.length > 0) {
var G__28__i = 0, G__28__a = new Array(arguments.length -  0);
while (G__28__i < G__28__a.length) {G__28__a[G__28__i] = arguments[G__28__i + 0]; ++G__28__i;}
  args = new cljs.core.IndexedSeq(G__28__a,0);
} 
return G__27__delegate.call(this,args);};
G__27.cljs$lang$maxFixedArity = 0;
G__27.cljs$lang$applyTo = (function (arglist__29){
var args = cljs.core.seq(arglist__29);
return G__27__delegate(args);
});
G__27.cljs$core$IFn$_invoke$arity$variadic = G__27__delegate;
return G__27;
})()
;

return null;
});
