goog.provide('hello_world.core');
goog.require('cljs.core');
goog.require('cljs.core.constants');
goog.require('cljs.nodejs');
cljs.nodejs.enable_util_print_BANG_();
hello_world.core.props = ({"friends": []});
hello_world.core._main = (function hello_world$core$_main(var_args){
var args__23374__auto__ = [];
var len__23372__auto___46 = arguments.length;
var i__23373__auto___47 = (0);
while(true){
if((i__23373__auto___47 < len__23372__auto___46)){
args__23374__auto__.push((arguments[i__23373__auto___47]));

var G__48 = (i__23373__auto___47 + (1));
i__23373__auto___47 = G__48;
continue;
} else {
}
break;
}

var argseq__23375__auto__ = ((((0) < args__23374__auto__.length))?(new cljs.core.IndexedSeq(args__23374__auto__.slice((0)),(0),null)):null);
return hello_world.core._main.cljs$core$IFn$_invoke$arity$variadic(argseq__23375__auto__);
});

hello_world.core._main.cljs$core$IFn$_invoke$arity$variadic = (function (args){
return cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.array_seq([((((hello_world.core.props["user"]) == null))?(hello_world.core.props["user"]):(((((hello_world.core.props["user"])["friends"]) == null))?((hello_world.core.props["user"])["friends"]):((((((hello_world.core.props["user"])["friends"])[(0)]) == null))?(((hello_world.core.props["user"])["friends"])[(0)]):(((hello_world.core.props["user"])["friends"])[(0)]["friends"]))))], 0));
});

hello_world.core._main.cljs$lang$maxFixedArity = (0);

hello_world.core._main.cljs$lang$applyTo = (function (seq45){
return hello_world.core._main.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq(seq45));
});

cljs.core._STAR_main_cli_fn_STAR_ = hello_world.core._main;
