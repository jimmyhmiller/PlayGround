goog.provide('cljs.repl');
cljs.repl.print_doc = (function cljs$repl$print_doc(p__48224){
var map__48225 = p__48224;
var map__48225__$1 = (((((!((map__48225 == null))))?(((((map__48225.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48225.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48225):map__48225);
var m = map__48225__$1;
var n = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48225__$1,new cljs.core.Keyword(null,"ns","ns",441598760));
var nm = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48225__$1,new cljs.core.Keyword(null,"name","name",1843675177));
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["-------------------------"], 0));

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([(function (){var or__4126__auto__ = new cljs.core.Keyword(null,"spec","spec",347520401).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return [(function (){var temp__5735__auto__ = new cljs.core.Keyword(null,"ns","ns",441598760).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(temp__5735__auto__)){
var ns = temp__5735__auto__;
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(ns),"/"].join('');
} else {
return null;
}
})(),cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join('');
}
})()], 0));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["Protocol"], 0));
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m))){
var seq__48231_48437 = cljs.core.seq(new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m));
var chunk__48232_48438 = null;
var count__48233_48439 = (0);
var i__48234_48440 = (0);
while(true){
if((i__48234_48440 < count__48233_48439)){
var f_48441 = chunk__48232_48438.cljs$core$IIndexed$_nth$arity$2(null,i__48234_48440);
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["  ",f_48441], 0));


var G__48442 = seq__48231_48437;
var G__48443 = chunk__48232_48438;
var G__48444 = count__48233_48439;
var G__48445 = (i__48234_48440 + (1));
seq__48231_48437 = G__48442;
chunk__48232_48438 = G__48443;
count__48233_48439 = G__48444;
i__48234_48440 = G__48445;
continue;
} else {
var temp__5735__auto___48446 = cljs.core.seq(seq__48231_48437);
if(temp__5735__auto___48446){
var seq__48231_48447__$1 = temp__5735__auto___48446;
if(cljs.core.chunked_seq_QMARK_(seq__48231_48447__$1)){
var c__4556__auto___48448 = cljs.core.chunk_first(seq__48231_48447__$1);
var G__48449 = cljs.core.chunk_rest(seq__48231_48447__$1);
var G__48450 = c__4556__auto___48448;
var G__48451 = cljs.core.count(c__4556__auto___48448);
var G__48452 = (0);
seq__48231_48437 = G__48449;
chunk__48232_48438 = G__48450;
count__48233_48439 = G__48451;
i__48234_48440 = G__48452;
continue;
} else {
var f_48453 = cljs.core.first(seq__48231_48447__$1);
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["  ",f_48453], 0));


var G__48454 = cljs.core.next(seq__48231_48447__$1);
var G__48455 = null;
var G__48456 = (0);
var G__48457 = (0);
seq__48231_48437 = G__48454;
chunk__48232_48438 = G__48455;
count__48233_48439 = G__48456;
i__48234_48440 = G__48457;
continue;
}
} else {
}
}
break;
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m))){
var arglists_48458 = new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_((function (){var or__4126__auto__ = new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m);
}
})())){
cljs.core.prn.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([arglists_48458], 0));
} else {
cljs.core.prn.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.first(arglists_48458)))?cljs.core.second(arglists_48458):arglists_48458)], 0));
}
} else {
}
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"special-form","special-form",-1326536374).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["Special Form"], 0));

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m)], 0));

if(cljs.core.contains_QMARK_(m,new cljs.core.Keyword(null,"url","url",276297046))){
if(cljs.core.truth_(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))){
return cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([["\n  Please see http://clojure.org/",cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))].join('')], 0));
} else {
return null;
}
} else {
return cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([["\n  Please see http://clojure.org/special_forms#",cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join('')], 0));
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["Macro"], 0));
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"spec","spec",347520401).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["Spec"], 0));
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["REPL Special Function"], 0));
} else {
}

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m)], 0));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
var seq__48237_48460 = cljs.core.seq(new cljs.core.Keyword(null,"methods","methods",453930866).cljs$core$IFn$_invoke$arity$1(m));
var chunk__48238_48461 = null;
var count__48239_48462 = (0);
var i__48240_48463 = (0);
while(true){
if((i__48240_48463 < count__48239_48462)){
var vec__48252_48464 = chunk__48238_48461.cljs$core$IIndexed$_nth$arity$2(null,i__48240_48463);
var name_48465 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48252_48464,(0),null);
var map__48255_48466 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48252_48464,(1),null);
var map__48255_48467__$1 = (((((!((map__48255_48466 == null))))?(((((map__48255_48466.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48255_48466.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48255_48466):map__48255_48466);
var doc_48468 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48255_48467__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_48469 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48255_48467__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println();

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",name_48465], 0));

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",arglists_48469], 0));

if(cljs.core.truth_(doc_48468)){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",doc_48468], 0));
} else {
}


var G__48470 = seq__48237_48460;
var G__48471 = chunk__48238_48461;
var G__48472 = count__48239_48462;
var G__48473 = (i__48240_48463 + (1));
seq__48237_48460 = G__48470;
chunk__48238_48461 = G__48471;
count__48239_48462 = G__48472;
i__48240_48463 = G__48473;
continue;
} else {
var temp__5735__auto___48474 = cljs.core.seq(seq__48237_48460);
if(temp__5735__auto___48474){
var seq__48237_48475__$1 = temp__5735__auto___48474;
if(cljs.core.chunked_seq_QMARK_(seq__48237_48475__$1)){
var c__4556__auto___48476 = cljs.core.chunk_first(seq__48237_48475__$1);
var G__48477 = cljs.core.chunk_rest(seq__48237_48475__$1);
var G__48478 = c__4556__auto___48476;
var G__48479 = cljs.core.count(c__4556__auto___48476);
var G__48480 = (0);
seq__48237_48460 = G__48477;
chunk__48238_48461 = G__48478;
count__48239_48462 = G__48479;
i__48240_48463 = G__48480;
continue;
} else {
var vec__48259_48481 = cljs.core.first(seq__48237_48475__$1);
var name_48482 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48259_48481,(0),null);
var map__48262_48483 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48259_48481,(1),null);
var map__48262_48484__$1 = (((((!((map__48262_48483 == null))))?(((((map__48262_48483.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48262_48483.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48262_48483):map__48262_48483);
var doc_48485 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48262_48484__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_48486 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48262_48484__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println();

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",name_48482], 0));

cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",arglists_48486], 0));

if(cljs.core.truth_(doc_48485)){
cljs.core.println.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([" ",doc_48485], 0));
} else {
}


var G__48487 = cljs.core.next(seq__48237_48475__$1);
var G__48488 = null;
var G__48489 = (0);
var G__48490 = (0);
seq__48237_48460 = G__48487;
chunk__48238_48461 = G__48488;
count__48239_48462 = G__48489;
i__48240_48463 = G__48490;
continue;
}
} else {
}
}
break;
}
} else {
}

if(cljs.core.truth_(n)){
var temp__5735__auto__ = cljs.spec.alpha.get_spec(cljs.core.symbol.cljs$core$IFn$_invoke$arity$2(cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.ns_name(n)),cljs.core.name(nm)));
if(cljs.core.truth_(temp__5735__auto__)){
var fnspec = temp__5735__auto__;
cljs.core.print.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["Spec"], 0));

var seq__48264 = cljs.core.seq(new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"args","args",1315556576),new cljs.core.Keyword(null,"ret","ret",-468222814),new cljs.core.Keyword(null,"fn","fn",-1175266204)], null));
var chunk__48265 = null;
var count__48266 = (0);
var i__48267 = (0);
while(true){
if((i__48267 < count__48266)){
var role = chunk__48265.cljs$core$IIndexed$_nth$arity$2(null,i__48267);
var temp__5735__auto___48491__$1 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(fnspec,role);
if(cljs.core.truth_(temp__5735__auto___48491__$1)){
var spec_48492 = temp__5735__auto___48491__$1;
cljs.core.print.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([["\n ",cljs.core.name(role),":"].join(''),cljs.spec.alpha.describe(spec_48492)], 0));
} else {
}


var G__48493 = seq__48264;
var G__48494 = chunk__48265;
var G__48495 = count__48266;
var G__48496 = (i__48267 + (1));
seq__48264 = G__48493;
chunk__48265 = G__48494;
count__48266 = G__48495;
i__48267 = G__48496;
continue;
} else {
var temp__5735__auto____$1 = cljs.core.seq(seq__48264);
if(temp__5735__auto____$1){
var seq__48264__$1 = temp__5735__auto____$1;
if(cljs.core.chunked_seq_QMARK_(seq__48264__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__48264__$1);
var G__48497 = cljs.core.chunk_rest(seq__48264__$1);
var G__48498 = c__4556__auto__;
var G__48499 = cljs.core.count(c__4556__auto__);
var G__48500 = (0);
seq__48264 = G__48497;
chunk__48265 = G__48498;
count__48266 = G__48499;
i__48267 = G__48500;
continue;
} else {
var role = cljs.core.first(seq__48264__$1);
var temp__5735__auto___48501__$2 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(fnspec,role);
if(cljs.core.truth_(temp__5735__auto___48501__$2)){
var spec_48502 = temp__5735__auto___48501__$2;
cljs.core.print.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([["\n ",cljs.core.name(role),":"].join(''),cljs.spec.alpha.describe(spec_48502)], 0));
} else {
}


var G__48503 = cljs.core.next(seq__48264__$1);
var G__48504 = null;
var G__48505 = (0);
var G__48506 = (0);
seq__48264 = G__48503;
chunk__48265 = G__48504;
count__48266 = G__48505;
i__48267 = G__48506;
continue;
}
} else {
return null;
}
}
break;
}
} else {
return null;
}
} else {
return null;
}
}
});
/**
 * Constructs a data representation for a Error with keys:
 *  :cause - root cause message
 *  :phase - error phase
 *  :via - cause chain, with cause keys:
 *           :type - exception class symbol
 *           :message - exception message
 *           :data - ex-data
 *           :at - top stack element
 *  :trace - root cause stack elements
 */
cljs.repl.Error__GT_map = (function cljs$repl$Error__GT_map(o){
var base = (function (t){
return cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"type","type",1174270348),(((t instanceof cljs.core.ExceptionInfo))?new cljs.core.Symbol(null,"ExceptionInfo","ExceptionInfo",294935087,null):(((t instanceof Error))?cljs.core.symbol.cljs$core$IFn$_invoke$arity$2("js",t.name):null
))], null),(function (){var temp__5735__auto__ = cljs.core.ex_message(t);
if(cljs.core.truth_(temp__5735__auto__)){
var msg = temp__5735__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"message","message",-406056002),msg], null);
} else {
return null;
}
})(),(function (){var temp__5735__auto__ = cljs.core.ex_data(t);
if(cljs.core.truth_(temp__5735__auto__)){
var ed = temp__5735__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"data","data",-232669377),ed], null);
} else {
return null;
}
})()], 0));
});
var via = (function (){var via = cljs.core.PersistentVector.EMPTY;
var t = o;
while(true){
if(cljs.core.truth_(t)){
var G__48511 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(via,t);
var G__48512 = cljs.core.ex_cause(t);
via = G__48511;
t = G__48512;
continue;
} else {
return via;
}
break;
}
})();
var root = cljs.core.peek(via);
return cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"via","via",-1904457336),cljs.core.vec(cljs.core.map.cljs$core$IFn$_invoke$arity$2(base,via)),new cljs.core.Keyword(null,"trace","trace",-1082747415),null], null),(function (){var temp__5735__auto__ = cljs.core.ex_message(root);
if(cljs.core.truth_(temp__5735__auto__)){
var root_msg = temp__5735__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"cause","cause",231901252),root_msg], null);
} else {
return null;
}
})(),(function (){var temp__5735__auto__ = cljs.core.ex_data(root);
if(cljs.core.truth_(temp__5735__auto__)){
var data = temp__5735__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"data","data",-232669377),data], null);
} else {
return null;
}
})(),(function (){var temp__5735__auto__ = new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358).cljs$core$IFn$_invoke$arity$1(cljs.core.ex_data(o));
if(cljs.core.truth_(temp__5735__auto__)){
var phase = temp__5735__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"phase","phase",575722892),phase], null);
} else {
return null;
}
})()], 0));
});
/**
 * Returns an analysis of the phase, error, cause, and location of an error that occurred
 *   based on Throwable data, as returned by Throwable->map. All attributes other than phase
 *   are optional:
 *  :clojure.error/phase - keyword phase indicator, one of:
 *    :read-source :compile-syntax-check :compilation :macro-syntax-check :macroexpansion
 *    :execution :read-eval-result :print-eval-result
 *  :clojure.error/source - file name (no path)
 *  :clojure.error/line - integer line number
 *  :clojure.error/column - integer column number
 *  :clojure.error/symbol - symbol being expanded/compiled/invoked
 *  :clojure.error/class - cause exception class symbol
 *  :clojure.error/cause - cause exception message
 *  :clojure.error/spec - explain-data for spec error
 */
cljs.repl.ex_triage = (function cljs$repl$ex_triage(datafied_throwable){
var map__48300 = datafied_throwable;
var map__48300__$1 = (((((!((map__48300 == null))))?(((((map__48300.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48300.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48300):map__48300);
var via = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48300__$1,new cljs.core.Keyword(null,"via","via",-1904457336));
var trace = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48300__$1,new cljs.core.Keyword(null,"trace","trace",-1082747415));
var phase = cljs.core.get.cljs$core$IFn$_invoke$arity$3(map__48300__$1,new cljs.core.Keyword(null,"phase","phase",575722892),new cljs.core.Keyword(null,"execution","execution",253283524));
var map__48302 = cljs.core.last(via);
var map__48302__$1 = (((((!((map__48302 == null))))?(((((map__48302.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48302.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48302):map__48302);
var type = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48302__$1,new cljs.core.Keyword(null,"type","type",1174270348));
var message = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48302__$1,new cljs.core.Keyword(null,"message","message",-406056002));
var data = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48302__$1,new cljs.core.Keyword(null,"data","data",-232669377));
var map__48303 = data;
var map__48303__$1 = (((((!((map__48303 == null))))?(((((map__48303.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48303.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48303):map__48303);
var problems = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48303__$1,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814));
var fn = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48303__$1,new cljs.core.Keyword("cljs.spec.alpha","fn","cljs.spec.alpha/fn",408600443));
var caller = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48303__$1,new cljs.core.Keyword("cljs.spec.test.alpha","caller","cljs.spec.test.alpha/caller",-398302390));
var map__48304 = new cljs.core.Keyword(null,"data","data",-232669377).cljs$core$IFn$_invoke$arity$1(cljs.core.first(via));
var map__48304__$1 = (((((!((map__48304 == null))))?(((((map__48304.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48304.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48304):map__48304);
var top_data = map__48304__$1;
var source = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48304__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397));
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3((function (){var G__48326 = phase;
var G__48326__$1 = (((G__48326 instanceof cljs.core.Keyword))?G__48326.fqn:null);
switch (G__48326__$1) {
case "read-source":
var map__48332 = data;
var map__48332__$1 = (((((!((map__48332 == null))))?(((((map__48332.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48332.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48332):map__48332);
var line = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48332__$1,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471));
var column = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48332__$1,new cljs.core.Keyword("clojure.error","column","clojure.error/column",304721553));
var G__48343 = cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.Keyword(null,"data","data",-232669377).cljs$core$IFn$_invoke$arity$1(cljs.core.second(via)),top_data], 0));
var G__48343__$1 = (cljs.core.truth_(source)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48343,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),source):G__48343);
var G__48343__$2 = (cljs.core.truth_((function (){var fexpr__48346 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null);
return (fexpr__48346.cljs$core$IFn$_invoke$arity$1 ? fexpr__48346.cljs$core$IFn$_invoke$arity$1(source) : fexpr__48346.call(null,source));
})())?cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(G__48343__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397)):G__48343__$1);
if(cljs.core.truth_(message)){
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48343__$2,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message);
} else {
return G__48343__$2;
}

break;
case "compile-syntax-check":
case "compilation":
case "macro-syntax-check":
case "macroexpansion":
var G__48352 = top_data;
var G__48352__$1 = (cljs.core.truth_(source)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48352,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),source):G__48352);
var G__48352__$2 = (cljs.core.truth_((function (){var fexpr__48353 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null);
return (fexpr__48353.cljs$core$IFn$_invoke$arity$1 ? fexpr__48353.cljs$core$IFn$_invoke$arity$1(source) : fexpr__48353.call(null,source));
})())?cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(G__48352__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397)):G__48352__$1);
var G__48352__$3 = (cljs.core.truth_(type)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48352__$2,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type):G__48352__$2);
var G__48352__$4 = (cljs.core.truth_(message)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48352__$3,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message):G__48352__$3);
if(cljs.core.truth_(problems)){
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48352__$4,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595),data);
} else {
return G__48352__$4;
}

break;
case "read-eval-result":
case "print-eval-result":
var vec__48365 = cljs.core.first(trace);
var source__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48365,(0),null);
var method = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48365,(1),null);
var file = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48365,(2),null);
var line = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48365,(3),null);
var G__48371 = top_data;
var G__48371__$1 = (cljs.core.truth_(line)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48371,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471),line):G__48371);
var G__48371__$2 = (cljs.core.truth_(file)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48371__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),file):G__48371__$1);
var G__48371__$3 = (cljs.core.truth_((function (){var and__4115__auto__ = source__$1;
if(cljs.core.truth_(and__4115__auto__)){
return method;
} else {
return and__4115__auto__;
}
})())?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48371__$2,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994),(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[source__$1,method],null))):G__48371__$2);
var G__48371__$4 = (cljs.core.truth_(type)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48371__$3,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type):G__48371__$3);
if(cljs.core.truth_(message)){
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48371__$4,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message);
} else {
return G__48371__$4;
}

break;
case "execution":
var vec__48380 = cljs.core.first(trace);
var source__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48380,(0),null);
var method = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48380,(1),null);
var file = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48380,(2),null);
var line = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__48380,(3),null);
var file__$1 = cljs.core.first(cljs.core.remove.cljs$core$IFn$_invoke$arity$2((function (p1__48293_SHARP_){
var or__4126__auto__ = (p1__48293_SHARP_ == null);
if(or__4126__auto__){
return or__4126__auto__;
} else {
var fexpr__48384 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null);
return (fexpr__48384.cljs$core$IFn$_invoke$arity$1 ? fexpr__48384.cljs$core$IFn$_invoke$arity$1(p1__48293_SHARP_) : fexpr__48384.call(null,p1__48293_SHARP_));
}
}),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(caller),file], null)));
var err_line = (function (){var or__4126__auto__ = new cljs.core.Keyword(null,"line","line",212345235).cljs$core$IFn$_invoke$arity$1(caller);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return line;
}
})();
var G__48389 = new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type], null);
var G__48389__$1 = (cljs.core.truth_(err_line)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48389,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471),err_line):G__48389);
var G__48389__$2 = (cljs.core.truth_(message)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48389__$1,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message):G__48389__$1);
var G__48389__$3 = (cljs.core.truth_((function (){var or__4126__auto__ = fn;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
var and__4115__auto__ = source__$1;
if(cljs.core.truth_(and__4115__auto__)){
return method;
} else {
return and__4115__auto__;
}
}
})())?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48389__$2,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994),(function (){var or__4126__auto__ = fn;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return (new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[source__$1,method],null));
}
})()):G__48389__$2);
var G__48389__$4 = (cljs.core.truth_(file__$1)?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48389__$3,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),file__$1):G__48389__$3);
if(cljs.core.truth_(problems)){
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(G__48389__$4,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595),data);
} else {
return G__48389__$4;
}

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__48326__$1)].join('')));

}
})(),new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358),phase);
});
/**
 * Returns a string from exception data, as produced by ex-triage.
 *   The first line summarizes the exception phase and location.
 *   The subsequent lines describe the cause.
 */
cljs.repl.ex_str = (function cljs$repl$ex_str(p__48393){
var map__48394 = p__48393;
var map__48394__$1 = (((((!((map__48394 == null))))?(((((map__48394.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__48394.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__48394):map__48394);
var triage_data = map__48394__$1;
var phase = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358));
var source = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397));
var line = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471));
var column = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","column","clojure.error/column",304721553));
var symbol = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994));
var class$ = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890));
var cause = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742));
var spec = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__48394__$1,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595));
var loc = [cljs.core.str.cljs$core$IFn$_invoke$arity$1((function (){var or__4126__auto__ = source;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return "<cljs repl>";
}
})()),":",cljs.core.str.cljs$core$IFn$_invoke$arity$1((function (){var or__4126__auto__ = line;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return (1);
}
})()),(cljs.core.truth_(column)?[":",cljs.core.str.cljs$core$IFn$_invoke$arity$1(column)].join(''):"")].join('');
var class_name = cljs.core.name((function (){var or__4126__auto__ = class$;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return "";
}
})());
var simple_class = class_name;
var cause_type = ((cljs.core.contains_QMARK_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["RuntimeException",null,"Exception",null], null), null),simple_class))?"":[" (",simple_class,")"].join(''));
var format = goog.string.format;
var G__48397 = phase;
var G__48397__$1 = (((G__48397 instanceof cljs.core.Keyword))?G__48397.fqn:null);
switch (G__48397__$1) {
case "read-source":
return (format.cljs$core$IFn$_invoke$arity$3 ? format.cljs$core$IFn$_invoke$arity$3("Syntax error reading source at (%s).\n%s\n",loc,cause) : format.call(null,"Syntax error reading source at (%s).\n%s\n",loc,cause));

break;
case "macro-syntax-check":
var G__48398 = "Syntax error macroexpanding %sat (%s).\n%s";
var G__48399 = (cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):"");
var G__48400 = loc;
var G__48401 = (cljs.core.truth_(spec)?(function (){var sb__4667__auto__ = (new goog.string.StringBuffer());
var _STAR_print_newline_STAR__orig_val__48402_48520 = cljs.core._STAR_print_newline_STAR_;
var _STAR_print_fn_STAR__orig_val__48403_48521 = cljs.core._STAR_print_fn_STAR_;
var _STAR_print_newline_STAR__temp_val__48404_48522 = true;
var _STAR_print_fn_STAR__temp_val__48405_48523 = (function (x__4668__auto__){
return sb__4667__auto__.append(x__4668__auto__);
});
(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__temp_val__48404_48522);

(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__temp_val__48405_48523);

try{cljs.spec.alpha.explain_out(cljs.core.update.cljs$core$IFn$_invoke$arity$3(spec,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814),(function (probs){
return cljs.core.map.cljs$core$IFn$_invoke$arity$2((function (p1__48390_SHARP_){
return cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(p1__48390_SHARP_,new cljs.core.Keyword(null,"in","in",-1531184865));
}),probs);
}))
);
}finally {(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__orig_val__48403_48521);

(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__orig_val__48402_48520);
}
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(sb__4667__auto__);
})():(format.cljs$core$IFn$_invoke$arity$2 ? format.cljs$core$IFn$_invoke$arity$2("%s\n",cause) : format.call(null,"%s\n",cause)));
return (format.cljs$core$IFn$_invoke$arity$4 ? format.cljs$core$IFn$_invoke$arity$4(G__48398,G__48399,G__48400,G__48401) : format.call(null,G__48398,G__48399,G__48400,G__48401));

break;
case "macroexpansion":
var G__48406 = "Unexpected error%s macroexpanding %sat (%s).\n%s\n";
var G__48407 = cause_type;
var G__48408 = (cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):"");
var G__48409 = loc;
var G__48410 = cause;
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5(G__48406,G__48407,G__48408,G__48409,G__48410) : format.call(null,G__48406,G__48407,G__48408,G__48409,G__48410));

break;
case "compile-syntax-check":
var G__48411 = "Syntax error%s compiling %sat (%s).\n%s\n";
var G__48412 = cause_type;
var G__48413 = (cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):"");
var G__48414 = loc;
var G__48415 = cause;
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5(G__48411,G__48412,G__48413,G__48414,G__48415) : format.call(null,G__48411,G__48412,G__48413,G__48414,G__48415));

break;
case "compilation":
var G__48416 = "Unexpected error%s compiling %sat (%s).\n%s\n";
var G__48417 = cause_type;
var G__48418 = (cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):"");
var G__48419 = loc;
var G__48420 = cause;
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5(G__48416,G__48417,G__48418,G__48419,G__48420) : format.call(null,G__48416,G__48417,G__48418,G__48419,G__48420));

break;
case "read-eval-result":
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5("Error reading eval result%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause) : format.call(null,"Error reading eval result%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause));

break;
case "print-eval-result":
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5("Error printing return value%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause) : format.call(null,"Error printing return value%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause));

break;
case "execution":
if(cljs.core.truth_(spec)){
var G__48422 = "Execution error - invalid arguments to %s at (%s).\n%s";
var G__48423 = symbol;
var G__48424 = loc;
var G__48425 = (function (){var sb__4667__auto__ = (new goog.string.StringBuffer());
var _STAR_print_newline_STAR__orig_val__48426_48525 = cljs.core._STAR_print_newline_STAR_;
var _STAR_print_fn_STAR__orig_val__48427_48526 = cljs.core._STAR_print_fn_STAR_;
var _STAR_print_newline_STAR__temp_val__48428_48527 = true;
var _STAR_print_fn_STAR__temp_val__48429_48528 = (function (x__4668__auto__){
return sb__4667__auto__.append(x__4668__auto__);
});
(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__temp_val__48428_48527);

(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__temp_val__48429_48528);

try{cljs.spec.alpha.explain_out(cljs.core.update.cljs$core$IFn$_invoke$arity$3(spec,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814),(function (probs){
return cljs.core.map.cljs$core$IFn$_invoke$arity$2((function (p1__48391_SHARP_){
return cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(p1__48391_SHARP_,new cljs.core.Keyword(null,"in","in",-1531184865));
}),probs);
}))
);
}finally {(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__orig_val__48427_48526);

(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__orig_val__48426_48525);
}
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(sb__4667__auto__);
})();
return (format.cljs$core$IFn$_invoke$arity$4 ? format.cljs$core$IFn$_invoke$arity$4(G__48422,G__48423,G__48424,G__48425) : format.call(null,G__48422,G__48423,G__48424,G__48425));
} else {
var G__48430 = "Execution error%s at %s(%s).\n%s\n";
var G__48431 = cause_type;
var G__48432 = (cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):"");
var G__48433 = loc;
var G__48434 = cause;
return (format.cljs$core$IFn$_invoke$arity$5 ? format.cljs$core$IFn$_invoke$arity$5(G__48430,G__48431,G__48432,G__48433,G__48434) : format.call(null,G__48430,G__48431,G__48432,G__48433,G__48434));
}

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__48397__$1)].join('')));

}
});
cljs.repl.error__GT_str = (function cljs$repl$error__GT_str(error){
return cljs.repl.ex_str(cljs.repl.ex_triage(cljs.repl.Error__GT_map(error)));
});

//# sourceMappingURL=cljs.repl.js.map
