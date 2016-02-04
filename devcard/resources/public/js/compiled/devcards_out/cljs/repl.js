// Compiled by ClojureScript 1.7.170 {}
goog.provide('cljs.repl');
goog.require('cljs.core');
cljs.repl.print_doc = (function cljs$repl$print_doc(m){
cljs.core.println.call(null,"-------------------------");

cljs.core.println.call(null,[cljs.core.str((function (){var temp__4425__auto__ = new cljs.core.Keyword(null,"ns","ns",441598760).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(temp__4425__auto__)){
var ns = temp__4425__auto__;
return [cljs.core.str(ns),cljs.core.str("/")].join('');
} else {
return null;
}
})()),cljs.core.str(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join(''));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Protocol");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m))){
var seq__26359_26373 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m));
var chunk__26360_26374 = null;
var count__26361_26375 = (0);
var i__26362_26376 = (0);
while(true){
if((i__26362_26376 < count__26361_26375)){
var f_26377 = cljs.core._nth.call(null,chunk__26360_26374,i__26362_26376);
cljs.core.println.call(null,"  ",f_26377);

var G__26378 = seq__26359_26373;
var G__26379 = chunk__26360_26374;
var G__26380 = count__26361_26375;
var G__26381 = (i__26362_26376 + (1));
seq__26359_26373 = G__26378;
chunk__26360_26374 = G__26379;
count__26361_26375 = G__26380;
i__26362_26376 = G__26381;
continue;
} else {
var temp__4425__auto___26382 = cljs.core.seq.call(null,seq__26359_26373);
if(temp__4425__auto___26382){
var seq__26359_26383__$1 = temp__4425__auto___26382;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__26359_26383__$1)){
var c__17569__auto___26384 = cljs.core.chunk_first.call(null,seq__26359_26383__$1);
var G__26385 = cljs.core.chunk_rest.call(null,seq__26359_26383__$1);
var G__26386 = c__17569__auto___26384;
var G__26387 = cljs.core.count.call(null,c__17569__auto___26384);
var G__26388 = (0);
seq__26359_26373 = G__26385;
chunk__26360_26374 = G__26386;
count__26361_26375 = G__26387;
i__26362_26376 = G__26388;
continue;
} else {
var f_26389 = cljs.core.first.call(null,seq__26359_26383__$1);
cljs.core.println.call(null,"  ",f_26389);

var G__26390 = cljs.core.next.call(null,seq__26359_26383__$1);
var G__26391 = null;
var G__26392 = (0);
var G__26393 = (0);
seq__26359_26373 = G__26390;
chunk__26360_26374 = G__26391;
count__26361_26375 = G__26392;
i__26362_26376 = G__26393;
continue;
}
} else {
}
}
break;
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m))){
var arglists_26394 = new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_((function (){var or__16766__auto__ = new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m);
}
})())){
cljs.core.prn.call(null,arglists_26394);
} else {
cljs.core.prn.call(null,((cljs.core._EQ_.call(null,new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.first.call(null,arglists_26394)))?cljs.core.second.call(null,arglists_26394):arglists_26394));
}
} else {
}
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"special-form","special-form",-1326536374).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Special Form");

cljs.core.println.call(null," ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m));

if(cljs.core.contains_QMARK_.call(null,m,new cljs.core.Keyword(null,"url","url",276297046))){
if(cljs.core.truth_(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))){
return cljs.core.println.call(null,[cljs.core.str("\n  Please see http://clojure.org/"),cljs.core.str(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))].join(''));
} else {
return null;
}
} else {
return cljs.core.println.call(null,[cljs.core.str("\n  Please see http://clojure.org/special_forms#"),cljs.core.str(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join(''));
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Macro");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"REPL Special Function");
} else {
}

cljs.core.println.call(null," ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
var seq__26363 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"methods","methods",453930866).cljs$core$IFn$_invoke$arity$1(m));
var chunk__26364 = null;
var count__26365 = (0);
var i__26366 = (0);
while(true){
if((i__26366 < count__26365)){
var vec__26367 = cljs.core._nth.call(null,chunk__26364,i__26366);
var name = cljs.core.nth.call(null,vec__26367,(0),null);
var map__26368 = cljs.core.nth.call(null,vec__26367,(1),null);
var map__26368__$1 = ((((!((map__26368 == null)))?((((map__26368.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26368.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26368):map__26368);
var doc = cljs.core.get.call(null,map__26368__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists = cljs.core.get.call(null,map__26368__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name);

cljs.core.println.call(null," ",arglists);

if(cljs.core.truth_(doc)){
cljs.core.println.call(null," ",doc);
} else {
}

var G__26395 = seq__26363;
var G__26396 = chunk__26364;
var G__26397 = count__26365;
var G__26398 = (i__26366 + (1));
seq__26363 = G__26395;
chunk__26364 = G__26396;
count__26365 = G__26397;
i__26366 = G__26398;
continue;
} else {
var temp__4425__auto__ = cljs.core.seq.call(null,seq__26363);
if(temp__4425__auto__){
var seq__26363__$1 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__26363__$1)){
var c__17569__auto__ = cljs.core.chunk_first.call(null,seq__26363__$1);
var G__26399 = cljs.core.chunk_rest.call(null,seq__26363__$1);
var G__26400 = c__17569__auto__;
var G__26401 = cljs.core.count.call(null,c__17569__auto__);
var G__26402 = (0);
seq__26363 = G__26399;
chunk__26364 = G__26400;
count__26365 = G__26401;
i__26366 = G__26402;
continue;
} else {
var vec__26370 = cljs.core.first.call(null,seq__26363__$1);
var name = cljs.core.nth.call(null,vec__26370,(0),null);
var map__26371 = cljs.core.nth.call(null,vec__26370,(1),null);
var map__26371__$1 = ((((!((map__26371 == null)))?((((map__26371.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26371.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26371):map__26371);
var doc = cljs.core.get.call(null,map__26371__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists = cljs.core.get.call(null,map__26371__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name);

cljs.core.println.call(null," ",arglists);

if(cljs.core.truth_(doc)){
cljs.core.println.call(null," ",doc);
} else {
}

var G__26403 = cljs.core.next.call(null,seq__26363__$1);
var G__26404 = null;
var G__26405 = (0);
var G__26406 = (0);
seq__26363 = G__26403;
chunk__26364 = G__26404;
count__26365 = G__26405;
i__26366 = G__26406;
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
}
});

//# sourceMappingURL=repl.js.map?rel=1454621293596