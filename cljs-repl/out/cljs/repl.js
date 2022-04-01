// Compiled by ClojureScript 1.10.758 {:target :nodejs}
goog.provide('cljs.repl');
goog.require('cljs.core');
goog.require('cljs.spec.alpha');
goog.require('goog.string');
goog.require('goog.string.format');
cljs.repl.print_doc = (function cljs$repl$print_doc(p__4052){
var map__4053 = p__4052;
var map__4053__$1 = (((((!((map__4053 == null))))?(((((map__4053.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4053.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4053):map__4053);
var m = map__4053__$1;
var n = cljs.core.get.call(null,map__4053__$1,new cljs.core.Keyword(null,"ns","ns",441598760));
var nm = cljs.core.get.call(null,map__4053__$1,new cljs.core.Keyword(null,"name","name",1843675177));
cljs.core.println.call(null,"-------------------------");

cljs.core.println.call(null,(function (){var or__4126__auto__ = new cljs.core.Keyword(null,"spec","spec",347520401).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return [(function (){var temp__5753__auto__ = new cljs.core.Keyword(null,"ns","ns",441598760).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(temp__5753__auto__)){
var ns = temp__5753__auto__;
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(ns),"/"].join('');
} else {
return null;
}
})(),cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join('');
}
})());

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Protocol");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m))){
var seq__4055_4087 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m));
var chunk__4056_4088 = null;
var count__4057_4089 = (0);
var i__4058_4090 = (0);
while(true){
if((i__4058_4090 < count__4057_4089)){
var f_4091 = cljs.core._nth.call(null,chunk__4056_4088,i__4058_4090);
cljs.core.println.call(null,"  ",f_4091);


var G__4092 = seq__4055_4087;
var G__4093 = chunk__4056_4088;
var G__4094 = count__4057_4089;
var G__4095 = (i__4058_4090 + (1));
seq__4055_4087 = G__4092;
chunk__4056_4088 = G__4093;
count__4057_4089 = G__4094;
i__4058_4090 = G__4095;
continue;
} else {
var temp__5753__auto___4096 = cljs.core.seq.call(null,seq__4055_4087);
if(temp__5753__auto___4096){
var seq__4055_4097__$1 = temp__5753__auto___4096;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__4055_4097__$1)){
var c__4556__auto___4098 = cljs.core.chunk_first.call(null,seq__4055_4097__$1);
var G__4099 = cljs.core.chunk_rest.call(null,seq__4055_4097__$1);
var G__4100 = c__4556__auto___4098;
var G__4101 = cljs.core.count.call(null,c__4556__auto___4098);
var G__4102 = (0);
seq__4055_4087 = G__4099;
chunk__4056_4088 = G__4100;
count__4057_4089 = G__4101;
i__4058_4090 = G__4102;
continue;
} else {
var f_4103 = cljs.core.first.call(null,seq__4055_4097__$1);
cljs.core.println.call(null,"  ",f_4103);


var G__4104 = cljs.core.next.call(null,seq__4055_4097__$1);
var G__4105 = null;
var G__4106 = (0);
var G__4107 = (0);
seq__4055_4087 = G__4104;
chunk__4056_4088 = G__4105;
count__4057_4089 = G__4106;
i__4058_4090 = G__4107;
continue;
}
} else {
}
}
break;
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m))){
var arglists_4108 = new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_((function (){var or__4126__auto__ = new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m);
}
})())){
cljs.core.prn.call(null,arglists_4108);
} else {
cljs.core.prn.call(null,((cljs.core._EQ_.call(null,new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.first.call(null,arglists_4108)))?cljs.core.second.call(null,arglists_4108):arglists_4108));
}
} else {
}
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"special-form","special-form",-1326536374).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Special Form");

cljs.core.println.call(null," ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m));

if(cljs.core.contains_QMARK_.call(null,m,new cljs.core.Keyword(null,"url","url",276297046))){
if(cljs.core.truth_(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))){
return cljs.core.println.call(null,["\n  Please see http://clojure.org/",cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"url","url",276297046).cljs$core$IFn$_invoke$arity$1(m))].join(''));
} else {
return null;
}
} else {
return cljs.core.println.call(null,["\n  Please see http://clojure.org/special_forms#",cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join(''));
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Macro");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"spec","spec",347520401).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Spec");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"REPL Special Function");
} else {
}

cljs.core.println.call(null," ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
var seq__4059_4109 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"methods","methods",453930866).cljs$core$IFn$_invoke$arity$1(m));
var chunk__4060_4110 = null;
var count__4061_4111 = (0);
var i__4062_4112 = (0);
while(true){
if((i__4062_4112 < count__4061_4111)){
var vec__4073_4113 = cljs.core._nth.call(null,chunk__4060_4110,i__4062_4112);
var name_4114 = cljs.core.nth.call(null,vec__4073_4113,(0),null);
var map__4076_4115 = cljs.core.nth.call(null,vec__4073_4113,(1),null);
var map__4076_4116__$1 = (((((!((map__4076_4115 == null))))?(((((map__4076_4115.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4076_4115.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4076_4115):map__4076_4115);
var doc_4117 = cljs.core.get.call(null,map__4076_4116__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_4118 = cljs.core.get.call(null,map__4076_4116__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name_4114);

cljs.core.println.call(null," ",arglists_4118);

if(cljs.core.truth_(doc_4117)){
cljs.core.println.call(null," ",doc_4117);
} else {
}


var G__4119 = seq__4059_4109;
var G__4120 = chunk__4060_4110;
var G__4121 = count__4061_4111;
var G__4122 = (i__4062_4112 + (1));
seq__4059_4109 = G__4119;
chunk__4060_4110 = G__4120;
count__4061_4111 = G__4121;
i__4062_4112 = G__4122;
continue;
} else {
var temp__5753__auto___4123 = cljs.core.seq.call(null,seq__4059_4109);
if(temp__5753__auto___4123){
var seq__4059_4124__$1 = temp__5753__auto___4123;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__4059_4124__$1)){
var c__4556__auto___4125 = cljs.core.chunk_first.call(null,seq__4059_4124__$1);
var G__4126 = cljs.core.chunk_rest.call(null,seq__4059_4124__$1);
var G__4127 = c__4556__auto___4125;
var G__4128 = cljs.core.count.call(null,c__4556__auto___4125);
var G__4129 = (0);
seq__4059_4109 = G__4126;
chunk__4060_4110 = G__4127;
count__4061_4111 = G__4128;
i__4062_4112 = G__4129;
continue;
} else {
var vec__4078_4130 = cljs.core.first.call(null,seq__4059_4124__$1);
var name_4131 = cljs.core.nth.call(null,vec__4078_4130,(0),null);
var map__4081_4132 = cljs.core.nth.call(null,vec__4078_4130,(1),null);
var map__4081_4133__$1 = (((((!((map__4081_4132 == null))))?(((((map__4081_4132.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4081_4132.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4081_4132):map__4081_4132);
var doc_4134 = cljs.core.get.call(null,map__4081_4133__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_4135 = cljs.core.get.call(null,map__4081_4133__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name_4131);

cljs.core.println.call(null," ",arglists_4135);

if(cljs.core.truth_(doc_4134)){
cljs.core.println.call(null," ",doc_4134);
} else {
}


var G__4136 = cljs.core.next.call(null,seq__4059_4124__$1);
var G__4137 = null;
var G__4138 = (0);
var G__4139 = (0);
seq__4059_4109 = G__4136;
chunk__4060_4110 = G__4137;
count__4061_4111 = G__4138;
i__4062_4112 = G__4139;
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
var temp__5753__auto__ = cljs.spec.alpha.get_spec.call(null,cljs.core.symbol.call(null,cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.ns_name.call(null,n)),cljs.core.name.call(null,nm)));
if(cljs.core.truth_(temp__5753__auto__)){
var fnspec = temp__5753__auto__;
cljs.core.print.call(null,"Spec");

var seq__4083 = cljs.core.seq.call(null,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"args","args",1315556576),new cljs.core.Keyword(null,"ret","ret",-468222814),new cljs.core.Keyword(null,"fn","fn",-1175266204)], null));
var chunk__4084 = null;
var count__4085 = (0);
var i__4086 = (0);
while(true){
if((i__4086 < count__4085)){
var role = cljs.core._nth.call(null,chunk__4084,i__4086);
var temp__5753__auto___4140__$1 = cljs.core.get.call(null,fnspec,role);
if(cljs.core.truth_(temp__5753__auto___4140__$1)){
var spec_4141 = temp__5753__auto___4140__$1;
cljs.core.print.call(null,["\n ",cljs.core.name.call(null,role),":"].join(''),cljs.spec.alpha.describe.call(null,spec_4141));
} else {
}


var G__4142 = seq__4083;
var G__4143 = chunk__4084;
var G__4144 = count__4085;
var G__4145 = (i__4086 + (1));
seq__4083 = G__4142;
chunk__4084 = G__4143;
count__4085 = G__4144;
i__4086 = G__4145;
continue;
} else {
var temp__5753__auto____$1 = cljs.core.seq.call(null,seq__4083);
if(temp__5753__auto____$1){
var seq__4083__$1 = temp__5753__auto____$1;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__4083__$1)){
var c__4556__auto__ = cljs.core.chunk_first.call(null,seq__4083__$1);
var G__4146 = cljs.core.chunk_rest.call(null,seq__4083__$1);
var G__4147 = c__4556__auto__;
var G__4148 = cljs.core.count.call(null,c__4556__auto__);
var G__4149 = (0);
seq__4083 = G__4146;
chunk__4084 = G__4147;
count__4085 = G__4148;
i__4086 = G__4149;
continue;
} else {
var role = cljs.core.first.call(null,seq__4083__$1);
var temp__5753__auto___4150__$2 = cljs.core.get.call(null,fnspec,role);
if(cljs.core.truth_(temp__5753__auto___4150__$2)){
var spec_4151 = temp__5753__auto___4150__$2;
cljs.core.print.call(null,["\n ",cljs.core.name.call(null,role),":"].join(''),cljs.spec.alpha.describe.call(null,spec_4151));
} else {
}


var G__4152 = cljs.core.next.call(null,seq__4083__$1);
var G__4153 = null;
var G__4154 = (0);
var G__4155 = (0);
seq__4083 = G__4152;
chunk__4084 = G__4153;
count__4085 = G__4154;
i__4086 = G__4155;
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
return cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"type","type",1174270348),(((t instanceof cljs.core.ExceptionInfo))?new cljs.core.Symbol(null,"ExceptionInfo","ExceptionInfo",294935087,null):(((t instanceof Error))?cljs.core.symbol.call(null,"js",t.name):null
))], null),(function (){var temp__5753__auto__ = cljs.core.ex_message.call(null,t);
if(cljs.core.truth_(temp__5753__auto__)){
var msg = temp__5753__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"message","message",-406056002),msg], null);
} else {
return null;
}
})(),(function (){var temp__5753__auto__ = cljs.core.ex_data.call(null,t);
if(cljs.core.truth_(temp__5753__auto__)){
var ed = temp__5753__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"data","data",-232669377),ed], null);
} else {
return null;
}
})());
});
var via = (function (){var via = cljs.core.PersistentVector.EMPTY;
var t = o;
while(true){
if(cljs.core.truth_(t)){
var G__4156 = cljs.core.conj.call(null,via,t);
var G__4157 = cljs.core.ex_cause.call(null,t);
via = G__4156;
t = G__4157;
continue;
} else {
return via;
}
break;
}
})();
var root = cljs.core.peek.call(null,via);
return cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"via","via",-1904457336),cljs.core.vec.call(null,cljs.core.map.call(null,base,via)),new cljs.core.Keyword(null,"trace","trace",-1082747415),null], null),(function (){var temp__5753__auto__ = cljs.core.ex_message.call(null,root);
if(cljs.core.truth_(temp__5753__auto__)){
var root_msg = temp__5753__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"cause","cause",231901252),root_msg], null);
} else {
return null;
}
})(),(function (){var temp__5753__auto__ = cljs.core.ex_data.call(null,root);
if(cljs.core.truth_(temp__5753__auto__)){
var data = temp__5753__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"data","data",-232669377),data], null);
} else {
return null;
}
})(),(function (){var temp__5753__auto__ = new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358).cljs$core$IFn$_invoke$arity$1(cljs.core.ex_data.call(null,o));
if(cljs.core.truth_(temp__5753__auto__)){
var phase = temp__5753__auto__;
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"phase","phase",575722892),phase], null);
} else {
return null;
}
})());
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
var map__4160 = datafied_throwable;
var map__4160__$1 = (((((!((map__4160 == null))))?(((((map__4160.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4160.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4160):map__4160);
var via = cljs.core.get.call(null,map__4160__$1,new cljs.core.Keyword(null,"via","via",-1904457336));
var trace = cljs.core.get.call(null,map__4160__$1,new cljs.core.Keyword(null,"trace","trace",-1082747415));
var phase = cljs.core.get.call(null,map__4160__$1,new cljs.core.Keyword(null,"phase","phase",575722892),new cljs.core.Keyword(null,"execution","execution",253283524));
var map__4161 = cljs.core.last.call(null,via);
var map__4161__$1 = (((((!((map__4161 == null))))?(((((map__4161.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4161.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4161):map__4161);
var type = cljs.core.get.call(null,map__4161__$1,new cljs.core.Keyword(null,"type","type",1174270348));
var message = cljs.core.get.call(null,map__4161__$1,new cljs.core.Keyword(null,"message","message",-406056002));
var data = cljs.core.get.call(null,map__4161__$1,new cljs.core.Keyword(null,"data","data",-232669377));
var map__4162 = data;
var map__4162__$1 = (((((!((map__4162 == null))))?(((((map__4162.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4162.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4162):map__4162);
var problems = cljs.core.get.call(null,map__4162__$1,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814));
var fn = cljs.core.get.call(null,map__4162__$1,new cljs.core.Keyword("cljs.spec.alpha","fn","cljs.spec.alpha/fn",408600443));
var caller = cljs.core.get.call(null,map__4162__$1,new cljs.core.Keyword("cljs.spec.test.alpha","caller","cljs.spec.test.alpha/caller",-398302390));
var map__4163 = new cljs.core.Keyword(null,"data","data",-232669377).cljs$core$IFn$_invoke$arity$1(cljs.core.first.call(null,via));
var map__4163__$1 = (((((!((map__4163 == null))))?(((((map__4163.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4163.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4163):map__4163);
var top_data = map__4163__$1;
var source = cljs.core.get.call(null,map__4163__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397));
return cljs.core.assoc.call(null,(function (){var G__4168 = phase;
var G__4168__$1 = (((G__4168 instanceof cljs.core.Keyword))?G__4168.fqn:null);
switch (G__4168__$1) {
case "read-source":
var map__4169 = data;
var map__4169__$1 = (((((!((map__4169 == null))))?(((((map__4169.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4169.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4169):map__4169);
var line = cljs.core.get.call(null,map__4169__$1,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471));
var column = cljs.core.get.call(null,map__4169__$1,new cljs.core.Keyword("clojure.error","column","clojure.error/column",304721553));
var G__4171 = cljs.core.merge.call(null,new cljs.core.Keyword(null,"data","data",-232669377).cljs$core$IFn$_invoke$arity$1(cljs.core.second.call(null,via)),top_data);
var G__4171__$1 = (cljs.core.truth_(source)?cljs.core.assoc.call(null,G__4171,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),source):G__4171);
var G__4171__$2 = (cljs.core.truth_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null).call(null,source))?cljs.core.dissoc.call(null,G__4171__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397)):G__4171__$1);
if(cljs.core.truth_(message)){
return cljs.core.assoc.call(null,G__4171__$2,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message);
} else {
return G__4171__$2;
}

break;
case "compile-syntax-check":
case "compilation":
case "macro-syntax-check":
case "macroexpansion":
var G__4172 = top_data;
var G__4172__$1 = (cljs.core.truth_(source)?cljs.core.assoc.call(null,G__4172,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),source):G__4172);
var G__4172__$2 = (cljs.core.truth_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null).call(null,source))?cljs.core.dissoc.call(null,G__4172__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397)):G__4172__$1);
var G__4172__$3 = (cljs.core.truth_(type)?cljs.core.assoc.call(null,G__4172__$2,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type):G__4172__$2);
var G__4172__$4 = (cljs.core.truth_(message)?cljs.core.assoc.call(null,G__4172__$3,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message):G__4172__$3);
if(cljs.core.truth_(problems)){
return cljs.core.assoc.call(null,G__4172__$4,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595),data);
} else {
return G__4172__$4;
}

break;
case "read-eval-result":
case "print-eval-result":
var vec__4173 = cljs.core.first.call(null,trace);
var source__$1 = cljs.core.nth.call(null,vec__4173,(0),null);
var method = cljs.core.nth.call(null,vec__4173,(1),null);
var file = cljs.core.nth.call(null,vec__4173,(2),null);
var line = cljs.core.nth.call(null,vec__4173,(3),null);
var G__4176 = top_data;
var G__4176__$1 = (cljs.core.truth_(line)?cljs.core.assoc.call(null,G__4176,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471),line):G__4176);
var G__4176__$2 = (cljs.core.truth_(file)?cljs.core.assoc.call(null,G__4176__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),file):G__4176__$1);
var G__4176__$3 = (cljs.core.truth_((function (){var and__4115__auto__ = source__$1;
if(cljs.core.truth_(and__4115__auto__)){
return method;
} else {
return and__4115__auto__;
}
})())?cljs.core.assoc.call(null,G__4176__$2,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994),(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[source__$1,method],null))):G__4176__$2);
var G__4176__$4 = (cljs.core.truth_(type)?cljs.core.assoc.call(null,G__4176__$3,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type):G__4176__$3);
if(cljs.core.truth_(message)){
return cljs.core.assoc.call(null,G__4176__$4,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message);
} else {
return G__4176__$4;
}

break;
case "execution":
var vec__4177 = cljs.core.first.call(null,trace);
var source__$1 = cljs.core.nth.call(null,vec__4177,(0),null);
var method = cljs.core.nth.call(null,vec__4177,(1),null);
var file = cljs.core.nth.call(null,vec__4177,(2),null);
var line = cljs.core.nth.call(null,vec__4177,(3),null);
var file__$1 = cljs.core.first.call(null,cljs.core.remove.call(null,(function (p1__4159_SHARP_){
var or__4126__auto__ = (p1__4159_SHARP_ == null);
if(or__4126__auto__){
return or__4126__auto__;
} else {
return new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["NO_SOURCE_PATH",null,"NO_SOURCE_FILE",null], null), null).call(null,p1__4159_SHARP_);
}
}),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(caller),file], null)));
var err_line = (function (){var or__4126__auto__ = new cljs.core.Keyword(null,"line","line",212345235).cljs$core$IFn$_invoke$arity$1(caller);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return line;
}
})();
var G__4180 = new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890),type], null);
var G__4180__$1 = (cljs.core.truth_(err_line)?cljs.core.assoc.call(null,G__4180,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471),err_line):G__4180);
var G__4180__$2 = (cljs.core.truth_(message)?cljs.core.assoc.call(null,G__4180__$1,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742),message):G__4180__$1);
var G__4180__$3 = (cljs.core.truth_((function (){var or__4126__auto__ = fn;
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
})())?cljs.core.assoc.call(null,G__4180__$2,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994),(function (){var or__4126__auto__ = fn;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return (new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[source__$1,method],null));
}
})()):G__4180__$2);
var G__4180__$4 = (cljs.core.truth_(file__$1)?cljs.core.assoc.call(null,G__4180__$3,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397),file__$1):G__4180__$3);
if(cljs.core.truth_(problems)){
return cljs.core.assoc.call(null,G__4180__$4,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595),data);
} else {
return G__4180__$4;
}

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__4168__$1)].join('')));

}
})(),new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358),phase);
});
/**
 * Returns a string from exception data, as produced by ex-triage.
 *   The first line summarizes the exception phase and location.
 *   The subsequent lines describe the cause.
 */
cljs.repl.ex_str = (function cljs$repl$ex_str(p__4184){
var map__4185 = p__4184;
var map__4185__$1 = (((((!((map__4185 == null))))?(((((map__4185.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__4185.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__4185):map__4185);
var triage_data = map__4185__$1;
var phase = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","phase","clojure.error/phase",275140358));
var source = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","source","clojure.error/source",-2011936397));
var line = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","line","clojure.error/line",-1816287471));
var column = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","column","clojure.error/column",304721553));
var symbol = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","symbol","clojure.error/symbol",1544821994));
var class$ = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","class","clojure.error/class",278435890));
var cause = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","cause","clojure.error/cause",-1879175742));
var spec = cljs.core.get.call(null,map__4185__$1,new cljs.core.Keyword("clojure.error","spec","clojure.error/spec",2055032595));
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
var class_name = cljs.core.name.call(null,(function (){var or__4126__auto__ = class$;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return "";
}
})());
var simple_class = class_name;
var cause_type = ((cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["RuntimeException",null,"Exception",null], null), null),simple_class))?"":[" (",simple_class,")"].join(''));
var format = goog.string.format;
var G__4187 = phase;
var G__4187__$1 = (((G__4187 instanceof cljs.core.Keyword))?G__4187.fqn:null);
switch (G__4187__$1) {
case "read-source":
return format.call(null,"Syntax error reading source at (%s).\n%s\n",loc,cause);

break;
case "macro-syntax-check":
return format.call(null,"Syntax error macroexpanding %sat (%s).\n%s",(cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):""),loc,(cljs.core.truth_(spec)?(function (){var sb__4667__auto__ = (new goog.string.StringBuffer());
var _STAR_print_newline_STAR__orig_val__4188_4197 = cljs.core._STAR_print_newline_STAR_;
var _STAR_print_fn_STAR__orig_val__4189_4198 = cljs.core._STAR_print_fn_STAR_;
var _STAR_print_newline_STAR__temp_val__4190_4199 = true;
var _STAR_print_fn_STAR__temp_val__4191_4200 = (function (x__4668__auto__){
return sb__4667__auto__.append(x__4668__auto__);
});
(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__temp_val__4190_4199);

(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__temp_val__4191_4200);

try{cljs.spec.alpha.explain_out.call(null,cljs.core.update.call(null,spec,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814),(function (probs){
return cljs.core.map.call(null,(function (p1__4182_SHARP_){
return cljs.core.dissoc.call(null,p1__4182_SHARP_,new cljs.core.Keyword(null,"in","in",-1531184865));
}),probs);
}))
);
}finally {(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__orig_val__4189_4198);

(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__orig_val__4188_4197);
}
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(sb__4667__auto__);
})():format.call(null,"%s\n",cause)));

break;
case "macroexpansion":
return format.call(null,"Unexpected error%s macroexpanding %sat (%s).\n%s\n",cause_type,(cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):""),loc,cause);

break;
case "compile-syntax-check":
return format.call(null,"Syntax error%s compiling %sat (%s).\n%s\n",cause_type,(cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):""),loc,cause);

break;
case "compilation":
return format.call(null,"Unexpected error%s compiling %sat (%s).\n%s\n",cause_type,(cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):""),loc,cause);

break;
case "read-eval-result":
return format.call(null,"Error reading eval result%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause);

break;
case "print-eval-result":
return format.call(null,"Error printing return value%s at %s (%s).\n%s\n",cause_type,symbol,loc,cause);

break;
case "execution":
if(cljs.core.truth_(spec)){
return format.call(null,"Execution error - invalid arguments to %s at (%s).\n%s",symbol,loc,(function (){var sb__4667__auto__ = (new goog.string.StringBuffer());
var _STAR_print_newline_STAR__orig_val__4192_4201 = cljs.core._STAR_print_newline_STAR_;
var _STAR_print_fn_STAR__orig_val__4193_4202 = cljs.core._STAR_print_fn_STAR_;
var _STAR_print_newline_STAR__temp_val__4194_4203 = true;
var _STAR_print_fn_STAR__temp_val__4195_4204 = (function (x__4668__auto__){
return sb__4667__auto__.append(x__4668__auto__);
});
(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__temp_val__4194_4203);

(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__temp_val__4195_4204);

try{cljs.spec.alpha.explain_out.call(null,cljs.core.update.call(null,spec,new cljs.core.Keyword("cljs.spec.alpha","problems","cljs.spec.alpha/problems",447400814),(function (probs){
return cljs.core.map.call(null,(function (p1__4183_SHARP_){
return cljs.core.dissoc.call(null,p1__4183_SHARP_,new cljs.core.Keyword(null,"in","in",-1531184865));
}),probs);
}))
);
}finally {(cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR__orig_val__4193_4202);

(cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR__orig_val__4192_4201);
}
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(sb__4667__auto__);
})());
} else {
return format.call(null,"Execution error%s at %s(%s).\n%s\n",cause_type,(cljs.core.truth_(symbol)?[cljs.core.str.cljs$core$IFn$_invoke$arity$1(symbol)," "].join(''):""),loc,cause);
}

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__4187__$1)].join('')));

}
});
cljs.repl.error__GT_str = (function cljs$repl$error__GT_str(error){
return cljs.repl.ex_str.call(null,cljs.repl.ex_triage.call(null,cljs.repl.Error__GT_map.call(null,error)));
});

//# sourceMappingURL=repl.js.map
