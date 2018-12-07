// Compiled by ClojureScript 1.10.238 {:target :nodejs}
goog.provide('cljs.repl');
goog.require('cljs.core');
goog.require('cljs.spec.alpha');
cljs.repl.print_doc = (function cljs$repl$print_doc(p__13599){
var map__13600 = p__13599;
var map__13600__$1 = ((((!((map__13600 == null)))?(((((map__13600.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__13600.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__13600):map__13600);
var m = map__13600__$1;
var n = cljs.core.get.call(null,map__13600__$1,new cljs.core.Keyword(null,"ns","ns",441598760));
var nm = cljs.core.get.call(null,map__13600__$1,new cljs.core.Keyword(null,"name","name",1843675177));
cljs.core.println.call(null,"-------------------------");

cljs.core.println.call(null,[cljs.core.str.cljs$core$IFn$_invoke$arity$1((function (){var temp__5457__auto__ = new cljs.core.Keyword(null,"ns","ns",441598760).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(temp__5457__auto__)){
var ns = temp__5457__auto__;
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(ns),"/"].join('');
} else {
return null;
}
})()),cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(m))].join(''));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"Protocol");
} else {
}

if(cljs.core.truth_(new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m))){
var seq__13602_13624 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"forms","forms",2045992350).cljs$core$IFn$_invoke$arity$1(m));
var chunk__13603_13625 = null;
var count__13604_13626 = (0);
var i__13605_13627 = (0);
while(true){
if((i__13605_13627 < count__13604_13626)){
var f_13628 = cljs.core._nth.call(null,chunk__13603_13625,i__13605_13627);
cljs.core.println.call(null,"  ",f_13628);


var G__13629 = seq__13602_13624;
var G__13630 = chunk__13603_13625;
var G__13631 = count__13604_13626;
var G__13632 = (i__13605_13627 + (1));
seq__13602_13624 = G__13629;
chunk__13603_13625 = G__13630;
count__13604_13626 = G__13631;
i__13605_13627 = G__13632;
continue;
} else {
var temp__5457__auto___13633 = cljs.core.seq.call(null,seq__13602_13624);
if(temp__5457__auto___13633){
var seq__13602_13634__$1 = temp__5457__auto___13633;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__13602_13634__$1)){
var c__4319__auto___13635 = cljs.core.chunk_first.call(null,seq__13602_13634__$1);
var G__13636 = cljs.core.chunk_rest.call(null,seq__13602_13634__$1);
var G__13637 = c__4319__auto___13635;
var G__13638 = cljs.core.count.call(null,c__4319__auto___13635);
var G__13639 = (0);
seq__13602_13624 = G__13636;
chunk__13603_13625 = G__13637;
count__13604_13626 = G__13638;
i__13605_13627 = G__13639;
continue;
} else {
var f_13640 = cljs.core.first.call(null,seq__13602_13634__$1);
cljs.core.println.call(null,"  ",f_13640);


var G__13641 = cljs.core.next.call(null,seq__13602_13634__$1);
var G__13642 = null;
var G__13643 = (0);
var G__13644 = (0);
seq__13602_13624 = G__13641;
chunk__13603_13625 = G__13642;
count__13604_13626 = G__13643;
i__13605_13627 = G__13644;
continue;
}
} else {
}
}
break;
}
} else {
if(cljs.core.truth_(new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m))){
var arglists_13645 = new cljs.core.Keyword(null,"arglists","arglists",1661989754).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_((function (){var or__3922__auto__ = new cljs.core.Keyword(null,"macro","macro",-867863404).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(or__3922__auto__)){
return or__3922__auto__;
} else {
return new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m);
}
})())){
cljs.core.prn.call(null,arglists_13645);
} else {
cljs.core.prn.call(null,((cljs.core._EQ_.call(null,new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.first.call(null,arglists_13645)))?cljs.core.second.call(null,arglists_13645):arglists_13645));
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

if(cljs.core.truth_(new cljs.core.Keyword(null,"repl-special-function","repl-special-function",1262603725).cljs$core$IFn$_invoke$arity$1(m))){
cljs.core.println.call(null,"REPL Special Function");
} else {
}

cljs.core.println.call(null," ",new cljs.core.Keyword(null,"doc","doc",1913296891).cljs$core$IFn$_invoke$arity$1(m));

if(cljs.core.truth_(new cljs.core.Keyword(null,"protocol","protocol",652470118).cljs$core$IFn$_invoke$arity$1(m))){
var seq__13606_13646 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"methods","methods",453930866).cljs$core$IFn$_invoke$arity$1(m));
var chunk__13607_13647 = null;
var count__13608_13648 = (0);
var i__13609_13649 = (0);
while(true){
if((i__13609_13649 < count__13608_13648)){
var vec__13610_13650 = cljs.core._nth.call(null,chunk__13607_13647,i__13609_13649);
var name_13651 = cljs.core.nth.call(null,vec__13610_13650,(0),null);
var map__13613_13652 = cljs.core.nth.call(null,vec__13610_13650,(1),null);
var map__13613_13653__$1 = ((((!((map__13613_13652 == null)))?(((((map__13613_13652.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__13613_13652.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__13613_13652):map__13613_13652);
var doc_13654 = cljs.core.get.call(null,map__13613_13653__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_13655 = cljs.core.get.call(null,map__13613_13653__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name_13651);

cljs.core.println.call(null," ",arglists_13655);

if(cljs.core.truth_(doc_13654)){
cljs.core.println.call(null," ",doc_13654);
} else {
}


var G__13656 = seq__13606_13646;
var G__13657 = chunk__13607_13647;
var G__13658 = count__13608_13648;
var G__13659 = (i__13609_13649 + (1));
seq__13606_13646 = G__13656;
chunk__13607_13647 = G__13657;
count__13608_13648 = G__13658;
i__13609_13649 = G__13659;
continue;
} else {
var temp__5457__auto___13660 = cljs.core.seq.call(null,seq__13606_13646);
if(temp__5457__auto___13660){
var seq__13606_13661__$1 = temp__5457__auto___13660;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__13606_13661__$1)){
var c__4319__auto___13662 = cljs.core.chunk_first.call(null,seq__13606_13661__$1);
var G__13663 = cljs.core.chunk_rest.call(null,seq__13606_13661__$1);
var G__13664 = c__4319__auto___13662;
var G__13665 = cljs.core.count.call(null,c__4319__auto___13662);
var G__13666 = (0);
seq__13606_13646 = G__13663;
chunk__13607_13647 = G__13664;
count__13608_13648 = G__13665;
i__13609_13649 = G__13666;
continue;
} else {
var vec__13615_13667 = cljs.core.first.call(null,seq__13606_13661__$1);
var name_13668 = cljs.core.nth.call(null,vec__13615_13667,(0),null);
var map__13618_13669 = cljs.core.nth.call(null,vec__13615_13667,(1),null);
var map__13618_13670__$1 = ((((!((map__13618_13669 == null)))?(((((map__13618_13669.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__13618_13669.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__13618_13669):map__13618_13669);
var doc_13671 = cljs.core.get.call(null,map__13618_13670__$1,new cljs.core.Keyword(null,"doc","doc",1913296891));
var arglists_13672 = cljs.core.get.call(null,map__13618_13670__$1,new cljs.core.Keyword(null,"arglists","arglists",1661989754));
cljs.core.println.call(null);

cljs.core.println.call(null," ",name_13668);

cljs.core.println.call(null," ",arglists_13672);

if(cljs.core.truth_(doc_13671)){
cljs.core.println.call(null," ",doc_13671);
} else {
}


var G__13673 = cljs.core.next.call(null,seq__13606_13661__$1);
var G__13674 = null;
var G__13675 = (0);
var G__13676 = (0);
seq__13606_13646 = G__13673;
chunk__13607_13647 = G__13674;
count__13608_13648 = G__13675;
i__13609_13649 = G__13676;
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
var temp__5457__auto__ = cljs.spec.alpha.get_spec.call(null,cljs.core.symbol.call(null,[cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.ns_name.call(null,n))].join(''),cljs.core.name.call(null,nm)));
if(cljs.core.truth_(temp__5457__auto__)){
var fnspec = temp__5457__auto__;
cljs.core.print.call(null,"Spec");

var seq__13620 = cljs.core.seq.call(null,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"args","args",1315556576),new cljs.core.Keyword(null,"ret","ret",-468222814),new cljs.core.Keyword(null,"fn","fn",-1175266204)], null));
var chunk__13621 = null;
var count__13622 = (0);
var i__13623 = (0);
while(true){
if((i__13623 < count__13622)){
var role = cljs.core._nth.call(null,chunk__13621,i__13623);
var temp__5457__auto___13677__$1 = cljs.core.get.call(null,fnspec,role);
if(cljs.core.truth_(temp__5457__auto___13677__$1)){
var spec_13678 = temp__5457__auto___13677__$1;
cljs.core.print.call(null,["\n ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.name.call(null,role)),":"].join(''),cljs.spec.alpha.describe.call(null,spec_13678));
} else {
}


var G__13679 = seq__13620;
var G__13680 = chunk__13621;
var G__13681 = count__13622;
var G__13682 = (i__13623 + (1));
seq__13620 = G__13679;
chunk__13621 = G__13680;
count__13622 = G__13681;
i__13623 = G__13682;
continue;
} else {
var temp__5457__auto____$1 = cljs.core.seq.call(null,seq__13620);
if(temp__5457__auto____$1){
var seq__13620__$1 = temp__5457__auto____$1;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__13620__$1)){
var c__4319__auto__ = cljs.core.chunk_first.call(null,seq__13620__$1);
var G__13683 = cljs.core.chunk_rest.call(null,seq__13620__$1);
var G__13684 = c__4319__auto__;
var G__13685 = cljs.core.count.call(null,c__4319__auto__);
var G__13686 = (0);
seq__13620 = G__13683;
chunk__13621 = G__13684;
count__13622 = G__13685;
i__13623 = G__13686;
continue;
} else {
var role = cljs.core.first.call(null,seq__13620__$1);
var temp__5457__auto___13687__$2 = cljs.core.get.call(null,fnspec,role);
if(cljs.core.truth_(temp__5457__auto___13687__$2)){
var spec_13688 = temp__5457__auto___13687__$2;
cljs.core.print.call(null,["\n ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.name.call(null,role)),":"].join(''),cljs.spec.alpha.describe.call(null,spec_13688));
} else {
}


var G__13689 = cljs.core.next.call(null,seq__13620__$1);
var G__13690 = null;
var G__13691 = (0);
var G__13692 = (0);
seq__13620 = G__13689;
chunk__13621 = G__13690;
count__13622 = G__13691;
i__13623 = G__13692;
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

//# sourceMappingURL=repl.js.map
