goog.provide('hipo.interpreter');
hipo.interpreter.set_attribute_BANG_ = (function hipo$interpreter$set_attribute_BANG_(el,ns,tag,sok,ov,nv,p__37260){
var map__37261 = p__37260;
var map__37261__$1 = (((((!((map__37261 == null))))?(((((map__37261.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37261.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37261):map__37261);
var m = map__37261__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37261__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
if((!((ov === nv)))){
var temp__5733__auto__ = hipo.hiccup.listener_name__GT_event_name(cljs.core.name(sok));
if(cljs.core.truth_(temp__5733__auto__)){
var en = temp__5733__auto__;
if((!(((cljs.core.map_QMARK_(ov)) && (cljs.core.map_QMARK_(nv)) && ((new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(ov) === new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(nv))))))){
var b__37191__auto__ = (function (){
var hn = ["hipo_listener_",en].join('');
var temp__5733__auto___37360__$1 = (el[hn]);
if(cljs.core.truth_(temp__5733__auto___37360__$1)){
var l_37361 = temp__5733__auto___37360__$1;
el.removeEventListener(en,l_37361);
} else {
}

var temp__5735__auto__ = (function (){var or__4126__auto__ = new cljs.core.Keyword(null,"fn","fn",-1175266204).cljs$core$IFn$_invoke$arity$1(nv);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return nv;
}
})();
if(cljs.core.truth_(temp__5735__auto__)){
var nv__$1 = temp__5735__auto__;
el.addEventListener(en,nv__$1);

return (el[hn] = nv__$1);
} else {
return null;
}
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,(cljs.core.truth_(nv)?new cljs.core.Keyword(null,"update-handler","update-handler",1389859106):new cljs.core.Keyword(null,"remove-handler","remove-handler",389960218)),cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"name","name",1843675177),sok,new cljs.core.Keyword(null,"old-value","old-value",862546795),ov], null),(cljs.core.truth_(nv)?new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"new-value","new-value",1087038368),nv], null):null)], 0)));
}
} else {
return null;
}
} else {
var b__37191__auto__ = (function (){
return hipo.attribute.set_value_BANG_(el,m,ns,tag,sok,ov,nv);
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,(cljs.core.truth_(nv)?new cljs.core.Keyword(null,"update-attribute","update-attribute",102770530):new cljs.core.Keyword(null,"remove-attribute","remove-attribute",552745626)),cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"name","name",1843675177),sok,new cljs.core.Keyword(null,"old-value","old-value",862546795),ov], null),(cljs.core.truth_(nv)?new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"new-value","new-value",1087038368),nv], null):null)], 0)));
}
}
} else {
return null;
}
});
hipo.interpreter.append_children_BANG_ = (function hipo$interpreter$append_children_BANG_(el,v,m){
if(cljs.core.vector_QMARK_(v)){
} else {
throw (new Error("Assert failed: (vector? v)"));
}

var v__$1 = hipo.hiccup.flatten_children(v);
while(true){
if(cljs.core.empty_QMARK_(v__$1)){
return null;
} else {
var temp__5733__auto___37374 = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(v__$1,(0));
if(cljs.core.truth_(temp__5733__auto___37374)){
var h_37375 = temp__5733__auto___37374;
el.appendChild((hipo.interpreter.create_child.cljs$core$IFn$_invoke$arity$2 ? hipo.interpreter.create_child.cljs$core$IFn$_invoke$arity$2(h_37375,m) : hipo.interpreter.create_child.call(null,h_37375,m)));
} else {
}

var G__37376 = cljs.core.rest(v__$1);
v__$1 = G__37376;
continue;
}
break;
}
});
hipo.interpreter.default_create_element = (function hipo$interpreter$default_create_element(ns,tag,attrs,m){
var el = hipo.dom.create_element(ns,tag);
var seq__37263_37377 = cljs.core.seq(attrs);
var chunk__37264_37378 = null;
var count__37265_37379 = (0);
var i__37266_37380 = (0);
while(true){
if((i__37266_37380 < count__37265_37379)){
var vec__37273_37381 = chunk__37264_37378.cljs$core$IIndexed$_nth$arity$2(null,i__37266_37380);
var sok_37382 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37273_37381,(0),null);
var v_37383 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37273_37381,(1),null);
if(cljs.core.truth_(v_37383)){
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok_37382,null,v_37383,m);
} else {
}


var G__37384 = seq__37263_37377;
var G__37385 = chunk__37264_37378;
var G__37386 = count__37265_37379;
var G__37387 = (i__37266_37380 + (1));
seq__37263_37377 = G__37384;
chunk__37264_37378 = G__37385;
count__37265_37379 = G__37386;
i__37266_37380 = G__37387;
continue;
} else {
var temp__5735__auto___37390 = cljs.core.seq(seq__37263_37377);
if(temp__5735__auto___37390){
var seq__37263_37391__$1 = temp__5735__auto___37390;
if(cljs.core.chunked_seq_QMARK_(seq__37263_37391__$1)){
var c__4556__auto___37395 = cljs.core.chunk_first(seq__37263_37391__$1);
var G__37396 = cljs.core.chunk_rest(seq__37263_37391__$1);
var G__37397 = c__4556__auto___37395;
var G__37398 = cljs.core.count(c__4556__auto___37395);
var G__37399 = (0);
seq__37263_37377 = G__37396;
chunk__37264_37378 = G__37397;
count__37265_37379 = G__37398;
i__37266_37380 = G__37399;
continue;
} else {
var vec__37276_37401 = cljs.core.first(seq__37263_37391__$1);
var sok_37402 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37276_37401,(0),null);
var v_37403 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37276_37401,(1),null);
if(cljs.core.truth_(v_37403)){
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok_37402,null,v_37403,m);
} else {
}


var G__37405 = cljs.core.next(seq__37263_37391__$1);
var G__37406 = null;
var G__37407 = (0);
var G__37408 = (0);
seq__37263_37377 = G__37405;
chunk__37264_37378 = G__37406;
count__37265_37379 = G__37407;
i__37266_37380 = G__37408;
continue;
}
} else {
}
}
break;
}

return el;
});
hipo.interpreter.create_element = (function hipo$interpreter$create_element(ns,tag,attrs,m){
var temp__5733__auto__ = new cljs.core.Keyword(null,"create-element-fn","create-element-fn",827380427).cljs$core$IFn$_invoke$arity$1(m);
if(cljs.core.truth_(temp__5733__auto__)){
var f = temp__5733__auto__;
return (f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(ns,tag,attrs,m) : f.call(null,ns,tag,attrs,m));
} else {
return hipo.interpreter.default_create_element(ns,tag,attrs,m);
}
});
hipo.interpreter.create_vector = (function hipo$interpreter$create_vector(h,m){
if(cljs.core.vector_QMARK_(h)){
} else {
throw (new Error("Assert failed: (vector? h)"));
}

var key = hipo.hiccup.keyns(h);
var tag = hipo.hiccup.tag(h);
var attrs = hipo.hiccup.attributes(h);
var children = hipo.hiccup.children(h);
var el = hipo.interpreter.create_element(hipo.hiccup.key__GT_namespace(key,m),tag,attrs,m);
if(cljs.core.truth_(children)){
hipo.interpreter.append_children_BANG_(el,children,m);
} else {
}

return el;
});
hipo.interpreter.create_child = (function hipo$interpreter$create_child(o,m){
if(((hipo.hiccup.literal_QMARK_(o)) || (cljs.core.vector_QMARK_(o)))){
} else {
throw (new Error("Assert failed: (or (hic/literal? o) (vector? o))"));
}

if(hipo.hiccup.literal_QMARK_(o)){
return document.createTextNode(o);
} else {
return hipo.interpreter.create_vector(o,m);
}
});
hipo.interpreter.append_to_parent = (function hipo$interpreter$append_to_parent(el,o,m){
if(cljs.core.seq_QMARK_(o)){
return hipo.interpreter.append_children_BANG_(el,cljs.core.vec(o),m);
} else {
if((!((o == null)))){
return el.appendChild(hipo.interpreter.create_child(o,m));
} else {
return null;
}
}
});
hipo.interpreter.create = (function hipo$interpreter$create(o,m){
if(cljs.core.seq_QMARK_(o)){
var f = document.createDocumentFragment();
hipo.interpreter.append_children_BANG_(f,cljs.core.vec(o),m);

return f;
} else {
if((!((o == null)))){
return hipo.interpreter.create_child(o,m);
} else {
return null;
}
}
});
hipo.interpreter.reconciliate_attributes_BANG_ = (function hipo$interpreter$reconciliate_attributes_BANG_(el,ns,tag,om,nm,m){
var seq__37279_37416 = cljs.core.seq(nm);
var chunk__37281_37417 = null;
var count__37282_37418 = (0);
var i__37283_37419 = (0);
while(true){
if((i__37283_37419 < count__37282_37418)){
var vec__37291_37421 = chunk__37281_37417.cljs$core$IIndexed$_nth$arity$2(null,i__37283_37419);
var sok_37422 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37291_37421,(0),null);
var nv_37423 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37291_37421,(1),null);
var ov_37424 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,sok_37422);
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok_37422,ov_37424,nv_37423,m);


var G__37425 = seq__37279_37416;
var G__37426 = chunk__37281_37417;
var G__37427 = count__37282_37418;
var G__37428 = (i__37283_37419 + (1));
seq__37279_37416 = G__37425;
chunk__37281_37417 = G__37426;
count__37282_37418 = G__37427;
i__37283_37419 = G__37428;
continue;
} else {
var temp__5735__auto___37433 = cljs.core.seq(seq__37279_37416);
if(temp__5735__auto___37433){
var seq__37279_37434__$1 = temp__5735__auto___37433;
if(cljs.core.chunked_seq_QMARK_(seq__37279_37434__$1)){
var c__4556__auto___37435 = cljs.core.chunk_first(seq__37279_37434__$1);
var G__37436 = cljs.core.chunk_rest(seq__37279_37434__$1);
var G__37437 = c__4556__auto___37435;
var G__37438 = cljs.core.count(c__4556__auto___37435);
var G__37439 = (0);
seq__37279_37416 = G__37436;
chunk__37281_37417 = G__37437;
count__37282_37418 = G__37438;
i__37283_37419 = G__37439;
continue;
} else {
var vec__37294_37441 = cljs.core.first(seq__37279_37434__$1);
var sok_37442 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37294_37441,(0),null);
var nv_37443 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37294_37441,(1),null);
var ov_37445 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,sok_37442);
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok_37442,ov_37445,nv_37443,m);


var G__37446 = cljs.core.next(seq__37279_37434__$1);
var G__37447 = null;
var G__37448 = (0);
var G__37449 = (0);
seq__37279_37416 = G__37446;
chunk__37281_37417 = G__37447;
count__37282_37418 = G__37448;
i__37283_37419 = G__37449;
continue;
}
} else {
}
}
break;
}

var seq__37297 = cljs.core.seq(clojure.set.difference.cljs$core$IFn$_invoke$arity$2(cljs.core.set(cljs.core.keys(om)),cljs.core.set(cljs.core.keys(nm))));
var chunk__37298 = null;
var count__37299 = (0);
var i__37300 = (0);
while(true){
if((i__37300 < count__37299)){
var sok = chunk__37298.cljs$core$IIndexed$_nth$arity$2(null,i__37300);
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok,cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,sok),null,m);


var G__37456 = seq__37297;
var G__37457 = chunk__37298;
var G__37458 = count__37299;
var G__37459 = (i__37300 + (1));
seq__37297 = G__37456;
chunk__37298 = G__37457;
count__37299 = G__37458;
i__37300 = G__37459;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__37297);
if(temp__5735__auto__){
var seq__37297__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__37297__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__37297__$1);
var G__37460 = cljs.core.chunk_rest(seq__37297__$1);
var G__37461 = c__4556__auto__;
var G__37462 = cljs.core.count(c__4556__auto__);
var G__37463 = (0);
seq__37297 = G__37460;
chunk__37298 = G__37461;
count__37299 = G__37462;
i__37300 = G__37463;
continue;
} else {
var sok = cljs.core.first(seq__37297__$1);
hipo.interpreter.set_attribute_BANG_(el,ns,tag,sok,cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,sok),null,m);


var G__37465 = cljs.core.next(seq__37297__$1);
var G__37466 = null;
var G__37467 = (0);
var G__37468 = (0);
seq__37297 = G__37465;
chunk__37298 = G__37466;
count__37299 = G__37467;
i__37300 = G__37468;
continue;
}
} else {
return null;
}
}
break;
}
});
hipo.interpreter.child_key = (function hipo$interpreter$child_key(h){
return new cljs.core.Keyword("hipo","key","hipo/key",-1519246363).cljs$core$IFn$_invoke$arity$1(cljs.core.meta(h));
});
hipo.interpreter.keyed_children__GT_indexed_map = (function hipo$interpreter$keyed_children__GT_indexed_map(v){
return cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,(function (){var iter__4529__auto__ = (function hipo$interpreter$keyed_children__GT_indexed_map_$_iter__37301(s__37302){
return (new cljs.core.LazySeq(null,(function (){
var s__37302__$1 = s__37302;
while(true){
var temp__5735__auto__ = cljs.core.seq(s__37302__$1);
if(temp__5735__auto__){
var s__37302__$2 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(s__37302__$2)){
var c__4527__auto__ = cljs.core.chunk_first(s__37302__$2);
var size__4528__auto__ = cljs.core.count(c__4527__auto__);
var b__37304 = cljs.core.chunk_buffer(size__4528__auto__);
if((function (){var i__37303 = (0);
while(true){
if((i__37303 < size__4528__auto__)){
var ih = cljs.core._nth(c__4527__auto__,i__37303);
cljs.core.chunk_append(b__37304,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [hipo.interpreter.child_key(cljs.core.nth.cljs$core$IFn$_invoke$arity$2(ih,(1))),ih], null));

var G__37474 = (i__37303 + (1));
i__37303 = G__37474;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons(cljs.core.chunk(b__37304),hipo$interpreter$keyed_children__GT_indexed_map_$_iter__37301(cljs.core.chunk_rest(s__37302__$2)));
} else {
return cljs.core.chunk_cons(cljs.core.chunk(b__37304),null);
}
} else {
var ih = cljs.core.first(s__37302__$2);
return cljs.core.cons(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [hipo.interpreter.child_key(cljs.core.nth.cljs$core$IFn$_invoke$arity$2(ih,(1))),ih], null),hipo$interpreter$keyed_children__GT_indexed_map_$_iter__37301(cljs.core.rest(s__37302__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__4529__auto__(cljs.core.map_indexed.cljs$core$IFn$_invoke$arity$2((function (idx,itm){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [idx,itm], null);
}),v));
})());
});
/**
 * Reconciliate a vector of children based on their associated key.
 */
hipo.interpreter.reconciliate_keyed_children_BANG_ = (function hipo$interpreter$reconciliate_keyed_children_BANG_(el,och,nch,p__37305){
var map__37306 = p__37305;
var map__37306__$1 = (((((!((map__37306 == null))))?(((((map__37306.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37306.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37306):map__37306);
var m = map__37306__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37306__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
var om = hipo.interpreter.keyed_children__GT_indexed_map(och);
var nm = hipo.interpreter.keyed_children__GT_indexed_map(nch);
var cs = hipo.dom.children.cljs$core$IFn$_invoke$arity$2(el,cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.max,clojure.set.intersection.cljs$core$IFn$_invoke$arity$2(cljs.core.set(cljs.core.keys(nm)),cljs.core.set(cljs.core.keys(om)))));
var seq__37308_37482 = cljs.core.seq(nm);
var chunk__37309_37483 = null;
var count__37310_37484 = (0);
var i__37311_37485 = (0);
while(true){
if((i__37311_37485 < count__37310_37484)){
var vec__37330_37489 = chunk__37309_37483.cljs$core$IIndexed$_nth$arity$2(null,i__37311_37485);
var i_37490 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37330_37489,(0),null);
var vec__37333_37491 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37330_37489,(1),null);
var ii_37492 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37333_37491,(0),null);
var h_37493 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37333_37491,(1),null);
var temp__5733__auto___37494 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,i_37490);
if(cljs.core.truth_(temp__5733__auto___37494)){
var vec__37336_37495 = temp__5733__auto___37494;
var iii_37496 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37336_37495,(0),null);
var oh_37497 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37336_37495,(1),null);
var cel_37499 = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(cs,iii_37496);
if((ii_37492 === iii_37496)){
(hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4 ? hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4(cel_37499,oh_37497,h_37493,m) : hipo.interpreter.reconciliate_BANG_.call(null,cel_37499,oh_37497,h_37493,m));
} else {
var b__37191__auto___37501 = ((function (seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,cel_37499,vec__37336_37495,iii_37496,oh_37497,temp__5733__auto___37494,vec__37330_37489,i_37490,vec__37333_37491,ii_37492,h_37493,om,nm,cs,map__37306,map__37306__$1,m,interceptors){
return (function (){
var ncel = el.removeChild(cel_37499);
(hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4 ? hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4(ncel,oh_37497,h_37493,m) : hipo.interpreter.reconciliate_BANG_.call(null,ncel,oh_37497,h_37493,m));

return hipo.dom.insert_child_BANG_(el,ii_37492,ncel);
});})(seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,cel_37499,vec__37336_37495,iii_37496,oh_37497,temp__5733__auto___37494,vec__37330_37489,i_37490,vec__37333_37491,ii_37492,h_37493,om,nm,cs,map__37306,map__37306__$1,m,interceptors))
;
var v__37192__auto___37502 = interceptors;
if(((cljs.core.not(v__37192__auto___37502)) || (cljs.core.empty_QMARK_(v__37192__auto___37502)))){
b__37191__auto___37501();
} else {
hipo.interceptor.call(b__37191__auto___37501,v__37192__auto___37502,new cljs.core.Keyword(null,"move","move",-2110884309),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),h_37493,new cljs.core.Keyword(null,"index","index",-1531685915),ii_37492], null));
}
}
} else {
var b__37191__auto___37511 = ((function (seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,temp__5733__auto___37494,vec__37330_37489,i_37490,vec__37333_37491,ii_37492,h_37493,om,nm,cs,map__37306,map__37306__$1,m,interceptors){
return (function (){
return hipo.dom.insert_child_BANG_(el,ii_37492,hipo.interpreter.create_child(h_37493,m));
});})(seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,temp__5733__auto___37494,vec__37330_37489,i_37490,vec__37333_37491,ii_37492,h_37493,om,nm,cs,map__37306,map__37306__$1,m,interceptors))
;
var v__37192__auto___37512 = interceptors;
if(((cljs.core.not(v__37192__auto___37512)) || (cljs.core.empty_QMARK_(v__37192__auto___37512)))){
b__37191__auto___37511();
} else {
hipo.interceptor.call(b__37191__auto___37511,v__37192__auto___37512,new cljs.core.Keyword(null,"insert","insert",1286475395),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),h_37493,new cljs.core.Keyword(null,"index","index",-1531685915),ii_37492], null));
}
}


var G__37514 = seq__37308_37482;
var G__37515 = chunk__37309_37483;
var G__37516 = count__37310_37484;
var G__37517 = (i__37311_37485 + (1));
seq__37308_37482 = G__37514;
chunk__37309_37483 = G__37515;
count__37310_37484 = G__37516;
i__37311_37485 = G__37517;
continue;
} else {
var temp__5735__auto___37518 = cljs.core.seq(seq__37308_37482);
if(temp__5735__auto___37518){
var seq__37308_37519__$1 = temp__5735__auto___37518;
if(cljs.core.chunked_seq_QMARK_(seq__37308_37519__$1)){
var c__4556__auto___37522 = cljs.core.chunk_first(seq__37308_37519__$1);
var G__37523 = cljs.core.chunk_rest(seq__37308_37519__$1);
var G__37524 = c__4556__auto___37522;
var G__37525 = cljs.core.count(c__4556__auto___37522);
var G__37526 = (0);
seq__37308_37482 = G__37523;
chunk__37309_37483 = G__37524;
count__37310_37484 = G__37525;
i__37311_37485 = G__37526;
continue;
} else {
var vec__37339_37529 = cljs.core.first(seq__37308_37519__$1);
var i_37530 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37339_37529,(0),null);
var vec__37342_37531 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37339_37529,(1),null);
var ii_37532 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37342_37531,(0),null);
var h_37533 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37342_37531,(1),null);
var temp__5733__auto___37536 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(om,i_37530);
if(cljs.core.truth_(temp__5733__auto___37536)){
var vec__37345_37539 = temp__5733__auto___37536;
var iii_37540 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37345_37539,(0),null);
var oh_37541 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37345_37539,(1),null);
var cel_37545 = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(cs,iii_37540);
if((ii_37532 === iii_37540)){
(hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4 ? hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4(cel_37545,oh_37541,h_37533,m) : hipo.interpreter.reconciliate_BANG_.call(null,cel_37545,oh_37541,h_37533,m));
} else {
var b__37191__auto___37546 = ((function (seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,cel_37545,vec__37345_37539,iii_37540,oh_37541,temp__5733__auto___37536,vec__37339_37529,i_37530,vec__37342_37531,ii_37532,h_37533,seq__37308_37519__$1,temp__5735__auto___37518,om,nm,cs,map__37306,map__37306__$1,m,interceptors){
return (function (){
var ncel = el.removeChild(cel_37545);
(hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4 ? hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4(ncel,oh_37541,h_37533,m) : hipo.interpreter.reconciliate_BANG_.call(null,ncel,oh_37541,h_37533,m));

return hipo.dom.insert_child_BANG_(el,ii_37532,ncel);
});})(seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,cel_37545,vec__37345_37539,iii_37540,oh_37541,temp__5733__auto___37536,vec__37339_37529,i_37530,vec__37342_37531,ii_37532,h_37533,seq__37308_37519__$1,temp__5735__auto___37518,om,nm,cs,map__37306,map__37306__$1,m,interceptors))
;
var v__37192__auto___37547 = interceptors;
if(((cljs.core.not(v__37192__auto___37547)) || (cljs.core.empty_QMARK_(v__37192__auto___37547)))){
b__37191__auto___37546();
} else {
hipo.interceptor.call(b__37191__auto___37546,v__37192__auto___37547,new cljs.core.Keyword(null,"move","move",-2110884309),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),h_37533,new cljs.core.Keyword(null,"index","index",-1531685915),ii_37532], null));
}
}
} else {
var b__37191__auto___37555 = ((function (seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,temp__5733__auto___37536,vec__37339_37529,i_37530,vec__37342_37531,ii_37532,h_37533,seq__37308_37519__$1,temp__5735__auto___37518,om,nm,cs,map__37306,map__37306__$1,m,interceptors){
return (function (){
return hipo.dom.insert_child_BANG_(el,ii_37532,hipo.interpreter.create_child(h_37533,m));
});})(seq__37308_37482,chunk__37309_37483,count__37310_37484,i__37311_37485,temp__5733__auto___37536,vec__37339_37529,i_37530,vec__37342_37531,ii_37532,h_37533,seq__37308_37519__$1,temp__5735__auto___37518,om,nm,cs,map__37306,map__37306__$1,m,interceptors))
;
var v__37192__auto___37556 = interceptors;
if(((cljs.core.not(v__37192__auto___37556)) || (cljs.core.empty_QMARK_(v__37192__auto___37556)))){
b__37191__auto___37555();
} else {
hipo.interceptor.call(b__37191__auto___37555,v__37192__auto___37556,new cljs.core.Keyword(null,"insert","insert",1286475395),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),h_37533,new cljs.core.Keyword(null,"index","index",-1531685915),ii_37532], null));
}
}


var G__37563 = cljs.core.next(seq__37308_37519__$1);
var G__37564 = null;
var G__37565 = (0);
var G__37566 = (0);
seq__37308_37482 = G__37563;
chunk__37309_37483 = G__37564;
count__37310_37484 = G__37565;
i__37311_37485 = G__37566;
continue;
}
} else {
}
}
break;
}

var d = cljs.core.count(clojure.set.difference.cljs$core$IFn$_invoke$arity$2(cljs.core.set(cljs.core.keys(om)),cljs.core.set(cljs.core.keys(nm))));
if((d > (0))){
var b__37191__auto__ = (function (){
return hipo.dom.remove_trailing_children_BANG_(el,d);
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"remove-trailing","remove-trailing",-1590009193),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"count","count",2139924085),d], null));
}
} else {
return null;
}
});
hipo.interpreter.reconciliate_non_keyed_children_BANG_ = (function hipo$interpreter$reconciliate_non_keyed_children_BANG_(el,och,nch,p__37348){
var map__37349 = p__37348;
var map__37349__$1 = (((((!((map__37349 == null))))?(((((map__37349.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37349.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37349):map__37349);
var m = map__37349__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37349__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
var oc = cljs.core.count(och);
var nc = cljs.core.count(nch);
var d = (oc - nc);
if((d > (0))){
var b__37191__auto___37576 = (function (){
return hipo.dom.remove_trailing_children_BANG_(el,d);
});
var v__37192__auto___37577 = interceptors;
if(((cljs.core.not(v__37192__auto___37577)) || (cljs.core.empty_QMARK_(v__37192__auto___37577)))){
b__37191__auto___37576();
} else {
hipo.interceptor.call(b__37191__auto___37576,v__37192__auto___37577,new cljs.core.Keyword(null,"remove-trailing","remove-trailing",-1590009193),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"count","count",2139924085),d], null));
}
} else {
}

var n__4613__auto___37584 = (function (){var x__4217__auto__ = oc;
var y__4218__auto__ = nc;
return ((x__4217__auto__ < y__4218__auto__) ? x__4217__auto__ : y__4218__auto__);
})();
var i_37586 = (0);
while(true){
if((i_37586 < n__4613__auto___37584)){
var ov_37587 = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(och,i_37586);
var nv_37588 = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(nch,i_37586);
if((!((((ov_37587 == null)) && ((nv_37588 == null)))))){
if((ov_37587 == null)){
var b__37191__auto___37590 = ((function (i_37586,ov_37587,nv_37588,n__4613__auto___37584,oc,nc,d,map__37349,map__37349__$1,m,interceptors){
return (function (){
return hipo.dom.insert_child_BANG_(el,i_37586,hipo.interpreter.create_child(nv_37588,m));
});})(i_37586,ov_37587,nv_37588,n__4613__auto___37584,oc,nc,d,map__37349,map__37349__$1,m,interceptors))
;
var v__37192__auto___37591 = interceptors;
if(((cljs.core.not(v__37192__auto___37591)) || (cljs.core.empty_QMARK_(v__37192__auto___37591)))){
b__37191__auto___37590();
} else {
hipo.interceptor.call(b__37191__auto___37590,v__37192__auto___37591,new cljs.core.Keyword(null,"insert","insert",1286475395),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),nv_37588,new cljs.core.Keyword(null,"index","index",-1531685915),i_37586], null));
}
} else {
if((nv_37588 == null)){
var b__37191__auto___37599 = ((function (i_37586,ov_37587,nv_37588,n__4613__auto___37584,oc,nc,d,map__37349,map__37349__$1,m,interceptors){
return (function (){
return hipo.dom.remove_child_BANG_(el,i_37586);
});})(i_37586,ov_37587,nv_37588,n__4613__auto___37584,oc,nc,d,map__37349,map__37349__$1,m,interceptors))
;
var v__37192__auto___37600 = interceptors;
if(((cljs.core.not(v__37192__auto___37600)) || (cljs.core.empty_QMARK_(v__37192__auto___37600)))){
b__37191__auto___37599();
} else {
hipo.interceptor.call(b__37191__auto___37599,v__37192__auto___37600,new cljs.core.Keyword(null,"remove","remove",-131428414),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"index","index",-1531685915),i_37586], null));
}
} else {
var temp__5733__auto___37601 = hipo.dom.child(el,i_37586);
if(cljs.core.truth_(temp__5733__auto___37601)){
var cel_37603 = temp__5733__auto___37601;
(hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4 ? hipo.interpreter.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$4(cel_37603,ov_37587,nv_37588,m) : hipo.interpreter.reconciliate_BANG_.call(null,cel_37603,ov_37587,nv_37588,m));
} else {
}

}
}
} else {
}

var G__37606 = (i_37586 + (1));
i_37586 = G__37606;
continue;
} else {
}
break;
}

if((d < (0))){
if(((-1) === d)){
var temp__5733__auto__ = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(nch,oc);
if(cljs.core.truth_(temp__5733__auto__)){
var h = temp__5733__auto__;
var b__37191__auto__ = (function (){
return el.appendChild(hipo.interpreter.create_child(h,m));
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"append","append",-291298229),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),h], null));
}
} else {
return null;
}
} else {
var f = document.createDocumentFragment();
var cs = ((((0) === oc))?nch:cljs.core.subvec.cljs$core$IFn$_invoke$arity$2(nch,oc));
var b__37191__auto___37619 = (function (){
return hipo.interpreter.append_children_BANG_(f,cs,m);
});
var v__37192__auto___37620 = interceptors;
if(((cljs.core.not(v__37192__auto___37620)) || (cljs.core.empty_QMARK_(v__37192__auto___37620)))){
b__37191__auto___37619();
} else {
hipo.interceptor.call(b__37191__auto___37619,v__37192__auto___37620,new cljs.core.Keyword(null,"append","append",-291298229),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),cs], null));
}

return el.appendChild(f);
}
} else {
return null;
}
});
hipo.interpreter.keyed_children_QMARK_ = (function hipo$interpreter$keyed_children_QMARK_(v){
return (!((hipo.interpreter.child_key(cljs.core.nth.cljs$core$IFn$_invoke$arity$2(v,(0))) == null)));
});
hipo.interpreter.reconciliate_children_BANG_ = (function hipo$interpreter$reconciliate_children_BANG_(el,och,nch,p__37351){
var map__37352 = p__37351;
var map__37352__$1 = (((((!((map__37352 == null))))?(((((map__37352.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37352.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37352):map__37352);
var m = map__37352__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37352__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
if(cljs.core.empty_QMARK_(nch)){
if((!(cljs.core.empty_QMARK_(och)))){
var b__37191__auto__ = (function (){
return hipo.dom.clear_BANG_(el);
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"clear","clear",1877104959),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"target","target",253001721),el], null));
}
} else {
return null;
}
} else {
if(hipo.interpreter.keyed_children_QMARK_(nch)){
return hipo.interpreter.reconciliate_keyed_children_BANG_(el,och,nch,m);
} else {
return hipo.interpreter.reconciliate_non_keyed_children_BANG_(el,och,nch,m);
}
}
});
hipo.interpreter.reconciliate_vector_BANG_ = (function hipo$interpreter$reconciliate_vector_BANG_(el,oh,nh,p__37354){
var map__37355 = p__37354;
var map__37355__$1 = (((((!((map__37355 == null))))?(((((map__37355.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37355.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37355):map__37355);
var m = map__37355__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37355__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
if(cljs.core.vector_QMARK_(nh)){
} else {
throw (new Error("Assert failed: (vector? nh)"));
}

if(((hipo.hiccup.literal_QMARK_(oh)) || ((!((hipo.hiccup.tag(nh) === hipo.hiccup.tag(oh))))))){
var nel = hipo.interpreter.create_child(nh,m);
var b__37191__auto__ = (function (){
if(cljs.core.truth_(el.parentElement)){
} else {
throw (new Error(["Assert failed: ","Can't replace root element. If you want to change root element's type it must be encapsulated in a static element.","\n","(.-parentElement el)"].join('')));
}

return hipo.dom.replace_BANG_(el,nel);
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"replace","replace",-786587770),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),nh], null));
}
} else {
var om = hipo.hiccup.attributes(oh);
var nm = hipo.hiccup.attributes(nh);
var och = hipo.hiccup.children(oh);
var nch = hipo.hiccup.children(nh);
var b__37191__auto___37640 = (function (){
return hipo.interpreter.reconciliate_children_BANG_(el,hipo.hiccup.flatten_children(och),hipo.hiccup.flatten_children(nch),m);
});
var v__37192__auto___37641 = interceptors;
if(((cljs.core.not(v__37192__auto___37641)) || (cljs.core.empty_QMARK_(v__37192__auto___37641)))){
b__37191__auto___37640();
} else {
hipo.interceptor.call(b__37191__auto___37640,v__37192__auto___37641,new cljs.core.Keyword(null,"reconciliate","reconciliate",-527400739),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"old-value","old-value",862546795),och,new cljs.core.Keyword(null,"new-value","new-value",1087038368),nch], null));
}

return hipo.interpreter.reconciliate_attributes_BANG_(el,hipo.hiccup.keyns(nh),hipo.hiccup.tag(nh),om,nm,m);
}
});
hipo.interpreter.reconciliate_BANG_ = (function hipo$interpreter$reconciliate_BANG_(el,oh,nh,p__37357){
var map__37358 = p__37357;
var map__37358__$1 = (((((!((map__37358 == null))))?(((((map__37358.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__37358.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__37358):map__37358);
var m = map__37358__$1;
var interceptors = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__37358__$1,new cljs.core.Keyword(null,"interceptors","interceptors",-1546782951));
if(((cljs.core.vector_QMARK_(nh)) || (hipo.hiccup.literal_QMARK_(nh)))){
} else {
throw (new Error("Assert failed: (or (vector? nh) (hic/literal? nh))"));
}

if((((m == null)) || (cljs.core.map_QMARK_(m)))){
} else {
throw (new Error("Assert failed: (or (nil? m) (map? m))"));
}

var b__37191__auto__ = (function (){
if(hipo.hiccup.literal_QMARK_(nh)){
if((!((oh === nh)))){
var b__37191__auto__ = (function (){
if(cljs.core.truth_(el.parentElement)){
} else {
throw (new Error(["Assert failed: ","Can't replace root element. If you want to change root element's type it must be encapsulated in a static element.","\n","(.-parentElement el)"].join('')));
}

return hipo.dom.replace_text_BANG_(el,cljs.core.str.cljs$core$IFn$_invoke$arity$1(nh));
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"replace","replace",-786587770),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"value","value",305978217),nh], null));
}
} else {
return null;
}
} else {
return hipo.interpreter.reconciliate_vector_BANG_(el,oh,nh,m);
}
});
var v__37192__auto__ = interceptors;
if(((cljs.core.not(v__37192__auto__)) || (cljs.core.empty_QMARK_(v__37192__auto__)))){
return b__37191__auto__();
} else {
return hipo.interceptor.call(b__37191__auto__,v__37192__auto__,new cljs.core.Keyword(null,"reconciliate","reconciliate",-527400739),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"target","target",253001721),el,new cljs.core.Keyword(null,"old-value","old-value",862546795),oh,new cljs.core.Keyword(null,"new-value","new-value",1087038368),nh], null));
}
});

//# sourceMappingURL=hipo.interpreter.js.map
