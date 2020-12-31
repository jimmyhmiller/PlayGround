goog.provide('shadow.remote.runtime.tap_support');
shadow.remote.runtime.tap_support.tap_subscribe = (function shadow$remote$runtime$tap_support$tap_subscribe(p__45391,p__45392){
var map__45393 = p__45391;
var map__45393__$1 = (((((!((map__45393 == null))))?(((((map__45393.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45393.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45393):map__45393);
var svc = map__45393__$1;
var subs_ref = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45393__$1,new cljs.core.Keyword(null,"subs-ref","subs-ref",-1355989911));
var obj_support = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45393__$1,new cljs.core.Keyword(null,"obj-support","obj-support",1522559229));
var runtime = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45393__$1,new cljs.core.Keyword(null,"runtime","runtime",-1331573996));
var map__45394 = p__45392;
var map__45394__$1 = (((((!((map__45394 == null))))?(((((map__45394.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45394.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45394):map__45394);
var msg = map__45394__$1;
var from = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45394__$1,new cljs.core.Keyword(null,"from","from",1815293044));
var summary = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45394__$1,new cljs.core.Keyword(null,"summary","summary",380847952));
var history = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45394__$1,new cljs.core.Keyword(null,"history","history",-247395220));
var num = cljs.core.get.cljs$core$IFn$_invoke$arity$3(map__45394__$1,new cljs.core.Keyword(null,"num","num",1985240673),(10));
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$4(subs_ref,cljs.core.assoc,from,msg);

if(cljs.core.truth_(history)){
return shadow.remote.runtime.shared.reply(runtime,msg,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"op","op",-1882987955),new cljs.core.Keyword(null,"tap-subscribed","tap-subscribed",-1882247432),new cljs.core.Keyword(null,"history","history",-247395220),cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentVector.EMPTY,cljs.core.map.cljs$core$IFn$_invoke$arity$2((function (oid){
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"oid","oid",-768692334),oid,new cljs.core.Keyword(null,"summary","summary",380847952),shadow.remote.runtime.obj_support.obj_describe_STAR_(obj_support,oid)], null);
}),shadow.remote.runtime.obj_support.get_tap_history(obj_support,num)))], null));
} else {
return null;
}
});
shadow.remote.runtime.tap_support.tap_unsubscribe = (function shadow$remote$runtime$tap_support$tap_unsubscribe(p__45405,p__45406){
var map__45408 = p__45405;
var map__45408__$1 = (((((!((map__45408 == null))))?(((((map__45408.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45408.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45408):map__45408);
var subs_ref = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45408__$1,new cljs.core.Keyword(null,"subs-ref","subs-ref",-1355989911));
var map__45409 = p__45406;
var map__45409__$1 = (((((!((map__45409 == null))))?(((((map__45409.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45409.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45409):map__45409);
var from = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45409__$1,new cljs.core.Keyword(null,"from","from",1815293044));
return cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(subs_ref,cljs.core.dissoc,from);
});
shadow.remote.runtime.tap_support.request_tap_history = (function shadow$remote$runtime$tap_support$request_tap_history(p__45415,p__45416){
var map__45419 = p__45415;
var map__45419__$1 = (((((!((map__45419 == null))))?(((((map__45419.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45419.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45419):map__45419);
var obj_support = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45419__$1,new cljs.core.Keyword(null,"obj-support","obj-support",1522559229));
var runtime = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45419__$1,new cljs.core.Keyword(null,"runtime","runtime",-1331573996));
var map__45420 = p__45416;
var map__45420__$1 = (((((!((map__45420 == null))))?(((((map__45420.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45420.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45420):map__45420);
var msg = map__45420__$1;
var num = cljs.core.get.cljs$core$IFn$_invoke$arity$3(map__45420__$1,new cljs.core.Keyword(null,"num","num",1985240673),(10));
var tap_ids = shadow.remote.runtime.obj_support.get_tap_history(obj_support,num);
return shadow.remote.runtime.shared.reply(runtime,msg,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"op","op",-1882987955),new cljs.core.Keyword(null,"tap-history","tap-history",-282803347),new cljs.core.Keyword(null,"oids","oids",-1580877688),tap_ids], null));
});
shadow.remote.runtime.tap_support.tool_disconnect = (function shadow$remote$runtime$tap_support$tool_disconnect(p__45427,tid){
var map__45428 = p__45427;
var map__45428__$1 = (((((!((map__45428 == null))))?(((((map__45428.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45428.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45428):map__45428);
var svc = map__45428__$1;
var subs_ref = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45428__$1,new cljs.core.Keyword(null,"subs-ref","subs-ref",-1355989911));
return cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(subs_ref,cljs.core.dissoc,tid);
});
shadow.remote.runtime.tap_support.start = (function shadow$remote$runtime$tap_support$start(runtime,obj_support){
var subs_ref = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var tap_fn = (function shadow$remote$runtime$tap_support$start_$_runtime_tap(obj){
if((!((obj == null)))){
var oid = shadow.remote.runtime.obj_support.register(obj_support,obj,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"from","from",1815293044),new cljs.core.Keyword(null,"tap","tap",-1086702463)], null));
var seq__45439 = cljs.core.seq(cljs.core.deref(subs_ref));
var chunk__45440 = null;
var count__45441 = (0);
var i__45442 = (0);
while(true){
if((i__45442 < count__45441)){
var vec__45453 = chunk__45440.cljs$core$IIndexed$_nth$arity$2(null,i__45442);
var tid = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45453,(0),null);
var tap_config = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45453,(1),null);
shadow.remote.runtime.api.relay_msg(runtime,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"op","op",-1882987955),new cljs.core.Keyword(null,"tap","tap",-1086702463),new cljs.core.Keyword(null,"to","to",192099007),tid,new cljs.core.Keyword(null,"oid","oid",-768692334),oid], null));


var G__45483 = seq__45439;
var G__45484 = chunk__45440;
var G__45485 = count__45441;
var G__45486 = (i__45442 + (1));
seq__45439 = G__45483;
chunk__45440 = G__45484;
count__45441 = G__45485;
i__45442 = G__45486;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__45439);
if(temp__5735__auto__){
var seq__45439__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__45439__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__45439__$1);
var G__45487 = cljs.core.chunk_rest(seq__45439__$1);
var G__45488 = c__4556__auto__;
var G__45489 = cljs.core.count(c__4556__auto__);
var G__45490 = (0);
seq__45439 = G__45487;
chunk__45440 = G__45488;
count__45441 = G__45489;
i__45442 = G__45490;
continue;
} else {
var vec__45458 = cljs.core.first(seq__45439__$1);
var tid = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45458,(0),null);
var tap_config = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45458,(1),null);
shadow.remote.runtime.api.relay_msg(runtime,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"op","op",-1882987955),new cljs.core.Keyword(null,"tap","tap",-1086702463),new cljs.core.Keyword(null,"to","to",192099007),tid,new cljs.core.Keyword(null,"oid","oid",-768692334),oid], null));


var G__45491 = cljs.core.next(seq__45439__$1);
var G__45492 = null;
var G__45493 = (0);
var G__45494 = (0);
seq__45439 = G__45491;
chunk__45440 = G__45492;
count__45441 = G__45493;
i__45442 = G__45494;
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
});
var svc = new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"runtime","runtime",-1331573996),runtime,new cljs.core.Keyword(null,"obj-support","obj-support",1522559229),obj_support,new cljs.core.Keyword(null,"tap-fn","tap-fn",1573556461),tap_fn,new cljs.core.Keyword(null,"subs-ref","subs-ref",-1355989911),subs_ref], null);
shadow.remote.runtime.api.add_extension(runtime,new cljs.core.Keyword("shadow.remote.runtime.tap-support","ext","shadow.remote.runtime.tap-support/ext",1019069674),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"ops","ops",1237330063),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"tap-subscribe","tap-subscribe",411179050),(function (p1__45430_SHARP_){
return shadow.remote.runtime.tap_support.tap_subscribe(svc,p1__45430_SHARP_);
}),new cljs.core.Keyword(null,"tap-unsubscribe","tap-unsubscribe",1183890755),(function (p1__45431_SHARP_){
return shadow.remote.runtime.tap_support.tap_unsubscribe(svc,p1__45431_SHARP_);
}),new cljs.core.Keyword(null,"request-tap-history","request-tap-history",-670837812),(function (p1__45432_SHARP_){
return shadow.remote.runtime.tap_support.request_tap_history(svc,p1__45432_SHARP_);
})], null),new cljs.core.Keyword(null,"on-tool-disconnect","on-tool-disconnect",693464366),(function (p1__45433_SHARP_){
return shadow.remote.runtime.tap_support.tool_disconnect(svc,p1__45433_SHARP_);
})], null));

cljs.core.add_tap(tap_fn);

return svc;
});
shadow.remote.runtime.tap_support.stop = (function shadow$remote$runtime$tap_support$stop(p__45466){
var map__45468 = p__45466;
var map__45468__$1 = (((((!((map__45468 == null))))?(((((map__45468.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__45468.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__45468):map__45468);
var svc = map__45468__$1;
var tap_fn = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45468__$1,new cljs.core.Keyword(null,"tap-fn","tap-fn",1573556461));
var runtime = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__45468__$1,new cljs.core.Keyword(null,"runtime","runtime",-1331573996));
cljs.core.remove_tap(tap_fn);

return shadow.remote.runtime.api.del_extension(runtime,new cljs.core.Keyword("shadow.remote.runtime.tap-support","ext","shadow.remote.runtime.tap-support/ext",1019069674));
});

//# sourceMappingURL=shadow.remote.runtime.tap_support.js.map
