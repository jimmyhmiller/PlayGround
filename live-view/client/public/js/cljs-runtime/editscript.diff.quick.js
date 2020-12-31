goog.provide('editscript.diff.quick');
editscript.diff.quick.diff_STAR_ = (function editscript$diff$quick$diff_STAR_(script,path,a,b,opts){
return null;
});
editscript.diff.quick.diff_map = (function editscript$diff$quick$diff_map(script,path,a,b,opts){
cljs.core.reduce_kv((function (_,ka,va){
var path_SINGLEQUOTE_ = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,ka);
if(cljs.core.contains_QMARK_(b,ka)){
var G__46239 = script;
var G__46240 = path_SINGLEQUOTE_;
var G__46241 = va;
var G__46242 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(b,ka);
var G__46243 = opts;
return (editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46239,G__46240,G__46241,G__46242,G__46243) : editscript.diff.quick.diff_STAR_.call(null,G__46239,G__46240,G__46241,G__46242,G__46243));
} else {
var G__46244 = script;
var G__46245 = path_SINGLEQUOTE_;
var G__46246 = va;
var G__46247 = editscript.edit.nada();
var G__46248 = opts;
return (editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46244,G__46245,G__46246,G__46247,G__46248) : editscript.diff.quick.diff_STAR_.call(null,G__46244,G__46245,G__46246,G__46247,G__46248));
}
}),null,a);

return cljs.core.reduce_kv((function (_,kb,vb){
if(cljs.core.contains_QMARK_(a,kb)){
return null;
} else {
var G__46256 = script;
var G__46257 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,kb);
var G__46258 = editscript.edit.nada();
var G__46259 = vb;
var G__46260 = opts;
return (editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46256,G__46257,G__46258,G__46259,G__46260) : editscript.diff.quick.diff_STAR_.call(null,G__46256,G__46257,G__46258,G__46259,G__46260));
}
}),null,b);
});
/**
 * Adjust the indices to have a correct editscript
 */
editscript.diff.quick.diff_vec = (function editscript$diff$quick$diff_vec(script,path,a,b,opts){
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (p__46270,op){
var vec__46271 = p__46270;
var ia = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46271,(0),null);
var ia_SINGLEQUOTE_ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46271,(1),null);
var ib = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46271,(2),null);
var G__46274 = op;
var G__46274__$1 = (((G__46274 instanceof cljs.core.Keyword))?G__46274.fqn:null);
switch (G__46274__$1) {
case "-":
var G__46275_46472 = script;
var G__46276_46473 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,ia_SINGLEQUOTE_);
var G__46277_46474 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(a,ia);
var G__46278_46475 = editscript.edit.nada();
var G__46279_46476 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46275_46472,G__46276_46473,G__46277_46474,G__46278_46475,G__46279_46476) : editscript.diff.quick.diff_STAR_.call(null,G__46275_46472,G__46276_46473,G__46277_46474,G__46278_46475,G__46279_46476));

return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [(ia + (1)),ia_SINGLEQUOTE_,ib], null);

break;
case "+":
var G__46280_46477 = script;
var G__46281_46478 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,ia_SINGLEQUOTE_);
var G__46282_46479 = editscript.edit.nada();
var G__46283_46480 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(b,ib);
var G__46284_46481 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46280_46477,G__46281_46478,G__46282_46479,G__46283_46480,G__46284_46481) : editscript.diff.quick.diff_STAR_.call(null,G__46280_46477,G__46281_46478,G__46282_46479,G__46283_46480,G__46284_46481));

return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [ia,(ia_SINGLEQUOTE_ + (1)),(ib + (1))], null);

break;
case "r":
var G__46287_46482 = script;
var G__46288_46483 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,ia_SINGLEQUOTE_);
var G__46289_46484 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(a,ia);
var G__46290_46485 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(b,ib);
var G__46291_46486 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46287_46482,G__46288_46483,G__46289_46484,G__46290_46485,G__46291_46486) : editscript.diff.quick.diff_STAR_.call(null,G__46287_46482,G__46288_46483,G__46289_46484,G__46290_46485,G__46291_46486));

return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [(ia + (1)),(ia_SINGLEQUOTE_ + (1)),(ib + (1))], null);

break;
default:
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [(ia + op),(ia_SINGLEQUOTE_ + op),(ib + op)], null);

}
}),cljs.core.transient$(new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [(0),(0),(0)], null)),editscript.util.common.vec_edits(a,b));
});
editscript.diff.quick.diff_set = (function editscript$diff$quick$diff_set(script,path,a,b,opts){
var seq__46295_46489 = cljs.core.seq(clojure.set.difference.cljs$core$IFn$_invoke$arity$2(a,b));
var chunk__46296_46490 = null;
var count__46297_46491 = (0);
var i__46298_46492 = (0);
while(true){
if((i__46298_46492 < count__46297_46491)){
var va_46493 = chunk__46296_46490.cljs$core$IIndexed$_nth$arity$2(null,i__46298_46492);
var G__46323_46494 = script;
var G__46324_46495 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,va_46493);
var G__46325_46496 = va_46493;
var G__46326_46497 = editscript.edit.nada();
var G__46327_46498 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46323_46494,G__46324_46495,G__46325_46496,G__46326_46497,G__46327_46498) : editscript.diff.quick.diff_STAR_.call(null,G__46323_46494,G__46324_46495,G__46325_46496,G__46326_46497,G__46327_46498));


var G__46499 = seq__46295_46489;
var G__46500 = chunk__46296_46490;
var G__46501 = count__46297_46491;
var G__46502 = (i__46298_46492 + (1));
seq__46295_46489 = G__46499;
chunk__46296_46490 = G__46500;
count__46297_46491 = G__46501;
i__46298_46492 = G__46502;
continue;
} else {
var temp__5735__auto___46503 = cljs.core.seq(seq__46295_46489);
if(temp__5735__auto___46503){
var seq__46295_46504__$1 = temp__5735__auto___46503;
if(cljs.core.chunked_seq_QMARK_(seq__46295_46504__$1)){
var c__4556__auto___46505 = cljs.core.chunk_first(seq__46295_46504__$1);
var G__46506 = cljs.core.chunk_rest(seq__46295_46504__$1);
var G__46507 = c__4556__auto___46505;
var G__46508 = cljs.core.count(c__4556__auto___46505);
var G__46509 = (0);
seq__46295_46489 = G__46506;
chunk__46296_46490 = G__46507;
count__46297_46491 = G__46508;
i__46298_46492 = G__46509;
continue;
} else {
var va_46510 = cljs.core.first(seq__46295_46504__$1);
var G__46330_46512 = script;
var G__46331_46513 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,va_46510);
var G__46332_46514 = va_46510;
var G__46333_46515 = editscript.edit.nada();
var G__46334_46516 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46330_46512,G__46331_46513,G__46332_46514,G__46333_46515,G__46334_46516) : editscript.diff.quick.diff_STAR_.call(null,G__46330_46512,G__46331_46513,G__46332_46514,G__46333_46515,G__46334_46516));


var G__46517 = cljs.core.next(seq__46295_46504__$1);
var G__46518 = null;
var G__46519 = (0);
var G__46520 = (0);
seq__46295_46489 = G__46517;
chunk__46296_46490 = G__46518;
count__46297_46491 = G__46519;
i__46298_46492 = G__46520;
continue;
}
} else {
}
}
break;
}

var seq__46337 = cljs.core.seq(clojure.set.difference.cljs$core$IFn$_invoke$arity$2(b,a));
var chunk__46338 = null;
var count__46339 = (0);
var i__46340 = (0);
while(true){
if((i__46340 < count__46339)){
var vb = chunk__46338.cljs$core$IIndexed$_nth$arity$2(null,i__46340);
var G__46363_46521 = script;
var G__46364_46522 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,vb);
var G__46365_46523 = editscript.edit.nada();
var G__46366_46524 = vb;
var G__46367_46525 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46363_46521,G__46364_46522,G__46365_46523,G__46366_46524,G__46367_46525) : editscript.diff.quick.diff_STAR_.call(null,G__46363_46521,G__46364_46522,G__46365_46523,G__46366_46524,G__46367_46525));


var G__46527 = seq__46337;
var G__46528 = chunk__46338;
var G__46529 = count__46339;
var G__46530 = (i__46340 + (1));
seq__46337 = G__46527;
chunk__46338 = G__46528;
count__46339 = G__46529;
i__46340 = G__46530;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46337);
if(temp__5735__auto__){
var seq__46337__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46337__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46337__$1);
var G__46531 = cljs.core.chunk_rest(seq__46337__$1);
var G__46532 = c__4556__auto__;
var G__46533 = cljs.core.count(c__4556__auto__);
var G__46534 = (0);
seq__46337 = G__46531;
chunk__46338 = G__46532;
count__46339 = G__46533;
i__46340 = G__46534;
continue;
} else {
var vb = cljs.core.first(seq__46337__$1);
var G__46369_46535 = script;
var G__46370_46536 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,vb);
var G__46371_46537 = editscript.edit.nada();
var G__46372_46538 = vb;
var G__46373_46539 = opts;
(editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5 ? editscript.diff.quick.diff_STAR_.cljs$core$IFn$_invoke$arity$5(G__46369_46535,G__46370_46536,G__46371_46537,G__46372_46538,G__46373_46539) : editscript.diff.quick.diff_STAR_.call(null,G__46369_46535,G__46370_46536,G__46371_46537,G__46372_46538,G__46373_46539));


var G__46540 = cljs.core.next(seq__46337__$1);
var G__46541 = null;
var G__46542 = (0);
var G__46543 = (0);
seq__46337 = G__46540;
chunk__46338 = G__46541;
count__46339 = G__46542;
i__46340 = G__46543;
continue;
}
} else {
return null;
}
}
break;
}
});
editscript.diff.quick.diff_lst = (function editscript$diff$quick$diff_lst(script,path,a,b,opts){
return editscript.diff.quick.diff_vec(script,path,cljs.core.vec(a),cljs.core.vec(b),opts);
});
editscript.diff.quick.diff_val = (function editscript$diff$quick$diff_val(script,path,a,b){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(editscript.edit.get_type(b),new cljs.core.Keyword(null,"nil","nil",99600501))){
return editscript.edit.delete_data(script,path);
} else {
return editscript.edit.replace_data(script,path,b);
}
});
editscript.diff.quick.diff_STAR_ = (function editscript$diff$quick$diff_STAR_(script,path,a,b,opts){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(a,b)){
return null;
} else {
var G__46381 = editscript.edit.get_type(a);
var G__46381__$1 = (((G__46381 instanceof cljs.core.Keyword))?G__46381.fqn:null);
switch (G__46381__$1) {
case "nil":
return editscript.edit.add_data(script,path,b);

break;
case "map":
var G__46384 = editscript.edit.get_type(b);
var G__46384__$1 = (((G__46384 instanceof cljs.core.Keyword))?G__46384.fqn:null);
switch (G__46384__$1) {
case "nil":
return editscript.edit.delete_data(script,path);

break;
case "map":
var fexpr__46390 = new cljs.core.Var(function(){return editscript.diff.quick.diff_map;},new cljs.core.Symbol("editscript.diff.quick","diff-map","editscript.diff.quick/diff-map",1973051147,null),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"private","private",-558947994),new cljs.core.Keyword(null,"ns","ns",441598760),new cljs.core.Keyword(null,"name","name",1843675177),new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"end-column","end-column",1425389514),new cljs.core.Keyword(null,"column","column",2078222095),new cljs.core.Keyword(null,"line","line",212345235),new cljs.core.Keyword(null,"end-line","end-line",1837326455),new cljs.core.Keyword(null,"arglists","arglists",1661989754),new cljs.core.Keyword(null,"doc","doc",1913296891),new cljs.core.Keyword(null,"test","test",577538877)],[true,cljs.core.with_meta(new cljs.core.Symbol(null,"editscript.diff.quick","editscript.diff.quick",165700457,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"no-doc","no-doc",1559921891),true], null)),new cljs.core.Symbol(null,"diff-map","diff-map",-711364586,null),"editscript/diff/quick.cljc",16,1,22,22,cljs.core.list(new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"script","script",336087726,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.Symbol(null,"opts","opts",1795607228,null)], null)),null,(cljs.core.truth_(editscript.diff.quick.diff_map)?editscript.diff.quick.diff_map.cljs$lang$test:null)]));
return (fexpr__46390.cljs$core$IFn$_invoke$arity$5 ? fexpr__46390.cljs$core$IFn$_invoke$arity$5(script,path,a,b,opts) : fexpr__46390.call(null,script,path,a,b,opts));

break;
default:
return editscript.edit.replace_data(script,path,b);

}

break;
case "vec":
var G__46392 = editscript.edit.get_type(b);
var G__46392__$1 = (((G__46392 instanceof cljs.core.Keyword))?G__46392.fqn:null);
switch (G__46392__$1) {
case "nil":
return editscript.edit.delete_data(script,path);

break;
case "vec":
var fexpr__46395 = new cljs.core.Var(function(){return editscript.diff.quick.diff_vec;},new cljs.core.Symbol("editscript.diff.quick","diff-vec","editscript.diff.quick/diff-vec",-1781503036,null),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"private","private",-558947994),new cljs.core.Keyword(null,"ns","ns",441598760),new cljs.core.Keyword(null,"name","name",1843675177),new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"end-column","end-column",1425389514),new cljs.core.Keyword(null,"column","column",2078222095),new cljs.core.Keyword(null,"line","line",212345235),new cljs.core.Keyword(null,"end-line","end-line",1837326455),new cljs.core.Keyword(null,"arglists","arglists",1661989754),new cljs.core.Keyword(null,"doc","doc",1913296891),new cljs.core.Keyword(null,"test","test",577538877)],[true,cljs.core.with_meta(new cljs.core.Symbol(null,"editscript.diff.quick","editscript.diff.quick",165700457,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"no-doc","no-doc",1559921891),true], null)),new cljs.core.Symbol(null,"diff-vec","diff-vec",-170849589,null),"editscript/diff/quick.cljc",16,1,39,39,cljs.core.list(new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"script","script",336087726,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.Symbol(null,"opts","opts",1795607228,null)], null)),"Adjust the indices to have a correct editscript",(cljs.core.truth_(editscript.diff.quick.diff_vec)?editscript.diff.quick.diff_vec.cljs$lang$test:null)]));
return (fexpr__46395.cljs$core$IFn$_invoke$arity$5 ? fexpr__46395.cljs$core$IFn$_invoke$arity$5(script,path,a,b,opts) : fexpr__46395.call(null,script,path,a,b,opts));

break;
default:
return editscript.edit.replace_data(script,path,b);

}

break;
case "set":
var G__46396 = editscript.edit.get_type(b);
var G__46396__$1 = (((G__46396 instanceof cljs.core.Keyword))?G__46396.fqn:null);
switch (G__46396__$1) {
case "nil":
return editscript.edit.delete_data(script,path);

break;
case "set":
var fexpr__46399 = new cljs.core.Var(function(){return editscript.diff.quick.diff_set;},new cljs.core.Symbol("editscript.diff.quick","diff-set","editscript.diff.quick/diff-set",1246078730,null),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"private","private",-558947994),new cljs.core.Keyword(null,"ns","ns",441598760),new cljs.core.Keyword(null,"name","name",1843675177),new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"end-column","end-column",1425389514),new cljs.core.Keyword(null,"column","column",2078222095),new cljs.core.Keyword(null,"line","line",212345235),new cljs.core.Keyword(null,"end-line","end-line",1837326455),new cljs.core.Keyword(null,"arglists","arglists",1661989754),new cljs.core.Keyword(null,"doc","doc",1913296891),new cljs.core.Keyword(null,"test","test",577538877)],[true,cljs.core.with_meta(new cljs.core.Symbol(null,"editscript.diff.quick","editscript.diff.quick",165700457,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"no-doc","no-doc",1559921891),true], null)),new cljs.core.Symbol(null,"diff-set","diff-set",-364591609,null),"editscript/diff/quick.cljc",16,1,55,55,cljs.core.list(new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"script","script",336087726,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.Symbol(null,"opts","opts",1795607228,null)], null)),null,(cljs.core.truth_(editscript.diff.quick.diff_set)?editscript.diff.quick.diff_set.cljs$lang$test:null)]));
return (fexpr__46399.cljs$core$IFn$_invoke$arity$5 ? fexpr__46399.cljs$core$IFn$_invoke$arity$5(script,path,a,b,opts) : fexpr__46399.call(null,script,path,a,b,opts));

break;
default:
return editscript.edit.replace_data(script,path,b);

}

break;
case "lst":
var G__46408 = editscript.edit.get_type(b);
var G__46408__$1 = (((G__46408 instanceof cljs.core.Keyword))?G__46408.fqn:null);
switch (G__46408__$1) {
case "nil":
return editscript.edit.delete_data(script,path);

break;
case "lst":
var fexpr__46411 = new cljs.core.Var(function(){return editscript.diff.quick.diff_lst;},new cljs.core.Symbol("editscript.diff.quick","diff-lst","editscript.diff.quick/diff-lst",-1885319001,null),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"private","private",-558947994),new cljs.core.Keyword(null,"ns","ns",441598760),new cljs.core.Keyword(null,"name","name",1843675177),new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"end-column","end-column",1425389514),new cljs.core.Keyword(null,"column","column",2078222095),new cljs.core.Keyword(null,"line","line",212345235),new cljs.core.Keyword(null,"end-line","end-line",1837326455),new cljs.core.Keyword(null,"arglists","arglists",1661989754),new cljs.core.Keyword(null,"doc","doc",1913296891),new cljs.core.Keyword(null,"test","test",577538877)],[true,cljs.core.with_meta(new cljs.core.Symbol(null,"editscript.diff.quick","editscript.diff.quick",165700457,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"no-doc","no-doc",1559921891),true], null)),new cljs.core.Symbol(null,"diff-lst","diff-lst",799109538,null),"editscript/diff/quick.cljc",16,1,62,62,cljs.core.list(new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"script","script",336087726,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.Symbol(null,"opts","opts",1795607228,null)], null)),null,(cljs.core.truth_(editscript.diff.quick.diff_lst)?editscript.diff.quick.diff_lst.cljs$lang$test:null)]));
return (fexpr__46411.cljs$core$IFn$_invoke$arity$5 ? fexpr__46411.cljs$core$IFn$_invoke$arity$5(script,path,a,b,opts) : fexpr__46411.call(null,script,path,a,b,opts));

break;
default:
return editscript.edit.replace_data(script,path,b);

}

break;
case "str":
if(cljs.core.truth_(new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760).cljs$core$IFn$_invoke$arity$1(opts))){
var G__46416 = editscript.edit.get_type(b);
var G__46416__$1 = (((G__46416 instanceof cljs.core.Keyword))?G__46416.fqn:null);
switch (G__46416__$1) {
case "nil":
return editscript.edit.delete_data(script,path);

break;
case "str":
var fexpr__46434 = new cljs.core.Var(function(){return editscript.util.common.diff_str;},new cljs.core.Symbol("editscript.util.common","diff-str","editscript.util.common/diff-str",283460236,null),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"ns","ns",441598760),new cljs.core.Keyword(null,"name","name",1843675177),new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"end-column","end-column",1425389514),new cljs.core.Keyword(null,"column","column",2078222095),new cljs.core.Keyword(null,"line","line",212345235),new cljs.core.Keyword(null,"end-line","end-line",1837326455),new cljs.core.Keyword(null,"arglists","arglists",1661989754),new cljs.core.Keyword(null,"doc","doc",1913296891),new cljs.core.Keyword(null,"test","test",577538877)],[new cljs.core.Symbol(null,"editscript.util.common","editscript.util.common",1209759084,null),new cljs.core.Symbol(null,"diff-str","diff-str",1255675210,null),"editscript/util/common.cljc",15,1,135,135,cljs.core.list(new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"script","script",336087726,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.Symbol(null,"_","_",-1201019570,null)], null)),null,(cljs.core.truth_(editscript.util.common.diff_str)?editscript.util.common.diff_str.cljs$lang$test:null)]));
return (fexpr__46434.cljs$core$IFn$_invoke$arity$5 ? fexpr__46434.cljs$core$IFn$_invoke$arity$5(script,path,a,b,opts) : fexpr__46434.call(null,script,path,a,b,opts));

break;
default:
return editscript.edit.replace_data(script,path,b);

}
} else {
return editscript.diff.quick.diff_val(script,path,a,b);
}

break;
case "val":
return editscript.diff.quick.diff_val(script,path,a,b);

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__46381__$1)].join('')));

}
}
});
/**
 * Create an EditScript that represents the difference between `b` and `a`
 *   This algorithm is fast, but it does not attempt to generate an EditScript
 *   that is minimal in size
 */
editscript.diff.quick.diff = (function editscript$diff$quick$diff(var_args){
var G__46445 = arguments.length;
switch (G__46445) {
case 2:
return editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$2 = (function (a,b){
return editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$3(a,b,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760),false], null));
}));

(editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$3 = (function (a,b,opts){
var script = editscript.edit.edits__GT_script(cljs.core.PersistentVector.EMPTY);
editscript.diff.quick.diff_STAR_(script,cljs.core.PersistentVector.EMPTY,a,b,opts);

return script;
}));

(editscript.diff.quick.diff.cljs$lang$maxFixedArity = 3);

editscript.util.common.diff_algo.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"quick","quick",841581564),(function (a,b,opts){
return editscript.diff.quick.diff.cljs$core$IFn$_invoke$arity$3(a,b,opts);
}));

//# sourceMappingURL=editscript.diff.quick.js.map
