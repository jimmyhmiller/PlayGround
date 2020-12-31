goog.provide('crate.binding');

/**
* @constructor
 * @implements {cljs.core.IWatchable}
 * @implements {cljs.core.IEquiv}
 * @implements {cljs.core.IHash}
 * @implements {cljs.core.IDeref}
 * @implements {cljs.core.IPrintWithWriter}
*/
crate.binding.SubAtom = (function (atm,path,prevhash,watches,key){
this.atm = atm;
this.path = path;
this.prevhash = prevhash;
this.watches = watches;
this.key = key;
this.cljs$lang$protocol_mask$partition0$ = 2153807872;
this.cljs$lang$protocol_mask$partition1$ = 2;
});
(crate.binding.SubAtom.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (o,other){
var self__ = this;
var o__$1 = this;
return (o__$1 === other);
}));

(crate.binding.SubAtom.prototype.cljs$core$IDeref$_deref$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
if(cljs.core.truth_(self__.atm)){
return cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(self__.atm),self__.path);
} else {
return null;
}
}));

(crate.binding.SubAtom.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this$,writer,opts){
var self__ = this;
var this$__$1 = this;
return cljs.core._write(writer,["#<SubAtom: ",cljs.core.pr_str.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(self__.atm),self__.path)], 0)),">"].join(''));
}));

(crate.binding.SubAtom.prototype.cljs$core$IWatchable$_notify_watches$arity$3 = (function (this$,oldval,newval){
var self__ = this;
var this$__$1 = this;
var seq__46249 = cljs.core.seq(self__.watches);
var chunk__46250 = null;
var count__46251 = (0);
var i__46252 = (0);
while(true){
if((i__46252 < count__46251)){
var vec__46264 = chunk__46250.cljs$core$IIndexed$_nth$arity$2(null,i__46252);
var key__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46264,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46264,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key__$1,this$__$1,oldval,newval) : f.call(null,key__$1,this$__$1,oldval,newval));


var G__46811 = seq__46249;
var G__46812 = chunk__46250;
var G__46813 = count__46251;
var G__46814 = (i__46252 + (1));
seq__46249 = G__46811;
chunk__46250 = G__46812;
count__46251 = G__46813;
i__46252 = G__46814;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46249);
if(temp__5735__auto__){
var seq__46249__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46249__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46249__$1);
var G__46819 = cljs.core.chunk_rest(seq__46249__$1);
var G__46820 = c__4556__auto__;
var G__46821 = cljs.core.count(c__4556__auto__);
var G__46822 = (0);
seq__46249 = G__46819;
chunk__46250 = G__46820;
count__46251 = G__46821;
i__46252 = G__46822;
continue;
} else {
var vec__46267 = cljs.core.first(seq__46249__$1);
var key__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46267,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46267,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key__$1,this$__$1,oldval,newval) : f.call(null,key__$1,this$__$1,oldval,newval));


var G__46831 = cljs.core.next(seq__46249__$1);
var G__46832 = null;
var G__46833 = (0);
var G__46834 = (0);
seq__46249 = G__46831;
chunk__46250 = G__46832;
count__46251 = G__46833;
i__46252 = G__46834;
continue;
}
} else {
return null;
}
}
break;
}
}));

(crate.binding.SubAtom.prototype.cljs$core$IWatchable$_add_watch$arity$3 = (function (this$,key__$1,f){
var self__ = this;
var this$__$1 = this;
if(cljs.core.truth_(f)){
return (this$__$1.watches = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.watches,key__$1,f));
} else {
return null;
}
}));

(crate.binding.SubAtom.prototype.cljs$core$IWatchable$_remove_watch$arity$2 = (function (this$,key__$1){
var self__ = this;
var this$__$1 = this;
return (this$__$1.watches = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(self__.watches,key__$1));
}));

(crate.binding.SubAtom.prototype.cljs$core$IHash$_hash$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return goog.getUid(this$__$1);
}));

(crate.binding.SubAtom.getBasis = (function (){
return new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"atm","atm",-1963551835,null),new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.Symbol(null,"prevhash","prevhash",1446045952,null),new cljs.core.Symbol(null,"watches","watches",1367433992,null),new cljs.core.Symbol(null,"key","key",124488940,null)], null);
}));

(crate.binding.SubAtom.cljs$lang$type = true);

(crate.binding.SubAtom.cljs$lang$ctorStr = "crate.binding/SubAtom");

(crate.binding.SubAtom.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"crate.binding/SubAtom");
}));

/**
 * Positional factory function for crate.binding/SubAtom.
 */
crate.binding.__GT_SubAtom = (function crate$binding$__GT_SubAtom(atm,path,prevhash,watches,key){
return (new crate.binding.SubAtom(atm,path,prevhash,watches,key));
});

crate.binding.subatom = (function crate$binding$subatom(atm,path){
var path__$1 = ((cljs.core.coll_QMARK_(path))?path:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [path], null));
var vec__46292 = (((atm instanceof crate.binding.SubAtom))?new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [atm.atm,cljs.core.concat.cljs$core$IFn$_invoke$arity$2(atm.path,path__$1)], null):new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [atm,path__$1], null));
var atm__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46292,(0),null);
var path__$2 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46292,(1),null);
var k = cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("subatom");
var sa = (new crate.binding.SubAtom(atm__$1,path__$2,cljs.core.hash(cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(atm__$1),path__$2)),null,k));
cljs.core.add_watch(atm__$1,k,(function (_,___$1,ov,nv){
var latest = cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(nv,path__$2);
var prev = cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(ov,path__$2);
var latest_hash = cljs.core.hash(latest);
if(((cljs.core.not_EQ_.cljs$core$IFn$_invoke$arity$2(sa.prevhash,latest_hash)) && (cljs.core.not_EQ_.cljs$core$IFn$_invoke$arity$2(prev,latest)))){
(sa.prevhash = latest_hash);

return sa.cljs$core$IWatchable$_notify_watches$arity$3(null,cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(ov,path__$2),latest);
} else {
return null;
}
}));

return sa;
});
/**
 * Sets the value of atom to newval without regard for the
 *   current value. Returns newval.
 */
crate.binding.sub_reset_BANG_ = (function crate$binding$sub_reset_BANG_(sa,new_value){
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$4(sa.atm,cljs.core.assoc_in,sa.path,new_value);

return new_value;
});
/**
 * Atomically swaps the value of atom to be:
 *   (apply f current-value-of-atom args). Note that f may be called
 *   multiple times, and thus should be free of side effects.  Returns
 *   the value that was swapped in.
 */
crate.binding.sub_swap_BANG_ = (function crate$binding$sub_swap_BANG_(var_args){
var G__46315 = arguments.length;
switch (G__46315) {
case 2:
return crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
case 5:
return crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$5((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]));

break;
default:
var args_arr__4757__auto__ = [];
var len__4736__auto___46894 = arguments.length;
var i__4737__auto___46895 = (0);
while(true){
if((i__4737__auto___46895 < len__4736__auto___46894)){
args_arr__4757__auto__.push((arguments[i__4737__auto___46895]));

var G__46900 = (i__4737__auto___46895 + (1));
i__4737__auto___46895 = G__46900;
continue;
} else {
}
break;
}

var argseq__4758__auto__ = (new cljs.core.IndexedSeq(args_arr__4757__auto__.slice((5)),(0),null));
return crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]),argseq__4758__auto__);

}
});

(crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (sa,f){
return crate.binding.sub_reset_BANG_(sa,(function (){var G__46328 = cljs.core.deref(sa);
return (f.cljs$core$IFn$_invoke$arity$1 ? f.cljs$core$IFn$_invoke$arity$1(G__46328) : f.call(null,G__46328));
})());
}));

(crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (sa,f,x){
return crate.binding.sub_reset_BANG_(sa,(function (){var G__46335 = cljs.core.deref(sa);
var G__46336 = x;
return (f.cljs$core$IFn$_invoke$arity$2 ? f.cljs$core$IFn$_invoke$arity$2(G__46335,G__46336) : f.call(null,G__46335,G__46336));
})());
}));

(crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$4 = (function (sa,f,x,y){
return crate.binding.sub_reset_BANG_(sa,(function (){var G__46344 = cljs.core.deref(sa);
var G__46345 = x;
var G__46346 = y;
return (f.cljs$core$IFn$_invoke$arity$3 ? f.cljs$core$IFn$_invoke$arity$3(G__46344,G__46345,G__46346) : f.call(null,G__46344,G__46345,G__46346));
})());
}));

(crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$5 = (function (sa,f,x,y,z){
return crate.binding.sub_reset_BANG_(sa,(function (){var G__46352 = cljs.core.deref(sa);
var G__46353 = x;
var G__46354 = y;
var G__46355 = z;
return (f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(G__46352,G__46353,G__46354,G__46355) : f.call(null,G__46352,G__46353,G__46354,G__46355));
})());
}));

(crate.binding.sub_swap_BANG_.cljs$core$IFn$_invoke$arity$variadic = (function (sa,f,x,y,z,more){
return crate.binding.sub_reset_BANG_(sa,cljs.core.apply.cljs$core$IFn$_invoke$arity$variadic(f,cljs.core.deref(sa),x,y,z,cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([more], 0)));
}));

/** @this {Function} */
(crate.binding.sub_swap_BANG_.cljs$lang$applyTo = (function (seq46305){
var G__46306 = cljs.core.first(seq46305);
var seq46305__$1 = cljs.core.next(seq46305);
var G__46308 = cljs.core.first(seq46305__$1);
var seq46305__$2 = cljs.core.next(seq46305__$1);
var G__46312 = cljs.core.first(seq46305__$2);
var seq46305__$3 = cljs.core.next(seq46305__$2);
var G__46313 = cljs.core.first(seq46305__$3);
var seq46305__$4 = cljs.core.next(seq46305__$3);
var G__46314 = cljs.core.first(seq46305__$4);
var seq46305__$5 = cljs.core.next(seq46305__$4);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46306,G__46308,G__46312,G__46313,G__46314,seq46305__$5);
}));

(crate.binding.sub_swap_BANG_.cljs$lang$maxFixedArity = (5));

crate.binding.sub_destroy_BANG_ = (function crate$binding$sub_destroy_BANG_(sa){
cljs.core.remove_watch(sa.atm,sa.key);

(sa.watches = null);

return (sa.atm = null);
});

/**
 * @interface
 */
crate.binding.computable = function(){};

var crate$binding$computable$_depend$dyn_46932 = (function (this$,atm){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (crate.binding._depend[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,atm) : m__4429__auto__.call(null,this$,atm));
} else {
var m__4426__auto__ = (crate.binding._depend["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,atm) : m__4426__auto__.call(null,this$,atm));
} else {
throw cljs.core.missing_protocol("computable.-depend",this$);
}
}
});
/**
 * depend on an atom
 */
crate.binding._depend = (function crate$binding$_depend(this$,atm){
if((((!((this$ == null)))) && ((!((this$.crate$binding$computable$_depend$arity$2 == null)))))){
return this$.crate$binding$computable$_depend$arity$2(this$,atm);
} else {
return crate$binding$computable$_depend$dyn_46932(this$,atm);
}
});

var crate$binding$computable$_compute$dyn_46964 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (crate.binding._compute[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (crate.binding._compute["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("computable.-compute",this$);
}
}
});
/**
 * compute the latest value
 */
crate.binding._compute = (function crate$binding$_compute(this$){
if((((!((this$ == null)))) && ((!((this$.crate$binding$computable$_compute$arity$1 == null)))))){
return this$.crate$binding$computable$_compute$arity$1(this$);
} else {
return crate$binding$computable$_compute$dyn_46964(this$);
}
});


/**
* @constructor
 * @implements {cljs.core.IWatchable}
 * @implements {crate.binding.computable}
 * @implements {cljs.core.IEquiv}
 * @implements {cljs.core.IHash}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IDeref}
 * @implements {cljs.core.IPrintWithWriter}
*/
crate.binding.Computed = (function (atms,value,func,watches,key,meta){
this.atms = atms;
this.value = value;
this.func = func;
this.watches = watches;
this.key = key;
this.meta = meta;
this.cljs$lang$protocol_mask$partition0$ = 2153938944;
this.cljs$lang$protocol_mask$partition1$ = 2;
});
(crate.binding.Computed.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (o,other){
var self__ = this;
var o__$1 = this;
return (o__$1 === other);
}));

(crate.binding.Computed.prototype.cljs$core$IDeref$_deref$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.value;
}));

(crate.binding.Computed.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this$,writer,opts){
var self__ = this;
var this$__$1 = this;
return cljs.core._write(writer,["#<Computed: ",cljs.core.pr_str.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([self__.value], 0)),">"].join(''));
}));

(crate.binding.Computed.prototype.cljs$core$IWatchable$_notify_watches$arity$3 = (function (this$,oldval,newval){
var self__ = this;
var this$__$1 = this;
var seq__46412 = cljs.core.seq(self__.watches);
var chunk__46413 = null;
var count__46414 = (0);
var i__46415 = (0);
while(true){
if((i__46415 < count__46414)){
var vec__46431 = chunk__46413.cljs$core$IIndexed$_nth$arity$2(null,i__46415);
var key__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46431,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46431,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key__$1,this$__$1,oldval,newval) : f.call(null,key__$1,this$__$1,oldval,newval));


var G__46966 = seq__46412;
var G__46967 = chunk__46413;
var G__46968 = count__46414;
var G__46969 = (i__46415 + (1));
seq__46412 = G__46966;
chunk__46413 = G__46967;
count__46414 = G__46968;
i__46415 = G__46969;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46412);
if(temp__5735__auto__){
var seq__46412__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46412__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46412__$1);
var G__46978 = cljs.core.chunk_rest(seq__46412__$1);
var G__46979 = c__4556__auto__;
var G__46980 = cljs.core.count(c__4556__auto__);
var G__46981 = (0);
seq__46412 = G__46978;
chunk__46413 = G__46979;
count__46414 = G__46980;
i__46415 = G__46981;
continue;
} else {
var vec__46439 = cljs.core.first(seq__46412__$1);
var key__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46439,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46439,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key__$1,this$__$1,oldval,newval) : f.call(null,key__$1,this$__$1,oldval,newval));


var G__46984 = cljs.core.next(seq__46412__$1);
var G__46985 = null;
var G__46986 = (0);
var G__46987 = (0);
seq__46412 = G__46984;
chunk__46413 = G__46985;
count__46414 = G__46986;
i__46415 = G__46987;
continue;
}
} else {
return null;
}
}
break;
}
}));

(crate.binding.Computed.prototype.cljs$core$IWatchable$_add_watch$arity$3 = (function (this$,key__$1,f){
var self__ = this;
var this$__$1 = this;
if(cljs.core.truth_(f)){
return (this$__$1.watches = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.watches,key__$1,f));
} else {
return null;
}
}));

(crate.binding.Computed.prototype.cljs$core$IWatchable$_remove_watch$arity$2 = (function (this$,key__$1){
var self__ = this;
var this$__$1 = this;
return (this$__$1.watches = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(self__.watches,key__$1));
}));

(crate.binding.Computed.prototype.cljs$core$IHash$_hash$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return goog.getUid(this$__$1);
}));

(crate.binding.Computed.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return self__.meta;
}));

(crate.binding.Computed.prototype.crate$binding$computable$ = cljs.core.PROTOCOL_SENTINEL);

(crate.binding.Computed.prototype.crate$binding$computable$_depend$arity$2 = (function (this$,atm){
var self__ = this;
var this$__$1 = this;
(this$__$1.atms = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(this$__$1.atms,atm));

return cljs.core.add_watch(atm,self__.key,(function (_,___$1,___$2,___$3){
return this$__$1.crate$binding$computable$_compute$arity$1(null);
}));
}));

(crate.binding.Computed.prototype.crate$binding$computable$_compute$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
var old = this$__$1.value;
var nv = cljs.core.apply.cljs$core$IFn$_invoke$arity$2(self__.func,cljs.core.map.cljs$core$IFn$_invoke$arity$2(cljs.core.deref,self__.atms));
(this$__$1.value = nv);

return this$__$1.cljs$core$IWatchable$_notify_watches$arity$3(null,old,nv);
}));

(crate.binding.Computed.getBasis = (function (){
return new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"atms","atms",-855465715,null),new cljs.core.Symbol(null,"value","value",1946509744,null),new cljs.core.Symbol(null,"func","func",1401825487,null),new cljs.core.Symbol(null,"watches","watches",1367433992,null),new cljs.core.Symbol(null,"key","key",124488940,null),new cljs.core.Symbol(null,"meta","meta",-1154898805,null)], null);
}));

(crate.binding.Computed.cljs$lang$type = true);

(crate.binding.Computed.cljs$lang$ctorStr = "crate.binding/Computed");

(crate.binding.Computed.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"crate.binding/Computed");
}));

/**
 * Positional factory function for crate.binding/Computed.
 */
crate.binding.__GT_Computed = (function crate$binding$__GT_Computed(atms,value,func,watches,key,meta){
return (new crate.binding.Computed(atms,value,func,watches,key,meta));
});

crate.binding.computed = (function crate$binding$computed(atms,func){
var k = cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("computed");
var neue = (new crate.binding.Computed(cljs.core.PersistentVector.EMPTY,null,func,null,k,null));
neue.crate$binding$computable$_compute$arity$1(null);

var seq__46452_46995 = cljs.core.seq(atms);
var chunk__46453_46996 = null;
var count__46454_46997 = (0);
var i__46455_46998 = (0);
while(true){
if((i__46455_46998 < count__46454_46997)){
var atm_46999 = chunk__46453_46996.cljs$core$IIndexed$_nth$arity$2(null,i__46455_46998);
neue.crate$binding$computable$_depend$arity$2(null,atm_46999);


var G__47000 = seq__46452_46995;
var G__47001 = chunk__46453_46996;
var G__47002 = count__46454_46997;
var G__47003 = (i__46455_46998 + (1));
seq__46452_46995 = G__47000;
chunk__46453_46996 = G__47001;
count__46454_46997 = G__47002;
i__46455_46998 = G__47003;
continue;
} else {
var temp__5735__auto___47005 = cljs.core.seq(seq__46452_46995);
if(temp__5735__auto___47005){
var seq__46452_47006__$1 = temp__5735__auto___47005;
if(cljs.core.chunked_seq_QMARK_(seq__46452_47006__$1)){
var c__4556__auto___47007 = cljs.core.chunk_first(seq__46452_47006__$1);
var G__47008 = cljs.core.chunk_rest(seq__46452_47006__$1);
var G__47009 = c__4556__auto___47007;
var G__47010 = cljs.core.count(c__4556__auto___47007);
var G__47011 = (0);
seq__46452_46995 = G__47008;
chunk__46453_46996 = G__47009;
count__46454_46997 = G__47010;
i__46455_46998 = G__47011;
continue;
} else {
var atm_47012 = cljs.core.first(seq__46452_47006__$1);
neue.crate$binding$computable$_depend$arity$2(null,atm_47012);


var G__47014 = cljs.core.next(seq__46452_47006__$1);
var G__47015 = null;
var G__47016 = (0);
var G__47017 = (0);
seq__46452_46995 = G__47014;
chunk__46453_46996 = G__47015;
count__46454_46997 = G__47016;
i__46455_46998 = G__47017;
continue;
}
} else {
}
}
break;
}

return neue;
});
crate.binding.compute = (function crate$binding$compute(compu){
return crate.binding._compute(compu);
});
crate.binding.depend_on = (function crate$binding$depend_on(compu,atm){
return crate.binding._depend(compu,atm);
});
crate.binding.notify = (function crate$binding$notify(w,o,v){
return cljs.core._notify_watches(w,o,v);
});

/**
 * @interface
 */
crate.binding.bindable_coll = function(){};


/**
 * @interface
 */
crate.binding.bindable = function(){};

var crate$binding$bindable$_value$dyn_47018 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (crate.binding._value[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (crate.binding._value["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("bindable.-value",this$);
}
}
});
/**
 * get the current value of this binding
 */
crate.binding._value = (function crate$binding$_value(this$){
if((((!((this$ == null)))) && ((!((this$.crate$binding$bindable$_value$arity$1 == null)))))){
return this$.crate$binding$bindable$_value$arity$1(this$);
} else {
return crate$binding$bindable$_value$dyn_47018(this$);
}
});

var crate$binding$bindable$_on_change$dyn_47026 = (function (this$,func){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (crate.binding._on_change[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,func) : m__4429__auto__.call(null,this$,func));
} else {
var m__4426__auto__ = (crate.binding._on_change["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,func) : m__4426__auto__.call(null,this$,func));
} else {
throw cljs.core.missing_protocol("bindable.-on-change",this$);
}
}
});
/**
 * On change of this binding execute func
 */
crate.binding._on_change = (function crate$binding$_on_change(this$,func){
if((((!((this$ == null)))) && ((!((this$.crate$binding$bindable$_on_change$arity$2 == null)))))){
return this$.crate$binding$bindable$_on_change$arity$2(this$,func);
} else {
return crate$binding$bindable$_on_change$dyn_47026(this$,func);
}
});


/**
* @constructor
 * @implements {crate.binding.bindable}
*/
crate.binding.atom_binding = (function (atm,value_func){
this.atm = atm;
this.value_func = value_func;
});
(crate.binding.atom_binding.prototype.crate$binding$bindable$ = cljs.core.PROTOCOL_SENTINEL);

(crate.binding.atom_binding.prototype.crate$binding$bindable$_value$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
var G__46578 = cljs.core.deref(self__.atm);
return (self__.value_func.cljs$core$IFn$_invoke$arity$1 ? self__.value_func.cljs$core$IFn$_invoke$arity$1(G__46578) : self__.value_func.call(null,G__46578));
}));

(crate.binding.atom_binding.prototype.crate$binding$bindable$_on_change$arity$2 = (function (this$,func){
var self__ = this;
var this$__$1 = this;
return cljs.core.add_watch(self__.atm,cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("atom-binding"),(function (){
var G__46580 = this$__$1.crate$binding$bindable$_value$arity$1(null);
return (func.cljs$core$IFn$_invoke$arity$1 ? func.cljs$core$IFn$_invoke$arity$1(G__46580) : func.call(null,G__46580));
}));
}));

(crate.binding.atom_binding.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"atm","atm",-1963551835,null),new cljs.core.Symbol(null,"value-func","value-func",2077951825,null)], null);
}));

(crate.binding.atom_binding.cljs$lang$type = true);

(crate.binding.atom_binding.cljs$lang$ctorStr = "crate.binding/atom-binding");

(crate.binding.atom_binding.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"crate.binding/atom-binding");
}));

/**
 * Positional factory function for crate.binding/atom-binding.
 */
crate.binding.__GT_atom_binding = (function crate$binding$__GT_atom_binding(atm,value_func){
return (new crate.binding.atom_binding(atm,value_func));
});


/**
* @constructor
 * @implements {cljs.core.IWatchable}
*/
crate.binding.notifier = (function (watches){
this.watches = watches;
this.cljs$lang$protocol_mask$partition1$ = 2;
this.cljs$lang$protocol_mask$partition0$ = 0;
});
(crate.binding.notifier.prototype.cljs$core$IWatchable$_notify_watches$arity$3 = (function (this$,oldval,newval){
var self__ = this;
var this$__$1 = this;
var seq__46591 = cljs.core.seq(self__.watches);
var chunk__46597 = null;
var count__46600 = (0);
var i__46601 = (0);
while(true){
if((i__46601 < count__46600)){
var vec__46649 = chunk__46597.cljs$core$IIndexed$_nth$arity$2(null,i__46601);
var key = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46649,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46649,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key,this$__$1,oldval,newval) : f.call(null,key,this$__$1,oldval,newval));


var G__47045 = seq__46591;
var G__47046 = chunk__46597;
var G__47047 = count__46600;
var G__47048 = (i__46601 + (1));
seq__46591 = G__47045;
chunk__46597 = G__47046;
count__46600 = G__47047;
i__46601 = G__47048;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46591);
if(temp__5735__auto__){
var seq__46591__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46591__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46591__$1);
var G__47049 = cljs.core.chunk_rest(seq__46591__$1);
var G__47050 = c__4556__auto__;
var G__47051 = cljs.core.count(c__4556__auto__);
var G__47052 = (0);
seq__46591 = G__47049;
chunk__46597 = G__47050;
count__46600 = G__47051;
i__46601 = G__47052;
continue;
} else {
var vec__46652 = cljs.core.first(seq__46591__$1);
var key = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46652,(0),null);
var f = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46652,(1),null);
(f.cljs$core$IFn$_invoke$arity$4 ? f.cljs$core$IFn$_invoke$arity$4(key,this$__$1,oldval,newval) : f.call(null,key,this$__$1,oldval,newval));


var G__47053 = cljs.core.next(seq__46591__$1);
var G__47054 = null;
var G__47055 = (0);
var G__47056 = (0);
seq__46591 = G__47053;
chunk__46597 = G__47054;
count__46600 = G__47055;
i__46601 = G__47056;
continue;
}
} else {
return null;
}
}
break;
}
}));

(crate.binding.notifier.prototype.cljs$core$IWatchable$_add_watch$arity$3 = (function (this$,key,f){
var self__ = this;
var this$__$1 = this;
return (this$__$1.watches = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.watches,key,f));
}));

(crate.binding.notifier.prototype.cljs$core$IWatchable$_remove_watch$arity$2 = (function (this$,key){
var self__ = this;
var this$__$1 = this;
return (this$__$1.watches = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(self__.watches,key));
}));

(crate.binding.notifier.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"watches","watches",1367433992,null)], null);
}));

(crate.binding.notifier.cljs$lang$type = true);

(crate.binding.notifier.cljs$lang$ctorStr = "crate.binding/notifier");

(crate.binding.notifier.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"crate.binding/notifier");
}));

/**
 * Positional factory function for crate.binding/notifier.
 */
crate.binding.__GT_notifier = (function crate$binding$__GT_notifier(watches){
return (new crate.binding.notifier(watches));
});


/**
* @constructor
 * @implements {crate.binding.bindable}
 * @implements {crate.binding.bindable_coll}
*/
crate.binding.bound_collection = (function (atm,notif,opts,stuff){
this.atm = atm;
this.notif = notif;
this.opts = opts;
this.stuff = stuff;
});
(crate.binding.bound_collection.prototype.crate$binding$bindable_coll$ = cljs.core.PROTOCOL_SENTINEL);

(crate.binding.bound_collection.prototype.crate$binding$bindable$ = cljs.core.PROTOCOL_SENTINEL);

(crate.binding.bound_collection.prototype.crate$binding$bindable$_value$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return cljs.core.map.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"elem","elem",618631056),cljs.core.vals(this$__$1.stuff));
}));

(crate.binding.bound_collection.prototype.crate$binding$bindable$_on_change$arity$2 = (function (this$,func){
var self__ = this;
var this$__$1 = this;
return cljs.core.add_watch(self__.notif,cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("bound-coll"),(function (_,___$1,___$2,p__46669){
var vec__46670 = p__46669;
var event = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46670,(0),null);
var el = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46670,(1),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46670,(2),null);
return (func.cljs$core$IFn$_invoke$arity$3 ? func.cljs$core$IFn$_invoke$arity$3(event,el,v) : func.call(null,event,el,v));
}));
}));

(crate.binding.bound_collection.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"atm","atm",-1963551835,null),new cljs.core.Symbol(null,"notif","notif",-1551848296,null),new cljs.core.Symbol(null,"opts","opts",1795607228,null),new cljs.core.Symbol(null,"stuff","stuff",-411032116,null)], null);
}));

(crate.binding.bound_collection.cljs$lang$type = true);

(crate.binding.bound_collection.cljs$lang$ctorStr = "crate.binding/bound-collection");

(crate.binding.bound_collection.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"crate.binding/bound-collection");
}));

/**
 * Positional factory function for crate.binding/bound-collection.
 */
crate.binding.__GT_bound_collection = (function crate$binding$__GT_bound_collection(atm,notif,opts,stuff){
return (new crate.binding.bound_collection(atm,notif,opts,stuff));
});

crate.binding.opt = (function crate$binding$opt(bc,k){
var fexpr__46675 = bc.opts;
return (fexpr__46675.cljs$core$IFn$_invoke$arity$1 ? fexpr__46675.cljs$core$IFn$_invoke$arity$1(k) : fexpr__46675.call(null,k));
});
crate.binding.bc_add = (function crate$binding$bc_add(bc,path,key){
var sa = crate.binding.subatom(bc.atm,path);
var elem = (function (){var fexpr__46680 = crate.binding.opt(bc,new cljs.core.Keyword(null,"as","as",1148689641));
return (fexpr__46680.cljs$core$IFn$_invoke$arity$1 ? fexpr__46680.cljs$core$IFn$_invoke$arity$1(sa) : fexpr__46680.call(null,sa));
})();
(bc.stuff = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(bc.stuff,key,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"elem","elem",618631056),elem,new cljs.core.Keyword(null,"subatom","subatom",-95454370),sa], null)));

return crate.binding.notify(bc.notif,null,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"add","add",235287739),elem,cljs.core.deref(sa)], null));
});
crate.binding.bc_remove = (function crate$binding$bc_remove(bc,key){
var notif = bc.notif;
var prev = (function (){var fexpr__46684 = bc.stuff;
return (fexpr__46684.cljs$core$IFn$_invoke$arity$1 ? fexpr__46684.cljs$core$IFn$_invoke$arity$1(key) : fexpr__46684.call(null,key));
})();
(bc.stuff = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(bc.stuff,key));

crate.binding.notify(bc.notif,null,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"remove","remove",-131428414),new cljs.core.Keyword(null,"elem","elem",618631056).cljs$core$IFn$_invoke$arity$1(prev),null], null));

return crate.binding.sub_destroy_BANG_(new cljs.core.Keyword(null,"subatom","subatom",-95454370).cljs$core$IFn$_invoke$arity$1(prev));
});
crate.binding.__GT_indexed = (function crate$binding$__GT_indexed(coll){
if(cljs.core.map_QMARK_(coll)){
return cljs.core.seq(coll);
} else {
if(cljs.core.set_QMARK_(coll)){
return cljs.core.map.cljs$core$IFn$_invoke$arity$2(cljs.core.juxt.cljs$core$IFn$_invoke$arity$2(cljs.core.identity,cljs.core.identity),coll);
} else {
return cljs.core.map_indexed.cljs$core$IFn$_invoke$arity$2(cljs.core.vector,coll);

}
}
});
crate.binding.__GT_keyed = (function crate$binding$__GT_keyed(coll,keyfn){
return cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentHashSet.EMPTY,cljs.core.map.cljs$core$IFn$_invoke$arity$2(keyfn,crate.binding.__GT_indexed(coll)));
});
crate.binding.__GT_path = (function crate$binding$__GT_path(var_args){
var args__4742__auto__ = [];
var len__4736__auto___47096 = arguments.length;
var i__4737__auto___47100 = (0);
while(true){
if((i__4737__auto___47100 < len__4736__auto___47096)){
args__4742__auto__.push((arguments[i__4737__auto___47100]));

var G__47101 = (i__4737__auto___47100 + (1));
i__4737__auto___47100 = G__47101;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return crate.binding.__GT_path.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(crate.binding.__GT_path.cljs$core$IFn$_invoke$arity$variadic = (function (bc,segs){
return cljs.core.concat.cljs$core$IFn$_invoke$arity$2((function (){var or__4126__auto__ = crate.binding.opt(bc,new cljs.core.Keyword(null,"path","path",-188191168));
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return cljs.core.PersistentVector.EMPTY;
}
})(),segs);
}));

(crate.binding.__GT_path.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(crate.binding.__GT_path.cljs$lang$applyTo = (function (seq46690){
var G__46691 = cljs.core.first(seq46690);
var seq46690__$1 = cljs.core.next(seq46690);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46691,seq46690__$1);
}));

crate.binding.bc_compare = (function crate$binding$bc_compare(bc,neue){
var prev = bc.stuff;
var pset = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentHashSet.EMPTY,cljs.core.keys(prev));
var nset = crate.binding.__GT_keyed(neue,crate.binding.opt(bc,new cljs.core.Keyword(null,"keyfn","keyfn",780060332)));
var added = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.sorted_set(),clojure.set.difference.cljs$core$IFn$_invoke$arity$2(nset,pset));
var removed = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.sorted_set(),clojure.set.difference.cljs$core$IFn$_invoke$arity$2(pset,nset));
var seq__46698_47109 = cljs.core.seq(added);
var chunk__46699_47110 = null;
var count__46700_47111 = (0);
var i__46701_47112 = (0);
while(true){
if((i__46701_47112 < count__46700_47111)){
var a_47113 = chunk__46699_47110.cljs$core$IIndexed$_nth$arity$2(null,i__46701_47112);
crate.binding.bc_add(bc,a_47113,a_47113);


var G__47117 = seq__46698_47109;
var G__47118 = chunk__46699_47110;
var G__47119 = count__46700_47111;
var G__47120 = (i__46701_47112 + (1));
seq__46698_47109 = G__47117;
chunk__46699_47110 = G__47118;
count__46700_47111 = G__47119;
i__46701_47112 = G__47120;
continue;
} else {
var temp__5735__auto___47121 = cljs.core.seq(seq__46698_47109);
if(temp__5735__auto___47121){
var seq__46698_47122__$1 = temp__5735__auto___47121;
if(cljs.core.chunked_seq_QMARK_(seq__46698_47122__$1)){
var c__4556__auto___47123 = cljs.core.chunk_first(seq__46698_47122__$1);
var G__47124 = cljs.core.chunk_rest(seq__46698_47122__$1);
var G__47125 = c__4556__auto___47123;
var G__47126 = cljs.core.count(c__4556__auto___47123);
var G__47127 = (0);
seq__46698_47109 = G__47124;
chunk__46699_47110 = G__47125;
count__46700_47111 = G__47126;
i__46701_47112 = G__47127;
continue;
} else {
var a_47129 = cljs.core.first(seq__46698_47122__$1);
crate.binding.bc_add(bc,a_47129,a_47129);


var G__47130 = cljs.core.next(seq__46698_47122__$1);
var G__47131 = null;
var G__47132 = (0);
var G__47133 = (0);
seq__46698_47109 = G__47130;
chunk__46699_47110 = G__47131;
count__46700_47111 = G__47132;
i__46701_47112 = G__47133;
continue;
}
} else {
}
}
break;
}

var seq__46705 = cljs.core.seq(removed);
var chunk__46706 = null;
var count__46707 = (0);
var i__46708 = (0);
while(true){
if((i__46708 < count__46707)){
var r = chunk__46706.cljs$core$IIndexed$_nth$arity$2(null,i__46708);
crate.binding.bc_remove(bc,r);


var G__47134 = seq__46705;
var G__47135 = chunk__46706;
var G__47136 = count__46707;
var G__47137 = (i__46708 + (1));
seq__46705 = G__47134;
chunk__46706 = G__47135;
count__46707 = G__47136;
i__46708 = G__47137;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46705);
if(temp__5735__auto__){
var seq__46705__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46705__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46705__$1);
var G__47138 = cljs.core.chunk_rest(seq__46705__$1);
var G__47139 = c__4556__auto__;
var G__47140 = cljs.core.count(c__4556__auto__);
var G__47141 = (0);
seq__46705 = G__47138;
chunk__46706 = G__47139;
count__46707 = G__47140;
i__46708 = G__47141;
continue;
} else {
var r = cljs.core.first(seq__46705__$1);
crate.binding.bc_remove(bc,r);


var G__47143 = cljs.core.next(seq__46705__$1);
var G__47144 = null;
var G__47145 = (0);
var G__47146 = (0);
seq__46705 = G__47143;
chunk__46706 = G__47144;
count__46707 = G__47145;
i__46708 = G__47146;
continue;
}
} else {
return null;
}
}
break;
}
});
crate.binding.bound_coll = (function crate$binding$bound_coll(var_args){
var args__4742__auto__ = [];
var len__4736__auto___47148 = arguments.length;
var i__4737__auto___47149 = (0);
while(true){
if((i__4737__auto___47149 < len__4736__auto___47148)){
args__4742__auto__.push((arguments[i__4737__auto___47149]));

var G__47150 = (i__4737__auto___47149 + (1));
i__4737__auto___47149 = G__47150;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return crate.binding.bound_coll.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(crate.binding.bound_coll.cljs$core$IFn$_invoke$arity$variadic = (function (atm,p__46717){
var vec__46718 = p__46717;
var path = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46718,(0),null);
var opts = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46718,(1),null);
var vec__46724 = (cljs.core.truth_(opts)?new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [path,opts], null):new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [null,path], null));
var path__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46724,(0),null);
var opts__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46724,(1),null);
var atm__$1 = ((cljs.core.not(path__$1))?atm:crate.binding.subatom(atm,path__$1));
var opts__$2 = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts__$1,new cljs.core.Keyword(null,"path","path",-188191168),path__$1);
var opts__$3 = ((cljs.core.not(new cljs.core.Keyword(null,"keyfn","keyfn",780060332).cljs$core$IFn$_invoke$arity$1(opts__$2)))?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts__$2,new cljs.core.Keyword(null,"keyfn","keyfn",780060332),cljs.core.first):cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts__$2,new cljs.core.Keyword(null,"keyfn","keyfn",780060332),cljs.core.comp.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"keyfn","keyfn",780060332).cljs$core$IFn$_invoke$arity$1(opts__$2),cljs.core.second)));
var bc = (new crate.binding.bound_collection(atm__$1,(new crate.binding.notifier(null)),opts__$3,cljs.core.sorted_map()));
cljs.core.add_watch(atm__$1,cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("bound-coll"),(function (_,___$1,___$2,neue){
return crate.binding.bc_compare(bc,neue);
}));

crate.binding.bc_compare(bc,cljs.core.deref(atm__$1));

return bc;
}));

(crate.binding.bound_coll.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(crate.binding.bound_coll.cljs$lang$applyTo = (function (seq46715){
var G__46716 = cljs.core.first(seq46715);
var seq46715__$1 = cljs.core.next(seq46715);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46716,seq46715__$1);
}));

crate.binding.map_bound = (function crate$binding$map_bound(var_args){
var args__4742__auto__ = [];
var len__4736__auto___47167 = arguments.length;
var i__4737__auto___47168 = (0);
while(true){
if((i__4737__auto___47168 < len__4736__auto___47167)){
args__4742__auto__.push((arguments[i__4737__auto___47168]));

var G__47171 = (i__4737__auto___47168 + (1));
i__4737__auto___47168 = G__47171;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((2) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((2)),(0),null)):null);
return crate.binding.map_bound.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),(arguments[(1)]),argseq__4743__auto__);
});

(crate.binding.map_bound.cljs$core$IFn$_invoke$arity$variadic = (function (as,atm,p__46739){
var vec__46741 = p__46739;
var opts = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46741,(0),null);
var opts__$1 = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts,new cljs.core.Keyword(null,"as","as",1148689641),as);
var atm__$1 = ((cljs.core.not(new cljs.core.Keyword(null,"path","path",-188191168).cljs$core$IFn$_invoke$arity$1(opts__$1)))?atm:crate.binding.subatom(atm,new cljs.core.Keyword(null,"path","path",-188191168).cljs$core$IFn$_invoke$arity$1(opts__$1)));
var opts__$2 = ((cljs.core.not(new cljs.core.Keyword(null,"keyfn","keyfn",780060332).cljs$core$IFn$_invoke$arity$1(opts__$1)))?cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts__$1,new cljs.core.Keyword(null,"keyfn","keyfn",780060332),cljs.core.first):cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(opts__$1,new cljs.core.Keyword(null,"keyfn","keyfn",780060332),cljs.core.comp.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"keyfn","keyfn",780060332).cljs$core$IFn$_invoke$arity$1(opts__$1),cljs.core.second)));
var bc = (new crate.binding.bound_collection(atm__$1,(new crate.binding.notifier(null)),opts__$2,cljs.core.sorted_map()));
cljs.core.add_watch(atm__$1,cljs.core.gensym.cljs$core$IFn$_invoke$arity$1("bound-coll"),(function (_,___$1,___$2,neue){
return crate.binding.bc_compare(bc,neue);
}));

crate.binding.bc_compare(bc,cljs.core.deref(atm__$1));

return bc;
}));

(crate.binding.map_bound.cljs$lang$maxFixedArity = (2));

/** @this {Function} */
(crate.binding.map_bound.cljs$lang$applyTo = (function (seq46728){
var G__46729 = cljs.core.first(seq46728);
var seq46728__$1 = cljs.core.next(seq46728);
var G__46730 = cljs.core.first(seq46728__$1);
var seq46728__$2 = cljs.core.next(seq46728__$1);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46729,G__46730,seq46728__$2);
}));

crate.binding.binding_QMARK_ = (function crate$binding$binding_QMARK_(b){
if((!((b == null)))){
if(((false) || ((cljs.core.PROTOCOL_SENTINEL === b.crate$binding$bindable$)))){
return true;
} else {
if((!b.cljs$lang$protocol_mask$partition$)){
return cljs.core.native_satisfies_QMARK_(crate.binding.bindable,b);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_(crate.binding.bindable,b);
}
});
crate.binding.binding_coll_QMARK_ = (function crate$binding$binding_coll_QMARK_(b){
if((!((b == null)))){
if(((false) || ((cljs.core.PROTOCOL_SENTINEL === b.crate$binding$bindable_coll$)))){
return true;
} else {
if((!b.cljs$lang$protocol_mask$partition$)){
return cljs.core.native_satisfies_QMARK_(crate.binding.bindable_coll,b);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_(crate.binding.bindable_coll,b);
}
});
crate.binding.deref_QMARK_ = (function crate$binding$deref_QMARK_(atm){
if((!((atm == null)))){
if((((atm.cljs$lang$protocol_mask$partition0$ & (32768))) || ((cljs.core.PROTOCOL_SENTINEL === atm.cljs$core$IDeref$)))){
return true;
} else {
if((!atm.cljs$lang$protocol_mask$partition0$)){
return cljs.core.native_satisfies_QMARK_(cljs.core.IDeref,atm);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_(cljs.core.IDeref,atm);
}
});
crate.binding.value = (function crate$binding$value(b){
return crate.binding._value(b);
});
crate.binding.index = (function crate$binding$index(sub_atom){
return cljs.core.last(sub_atom.path);
});
crate.binding.on_change = (function crate$binding$on_change(b,func){
return crate.binding._on_change(b,func);
});
crate.binding.bound = (function crate$binding$bound(var_args){
var args__4742__auto__ = [];
var len__4736__auto___47189 = arguments.length;
var i__4737__auto___47190 = (0);
while(true){
if((i__4737__auto___47190 < len__4736__auto___47189)){
args__4742__auto__.push((arguments[i__4737__auto___47190]));

var G__47194 = (i__4737__auto___47190 + (1));
i__4737__auto___47190 = G__47194;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return crate.binding.bound.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(crate.binding.bound.cljs$core$IFn$_invoke$arity$variadic = (function (atm,p__46787){
var vec__46793 = p__46787;
var func = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46793,(0),null);
var func__$1 = (function (){var or__4126__auto__ = func;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return cljs.core.identity;
}
})();
return (new crate.binding.atom_binding(atm,func__$1));
}));

(crate.binding.bound.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(crate.binding.bound.cljs$lang$applyTo = (function (seq46772){
var G__46773 = cljs.core.first(seq46772);
var seq46772__$1 = cljs.core.next(seq46772);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46773,seq46772__$1);
}));


//# sourceMappingURL=crate.binding.js.map
