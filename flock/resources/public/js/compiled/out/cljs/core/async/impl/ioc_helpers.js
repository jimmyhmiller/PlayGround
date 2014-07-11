// Compiled by ClojureScript 0.0-2202
goog.provide('cljs.core.async.impl.ioc_helpers');
goog.require('cljs.core');
goog.require('cljs.core.async.impl.protocols');
goog.require('cljs.core.async.impl.protocols');
cljs.core.async.impl.ioc_helpers.FN_IDX = 0;
cljs.core.async.impl.ioc_helpers.STATE_IDX = 1;
cljs.core.async.impl.ioc_helpers.VALUE_IDX = 2;
cljs.core.async.impl.ioc_helpers.BINDINGS_IDX = 3;
cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES = 4;
cljs.core.async.impl.ioc_helpers.CURRENT_EXCEPTION = 5;
cljs.core.async.impl.ioc_helpers.USER_START_IDX = 6;
cljs.core.async.impl.ioc_helpers.aset_object = (function aset_object(arr,idx,o){return (arr[idx][o]);
});
cljs.core.async.impl.ioc_helpers.aget_object = (function aget_object(arr,idx){return (arr[idx]);
});
/**
* Returns true if the machine is in a finished state
*/
cljs.core.async.impl.ioc_helpers.finished_QMARK_ = (function finished_QMARK_(state_array){return cljs.core.keyword_identical_QMARK_.call(null,(state_array[cljs.core.async.impl.ioc_helpers.STATE_IDX]),new cljs.core.Keyword(null,"finished","finished",4635210724));
});
cljs.core.async.impl.ioc_helpers.fn_handler = (function fn_handler(f){if(typeof cljs.core.async.impl.ioc_helpers.t18608 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.impl.ioc_helpers.t18608 = (function (f,fn_handler,meta18609){
this.f = f;
this.fn_handler = fn_handler;
this.meta18609 = meta18609;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.impl.ioc_helpers.t18608.cljs$lang$type = true;
cljs.core.async.impl.ioc_helpers.t18608.cljs$lang$ctorStr = "cljs.core.async.impl.ioc-helpers/t18608";
cljs.core.async.impl.ioc_helpers.t18608.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async.impl.ioc-helpers/t18608");
});
cljs.core.async.impl.ioc_helpers.t18608.prototype.cljs$core$async$impl$protocols$Handler$ = true;
cljs.core.async.impl.ioc_helpers.t18608.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return true;
});
cljs.core.async.impl.ioc_helpers.t18608.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return self__.f;
});
cljs.core.async.impl.ioc_helpers.t18608.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_18610){var self__ = this;
var _18610__$1 = this;return self__.meta18609;
});
cljs.core.async.impl.ioc_helpers.t18608.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_18610,meta18609__$1){var self__ = this;
var _18610__$1 = this;return (new cljs.core.async.impl.ioc_helpers.t18608(self__.f,self__.fn_handler,meta18609__$1));
});
cljs.core.async.impl.ioc_helpers.__GT_t18608 = (function __GT_t18608(f__$1,fn_handler__$1,meta18609){return (new cljs.core.async.impl.ioc_helpers.t18608(f__$1,fn_handler__$1,meta18609));
});
}
return (new cljs.core.async.impl.ioc_helpers.t18608(f,fn_handler,null));
});
cljs.core.async.impl.ioc_helpers.run_state_machine = (function run_state_machine(state){return cljs.core.async.impl.ioc_helpers.aget_object.call(null,state,cljs.core.async.impl.ioc_helpers.FN_IDX).call(null,state);
});
cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped = (function run_state_machine_wrapped(state){try{return cljs.core.async.impl.ioc_helpers.run_state_machine.call(null,state);
}catch (e18612){if((e18612 instanceof Object))
{var ex = e18612;cljs.core.async.impl.protocols.close_BANG_.call(null,cljs.core.async.impl.ioc_helpers.aget_object.call(null,state,cljs.core.async.impl.ioc_helpers.USER_START_IDX));
throw ex;
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18612;
} else
{return null;
}
}
}});
cljs.core.async.impl.ioc_helpers.take_BANG_ = (function take_BANG_(state,blk,c){var temp__4124__auto__ = cljs.core.async.impl.protocols.take_BANG_.call(null,c,cljs.core.async.impl.ioc_helpers.fn_handler.call(null,(function (x){var statearr_18615_18617 = state;(statearr_18615_18617[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = x);
(statearr_18615_18617[cljs.core.async.impl.ioc_helpers.STATE_IDX] = blk);
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state);
})));if(cljs.core.truth_(temp__4124__auto__))
{var cb = temp__4124__auto__;var statearr_18616_18618 = state;(statearr_18616_18618[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = cljs.core.deref.call(null,cb));
(statearr_18616_18618[cljs.core.async.impl.ioc_helpers.STATE_IDX] = blk);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
});
cljs.core.async.impl.ioc_helpers.put_BANG_ = (function put_BANG_(state,blk,c,val){var temp__4124__auto__ = cljs.core.async.impl.protocols.put_BANG_.call(null,c,val,cljs.core.async.impl.ioc_helpers.fn_handler.call(null,(function (ret_val){var statearr_18621_18623 = state;(statearr_18621_18623[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = ret_val);
(statearr_18621_18623[cljs.core.async.impl.ioc_helpers.STATE_IDX] = blk);
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state);
})));if(cljs.core.truth_(temp__4124__auto__))
{var cb = temp__4124__auto__;var statearr_18622_18624 = state;(statearr_18622_18624[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = cljs.core.deref.call(null,cb));
(statearr_18622_18624[cljs.core.async.impl.ioc_helpers.STATE_IDX] = blk);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
});
/**
* @param {...*} var_args
*/
cljs.core.async.impl.ioc_helpers.ioc_alts_BANG_ = (function() { 
var ioc_alts_BANG___delegate = function (state,cont_block,ports,p__18625){var map__18630 = p__18625;var map__18630__$1 = ((cljs.core.seq_QMARK_.call(null,map__18630))?cljs.core.apply.call(null,cljs.core.hash_map,map__18630):map__18630);var opts = map__18630__$1;var statearr_18631_18634 = state;(statearr_18631_18634[cljs.core.async.impl.ioc_helpers.STATE_IDX] = cont_block);
var temp__4126__auto__ = cljs.core.async.do_alts.call(null,((function (map__18630,map__18630__$1,opts){
return (function (val){var statearr_18632_18635 = state;(statearr_18632_18635[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = val);
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state);
});})(map__18630,map__18630__$1,opts))
,ports,opts);if(cljs.core.truth_(temp__4126__auto__))
{var cb = temp__4126__auto__;var statearr_18633_18636 = state;(statearr_18633_18636[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = cljs.core.deref.call(null,cb));
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
};
var ioc_alts_BANG_ = function (state,cont_block,ports,var_args){
var p__18625 = null;if (arguments.length > 3) {
  p__18625 = cljs.core.array_seq(Array.prototype.slice.call(arguments, 3),0);} 
return ioc_alts_BANG___delegate.call(this,state,cont_block,ports,p__18625);};
ioc_alts_BANG_.cljs$lang$maxFixedArity = 3;
ioc_alts_BANG_.cljs$lang$applyTo = (function (arglist__18637){
var state = cljs.core.first(arglist__18637);
arglist__18637 = cljs.core.next(arglist__18637);
var cont_block = cljs.core.first(arglist__18637);
arglist__18637 = cljs.core.next(arglist__18637);
var ports = cljs.core.first(arglist__18637);
var p__18625 = cljs.core.rest(arglist__18637);
return ioc_alts_BANG___delegate(state,cont_block,ports,p__18625);
});
ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = ioc_alts_BANG___delegate;
return ioc_alts_BANG_;
})()
;
cljs.core.async.impl.ioc_helpers.return_chan = (function return_chan(state,value){var c = (state[cljs.core.async.impl.ioc_helpers.USER_START_IDX]);if((value == null))
{} else
{cljs.core.async.impl.protocols.put_BANG_.call(null,c,value,cljs.core.async.impl.ioc_helpers.fn_handler.call(null,((function (c){
return (function (){return null;
});})(c))
));
}
cljs.core.async.impl.protocols.close_BANG_.call(null,c);
return c;
});

/**
* @constructor
* @param {*} catch_block
* @param {*} catch_exception
* @param {*} finally_block
* @param {*} continue_block
* @param {*} prev
* @param {*} __meta
* @param {*} __extmap
* @param {*=} __meta 
* @param {*=} __extmap
*/
cljs.core.async.impl.ioc_helpers.ExceptionFrame = (function (catch_block,catch_exception,finally_block,continue_block,prev,__meta,__extmap){
this.catch_block = catch_block;
this.catch_exception = catch_exception;
this.finally_block = finally_block;
this.continue_block = continue_block;
this.prev = prev;
this.__meta = __meta;
this.__extmap = __extmap;
this.cljs$lang$protocol_mask$partition0$ = 2229667594;
this.cljs$lang$protocol_mask$partition1$ = 8192;
if(arguments.length>5){
this.__meta = __meta;
this.__extmap = __extmap;
} else {
this.__meta=null;
this.__extmap=null;
}
})
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__8460__auto__,k__8461__auto__){var self__ = this;
var this__8460__auto____$1 = this;return cljs.core._lookup.call(null,this__8460__auto____$1,k__8461__auto__,null);
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__8462__auto__,k18639,else__8463__auto__){var self__ = this;
var this__8462__auto____$1 = this;if(cljs.core.keyword_identical_QMARK_.call(null,k18639,new cljs.core.Keyword(null,"catch-block","catch-block",2343862893)))
{return self__.catch_block;
} else
{if(cljs.core.keyword_identical_QMARK_.call(null,k18639,new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687)))
{return self__.catch_exception;
} else
{if(cljs.core.keyword_identical_QMARK_.call(null,k18639,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429)))
{return self__.finally_block;
} else
{if(cljs.core.keyword_identical_QMARK_.call(null,k18639,new cljs.core.Keyword(null,"continue-block","continue-block",1486987097)))
{return self__.continue_block;
} else
{if(cljs.core.keyword_identical_QMARK_.call(null,k18639,new cljs.core.Keyword(null,"prev","prev",1017353637)))
{return self__.prev;
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{return cljs.core.get.call(null,self__.__extmap,k18639,else__8463__auto__);
} else
{return null;
}
}
}
}
}
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__8474__auto__,writer__8475__auto__,opts__8476__auto__){var self__ = this;
var this__8474__auto____$1 = this;var pr_pair__8477__auto__ = ((function (this__8474__auto____$1){
return (function (keyval__8478__auto__){return cljs.core.pr_sequential_writer.call(null,writer__8475__auto__,cljs.core.pr_writer,""," ","",opts__8476__auto__,keyval__8478__auto__);
});})(this__8474__auto____$1))
;return cljs.core.pr_sequential_writer.call(null,writer__8475__auto__,pr_pair__8477__auto__,"#cljs.core.async.impl.ioc-helpers.ExceptionFrame{",", ","}",opts__8476__auto__,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),self__.catch_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),self__.catch_exception],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),self__.finally_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"continue-block","continue-block",1486987097),self__.continue_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"prev","prev",1017353637),self__.prev],null))], null),self__.__extmap));
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__8458__auto__){var self__ = this;
var this__8458__auto____$1 = this;return self__.__meta;
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__8454__auto__){var self__ = this;
var this__8454__auto____$1 = this;return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,self__.continue_block,self__.prev,self__.__meta,self__.__extmap,self__.__hash));
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__8464__auto__){var self__ = this;
var this__8464__auto____$1 = this;return (5 + cljs.core.count.call(null,self__.__extmap));
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__8455__auto__){var self__ = this;
var this__8455__auto____$1 = this;var h__8285__auto__ = self__.__hash;if(!((h__8285__auto__ == null)))
{return h__8285__auto__;
} else
{var h__8285__auto____$1 = cljs.core.hash_imap.call(null,this__8455__auto____$1);self__.__hash = h__8285__auto____$1;
return h__8285__auto____$1;
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this__8456__auto__,other__8457__auto__){var self__ = this;
var this__8456__auto____$1 = this;if(cljs.core.truth_((function (){var and__7862__auto__ = other__8457__auto__;if(cljs.core.truth_(and__7862__auto__))
{return ((this__8456__auto____$1.constructor === other__8457__auto__.constructor)) && (cljs.core.equiv_map.call(null,this__8456__auto____$1,other__8457__auto__));
} else
{return and__7862__auto__;
}
})()))
{return true;
} else
{return false;
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__8469__auto__,k__8470__auto__){var self__ = this;
var this__8469__auto____$1 = this;if(cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),null,new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),null,new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),null,new cljs.core.Keyword(null,"prev","prev",1017353637),null,new cljs.core.Keyword(null,"continue-block","continue-block",1486987097),null], null), null),k__8470__auto__))
{return cljs.core.dissoc.call(null,cljs.core.with_meta.call(null,cljs.core.into.call(null,cljs.core.PersistentArrayMap.EMPTY,this__8469__auto____$1),self__.__meta),k__8470__auto__);
} else
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,self__.continue_block,self__.prev,self__.__meta,cljs.core.not_empty.call(null,cljs.core.dissoc.call(null,self__.__extmap,k__8470__auto__)),null));
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__8467__auto__,k__8468__auto__,G__18638){var self__ = this;
var this__8467__auto____$1 = this;var pred__18641 = cljs.core.keyword_identical_QMARK_;var expr__18642 = k__8468__auto__;if(cljs.core.truth_(pred__18641.call(null,new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),expr__18642)))
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(G__18638,self__.catch_exception,self__.finally_block,self__.continue_block,self__.prev,self__.__meta,self__.__extmap,null));
} else
{if(cljs.core.truth_(pred__18641.call(null,new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),expr__18642)))
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,G__18638,self__.finally_block,self__.continue_block,self__.prev,self__.__meta,self__.__extmap,null));
} else
{if(cljs.core.truth_(pred__18641.call(null,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),expr__18642)))
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,G__18638,self__.continue_block,self__.prev,self__.__meta,self__.__extmap,null));
} else
{if(cljs.core.truth_(pred__18641.call(null,new cljs.core.Keyword(null,"continue-block","continue-block",1486987097),expr__18642)))
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,G__18638,self__.prev,self__.__meta,self__.__extmap,null));
} else
{if(cljs.core.truth_(pred__18641.call(null,new cljs.core.Keyword(null,"prev","prev",1017353637),expr__18642)))
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,self__.continue_block,G__18638,self__.__meta,self__.__extmap,null));
} else
{return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,self__.continue_block,self__.prev,self__.__meta,cljs.core.assoc.call(null,self__.__extmap,k__8468__auto__,G__18638),null));
}
}
}
}
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__8472__auto__){var self__ = this;
var this__8472__auto____$1 = this;return cljs.core.seq.call(null,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),self__.catch_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),self__.catch_exception],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),self__.finally_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"continue-block","continue-block",1486987097),self__.continue_block],null)),(new cljs.core.PersistentVector(null,2,5,cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"prev","prev",1017353637),self__.prev],null))], null),self__.__extmap));
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__8459__auto__,G__18638){var self__ = this;
var this__8459__auto____$1 = this;return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(self__.catch_block,self__.catch_exception,self__.finally_block,self__.continue_block,self__.prev,G__18638,self__.__extmap,self__.__hash));
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__8465__auto__,entry__8466__auto__){var self__ = this;
var this__8465__auto____$1 = this;if(cljs.core.vector_QMARK_.call(null,entry__8466__auto__))
{return cljs.core._assoc.call(null,this__8465__auto____$1,cljs.core._nth.call(null,entry__8466__auto__,0),cljs.core._nth.call(null,entry__8466__auto__,1));
} else
{return cljs.core.reduce.call(null,cljs.core._conj,this__8465__auto____$1,entry__8466__auto__);
}
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.cljs$lang$type = true;
cljs.core.async.impl.ioc_helpers.ExceptionFrame.cljs$lang$ctorPrSeq = (function (this__8494__auto__){return cljs.core._conj.call(null,cljs.core.List.EMPTY,"cljs.core.async.impl.ioc-helpers/ExceptionFrame");
});
cljs.core.async.impl.ioc_helpers.ExceptionFrame.cljs$lang$ctorPrWriter = (function (this__8494__auto__,writer__8495__auto__){return cljs.core._write.call(null,writer__8495__auto__,"cljs.core.async.impl.ioc-helpers/ExceptionFrame");
});
cljs.core.async.impl.ioc_helpers.__GT_ExceptionFrame = (function __GT_ExceptionFrame(catch_block,catch_exception,finally_block,continue_block,prev){return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(catch_block,catch_exception,finally_block,continue_block,prev));
});
cljs.core.async.impl.ioc_helpers.map__GT_ExceptionFrame = (function map__GT_ExceptionFrame(G__18640){return (new cljs.core.async.impl.ioc_helpers.ExceptionFrame(new cljs.core.Keyword(null,"catch-block","catch-block",2343862893).cljs$core$IFn$_invoke$arity$1(G__18640),new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687).cljs$core$IFn$_invoke$arity$1(G__18640),new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(G__18640),new cljs.core.Keyword(null,"continue-block","continue-block",1486987097).cljs$core$IFn$_invoke$arity$1(G__18640),new cljs.core.Keyword(null,"prev","prev",1017353637).cljs$core$IFn$_invoke$arity$1(G__18640),null,cljs.core.dissoc.call(null,G__18640,new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),new cljs.core.Keyword(null,"continue-block","continue-block",1486987097),new cljs.core.Keyword(null,"prev","prev",1017353637))));
});
cljs.core.async.impl.ioc_helpers.add_exception_frame = (function add_exception_frame(state,catch_block,catch_exception,finally_block,continue_block){var statearr_18645 = state;(statearr_18645[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = cljs.core.async.impl.ioc_helpers.__GT_ExceptionFrame.call(null,catch_block,catch_exception,finally_block,continue_block,cljs.core.async.impl.ioc_helpers.aget_object.call(null,state,cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES)));
return statearr_18645;
});
cljs.core.async.impl.ioc_helpers.process_exception = (function process_exception(state){while(true){
var exception_frame = cljs.core.async.impl.ioc_helpers.aget_object.call(null,state,cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES);var catch_block = new cljs.core.Keyword(null,"catch-block","catch-block",2343862893).cljs$core$IFn$_invoke$arity$1(exception_frame);var catch_exception = new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687).cljs$core$IFn$_invoke$arity$1(exception_frame);var exception = cljs.core.async.impl.ioc_helpers.aget_object.call(null,state,cljs.core.async.impl.ioc_helpers.CURRENT_EXCEPTION);if(cljs.core.truth_((function (){var and__7862__auto__ = exception;if(cljs.core.truth_(and__7862__auto__))
{return cljs.core.not.call(null,exception_frame);
} else
{return and__7862__auto__;
}
})()))
{throw exception;
} else
{if(cljs.core.truth_((function (){var and__7862__auto__ = exception;if(cljs.core.truth_(and__7862__auto__))
{var and__7862__auto____$1 = catch_block;if(cljs.core.truth_(and__7862__auto____$1))
{return (exception instanceof catch_exception);
} else
{return and__7862__auto____$1;
}
} else
{return and__7862__auto__;
}
})()))
{var statearr_18651 = state;(statearr_18651[cljs.core.async.impl.ioc_helpers.STATE_IDX] = catch_block);
(statearr_18651[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = exception);
(statearr_18651[cljs.core.async.impl.ioc_helpers.CURRENT_EXCEPTION] = null);
(statearr_18651[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = cljs.core.assoc.call(null,exception_frame,new cljs.core.Keyword(null,"catch-block","catch-block",2343862893),null,new cljs.core.Keyword(null,"catch-exception","catch-exception",1686480687),null));
return statearr_18651;
} else
{if(cljs.core.truth_((function (){var and__7862__auto__ = exception;if(cljs.core.truth_(and__7862__auto__))
{return (cljs.core.not.call(null,catch_block)) && (cljs.core.not.call(null,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame)));
} else
{return and__7862__auto__;
}
})()))
{var statearr_18652_18656 = state;(statearr_18652_18656[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = new cljs.core.Keyword(null,"prev","prev",1017353637).cljs$core$IFn$_invoke$arity$1(exception_frame));
{
var G__18657 = state;
state = G__18657;
continue;
}
} else
{if(cljs.core.truth_((function (){var and__7862__auto__ = exception;if(cljs.core.truth_(and__7862__auto__))
{var and__7862__auto____$1 = cljs.core.not.call(null,catch_block);if(and__7862__auto____$1)
{return new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame);
} else
{return and__7862__auto____$1;
}
} else
{return and__7862__auto__;
}
})()))
{var statearr_18653 = state;(statearr_18653[cljs.core.async.impl.ioc_helpers.STATE_IDX] = new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame));
(statearr_18653[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = cljs.core.assoc.call(null,exception_frame,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),null));
return statearr_18653;
} else
{if(cljs.core.truth_((function (){var and__7862__auto__ = cljs.core.not.call(null,exception);if(and__7862__auto__)
{return new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame);
} else
{return and__7862__auto__;
}
})()))
{var statearr_18654 = state;(statearr_18654[cljs.core.async.impl.ioc_helpers.STATE_IDX] = new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame));
(statearr_18654[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = cljs.core.assoc.call(null,exception_frame,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429),null));
return statearr_18654;
} else
{if((cljs.core.not.call(null,exception)) && (cljs.core.not.call(null,new cljs.core.Keyword(null,"finally-block","finally-block",2846533429).cljs$core$IFn$_invoke$arity$1(exception_frame))))
{var statearr_18655 = state;(statearr_18655[cljs.core.async.impl.ioc_helpers.STATE_IDX] = new cljs.core.Keyword(null,"continue-block","continue-block",1486987097).cljs$core$IFn$_invoke$arity$1(exception_frame));
(statearr_18655[cljs.core.async.impl.ioc_helpers.EXCEPTION_FRAMES] = new cljs.core.Keyword(null,"prev","prev",1017353637).cljs$core$IFn$_invoke$arity$1(exception_frame));
return statearr_18655;
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{if(false)
{return null;
} else
{throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str("No matching clause"),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,false))].join('')));
}
} else
{return null;
}
}
}
}
}
}
}
break;
}
});
