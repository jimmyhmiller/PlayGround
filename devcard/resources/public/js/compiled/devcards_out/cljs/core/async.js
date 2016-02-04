// Compiled by ClojureScript 1.7.170 {}
goog.provide('cljs.core.async');
goog.require('cljs.core');
goog.require('cljs.core.async.impl.channels');
goog.require('cljs.core.async.impl.dispatch');
goog.require('cljs.core.async.impl.ioc_helpers');
goog.require('cljs.core.async.impl.protocols');
goog.require('cljs.core.async.impl.buffers');
goog.require('cljs.core.async.impl.timers');
cljs.core.async.fn_handler = (function cljs$core$async$fn_handler(var_args){
var args18978 = [];
var len__17824__auto___18984 = arguments.length;
var i__17825__auto___18985 = (0);
while(true){
if((i__17825__auto___18985 < len__17824__auto___18984)){
args18978.push((arguments[i__17825__auto___18985]));

var G__18986 = (i__17825__auto___18985 + (1));
i__17825__auto___18985 = G__18986;
continue;
} else {
}
break;
}

var G__18980 = args18978.length;
switch (G__18980) {
case 1:
return cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args18978.length)].join('')));

}
});

cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1 = (function (f){
return cljs.core.async.fn_handler.call(null,f,true);
});

cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2 = (function (f,blockable){
if(typeof cljs.core.async.t_cljs$core$async18981 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async18981 = (function (f,blockable,meta18982){
this.f = f;
this.blockable = blockable;
this.meta18982 = meta18982;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_18983,meta18982__$1){
var self__ = this;
var _18983__$1 = this;
return (new cljs.core.async.t_cljs$core$async18981(self__.f,self__.blockable,meta18982__$1));
});

cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_18983){
var self__ = this;
var _18983__$1 = this;
return self__.meta18982;
});

cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$async$impl$protocols$Handler$ = true;

cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return true;
});

cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.blockable;
});

cljs.core.async.t_cljs$core$async18981.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.f;
});

cljs.core.async.t_cljs$core$async18981.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"blockable","blockable",-28395259,null),new cljs.core.Symbol(null,"meta18982","meta18982",1535622489,null)], null);
});

cljs.core.async.t_cljs$core$async18981.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async18981.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async18981";

cljs.core.async.t_cljs$core$async18981.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async18981");
});

cljs.core.async.__GT_t_cljs$core$async18981 = (function cljs$core$async$__GT_t_cljs$core$async18981(f__$1,blockable__$1,meta18982){
return (new cljs.core.async.t_cljs$core$async18981(f__$1,blockable__$1,meta18982));
});

}

return (new cljs.core.async.t_cljs$core$async18981(f,blockable,cljs.core.PersistentArrayMap.EMPTY));
});

cljs.core.async.fn_handler.cljs$lang$maxFixedArity = 2;
/**
 * Returns a fixed buffer of size n. When full, puts will block/park.
 */
cljs.core.async.buffer = (function cljs$core$async$buffer(n){
return cljs.core.async.impl.buffers.fixed_buffer.call(null,n);
});
/**
 * Returns a buffer of size n. When full, puts will complete but
 *   val will be dropped (no transfer).
 */
cljs.core.async.dropping_buffer = (function cljs$core$async$dropping_buffer(n){
return cljs.core.async.impl.buffers.dropping_buffer.call(null,n);
});
/**
 * Returns a buffer of size n. When full, puts will complete, and be
 *   buffered, but oldest elements in buffer will be dropped (not
 *   transferred).
 */
cljs.core.async.sliding_buffer = (function cljs$core$async$sliding_buffer(n){
return cljs.core.async.impl.buffers.sliding_buffer.call(null,n);
});
/**
 * Returns true if a channel created with buff will never block. That is to say,
 * puts into this buffer will never cause the buffer to be full. 
 */
cljs.core.async.unblocking_buffer_QMARK_ = (function cljs$core$async$unblocking_buffer_QMARK_(buff){
if(!((buff == null))){
if((false) || (buff.cljs$core$async$impl$protocols$UnblockingBuffer$)){
return true;
} else {
if((!buff.cljs$lang$protocol_mask$partition$)){
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.async.impl.protocols.UnblockingBuffer,buff);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.async.impl.protocols.UnblockingBuffer,buff);
}
});
/**
 * Creates a channel with an optional buffer, an optional transducer (like (map f),
 *   (filter p) etc or a composition thereof), and an optional exception handler.
 *   If buf-or-n is a number, will create and use a fixed buffer of that size. If a
 *   transducer is supplied a buffer must be specified. ex-handler must be a
 *   fn of one argument - if an exception occurs during transformation it will be called
 *   with the thrown value as an argument, and any non-nil return value will be placed
 *   in the channel.
 */
cljs.core.async.chan = (function cljs$core$async$chan(var_args){
var args18990 = [];
var len__17824__auto___18993 = arguments.length;
var i__17825__auto___18994 = (0);
while(true){
if((i__17825__auto___18994 < len__17824__auto___18993)){
args18990.push((arguments[i__17825__auto___18994]));

var G__18995 = (i__17825__auto___18994 + (1));
i__17825__auto___18994 = G__18995;
continue;
} else {
}
break;
}

var G__18992 = args18990.length;
switch (G__18992) {
case 0:
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$0();

break;
case 1:
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args18990.length)].join('')));

}
});

cljs.core.async.chan.cljs$core$IFn$_invoke$arity$0 = (function (){
return cljs.core.async.chan.call(null,null);
});

cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1 = (function (buf_or_n){
return cljs.core.async.chan.call(null,buf_or_n,null,null);
});

cljs.core.async.chan.cljs$core$IFn$_invoke$arity$2 = (function (buf_or_n,xform){
return cljs.core.async.chan.call(null,buf_or_n,xform,null);
});

cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3 = (function (buf_or_n,xform,ex_handler){
var buf_or_n__$1 = ((cljs.core._EQ_.call(null,buf_or_n,(0)))?null:buf_or_n);
if(cljs.core.truth_(xform)){
if(cljs.core.truth_(buf_or_n__$1)){
} else {
throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str("buffer must be supplied when transducer is"),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,new cljs.core.Symbol(null,"buf-or-n","buf-or-n",-1646815050,null)))].join('')));
}
} else {
}

return cljs.core.async.impl.channels.chan.call(null,((typeof buf_or_n__$1 === 'number')?cljs.core.async.buffer.call(null,buf_or_n__$1):buf_or_n__$1),xform,ex_handler);
});

cljs.core.async.chan.cljs$lang$maxFixedArity = 3;
/**
 * Creates a promise channel with an optional transducer, and an optional
 *   exception-handler. A promise channel can take exactly one value that consumers
 *   will receive. Once full, puts complete but val is dropped (no transfer).
 *   Consumers will block until either a value is placed in the channel or the
 *   channel is closed. See chan for the semantics of xform and ex-handler.
 */
cljs.core.async.promise_chan = (function cljs$core$async$promise_chan(var_args){
var args18997 = [];
var len__17824__auto___19000 = arguments.length;
var i__17825__auto___19001 = (0);
while(true){
if((i__17825__auto___19001 < len__17824__auto___19000)){
args18997.push((arguments[i__17825__auto___19001]));

var G__19002 = (i__17825__auto___19001 + (1));
i__17825__auto___19001 = G__19002;
continue;
} else {
}
break;
}

var G__18999 = args18997.length;
switch (G__18999) {
case 0:
return cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$0();

break;
case 1:
return cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args18997.length)].join('')));

}
});

cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$0 = (function (){
return cljs.core.async.promise_chan.call(null,null);
});

cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$1 = (function (xform){
return cljs.core.async.promise_chan.call(null,xform,null);
});

cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$2 = (function (xform,ex_handler){
return cljs.core.async.chan.call(null,cljs.core.async.impl.buffers.promise_buffer.call(null),xform,ex_handler);
});

cljs.core.async.promise_chan.cljs$lang$maxFixedArity = 2;
/**
 * Returns a channel that will close after msecs
 */
cljs.core.async.timeout = (function cljs$core$async$timeout(msecs){
return cljs.core.async.impl.timers.timeout.call(null,msecs);
});
/**
 * takes a val from port. Must be called inside a (go ...) block. Will
 *   return nil if closed. Will park if nothing is available.
 *   Returns true unless port is already closed
 */
cljs.core.async._LT__BANG_ = (function cljs$core$async$_LT__BANG_(port){
throw (new Error("<! used not in (go ...) block"));
});
/**
 * Asynchronously takes a val from port, passing to fn1. Will pass nil
 * if closed. If on-caller? (default true) is true, and value is
 * immediately available, will call fn1 on calling thread.
 * Returns nil.
 */
cljs.core.async.take_BANG_ = (function cljs$core$async$take_BANG_(var_args){
var args19004 = [];
var len__17824__auto___19007 = arguments.length;
var i__17825__auto___19008 = (0);
while(true){
if((i__17825__auto___19008 < len__17824__auto___19007)){
args19004.push((arguments[i__17825__auto___19008]));

var G__19009 = (i__17825__auto___19008 + (1));
i__17825__auto___19008 = G__19009;
continue;
} else {
}
break;
}

var G__19006 = args19004.length;
switch (G__19006) {
case 2:
return cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19004.length)].join('')));

}
});

cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (port,fn1){
return cljs.core.async.take_BANG_.call(null,port,fn1,true);
});

cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (port,fn1,on_caller_QMARK_){
var ret = cljs.core.async.impl.protocols.take_BANG_.call(null,port,cljs.core.async.fn_handler.call(null,fn1));
if(cljs.core.truth_(ret)){
var val_19011 = cljs.core.deref.call(null,ret);
if(cljs.core.truth_(on_caller_QMARK_)){
fn1.call(null,val_19011);
} else {
cljs.core.async.impl.dispatch.run.call(null,((function (val_19011,ret){
return (function (){
return fn1.call(null,val_19011);
});})(val_19011,ret))
);
}
} else {
}

return null;
});

cljs.core.async.take_BANG_.cljs$lang$maxFixedArity = 3;
cljs.core.async.nop = (function cljs$core$async$nop(_){
return null;
});
cljs.core.async.fhnop = cljs.core.async.fn_handler.call(null,cljs.core.async.nop);
/**
 * puts a val into port. nil values are not allowed. Must be called
 *   inside a (go ...) block. Will park if no buffer space is available.
 *   Returns true unless port is already closed.
 */
cljs.core.async._GT__BANG_ = (function cljs$core$async$_GT__BANG_(port,val){
throw (new Error(">! used not in (go ...) block"));
});
/**
 * Asynchronously puts a val into port, calling fn0 (if supplied) when
 * complete. nil values are not allowed. Will throw if closed. If
 * on-caller? (default true) is true, and the put is immediately
 * accepted, will call fn0 on calling thread.  Returns nil.
 */
cljs.core.async.put_BANG_ = (function cljs$core$async$put_BANG_(var_args){
var args19012 = [];
var len__17824__auto___19015 = arguments.length;
var i__17825__auto___19016 = (0);
while(true){
if((i__17825__auto___19016 < len__17824__auto___19015)){
args19012.push((arguments[i__17825__auto___19016]));

var G__19017 = (i__17825__auto___19016 + (1));
i__17825__auto___19016 = G__19017;
continue;
} else {
}
break;
}

var G__19014 = args19012.length;
switch (G__19014) {
case 2:
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19012.length)].join('')));

}
});

cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (port,val){
var temp__4423__auto__ = cljs.core.async.impl.protocols.put_BANG_.call(null,port,val,cljs.core.async.fhnop);
if(cljs.core.truth_(temp__4423__auto__)){
var ret = temp__4423__auto__;
return cljs.core.deref.call(null,ret);
} else {
return true;
}
});

cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (port,val,fn1){
return cljs.core.async.put_BANG_.call(null,port,val,fn1,true);
});

cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$4 = (function (port,val,fn1,on_caller_QMARK_){
var temp__4423__auto__ = cljs.core.async.impl.protocols.put_BANG_.call(null,port,val,cljs.core.async.fn_handler.call(null,fn1));
if(cljs.core.truth_(temp__4423__auto__)){
var retb = temp__4423__auto__;
var ret = cljs.core.deref.call(null,retb);
if(cljs.core.truth_(on_caller_QMARK_)){
fn1.call(null,ret);
} else {
cljs.core.async.impl.dispatch.run.call(null,((function (ret,retb,temp__4423__auto__){
return (function (){
return fn1.call(null,ret);
});})(ret,retb,temp__4423__auto__))
);
}

return ret;
} else {
return true;
}
});

cljs.core.async.put_BANG_.cljs$lang$maxFixedArity = 4;
cljs.core.async.close_BANG_ = (function cljs$core$async$close_BANG_(port){
return cljs.core.async.impl.protocols.close_BANG_.call(null,port);
});
cljs.core.async.random_array = (function cljs$core$async$random_array(n){
var a = (new Array(n));
var n__17669__auto___19019 = n;
var x_19020 = (0);
while(true){
if((x_19020 < n__17669__auto___19019)){
(a[x_19020] = (0));

var G__19021 = (x_19020 + (1));
x_19020 = G__19021;
continue;
} else {
}
break;
}

var i = (1);
while(true){
if(cljs.core._EQ_.call(null,i,n)){
return a;
} else {
var j = cljs.core.rand_int.call(null,i);
(a[i] = (a[j]));

(a[j] = i);

var G__19022 = (i + (1));
i = G__19022;
continue;
}
break;
}
});
cljs.core.async.alt_flag = (function cljs$core$async$alt_flag(){
var flag = cljs.core.atom.call(null,true);
if(typeof cljs.core.async.t_cljs$core$async19026 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async19026 = (function (alt_flag,flag,meta19027){
this.alt_flag = alt_flag;
this.flag = flag;
this.meta19027 = meta19027;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (flag){
return (function (_19028,meta19027__$1){
var self__ = this;
var _19028__$1 = this;
return (new cljs.core.async.t_cljs$core$async19026(self__.alt_flag,self__.flag,meta19027__$1));
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (flag){
return (function (_19028){
var self__ = this;
var _19028__$1 = this;
return self__.meta19027;
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$async$impl$protocols$Handler$ = true;

cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = ((function (flag){
return (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.deref.call(null,self__.flag);
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = ((function (flag){
return (function (_){
var self__ = this;
var ___$1 = this;
return true;
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = ((function (flag){
return (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_.call(null,self__.flag,null);

return true;
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.getBasis = ((function (flag){
return (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"alt-flag","alt-flag",-1794972754,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"private","private",-558947994),true,new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(cljs.core.PersistentVector.EMPTY))], null)),new cljs.core.Symbol(null,"flag","flag",-1565787888,null),new cljs.core.Symbol(null,"meta19027","meta19027",-1888318779,null)], null);
});})(flag))
;

cljs.core.async.t_cljs$core$async19026.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async19026.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async19026";

cljs.core.async.t_cljs$core$async19026.cljs$lang$ctorPrWriter = ((function (flag){
return (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async19026");
});})(flag))
;

cljs.core.async.__GT_t_cljs$core$async19026 = ((function (flag){
return (function cljs$core$async$alt_flag_$___GT_t_cljs$core$async19026(alt_flag__$1,flag__$1,meta19027){
return (new cljs.core.async.t_cljs$core$async19026(alt_flag__$1,flag__$1,meta19027));
});})(flag))
;

}

return (new cljs.core.async.t_cljs$core$async19026(cljs$core$async$alt_flag,flag,cljs.core.PersistentArrayMap.EMPTY));
});
cljs.core.async.alt_handler = (function cljs$core$async$alt_handler(flag,cb){
if(typeof cljs.core.async.t_cljs$core$async19032 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async19032 = (function (alt_handler,flag,cb,meta19033){
this.alt_handler = alt_handler;
this.flag = flag;
this.cb = cb;
this.meta19033 = meta19033;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_19034,meta19033__$1){
var self__ = this;
var _19034__$1 = this;
return (new cljs.core.async.t_cljs$core$async19032(self__.alt_handler,self__.flag,self__.cb,meta19033__$1));
});

cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_19034){
var self__ = this;
var _19034__$1 = this;
return self__.meta19033;
});

cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$async$impl$protocols$Handler$ = true;

cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.active_QMARK_.call(null,self__.flag);
});

cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return true;
});

cljs.core.async.t_cljs$core$async19032.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.async.impl.protocols.commit.call(null,self__.flag);

return self__.cb;
});

cljs.core.async.t_cljs$core$async19032.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"alt-handler","alt-handler",963786170,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"private","private",-558947994),true,new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"flag","flag",-1565787888,null),new cljs.core.Symbol(null,"cb","cb",-2064487928,null)], null)))], null)),new cljs.core.Symbol(null,"flag","flag",-1565787888,null),new cljs.core.Symbol(null,"cb","cb",-2064487928,null),new cljs.core.Symbol(null,"meta19033","meta19033",1422629552,null)], null);
});

cljs.core.async.t_cljs$core$async19032.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async19032.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async19032";

cljs.core.async.t_cljs$core$async19032.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async19032");
});

cljs.core.async.__GT_t_cljs$core$async19032 = (function cljs$core$async$alt_handler_$___GT_t_cljs$core$async19032(alt_handler__$1,flag__$1,cb__$1,meta19033){
return (new cljs.core.async.t_cljs$core$async19032(alt_handler__$1,flag__$1,cb__$1,meta19033));
});

}

return (new cljs.core.async.t_cljs$core$async19032(cljs$core$async$alt_handler,flag,cb,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * returns derefable [val port] if immediate, nil if enqueued
 */
cljs.core.async.do_alts = (function cljs$core$async$do_alts(fret,ports,opts){
var flag = cljs.core.async.alt_flag.call(null);
var n = cljs.core.count.call(null,ports);
var idxs = cljs.core.async.random_array.call(null,n);
var priority = new cljs.core.Keyword(null,"priority","priority",1431093715).cljs$core$IFn$_invoke$arity$1(opts);
var ret = (function (){var i = (0);
while(true){
if((i < n)){
var idx = (cljs.core.truth_(priority)?i:(idxs[i]));
var port = cljs.core.nth.call(null,ports,idx);
var wport = ((cljs.core.vector_QMARK_.call(null,port))?port.call(null,(0)):null);
var vbox = (cljs.core.truth_(wport)?(function (){var val = port.call(null,(1));
return cljs.core.async.impl.protocols.put_BANG_.call(null,wport,val,cljs.core.async.alt_handler.call(null,flag,((function (i,val,idx,port,wport,flag,n,idxs,priority){
return (function (p1__19035_SHARP_){
return fret.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__19035_SHARP_,wport], null));
});})(i,val,idx,port,wport,flag,n,idxs,priority))
));
})():cljs.core.async.impl.protocols.take_BANG_.call(null,port,cljs.core.async.alt_handler.call(null,flag,((function (i,idx,port,wport,flag,n,idxs,priority){
return (function (p1__19036_SHARP_){
return fret.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__19036_SHARP_,port], null));
});})(i,idx,port,wport,flag,n,idxs,priority))
)));
if(cljs.core.truth_(vbox)){
return cljs.core.async.impl.channels.box.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.deref.call(null,vbox),(function (){var or__16766__auto__ = wport;
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return port;
}
})()], null));
} else {
var G__19037 = (i + (1));
i = G__19037;
continue;
}
} else {
return null;
}
break;
}
})();
var or__16766__auto__ = ret;
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
if(cljs.core.contains_QMARK_.call(null,opts,new cljs.core.Keyword(null,"default","default",-1987822328))){
var temp__4425__auto__ = (function (){var and__16754__auto__ = cljs.core.async.impl.protocols.active_QMARK_.call(null,flag);
if(cljs.core.truth_(and__16754__auto__)){
return cljs.core.async.impl.protocols.commit.call(null,flag);
} else {
return and__16754__auto__;
}
})();
if(cljs.core.truth_(temp__4425__auto__)){
var got = temp__4425__auto__;
return cljs.core.async.impl.channels.box.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"default","default",-1987822328).cljs$core$IFn$_invoke$arity$1(opts),new cljs.core.Keyword(null,"default","default",-1987822328)], null));
} else {
return null;
}
} else {
return null;
}
}
});
/**
 * Completes at most one of several channel operations. Must be called
 * inside a (go ...) block. ports is a vector of channel endpoints,
 * which can be either a channel to take from or a vector of
 *   [channel-to-put-to val-to-put], in any combination. Takes will be
 *   made as if by <!, and puts will be made as if by >!. Unless
 *   the :priority option is true, if more than one port operation is
 *   ready a non-deterministic choice will be made. If no operation is
 *   ready and a :default value is supplied, [default-val :default] will
 *   be returned, otherwise alts! will park until the first operation to
 *   become ready completes. Returns [val port] of the completed
 *   operation, where val is the value taken for takes, and a
 *   boolean (true unless already closed, as per put!) for puts.
 * 
 *   opts are passed as :key val ... Supported options:
 * 
 *   :default val - the value to use if none of the operations are immediately ready
 *   :priority true - (default nil) when true, the operations will be tried in order.
 * 
 *   Note: there is no guarantee that the port exps or val exprs will be
 *   used, nor in what order should they be, so they should not be
 *   depended upon for side effects.
 */
cljs.core.async.alts_BANG_ = (function cljs$core$async$alts_BANG_(var_args){
var args__17831__auto__ = [];
var len__17824__auto___19043 = arguments.length;
var i__17825__auto___19044 = (0);
while(true){
if((i__17825__auto___19044 < len__17824__auto___19043)){
args__17831__auto__.push((arguments[i__17825__auto___19044]));

var G__19045 = (i__17825__auto___19044 + (1));
i__17825__auto___19044 = G__19045;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((1) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((1)),(0))):null);
return cljs.core.async.alts_BANG_.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__17832__auto__);
});

cljs.core.async.alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = (function (ports,p__19040){
var map__19041 = p__19040;
var map__19041__$1 = ((((!((map__19041 == null)))?((((map__19041.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19041.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19041):map__19041);
var opts = map__19041__$1;
throw (new Error("alts! used not in (go ...) block"));
});

cljs.core.async.alts_BANG_.cljs$lang$maxFixedArity = (1);

cljs.core.async.alts_BANG_.cljs$lang$applyTo = (function (seq19038){
var G__19039 = cljs.core.first.call(null,seq19038);
var seq19038__$1 = cljs.core.next.call(null,seq19038);
return cljs.core.async.alts_BANG_.cljs$core$IFn$_invoke$arity$variadic(G__19039,seq19038__$1);
});
/**
 * Puts a val into port if it's possible to do so immediately.
 *   nil values are not allowed. Never blocks. Returns true if offer succeeds.
 */
cljs.core.async.offer_BANG_ = (function cljs$core$async$offer_BANG_(port,val){
var ret = cljs.core.async.impl.protocols.put_BANG_.call(null,port,val,cljs.core.async.fn_handler.call(null,cljs.core.async.nop,false));
if(cljs.core.truth_(ret)){
return cljs.core.deref.call(null,ret);
} else {
return null;
}
});
/**
 * Takes a val from port if it's possible to do so immediately.
 *   Never blocks. Returns value if successful, nil otherwise.
 */
cljs.core.async.poll_BANG_ = (function cljs$core$async$poll_BANG_(port){
var ret = cljs.core.async.impl.protocols.take_BANG_.call(null,port,cljs.core.async.fn_handler.call(null,cljs.core.async.nop,false));
if(cljs.core.truth_(ret)){
return cljs.core.deref.call(null,ret);
} else {
return null;
}
});
/**
 * Takes elements from the from channel and supplies them to the to
 * channel. By default, the to channel will be closed when the from
 * channel closes, but can be determined by the close?  parameter. Will
 * stop consuming the from channel if the to channel closes
 */
cljs.core.async.pipe = (function cljs$core$async$pipe(var_args){
var args19046 = [];
var len__17824__auto___19096 = arguments.length;
var i__17825__auto___19097 = (0);
while(true){
if((i__17825__auto___19097 < len__17824__auto___19096)){
args19046.push((arguments[i__17825__auto___19097]));

var G__19098 = (i__17825__auto___19097 + (1));
i__17825__auto___19097 = G__19098;
continue;
} else {
}
break;
}

var G__19048 = args19046.length;
switch (G__19048) {
case 2:
return cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19046.length)].join('')));

}
});

cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$2 = (function (from,to){
return cljs.core.async.pipe.call(null,from,to,true);
});

cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$3 = (function (from,to,close_QMARK_){
var c__18933__auto___19100 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___19100){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___19100){
return (function (state_19072){
var state_val_19073 = (state_19072[(1)]);
if((state_val_19073 === (7))){
var inst_19068 = (state_19072[(2)]);
var state_19072__$1 = state_19072;
var statearr_19074_19101 = state_19072__$1;
(statearr_19074_19101[(2)] = inst_19068);

(statearr_19074_19101[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (1))){
var state_19072__$1 = state_19072;
var statearr_19075_19102 = state_19072__$1;
(statearr_19075_19102[(2)] = null);

(statearr_19075_19102[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (4))){
var inst_19051 = (state_19072[(7)]);
var inst_19051__$1 = (state_19072[(2)]);
var inst_19052 = (inst_19051__$1 == null);
var state_19072__$1 = (function (){var statearr_19076 = state_19072;
(statearr_19076[(7)] = inst_19051__$1);

return statearr_19076;
})();
if(cljs.core.truth_(inst_19052)){
var statearr_19077_19103 = state_19072__$1;
(statearr_19077_19103[(1)] = (5));

} else {
var statearr_19078_19104 = state_19072__$1;
(statearr_19078_19104[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (13))){
var state_19072__$1 = state_19072;
var statearr_19079_19105 = state_19072__$1;
(statearr_19079_19105[(2)] = null);

(statearr_19079_19105[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (6))){
var inst_19051 = (state_19072[(7)]);
var state_19072__$1 = state_19072;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19072__$1,(11),to,inst_19051);
} else {
if((state_val_19073 === (3))){
var inst_19070 = (state_19072[(2)]);
var state_19072__$1 = state_19072;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19072__$1,inst_19070);
} else {
if((state_val_19073 === (12))){
var state_19072__$1 = state_19072;
var statearr_19080_19106 = state_19072__$1;
(statearr_19080_19106[(2)] = null);

(statearr_19080_19106[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (2))){
var state_19072__$1 = state_19072;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19072__$1,(4),from);
} else {
if((state_val_19073 === (11))){
var inst_19061 = (state_19072[(2)]);
var state_19072__$1 = state_19072;
if(cljs.core.truth_(inst_19061)){
var statearr_19081_19107 = state_19072__$1;
(statearr_19081_19107[(1)] = (12));

} else {
var statearr_19082_19108 = state_19072__$1;
(statearr_19082_19108[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (9))){
var state_19072__$1 = state_19072;
var statearr_19083_19109 = state_19072__$1;
(statearr_19083_19109[(2)] = null);

(statearr_19083_19109[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (5))){
var state_19072__$1 = state_19072;
if(cljs.core.truth_(close_QMARK_)){
var statearr_19084_19110 = state_19072__$1;
(statearr_19084_19110[(1)] = (8));

} else {
var statearr_19085_19111 = state_19072__$1;
(statearr_19085_19111[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (14))){
var inst_19066 = (state_19072[(2)]);
var state_19072__$1 = state_19072;
var statearr_19086_19112 = state_19072__$1;
(statearr_19086_19112[(2)] = inst_19066);

(statearr_19086_19112[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (10))){
var inst_19058 = (state_19072[(2)]);
var state_19072__$1 = state_19072;
var statearr_19087_19113 = state_19072__$1;
(statearr_19087_19113[(2)] = inst_19058);

(statearr_19087_19113[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19073 === (8))){
var inst_19055 = cljs.core.async.close_BANG_.call(null,to);
var state_19072__$1 = state_19072;
var statearr_19088_19114 = state_19072__$1;
(statearr_19088_19114[(2)] = inst_19055);

(statearr_19088_19114[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___19100))
;
return ((function (switch__18821__auto__,c__18933__auto___19100){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_19092 = [null,null,null,null,null,null,null,null];
(statearr_19092[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_19092[(1)] = (1));

return statearr_19092;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_19072){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19072);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19093){if((e19093 instanceof Object)){
var ex__18825__auto__ = e19093;
var statearr_19094_19115 = state_19072;
(statearr_19094_19115[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19072);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19093;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19116 = state_19072;
state_19072 = G__19116;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_19072){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_19072);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___19100))
})();
var state__18935__auto__ = (function (){var statearr_19095 = f__18934__auto__.call(null);
(statearr_19095[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19100);

return statearr_19095;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___19100))
);


return to;
});

cljs.core.async.pipe.cljs$lang$maxFixedArity = 3;
cljs.core.async.pipeline_STAR_ = (function cljs$core$async$pipeline_STAR_(n,to,xf,from,close_QMARK_,ex_handler,type){
if((n > (0))){
} else {
throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str(cljs.core.pr_str.call(null,cljs.core.list(new cljs.core.Symbol(null,"pos?","pos?",-244377722,null),new cljs.core.Symbol(null,"n","n",-2092305744,null))))].join('')));
}

var jobs = cljs.core.async.chan.call(null,n);
var results = cljs.core.async.chan.call(null,n);
var process = ((function (jobs,results){
return (function (p__19300){
var vec__19301 = p__19300;
var v = cljs.core.nth.call(null,vec__19301,(0),null);
var p = cljs.core.nth.call(null,vec__19301,(1),null);
var job = vec__19301;
if((job == null)){
cljs.core.async.close_BANG_.call(null,results);

return null;
} else {
var res = cljs.core.async.chan.call(null,(1),xf,ex_handler);
var c__18933__auto___19483 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results){
return (function (state_19306){
var state_val_19307 = (state_19306[(1)]);
if((state_val_19307 === (1))){
var state_19306__$1 = state_19306;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19306__$1,(2),res,v);
} else {
if((state_val_19307 === (2))){
var inst_19303 = (state_19306[(2)]);
var inst_19304 = cljs.core.async.close_BANG_.call(null,res);
var state_19306__$1 = (function (){var statearr_19308 = state_19306;
(statearr_19308[(7)] = inst_19303);

return statearr_19308;
})();
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19306__$1,inst_19304);
} else {
return null;
}
}
});})(c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results))
;
return ((function (switch__18821__auto__,c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_19312 = [null,null,null,null,null,null,null,null];
(statearr_19312[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__);

(statearr_19312[(1)] = (1));

return statearr_19312;
});
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1 = (function (state_19306){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19306);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19313){if((e19313 instanceof Object)){
var ex__18825__auto__ = e19313;
var statearr_19314_19484 = state_19306;
(statearr_19314_19484[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19306);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19313;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19485 = state_19306;
state_19306 = G__19485;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = function(state_19306){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1.call(this,state_19306);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results))
})();
var state__18935__auto__ = (function (){var statearr_19315 = f__18934__auto__.call(null);
(statearr_19315[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19483);

return statearr_19315;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___19483,res,vec__19301,v,p,job,jobs,results))
);


cljs.core.async.put_BANG_.call(null,p,res);

return true;
}
});})(jobs,results))
;
var async = ((function (jobs,results,process){
return (function (p__19316){
var vec__19317 = p__19316;
var v = cljs.core.nth.call(null,vec__19317,(0),null);
var p = cljs.core.nth.call(null,vec__19317,(1),null);
var job = vec__19317;
if((job == null)){
cljs.core.async.close_BANG_.call(null,results);

return null;
} else {
var res = cljs.core.async.chan.call(null,(1));
xf.call(null,v,res);

cljs.core.async.put_BANG_.call(null,p,res);

return true;
}
});})(jobs,results,process))
;
var n__17669__auto___19486 = n;
var __19487 = (0);
while(true){
if((__19487 < n__17669__auto___19486)){
var G__19318_19488 = (((type instanceof cljs.core.Keyword))?type.fqn:null);
switch (G__19318_19488) {
case "compute":
var c__18933__auto___19490 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (__19487,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (__19487,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function (state_19331){
var state_val_19332 = (state_19331[(1)]);
if((state_val_19332 === (1))){
var state_19331__$1 = state_19331;
var statearr_19333_19491 = state_19331__$1;
(statearr_19333_19491[(2)] = null);

(statearr_19333_19491[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19332 === (2))){
var state_19331__$1 = state_19331;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19331__$1,(4),jobs);
} else {
if((state_val_19332 === (3))){
var inst_19329 = (state_19331[(2)]);
var state_19331__$1 = state_19331;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19331__$1,inst_19329);
} else {
if((state_val_19332 === (4))){
var inst_19321 = (state_19331[(2)]);
var inst_19322 = process.call(null,inst_19321);
var state_19331__$1 = state_19331;
if(cljs.core.truth_(inst_19322)){
var statearr_19334_19492 = state_19331__$1;
(statearr_19334_19492[(1)] = (5));

} else {
var statearr_19335_19493 = state_19331__$1;
(statearr_19335_19493[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19332 === (5))){
var state_19331__$1 = state_19331;
var statearr_19336_19494 = state_19331__$1;
(statearr_19336_19494[(2)] = null);

(statearr_19336_19494[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19332 === (6))){
var state_19331__$1 = state_19331;
var statearr_19337_19495 = state_19331__$1;
(statearr_19337_19495[(2)] = null);

(statearr_19337_19495[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19332 === (7))){
var inst_19327 = (state_19331[(2)]);
var state_19331__$1 = state_19331;
var statearr_19338_19496 = state_19331__$1;
(statearr_19338_19496[(2)] = inst_19327);

(statearr_19338_19496[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
});})(__19487,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
;
return ((function (__19487,switch__18821__auto__,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_19342 = [null,null,null,null,null,null,null];
(statearr_19342[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__);

(statearr_19342[(1)] = (1));

return statearr_19342;
});
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1 = (function (state_19331){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19331);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19343){if((e19343 instanceof Object)){
var ex__18825__auto__ = e19343;
var statearr_19344_19497 = state_19331;
(statearr_19344_19497[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19331);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19343;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19498 = state_19331;
state_19331 = G__19498;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = function(state_19331){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1.call(this,state_19331);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__;
})()
;})(__19487,switch__18821__auto__,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
})();
var state__18935__auto__ = (function (){var statearr_19345 = f__18934__auto__.call(null);
(statearr_19345[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19490);

return statearr_19345;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(__19487,c__18933__auto___19490,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
);


break;
case "async":
var c__18933__auto___19499 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (__19487,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (__19487,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function (state_19358){
var state_val_19359 = (state_19358[(1)]);
if((state_val_19359 === (1))){
var state_19358__$1 = state_19358;
var statearr_19360_19500 = state_19358__$1;
(statearr_19360_19500[(2)] = null);

(statearr_19360_19500[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19359 === (2))){
var state_19358__$1 = state_19358;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19358__$1,(4),jobs);
} else {
if((state_val_19359 === (3))){
var inst_19356 = (state_19358[(2)]);
var state_19358__$1 = state_19358;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19358__$1,inst_19356);
} else {
if((state_val_19359 === (4))){
var inst_19348 = (state_19358[(2)]);
var inst_19349 = async.call(null,inst_19348);
var state_19358__$1 = state_19358;
if(cljs.core.truth_(inst_19349)){
var statearr_19361_19501 = state_19358__$1;
(statearr_19361_19501[(1)] = (5));

} else {
var statearr_19362_19502 = state_19358__$1;
(statearr_19362_19502[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19359 === (5))){
var state_19358__$1 = state_19358;
var statearr_19363_19503 = state_19358__$1;
(statearr_19363_19503[(2)] = null);

(statearr_19363_19503[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19359 === (6))){
var state_19358__$1 = state_19358;
var statearr_19364_19504 = state_19358__$1;
(statearr_19364_19504[(2)] = null);

(statearr_19364_19504[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19359 === (7))){
var inst_19354 = (state_19358[(2)]);
var state_19358__$1 = state_19358;
var statearr_19365_19505 = state_19358__$1;
(statearr_19365_19505[(2)] = inst_19354);

(statearr_19365_19505[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
});})(__19487,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
;
return ((function (__19487,switch__18821__auto__,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_19369 = [null,null,null,null,null,null,null];
(statearr_19369[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__);

(statearr_19369[(1)] = (1));

return statearr_19369;
});
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1 = (function (state_19358){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19358);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19370){if((e19370 instanceof Object)){
var ex__18825__auto__ = e19370;
var statearr_19371_19506 = state_19358;
(statearr_19371_19506[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19358);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19370;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19507 = state_19358;
state_19358 = G__19507;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = function(state_19358){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1.call(this,state_19358);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__;
})()
;})(__19487,switch__18821__auto__,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
})();
var state__18935__auto__ = (function (){var statearr_19372 = f__18934__auto__.call(null);
(statearr_19372[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19499);

return statearr_19372;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(__19487,c__18933__auto___19499,G__19318_19488,n__17669__auto___19486,jobs,results,process,async))
);


break;
default:
throw (new Error([cljs.core.str("No matching clause: "),cljs.core.str(type)].join('')));

}

var G__19508 = (__19487 + (1));
__19487 = G__19508;
continue;
} else {
}
break;
}

var c__18933__auto___19509 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___19509,jobs,results,process,async){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___19509,jobs,results,process,async){
return (function (state_19394){
var state_val_19395 = (state_19394[(1)]);
if((state_val_19395 === (1))){
var state_19394__$1 = state_19394;
var statearr_19396_19510 = state_19394__$1;
(statearr_19396_19510[(2)] = null);

(statearr_19396_19510[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19395 === (2))){
var state_19394__$1 = state_19394;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19394__$1,(4),from);
} else {
if((state_val_19395 === (3))){
var inst_19392 = (state_19394[(2)]);
var state_19394__$1 = state_19394;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19394__$1,inst_19392);
} else {
if((state_val_19395 === (4))){
var inst_19375 = (state_19394[(7)]);
var inst_19375__$1 = (state_19394[(2)]);
var inst_19376 = (inst_19375__$1 == null);
var state_19394__$1 = (function (){var statearr_19397 = state_19394;
(statearr_19397[(7)] = inst_19375__$1);

return statearr_19397;
})();
if(cljs.core.truth_(inst_19376)){
var statearr_19398_19511 = state_19394__$1;
(statearr_19398_19511[(1)] = (5));

} else {
var statearr_19399_19512 = state_19394__$1;
(statearr_19399_19512[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19395 === (5))){
var inst_19378 = cljs.core.async.close_BANG_.call(null,jobs);
var state_19394__$1 = state_19394;
var statearr_19400_19513 = state_19394__$1;
(statearr_19400_19513[(2)] = inst_19378);

(statearr_19400_19513[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19395 === (6))){
var inst_19380 = (state_19394[(8)]);
var inst_19375 = (state_19394[(7)]);
var inst_19380__$1 = cljs.core.async.chan.call(null,(1));
var inst_19381 = cljs.core.PersistentVector.EMPTY_NODE;
var inst_19382 = [inst_19375,inst_19380__$1];
var inst_19383 = (new cljs.core.PersistentVector(null,2,(5),inst_19381,inst_19382,null));
var state_19394__$1 = (function (){var statearr_19401 = state_19394;
(statearr_19401[(8)] = inst_19380__$1);

return statearr_19401;
})();
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19394__$1,(8),jobs,inst_19383);
} else {
if((state_val_19395 === (7))){
var inst_19390 = (state_19394[(2)]);
var state_19394__$1 = state_19394;
var statearr_19402_19514 = state_19394__$1;
(statearr_19402_19514[(2)] = inst_19390);

(statearr_19402_19514[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19395 === (8))){
var inst_19380 = (state_19394[(8)]);
var inst_19385 = (state_19394[(2)]);
var state_19394__$1 = (function (){var statearr_19403 = state_19394;
(statearr_19403[(9)] = inst_19385);

return statearr_19403;
})();
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19394__$1,(9),results,inst_19380);
} else {
if((state_val_19395 === (9))){
var inst_19387 = (state_19394[(2)]);
var state_19394__$1 = (function (){var statearr_19404 = state_19394;
(statearr_19404[(10)] = inst_19387);

return statearr_19404;
})();
var statearr_19405_19515 = state_19394__$1;
(statearr_19405_19515[(2)] = null);

(statearr_19405_19515[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___19509,jobs,results,process,async))
;
return ((function (switch__18821__auto__,c__18933__auto___19509,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_19409 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_19409[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__);

(statearr_19409[(1)] = (1));

return statearr_19409;
});
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1 = (function (state_19394){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19394);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19410){if((e19410 instanceof Object)){
var ex__18825__auto__ = e19410;
var statearr_19411_19516 = state_19394;
(statearr_19411_19516[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19394);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19410;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19517 = state_19394;
state_19394 = G__19517;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = function(state_19394){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1.call(this,state_19394);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___19509,jobs,results,process,async))
})();
var state__18935__auto__ = (function (){var statearr_19412 = f__18934__auto__.call(null);
(statearr_19412[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19509);

return statearr_19412;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___19509,jobs,results,process,async))
);


var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__,jobs,results,process,async){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__,jobs,results,process,async){
return (function (state_19450){
var state_val_19451 = (state_19450[(1)]);
if((state_val_19451 === (7))){
var inst_19446 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
var statearr_19452_19518 = state_19450__$1;
(statearr_19452_19518[(2)] = inst_19446);

(statearr_19452_19518[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (20))){
var state_19450__$1 = state_19450;
var statearr_19453_19519 = state_19450__$1;
(statearr_19453_19519[(2)] = null);

(statearr_19453_19519[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (1))){
var state_19450__$1 = state_19450;
var statearr_19454_19520 = state_19450__$1;
(statearr_19454_19520[(2)] = null);

(statearr_19454_19520[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (4))){
var inst_19415 = (state_19450[(7)]);
var inst_19415__$1 = (state_19450[(2)]);
var inst_19416 = (inst_19415__$1 == null);
var state_19450__$1 = (function (){var statearr_19455 = state_19450;
(statearr_19455[(7)] = inst_19415__$1);

return statearr_19455;
})();
if(cljs.core.truth_(inst_19416)){
var statearr_19456_19521 = state_19450__$1;
(statearr_19456_19521[(1)] = (5));

} else {
var statearr_19457_19522 = state_19450__$1;
(statearr_19457_19522[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (15))){
var inst_19428 = (state_19450[(8)]);
var state_19450__$1 = state_19450;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19450__$1,(18),to,inst_19428);
} else {
if((state_val_19451 === (21))){
var inst_19441 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
var statearr_19458_19523 = state_19450__$1;
(statearr_19458_19523[(2)] = inst_19441);

(statearr_19458_19523[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (13))){
var inst_19443 = (state_19450[(2)]);
var state_19450__$1 = (function (){var statearr_19459 = state_19450;
(statearr_19459[(9)] = inst_19443);

return statearr_19459;
})();
var statearr_19460_19524 = state_19450__$1;
(statearr_19460_19524[(2)] = null);

(statearr_19460_19524[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (6))){
var inst_19415 = (state_19450[(7)]);
var state_19450__$1 = state_19450;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19450__$1,(11),inst_19415);
} else {
if((state_val_19451 === (17))){
var inst_19436 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
if(cljs.core.truth_(inst_19436)){
var statearr_19461_19525 = state_19450__$1;
(statearr_19461_19525[(1)] = (19));

} else {
var statearr_19462_19526 = state_19450__$1;
(statearr_19462_19526[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (3))){
var inst_19448 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19450__$1,inst_19448);
} else {
if((state_val_19451 === (12))){
var inst_19425 = (state_19450[(10)]);
var state_19450__$1 = state_19450;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19450__$1,(14),inst_19425);
} else {
if((state_val_19451 === (2))){
var state_19450__$1 = state_19450;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19450__$1,(4),results);
} else {
if((state_val_19451 === (19))){
var state_19450__$1 = state_19450;
var statearr_19463_19527 = state_19450__$1;
(statearr_19463_19527[(2)] = null);

(statearr_19463_19527[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (11))){
var inst_19425 = (state_19450[(2)]);
var state_19450__$1 = (function (){var statearr_19464 = state_19450;
(statearr_19464[(10)] = inst_19425);

return statearr_19464;
})();
var statearr_19465_19528 = state_19450__$1;
(statearr_19465_19528[(2)] = null);

(statearr_19465_19528[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (9))){
var state_19450__$1 = state_19450;
var statearr_19466_19529 = state_19450__$1;
(statearr_19466_19529[(2)] = null);

(statearr_19466_19529[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (5))){
var state_19450__$1 = state_19450;
if(cljs.core.truth_(close_QMARK_)){
var statearr_19467_19530 = state_19450__$1;
(statearr_19467_19530[(1)] = (8));

} else {
var statearr_19468_19531 = state_19450__$1;
(statearr_19468_19531[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (14))){
var inst_19430 = (state_19450[(11)]);
var inst_19428 = (state_19450[(8)]);
var inst_19428__$1 = (state_19450[(2)]);
var inst_19429 = (inst_19428__$1 == null);
var inst_19430__$1 = cljs.core.not.call(null,inst_19429);
var state_19450__$1 = (function (){var statearr_19469 = state_19450;
(statearr_19469[(11)] = inst_19430__$1);

(statearr_19469[(8)] = inst_19428__$1);

return statearr_19469;
})();
if(inst_19430__$1){
var statearr_19470_19532 = state_19450__$1;
(statearr_19470_19532[(1)] = (15));

} else {
var statearr_19471_19533 = state_19450__$1;
(statearr_19471_19533[(1)] = (16));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (16))){
var inst_19430 = (state_19450[(11)]);
var state_19450__$1 = state_19450;
var statearr_19472_19534 = state_19450__$1;
(statearr_19472_19534[(2)] = inst_19430);

(statearr_19472_19534[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (10))){
var inst_19422 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
var statearr_19473_19535 = state_19450__$1;
(statearr_19473_19535[(2)] = inst_19422);

(statearr_19473_19535[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (18))){
var inst_19433 = (state_19450[(2)]);
var state_19450__$1 = state_19450;
var statearr_19474_19536 = state_19450__$1;
(statearr_19474_19536[(2)] = inst_19433);

(statearr_19474_19536[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19451 === (8))){
var inst_19419 = cljs.core.async.close_BANG_.call(null,to);
var state_19450__$1 = state_19450;
var statearr_19475_19537 = state_19450__$1;
(statearr_19475_19537[(2)] = inst_19419);

(statearr_19475_19537[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto__,jobs,results,process,async))
;
return ((function (switch__18821__auto__,c__18933__auto__,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_19479 = [null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_19479[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__);

(statearr_19479[(1)] = (1));

return statearr_19479;
});
var cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1 = (function (state_19450){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19450);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19480){if((e19480 instanceof Object)){
var ex__18825__auto__ = e19480;
var statearr_19481_19538 = state_19450;
(statearr_19481_19538[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19450);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19480;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19539 = state_19450;
state_19450 = G__19539;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__ = function(state_19450){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1.call(this,state_19450);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__,jobs,results,process,async))
})();
var state__18935__auto__ = (function (){var statearr_19482 = f__18934__auto__.call(null);
(statearr_19482[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_19482;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__,jobs,results,process,async))
);

return c__18933__auto__;
});
/**
 * Takes elements from the from channel and supplies them to the to
 *   channel, subject to the async function af, with parallelism n. af
 *   must be a function of two arguments, the first an input value and
 *   the second a channel on which to place the result(s). af must close!
 *   the channel before returning.  The presumption is that af will
 *   return immediately, having launched some asynchronous operation
 *   whose completion/callback will manipulate the result channel. Outputs
 *   will be returned in order relative to  the inputs. By default, the to
 *   channel will be closed when the from channel closes, but can be
 *   determined by the close?  parameter. Will stop consuming the from
 *   channel if the to channel closes.
 */
cljs.core.async.pipeline_async = (function cljs$core$async$pipeline_async(var_args){
var args19540 = [];
var len__17824__auto___19543 = arguments.length;
var i__17825__auto___19544 = (0);
while(true){
if((i__17825__auto___19544 < len__17824__auto___19543)){
args19540.push((arguments[i__17825__auto___19544]));

var G__19545 = (i__17825__auto___19544 + (1));
i__17825__auto___19544 = G__19545;
continue;
} else {
}
break;
}

var G__19542 = args19540.length;
switch (G__19542) {
case 4:
return cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
case 5:
return cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$5((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19540.length)].join('')));

}
});

cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$4 = (function (n,to,af,from){
return cljs.core.async.pipeline_async.call(null,n,to,af,from,true);
});

cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$5 = (function (n,to,af,from,close_QMARK_){
return cljs.core.async.pipeline_STAR_.call(null,n,to,af,from,close_QMARK_,null,new cljs.core.Keyword(null,"async","async",1050769601));
});

cljs.core.async.pipeline_async.cljs$lang$maxFixedArity = 5;
/**
 * Takes elements from the from channel and supplies them to the to
 *   channel, subject to the transducer xf, with parallelism n. Because
 *   it is parallel, the transducer will be applied independently to each
 *   element, not across elements, and may produce zero or more outputs
 *   per input.  Outputs will be returned in order relative to the
 *   inputs. By default, the to channel will be closed when the from
 *   channel closes, but can be determined by the close?  parameter. Will
 *   stop consuming the from channel if the to channel closes.
 * 
 *   Note this is supplied for API compatibility with the Clojure version.
 *   Values of N > 1 will not result in actual concurrency in a
 *   single-threaded runtime.
 */
cljs.core.async.pipeline = (function cljs$core$async$pipeline(var_args){
var args19547 = [];
var len__17824__auto___19550 = arguments.length;
var i__17825__auto___19551 = (0);
while(true){
if((i__17825__auto___19551 < len__17824__auto___19550)){
args19547.push((arguments[i__17825__auto___19551]));

var G__19552 = (i__17825__auto___19551 + (1));
i__17825__auto___19551 = G__19552;
continue;
} else {
}
break;
}

var G__19549 = args19547.length;
switch (G__19549) {
case 4:
return cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
case 5:
return cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$5((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]));

break;
case 6:
return cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$6((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]),(arguments[(5)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19547.length)].join('')));

}
});

cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$4 = (function (n,to,xf,from){
return cljs.core.async.pipeline.call(null,n,to,xf,from,true);
});

cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$5 = (function (n,to,xf,from,close_QMARK_){
return cljs.core.async.pipeline.call(null,n,to,xf,from,close_QMARK_,null);
});

cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$6 = (function (n,to,xf,from,close_QMARK_,ex_handler){
return cljs.core.async.pipeline_STAR_.call(null,n,to,xf,from,close_QMARK_,ex_handler,new cljs.core.Keyword(null,"compute","compute",1555393130));
});

cljs.core.async.pipeline.cljs$lang$maxFixedArity = 6;
/**
 * Takes a predicate and a source channel and returns a vector of two
 *   channels, the first of which will contain the values for which the
 *   predicate returned true, the second those for which it returned
 *   false.
 * 
 *   The out channels will be unbuffered by default, or two buf-or-ns can
 *   be supplied. The channels will close after the source channel has
 *   closed.
 */
cljs.core.async.split = (function cljs$core$async$split(var_args){
var args19554 = [];
var len__17824__auto___19607 = arguments.length;
var i__17825__auto___19608 = (0);
while(true){
if((i__17825__auto___19608 < len__17824__auto___19607)){
args19554.push((arguments[i__17825__auto___19608]));

var G__19609 = (i__17825__auto___19608 + (1));
i__17825__auto___19608 = G__19609;
continue;
} else {
}
break;
}

var G__19556 = args19554.length;
switch (G__19556) {
case 2:
return cljs.core.async.split.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 4:
return cljs.core.async.split.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19554.length)].join('')));

}
});

cljs.core.async.split.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.split.call(null,p,ch,null,null);
});

cljs.core.async.split.cljs$core$IFn$_invoke$arity$4 = (function (p,ch,t_buf_or_n,f_buf_or_n){
var tc = cljs.core.async.chan.call(null,t_buf_or_n);
var fc = cljs.core.async.chan.call(null,f_buf_or_n);
var c__18933__auto___19611 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___19611,tc,fc){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___19611,tc,fc){
return (function (state_19582){
var state_val_19583 = (state_19582[(1)]);
if((state_val_19583 === (7))){
var inst_19578 = (state_19582[(2)]);
var state_19582__$1 = state_19582;
var statearr_19584_19612 = state_19582__$1;
(statearr_19584_19612[(2)] = inst_19578);

(statearr_19584_19612[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (1))){
var state_19582__$1 = state_19582;
var statearr_19585_19613 = state_19582__$1;
(statearr_19585_19613[(2)] = null);

(statearr_19585_19613[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (4))){
var inst_19559 = (state_19582[(7)]);
var inst_19559__$1 = (state_19582[(2)]);
var inst_19560 = (inst_19559__$1 == null);
var state_19582__$1 = (function (){var statearr_19586 = state_19582;
(statearr_19586[(7)] = inst_19559__$1);

return statearr_19586;
})();
if(cljs.core.truth_(inst_19560)){
var statearr_19587_19614 = state_19582__$1;
(statearr_19587_19614[(1)] = (5));

} else {
var statearr_19588_19615 = state_19582__$1;
(statearr_19588_19615[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (13))){
var state_19582__$1 = state_19582;
var statearr_19589_19616 = state_19582__$1;
(statearr_19589_19616[(2)] = null);

(statearr_19589_19616[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (6))){
var inst_19559 = (state_19582[(7)]);
var inst_19565 = p.call(null,inst_19559);
var state_19582__$1 = state_19582;
if(cljs.core.truth_(inst_19565)){
var statearr_19590_19617 = state_19582__$1;
(statearr_19590_19617[(1)] = (9));

} else {
var statearr_19591_19618 = state_19582__$1;
(statearr_19591_19618[(1)] = (10));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (3))){
var inst_19580 = (state_19582[(2)]);
var state_19582__$1 = state_19582;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19582__$1,inst_19580);
} else {
if((state_val_19583 === (12))){
var state_19582__$1 = state_19582;
var statearr_19592_19619 = state_19582__$1;
(statearr_19592_19619[(2)] = null);

(statearr_19592_19619[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (2))){
var state_19582__$1 = state_19582;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19582__$1,(4),ch);
} else {
if((state_val_19583 === (11))){
var inst_19559 = (state_19582[(7)]);
var inst_19569 = (state_19582[(2)]);
var state_19582__$1 = state_19582;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19582__$1,(8),inst_19569,inst_19559);
} else {
if((state_val_19583 === (9))){
var state_19582__$1 = state_19582;
var statearr_19593_19620 = state_19582__$1;
(statearr_19593_19620[(2)] = tc);

(statearr_19593_19620[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (5))){
var inst_19562 = cljs.core.async.close_BANG_.call(null,tc);
var inst_19563 = cljs.core.async.close_BANG_.call(null,fc);
var state_19582__$1 = (function (){var statearr_19594 = state_19582;
(statearr_19594[(8)] = inst_19562);

return statearr_19594;
})();
var statearr_19595_19621 = state_19582__$1;
(statearr_19595_19621[(2)] = inst_19563);

(statearr_19595_19621[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (14))){
var inst_19576 = (state_19582[(2)]);
var state_19582__$1 = state_19582;
var statearr_19596_19622 = state_19582__$1;
(statearr_19596_19622[(2)] = inst_19576);

(statearr_19596_19622[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (10))){
var state_19582__$1 = state_19582;
var statearr_19597_19623 = state_19582__$1;
(statearr_19597_19623[(2)] = fc);

(statearr_19597_19623[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19583 === (8))){
var inst_19571 = (state_19582[(2)]);
var state_19582__$1 = state_19582;
if(cljs.core.truth_(inst_19571)){
var statearr_19598_19624 = state_19582__$1;
(statearr_19598_19624[(1)] = (12));

} else {
var statearr_19599_19625 = state_19582__$1;
(statearr_19599_19625[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___19611,tc,fc))
;
return ((function (switch__18821__auto__,c__18933__auto___19611,tc,fc){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_19603 = [null,null,null,null,null,null,null,null,null];
(statearr_19603[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_19603[(1)] = (1));

return statearr_19603;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_19582){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19582);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19604){if((e19604 instanceof Object)){
var ex__18825__auto__ = e19604;
var statearr_19605_19626 = state_19582;
(statearr_19605_19626[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19582);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19604;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19627 = state_19582;
state_19582 = G__19627;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_19582){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_19582);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___19611,tc,fc))
})();
var state__18935__auto__ = (function (){var statearr_19606 = f__18934__auto__.call(null);
(statearr_19606[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___19611);

return statearr_19606;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___19611,tc,fc))
);


return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [tc,fc], null);
});

cljs.core.async.split.cljs$lang$maxFixedArity = 4;
/**
 * f should be a function of 2 arguments. Returns a channel containing
 *   the single result of applying f to init and the first item from the
 *   channel, then applying f to that result and the 2nd item, etc. If
 *   the channel closes without yielding items, returns init and f is not
 *   called. ch must close before reduce produces a result.
 */
cljs.core.async.reduce = (function cljs$core$async$reduce(f,init,ch){
var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__){
return (function (state_19691){
var state_val_19692 = (state_19691[(1)]);
if((state_val_19692 === (7))){
var inst_19687 = (state_19691[(2)]);
var state_19691__$1 = state_19691;
var statearr_19693_19714 = state_19691__$1;
(statearr_19693_19714[(2)] = inst_19687);

(statearr_19693_19714[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (1))){
var inst_19671 = init;
var state_19691__$1 = (function (){var statearr_19694 = state_19691;
(statearr_19694[(7)] = inst_19671);

return statearr_19694;
})();
var statearr_19695_19715 = state_19691__$1;
(statearr_19695_19715[(2)] = null);

(statearr_19695_19715[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (4))){
var inst_19674 = (state_19691[(8)]);
var inst_19674__$1 = (state_19691[(2)]);
var inst_19675 = (inst_19674__$1 == null);
var state_19691__$1 = (function (){var statearr_19696 = state_19691;
(statearr_19696[(8)] = inst_19674__$1);

return statearr_19696;
})();
if(cljs.core.truth_(inst_19675)){
var statearr_19697_19716 = state_19691__$1;
(statearr_19697_19716[(1)] = (5));

} else {
var statearr_19698_19717 = state_19691__$1;
(statearr_19698_19717[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (6))){
var inst_19678 = (state_19691[(9)]);
var inst_19671 = (state_19691[(7)]);
var inst_19674 = (state_19691[(8)]);
var inst_19678__$1 = f.call(null,inst_19671,inst_19674);
var inst_19679 = cljs.core.reduced_QMARK_.call(null,inst_19678__$1);
var state_19691__$1 = (function (){var statearr_19699 = state_19691;
(statearr_19699[(9)] = inst_19678__$1);

return statearr_19699;
})();
if(inst_19679){
var statearr_19700_19718 = state_19691__$1;
(statearr_19700_19718[(1)] = (8));

} else {
var statearr_19701_19719 = state_19691__$1;
(statearr_19701_19719[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (3))){
var inst_19689 = (state_19691[(2)]);
var state_19691__$1 = state_19691;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19691__$1,inst_19689);
} else {
if((state_val_19692 === (2))){
var state_19691__$1 = state_19691;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19691__$1,(4),ch);
} else {
if((state_val_19692 === (9))){
var inst_19678 = (state_19691[(9)]);
var inst_19671 = inst_19678;
var state_19691__$1 = (function (){var statearr_19702 = state_19691;
(statearr_19702[(7)] = inst_19671);

return statearr_19702;
})();
var statearr_19703_19720 = state_19691__$1;
(statearr_19703_19720[(2)] = null);

(statearr_19703_19720[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (5))){
var inst_19671 = (state_19691[(7)]);
var state_19691__$1 = state_19691;
var statearr_19704_19721 = state_19691__$1;
(statearr_19704_19721[(2)] = inst_19671);

(statearr_19704_19721[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (10))){
var inst_19685 = (state_19691[(2)]);
var state_19691__$1 = state_19691;
var statearr_19705_19722 = state_19691__$1;
(statearr_19705_19722[(2)] = inst_19685);

(statearr_19705_19722[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19692 === (8))){
var inst_19678 = (state_19691[(9)]);
var inst_19681 = cljs.core.deref.call(null,inst_19678);
var state_19691__$1 = state_19691;
var statearr_19706_19723 = state_19691__$1;
(statearr_19706_19723[(2)] = inst_19681);

(statearr_19706_19723[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto__))
;
return ((function (switch__18821__auto__,c__18933__auto__){
return (function() {
var cljs$core$async$reduce_$_state_machine__18822__auto__ = null;
var cljs$core$async$reduce_$_state_machine__18822__auto____0 = (function (){
var statearr_19710 = [null,null,null,null,null,null,null,null,null,null];
(statearr_19710[(0)] = cljs$core$async$reduce_$_state_machine__18822__auto__);

(statearr_19710[(1)] = (1));

return statearr_19710;
});
var cljs$core$async$reduce_$_state_machine__18822__auto____1 = (function (state_19691){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19691);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19711){if((e19711 instanceof Object)){
var ex__18825__auto__ = e19711;
var statearr_19712_19724 = state_19691;
(statearr_19712_19724[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19691);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19711;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19725 = state_19691;
state_19691 = G__19725;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$reduce_$_state_machine__18822__auto__ = function(state_19691){
switch(arguments.length){
case 0:
return cljs$core$async$reduce_$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$reduce_$_state_machine__18822__auto____1.call(this,state_19691);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$reduce_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$reduce_$_state_machine__18822__auto____0;
cljs$core$async$reduce_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$reduce_$_state_machine__18822__auto____1;
return cljs$core$async$reduce_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__))
})();
var state__18935__auto__ = (function (){var statearr_19713 = f__18934__auto__.call(null);
(statearr_19713[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_19713;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__))
);

return c__18933__auto__;
});
/**
 * Puts the contents of coll into the supplied channel.
 * 
 *   By default the channel will be closed after the items are copied,
 *   but can be determined by the close? parameter.
 * 
 *   Returns a channel which will close after the items are copied.
 */
cljs.core.async.onto_chan = (function cljs$core$async$onto_chan(var_args){
var args19726 = [];
var len__17824__auto___19778 = arguments.length;
var i__17825__auto___19779 = (0);
while(true){
if((i__17825__auto___19779 < len__17824__auto___19778)){
args19726.push((arguments[i__17825__auto___19779]));

var G__19780 = (i__17825__auto___19779 + (1));
i__17825__auto___19779 = G__19780;
continue;
} else {
}
break;
}

var G__19728 = args19726.length;
switch (G__19728) {
case 2:
return cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19726.length)].join('')));

}
});

cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$2 = (function (ch,coll){
return cljs.core.async.onto_chan.call(null,ch,coll,true);
});

cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$3 = (function (ch,coll,close_QMARK_){
var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__){
return (function (state_19753){
var state_val_19754 = (state_19753[(1)]);
if((state_val_19754 === (7))){
var inst_19735 = (state_19753[(2)]);
var state_19753__$1 = state_19753;
var statearr_19755_19782 = state_19753__$1;
(statearr_19755_19782[(2)] = inst_19735);

(statearr_19755_19782[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (1))){
var inst_19729 = cljs.core.seq.call(null,coll);
var inst_19730 = inst_19729;
var state_19753__$1 = (function (){var statearr_19756 = state_19753;
(statearr_19756[(7)] = inst_19730);

return statearr_19756;
})();
var statearr_19757_19783 = state_19753__$1;
(statearr_19757_19783[(2)] = null);

(statearr_19757_19783[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (4))){
var inst_19730 = (state_19753[(7)]);
var inst_19733 = cljs.core.first.call(null,inst_19730);
var state_19753__$1 = state_19753;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_19753__$1,(7),ch,inst_19733);
} else {
if((state_val_19754 === (13))){
var inst_19747 = (state_19753[(2)]);
var state_19753__$1 = state_19753;
var statearr_19758_19784 = state_19753__$1;
(statearr_19758_19784[(2)] = inst_19747);

(statearr_19758_19784[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (6))){
var inst_19738 = (state_19753[(2)]);
var state_19753__$1 = state_19753;
if(cljs.core.truth_(inst_19738)){
var statearr_19759_19785 = state_19753__$1;
(statearr_19759_19785[(1)] = (8));

} else {
var statearr_19760_19786 = state_19753__$1;
(statearr_19760_19786[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (3))){
var inst_19751 = (state_19753[(2)]);
var state_19753__$1 = state_19753;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19753__$1,inst_19751);
} else {
if((state_val_19754 === (12))){
var state_19753__$1 = state_19753;
var statearr_19761_19787 = state_19753__$1;
(statearr_19761_19787[(2)] = null);

(statearr_19761_19787[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (2))){
var inst_19730 = (state_19753[(7)]);
var state_19753__$1 = state_19753;
if(cljs.core.truth_(inst_19730)){
var statearr_19762_19788 = state_19753__$1;
(statearr_19762_19788[(1)] = (4));

} else {
var statearr_19763_19789 = state_19753__$1;
(statearr_19763_19789[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (11))){
var inst_19744 = cljs.core.async.close_BANG_.call(null,ch);
var state_19753__$1 = state_19753;
var statearr_19764_19790 = state_19753__$1;
(statearr_19764_19790[(2)] = inst_19744);

(statearr_19764_19790[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (9))){
var state_19753__$1 = state_19753;
if(cljs.core.truth_(close_QMARK_)){
var statearr_19765_19791 = state_19753__$1;
(statearr_19765_19791[(1)] = (11));

} else {
var statearr_19766_19792 = state_19753__$1;
(statearr_19766_19792[(1)] = (12));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (5))){
var inst_19730 = (state_19753[(7)]);
var state_19753__$1 = state_19753;
var statearr_19767_19793 = state_19753__$1;
(statearr_19767_19793[(2)] = inst_19730);

(statearr_19767_19793[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (10))){
var inst_19749 = (state_19753[(2)]);
var state_19753__$1 = state_19753;
var statearr_19768_19794 = state_19753__$1;
(statearr_19768_19794[(2)] = inst_19749);

(statearr_19768_19794[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19754 === (8))){
var inst_19730 = (state_19753[(7)]);
var inst_19740 = cljs.core.next.call(null,inst_19730);
var inst_19730__$1 = inst_19740;
var state_19753__$1 = (function (){var statearr_19769 = state_19753;
(statearr_19769[(7)] = inst_19730__$1);

return statearr_19769;
})();
var statearr_19770_19795 = state_19753__$1;
(statearr_19770_19795[(2)] = null);

(statearr_19770_19795[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto__))
;
return ((function (switch__18821__auto__,c__18933__auto__){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_19774 = [null,null,null,null,null,null,null,null];
(statearr_19774[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_19774[(1)] = (1));

return statearr_19774;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_19753){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_19753);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e19775){if((e19775 instanceof Object)){
var ex__18825__auto__ = e19775;
var statearr_19776_19796 = state_19753;
(statearr_19776_19796[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19753);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e19775;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__19797 = state_19753;
state_19753 = G__19797;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_19753){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_19753);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__))
})();
var state__18935__auto__ = (function (){var statearr_19777 = f__18934__auto__.call(null);
(statearr_19777[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_19777;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__))
);

return c__18933__auto__;
});

cljs.core.async.onto_chan.cljs$lang$maxFixedArity = 3;
/**
 * Creates and returns a channel which contains the contents of coll,
 *   closing when exhausted.
 */
cljs.core.async.to_chan = (function cljs$core$async$to_chan(coll){
var ch = cljs.core.async.chan.call(null,cljs.core.bounded_count.call(null,(100),coll));
cljs.core.async.onto_chan.call(null,ch,coll);

return ch;
});

/**
 * @interface
 */
cljs.core.async.Mux = function(){};

cljs.core.async.muxch_STAR_ = (function cljs$core$async$muxch_STAR_(_){
if((!((_ == null))) && (!((_.cljs$core$async$Mux$muxch_STAR_$arity$1 == null)))){
return _.cljs$core$async$Mux$muxch_STAR_$arity$1(_);
} else {
var x__17421__auto__ = (((_ == null))?null:_);
var m__17422__auto__ = (cljs.core.async.muxch_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,_);
} else {
var m__17422__auto____$1 = (cljs.core.async.muxch_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,_);
} else {
throw cljs.core.missing_protocol.call(null,"Mux.muxch*",_);
}
}
}
});


/**
 * @interface
 */
cljs.core.async.Mult = function(){};

cljs.core.async.tap_STAR_ = (function cljs$core$async$tap_STAR_(m,ch,close_QMARK_){
if((!((m == null))) && (!((m.cljs$core$async$Mult$tap_STAR_$arity$3 == null)))){
return m.cljs$core$async$Mult$tap_STAR_$arity$3(m,ch,close_QMARK_);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.tap_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,ch,close_QMARK_);
} else {
var m__17422__auto____$1 = (cljs.core.async.tap_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,ch,close_QMARK_);
} else {
throw cljs.core.missing_protocol.call(null,"Mult.tap*",m);
}
}
}
});

cljs.core.async.untap_STAR_ = (function cljs$core$async$untap_STAR_(m,ch){
if((!((m == null))) && (!((m.cljs$core$async$Mult$untap_STAR_$arity$2 == null)))){
return m.cljs$core$async$Mult$untap_STAR_$arity$2(m,ch);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.untap_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,ch);
} else {
var m__17422__auto____$1 = (cljs.core.async.untap_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,ch);
} else {
throw cljs.core.missing_protocol.call(null,"Mult.untap*",m);
}
}
}
});

cljs.core.async.untap_all_STAR_ = (function cljs$core$async$untap_all_STAR_(m){
if((!((m == null))) && (!((m.cljs$core$async$Mult$untap_all_STAR_$arity$1 == null)))){
return m.cljs$core$async$Mult$untap_all_STAR_$arity$1(m);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.untap_all_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m);
} else {
var m__17422__auto____$1 = (cljs.core.async.untap_all_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m);
} else {
throw cljs.core.missing_protocol.call(null,"Mult.untap-all*",m);
}
}
}
});

/**
 * Creates and returns a mult(iple) of the supplied channel. Channels
 *   containing copies of the channel can be created with 'tap', and
 *   detached with 'untap'.
 * 
 *   Each item is distributed to all taps in parallel and synchronously,
 *   i.e. each tap must accept before the next item is distributed. Use
 *   buffering/windowing to prevent slow taps from holding up the mult.
 * 
 *   Items received when there are no taps get dropped.
 * 
 *   If a tap puts to a closed channel, it will be removed from the mult.
 */
cljs.core.async.mult = (function cljs$core$async$mult(ch){
var cs = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var m = (function (){
if(typeof cljs.core.async.t_cljs$core$async20019 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.Mult}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async20019 = (function (mult,ch,cs,meta20020){
this.mult = mult;
this.ch = ch;
this.cs = cs;
this.meta20020 = meta20020;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (cs){
return (function (_20021,meta20020__$1){
var self__ = this;
var _20021__$1 = this;
return (new cljs.core.async.t_cljs$core$async20019(self__.mult,self__.ch,self__.cs,meta20020__$1));
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (cs){
return (function (_20021){
var self__ = this;
var _20021__$1 = this;
return self__.meta20020;
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mux$ = true;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (cs){
return (function (_){
var self__ = this;
var ___$1 = this;
return self__.ch;
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mult$ = true;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mult$tap_STAR_$arity$3 = ((function (cs){
return (function (_,ch__$1,close_QMARK_){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.assoc,ch__$1,close_QMARK_);

return null;
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mult$untap_STAR_$arity$2 = ((function (cs){
return (function (_,ch__$1){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.dissoc,ch__$1);

return null;
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.prototype.cljs$core$async$Mult$untap_all_STAR_$arity$1 = ((function (cs){
return (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_.call(null,self__.cs,cljs.core.PersistentArrayMap.EMPTY);

return null;
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.getBasis = ((function (cs){
return (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"mult","mult",-1187640995,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"ch","ch",1085813622,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Creates and returns a mult(iple) of the supplied channel. Channels\n  containing copies of the channel can be created with 'tap', and\n  detached with 'untap'.\n\n  Each item is distributed to all taps in parallel and synchronously,\n  i.e. each tap must accept before the next item is distributed. Use\n  buffering/windowing to prevent slow taps from holding up the mult.\n\n  Items received when there are no taps get dropped.\n\n  If a tap puts to a closed channel, it will be removed from the mult."], null)),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"cs","cs",-117024463,null),new cljs.core.Symbol(null,"meta20020","meta20020",1040639285,null)], null);
});})(cs))
;

cljs.core.async.t_cljs$core$async20019.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async20019.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async20019";

cljs.core.async.t_cljs$core$async20019.cljs$lang$ctorPrWriter = ((function (cs){
return (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async20019");
});})(cs))
;

cljs.core.async.__GT_t_cljs$core$async20019 = ((function (cs){
return (function cljs$core$async$mult_$___GT_t_cljs$core$async20019(mult__$1,ch__$1,cs__$1,meta20020){
return (new cljs.core.async.t_cljs$core$async20019(mult__$1,ch__$1,cs__$1,meta20020));
});})(cs))
;

}

return (new cljs.core.async.t_cljs$core$async20019(cljs$core$async$mult,ch,cs,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var dchan = cljs.core.async.chan.call(null,(1));
var dctr = cljs.core.atom.call(null,null);
var done = ((function (cs,m,dchan,dctr){
return (function (_){
if((cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec) === (0))){
return cljs.core.async.put_BANG_.call(null,dchan,true);
} else {
return null;
}
});})(cs,m,dchan,dctr))
;
var c__18933__auto___20240 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___20240,cs,m,dchan,dctr,done){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___20240,cs,m,dchan,dctr,done){
return (function (state_20152){
var state_val_20153 = (state_20152[(1)]);
if((state_val_20153 === (7))){
var inst_20148 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20154_20241 = state_20152__$1;
(statearr_20154_20241[(2)] = inst_20148);

(statearr_20154_20241[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (20))){
var inst_20053 = (state_20152[(7)]);
var inst_20063 = cljs.core.first.call(null,inst_20053);
var inst_20064 = cljs.core.nth.call(null,inst_20063,(0),null);
var inst_20065 = cljs.core.nth.call(null,inst_20063,(1),null);
var state_20152__$1 = (function (){var statearr_20155 = state_20152;
(statearr_20155[(8)] = inst_20064);

return statearr_20155;
})();
if(cljs.core.truth_(inst_20065)){
var statearr_20156_20242 = state_20152__$1;
(statearr_20156_20242[(1)] = (22));

} else {
var statearr_20157_20243 = state_20152__$1;
(statearr_20157_20243[(1)] = (23));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (27))){
var inst_20093 = (state_20152[(9)]);
var inst_20100 = (state_20152[(10)]);
var inst_20024 = (state_20152[(11)]);
var inst_20095 = (state_20152[(12)]);
var inst_20100__$1 = cljs.core._nth.call(null,inst_20093,inst_20095);
var inst_20101 = cljs.core.async.put_BANG_.call(null,inst_20100__$1,inst_20024,done);
var state_20152__$1 = (function (){var statearr_20158 = state_20152;
(statearr_20158[(10)] = inst_20100__$1);

return statearr_20158;
})();
if(cljs.core.truth_(inst_20101)){
var statearr_20159_20244 = state_20152__$1;
(statearr_20159_20244[(1)] = (30));

} else {
var statearr_20160_20245 = state_20152__$1;
(statearr_20160_20245[(1)] = (31));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (1))){
var state_20152__$1 = state_20152;
var statearr_20161_20246 = state_20152__$1;
(statearr_20161_20246[(2)] = null);

(statearr_20161_20246[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (24))){
var inst_20053 = (state_20152[(7)]);
var inst_20070 = (state_20152[(2)]);
var inst_20071 = cljs.core.next.call(null,inst_20053);
var inst_20033 = inst_20071;
var inst_20034 = null;
var inst_20035 = (0);
var inst_20036 = (0);
var state_20152__$1 = (function (){var statearr_20162 = state_20152;
(statearr_20162[(13)] = inst_20070);

(statearr_20162[(14)] = inst_20036);

(statearr_20162[(15)] = inst_20035);

(statearr_20162[(16)] = inst_20033);

(statearr_20162[(17)] = inst_20034);

return statearr_20162;
})();
var statearr_20163_20247 = state_20152__$1;
(statearr_20163_20247[(2)] = null);

(statearr_20163_20247[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (39))){
var state_20152__$1 = state_20152;
var statearr_20167_20248 = state_20152__$1;
(statearr_20167_20248[(2)] = null);

(statearr_20167_20248[(1)] = (41));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (4))){
var inst_20024 = (state_20152[(11)]);
var inst_20024__$1 = (state_20152[(2)]);
var inst_20025 = (inst_20024__$1 == null);
var state_20152__$1 = (function (){var statearr_20168 = state_20152;
(statearr_20168[(11)] = inst_20024__$1);

return statearr_20168;
})();
if(cljs.core.truth_(inst_20025)){
var statearr_20169_20249 = state_20152__$1;
(statearr_20169_20249[(1)] = (5));

} else {
var statearr_20170_20250 = state_20152__$1;
(statearr_20170_20250[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (15))){
var inst_20036 = (state_20152[(14)]);
var inst_20035 = (state_20152[(15)]);
var inst_20033 = (state_20152[(16)]);
var inst_20034 = (state_20152[(17)]);
var inst_20049 = (state_20152[(2)]);
var inst_20050 = (inst_20036 + (1));
var tmp20164 = inst_20035;
var tmp20165 = inst_20033;
var tmp20166 = inst_20034;
var inst_20033__$1 = tmp20165;
var inst_20034__$1 = tmp20166;
var inst_20035__$1 = tmp20164;
var inst_20036__$1 = inst_20050;
var state_20152__$1 = (function (){var statearr_20171 = state_20152;
(statearr_20171[(18)] = inst_20049);

(statearr_20171[(14)] = inst_20036__$1);

(statearr_20171[(15)] = inst_20035__$1);

(statearr_20171[(16)] = inst_20033__$1);

(statearr_20171[(17)] = inst_20034__$1);

return statearr_20171;
})();
var statearr_20172_20251 = state_20152__$1;
(statearr_20172_20251[(2)] = null);

(statearr_20172_20251[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (21))){
var inst_20074 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20176_20252 = state_20152__$1;
(statearr_20176_20252[(2)] = inst_20074);

(statearr_20176_20252[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (31))){
var inst_20100 = (state_20152[(10)]);
var inst_20104 = done.call(null,null);
var inst_20105 = cljs.core.async.untap_STAR_.call(null,m,inst_20100);
var state_20152__$1 = (function (){var statearr_20177 = state_20152;
(statearr_20177[(19)] = inst_20104);

return statearr_20177;
})();
var statearr_20178_20253 = state_20152__$1;
(statearr_20178_20253[(2)] = inst_20105);

(statearr_20178_20253[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (32))){
var inst_20092 = (state_20152[(20)]);
var inst_20093 = (state_20152[(9)]);
var inst_20094 = (state_20152[(21)]);
var inst_20095 = (state_20152[(12)]);
var inst_20107 = (state_20152[(2)]);
var inst_20108 = (inst_20095 + (1));
var tmp20173 = inst_20092;
var tmp20174 = inst_20093;
var tmp20175 = inst_20094;
var inst_20092__$1 = tmp20173;
var inst_20093__$1 = tmp20174;
var inst_20094__$1 = tmp20175;
var inst_20095__$1 = inst_20108;
var state_20152__$1 = (function (){var statearr_20179 = state_20152;
(statearr_20179[(20)] = inst_20092__$1);

(statearr_20179[(9)] = inst_20093__$1);

(statearr_20179[(22)] = inst_20107);

(statearr_20179[(21)] = inst_20094__$1);

(statearr_20179[(12)] = inst_20095__$1);

return statearr_20179;
})();
var statearr_20180_20254 = state_20152__$1;
(statearr_20180_20254[(2)] = null);

(statearr_20180_20254[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (40))){
var inst_20120 = (state_20152[(23)]);
var inst_20124 = done.call(null,null);
var inst_20125 = cljs.core.async.untap_STAR_.call(null,m,inst_20120);
var state_20152__$1 = (function (){var statearr_20181 = state_20152;
(statearr_20181[(24)] = inst_20124);

return statearr_20181;
})();
var statearr_20182_20255 = state_20152__$1;
(statearr_20182_20255[(2)] = inst_20125);

(statearr_20182_20255[(1)] = (41));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (33))){
var inst_20111 = (state_20152[(25)]);
var inst_20113 = cljs.core.chunked_seq_QMARK_.call(null,inst_20111);
var state_20152__$1 = state_20152;
if(inst_20113){
var statearr_20183_20256 = state_20152__$1;
(statearr_20183_20256[(1)] = (36));

} else {
var statearr_20184_20257 = state_20152__$1;
(statearr_20184_20257[(1)] = (37));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (13))){
var inst_20043 = (state_20152[(26)]);
var inst_20046 = cljs.core.async.close_BANG_.call(null,inst_20043);
var state_20152__$1 = state_20152;
var statearr_20185_20258 = state_20152__$1;
(statearr_20185_20258[(2)] = inst_20046);

(statearr_20185_20258[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (22))){
var inst_20064 = (state_20152[(8)]);
var inst_20067 = cljs.core.async.close_BANG_.call(null,inst_20064);
var state_20152__$1 = state_20152;
var statearr_20186_20259 = state_20152__$1;
(statearr_20186_20259[(2)] = inst_20067);

(statearr_20186_20259[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (36))){
var inst_20111 = (state_20152[(25)]);
var inst_20115 = cljs.core.chunk_first.call(null,inst_20111);
var inst_20116 = cljs.core.chunk_rest.call(null,inst_20111);
var inst_20117 = cljs.core.count.call(null,inst_20115);
var inst_20092 = inst_20116;
var inst_20093 = inst_20115;
var inst_20094 = inst_20117;
var inst_20095 = (0);
var state_20152__$1 = (function (){var statearr_20187 = state_20152;
(statearr_20187[(20)] = inst_20092);

(statearr_20187[(9)] = inst_20093);

(statearr_20187[(21)] = inst_20094);

(statearr_20187[(12)] = inst_20095);

return statearr_20187;
})();
var statearr_20188_20260 = state_20152__$1;
(statearr_20188_20260[(2)] = null);

(statearr_20188_20260[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (41))){
var inst_20111 = (state_20152[(25)]);
var inst_20127 = (state_20152[(2)]);
var inst_20128 = cljs.core.next.call(null,inst_20111);
var inst_20092 = inst_20128;
var inst_20093 = null;
var inst_20094 = (0);
var inst_20095 = (0);
var state_20152__$1 = (function (){var statearr_20189 = state_20152;
(statearr_20189[(20)] = inst_20092);

(statearr_20189[(9)] = inst_20093);

(statearr_20189[(27)] = inst_20127);

(statearr_20189[(21)] = inst_20094);

(statearr_20189[(12)] = inst_20095);

return statearr_20189;
})();
var statearr_20190_20261 = state_20152__$1;
(statearr_20190_20261[(2)] = null);

(statearr_20190_20261[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (43))){
var state_20152__$1 = state_20152;
var statearr_20191_20262 = state_20152__$1;
(statearr_20191_20262[(2)] = null);

(statearr_20191_20262[(1)] = (44));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (29))){
var inst_20136 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20192_20263 = state_20152__$1;
(statearr_20192_20263[(2)] = inst_20136);

(statearr_20192_20263[(1)] = (26));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (44))){
var inst_20145 = (state_20152[(2)]);
var state_20152__$1 = (function (){var statearr_20193 = state_20152;
(statearr_20193[(28)] = inst_20145);

return statearr_20193;
})();
var statearr_20194_20264 = state_20152__$1;
(statearr_20194_20264[(2)] = null);

(statearr_20194_20264[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (6))){
var inst_20084 = (state_20152[(29)]);
var inst_20083 = cljs.core.deref.call(null,cs);
var inst_20084__$1 = cljs.core.keys.call(null,inst_20083);
var inst_20085 = cljs.core.count.call(null,inst_20084__$1);
var inst_20086 = cljs.core.reset_BANG_.call(null,dctr,inst_20085);
var inst_20091 = cljs.core.seq.call(null,inst_20084__$1);
var inst_20092 = inst_20091;
var inst_20093 = null;
var inst_20094 = (0);
var inst_20095 = (0);
var state_20152__$1 = (function (){var statearr_20195 = state_20152;
(statearr_20195[(20)] = inst_20092);

(statearr_20195[(9)] = inst_20093);

(statearr_20195[(30)] = inst_20086);

(statearr_20195[(29)] = inst_20084__$1);

(statearr_20195[(21)] = inst_20094);

(statearr_20195[(12)] = inst_20095);

return statearr_20195;
})();
var statearr_20196_20265 = state_20152__$1;
(statearr_20196_20265[(2)] = null);

(statearr_20196_20265[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (28))){
var inst_20092 = (state_20152[(20)]);
var inst_20111 = (state_20152[(25)]);
var inst_20111__$1 = cljs.core.seq.call(null,inst_20092);
var state_20152__$1 = (function (){var statearr_20197 = state_20152;
(statearr_20197[(25)] = inst_20111__$1);

return statearr_20197;
})();
if(inst_20111__$1){
var statearr_20198_20266 = state_20152__$1;
(statearr_20198_20266[(1)] = (33));

} else {
var statearr_20199_20267 = state_20152__$1;
(statearr_20199_20267[(1)] = (34));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (25))){
var inst_20094 = (state_20152[(21)]);
var inst_20095 = (state_20152[(12)]);
var inst_20097 = (inst_20095 < inst_20094);
var inst_20098 = inst_20097;
var state_20152__$1 = state_20152;
if(cljs.core.truth_(inst_20098)){
var statearr_20200_20268 = state_20152__$1;
(statearr_20200_20268[(1)] = (27));

} else {
var statearr_20201_20269 = state_20152__$1;
(statearr_20201_20269[(1)] = (28));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (34))){
var state_20152__$1 = state_20152;
var statearr_20202_20270 = state_20152__$1;
(statearr_20202_20270[(2)] = null);

(statearr_20202_20270[(1)] = (35));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (17))){
var state_20152__$1 = state_20152;
var statearr_20203_20271 = state_20152__$1;
(statearr_20203_20271[(2)] = null);

(statearr_20203_20271[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (3))){
var inst_20150 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_20152__$1,inst_20150);
} else {
if((state_val_20153 === (12))){
var inst_20079 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20204_20272 = state_20152__$1;
(statearr_20204_20272[(2)] = inst_20079);

(statearr_20204_20272[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (2))){
var state_20152__$1 = state_20152;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20152__$1,(4),ch);
} else {
if((state_val_20153 === (23))){
var state_20152__$1 = state_20152;
var statearr_20205_20273 = state_20152__$1;
(statearr_20205_20273[(2)] = null);

(statearr_20205_20273[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (35))){
var inst_20134 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20206_20274 = state_20152__$1;
(statearr_20206_20274[(2)] = inst_20134);

(statearr_20206_20274[(1)] = (29));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (19))){
var inst_20053 = (state_20152[(7)]);
var inst_20057 = cljs.core.chunk_first.call(null,inst_20053);
var inst_20058 = cljs.core.chunk_rest.call(null,inst_20053);
var inst_20059 = cljs.core.count.call(null,inst_20057);
var inst_20033 = inst_20058;
var inst_20034 = inst_20057;
var inst_20035 = inst_20059;
var inst_20036 = (0);
var state_20152__$1 = (function (){var statearr_20207 = state_20152;
(statearr_20207[(14)] = inst_20036);

(statearr_20207[(15)] = inst_20035);

(statearr_20207[(16)] = inst_20033);

(statearr_20207[(17)] = inst_20034);

return statearr_20207;
})();
var statearr_20208_20275 = state_20152__$1;
(statearr_20208_20275[(2)] = null);

(statearr_20208_20275[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (11))){
var inst_20053 = (state_20152[(7)]);
var inst_20033 = (state_20152[(16)]);
var inst_20053__$1 = cljs.core.seq.call(null,inst_20033);
var state_20152__$1 = (function (){var statearr_20209 = state_20152;
(statearr_20209[(7)] = inst_20053__$1);

return statearr_20209;
})();
if(inst_20053__$1){
var statearr_20210_20276 = state_20152__$1;
(statearr_20210_20276[(1)] = (16));

} else {
var statearr_20211_20277 = state_20152__$1;
(statearr_20211_20277[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (9))){
var inst_20081 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20212_20278 = state_20152__$1;
(statearr_20212_20278[(2)] = inst_20081);

(statearr_20212_20278[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (5))){
var inst_20031 = cljs.core.deref.call(null,cs);
var inst_20032 = cljs.core.seq.call(null,inst_20031);
var inst_20033 = inst_20032;
var inst_20034 = null;
var inst_20035 = (0);
var inst_20036 = (0);
var state_20152__$1 = (function (){var statearr_20213 = state_20152;
(statearr_20213[(14)] = inst_20036);

(statearr_20213[(15)] = inst_20035);

(statearr_20213[(16)] = inst_20033);

(statearr_20213[(17)] = inst_20034);

return statearr_20213;
})();
var statearr_20214_20279 = state_20152__$1;
(statearr_20214_20279[(2)] = null);

(statearr_20214_20279[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (14))){
var state_20152__$1 = state_20152;
var statearr_20215_20280 = state_20152__$1;
(statearr_20215_20280[(2)] = null);

(statearr_20215_20280[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (45))){
var inst_20142 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20216_20281 = state_20152__$1;
(statearr_20216_20281[(2)] = inst_20142);

(statearr_20216_20281[(1)] = (44));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (26))){
var inst_20084 = (state_20152[(29)]);
var inst_20138 = (state_20152[(2)]);
var inst_20139 = cljs.core.seq.call(null,inst_20084);
var state_20152__$1 = (function (){var statearr_20217 = state_20152;
(statearr_20217[(31)] = inst_20138);

return statearr_20217;
})();
if(inst_20139){
var statearr_20218_20282 = state_20152__$1;
(statearr_20218_20282[(1)] = (42));

} else {
var statearr_20219_20283 = state_20152__$1;
(statearr_20219_20283[(1)] = (43));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (16))){
var inst_20053 = (state_20152[(7)]);
var inst_20055 = cljs.core.chunked_seq_QMARK_.call(null,inst_20053);
var state_20152__$1 = state_20152;
if(inst_20055){
var statearr_20220_20284 = state_20152__$1;
(statearr_20220_20284[(1)] = (19));

} else {
var statearr_20221_20285 = state_20152__$1;
(statearr_20221_20285[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (38))){
var inst_20131 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20222_20286 = state_20152__$1;
(statearr_20222_20286[(2)] = inst_20131);

(statearr_20222_20286[(1)] = (35));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (30))){
var state_20152__$1 = state_20152;
var statearr_20223_20287 = state_20152__$1;
(statearr_20223_20287[(2)] = null);

(statearr_20223_20287[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (10))){
var inst_20036 = (state_20152[(14)]);
var inst_20034 = (state_20152[(17)]);
var inst_20042 = cljs.core._nth.call(null,inst_20034,inst_20036);
var inst_20043 = cljs.core.nth.call(null,inst_20042,(0),null);
var inst_20044 = cljs.core.nth.call(null,inst_20042,(1),null);
var state_20152__$1 = (function (){var statearr_20224 = state_20152;
(statearr_20224[(26)] = inst_20043);

return statearr_20224;
})();
if(cljs.core.truth_(inst_20044)){
var statearr_20225_20288 = state_20152__$1;
(statearr_20225_20288[(1)] = (13));

} else {
var statearr_20226_20289 = state_20152__$1;
(statearr_20226_20289[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (18))){
var inst_20077 = (state_20152[(2)]);
var state_20152__$1 = state_20152;
var statearr_20227_20290 = state_20152__$1;
(statearr_20227_20290[(2)] = inst_20077);

(statearr_20227_20290[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (42))){
var state_20152__$1 = state_20152;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20152__$1,(45),dchan);
} else {
if((state_val_20153 === (37))){
var inst_20111 = (state_20152[(25)]);
var inst_20120 = (state_20152[(23)]);
var inst_20024 = (state_20152[(11)]);
var inst_20120__$1 = cljs.core.first.call(null,inst_20111);
var inst_20121 = cljs.core.async.put_BANG_.call(null,inst_20120__$1,inst_20024,done);
var state_20152__$1 = (function (){var statearr_20228 = state_20152;
(statearr_20228[(23)] = inst_20120__$1);

return statearr_20228;
})();
if(cljs.core.truth_(inst_20121)){
var statearr_20229_20291 = state_20152__$1;
(statearr_20229_20291[(1)] = (39));

} else {
var statearr_20230_20292 = state_20152__$1;
(statearr_20230_20292[(1)] = (40));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20153 === (8))){
var inst_20036 = (state_20152[(14)]);
var inst_20035 = (state_20152[(15)]);
var inst_20038 = (inst_20036 < inst_20035);
var inst_20039 = inst_20038;
var state_20152__$1 = state_20152;
if(cljs.core.truth_(inst_20039)){
var statearr_20231_20293 = state_20152__$1;
(statearr_20231_20293[(1)] = (10));

} else {
var statearr_20232_20294 = state_20152__$1;
(statearr_20232_20294[(1)] = (11));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___20240,cs,m,dchan,dctr,done))
;
return ((function (switch__18821__auto__,c__18933__auto___20240,cs,m,dchan,dctr,done){
return (function() {
var cljs$core$async$mult_$_state_machine__18822__auto__ = null;
var cljs$core$async$mult_$_state_machine__18822__auto____0 = (function (){
var statearr_20236 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_20236[(0)] = cljs$core$async$mult_$_state_machine__18822__auto__);

(statearr_20236[(1)] = (1));

return statearr_20236;
});
var cljs$core$async$mult_$_state_machine__18822__auto____1 = (function (state_20152){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_20152);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e20237){if((e20237 instanceof Object)){
var ex__18825__auto__ = e20237;
var statearr_20238_20295 = state_20152;
(statearr_20238_20295[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20152);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20237;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20296 = state_20152;
state_20152 = G__20296;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$mult_$_state_machine__18822__auto__ = function(state_20152){
switch(arguments.length){
case 0:
return cljs$core$async$mult_$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$mult_$_state_machine__18822__auto____1.call(this,state_20152);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mult_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mult_$_state_machine__18822__auto____0;
cljs$core$async$mult_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mult_$_state_machine__18822__auto____1;
return cljs$core$async$mult_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___20240,cs,m,dchan,dctr,done))
})();
var state__18935__auto__ = (function (){var statearr_20239 = f__18934__auto__.call(null);
(statearr_20239[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___20240);

return statearr_20239;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___20240,cs,m,dchan,dctr,done))
);


return m;
});
/**
 * Copies the mult source onto the supplied channel.
 * 
 *   By default the channel will be closed when the source closes,
 *   but can be determined by the close? parameter.
 */
cljs.core.async.tap = (function cljs$core$async$tap(var_args){
var args20297 = [];
var len__17824__auto___20300 = arguments.length;
var i__17825__auto___20301 = (0);
while(true){
if((i__17825__auto___20301 < len__17824__auto___20300)){
args20297.push((arguments[i__17825__auto___20301]));

var G__20302 = (i__17825__auto___20301 + (1));
i__17825__auto___20301 = G__20302;
continue;
} else {
}
break;
}

var G__20299 = args20297.length;
switch (G__20299) {
case 2:
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20297.length)].join('')));

}
});

cljs.core.async.tap.cljs$core$IFn$_invoke$arity$2 = (function (mult,ch){
return cljs.core.async.tap.call(null,mult,ch,true);
});

cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3 = (function (mult,ch,close_QMARK_){
cljs.core.async.tap_STAR_.call(null,mult,ch,close_QMARK_);

return ch;
});

cljs.core.async.tap.cljs$lang$maxFixedArity = 3;
/**
 * Disconnects a target channel from a mult
 */
cljs.core.async.untap = (function cljs$core$async$untap(mult,ch){
return cljs.core.async.untap_STAR_.call(null,mult,ch);
});
/**
 * Disconnects all target channels from a mult
 */
cljs.core.async.untap_all = (function cljs$core$async$untap_all(mult){
return cljs.core.async.untap_all_STAR_.call(null,mult);
});

/**
 * @interface
 */
cljs.core.async.Mix = function(){};

cljs.core.async.admix_STAR_ = (function cljs$core$async$admix_STAR_(m,ch){
if((!((m == null))) && (!((m.cljs$core$async$Mix$admix_STAR_$arity$2 == null)))){
return m.cljs$core$async$Mix$admix_STAR_$arity$2(m,ch);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.admix_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,ch);
} else {
var m__17422__auto____$1 = (cljs.core.async.admix_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,ch);
} else {
throw cljs.core.missing_protocol.call(null,"Mix.admix*",m);
}
}
}
});

cljs.core.async.unmix_STAR_ = (function cljs$core$async$unmix_STAR_(m,ch){
if((!((m == null))) && (!((m.cljs$core$async$Mix$unmix_STAR_$arity$2 == null)))){
return m.cljs$core$async$Mix$unmix_STAR_$arity$2(m,ch);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.unmix_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,ch);
} else {
var m__17422__auto____$1 = (cljs.core.async.unmix_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,ch);
} else {
throw cljs.core.missing_protocol.call(null,"Mix.unmix*",m);
}
}
}
});

cljs.core.async.unmix_all_STAR_ = (function cljs$core$async$unmix_all_STAR_(m){
if((!((m == null))) && (!((m.cljs$core$async$Mix$unmix_all_STAR_$arity$1 == null)))){
return m.cljs$core$async$Mix$unmix_all_STAR_$arity$1(m);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.unmix_all_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m);
} else {
var m__17422__auto____$1 = (cljs.core.async.unmix_all_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m);
} else {
throw cljs.core.missing_protocol.call(null,"Mix.unmix-all*",m);
}
}
}
});

cljs.core.async.toggle_STAR_ = (function cljs$core$async$toggle_STAR_(m,state_map){
if((!((m == null))) && (!((m.cljs$core$async$Mix$toggle_STAR_$arity$2 == null)))){
return m.cljs$core$async$Mix$toggle_STAR_$arity$2(m,state_map);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.toggle_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,state_map);
} else {
var m__17422__auto____$1 = (cljs.core.async.toggle_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,state_map);
} else {
throw cljs.core.missing_protocol.call(null,"Mix.toggle*",m);
}
}
}
});

cljs.core.async.solo_mode_STAR_ = (function cljs$core$async$solo_mode_STAR_(m,mode){
if((!((m == null))) && (!((m.cljs$core$async$Mix$solo_mode_STAR_$arity$2 == null)))){
return m.cljs$core$async$Mix$solo_mode_STAR_$arity$2(m,mode);
} else {
var x__17421__auto__ = (((m == null))?null:m);
var m__17422__auto__ = (cljs.core.async.solo_mode_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,m,mode);
} else {
var m__17422__auto____$1 = (cljs.core.async.solo_mode_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,m,mode);
} else {
throw cljs.core.missing_protocol.call(null,"Mix.solo-mode*",m);
}
}
}
});

cljs.core.async.ioc_alts_BANG_ = (function cljs$core$async$ioc_alts_BANG_(var_args){
var args__17831__auto__ = [];
var len__17824__auto___20314 = arguments.length;
var i__17825__auto___20315 = (0);
while(true){
if((i__17825__auto___20315 < len__17824__auto___20314)){
args__17831__auto__.push((arguments[i__17825__auto___20315]));

var G__20316 = (i__17825__auto___20315 + (1));
i__17825__auto___20315 = G__20316;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((3) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((3)),(0))):null);
return cljs.core.async.ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),argseq__17832__auto__);
});

cljs.core.async.ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = (function (state,cont_block,ports,p__20308){
var map__20309 = p__20308;
var map__20309__$1 = ((((!((map__20309 == null)))?((((map__20309.cljs$lang$protocol_mask$partition0$ & (64))) || (map__20309.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__20309):map__20309);
var opts = map__20309__$1;
var statearr_20311_20317 = state;
(statearr_20311_20317[cljs.core.async.impl.ioc_helpers.STATE_IDX] = cont_block);


var temp__4425__auto__ = cljs.core.async.do_alts.call(null,((function (map__20309,map__20309__$1,opts){
return (function (val){
var statearr_20312_20318 = state;
(statearr_20312_20318[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = val);


return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state);
});})(map__20309,map__20309__$1,opts))
,ports,opts);
if(cljs.core.truth_(temp__4425__auto__)){
var cb = temp__4425__auto__;
var statearr_20313_20319 = state;
(statearr_20313_20319[cljs.core.async.impl.ioc_helpers.VALUE_IDX] = cljs.core.deref.call(null,cb));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
});

cljs.core.async.ioc_alts_BANG_.cljs$lang$maxFixedArity = (3);

cljs.core.async.ioc_alts_BANG_.cljs$lang$applyTo = (function (seq20304){
var G__20305 = cljs.core.first.call(null,seq20304);
var seq20304__$1 = cljs.core.next.call(null,seq20304);
var G__20306 = cljs.core.first.call(null,seq20304__$1);
var seq20304__$2 = cljs.core.next.call(null,seq20304__$1);
var G__20307 = cljs.core.first.call(null,seq20304__$2);
var seq20304__$3 = cljs.core.next.call(null,seq20304__$2);
return cljs.core.async.ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic(G__20305,G__20306,G__20307,seq20304__$3);
});
/**
 * Creates and returns a mix of one or more input channels which will
 *   be put on the supplied out channel. Input sources can be added to
 *   the mix with 'admix', and removed with 'unmix'. A mix supports
 *   soloing, muting and pausing multiple inputs atomically using
 *   'toggle', and can solo using either muting or pausing as determined
 *   by 'solo-mode'.
 * 
 *   Each channel can have zero or more boolean modes set via 'toggle':
 * 
 *   :solo - when true, only this (ond other soloed) channel(s) will appear
 *        in the mix output channel. :mute and :pause states of soloed
 *        channels are ignored. If solo-mode is :mute, non-soloed
 *        channels are muted, if :pause, non-soloed channels are
 *        paused.
 * 
 *   :mute - muted channels will have their contents consumed but not included in the mix
 *   :pause - paused channels will not have their contents consumed (and thus also not included in the mix)
 */
cljs.core.async.mix = (function cljs$core$async$mix(out){
var cs = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var solo_modes = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"pause","pause",-2095325672),null,new cljs.core.Keyword(null,"mute","mute",1151223646),null], null), null);
var attrs = cljs.core.conj.call(null,solo_modes,new cljs.core.Keyword(null,"solo","solo",-316350075));
var solo_mode = cljs.core.atom.call(null,new cljs.core.Keyword(null,"mute","mute",1151223646));
var change = cljs.core.async.chan.call(null);
var changed = ((function (cs,solo_modes,attrs,solo_mode,change){
return (function (){
return cljs.core.async.put_BANG_.call(null,change,true);
});})(cs,solo_modes,attrs,solo_mode,change))
;
var pick = ((function (cs,solo_modes,attrs,solo_mode,change,changed){
return (function (attr,chs){
return cljs.core.reduce_kv.call(null,((function (cs,solo_modes,attrs,solo_mode,change,changed){
return (function (ret,c,v){
if(cljs.core.truth_(attr.call(null,v))){
return cljs.core.conj.call(null,ret,c);
} else {
return ret;
}
});})(cs,solo_modes,attrs,solo_mode,change,changed))
,cljs.core.PersistentHashSet.EMPTY,chs);
});})(cs,solo_modes,attrs,solo_mode,change,changed))
;
var calc_state = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick){
return (function (){
var chs = cljs.core.deref.call(null,cs);
var mode = cljs.core.deref.call(null,solo_mode);
var solos = pick.call(null,new cljs.core.Keyword(null,"solo","solo",-316350075),chs);
var pauses = pick.call(null,new cljs.core.Keyword(null,"pause","pause",-2095325672),chs);
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"solos","solos",1441458643),solos,new cljs.core.Keyword(null,"mutes","mutes",1068806309),pick.call(null,new cljs.core.Keyword(null,"mute","mute",1151223646),chs),new cljs.core.Keyword(null,"reads","reads",-1215067361),cljs.core.conj.call(null,(((cljs.core._EQ_.call(null,mode,new cljs.core.Keyword(null,"pause","pause",-2095325672))) && (!(cljs.core.empty_QMARK_.call(null,solos))))?cljs.core.vec.call(null,solos):cljs.core.vec.call(null,cljs.core.remove.call(null,pauses,cljs.core.keys.call(null,chs)))),change)], null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick))
;
var m = (function (){
if(typeof cljs.core.async.t_cljs$core$async20483 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mix}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async20483 = (function (change,mix,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,meta20484){
this.change = change;
this.mix = mix;
this.solo_mode = solo_mode;
this.pick = pick;
this.cs = cs;
this.calc_state = calc_state;
this.out = out;
this.changed = changed;
this.solo_modes = solo_modes;
this.attrs = attrs;
this.meta20484 = meta20484;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_20485,meta20484__$1){
var self__ = this;
var _20485__$1 = this;
return (new cljs.core.async.t_cljs$core$async20483(self__.change,self__.mix,self__.solo_mode,self__.pick,self__.cs,self__.calc_state,self__.out,self__.changed,self__.solo_modes,self__.attrs,meta20484__$1));
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_20485){
var self__ = this;
var _20485__$1 = this;
return self__.meta20484;
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mux$ = true;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_){
var self__ = this;
var ___$1 = this;
return self__.out;
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$ = true;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$admix_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,ch){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.assoc,ch,cljs.core.PersistentArrayMap.EMPTY);

return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$unmix_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,ch){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.dissoc,ch);

return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$unmix_all_STAR_$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_.call(null,self__.cs,cljs.core.PersistentArrayMap.EMPTY);

return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$toggle_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,state_map){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.partial.call(null,cljs.core.merge_with,cljs.core.merge),state_map);

return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.prototype.cljs$core$async$Mix$solo_mode_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,mode){
var self__ = this;
var ___$1 = this;
if(cljs.core.truth_(self__.solo_modes.call(null,mode))){
} else {
throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str([cljs.core.str("mode must be one of: "),cljs.core.str(self__.solo_modes)].join('')),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,cljs.core.list(new cljs.core.Symbol(null,"solo-modes","solo-modes",882180540,null),new cljs.core.Symbol(null,"mode","mode",-2000032078,null))))].join('')));
}

cljs.core.reset_BANG_.call(null,self__.solo_mode,mode);

return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.getBasis = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (){
return new cljs.core.PersistentVector(null, 11, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"change","change",477485025,null),cljs.core.with_meta(new cljs.core.Symbol(null,"mix","mix",2121373763,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"out","out",729986010,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Creates and returns a mix of one or more input channels which will\n  be put on the supplied out channel. Input sources can be added to\n  the mix with 'admix', and removed with 'unmix'. A mix supports\n  soloing, muting and pausing multiple inputs atomically using\n  'toggle', and can solo using either muting or pausing as determined\n  by 'solo-mode'.\n\n  Each channel can have zero or more boolean modes set via 'toggle':\n\n  :solo - when true, only this (ond other soloed) channel(s) will appear\n          in the mix output channel. :mute and :pause states of soloed\n          channels are ignored. If solo-mode is :mute, non-soloed\n          channels are muted, if :pause, non-soloed channels are\n          paused.\n\n  :mute - muted channels will have their contents consumed but not included in the mix\n  :pause - paused channels will not have their contents consumed (and thus also not included in the mix)\n"], null)),new cljs.core.Symbol(null,"solo-mode","solo-mode",2031788074,null),new cljs.core.Symbol(null,"pick","pick",1300068175,null),new cljs.core.Symbol(null,"cs","cs",-117024463,null),new cljs.core.Symbol(null,"calc-state","calc-state",-349968968,null),new cljs.core.Symbol(null,"out","out",729986010,null),new cljs.core.Symbol(null,"changed","changed",-2083710852,null),new cljs.core.Symbol(null,"solo-modes","solo-modes",882180540,null),new cljs.core.Symbol(null,"attrs","attrs",-450137186,null),new cljs.core.Symbol(null,"meta20484","meta20484",-566408226,null)], null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.t_cljs$core$async20483.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async20483.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async20483";

cljs.core.async.t_cljs$core$async20483.cljs$lang$ctorPrWriter = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async20483");
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

cljs.core.async.__GT_t_cljs$core$async20483 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function cljs$core$async$mix_$___GT_t_cljs$core$async20483(change__$1,mix__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta20484){
return (new cljs.core.async.t_cljs$core$async20483(change__$1,mix__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta20484));
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;

}

return (new cljs.core.async.t_cljs$core$async20483(change,cljs$core$async$mix,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var c__18933__auto___20646 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function (state_20583){
var state_val_20584 = (state_20583[(1)]);
if((state_val_20584 === (7))){
var inst_20501 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
var statearr_20585_20647 = state_20583__$1;
(statearr_20585_20647[(2)] = inst_20501);

(statearr_20585_20647[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (20))){
var inst_20513 = (state_20583[(7)]);
var state_20583__$1 = state_20583;
var statearr_20586_20648 = state_20583__$1;
(statearr_20586_20648[(2)] = inst_20513);

(statearr_20586_20648[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (27))){
var state_20583__$1 = state_20583;
var statearr_20587_20649 = state_20583__$1;
(statearr_20587_20649[(2)] = null);

(statearr_20587_20649[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (1))){
var inst_20489 = (state_20583[(8)]);
var inst_20489__$1 = calc_state.call(null);
var inst_20491 = (inst_20489__$1 == null);
var inst_20492 = cljs.core.not.call(null,inst_20491);
var state_20583__$1 = (function (){var statearr_20588 = state_20583;
(statearr_20588[(8)] = inst_20489__$1);

return statearr_20588;
})();
if(inst_20492){
var statearr_20589_20650 = state_20583__$1;
(statearr_20589_20650[(1)] = (2));

} else {
var statearr_20590_20651 = state_20583__$1;
(statearr_20590_20651[(1)] = (3));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (24))){
var inst_20536 = (state_20583[(9)]);
var inst_20543 = (state_20583[(10)]);
var inst_20557 = (state_20583[(11)]);
var inst_20557__$1 = inst_20536.call(null,inst_20543);
var state_20583__$1 = (function (){var statearr_20591 = state_20583;
(statearr_20591[(11)] = inst_20557__$1);

return statearr_20591;
})();
if(cljs.core.truth_(inst_20557__$1)){
var statearr_20592_20652 = state_20583__$1;
(statearr_20592_20652[(1)] = (29));

} else {
var statearr_20593_20653 = state_20583__$1;
(statearr_20593_20653[(1)] = (30));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (4))){
var inst_20504 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20504)){
var statearr_20594_20654 = state_20583__$1;
(statearr_20594_20654[(1)] = (8));

} else {
var statearr_20595_20655 = state_20583__$1;
(statearr_20595_20655[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (15))){
var inst_20530 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20530)){
var statearr_20596_20656 = state_20583__$1;
(statearr_20596_20656[(1)] = (19));

} else {
var statearr_20597_20657 = state_20583__$1;
(statearr_20597_20657[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (21))){
var inst_20535 = (state_20583[(12)]);
var inst_20535__$1 = (state_20583[(2)]);
var inst_20536 = cljs.core.get.call(null,inst_20535__$1,new cljs.core.Keyword(null,"solos","solos",1441458643));
var inst_20537 = cljs.core.get.call(null,inst_20535__$1,new cljs.core.Keyword(null,"mutes","mutes",1068806309));
var inst_20538 = cljs.core.get.call(null,inst_20535__$1,new cljs.core.Keyword(null,"reads","reads",-1215067361));
var state_20583__$1 = (function (){var statearr_20598 = state_20583;
(statearr_20598[(9)] = inst_20536);

(statearr_20598[(13)] = inst_20537);

(statearr_20598[(12)] = inst_20535__$1);

return statearr_20598;
})();
return cljs.core.async.ioc_alts_BANG_.call(null,state_20583__$1,(22),inst_20538);
} else {
if((state_val_20584 === (31))){
var inst_20565 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20565)){
var statearr_20599_20658 = state_20583__$1;
(statearr_20599_20658[(1)] = (32));

} else {
var statearr_20600_20659 = state_20583__$1;
(statearr_20600_20659[(1)] = (33));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (32))){
var inst_20542 = (state_20583[(14)]);
var state_20583__$1 = state_20583;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_20583__$1,(35),out,inst_20542);
} else {
if((state_val_20584 === (33))){
var inst_20535 = (state_20583[(12)]);
var inst_20513 = inst_20535;
var state_20583__$1 = (function (){var statearr_20601 = state_20583;
(statearr_20601[(7)] = inst_20513);

return statearr_20601;
})();
var statearr_20602_20660 = state_20583__$1;
(statearr_20602_20660[(2)] = null);

(statearr_20602_20660[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (13))){
var inst_20513 = (state_20583[(7)]);
var inst_20520 = inst_20513.cljs$lang$protocol_mask$partition0$;
var inst_20521 = (inst_20520 & (64));
var inst_20522 = inst_20513.cljs$core$ISeq$;
var inst_20523 = (inst_20521) || (inst_20522);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20523)){
var statearr_20603_20661 = state_20583__$1;
(statearr_20603_20661[(1)] = (16));

} else {
var statearr_20604_20662 = state_20583__$1;
(statearr_20604_20662[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (22))){
var inst_20542 = (state_20583[(14)]);
var inst_20543 = (state_20583[(10)]);
var inst_20541 = (state_20583[(2)]);
var inst_20542__$1 = cljs.core.nth.call(null,inst_20541,(0),null);
var inst_20543__$1 = cljs.core.nth.call(null,inst_20541,(1),null);
var inst_20544 = (inst_20542__$1 == null);
var inst_20545 = cljs.core._EQ_.call(null,inst_20543__$1,change);
var inst_20546 = (inst_20544) || (inst_20545);
var state_20583__$1 = (function (){var statearr_20605 = state_20583;
(statearr_20605[(14)] = inst_20542__$1);

(statearr_20605[(10)] = inst_20543__$1);

return statearr_20605;
})();
if(cljs.core.truth_(inst_20546)){
var statearr_20606_20663 = state_20583__$1;
(statearr_20606_20663[(1)] = (23));

} else {
var statearr_20607_20664 = state_20583__$1;
(statearr_20607_20664[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (36))){
var inst_20535 = (state_20583[(12)]);
var inst_20513 = inst_20535;
var state_20583__$1 = (function (){var statearr_20608 = state_20583;
(statearr_20608[(7)] = inst_20513);

return statearr_20608;
})();
var statearr_20609_20665 = state_20583__$1;
(statearr_20609_20665[(2)] = null);

(statearr_20609_20665[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (29))){
var inst_20557 = (state_20583[(11)]);
var state_20583__$1 = state_20583;
var statearr_20610_20666 = state_20583__$1;
(statearr_20610_20666[(2)] = inst_20557);

(statearr_20610_20666[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (6))){
var state_20583__$1 = state_20583;
var statearr_20611_20667 = state_20583__$1;
(statearr_20611_20667[(2)] = false);

(statearr_20611_20667[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (28))){
var inst_20553 = (state_20583[(2)]);
var inst_20554 = calc_state.call(null);
var inst_20513 = inst_20554;
var state_20583__$1 = (function (){var statearr_20612 = state_20583;
(statearr_20612[(7)] = inst_20513);

(statearr_20612[(15)] = inst_20553);

return statearr_20612;
})();
var statearr_20613_20668 = state_20583__$1;
(statearr_20613_20668[(2)] = null);

(statearr_20613_20668[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (25))){
var inst_20579 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
var statearr_20614_20669 = state_20583__$1;
(statearr_20614_20669[(2)] = inst_20579);

(statearr_20614_20669[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (34))){
var inst_20577 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
var statearr_20615_20670 = state_20583__$1;
(statearr_20615_20670[(2)] = inst_20577);

(statearr_20615_20670[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (17))){
var state_20583__$1 = state_20583;
var statearr_20616_20671 = state_20583__$1;
(statearr_20616_20671[(2)] = false);

(statearr_20616_20671[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (3))){
var state_20583__$1 = state_20583;
var statearr_20617_20672 = state_20583__$1;
(statearr_20617_20672[(2)] = false);

(statearr_20617_20672[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (12))){
var inst_20581 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_20583__$1,inst_20581);
} else {
if((state_val_20584 === (2))){
var inst_20489 = (state_20583[(8)]);
var inst_20494 = inst_20489.cljs$lang$protocol_mask$partition0$;
var inst_20495 = (inst_20494 & (64));
var inst_20496 = inst_20489.cljs$core$ISeq$;
var inst_20497 = (inst_20495) || (inst_20496);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20497)){
var statearr_20618_20673 = state_20583__$1;
(statearr_20618_20673[(1)] = (5));

} else {
var statearr_20619_20674 = state_20583__$1;
(statearr_20619_20674[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (23))){
var inst_20542 = (state_20583[(14)]);
var inst_20548 = (inst_20542 == null);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20548)){
var statearr_20620_20675 = state_20583__$1;
(statearr_20620_20675[(1)] = (26));

} else {
var statearr_20621_20676 = state_20583__$1;
(statearr_20621_20676[(1)] = (27));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (35))){
var inst_20568 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
if(cljs.core.truth_(inst_20568)){
var statearr_20622_20677 = state_20583__$1;
(statearr_20622_20677[(1)] = (36));

} else {
var statearr_20623_20678 = state_20583__$1;
(statearr_20623_20678[(1)] = (37));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (19))){
var inst_20513 = (state_20583[(7)]);
var inst_20532 = cljs.core.apply.call(null,cljs.core.hash_map,inst_20513);
var state_20583__$1 = state_20583;
var statearr_20624_20679 = state_20583__$1;
(statearr_20624_20679[(2)] = inst_20532);

(statearr_20624_20679[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (11))){
var inst_20513 = (state_20583[(7)]);
var inst_20517 = (inst_20513 == null);
var inst_20518 = cljs.core.not.call(null,inst_20517);
var state_20583__$1 = state_20583;
if(inst_20518){
var statearr_20625_20680 = state_20583__$1;
(statearr_20625_20680[(1)] = (13));

} else {
var statearr_20626_20681 = state_20583__$1;
(statearr_20626_20681[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (9))){
var inst_20489 = (state_20583[(8)]);
var state_20583__$1 = state_20583;
var statearr_20627_20682 = state_20583__$1;
(statearr_20627_20682[(2)] = inst_20489);

(statearr_20627_20682[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (5))){
var state_20583__$1 = state_20583;
var statearr_20628_20683 = state_20583__$1;
(statearr_20628_20683[(2)] = true);

(statearr_20628_20683[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (14))){
var state_20583__$1 = state_20583;
var statearr_20629_20684 = state_20583__$1;
(statearr_20629_20684[(2)] = false);

(statearr_20629_20684[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (26))){
var inst_20543 = (state_20583[(10)]);
var inst_20550 = cljs.core.swap_BANG_.call(null,cs,cljs.core.dissoc,inst_20543);
var state_20583__$1 = state_20583;
var statearr_20630_20685 = state_20583__$1;
(statearr_20630_20685[(2)] = inst_20550);

(statearr_20630_20685[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (16))){
var state_20583__$1 = state_20583;
var statearr_20631_20686 = state_20583__$1;
(statearr_20631_20686[(2)] = true);

(statearr_20631_20686[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (38))){
var inst_20573 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
var statearr_20632_20687 = state_20583__$1;
(statearr_20632_20687[(2)] = inst_20573);

(statearr_20632_20687[(1)] = (34));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (30))){
var inst_20536 = (state_20583[(9)]);
var inst_20543 = (state_20583[(10)]);
var inst_20537 = (state_20583[(13)]);
var inst_20560 = cljs.core.empty_QMARK_.call(null,inst_20536);
var inst_20561 = inst_20537.call(null,inst_20543);
var inst_20562 = cljs.core.not.call(null,inst_20561);
var inst_20563 = (inst_20560) && (inst_20562);
var state_20583__$1 = state_20583;
var statearr_20633_20688 = state_20583__$1;
(statearr_20633_20688[(2)] = inst_20563);

(statearr_20633_20688[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (10))){
var inst_20489 = (state_20583[(8)]);
var inst_20509 = (state_20583[(2)]);
var inst_20510 = cljs.core.get.call(null,inst_20509,new cljs.core.Keyword(null,"solos","solos",1441458643));
var inst_20511 = cljs.core.get.call(null,inst_20509,new cljs.core.Keyword(null,"mutes","mutes",1068806309));
var inst_20512 = cljs.core.get.call(null,inst_20509,new cljs.core.Keyword(null,"reads","reads",-1215067361));
var inst_20513 = inst_20489;
var state_20583__$1 = (function (){var statearr_20634 = state_20583;
(statearr_20634[(16)] = inst_20510);

(statearr_20634[(7)] = inst_20513);

(statearr_20634[(17)] = inst_20511);

(statearr_20634[(18)] = inst_20512);

return statearr_20634;
})();
var statearr_20635_20689 = state_20583__$1;
(statearr_20635_20689[(2)] = null);

(statearr_20635_20689[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (18))){
var inst_20527 = (state_20583[(2)]);
var state_20583__$1 = state_20583;
var statearr_20636_20690 = state_20583__$1;
(statearr_20636_20690[(2)] = inst_20527);

(statearr_20636_20690[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (37))){
var state_20583__$1 = state_20583;
var statearr_20637_20691 = state_20583__$1;
(statearr_20637_20691[(2)] = null);

(statearr_20637_20691[(1)] = (38));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20584 === (8))){
var inst_20489 = (state_20583[(8)]);
var inst_20506 = cljs.core.apply.call(null,cljs.core.hash_map,inst_20489);
var state_20583__$1 = state_20583;
var statearr_20638_20692 = state_20583__$1;
(statearr_20638_20692[(2)] = inst_20506);

(statearr_20638_20692[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
;
return ((function (switch__18821__auto__,c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function() {
var cljs$core$async$mix_$_state_machine__18822__auto__ = null;
var cljs$core$async$mix_$_state_machine__18822__auto____0 = (function (){
var statearr_20642 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_20642[(0)] = cljs$core$async$mix_$_state_machine__18822__auto__);

(statearr_20642[(1)] = (1));

return statearr_20642;
});
var cljs$core$async$mix_$_state_machine__18822__auto____1 = (function (state_20583){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_20583);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e20643){if((e20643 instanceof Object)){
var ex__18825__auto__ = e20643;
var statearr_20644_20693 = state_20583;
(statearr_20644_20693[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20583);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20643;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20694 = state_20583;
state_20583 = G__20694;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$mix_$_state_machine__18822__auto__ = function(state_20583){
switch(arguments.length){
case 0:
return cljs$core$async$mix_$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$mix_$_state_machine__18822__auto____1.call(this,state_20583);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mix_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mix_$_state_machine__18822__auto____0;
cljs$core$async$mix_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mix_$_state_machine__18822__auto____1;
return cljs$core$async$mix_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
})();
var state__18935__auto__ = (function (){var statearr_20645 = f__18934__auto__.call(null);
(statearr_20645[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___20646);

return statearr_20645;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___20646,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
);


return m;
});
/**
 * Adds ch as an input to the mix
 */
cljs.core.async.admix = (function cljs$core$async$admix(mix,ch){
return cljs.core.async.admix_STAR_.call(null,mix,ch);
});
/**
 * Removes ch as an input to the mix
 */
cljs.core.async.unmix = (function cljs$core$async$unmix(mix,ch){
return cljs.core.async.unmix_STAR_.call(null,mix,ch);
});
/**
 * removes all inputs from the mix
 */
cljs.core.async.unmix_all = (function cljs$core$async$unmix_all(mix){
return cljs.core.async.unmix_all_STAR_.call(null,mix);
});
/**
 * Atomically sets the state(s) of one or more channels in a mix. The
 *   state map is a map of channels -> channel-state-map. A
 *   channel-state-map is a map of attrs -> boolean, where attr is one or
 *   more of :mute, :pause or :solo. Any states supplied are merged with
 *   the current state.
 * 
 *   Note that channels can be added to a mix via toggle, which can be
 *   used to add channels in a particular (e.g. paused) state.
 */
cljs.core.async.toggle = (function cljs$core$async$toggle(mix,state_map){
return cljs.core.async.toggle_STAR_.call(null,mix,state_map);
});
/**
 * Sets the solo mode of the mix. mode must be one of :mute or :pause
 */
cljs.core.async.solo_mode = (function cljs$core$async$solo_mode(mix,mode){
return cljs.core.async.solo_mode_STAR_.call(null,mix,mode);
});

/**
 * @interface
 */
cljs.core.async.Pub = function(){};

cljs.core.async.sub_STAR_ = (function cljs$core$async$sub_STAR_(p,v,ch,close_QMARK_){
if((!((p == null))) && (!((p.cljs$core$async$Pub$sub_STAR_$arity$4 == null)))){
return p.cljs$core$async$Pub$sub_STAR_$arity$4(p,v,ch,close_QMARK_);
} else {
var x__17421__auto__ = (((p == null))?null:p);
var m__17422__auto__ = (cljs.core.async.sub_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,p,v,ch,close_QMARK_);
} else {
var m__17422__auto____$1 = (cljs.core.async.sub_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,p,v,ch,close_QMARK_);
} else {
throw cljs.core.missing_protocol.call(null,"Pub.sub*",p);
}
}
}
});

cljs.core.async.unsub_STAR_ = (function cljs$core$async$unsub_STAR_(p,v,ch){
if((!((p == null))) && (!((p.cljs$core$async$Pub$unsub_STAR_$arity$3 == null)))){
return p.cljs$core$async$Pub$unsub_STAR_$arity$3(p,v,ch);
} else {
var x__17421__auto__ = (((p == null))?null:p);
var m__17422__auto__ = (cljs.core.async.unsub_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,p,v,ch);
} else {
var m__17422__auto____$1 = (cljs.core.async.unsub_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,p,v,ch);
} else {
throw cljs.core.missing_protocol.call(null,"Pub.unsub*",p);
}
}
}
});

cljs.core.async.unsub_all_STAR_ = (function cljs$core$async$unsub_all_STAR_(var_args){
var args20695 = [];
var len__17824__auto___20698 = arguments.length;
var i__17825__auto___20699 = (0);
while(true){
if((i__17825__auto___20699 < len__17824__auto___20698)){
args20695.push((arguments[i__17825__auto___20699]));

var G__20700 = (i__17825__auto___20699 + (1));
i__17825__auto___20699 = G__20700;
continue;
} else {
}
break;
}

var G__20697 = args20695.length;
switch (G__20697) {
case 1:
return cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20695.length)].join('')));

}
});

cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$1 = (function (p){
if((!((p == null))) && (!((p.cljs$core$async$Pub$unsub_all_STAR_$arity$1 == null)))){
return p.cljs$core$async$Pub$unsub_all_STAR_$arity$1(p);
} else {
var x__17421__auto__ = (((p == null))?null:p);
var m__17422__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,p);
} else {
var m__17422__auto____$1 = (cljs.core.async.unsub_all_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,p);
} else {
throw cljs.core.missing_protocol.call(null,"Pub.unsub-all*",p);
}
}
}
});

cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$2 = (function (p,v){
if((!((p == null))) && (!((p.cljs$core$async$Pub$unsub_all_STAR_$arity$2 == null)))){
return p.cljs$core$async$Pub$unsub_all_STAR_$arity$2(p,v);
} else {
var x__17421__auto__ = (((p == null))?null:p);
var m__17422__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,p,v);
} else {
var m__17422__auto____$1 = (cljs.core.async.unsub_all_STAR_["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,p,v);
} else {
throw cljs.core.missing_protocol.call(null,"Pub.unsub-all*",p);
}
}
}
});

cljs.core.async.unsub_all_STAR_.cljs$lang$maxFixedArity = 2;

/**
 * Creates and returns a pub(lication) of the supplied channel,
 *   partitioned into topics by the topic-fn. topic-fn will be applied to
 *   each value on the channel and the result will determine the 'topic'
 *   on which that value will be put. Channels can be subscribed to
 *   receive copies of topics using 'sub', and unsubscribed using
 *   'unsub'. Each topic will be handled by an internal mult on a
 *   dedicated channel. By default these internal channels are
 *   unbuffered, but a buf-fn can be supplied which, given a topic,
 *   creates a buffer with desired properties.
 * 
 *   Each item is distributed to all subs in parallel and synchronously,
 *   i.e. each sub must accept before the next item is distributed. Use
 *   buffering/windowing to prevent slow subs from holding up the pub.
 * 
 *   Items received when there are no matching subs get dropped.
 * 
 *   Note that if buf-fns are used then each topic is handled
 *   asynchronously, i.e. if a channel is subscribed to more than one
 *   topic it should not expect them to be interleaved identically with
 *   the source.
 */
cljs.core.async.pub = (function cljs$core$async$pub(var_args){
var args20703 = [];
var len__17824__auto___20828 = arguments.length;
var i__17825__auto___20829 = (0);
while(true){
if((i__17825__auto___20829 < len__17824__auto___20828)){
args20703.push((arguments[i__17825__auto___20829]));

var G__20830 = (i__17825__auto___20829 + (1));
i__17825__auto___20829 = G__20830;
continue;
} else {
}
break;
}

var G__20705 = args20703.length;
switch (G__20705) {
case 2:
return cljs.core.async.pub.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.pub.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20703.length)].join('')));

}
});

cljs.core.async.pub.cljs$core$IFn$_invoke$arity$2 = (function (ch,topic_fn){
return cljs.core.async.pub.call(null,ch,topic_fn,cljs.core.constantly.call(null,null));
});

cljs.core.async.pub.cljs$core$IFn$_invoke$arity$3 = (function (ch,topic_fn,buf_fn){
var mults = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var ensure_mult = ((function (mults){
return (function (topic){
var or__16766__auto__ = cljs.core.get.call(null,cljs.core.deref.call(null,mults),topic);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core.get.call(null,cljs.core.swap_BANG_.call(null,mults,((function (or__16766__auto__,mults){
return (function (p1__20702_SHARP_){
if(cljs.core.truth_(p1__20702_SHARP_.call(null,topic))){
return p1__20702_SHARP_;
} else {
return cljs.core.assoc.call(null,p1__20702_SHARP_,topic,cljs.core.async.mult.call(null,cljs.core.async.chan.call(null,buf_fn.call(null,topic))));
}
});})(or__16766__auto__,mults))
),topic);
}
});})(mults))
;
var p = (function (){
if(typeof cljs.core.async.t_cljs$core$async20706 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.Pub}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async20706 = (function (ch,topic_fn,buf_fn,mults,ensure_mult,meta20707){
this.ch = ch;
this.topic_fn = topic_fn;
this.buf_fn = buf_fn;
this.mults = mults;
this.ensure_mult = ensure_mult;
this.meta20707 = meta20707;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (mults,ensure_mult){
return (function (_20708,meta20707__$1){
var self__ = this;
var _20708__$1 = this;
return (new cljs.core.async.t_cljs$core$async20706(self__.ch,self__.topic_fn,self__.buf_fn,self__.mults,self__.ensure_mult,meta20707__$1));
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (mults,ensure_mult){
return (function (_20708){
var self__ = this;
var _20708__$1 = this;
return self__.meta20707;
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Mux$ = true;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (mults,ensure_mult){
return (function (_){
var self__ = this;
var ___$1 = this;
return self__.ch;
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Pub$ = true;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Pub$sub_STAR_$arity$4 = ((function (mults,ensure_mult){
return (function (p,topic,ch__$1,close_QMARK_){
var self__ = this;
var p__$1 = this;
var m = self__.ensure_mult.call(null,topic);
return cljs.core.async.tap.call(null,m,ch__$1,close_QMARK_);
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Pub$unsub_STAR_$arity$3 = ((function (mults,ensure_mult){
return (function (p,topic,ch__$1){
var self__ = this;
var p__$1 = this;
var temp__4425__auto__ = cljs.core.get.call(null,cljs.core.deref.call(null,self__.mults),topic);
if(cljs.core.truth_(temp__4425__auto__)){
var m = temp__4425__auto__;
return cljs.core.async.untap.call(null,m,ch__$1);
} else {
return null;
}
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$1 = ((function (mults,ensure_mult){
return (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.reset_BANG_.call(null,self__.mults,cljs.core.PersistentArrayMap.EMPTY);
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$2 = ((function (mults,ensure_mult){
return (function (_,topic){
var self__ = this;
var ___$1 = this;
return cljs.core.swap_BANG_.call(null,self__.mults,cljs.core.dissoc,topic);
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.getBasis = ((function (mults,ensure_mult){
return (function (){
return new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"topic-fn","topic-fn",-862449736,null),new cljs.core.Symbol(null,"buf-fn","buf-fn",-1200281591,null),new cljs.core.Symbol(null,"mults","mults",-461114485,null),new cljs.core.Symbol(null,"ensure-mult","ensure-mult",1796584816,null),new cljs.core.Symbol(null,"meta20707","meta20707",-1665894657,null)], null);
});})(mults,ensure_mult))
;

cljs.core.async.t_cljs$core$async20706.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async20706.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async20706";

cljs.core.async.t_cljs$core$async20706.cljs$lang$ctorPrWriter = ((function (mults,ensure_mult){
return (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async20706");
});})(mults,ensure_mult))
;

cljs.core.async.__GT_t_cljs$core$async20706 = ((function (mults,ensure_mult){
return (function cljs$core$async$__GT_t_cljs$core$async20706(ch__$1,topic_fn__$1,buf_fn__$1,mults__$1,ensure_mult__$1,meta20707){
return (new cljs.core.async.t_cljs$core$async20706(ch__$1,topic_fn__$1,buf_fn__$1,mults__$1,ensure_mult__$1,meta20707));
});})(mults,ensure_mult))
;

}

return (new cljs.core.async.t_cljs$core$async20706(ch,topic_fn,buf_fn,mults,ensure_mult,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var c__18933__auto___20832 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___20832,mults,ensure_mult,p){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___20832,mults,ensure_mult,p){
return (function (state_20780){
var state_val_20781 = (state_20780[(1)]);
if((state_val_20781 === (7))){
var inst_20776 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20782_20833 = state_20780__$1;
(statearr_20782_20833[(2)] = inst_20776);

(statearr_20782_20833[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (20))){
var state_20780__$1 = state_20780;
var statearr_20783_20834 = state_20780__$1;
(statearr_20783_20834[(2)] = null);

(statearr_20783_20834[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (1))){
var state_20780__$1 = state_20780;
var statearr_20784_20835 = state_20780__$1;
(statearr_20784_20835[(2)] = null);

(statearr_20784_20835[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (24))){
var inst_20759 = (state_20780[(7)]);
var inst_20768 = cljs.core.swap_BANG_.call(null,mults,cljs.core.dissoc,inst_20759);
var state_20780__$1 = state_20780;
var statearr_20785_20836 = state_20780__$1;
(statearr_20785_20836[(2)] = inst_20768);

(statearr_20785_20836[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (4))){
var inst_20711 = (state_20780[(8)]);
var inst_20711__$1 = (state_20780[(2)]);
var inst_20712 = (inst_20711__$1 == null);
var state_20780__$1 = (function (){var statearr_20786 = state_20780;
(statearr_20786[(8)] = inst_20711__$1);

return statearr_20786;
})();
if(cljs.core.truth_(inst_20712)){
var statearr_20787_20837 = state_20780__$1;
(statearr_20787_20837[(1)] = (5));

} else {
var statearr_20788_20838 = state_20780__$1;
(statearr_20788_20838[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (15))){
var inst_20753 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20789_20839 = state_20780__$1;
(statearr_20789_20839[(2)] = inst_20753);

(statearr_20789_20839[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (21))){
var inst_20773 = (state_20780[(2)]);
var state_20780__$1 = (function (){var statearr_20790 = state_20780;
(statearr_20790[(9)] = inst_20773);

return statearr_20790;
})();
var statearr_20791_20840 = state_20780__$1;
(statearr_20791_20840[(2)] = null);

(statearr_20791_20840[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (13))){
var inst_20735 = (state_20780[(10)]);
var inst_20737 = cljs.core.chunked_seq_QMARK_.call(null,inst_20735);
var state_20780__$1 = state_20780;
if(inst_20737){
var statearr_20792_20841 = state_20780__$1;
(statearr_20792_20841[(1)] = (16));

} else {
var statearr_20793_20842 = state_20780__$1;
(statearr_20793_20842[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (22))){
var inst_20765 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
if(cljs.core.truth_(inst_20765)){
var statearr_20794_20843 = state_20780__$1;
(statearr_20794_20843[(1)] = (23));

} else {
var statearr_20795_20844 = state_20780__$1;
(statearr_20795_20844[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (6))){
var inst_20759 = (state_20780[(7)]);
var inst_20711 = (state_20780[(8)]);
var inst_20761 = (state_20780[(11)]);
var inst_20759__$1 = topic_fn.call(null,inst_20711);
var inst_20760 = cljs.core.deref.call(null,mults);
var inst_20761__$1 = cljs.core.get.call(null,inst_20760,inst_20759__$1);
var state_20780__$1 = (function (){var statearr_20796 = state_20780;
(statearr_20796[(7)] = inst_20759__$1);

(statearr_20796[(11)] = inst_20761__$1);

return statearr_20796;
})();
if(cljs.core.truth_(inst_20761__$1)){
var statearr_20797_20845 = state_20780__$1;
(statearr_20797_20845[(1)] = (19));

} else {
var statearr_20798_20846 = state_20780__$1;
(statearr_20798_20846[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (25))){
var inst_20770 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20799_20847 = state_20780__$1;
(statearr_20799_20847[(2)] = inst_20770);

(statearr_20799_20847[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (17))){
var inst_20735 = (state_20780[(10)]);
var inst_20744 = cljs.core.first.call(null,inst_20735);
var inst_20745 = cljs.core.async.muxch_STAR_.call(null,inst_20744);
var inst_20746 = cljs.core.async.close_BANG_.call(null,inst_20745);
var inst_20747 = cljs.core.next.call(null,inst_20735);
var inst_20721 = inst_20747;
var inst_20722 = null;
var inst_20723 = (0);
var inst_20724 = (0);
var state_20780__$1 = (function (){var statearr_20800 = state_20780;
(statearr_20800[(12)] = inst_20723);

(statearr_20800[(13)] = inst_20746);

(statearr_20800[(14)] = inst_20721);

(statearr_20800[(15)] = inst_20722);

(statearr_20800[(16)] = inst_20724);

return statearr_20800;
})();
var statearr_20801_20848 = state_20780__$1;
(statearr_20801_20848[(2)] = null);

(statearr_20801_20848[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (3))){
var inst_20778 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_20780__$1,inst_20778);
} else {
if((state_val_20781 === (12))){
var inst_20755 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20802_20849 = state_20780__$1;
(statearr_20802_20849[(2)] = inst_20755);

(statearr_20802_20849[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (2))){
var state_20780__$1 = state_20780;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20780__$1,(4),ch);
} else {
if((state_val_20781 === (23))){
var state_20780__$1 = state_20780;
var statearr_20803_20850 = state_20780__$1;
(statearr_20803_20850[(2)] = null);

(statearr_20803_20850[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (19))){
var inst_20711 = (state_20780[(8)]);
var inst_20761 = (state_20780[(11)]);
var inst_20763 = cljs.core.async.muxch_STAR_.call(null,inst_20761);
var state_20780__$1 = state_20780;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_20780__$1,(22),inst_20763,inst_20711);
} else {
if((state_val_20781 === (11))){
var inst_20721 = (state_20780[(14)]);
var inst_20735 = (state_20780[(10)]);
var inst_20735__$1 = cljs.core.seq.call(null,inst_20721);
var state_20780__$1 = (function (){var statearr_20804 = state_20780;
(statearr_20804[(10)] = inst_20735__$1);

return statearr_20804;
})();
if(inst_20735__$1){
var statearr_20805_20851 = state_20780__$1;
(statearr_20805_20851[(1)] = (13));

} else {
var statearr_20806_20852 = state_20780__$1;
(statearr_20806_20852[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (9))){
var inst_20757 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20807_20853 = state_20780__$1;
(statearr_20807_20853[(2)] = inst_20757);

(statearr_20807_20853[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (5))){
var inst_20718 = cljs.core.deref.call(null,mults);
var inst_20719 = cljs.core.vals.call(null,inst_20718);
var inst_20720 = cljs.core.seq.call(null,inst_20719);
var inst_20721 = inst_20720;
var inst_20722 = null;
var inst_20723 = (0);
var inst_20724 = (0);
var state_20780__$1 = (function (){var statearr_20808 = state_20780;
(statearr_20808[(12)] = inst_20723);

(statearr_20808[(14)] = inst_20721);

(statearr_20808[(15)] = inst_20722);

(statearr_20808[(16)] = inst_20724);

return statearr_20808;
})();
var statearr_20809_20854 = state_20780__$1;
(statearr_20809_20854[(2)] = null);

(statearr_20809_20854[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (14))){
var state_20780__$1 = state_20780;
var statearr_20813_20855 = state_20780__$1;
(statearr_20813_20855[(2)] = null);

(statearr_20813_20855[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (16))){
var inst_20735 = (state_20780[(10)]);
var inst_20739 = cljs.core.chunk_first.call(null,inst_20735);
var inst_20740 = cljs.core.chunk_rest.call(null,inst_20735);
var inst_20741 = cljs.core.count.call(null,inst_20739);
var inst_20721 = inst_20740;
var inst_20722 = inst_20739;
var inst_20723 = inst_20741;
var inst_20724 = (0);
var state_20780__$1 = (function (){var statearr_20814 = state_20780;
(statearr_20814[(12)] = inst_20723);

(statearr_20814[(14)] = inst_20721);

(statearr_20814[(15)] = inst_20722);

(statearr_20814[(16)] = inst_20724);

return statearr_20814;
})();
var statearr_20815_20856 = state_20780__$1;
(statearr_20815_20856[(2)] = null);

(statearr_20815_20856[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (10))){
var inst_20723 = (state_20780[(12)]);
var inst_20721 = (state_20780[(14)]);
var inst_20722 = (state_20780[(15)]);
var inst_20724 = (state_20780[(16)]);
var inst_20729 = cljs.core._nth.call(null,inst_20722,inst_20724);
var inst_20730 = cljs.core.async.muxch_STAR_.call(null,inst_20729);
var inst_20731 = cljs.core.async.close_BANG_.call(null,inst_20730);
var inst_20732 = (inst_20724 + (1));
var tmp20810 = inst_20723;
var tmp20811 = inst_20721;
var tmp20812 = inst_20722;
var inst_20721__$1 = tmp20811;
var inst_20722__$1 = tmp20812;
var inst_20723__$1 = tmp20810;
var inst_20724__$1 = inst_20732;
var state_20780__$1 = (function (){var statearr_20816 = state_20780;
(statearr_20816[(17)] = inst_20731);

(statearr_20816[(12)] = inst_20723__$1);

(statearr_20816[(14)] = inst_20721__$1);

(statearr_20816[(15)] = inst_20722__$1);

(statearr_20816[(16)] = inst_20724__$1);

return statearr_20816;
})();
var statearr_20817_20857 = state_20780__$1;
(statearr_20817_20857[(2)] = null);

(statearr_20817_20857[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (18))){
var inst_20750 = (state_20780[(2)]);
var state_20780__$1 = state_20780;
var statearr_20818_20858 = state_20780__$1;
(statearr_20818_20858[(2)] = inst_20750);

(statearr_20818_20858[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20781 === (8))){
var inst_20723 = (state_20780[(12)]);
var inst_20724 = (state_20780[(16)]);
var inst_20726 = (inst_20724 < inst_20723);
var inst_20727 = inst_20726;
var state_20780__$1 = state_20780;
if(cljs.core.truth_(inst_20727)){
var statearr_20819_20859 = state_20780__$1;
(statearr_20819_20859[(1)] = (10));

} else {
var statearr_20820_20860 = state_20780__$1;
(statearr_20820_20860[(1)] = (11));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___20832,mults,ensure_mult,p))
;
return ((function (switch__18821__auto__,c__18933__auto___20832,mults,ensure_mult,p){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_20824 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_20824[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_20824[(1)] = (1));

return statearr_20824;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_20780){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_20780);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e20825){if((e20825 instanceof Object)){
var ex__18825__auto__ = e20825;
var statearr_20826_20861 = state_20780;
(statearr_20826_20861[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20780);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20825;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20862 = state_20780;
state_20780 = G__20862;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_20780){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_20780);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___20832,mults,ensure_mult,p))
})();
var state__18935__auto__ = (function (){var statearr_20827 = f__18934__auto__.call(null);
(statearr_20827[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___20832);

return statearr_20827;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___20832,mults,ensure_mult,p))
);


return p;
});

cljs.core.async.pub.cljs$lang$maxFixedArity = 3;
/**
 * Subscribes a channel to a topic of a pub.
 * 
 *   By default the channel will be closed when the source closes,
 *   but can be determined by the close? parameter.
 */
cljs.core.async.sub = (function cljs$core$async$sub(var_args){
var args20863 = [];
var len__17824__auto___20866 = arguments.length;
var i__17825__auto___20867 = (0);
while(true){
if((i__17825__auto___20867 < len__17824__auto___20866)){
args20863.push((arguments[i__17825__auto___20867]));

var G__20868 = (i__17825__auto___20867 + (1));
i__17825__auto___20867 = G__20868;
continue;
} else {
}
break;
}

var G__20865 = args20863.length;
switch (G__20865) {
case 3:
return cljs.core.async.sub.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return cljs.core.async.sub.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20863.length)].join('')));

}
});

cljs.core.async.sub.cljs$core$IFn$_invoke$arity$3 = (function (p,topic,ch){
return cljs.core.async.sub.call(null,p,topic,ch,true);
});

cljs.core.async.sub.cljs$core$IFn$_invoke$arity$4 = (function (p,topic,ch,close_QMARK_){
return cljs.core.async.sub_STAR_.call(null,p,topic,ch,close_QMARK_);
});

cljs.core.async.sub.cljs$lang$maxFixedArity = 4;
/**
 * Unsubscribes a channel from a topic of a pub
 */
cljs.core.async.unsub = (function cljs$core$async$unsub(p,topic,ch){
return cljs.core.async.unsub_STAR_.call(null,p,topic,ch);
});
/**
 * Unsubscribes all channels from a pub, or a topic of a pub
 */
cljs.core.async.unsub_all = (function cljs$core$async$unsub_all(var_args){
var args20870 = [];
var len__17824__auto___20873 = arguments.length;
var i__17825__auto___20874 = (0);
while(true){
if((i__17825__auto___20874 < len__17824__auto___20873)){
args20870.push((arguments[i__17825__auto___20874]));

var G__20875 = (i__17825__auto___20874 + (1));
i__17825__auto___20874 = G__20875;
continue;
} else {
}
break;
}

var G__20872 = args20870.length;
switch (G__20872) {
case 1:
return cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20870.length)].join('')));

}
});

cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$1 = (function (p){
return cljs.core.async.unsub_all_STAR_.call(null,p);
});

cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$2 = (function (p,topic){
return cljs.core.async.unsub_all_STAR_.call(null,p,topic);
});

cljs.core.async.unsub_all.cljs$lang$maxFixedArity = 2;
/**
 * Takes a function and a collection of source channels, and returns a
 *   channel which contains the values produced by applying f to the set
 *   of first items taken from each source channel, followed by applying
 *   f to the set of second items from each channel, until any one of the
 *   channels is closed, at which point the output channel will be
 *   closed. The returned channel will be unbuffered by default, or a
 *   buf-or-n can be supplied
 */
cljs.core.async.map = (function cljs$core$async$map(var_args){
var args20877 = [];
var len__17824__auto___20948 = arguments.length;
var i__17825__auto___20949 = (0);
while(true){
if((i__17825__auto___20949 < len__17824__auto___20948)){
args20877.push((arguments[i__17825__auto___20949]));

var G__20950 = (i__17825__auto___20949 + (1));
i__17825__auto___20949 = G__20950;
continue;
} else {
}
break;
}

var G__20879 = args20877.length;
switch (G__20879) {
case 2:
return cljs.core.async.map.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.map.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20877.length)].join('')));

}
});

cljs.core.async.map.cljs$core$IFn$_invoke$arity$2 = (function (f,chs){
return cljs.core.async.map.call(null,f,chs,null);
});

cljs.core.async.map.cljs$core$IFn$_invoke$arity$3 = (function (f,chs,buf_or_n){
var chs__$1 = cljs.core.vec.call(null,chs);
var out = cljs.core.async.chan.call(null,buf_or_n);
var cnt = cljs.core.count.call(null,chs__$1);
var rets = cljs.core.object_array.call(null,cnt);
var dchan = cljs.core.async.chan.call(null,(1));
var dctr = cljs.core.atom.call(null,null);
var done = cljs.core.mapv.call(null,((function (chs__$1,out,cnt,rets,dchan,dctr){
return (function (i){
return ((function (chs__$1,out,cnt,rets,dchan,dctr){
return (function (ret){
(rets[i] = ret);

if((cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec) === (0))){
return cljs.core.async.put_BANG_.call(null,dchan,rets.slice((0)));
} else {
return null;
}
});
;})(chs__$1,out,cnt,rets,dchan,dctr))
});})(chs__$1,out,cnt,rets,dchan,dctr))
,cljs.core.range.call(null,cnt));
var c__18933__auto___20952 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function (state_20918){
var state_val_20919 = (state_20918[(1)]);
if((state_val_20919 === (7))){
var state_20918__$1 = state_20918;
var statearr_20920_20953 = state_20918__$1;
(statearr_20920_20953[(2)] = null);

(statearr_20920_20953[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (1))){
var state_20918__$1 = state_20918;
var statearr_20921_20954 = state_20918__$1;
(statearr_20921_20954[(2)] = null);

(statearr_20921_20954[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (4))){
var inst_20882 = (state_20918[(7)]);
var inst_20884 = (inst_20882 < cnt);
var state_20918__$1 = state_20918;
if(cljs.core.truth_(inst_20884)){
var statearr_20922_20955 = state_20918__$1;
(statearr_20922_20955[(1)] = (6));

} else {
var statearr_20923_20956 = state_20918__$1;
(statearr_20923_20956[(1)] = (7));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (15))){
var inst_20914 = (state_20918[(2)]);
var state_20918__$1 = state_20918;
var statearr_20924_20957 = state_20918__$1;
(statearr_20924_20957[(2)] = inst_20914);

(statearr_20924_20957[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (13))){
var inst_20907 = cljs.core.async.close_BANG_.call(null,out);
var state_20918__$1 = state_20918;
var statearr_20925_20958 = state_20918__$1;
(statearr_20925_20958[(2)] = inst_20907);

(statearr_20925_20958[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (6))){
var state_20918__$1 = state_20918;
var statearr_20926_20959 = state_20918__$1;
(statearr_20926_20959[(2)] = null);

(statearr_20926_20959[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (3))){
var inst_20916 = (state_20918[(2)]);
var state_20918__$1 = state_20918;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_20918__$1,inst_20916);
} else {
if((state_val_20919 === (12))){
var inst_20904 = (state_20918[(8)]);
var inst_20904__$1 = (state_20918[(2)]);
var inst_20905 = cljs.core.some.call(null,cljs.core.nil_QMARK_,inst_20904__$1);
var state_20918__$1 = (function (){var statearr_20927 = state_20918;
(statearr_20927[(8)] = inst_20904__$1);

return statearr_20927;
})();
if(cljs.core.truth_(inst_20905)){
var statearr_20928_20960 = state_20918__$1;
(statearr_20928_20960[(1)] = (13));

} else {
var statearr_20929_20961 = state_20918__$1;
(statearr_20929_20961[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (2))){
var inst_20881 = cljs.core.reset_BANG_.call(null,dctr,cnt);
var inst_20882 = (0);
var state_20918__$1 = (function (){var statearr_20930 = state_20918;
(statearr_20930[(9)] = inst_20881);

(statearr_20930[(7)] = inst_20882);

return statearr_20930;
})();
var statearr_20931_20962 = state_20918__$1;
(statearr_20931_20962[(2)] = null);

(statearr_20931_20962[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (11))){
var inst_20882 = (state_20918[(7)]);
var _ = cljs.core.async.impl.ioc_helpers.add_exception_frame.call(null,state_20918,(10),Object,null,(9));
var inst_20891 = chs__$1.call(null,inst_20882);
var inst_20892 = done.call(null,inst_20882);
var inst_20893 = cljs.core.async.take_BANG_.call(null,inst_20891,inst_20892);
var state_20918__$1 = state_20918;
var statearr_20932_20963 = state_20918__$1;
(statearr_20932_20963[(2)] = inst_20893);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20918__$1);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (9))){
var inst_20882 = (state_20918[(7)]);
var inst_20895 = (state_20918[(2)]);
var inst_20896 = (inst_20882 + (1));
var inst_20882__$1 = inst_20896;
var state_20918__$1 = (function (){var statearr_20933 = state_20918;
(statearr_20933[(10)] = inst_20895);

(statearr_20933[(7)] = inst_20882__$1);

return statearr_20933;
})();
var statearr_20934_20964 = state_20918__$1;
(statearr_20934_20964[(2)] = null);

(statearr_20934_20964[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (5))){
var inst_20902 = (state_20918[(2)]);
var state_20918__$1 = (function (){var statearr_20935 = state_20918;
(statearr_20935[(11)] = inst_20902);

return statearr_20935;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20918__$1,(12),dchan);
} else {
if((state_val_20919 === (14))){
var inst_20904 = (state_20918[(8)]);
var inst_20909 = cljs.core.apply.call(null,f,inst_20904);
var state_20918__$1 = state_20918;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_20918__$1,(16),out,inst_20909);
} else {
if((state_val_20919 === (16))){
var inst_20911 = (state_20918[(2)]);
var state_20918__$1 = (function (){var statearr_20936 = state_20918;
(statearr_20936[(12)] = inst_20911);

return statearr_20936;
})();
var statearr_20937_20965 = state_20918__$1;
(statearr_20937_20965[(2)] = null);

(statearr_20937_20965[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (10))){
var inst_20886 = (state_20918[(2)]);
var inst_20887 = cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec);
var state_20918__$1 = (function (){var statearr_20938 = state_20918;
(statearr_20938[(13)] = inst_20886);

return statearr_20938;
})();
var statearr_20939_20966 = state_20918__$1;
(statearr_20939_20966[(2)] = inst_20887);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20918__$1);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_20919 === (8))){
var inst_20900 = (state_20918[(2)]);
var state_20918__$1 = state_20918;
var statearr_20940_20967 = state_20918__$1;
(statearr_20940_20967[(2)] = inst_20900);

(statearr_20940_20967[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done))
;
return ((function (switch__18821__auto__,c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_20944 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_20944[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_20944[(1)] = (1));

return statearr_20944;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_20918){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_20918);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e20945){if((e20945 instanceof Object)){
var ex__18825__auto__ = e20945;
var statearr_20946_20968 = state_20918;
(statearr_20946_20968[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20918);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20945;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20969 = state_20918;
state_20918 = G__20969;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_20918){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_20918);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done))
})();
var state__18935__auto__ = (function (){var statearr_20947 = f__18934__auto__.call(null);
(statearr_20947[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___20952);

return statearr_20947;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___20952,chs__$1,out,cnt,rets,dchan,dctr,done))
);


return out;
});

cljs.core.async.map.cljs$lang$maxFixedArity = 3;
/**
 * Takes a collection of source channels and returns a channel which
 *   contains all values taken from them. The returned channel will be
 *   unbuffered by default, or a buf-or-n can be supplied. The channel
 *   will close after all the source channels have closed.
 */
cljs.core.async.merge = (function cljs$core$async$merge(var_args){
var args20971 = [];
var len__17824__auto___21027 = arguments.length;
var i__17825__auto___21028 = (0);
while(true){
if((i__17825__auto___21028 < len__17824__auto___21027)){
args20971.push((arguments[i__17825__auto___21028]));

var G__21029 = (i__17825__auto___21028 + (1));
i__17825__auto___21028 = G__21029;
continue;
} else {
}
break;
}

var G__20973 = args20971.length;
switch (G__20973) {
case 1:
return cljs.core.async.merge.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.merge.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args20971.length)].join('')));

}
});

cljs.core.async.merge.cljs$core$IFn$_invoke$arity$1 = (function (chs){
return cljs.core.async.merge.call(null,chs,null);
});

cljs.core.async.merge.cljs$core$IFn$_invoke$arity$2 = (function (chs,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21031 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21031,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21031,out){
return (function (state_21003){
var state_val_21004 = (state_21003[(1)]);
if((state_val_21004 === (7))){
var inst_20982 = (state_21003[(7)]);
var inst_20983 = (state_21003[(8)]);
var inst_20982__$1 = (state_21003[(2)]);
var inst_20983__$1 = cljs.core.nth.call(null,inst_20982__$1,(0),null);
var inst_20984 = cljs.core.nth.call(null,inst_20982__$1,(1),null);
var inst_20985 = (inst_20983__$1 == null);
var state_21003__$1 = (function (){var statearr_21005 = state_21003;
(statearr_21005[(9)] = inst_20984);

(statearr_21005[(7)] = inst_20982__$1);

(statearr_21005[(8)] = inst_20983__$1);

return statearr_21005;
})();
if(cljs.core.truth_(inst_20985)){
var statearr_21006_21032 = state_21003__$1;
(statearr_21006_21032[(1)] = (8));

} else {
var statearr_21007_21033 = state_21003__$1;
(statearr_21007_21033[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (1))){
var inst_20974 = cljs.core.vec.call(null,chs);
var inst_20975 = inst_20974;
var state_21003__$1 = (function (){var statearr_21008 = state_21003;
(statearr_21008[(10)] = inst_20975);

return statearr_21008;
})();
var statearr_21009_21034 = state_21003__$1;
(statearr_21009_21034[(2)] = null);

(statearr_21009_21034[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (4))){
var inst_20975 = (state_21003[(10)]);
var state_21003__$1 = state_21003;
return cljs.core.async.ioc_alts_BANG_.call(null,state_21003__$1,(7),inst_20975);
} else {
if((state_val_21004 === (6))){
var inst_20999 = (state_21003[(2)]);
var state_21003__$1 = state_21003;
var statearr_21010_21035 = state_21003__$1;
(statearr_21010_21035[(2)] = inst_20999);

(statearr_21010_21035[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (3))){
var inst_21001 = (state_21003[(2)]);
var state_21003__$1 = state_21003;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21003__$1,inst_21001);
} else {
if((state_val_21004 === (2))){
var inst_20975 = (state_21003[(10)]);
var inst_20977 = cljs.core.count.call(null,inst_20975);
var inst_20978 = (inst_20977 > (0));
var state_21003__$1 = state_21003;
if(cljs.core.truth_(inst_20978)){
var statearr_21012_21036 = state_21003__$1;
(statearr_21012_21036[(1)] = (4));

} else {
var statearr_21013_21037 = state_21003__$1;
(statearr_21013_21037[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (11))){
var inst_20975 = (state_21003[(10)]);
var inst_20992 = (state_21003[(2)]);
var tmp21011 = inst_20975;
var inst_20975__$1 = tmp21011;
var state_21003__$1 = (function (){var statearr_21014 = state_21003;
(statearr_21014[(11)] = inst_20992);

(statearr_21014[(10)] = inst_20975__$1);

return statearr_21014;
})();
var statearr_21015_21038 = state_21003__$1;
(statearr_21015_21038[(2)] = null);

(statearr_21015_21038[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (9))){
var inst_20983 = (state_21003[(8)]);
var state_21003__$1 = state_21003;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21003__$1,(11),out,inst_20983);
} else {
if((state_val_21004 === (5))){
var inst_20997 = cljs.core.async.close_BANG_.call(null,out);
var state_21003__$1 = state_21003;
var statearr_21016_21039 = state_21003__$1;
(statearr_21016_21039[(2)] = inst_20997);

(statearr_21016_21039[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (10))){
var inst_20995 = (state_21003[(2)]);
var state_21003__$1 = state_21003;
var statearr_21017_21040 = state_21003__$1;
(statearr_21017_21040[(2)] = inst_20995);

(statearr_21017_21040[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21004 === (8))){
var inst_20984 = (state_21003[(9)]);
var inst_20982 = (state_21003[(7)]);
var inst_20975 = (state_21003[(10)]);
var inst_20983 = (state_21003[(8)]);
var inst_20987 = (function (){var cs = inst_20975;
var vec__20980 = inst_20982;
var v = inst_20983;
var c = inst_20984;
return ((function (cs,vec__20980,v,c,inst_20984,inst_20982,inst_20975,inst_20983,state_val_21004,c__18933__auto___21031,out){
return (function (p1__20970_SHARP_){
return cljs.core.not_EQ_.call(null,c,p1__20970_SHARP_);
});
;})(cs,vec__20980,v,c,inst_20984,inst_20982,inst_20975,inst_20983,state_val_21004,c__18933__auto___21031,out))
})();
var inst_20988 = cljs.core.filterv.call(null,inst_20987,inst_20975);
var inst_20975__$1 = inst_20988;
var state_21003__$1 = (function (){var statearr_21018 = state_21003;
(statearr_21018[(10)] = inst_20975__$1);

return statearr_21018;
})();
var statearr_21019_21041 = state_21003__$1;
(statearr_21019_21041[(2)] = null);

(statearr_21019_21041[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21031,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21031,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21023 = [null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_21023[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21023[(1)] = (1));

return statearr_21023;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21003){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21003);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21024){if((e21024 instanceof Object)){
var ex__18825__auto__ = e21024;
var statearr_21025_21042 = state_21003;
(statearr_21025_21042[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21003);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21024;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21043 = state_21003;
state_21003 = G__21043;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21003){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21003);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21031,out))
})();
var state__18935__auto__ = (function (){var statearr_21026 = f__18934__auto__.call(null);
(statearr_21026[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21031);

return statearr_21026;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21031,out))
);


return out;
});

cljs.core.async.merge.cljs$lang$maxFixedArity = 2;
/**
 * Returns a channel containing the single (collection) result of the
 *   items taken from the channel conjoined to the supplied
 *   collection. ch must close before into produces a result.
 */
cljs.core.async.into = (function cljs$core$async$into(coll,ch){
return cljs.core.async.reduce.call(null,cljs.core.conj,coll,ch);
});
/**
 * Returns a channel that will return, at most, n items from ch. After n items
 * have been returned, or ch has been closed, the return chanel will close.
 * 
 *   The output channel is unbuffered by default, unless buf-or-n is given.
 */
cljs.core.async.take = (function cljs$core$async$take(var_args){
var args21044 = [];
var len__17824__auto___21093 = arguments.length;
var i__17825__auto___21094 = (0);
while(true){
if((i__17825__auto___21094 < len__17824__auto___21093)){
args21044.push((arguments[i__17825__auto___21094]));

var G__21095 = (i__17825__auto___21094 + (1));
i__17825__auto___21094 = G__21095;
continue;
} else {
}
break;
}

var G__21046 = args21044.length;
switch (G__21046) {
case 2:
return cljs.core.async.take.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.take.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21044.length)].join('')));

}
});

cljs.core.async.take.cljs$core$IFn$_invoke$arity$2 = (function (n,ch){
return cljs.core.async.take.call(null,n,ch,null);
});

cljs.core.async.take.cljs$core$IFn$_invoke$arity$3 = (function (n,ch,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21097 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21097,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21097,out){
return (function (state_21070){
var state_val_21071 = (state_21070[(1)]);
if((state_val_21071 === (7))){
var inst_21052 = (state_21070[(7)]);
var inst_21052__$1 = (state_21070[(2)]);
var inst_21053 = (inst_21052__$1 == null);
var inst_21054 = cljs.core.not.call(null,inst_21053);
var state_21070__$1 = (function (){var statearr_21072 = state_21070;
(statearr_21072[(7)] = inst_21052__$1);

return statearr_21072;
})();
if(inst_21054){
var statearr_21073_21098 = state_21070__$1;
(statearr_21073_21098[(1)] = (8));

} else {
var statearr_21074_21099 = state_21070__$1;
(statearr_21074_21099[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (1))){
var inst_21047 = (0);
var state_21070__$1 = (function (){var statearr_21075 = state_21070;
(statearr_21075[(8)] = inst_21047);

return statearr_21075;
})();
var statearr_21076_21100 = state_21070__$1;
(statearr_21076_21100[(2)] = null);

(statearr_21076_21100[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (4))){
var state_21070__$1 = state_21070;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21070__$1,(7),ch);
} else {
if((state_val_21071 === (6))){
var inst_21065 = (state_21070[(2)]);
var state_21070__$1 = state_21070;
var statearr_21077_21101 = state_21070__$1;
(statearr_21077_21101[(2)] = inst_21065);

(statearr_21077_21101[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (3))){
var inst_21067 = (state_21070[(2)]);
var inst_21068 = cljs.core.async.close_BANG_.call(null,out);
var state_21070__$1 = (function (){var statearr_21078 = state_21070;
(statearr_21078[(9)] = inst_21067);

return statearr_21078;
})();
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21070__$1,inst_21068);
} else {
if((state_val_21071 === (2))){
var inst_21047 = (state_21070[(8)]);
var inst_21049 = (inst_21047 < n);
var state_21070__$1 = state_21070;
if(cljs.core.truth_(inst_21049)){
var statearr_21079_21102 = state_21070__$1;
(statearr_21079_21102[(1)] = (4));

} else {
var statearr_21080_21103 = state_21070__$1;
(statearr_21080_21103[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (11))){
var inst_21047 = (state_21070[(8)]);
var inst_21057 = (state_21070[(2)]);
var inst_21058 = (inst_21047 + (1));
var inst_21047__$1 = inst_21058;
var state_21070__$1 = (function (){var statearr_21081 = state_21070;
(statearr_21081[(8)] = inst_21047__$1);

(statearr_21081[(10)] = inst_21057);

return statearr_21081;
})();
var statearr_21082_21104 = state_21070__$1;
(statearr_21082_21104[(2)] = null);

(statearr_21082_21104[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (9))){
var state_21070__$1 = state_21070;
var statearr_21083_21105 = state_21070__$1;
(statearr_21083_21105[(2)] = null);

(statearr_21083_21105[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (5))){
var state_21070__$1 = state_21070;
var statearr_21084_21106 = state_21070__$1;
(statearr_21084_21106[(2)] = null);

(statearr_21084_21106[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (10))){
var inst_21062 = (state_21070[(2)]);
var state_21070__$1 = state_21070;
var statearr_21085_21107 = state_21070__$1;
(statearr_21085_21107[(2)] = inst_21062);

(statearr_21085_21107[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21071 === (8))){
var inst_21052 = (state_21070[(7)]);
var state_21070__$1 = state_21070;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21070__$1,(11),out,inst_21052);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21097,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21097,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21089 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_21089[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21089[(1)] = (1));

return statearr_21089;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21070){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21070);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21090){if((e21090 instanceof Object)){
var ex__18825__auto__ = e21090;
var statearr_21091_21108 = state_21070;
(statearr_21091_21108[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21070);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21090;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21109 = state_21070;
state_21070 = G__21109;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21070){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21070);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21097,out))
})();
var state__18935__auto__ = (function (){var statearr_21092 = f__18934__auto__.call(null);
(statearr_21092[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21097);

return statearr_21092;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21097,out))
);


return out;
});

cljs.core.async.take.cljs$lang$maxFixedArity = 3;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.map_LT_ = (function cljs$core$async$map_LT_(f,ch){
if(typeof cljs.core.async.t_cljs$core$async21117 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async21117 = (function (map_LT_,f,ch,meta21118){
this.map_LT_ = map_LT_;
this.f = f;
this.ch = ch;
this.meta21118 = meta21118;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_21119,meta21118__$1){
var self__ = this;
var _21119__$1 = this;
return (new cljs.core.async.t_cljs$core$async21117(self__.map_LT_,self__.f,self__.ch,meta21118__$1));
});

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_21119){
var self__ = this;
var _21119__$1 = this;
return self__.meta21118;
});

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$Channel$ = true;

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch);
});

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
var ret = cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,(function (){
if(typeof cljs.core.async.t_cljs$core$async21120 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async21120 = (function (map_LT_,f,ch,meta21118,_,fn1,meta21121){
this.map_LT_ = map_LT_;
this.f = f;
this.ch = ch;
this.meta21118 = meta21118;
this._ = _;
this.fn1 = fn1;
this.meta21121 = meta21121;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (___$1){
return (function (_21122,meta21121__$1){
var self__ = this;
var _21122__$1 = this;
return (new cljs.core.async.t_cljs$core$async21120(self__.map_LT_,self__.f,self__.ch,self__.meta21118,self__._,self__.fn1,meta21121__$1));
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (___$1){
return (function (_21122){
var self__ = this;
var _21122__$1 = this;
return self__.meta21121;
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$async$impl$protocols$Handler$ = true;

cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = ((function (___$1){
return (function (___$1){
var self__ = this;
var ___$2 = this;
return cljs.core.async.impl.protocols.active_QMARK_.call(null,self__.fn1);
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = ((function (___$1){
return (function (___$1){
var self__ = this;
var ___$2 = this;
return true;
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = ((function (___$1){
return (function (___$1){
var self__ = this;
var ___$2 = this;
var f1 = cljs.core.async.impl.protocols.commit.call(null,self__.fn1);
return ((function (f1,___$2,___$1){
return (function (p1__21110_SHARP_){
return f1.call(null,(((p1__21110_SHARP_ == null))?null:self__.f.call(null,p1__21110_SHARP_)));
});
;})(f1,___$2,___$1))
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.getBasis = ((function (___$1){
return (function (){
return new cljs.core.PersistentVector(null, 7, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"map<","map<",-1235808357,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Deprecated - this function will be removed. Use transducer instead"], null)),new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta21118","meta21118",1983100157,null),cljs.core.with_meta(new cljs.core.Symbol(null,"_","_",-1201019570,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol("cljs.core.async","t_cljs$core$async21117","cljs.core.async/t_cljs$core$async21117",914899689,null)], null)),new cljs.core.Symbol(null,"fn1","fn1",895834444,null),new cljs.core.Symbol(null,"meta21121","meta21121",-1523246389,null)], null);
});})(___$1))
;

cljs.core.async.t_cljs$core$async21120.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async21120.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async21120";

cljs.core.async.t_cljs$core$async21120.cljs$lang$ctorPrWriter = ((function (___$1){
return (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async21120");
});})(___$1))
;

cljs.core.async.__GT_t_cljs$core$async21120 = ((function (___$1){
return (function cljs$core$async$map_LT__$___GT_t_cljs$core$async21120(map_LT___$1,f__$1,ch__$1,meta21118__$1,___$2,fn1__$1,meta21121){
return (new cljs.core.async.t_cljs$core$async21120(map_LT___$1,f__$1,ch__$1,meta21118__$1,___$2,fn1__$1,meta21121));
});})(___$1))
;

}

return (new cljs.core.async.t_cljs$core$async21120(self__.map_LT_,self__.f,self__.ch,self__.meta21118,___$1,fn1,cljs.core.PersistentArrayMap.EMPTY));
})()
);
if(cljs.core.truth_((function (){var and__16754__auto__ = ret;
if(cljs.core.truth_(and__16754__auto__)){
return !((cljs.core.deref.call(null,ret) == null));
} else {
return and__16754__auto__;
}
})())){
return cljs.core.async.impl.channels.box.call(null,self__.f.call(null,cljs.core.deref.call(null,ret)));
} else {
return ret;
}
});

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$WritePort$ = true;

cljs.core.async.t_cljs$core$async21117.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,val,fn1);
});

cljs.core.async.t_cljs$core$async21117.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"map<","map<",-1235808357,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Deprecated - this function will be removed. Use transducer instead"], null)),new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta21118","meta21118",1983100157,null)], null);
});

cljs.core.async.t_cljs$core$async21117.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async21117.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async21117";

cljs.core.async.t_cljs$core$async21117.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async21117");
});

cljs.core.async.__GT_t_cljs$core$async21117 = (function cljs$core$async$map_LT__$___GT_t_cljs$core$async21117(map_LT___$1,f__$1,ch__$1,meta21118){
return (new cljs.core.async.t_cljs$core$async21117(map_LT___$1,f__$1,ch__$1,meta21118));
});

}

return (new cljs.core.async.t_cljs$core$async21117(cljs$core$async$map_LT_,f,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.map_GT_ = (function cljs$core$async$map_GT_(f,ch){
if(typeof cljs.core.async.t_cljs$core$async21126 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async21126 = (function (map_GT_,f,ch,meta21127){
this.map_GT_ = map_GT_;
this.f = f;
this.ch = ch;
this.meta21127 = meta21127;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_21128,meta21127__$1){
var self__ = this;
var _21128__$1 = this;
return (new cljs.core.async.t_cljs$core$async21126(self__.map_GT_,self__.f,self__.ch,meta21127__$1));
});

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_21128){
var self__ = this;
var _21128__$1 = this;
return self__.meta21127;
});

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$Channel$ = true;

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,fn1);
});

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$WritePort$ = true;

cljs.core.async.t_cljs$core$async21126.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,self__.f.call(null,val),fn1);
});

cljs.core.async.t_cljs$core$async21126.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"map>","map>",1676369295,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Deprecated - this function will be removed. Use transducer instead"], null)),new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta21127","meta21127",1916326668,null)], null);
});

cljs.core.async.t_cljs$core$async21126.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async21126.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async21126";

cljs.core.async.t_cljs$core$async21126.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async21126");
});

cljs.core.async.__GT_t_cljs$core$async21126 = (function cljs$core$async$map_GT__$___GT_t_cljs$core$async21126(map_GT___$1,f__$1,ch__$1,meta21127){
return (new cljs.core.async.t_cljs$core$async21126(map_GT___$1,f__$1,ch__$1,meta21127));
});

}

return (new cljs.core.async.t_cljs$core$async21126(cljs$core$async$map_GT_,f,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.filter_GT_ = (function cljs$core$async$filter_GT_(p,ch){
if(typeof cljs.core.async.t_cljs$core$async21132 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async21132 = (function (filter_GT_,p,ch,meta21133){
this.filter_GT_ = filter_GT_;
this.p = p;
this.ch = ch;
this.meta21133 = meta21133;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_21134,meta21133__$1){
var self__ = this;
var _21134__$1 = this;
return (new cljs.core.async.t_cljs$core$async21132(self__.filter_GT_,self__.p,self__.ch,meta21133__$1));
});

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_21134){
var self__ = this;
var _21134__$1 = this;
return self__.meta21133;
});

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$Channel$ = true;

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch);
});

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,fn1);
});

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$WritePort$ = true;

cljs.core.async.t_cljs$core$async21132.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
if(cljs.core.truth_(self__.p.call(null,val))){
return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,val,fn1);
} else {
return cljs.core.async.impl.channels.box.call(null,cljs.core.not.call(null,cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch)));
}
});

cljs.core.async.t_cljs$core$async21132.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"filter>","filter>",-37644455,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"p","p",1791580836,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null)], null))),new cljs.core.Keyword(null,"doc","doc",1913296891),"Deprecated - this function will be removed. Use transducer instead"], null)),new cljs.core.Symbol(null,"p","p",1791580836,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta21133","meta21133",-1249868455,null)], null);
});

cljs.core.async.t_cljs$core$async21132.cljs$lang$type = true;

cljs.core.async.t_cljs$core$async21132.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async21132";

cljs.core.async.t_cljs$core$async21132.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"cljs.core.async/t_cljs$core$async21132");
});

cljs.core.async.__GT_t_cljs$core$async21132 = (function cljs$core$async$filter_GT__$___GT_t_cljs$core$async21132(filter_GT___$1,p__$1,ch__$1,meta21133){
return (new cljs.core.async.t_cljs$core$async21132(filter_GT___$1,p__$1,ch__$1,meta21133));
});

}

return (new cljs.core.async.t_cljs$core$async21132(cljs$core$async$filter_GT_,p,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.remove_GT_ = (function cljs$core$async$remove_GT_(p,ch){
return cljs.core.async.filter_GT_.call(null,cljs.core.complement.call(null,p),ch);
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.filter_LT_ = (function cljs$core$async$filter_LT_(var_args){
var args21135 = [];
var len__17824__auto___21179 = arguments.length;
var i__17825__auto___21180 = (0);
while(true){
if((i__17825__auto___21180 < len__17824__auto___21179)){
args21135.push((arguments[i__17825__auto___21180]));

var G__21181 = (i__17825__auto___21180 + (1));
i__17825__auto___21180 = G__21181;
continue;
} else {
}
break;
}

var G__21137 = args21135.length;
switch (G__21137) {
case 2:
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21135.length)].join('')));

}
});

cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.filter_LT_.call(null,p,ch,null);
});

cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3 = (function (p,ch,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21183 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21183,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21183,out){
return (function (state_21158){
var state_val_21159 = (state_21158[(1)]);
if((state_val_21159 === (7))){
var inst_21154 = (state_21158[(2)]);
var state_21158__$1 = state_21158;
var statearr_21160_21184 = state_21158__$1;
(statearr_21160_21184[(2)] = inst_21154);

(statearr_21160_21184[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (1))){
var state_21158__$1 = state_21158;
var statearr_21161_21185 = state_21158__$1;
(statearr_21161_21185[(2)] = null);

(statearr_21161_21185[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (4))){
var inst_21140 = (state_21158[(7)]);
var inst_21140__$1 = (state_21158[(2)]);
var inst_21141 = (inst_21140__$1 == null);
var state_21158__$1 = (function (){var statearr_21162 = state_21158;
(statearr_21162[(7)] = inst_21140__$1);

return statearr_21162;
})();
if(cljs.core.truth_(inst_21141)){
var statearr_21163_21186 = state_21158__$1;
(statearr_21163_21186[(1)] = (5));

} else {
var statearr_21164_21187 = state_21158__$1;
(statearr_21164_21187[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (6))){
var inst_21140 = (state_21158[(7)]);
var inst_21145 = p.call(null,inst_21140);
var state_21158__$1 = state_21158;
if(cljs.core.truth_(inst_21145)){
var statearr_21165_21188 = state_21158__$1;
(statearr_21165_21188[(1)] = (8));

} else {
var statearr_21166_21189 = state_21158__$1;
(statearr_21166_21189[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (3))){
var inst_21156 = (state_21158[(2)]);
var state_21158__$1 = state_21158;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21158__$1,inst_21156);
} else {
if((state_val_21159 === (2))){
var state_21158__$1 = state_21158;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21158__$1,(4),ch);
} else {
if((state_val_21159 === (11))){
var inst_21148 = (state_21158[(2)]);
var state_21158__$1 = state_21158;
var statearr_21167_21190 = state_21158__$1;
(statearr_21167_21190[(2)] = inst_21148);

(statearr_21167_21190[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (9))){
var state_21158__$1 = state_21158;
var statearr_21168_21191 = state_21158__$1;
(statearr_21168_21191[(2)] = null);

(statearr_21168_21191[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (5))){
var inst_21143 = cljs.core.async.close_BANG_.call(null,out);
var state_21158__$1 = state_21158;
var statearr_21169_21192 = state_21158__$1;
(statearr_21169_21192[(2)] = inst_21143);

(statearr_21169_21192[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (10))){
var inst_21151 = (state_21158[(2)]);
var state_21158__$1 = (function (){var statearr_21170 = state_21158;
(statearr_21170[(8)] = inst_21151);

return statearr_21170;
})();
var statearr_21171_21193 = state_21158__$1;
(statearr_21171_21193[(2)] = null);

(statearr_21171_21193[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21159 === (8))){
var inst_21140 = (state_21158[(7)]);
var state_21158__$1 = state_21158;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21158__$1,(11),out,inst_21140);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21183,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21183,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21175 = [null,null,null,null,null,null,null,null,null];
(statearr_21175[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21175[(1)] = (1));

return statearr_21175;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21158){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21158);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21176){if((e21176 instanceof Object)){
var ex__18825__auto__ = e21176;
var statearr_21177_21194 = state_21158;
(statearr_21177_21194[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21158);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21176;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21195 = state_21158;
state_21158 = G__21195;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21158){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21158);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21183,out))
})();
var state__18935__auto__ = (function (){var statearr_21178 = f__18934__auto__.call(null);
(statearr_21178[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21183);

return statearr_21178;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21183,out))
);


return out;
});

cljs.core.async.filter_LT_.cljs$lang$maxFixedArity = 3;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.remove_LT_ = (function cljs$core$async$remove_LT_(var_args){
var args21196 = [];
var len__17824__auto___21199 = arguments.length;
var i__17825__auto___21200 = (0);
while(true){
if((i__17825__auto___21200 < len__17824__auto___21199)){
args21196.push((arguments[i__17825__auto___21200]));

var G__21201 = (i__17825__auto___21200 + (1));
i__17825__auto___21200 = G__21201;
continue;
} else {
}
break;
}

var G__21198 = args21196.length;
switch (G__21198) {
case 2:
return cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21196.length)].join('')));

}
});

cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.remove_LT_.call(null,p,ch,null);
});

cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$3 = (function (p,ch,buf_or_n){
return cljs.core.async.filter_LT_.call(null,cljs.core.complement.call(null,p),ch,buf_or_n);
});

cljs.core.async.remove_LT_.cljs$lang$maxFixedArity = 3;
cljs.core.async.mapcat_STAR_ = (function cljs$core$async$mapcat_STAR_(f,in$,out){
var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__){
return (function (state_21368){
var state_val_21369 = (state_21368[(1)]);
if((state_val_21369 === (7))){
var inst_21364 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
var statearr_21370_21411 = state_21368__$1;
(statearr_21370_21411[(2)] = inst_21364);

(statearr_21370_21411[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (20))){
var inst_21334 = (state_21368[(7)]);
var inst_21345 = (state_21368[(2)]);
var inst_21346 = cljs.core.next.call(null,inst_21334);
var inst_21320 = inst_21346;
var inst_21321 = null;
var inst_21322 = (0);
var inst_21323 = (0);
var state_21368__$1 = (function (){var statearr_21371 = state_21368;
(statearr_21371[(8)] = inst_21321);

(statearr_21371[(9)] = inst_21320);

(statearr_21371[(10)] = inst_21322);

(statearr_21371[(11)] = inst_21323);

(statearr_21371[(12)] = inst_21345);

return statearr_21371;
})();
var statearr_21372_21412 = state_21368__$1;
(statearr_21372_21412[(2)] = null);

(statearr_21372_21412[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (1))){
var state_21368__$1 = state_21368;
var statearr_21373_21413 = state_21368__$1;
(statearr_21373_21413[(2)] = null);

(statearr_21373_21413[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (4))){
var inst_21309 = (state_21368[(13)]);
var inst_21309__$1 = (state_21368[(2)]);
var inst_21310 = (inst_21309__$1 == null);
var state_21368__$1 = (function (){var statearr_21374 = state_21368;
(statearr_21374[(13)] = inst_21309__$1);

return statearr_21374;
})();
if(cljs.core.truth_(inst_21310)){
var statearr_21375_21414 = state_21368__$1;
(statearr_21375_21414[(1)] = (5));

} else {
var statearr_21376_21415 = state_21368__$1;
(statearr_21376_21415[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (15))){
var state_21368__$1 = state_21368;
var statearr_21380_21416 = state_21368__$1;
(statearr_21380_21416[(2)] = null);

(statearr_21380_21416[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (21))){
var state_21368__$1 = state_21368;
var statearr_21381_21417 = state_21368__$1;
(statearr_21381_21417[(2)] = null);

(statearr_21381_21417[(1)] = (23));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (13))){
var inst_21321 = (state_21368[(8)]);
var inst_21320 = (state_21368[(9)]);
var inst_21322 = (state_21368[(10)]);
var inst_21323 = (state_21368[(11)]);
var inst_21330 = (state_21368[(2)]);
var inst_21331 = (inst_21323 + (1));
var tmp21377 = inst_21321;
var tmp21378 = inst_21320;
var tmp21379 = inst_21322;
var inst_21320__$1 = tmp21378;
var inst_21321__$1 = tmp21377;
var inst_21322__$1 = tmp21379;
var inst_21323__$1 = inst_21331;
var state_21368__$1 = (function (){var statearr_21382 = state_21368;
(statearr_21382[(8)] = inst_21321__$1);

(statearr_21382[(9)] = inst_21320__$1);

(statearr_21382[(10)] = inst_21322__$1);

(statearr_21382[(11)] = inst_21323__$1);

(statearr_21382[(14)] = inst_21330);

return statearr_21382;
})();
var statearr_21383_21418 = state_21368__$1;
(statearr_21383_21418[(2)] = null);

(statearr_21383_21418[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (22))){
var state_21368__$1 = state_21368;
var statearr_21384_21419 = state_21368__$1;
(statearr_21384_21419[(2)] = null);

(statearr_21384_21419[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (6))){
var inst_21309 = (state_21368[(13)]);
var inst_21318 = f.call(null,inst_21309);
var inst_21319 = cljs.core.seq.call(null,inst_21318);
var inst_21320 = inst_21319;
var inst_21321 = null;
var inst_21322 = (0);
var inst_21323 = (0);
var state_21368__$1 = (function (){var statearr_21385 = state_21368;
(statearr_21385[(8)] = inst_21321);

(statearr_21385[(9)] = inst_21320);

(statearr_21385[(10)] = inst_21322);

(statearr_21385[(11)] = inst_21323);

return statearr_21385;
})();
var statearr_21386_21420 = state_21368__$1;
(statearr_21386_21420[(2)] = null);

(statearr_21386_21420[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (17))){
var inst_21334 = (state_21368[(7)]);
var inst_21338 = cljs.core.chunk_first.call(null,inst_21334);
var inst_21339 = cljs.core.chunk_rest.call(null,inst_21334);
var inst_21340 = cljs.core.count.call(null,inst_21338);
var inst_21320 = inst_21339;
var inst_21321 = inst_21338;
var inst_21322 = inst_21340;
var inst_21323 = (0);
var state_21368__$1 = (function (){var statearr_21387 = state_21368;
(statearr_21387[(8)] = inst_21321);

(statearr_21387[(9)] = inst_21320);

(statearr_21387[(10)] = inst_21322);

(statearr_21387[(11)] = inst_21323);

return statearr_21387;
})();
var statearr_21388_21421 = state_21368__$1;
(statearr_21388_21421[(2)] = null);

(statearr_21388_21421[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (3))){
var inst_21366 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21368__$1,inst_21366);
} else {
if((state_val_21369 === (12))){
var inst_21354 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
var statearr_21389_21422 = state_21368__$1;
(statearr_21389_21422[(2)] = inst_21354);

(statearr_21389_21422[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (2))){
var state_21368__$1 = state_21368;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21368__$1,(4),in$);
} else {
if((state_val_21369 === (23))){
var inst_21362 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
var statearr_21390_21423 = state_21368__$1;
(statearr_21390_21423[(2)] = inst_21362);

(statearr_21390_21423[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (19))){
var inst_21349 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
var statearr_21391_21424 = state_21368__$1;
(statearr_21391_21424[(2)] = inst_21349);

(statearr_21391_21424[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (11))){
var inst_21320 = (state_21368[(9)]);
var inst_21334 = (state_21368[(7)]);
var inst_21334__$1 = cljs.core.seq.call(null,inst_21320);
var state_21368__$1 = (function (){var statearr_21392 = state_21368;
(statearr_21392[(7)] = inst_21334__$1);

return statearr_21392;
})();
if(inst_21334__$1){
var statearr_21393_21425 = state_21368__$1;
(statearr_21393_21425[(1)] = (14));

} else {
var statearr_21394_21426 = state_21368__$1;
(statearr_21394_21426[(1)] = (15));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (9))){
var inst_21356 = (state_21368[(2)]);
var inst_21357 = cljs.core.async.impl.protocols.closed_QMARK_.call(null,out);
var state_21368__$1 = (function (){var statearr_21395 = state_21368;
(statearr_21395[(15)] = inst_21356);

return statearr_21395;
})();
if(cljs.core.truth_(inst_21357)){
var statearr_21396_21427 = state_21368__$1;
(statearr_21396_21427[(1)] = (21));

} else {
var statearr_21397_21428 = state_21368__$1;
(statearr_21397_21428[(1)] = (22));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (5))){
var inst_21312 = cljs.core.async.close_BANG_.call(null,out);
var state_21368__$1 = state_21368;
var statearr_21398_21429 = state_21368__$1;
(statearr_21398_21429[(2)] = inst_21312);

(statearr_21398_21429[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (14))){
var inst_21334 = (state_21368[(7)]);
var inst_21336 = cljs.core.chunked_seq_QMARK_.call(null,inst_21334);
var state_21368__$1 = state_21368;
if(inst_21336){
var statearr_21399_21430 = state_21368__$1;
(statearr_21399_21430[(1)] = (17));

} else {
var statearr_21400_21431 = state_21368__$1;
(statearr_21400_21431[(1)] = (18));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (16))){
var inst_21352 = (state_21368[(2)]);
var state_21368__$1 = state_21368;
var statearr_21401_21432 = state_21368__$1;
(statearr_21401_21432[(2)] = inst_21352);

(statearr_21401_21432[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21369 === (10))){
var inst_21321 = (state_21368[(8)]);
var inst_21323 = (state_21368[(11)]);
var inst_21328 = cljs.core._nth.call(null,inst_21321,inst_21323);
var state_21368__$1 = state_21368;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21368__$1,(13),out,inst_21328);
} else {
if((state_val_21369 === (18))){
var inst_21334 = (state_21368[(7)]);
var inst_21343 = cljs.core.first.call(null,inst_21334);
var state_21368__$1 = state_21368;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21368__$1,(20),out,inst_21343);
} else {
if((state_val_21369 === (8))){
var inst_21322 = (state_21368[(10)]);
var inst_21323 = (state_21368[(11)]);
var inst_21325 = (inst_21323 < inst_21322);
var inst_21326 = inst_21325;
var state_21368__$1 = state_21368;
if(cljs.core.truth_(inst_21326)){
var statearr_21402_21433 = state_21368__$1;
(statearr_21402_21433[(1)] = (10));

} else {
var statearr_21403_21434 = state_21368__$1;
(statearr_21403_21434[(1)] = (11));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto__))
;
return ((function (switch__18821__auto__,c__18933__auto__){
return (function() {
var cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__ = null;
var cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____0 = (function (){
var statearr_21407 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_21407[(0)] = cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__);

(statearr_21407[(1)] = (1));

return statearr_21407;
});
var cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____1 = (function (state_21368){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21368);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21408){if((e21408 instanceof Object)){
var ex__18825__auto__ = e21408;
var statearr_21409_21435 = state_21368;
(statearr_21409_21435[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21368);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21408;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21436 = state_21368;
state_21368 = G__21436;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__ = function(state_21368){
switch(arguments.length){
case 0:
return cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____1.call(this,state_21368);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____0;
cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mapcat_STAR__$_state_machine__18822__auto____1;
return cljs$core$async$mapcat_STAR__$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__))
})();
var state__18935__auto__ = (function (){var statearr_21410 = f__18934__auto__.call(null);
(statearr_21410[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_21410;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__))
);

return c__18933__auto__;
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.mapcat_LT_ = (function cljs$core$async$mapcat_LT_(var_args){
var args21437 = [];
var len__17824__auto___21440 = arguments.length;
var i__17825__auto___21441 = (0);
while(true){
if((i__17825__auto___21441 < len__17824__auto___21440)){
args21437.push((arguments[i__17825__auto___21441]));

var G__21442 = (i__17825__auto___21441 + (1));
i__17825__auto___21441 = G__21442;
continue;
} else {
}
break;
}

var G__21439 = args21437.length;
switch (G__21439) {
case 2:
return cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21437.length)].join('')));

}
});

cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$2 = (function (f,in$){
return cljs.core.async.mapcat_LT_.call(null,f,in$,null);
});

cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$3 = (function (f,in$,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
cljs.core.async.mapcat_STAR_.call(null,f,in$,out);

return out;
});

cljs.core.async.mapcat_LT_.cljs$lang$maxFixedArity = 3;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.mapcat_GT_ = (function cljs$core$async$mapcat_GT_(var_args){
var args21444 = [];
var len__17824__auto___21447 = arguments.length;
var i__17825__auto___21448 = (0);
while(true){
if((i__17825__auto___21448 < len__17824__auto___21447)){
args21444.push((arguments[i__17825__auto___21448]));

var G__21449 = (i__17825__auto___21448 + (1));
i__17825__auto___21448 = G__21449;
continue;
} else {
}
break;
}

var G__21446 = args21444.length;
switch (G__21446) {
case 2:
return cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21444.length)].join('')));

}
});

cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$2 = (function (f,out){
return cljs.core.async.mapcat_GT_.call(null,f,out,null);
});

cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$3 = (function (f,out,buf_or_n){
var in$ = cljs.core.async.chan.call(null,buf_or_n);
cljs.core.async.mapcat_STAR_.call(null,f,in$,out);

return in$;
});

cljs.core.async.mapcat_GT_.cljs$lang$maxFixedArity = 3;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.unique = (function cljs$core$async$unique(var_args){
var args21451 = [];
var len__17824__auto___21502 = arguments.length;
var i__17825__auto___21503 = (0);
while(true){
if((i__17825__auto___21503 < len__17824__auto___21502)){
args21451.push((arguments[i__17825__auto___21503]));

var G__21504 = (i__17825__auto___21503 + (1));
i__17825__auto___21503 = G__21504;
continue;
} else {
}
break;
}

var G__21453 = args21451.length;
switch (G__21453) {
case 1:
return cljs.core.async.unique.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unique.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21451.length)].join('')));

}
});

cljs.core.async.unique.cljs$core$IFn$_invoke$arity$1 = (function (ch){
return cljs.core.async.unique.call(null,ch,null);
});

cljs.core.async.unique.cljs$core$IFn$_invoke$arity$2 = (function (ch,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21506 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21506,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21506,out){
return (function (state_21477){
var state_val_21478 = (state_21477[(1)]);
if((state_val_21478 === (7))){
var inst_21472 = (state_21477[(2)]);
var state_21477__$1 = state_21477;
var statearr_21479_21507 = state_21477__$1;
(statearr_21479_21507[(2)] = inst_21472);

(statearr_21479_21507[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (1))){
var inst_21454 = null;
var state_21477__$1 = (function (){var statearr_21480 = state_21477;
(statearr_21480[(7)] = inst_21454);

return statearr_21480;
})();
var statearr_21481_21508 = state_21477__$1;
(statearr_21481_21508[(2)] = null);

(statearr_21481_21508[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (4))){
var inst_21457 = (state_21477[(8)]);
var inst_21457__$1 = (state_21477[(2)]);
var inst_21458 = (inst_21457__$1 == null);
var inst_21459 = cljs.core.not.call(null,inst_21458);
var state_21477__$1 = (function (){var statearr_21482 = state_21477;
(statearr_21482[(8)] = inst_21457__$1);

return statearr_21482;
})();
if(inst_21459){
var statearr_21483_21509 = state_21477__$1;
(statearr_21483_21509[(1)] = (5));

} else {
var statearr_21484_21510 = state_21477__$1;
(statearr_21484_21510[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (6))){
var state_21477__$1 = state_21477;
var statearr_21485_21511 = state_21477__$1;
(statearr_21485_21511[(2)] = null);

(statearr_21485_21511[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (3))){
var inst_21474 = (state_21477[(2)]);
var inst_21475 = cljs.core.async.close_BANG_.call(null,out);
var state_21477__$1 = (function (){var statearr_21486 = state_21477;
(statearr_21486[(9)] = inst_21474);

return statearr_21486;
})();
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21477__$1,inst_21475);
} else {
if((state_val_21478 === (2))){
var state_21477__$1 = state_21477;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21477__$1,(4),ch);
} else {
if((state_val_21478 === (11))){
var inst_21457 = (state_21477[(8)]);
var inst_21466 = (state_21477[(2)]);
var inst_21454 = inst_21457;
var state_21477__$1 = (function (){var statearr_21487 = state_21477;
(statearr_21487[(7)] = inst_21454);

(statearr_21487[(10)] = inst_21466);

return statearr_21487;
})();
var statearr_21488_21512 = state_21477__$1;
(statearr_21488_21512[(2)] = null);

(statearr_21488_21512[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (9))){
var inst_21457 = (state_21477[(8)]);
var state_21477__$1 = state_21477;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21477__$1,(11),out,inst_21457);
} else {
if((state_val_21478 === (5))){
var inst_21454 = (state_21477[(7)]);
var inst_21457 = (state_21477[(8)]);
var inst_21461 = cljs.core._EQ_.call(null,inst_21457,inst_21454);
var state_21477__$1 = state_21477;
if(inst_21461){
var statearr_21490_21513 = state_21477__$1;
(statearr_21490_21513[(1)] = (8));

} else {
var statearr_21491_21514 = state_21477__$1;
(statearr_21491_21514[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (10))){
var inst_21469 = (state_21477[(2)]);
var state_21477__$1 = state_21477;
var statearr_21492_21515 = state_21477__$1;
(statearr_21492_21515[(2)] = inst_21469);

(statearr_21492_21515[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21478 === (8))){
var inst_21454 = (state_21477[(7)]);
var tmp21489 = inst_21454;
var inst_21454__$1 = tmp21489;
var state_21477__$1 = (function (){var statearr_21493 = state_21477;
(statearr_21493[(7)] = inst_21454__$1);

return statearr_21493;
})();
var statearr_21494_21516 = state_21477__$1;
(statearr_21494_21516[(2)] = null);

(statearr_21494_21516[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21506,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21506,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21498 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_21498[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21498[(1)] = (1));

return statearr_21498;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21477){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21477);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21499){if((e21499 instanceof Object)){
var ex__18825__auto__ = e21499;
var statearr_21500_21517 = state_21477;
(statearr_21500_21517[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21477);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21499;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21518 = state_21477;
state_21477 = G__21518;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21477){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21477);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21506,out))
})();
var state__18935__auto__ = (function (){var statearr_21501 = f__18934__auto__.call(null);
(statearr_21501[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21506);

return statearr_21501;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21506,out))
);


return out;
});

cljs.core.async.unique.cljs$lang$maxFixedArity = 2;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.partition = (function cljs$core$async$partition(var_args){
var args21519 = [];
var len__17824__auto___21589 = arguments.length;
var i__17825__auto___21590 = (0);
while(true){
if((i__17825__auto___21590 < len__17824__auto___21589)){
args21519.push((arguments[i__17825__auto___21590]));

var G__21591 = (i__17825__auto___21590 + (1));
i__17825__auto___21590 = G__21591;
continue;
} else {
}
break;
}

var G__21521 = args21519.length;
switch (G__21521) {
case 2:
return cljs.core.async.partition.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.partition.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21519.length)].join('')));

}
});

cljs.core.async.partition.cljs$core$IFn$_invoke$arity$2 = (function (n,ch){
return cljs.core.async.partition.call(null,n,ch,null);
});

cljs.core.async.partition.cljs$core$IFn$_invoke$arity$3 = (function (n,ch,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21593 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21593,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21593,out){
return (function (state_21559){
var state_val_21560 = (state_21559[(1)]);
if((state_val_21560 === (7))){
var inst_21555 = (state_21559[(2)]);
var state_21559__$1 = state_21559;
var statearr_21561_21594 = state_21559__$1;
(statearr_21561_21594[(2)] = inst_21555);

(statearr_21561_21594[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (1))){
var inst_21522 = (new Array(n));
var inst_21523 = inst_21522;
var inst_21524 = (0);
var state_21559__$1 = (function (){var statearr_21562 = state_21559;
(statearr_21562[(7)] = inst_21524);

(statearr_21562[(8)] = inst_21523);

return statearr_21562;
})();
var statearr_21563_21595 = state_21559__$1;
(statearr_21563_21595[(2)] = null);

(statearr_21563_21595[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (4))){
var inst_21527 = (state_21559[(9)]);
var inst_21527__$1 = (state_21559[(2)]);
var inst_21528 = (inst_21527__$1 == null);
var inst_21529 = cljs.core.not.call(null,inst_21528);
var state_21559__$1 = (function (){var statearr_21564 = state_21559;
(statearr_21564[(9)] = inst_21527__$1);

return statearr_21564;
})();
if(inst_21529){
var statearr_21565_21596 = state_21559__$1;
(statearr_21565_21596[(1)] = (5));

} else {
var statearr_21566_21597 = state_21559__$1;
(statearr_21566_21597[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (15))){
var inst_21549 = (state_21559[(2)]);
var state_21559__$1 = state_21559;
var statearr_21567_21598 = state_21559__$1;
(statearr_21567_21598[(2)] = inst_21549);

(statearr_21567_21598[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (13))){
var state_21559__$1 = state_21559;
var statearr_21568_21599 = state_21559__$1;
(statearr_21568_21599[(2)] = null);

(statearr_21568_21599[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (6))){
var inst_21524 = (state_21559[(7)]);
var inst_21545 = (inst_21524 > (0));
var state_21559__$1 = state_21559;
if(cljs.core.truth_(inst_21545)){
var statearr_21569_21600 = state_21559__$1;
(statearr_21569_21600[(1)] = (12));

} else {
var statearr_21570_21601 = state_21559__$1;
(statearr_21570_21601[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (3))){
var inst_21557 = (state_21559[(2)]);
var state_21559__$1 = state_21559;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21559__$1,inst_21557);
} else {
if((state_val_21560 === (12))){
var inst_21523 = (state_21559[(8)]);
var inst_21547 = cljs.core.vec.call(null,inst_21523);
var state_21559__$1 = state_21559;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21559__$1,(15),out,inst_21547);
} else {
if((state_val_21560 === (2))){
var state_21559__$1 = state_21559;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21559__$1,(4),ch);
} else {
if((state_val_21560 === (11))){
var inst_21539 = (state_21559[(2)]);
var inst_21540 = (new Array(n));
var inst_21523 = inst_21540;
var inst_21524 = (0);
var state_21559__$1 = (function (){var statearr_21571 = state_21559;
(statearr_21571[(7)] = inst_21524);

(statearr_21571[(8)] = inst_21523);

(statearr_21571[(10)] = inst_21539);

return statearr_21571;
})();
var statearr_21572_21602 = state_21559__$1;
(statearr_21572_21602[(2)] = null);

(statearr_21572_21602[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (9))){
var inst_21523 = (state_21559[(8)]);
var inst_21537 = cljs.core.vec.call(null,inst_21523);
var state_21559__$1 = state_21559;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21559__$1,(11),out,inst_21537);
} else {
if((state_val_21560 === (5))){
var inst_21532 = (state_21559[(11)]);
var inst_21524 = (state_21559[(7)]);
var inst_21527 = (state_21559[(9)]);
var inst_21523 = (state_21559[(8)]);
var inst_21531 = (inst_21523[inst_21524] = inst_21527);
var inst_21532__$1 = (inst_21524 + (1));
var inst_21533 = (inst_21532__$1 < n);
var state_21559__$1 = (function (){var statearr_21573 = state_21559;
(statearr_21573[(11)] = inst_21532__$1);

(statearr_21573[(12)] = inst_21531);

return statearr_21573;
})();
if(cljs.core.truth_(inst_21533)){
var statearr_21574_21603 = state_21559__$1;
(statearr_21574_21603[(1)] = (8));

} else {
var statearr_21575_21604 = state_21559__$1;
(statearr_21575_21604[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (14))){
var inst_21552 = (state_21559[(2)]);
var inst_21553 = cljs.core.async.close_BANG_.call(null,out);
var state_21559__$1 = (function (){var statearr_21577 = state_21559;
(statearr_21577[(13)] = inst_21552);

return statearr_21577;
})();
var statearr_21578_21605 = state_21559__$1;
(statearr_21578_21605[(2)] = inst_21553);

(statearr_21578_21605[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (10))){
var inst_21543 = (state_21559[(2)]);
var state_21559__$1 = state_21559;
var statearr_21579_21606 = state_21559__$1;
(statearr_21579_21606[(2)] = inst_21543);

(statearr_21579_21606[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21560 === (8))){
var inst_21532 = (state_21559[(11)]);
var inst_21523 = (state_21559[(8)]);
var tmp21576 = inst_21523;
var inst_21523__$1 = tmp21576;
var inst_21524 = inst_21532;
var state_21559__$1 = (function (){var statearr_21580 = state_21559;
(statearr_21580[(7)] = inst_21524);

(statearr_21580[(8)] = inst_21523__$1);

return statearr_21580;
})();
var statearr_21581_21607 = state_21559__$1;
(statearr_21581_21607[(2)] = null);

(statearr_21581_21607[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21593,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21593,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21585 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_21585[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21585[(1)] = (1));

return statearr_21585;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21559){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21559);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21586){if((e21586 instanceof Object)){
var ex__18825__auto__ = e21586;
var statearr_21587_21608 = state_21559;
(statearr_21587_21608[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21559);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21586;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21609 = state_21559;
state_21559 = G__21609;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21559){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21559);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21593,out))
})();
var state__18935__auto__ = (function (){var statearr_21588 = f__18934__auto__.call(null);
(statearr_21588[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21593);

return statearr_21588;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21593,out))
);


return out;
});

cljs.core.async.partition.cljs$lang$maxFixedArity = 3;
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.partition_by = (function cljs$core$async$partition_by(var_args){
var args21610 = [];
var len__17824__auto___21684 = arguments.length;
var i__17825__auto___21685 = (0);
while(true){
if((i__17825__auto___21685 < len__17824__auto___21684)){
args21610.push((arguments[i__17825__auto___21685]));

var G__21686 = (i__17825__auto___21685 + (1));
i__17825__auto___21685 = G__21686;
continue;
} else {
}
break;
}

var G__21612 = args21610.length;
switch (G__21612) {
case 2:
return cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args21610.length)].join('')));

}
});

cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$2 = (function (f,ch){
return cljs.core.async.partition_by.call(null,f,ch,null);
});

cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$3 = (function (f,ch,buf_or_n){
var out = cljs.core.async.chan.call(null,buf_or_n);
var c__18933__auto___21688 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___21688,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___21688,out){
return (function (state_21654){
var state_val_21655 = (state_21654[(1)]);
if((state_val_21655 === (7))){
var inst_21650 = (state_21654[(2)]);
var state_21654__$1 = state_21654;
var statearr_21656_21689 = state_21654__$1;
(statearr_21656_21689[(2)] = inst_21650);

(statearr_21656_21689[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (1))){
var inst_21613 = [];
var inst_21614 = inst_21613;
var inst_21615 = new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",-69252123);
var state_21654__$1 = (function (){var statearr_21657 = state_21654;
(statearr_21657[(7)] = inst_21615);

(statearr_21657[(8)] = inst_21614);

return statearr_21657;
})();
var statearr_21658_21690 = state_21654__$1;
(statearr_21658_21690[(2)] = null);

(statearr_21658_21690[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (4))){
var inst_21618 = (state_21654[(9)]);
var inst_21618__$1 = (state_21654[(2)]);
var inst_21619 = (inst_21618__$1 == null);
var inst_21620 = cljs.core.not.call(null,inst_21619);
var state_21654__$1 = (function (){var statearr_21659 = state_21654;
(statearr_21659[(9)] = inst_21618__$1);

return statearr_21659;
})();
if(inst_21620){
var statearr_21660_21691 = state_21654__$1;
(statearr_21660_21691[(1)] = (5));

} else {
var statearr_21661_21692 = state_21654__$1;
(statearr_21661_21692[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (15))){
var inst_21644 = (state_21654[(2)]);
var state_21654__$1 = state_21654;
var statearr_21662_21693 = state_21654__$1;
(statearr_21662_21693[(2)] = inst_21644);

(statearr_21662_21693[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (13))){
var state_21654__$1 = state_21654;
var statearr_21663_21694 = state_21654__$1;
(statearr_21663_21694[(2)] = null);

(statearr_21663_21694[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (6))){
var inst_21614 = (state_21654[(8)]);
var inst_21639 = inst_21614.length;
var inst_21640 = (inst_21639 > (0));
var state_21654__$1 = state_21654;
if(cljs.core.truth_(inst_21640)){
var statearr_21664_21695 = state_21654__$1;
(statearr_21664_21695[(1)] = (12));

} else {
var statearr_21665_21696 = state_21654__$1;
(statearr_21665_21696[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (3))){
var inst_21652 = (state_21654[(2)]);
var state_21654__$1 = state_21654;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_21654__$1,inst_21652);
} else {
if((state_val_21655 === (12))){
var inst_21614 = (state_21654[(8)]);
var inst_21642 = cljs.core.vec.call(null,inst_21614);
var state_21654__$1 = state_21654;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21654__$1,(15),out,inst_21642);
} else {
if((state_val_21655 === (2))){
var state_21654__$1 = state_21654;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_21654__$1,(4),ch);
} else {
if((state_val_21655 === (11))){
var inst_21618 = (state_21654[(9)]);
var inst_21622 = (state_21654[(10)]);
var inst_21632 = (state_21654[(2)]);
var inst_21633 = [];
var inst_21634 = inst_21633.push(inst_21618);
var inst_21614 = inst_21633;
var inst_21615 = inst_21622;
var state_21654__$1 = (function (){var statearr_21666 = state_21654;
(statearr_21666[(11)] = inst_21632);

(statearr_21666[(12)] = inst_21634);

(statearr_21666[(7)] = inst_21615);

(statearr_21666[(8)] = inst_21614);

return statearr_21666;
})();
var statearr_21667_21697 = state_21654__$1;
(statearr_21667_21697[(2)] = null);

(statearr_21667_21697[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (9))){
var inst_21614 = (state_21654[(8)]);
var inst_21630 = cljs.core.vec.call(null,inst_21614);
var state_21654__$1 = state_21654;
return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_21654__$1,(11),out,inst_21630);
} else {
if((state_val_21655 === (5))){
var inst_21618 = (state_21654[(9)]);
var inst_21615 = (state_21654[(7)]);
var inst_21622 = (state_21654[(10)]);
var inst_21622__$1 = f.call(null,inst_21618);
var inst_21623 = cljs.core._EQ_.call(null,inst_21622__$1,inst_21615);
var inst_21624 = cljs.core.keyword_identical_QMARK_.call(null,inst_21615,new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",-69252123));
var inst_21625 = (inst_21623) || (inst_21624);
var state_21654__$1 = (function (){var statearr_21668 = state_21654;
(statearr_21668[(10)] = inst_21622__$1);

return statearr_21668;
})();
if(cljs.core.truth_(inst_21625)){
var statearr_21669_21698 = state_21654__$1;
(statearr_21669_21698[(1)] = (8));

} else {
var statearr_21670_21699 = state_21654__$1;
(statearr_21670_21699[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (14))){
var inst_21647 = (state_21654[(2)]);
var inst_21648 = cljs.core.async.close_BANG_.call(null,out);
var state_21654__$1 = (function (){var statearr_21672 = state_21654;
(statearr_21672[(13)] = inst_21647);

return statearr_21672;
})();
var statearr_21673_21700 = state_21654__$1;
(statearr_21673_21700[(2)] = inst_21648);

(statearr_21673_21700[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (10))){
var inst_21637 = (state_21654[(2)]);
var state_21654__$1 = state_21654;
var statearr_21674_21701 = state_21654__$1;
(statearr_21674_21701[(2)] = inst_21637);

(statearr_21674_21701[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_21655 === (8))){
var inst_21618 = (state_21654[(9)]);
var inst_21614 = (state_21654[(8)]);
var inst_21622 = (state_21654[(10)]);
var inst_21627 = inst_21614.push(inst_21618);
var tmp21671 = inst_21614;
var inst_21614__$1 = tmp21671;
var inst_21615 = inst_21622;
var state_21654__$1 = (function (){var statearr_21675 = state_21654;
(statearr_21675[(14)] = inst_21627);

(statearr_21675[(7)] = inst_21615);

(statearr_21675[(8)] = inst_21614__$1);

return statearr_21675;
})();
var statearr_21676_21702 = state_21654__$1;
(statearr_21676_21702[(2)] = null);

(statearr_21676_21702[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__18933__auto___21688,out))
;
return ((function (switch__18821__auto__,c__18933__auto___21688,out){
return (function() {
var cljs$core$async$state_machine__18822__auto__ = null;
var cljs$core$async$state_machine__18822__auto____0 = (function (){
var statearr_21680 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_21680[(0)] = cljs$core$async$state_machine__18822__auto__);

(statearr_21680[(1)] = (1));

return statearr_21680;
});
var cljs$core$async$state_machine__18822__auto____1 = (function (state_21654){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_21654);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e21681){if((e21681 instanceof Object)){
var ex__18825__auto__ = e21681;
var statearr_21682_21703 = state_21654;
(statearr_21682_21703[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_21654);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e21681;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__21704 = state_21654;
state_21654 = G__21704;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
cljs$core$async$state_machine__18822__auto__ = function(state_21654){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__18822__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__18822__auto____1.call(this,state_21654);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__18822__auto____0;
cljs$core$async$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__18822__auto____1;
return cljs$core$async$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___21688,out))
})();
var state__18935__auto__ = (function (){var statearr_21683 = f__18934__auto__.call(null);
(statearr_21683[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___21688);

return statearr_21683;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___21688,out))
);


return out;
});

cljs.core.async.partition_by.cljs$lang$maxFixedArity = 3;

//# sourceMappingURL=async.js.map?rel=1454621288670