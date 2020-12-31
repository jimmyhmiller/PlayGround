goog.provide('cljs.core.async');
cljs.core.async.fn_handler = (function cljs$core$async$fn_handler(var_args){
var G__39525 = arguments.length;
switch (G__39525) {
case 1:
return cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1 = (function (f){
return cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2(f,true);
}));

(cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2 = (function (f,blockable){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async39526 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async39526 = (function (f,blockable,meta39527){
this.f = f;
this.blockable = blockable;
this.meta39527 = meta39527;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_39528,meta39527__$1){
var self__ = this;
var _39528__$1 = this;
return (new cljs.core.async.t_cljs$core$async39526(self__.f,self__.blockable,meta39527__$1));
}));

(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_39528){
var self__ = this;
var _39528__$1 = this;
return self__.meta39527;
}));

(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$async$impl$protocols$Handler$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return true;
}));

(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.blockable;
}));

(cljs.core.async.t_cljs$core$async39526.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.f;
}));

(cljs.core.async.t_cljs$core$async39526.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"blockable","blockable",-28395259,null),new cljs.core.Symbol(null,"meta39527","meta39527",-626072693,null)], null);
}));

(cljs.core.async.t_cljs$core$async39526.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async39526.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async39526");

(cljs.core.async.t_cljs$core$async39526.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async39526");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async39526.
 */
cljs.core.async.__GT_t_cljs$core$async39526 = (function cljs$core$async$__GT_t_cljs$core$async39526(f__$1,blockable__$1,meta39527){
return (new cljs.core.async.t_cljs$core$async39526(f__$1,blockable__$1,meta39527));
});

}

return (new cljs.core.async.t_cljs$core$async39526(f,blockable,cljs.core.PersistentArrayMap.EMPTY));
}));

(cljs.core.async.fn_handler.cljs$lang$maxFixedArity = 2);

/**
 * Returns a fixed buffer of size n. When full, puts will block/park.
 */
cljs.core.async.buffer = (function cljs$core$async$buffer(n){
return cljs.core.async.impl.buffers.fixed_buffer(n);
});
/**
 * Returns a buffer of size n. When full, puts will complete but
 *   val will be dropped (no transfer).
 */
cljs.core.async.dropping_buffer = (function cljs$core$async$dropping_buffer(n){
return cljs.core.async.impl.buffers.dropping_buffer(n);
});
/**
 * Returns a buffer of size n. When full, puts will complete, and be
 *   buffered, but oldest elements in buffer will be dropped (not
 *   transferred).
 */
cljs.core.async.sliding_buffer = (function cljs$core$async$sliding_buffer(n){
return cljs.core.async.impl.buffers.sliding_buffer(n);
});
/**
 * Returns true if a channel created with buff will never block. That is to say,
 * puts into this buffer will never cause the buffer to be full. 
 */
cljs.core.async.unblocking_buffer_QMARK_ = (function cljs$core$async$unblocking_buffer_QMARK_(buff){
if((!((buff == null)))){
if(((false) || ((cljs.core.PROTOCOL_SENTINEL === buff.cljs$core$async$impl$protocols$UnblockingBuffer$)))){
return true;
} else {
if((!buff.cljs$lang$protocol_mask$partition$)){
return cljs.core.native_satisfies_QMARK_(cljs.core.async.impl.protocols.UnblockingBuffer,buff);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_(cljs.core.async.impl.protocols.UnblockingBuffer,buff);
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
var G__39545 = arguments.length;
switch (G__39545) {
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
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.chan.cljs$core$IFn$_invoke$arity$0 = (function (){
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(null);
}));

(cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1 = (function (buf_or_n){
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3(buf_or_n,null,null);
}));

(cljs.core.async.chan.cljs$core$IFn$_invoke$arity$2 = (function (buf_or_n,xform){
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3(buf_or_n,xform,null);
}));

(cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3 = (function (buf_or_n,xform,ex_handler){
var buf_or_n__$1 = ((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(buf_or_n,(0)))?null:buf_or_n);
if(cljs.core.truth_(xform)){
if(cljs.core.truth_(buf_or_n__$1)){
} else {
throw (new Error(["Assert failed: ","buffer must be supplied when transducer is","\n","buf-or-n"].join('')));
}
} else {
}

return cljs.core.async.impl.channels.chan.cljs$core$IFn$_invoke$arity$3(((typeof buf_or_n__$1 === 'number')?cljs.core.async.buffer(buf_or_n__$1):buf_or_n__$1),xform,ex_handler);
}));

(cljs.core.async.chan.cljs$lang$maxFixedArity = 3);

/**
 * Creates a promise channel with an optional transducer, and an optional
 *   exception-handler. A promise channel can take exactly one value that consumers
 *   will receive. Once full, puts complete but val is dropped (no transfer).
 *   Consumers will block until either a value is placed in the channel or the
 *   channel is closed. See chan for the semantics of xform and ex-handler.
 */
cljs.core.async.promise_chan = (function cljs$core$async$promise_chan(var_args){
var G__39550 = arguments.length;
switch (G__39550) {
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
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$0 = (function (){
return cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$1(null);
}));

(cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$1 = (function (xform){
return cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$2(xform,null);
}));

(cljs.core.async.promise_chan.cljs$core$IFn$_invoke$arity$2 = (function (xform,ex_handler){
return cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3(cljs.core.async.impl.buffers.promise_buffer(),xform,ex_handler);
}));

(cljs.core.async.promise_chan.cljs$lang$maxFixedArity = 2);

/**
 * Returns a channel that will close after msecs
 */
cljs.core.async.timeout = (function cljs$core$async$timeout(msecs){
return cljs.core.async.impl.timers.timeout(msecs);
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
var G__39556 = arguments.length;
switch (G__39556) {
case 2:
return cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (port,fn1){
return cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$3(port,fn1,true);
}));

(cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (port,fn1,on_caller_QMARK_){
var ret = cljs.core.async.impl.protocols.take_BANG_(port,cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1(fn1));
if(cljs.core.truth_(ret)){
var val_42343 = cljs.core.deref(ret);
if(cljs.core.truth_(on_caller_QMARK_)){
(fn1.cljs$core$IFn$_invoke$arity$1 ? fn1.cljs$core$IFn$_invoke$arity$1(val_42343) : fn1.call(null,val_42343));
} else {
cljs.core.async.impl.dispatch.run((function (){
return (fn1.cljs$core$IFn$_invoke$arity$1 ? fn1.cljs$core$IFn$_invoke$arity$1(val_42343) : fn1.call(null,val_42343));
}));
}
} else {
}

return null;
}));

(cljs.core.async.take_BANG_.cljs$lang$maxFixedArity = 3);

cljs.core.async.nop = (function cljs$core$async$nop(_){
return null;
});
cljs.core.async.fhnop = cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1(cljs.core.async.nop);
/**
 * puts a val into port. nil values are not allowed. Must be called
 *   inside a (go ...) block. Will park if no buffer space is available.
 *   Returns true unless port is already closed.
 */
cljs.core.async._GT__BANG_ = (function cljs$core$async$_GT__BANG_(port,val){
throw (new Error(">! used not in (go ...) block"));
});
/**
 * Asynchronously puts a val into port, calling fn1 (if supplied) when
 * complete. nil values are not allowed. Will throw if closed. If
 * on-caller? (default true) is true, and the put is immediately
 * accepted, will call fn1 on calling thread.  Returns nil.
 */
cljs.core.async.put_BANG_ = (function cljs$core$async$put_BANG_(var_args){
var G__39569 = arguments.length;
switch (G__39569) {
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
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (port,val){
var temp__5733__auto__ = cljs.core.async.impl.protocols.put_BANG_(port,val,cljs.core.async.fhnop);
if(cljs.core.truth_(temp__5733__auto__)){
var ret = temp__5733__auto__;
return cljs.core.deref(ret);
} else {
return true;
}
}));

(cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (port,val,fn1){
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$4(port,val,fn1,true);
}));

(cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$4 = (function (port,val,fn1,on_caller_QMARK_){
var temp__5733__auto__ = cljs.core.async.impl.protocols.put_BANG_(port,val,cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$1(fn1));
if(cljs.core.truth_(temp__5733__auto__)){
var retb = temp__5733__auto__;
var ret = cljs.core.deref(retb);
if(cljs.core.truth_(on_caller_QMARK_)){
(fn1.cljs$core$IFn$_invoke$arity$1 ? fn1.cljs$core$IFn$_invoke$arity$1(ret) : fn1.call(null,ret));
} else {
cljs.core.async.impl.dispatch.run((function (){
return (fn1.cljs$core$IFn$_invoke$arity$1 ? fn1.cljs$core$IFn$_invoke$arity$1(ret) : fn1.call(null,ret));
}));
}

return ret;
} else {
return true;
}
}));

(cljs.core.async.put_BANG_.cljs$lang$maxFixedArity = 4);

cljs.core.async.close_BANG_ = (function cljs$core$async$close_BANG_(port){
return cljs.core.async.impl.protocols.close_BANG_(port);
});
cljs.core.async.random_array = (function cljs$core$async$random_array(n){
var a = (new Array(n));
var n__4613__auto___42351 = n;
var x_42352 = (0);
while(true){
if((x_42352 < n__4613__auto___42351)){
(a[x_42352] = x_42352);

var G__42353 = (x_42352 + (1));
x_42352 = G__42353;
continue;
} else {
}
break;
}

goog.array.shuffle(a);

return a;
});
cljs.core.async.alt_flag = (function cljs$core$async$alt_flag(){
var flag = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(true);
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async39592 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async39592 = (function (flag,meta39593){
this.flag = flag;
this.meta39593 = meta39593;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_39594,meta39593__$1){
var self__ = this;
var _39594__$1 = this;
return (new cljs.core.async.t_cljs$core$async39592(self__.flag,meta39593__$1));
}));

(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_39594){
var self__ = this;
var _39594__$1 = this;
return self__.meta39593;
}));

(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$async$impl$protocols$Handler$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.deref(self__.flag);
}));

(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return true;
}));

(cljs.core.async.t_cljs$core$async39592.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_(self__.flag,null);

return true;
}));

(cljs.core.async.t_cljs$core$async39592.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"flag","flag",-1565787888,null),new cljs.core.Symbol(null,"meta39593","meta39593",-1933470617,null)], null);
}));

(cljs.core.async.t_cljs$core$async39592.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async39592.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async39592");

(cljs.core.async.t_cljs$core$async39592.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async39592");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async39592.
 */
cljs.core.async.__GT_t_cljs$core$async39592 = (function cljs$core$async$alt_flag_$___GT_t_cljs$core$async39592(flag__$1,meta39593){
return (new cljs.core.async.t_cljs$core$async39592(flag__$1,meta39593));
});

}

return (new cljs.core.async.t_cljs$core$async39592(flag,cljs.core.PersistentArrayMap.EMPTY));
});
cljs.core.async.alt_handler = (function cljs$core$async$alt_handler(flag,cb){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async39612 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async39612 = (function (flag,cb,meta39613){
this.flag = flag;
this.cb = cb;
this.meta39613 = meta39613;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_39614,meta39613__$1){
var self__ = this;
var _39614__$1 = this;
return (new cljs.core.async.t_cljs$core$async39612(self__.flag,self__.cb,meta39613__$1));
}));

(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_39614){
var self__ = this;
var _39614__$1 = this;
return self__.meta39613;
}));

(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$async$impl$protocols$Handler$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.active_QMARK_(self__.flag);
}));

(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return true;
}));

(cljs.core.async.t_cljs$core$async39612.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.async.impl.protocols.commit(self__.flag);

return self__.cb;
}));

(cljs.core.async.t_cljs$core$async39612.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"flag","flag",-1565787888,null),new cljs.core.Symbol(null,"cb","cb",-2064487928,null),new cljs.core.Symbol(null,"meta39613","meta39613",-608648641,null)], null);
}));

(cljs.core.async.t_cljs$core$async39612.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async39612.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async39612");

(cljs.core.async.t_cljs$core$async39612.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async39612");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async39612.
 */
cljs.core.async.__GT_t_cljs$core$async39612 = (function cljs$core$async$alt_handler_$___GT_t_cljs$core$async39612(flag__$1,cb__$1,meta39613){
return (new cljs.core.async.t_cljs$core$async39612(flag__$1,cb__$1,meta39613));
});

}

return (new cljs.core.async.t_cljs$core$async39612(flag,cb,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * returns derefable [val port] if immediate, nil if enqueued
 */
cljs.core.async.do_alts = (function cljs$core$async$do_alts(fret,ports,opts){
if((cljs.core.count(ports) > (0))){
} else {
throw (new Error(["Assert failed: ","alts must have at least one channel operation","\n","(pos? (count ports))"].join('')));
}

var flag = cljs.core.async.alt_flag();
var n = cljs.core.count(ports);
var idxs = cljs.core.async.random_array(n);
var priority = new cljs.core.Keyword(null,"priority","priority",1431093715).cljs$core$IFn$_invoke$arity$1(opts);
var ret = (function (){var i = (0);
while(true){
if((i < n)){
var idx = (cljs.core.truth_(priority)?i:(idxs[i]));
var port = cljs.core.nth.cljs$core$IFn$_invoke$arity$2(ports,idx);
var wport = ((cljs.core.vector_QMARK_(port))?(port.cljs$core$IFn$_invoke$arity$1 ? port.cljs$core$IFn$_invoke$arity$1((0)) : port.call(null,(0))):null);
var vbox = (cljs.core.truth_(wport)?(function (){var val = (port.cljs$core$IFn$_invoke$arity$1 ? port.cljs$core$IFn$_invoke$arity$1((1)) : port.call(null,(1)));
return cljs.core.async.impl.protocols.put_BANG_(wport,val,cljs.core.async.alt_handler(flag,((function (i,val,idx,port,wport,flag,n,idxs,priority){
return (function (p1__39626_SHARP_){
var G__39635 = new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__39626_SHARP_,wport], null);
return (fret.cljs$core$IFn$_invoke$arity$1 ? fret.cljs$core$IFn$_invoke$arity$1(G__39635) : fret.call(null,G__39635));
});})(i,val,idx,port,wport,flag,n,idxs,priority))
));
})():cljs.core.async.impl.protocols.take_BANG_(port,cljs.core.async.alt_handler(flag,((function (i,idx,port,wport,flag,n,idxs,priority){
return (function (p1__39627_SHARP_){
var G__39636 = new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__39627_SHARP_,port], null);
return (fret.cljs$core$IFn$_invoke$arity$1 ? fret.cljs$core$IFn$_invoke$arity$1(G__39636) : fret.call(null,G__39636));
});})(i,idx,port,wport,flag,n,idxs,priority))
)));
if(cljs.core.truth_(vbox)){
return cljs.core.async.impl.channels.box(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.deref(vbox),(function (){var or__4126__auto__ = wport;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return port;
}
})()], null));
} else {
var G__42359 = (i + (1));
i = G__42359;
continue;
}
} else {
return null;
}
break;
}
})();
var or__4126__auto__ = ret;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
if(cljs.core.contains_QMARK_(opts,new cljs.core.Keyword(null,"default","default",-1987822328))){
var temp__5735__auto__ = (function (){var and__4115__auto__ = flag.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1(null);
if(cljs.core.truth_(and__4115__auto__)){
return flag.cljs$core$async$impl$protocols$Handler$commit$arity$1(null);
} else {
return and__4115__auto__;
}
})();
if(cljs.core.truth_(temp__5735__auto__)){
var got = temp__5735__auto__;
return cljs.core.async.impl.channels.box(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"default","default",-1987822328).cljs$core$IFn$_invoke$arity$1(opts),new cljs.core.Keyword(null,"default","default",-1987822328)], null));
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
var args__4742__auto__ = [];
var len__4736__auto___42364 = arguments.length;
var i__4737__auto___42365 = (0);
while(true){
if((i__4737__auto___42365 < len__4736__auto___42364)){
args__4742__auto__.push((arguments[i__4737__auto___42365]));

var G__42366 = (i__4737__auto___42365 + (1));
i__4737__auto___42365 = G__42366;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return cljs.core.async.alts_BANG_.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(cljs.core.async.alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = (function (ports,p__39643){
var map__39644 = p__39643;
var map__39644__$1 = (((((!((map__39644 == null))))?(((((map__39644.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__39644.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__39644):map__39644);
var opts = map__39644__$1;
throw (new Error("alts! used not in (go ...) block"));
}));

(cljs.core.async.alts_BANG_.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(cljs.core.async.alts_BANG_.cljs$lang$applyTo = (function (seq39639){
var G__39640 = cljs.core.first(seq39639);
var seq39639__$1 = cljs.core.next(seq39639);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__39640,seq39639__$1);
}));

/**
 * Puts a val into port if it's possible to do so immediately.
 *   nil values are not allowed. Never blocks. Returns true if offer succeeds.
 */
cljs.core.async.offer_BANG_ = (function cljs$core$async$offer_BANG_(port,val){
var ret = cljs.core.async.impl.protocols.put_BANG_(port,val,cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2(cljs.core.async.nop,false));
if(cljs.core.truth_(ret)){
return cljs.core.deref(ret);
} else {
return null;
}
});
/**
 * Takes a val from port if it's possible to do so immediately.
 *   Never blocks. Returns value if successful, nil otherwise.
 */
cljs.core.async.poll_BANG_ = (function cljs$core$async$poll_BANG_(port){
var ret = cljs.core.async.impl.protocols.take_BANG_(port,cljs.core.async.fn_handler.cljs$core$IFn$_invoke$arity$2(cljs.core.async.nop,false));
if(cljs.core.truth_(ret)){
return cljs.core.deref(ret);
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
var G__39664 = arguments.length;
switch (G__39664) {
case 2:
return cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$2 = (function (from,to){
return cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$3(from,to,true);
}));

(cljs.core.async.pipe.cljs$core$IFn$_invoke$arity$3 = (function (from,to,close_QMARK_){
var c__39440__auto___42376 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_39720){
var state_val_39721 = (state_39720[(1)]);
if((state_val_39721 === (7))){
var inst_39706 = (state_39720[(2)]);
var state_39720__$1 = state_39720;
var statearr_39747_42380 = state_39720__$1;
(statearr_39747_42380[(2)] = inst_39706);

(statearr_39747_42380[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (1))){
var state_39720__$1 = state_39720;
var statearr_39748_42384 = state_39720__$1;
(statearr_39748_42384[(2)] = null);

(statearr_39748_42384[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (4))){
var inst_39680 = (state_39720[(7)]);
var inst_39680__$1 = (state_39720[(2)]);
var inst_39689 = (inst_39680__$1 == null);
var state_39720__$1 = (function (){var statearr_39754 = state_39720;
(statearr_39754[(7)] = inst_39680__$1);

return statearr_39754;
})();
if(cljs.core.truth_(inst_39689)){
var statearr_39755_42387 = state_39720__$1;
(statearr_39755_42387[(1)] = (5));

} else {
var statearr_39756_42388 = state_39720__$1;
(statearr_39756_42388[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (13))){
var state_39720__$1 = state_39720;
var statearr_39758_42391 = state_39720__$1;
(statearr_39758_42391[(2)] = null);

(statearr_39758_42391[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (6))){
var inst_39680 = (state_39720[(7)]);
var state_39720__$1 = state_39720;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_39720__$1,(11),to,inst_39680);
} else {
if((state_val_39721 === (3))){
var inst_39712 = (state_39720[(2)]);
var state_39720__$1 = state_39720;
return cljs.core.async.impl.ioc_helpers.return_chan(state_39720__$1,inst_39712);
} else {
if((state_val_39721 === (12))){
var state_39720__$1 = state_39720;
var statearr_39765_42392 = state_39720__$1;
(statearr_39765_42392[(2)] = null);

(statearr_39765_42392[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (2))){
var state_39720__$1 = state_39720;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_39720__$1,(4),from);
} else {
if((state_val_39721 === (11))){
var inst_39699 = (state_39720[(2)]);
var state_39720__$1 = state_39720;
if(cljs.core.truth_(inst_39699)){
var statearr_39769_42394 = state_39720__$1;
(statearr_39769_42394[(1)] = (12));

} else {
var statearr_39770_42395 = state_39720__$1;
(statearr_39770_42395[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (9))){
var state_39720__$1 = state_39720;
var statearr_39771_42397 = state_39720__$1;
(statearr_39771_42397[(2)] = null);

(statearr_39771_42397[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (5))){
var state_39720__$1 = state_39720;
if(cljs.core.truth_(close_QMARK_)){
var statearr_39772_42398 = state_39720__$1;
(statearr_39772_42398[(1)] = (8));

} else {
var statearr_39773_42399 = state_39720__$1;
(statearr_39773_42399[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (14))){
var inst_39704 = (state_39720[(2)]);
var state_39720__$1 = state_39720;
var statearr_39777_42400 = state_39720__$1;
(statearr_39777_42400[(2)] = inst_39704);

(statearr_39777_42400[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (10))){
var inst_39696 = (state_39720[(2)]);
var state_39720__$1 = state_39720;
var statearr_39778_42401 = state_39720__$1;
(statearr_39778_42401[(2)] = inst_39696);

(statearr_39778_42401[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39721 === (8))){
var inst_39693 = cljs.core.async.close_BANG_(to);
var state_39720__$1 = state_39720;
var statearr_39780_42402 = state_39720__$1;
(statearr_39780_42402[(2)] = inst_39693);

(statearr_39780_42402[(1)] = (10));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_39785 = [null,null,null,null,null,null,null,null];
(statearr_39785[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_39785[(1)] = (1));

return statearr_39785;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_39720){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_39720);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e39786){var ex__39237__auto__ = e39786;
var statearr_39787_42410 = state_39720;
(statearr_39787_42410[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_39720[(4)]))){
var statearr_39788_42411 = state_39720;
(statearr_39788_42411[(1)] = cljs.core.first((state_39720[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42412 = state_39720;
state_39720 = G__42412;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_39720){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_39720);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_39792 = f__39441__auto__();
(statearr_39792[(6)] = c__39440__auto___42376);

return statearr_39792;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return to;
}));

(cljs.core.async.pipe.cljs$lang$maxFixedArity = 3);

cljs.core.async.pipeline_STAR_ = (function cljs$core$async$pipeline_STAR_(n,to,xf,from,close_QMARK_,ex_handler,type){
if((n > (0))){
} else {
throw (new Error("Assert failed: (pos? n)"));
}

var jobs = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(n);
var results = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(n);
var process = (function (p__39793){
var vec__39794 = p__39793;
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__39794,(0),null);
var p = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__39794,(1),null);
var job = vec__39794;
if((job == null)){
cljs.core.async.close_BANG_(results);

return null;
} else {
var res = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$3((1),xf,ex_handler);
var c__39440__auto___42417 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_39801){
var state_val_39802 = (state_39801[(1)]);
if((state_val_39802 === (1))){
var state_39801__$1 = state_39801;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_39801__$1,(2),res,v);
} else {
if((state_val_39802 === (2))){
var inst_39798 = (state_39801[(2)]);
var inst_39799 = cljs.core.async.close_BANG_(res);
var state_39801__$1 = (function (){var statearr_39805 = state_39801;
(statearr_39805[(7)] = inst_39798);

return statearr_39805;
})();
return cljs.core.async.impl.ioc_helpers.return_chan(state_39801__$1,inst_39799);
} else {
return null;
}
}
});
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_39807 = [null,null,null,null,null,null,null,null];
(statearr_39807[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__);

(statearr_39807[(1)] = (1));

return statearr_39807;
});
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1 = (function (state_39801){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_39801);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e39808){var ex__39237__auto__ = e39808;
var statearr_39809_42426 = state_39801;
(statearr_39809_42426[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_39801[(4)]))){
var statearr_39810_42427 = state_39801;
(statearr_39810_42427[(1)] = cljs.core.first((state_39801[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42432 = state_39801;
state_39801 = G__42432;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = function(state_39801){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1.call(this,state_39801);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_39815 = f__39441__auto__();
(statearr_39815[(6)] = c__39440__auto___42417);

return statearr_39815;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(p,res);

return true;
}
});
var async = (function (p__39818){
var vec__39819 = p__39818;
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__39819,(0),null);
var p = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__39819,(1),null);
var job = vec__39819;
if((job == null)){
cljs.core.async.close_BANG_(results);

return null;
} else {
var res = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
(xf.cljs$core$IFn$_invoke$arity$2 ? xf.cljs$core$IFn$_invoke$arity$2(v,res) : xf.call(null,v,res));

cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(p,res);

return true;
}
});
var n__4613__auto___42433 = n;
var __42434 = (0);
while(true){
if((__42434 < n__4613__auto___42433)){
var G__39822_42436 = type;
var G__39822_42437__$1 = (((G__39822_42436 instanceof cljs.core.Keyword))?G__39822_42436.fqn:null);
switch (G__39822_42437__$1) {
case "compute":
var c__39440__auto___42439 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run(((function (__42434,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = ((function (__42434,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function (state_39839){
var state_val_39840 = (state_39839[(1)]);
if((state_val_39840 === (1))){
var state_39839__$1 = state_39839;
var statearr_39842_42443 = state_39839__$1;
(statearr_39842_42443[(2)] = null);

(statearr_39842_42443[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39840 === (2))){
var state_39839__$1 = state_39839;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_39839__$1,(4),jobs);
} else {
if((state_val_39840 === (3))){
var inst_39834 = (state_39839[(2)]);
var state_39839__$1 = state_39839;
return cljs.core.async.impl.ioc_helpers.return_chan(state_39839__$1,inst_39834);
} else {
if((state_val_39840 === (4))){
var inst_39825 = (state_39839[(2)]);
var inst_39826 = process(inst_39825);
var state_39839__$1 = state_39839;
if(cljs.core.truth_(inst_39826)){
var statearr_39846_42446 = state_39839__$1;
(statearr_39846_42446[(1)] = (5));

} else {
var statearr_39847_42447 = state_39839__$1;
(statearr_39847_42447[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39840 === (5))){
var state_39839__$1 = state_39839;
var statearr_39849_42448 = state_39839__$1;
(statearr_39849_42448[(2)] = null);

(statearr_39849_42448[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39840 === (6))){
var state_39839__$1 = state_39839;
var statearr_39853_42449 = state_39839__$1;
(statearr_39853_42449[(2)] = null);

(statearr_39853_42449[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39840 === (7))){
var inst_39832 = (state_39839[(2)]);
var state_39839__$1 = state_39839;
var statearr_39857_42451 = state_39839__$1;
(statearr_39857_42451[(2)] = inst_39832);

(statearr_39857_42451[(1)] = (3));


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
});})(__42434,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
;
return ((function (__42434,switch__39233__auto__,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_39858 = [null,null,null,null,null,null,null];
(statearr_39858[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__);

(statearr_39858[(1)] = (1));

return statearr_39858;
});
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1 = (function (state_39839){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_39839);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e39860){var ex__39237__auto__ = e39860;
var statearr_39862_42452 = state_39839;
(statearr_39862_42452[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_39839[(4)]))){
var statearr_39864_42453 = state_39839;
(statearr_39864_42453[(1)] = cljs.core.first((state_39839[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42454 = state_39839;
state_39839 = G__42454;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = function(state_39839){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1.call(this,state_39839);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__;
})()
;})(__42434,switch__39233__auto__,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
})();
var state__39442__auto__ = (function (){var statearr_39866 = f__39441__auto__();
(statearr_39866[(6)] = c__39440__auto___42439);

return statearr_39866;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
});})(__42434,c__39440__auto___42439,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
);


break;
case "async":
var c__39440__auto___42455 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run(((function (__42434,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = ((function (__42434,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function (state_39883){
var state_val_39884 = (state_39883[(1)]);
if((state_val_39884 === (1))){
var state_39883__$1 = state_39883;
var statearr_39887_42456 = state_39883__$1;
(statearr_39887_42456[(2)] = null);

(statearr_39887_42456[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39884 === (2))){
var state_39883__$1 = state_39883;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_39883__$1,(4),jobs);
} else {
if((state_val_39884 === (3))){
var inst_39877 = (state_39883[(2)]);
var state_39883__$1 = state_39883;
return cljs.core.async.impl.ioc_helpers.return_chan(state_39883__$1,inst_39877);
} else {
if((state_val_39884 === (4))){
var inst_39869 = (state_39883[(2)]);
var inst_39870 = async(inst_39869);
var state_39883__$1 = state_39883;
if(cljs.core.truth_(inst_39870)){
var statearr_39894_42461 = state_39883__$1;
(statearr_39894_42461[(1)] = (5));

} else {
var statearr_39896_42462 = state_39883__$1;
(statearr_39896_42462[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39884 === (5))){
var state_39883__$1 = state_39883;
var statearr_39900_42463 = state_39883__$1;
(statearr_39900_42463[(2)] = null);

(statearr_39900_42463[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39884 === (6))){
var state_39883__$1 = state_39883;
var statearr_39903_42464 = state_39883__$1;
(statearr_39903_42464[(2)] = null);

(statearr_39903_42464[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39884 === (7))){
var inst_39875 = (state_39883[(2)]);
var state_39883__$1 = state_39883;
var statearr_39907_42465 = state_39883__$1;
(statearr_39907_42465[(2)] = inst_39875);

(statearr_39907_42465[(1)] = (3));


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
});})(__42434,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
;
return ((function (__42434,switch__39233__auto__,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async){
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_39911 = [null,null,null,null,null,null,null];
(statearr_39911[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__);

(statearr_39911[(1)] = (1));

return statearr_39911;
});
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1 = (function (state_39883){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_39883);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e39916){var ex__39237__auto__ = e39916;
var statearr_39918_42467 = state_39883;
(statearr_39918_42467[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_39883[(4)]))){
var statearr_39922_42468 = state_39883;
(statearr_39922_42468[(1)] = cljs.core.first((state_39883[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42469 = state_39883;
state_39883 = G__42469;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = function(state_39883){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1.call(this,state_39883);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__;
})()
;})(__42434,switch__39233__auto__,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
})();
var state__39442__auto__ = (function (){var statearr_39924 = f__39441__auto__();
(statearr_39924[(6)] = c__39440__auto___42455);

return statearr_39924;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
});})(__42434,c__39440__auto___42455,G__39822_42436,G__39822_42437__$1,n__4613__auto___42433,jobs,results,process,async))
);


break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__39822_42437__$1)].join('')));

}

var G__42470 = (__42434 + (1));
__42434 = G__42470;
continue;
} else {
}
break;
}

var c__39440__auto___42471 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_39957){
var state_val_39958 = (state_39957[(1)]);
if((state_val_39958 === (7))){
var inst_39953 = (state_39957[(2)]);
var state_39957__$1 = state_39957;
var statearr_39977_42472 = state_39957__$1;
(statearr_39977_42472[(2)] = inst_39953);

(statearr_39977_42472[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39958 === (1))){
var state_39957__$1 = state_39957;
var statearr_39994_42473 = state_39957__$1;
(statearr_39994_42473[(2)] = null);

(statearr_39994_42473[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39958 === (4))){
var inst_39936 = (state_39957[(7)]);
var inst_39936__$1 = (state_39957[(2)]);
var inst_39937 = (inst_39936__$1 == null);
var state_39957__$1 = (function (){var statearr_40004 = state_39957;
(statearr_40004[(7)] = inst_39936__$1);

return statearr_40004;
})();
if(cljs.core.truth_(inst_39937)){
var statearr_40013_42474 = state_39957__$1;
(statearr_40013_42474[(1)] = (5));

} else {
var statearr_40019_42475 = state_39957__$1;
(statearr_40019_42475[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39958 === (6))){
var inst_39942 = (state_39957[(8)]);
var inst_39936 = (state_39957[(7)]);
var inst_39942__$1 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
var inst_39944 = cljs.core.PersistentVector.EMPTY_NODE;
var inst_39945 = [inst_39936,inst_39942__$1];
var inst_39946 = (new cljs.core.PersistentVector(null,2,(5),inst_39944,inst_39945,null));
var state_39957__$1 = (function (){var statearr_40031 = state_39957;
(statearr_40031[(8)] = inst_39942__$1);

return statearr_40031;
})();
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_39957__$1,(8),jobs,inst_39946);
} else {
if((state_val_39958 === (3))){
var inst_39955 = (state_39957[(2)]);
var state_39957__$1 = state_39957;
return cljs.core.async.impl.ioc_helpers.return_chan(state_39957__$1,inst_39955);
} else {
if((state_val_39958 === (2))){
var state_39957__$1 = state_39957;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_39957__$1,(4),from);
} else {
if((state_val_39958 === (9))){
var inst_39950 = (state_39957[(2)]);
var state_39957__$1 = (function (){var statearr_40038 = state_39957;
(statearr_40038[(9)] = inst_39950);

return statearr_40038;
})();
var statearr_40040_42478 = state_39957__$1;
(statearr_40040_42478[(2)] = null);

(statearr_40040_42478[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39958 === (5))){
var inst_39939 = cljs.core.async.close_BANG_(jobs);
var state_39957__$1 = state_39957;
var statearr_40041_42479 = state_39957__$1;
(statearr_40041_42479[(2)] = inst_39939);

(statearr_40041_42479[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_39958 === (8))){
var inst_39942 = (state_39957[(8)]);
var inst_39948 = (state_39957[(2)]);
var state_39957__$1 = (function (){var statearr_40042 = state_39957;
(statearr_40042[(10)] = inst_39948);

return statearr_40042;
})();
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_39957__$1,(9),results,inst_39942);
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
});
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_40047 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_40047[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__);

(statearr_40047[(1)] = (1));

return statearr_40047;
});
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1 = (function (state_39957){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_39957);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40048){var ex__39237__auto__ = e40048;
var statearr_40049_42480 = state_39957;
(statearr_40049_42480[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_39957[(4)]))){
var statearr_40050_42481 = state_39957;
(statearr_40050_42481[(1)] = cljs.core.first((state_39957[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42482 = state_39957;
state_39957 = G__42482;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = function(state_39957){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1.call(this,state_39957);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40052 = f__39441__auto__();
(statearr_40052[(6)] = c__39440__auto___42471);

return statearr_40052;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


var c__39440__auto__ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40090){
var state_val_40091 = (state_40090[(1)]);
if((state_val_40091 === (7))){
var inst_40086 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
var statearr_40102_42490 = state_40090__$1;
(statearr_40102_42490[(2)] = inst_40086);

(statearr_40102_42490[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (20))){
var state_40090__$1 = state_40090;
var statearr_40103_42491 = state_40090__$1;
(statearr_40103_42491[(2)] = null);

(statearr_40103_42491[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (1))){
var state_40090__$1 = state_40090;
var statearr_40107_42492 = state_40090__$1;
(statearr_40107_42492[(2)] = null);

(statearr_40107_42492[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (4))){
var inst_40055 = (state_40090[(7)]);
var inst_40055__$1 = (state_40090[(2)]);
var inst_40056 = (inst_40055__$1 == null);
var state_40090__$1 = (function (){var statearr_40108 = state_40090;
(statearr_40108[(7)] = inst_40055__$1);

return statearr_40108;
})();
if(cljs.core.truth_(inst_40056)){
var statearr_40109_42500 = state_40090__$1;
(statearr_40109_42500[(1)] = (5));

} else {
var statearr_40110_42501 = state_40090__$1;
(statearr_40110_42501[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (15))){
var inst_40068 = (state_40090[(8)]);
var state_40090__$1 = state_40090;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_40090__$1,(18),to,inst_40068);
} else {
if((state_val_40091 === (21))){
var inst_40081 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
var statearr_40111_42502 = state_40090__$1;
(statearr_40111_42502[(2)] = inst_40081);

(statearr_40111_42502[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (13))){
var inst_40083 = (state_40090[(2)]);
var state_40090__$1 = (function (){var statearr_40112 = state_40090;
(statearr_40112[(9)] = inst_40083);

return statearr_40112;
})();
var statearr_40113_42508 = state_40090__$1;
(statearr_40113_42508[(2)] = null);

(statearr_40113_42508[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (6))){
var inst_40055 = (state_40090[(7)]);
var state_40090__$1 = state_40090;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40090__$1,(11),inst_40055);
} else {
if((state_val_40091 === (17))){
var inst_40076 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
if(cljs.core.truth_(inst_40076)){
var statearr_40119_42510 = state_40090__$1;
(statearr_40119_42510[(1)] = (19));

} else {
var statearr_40120_42511 = state_40090__$1;
(statearr_40120_42511[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (3))){
var inst_40088 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40090__$1,inst_40088);
} else {
if((state_val_40091 === (12))){
var inst_40065 = (state_40090[(10)]);
var state_40090__$1 = state_40090;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40090__$1,(14),inst_40065);
} else {
if((state_val_40091 === (2))){
var state_40090__$1 = state_40090;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40090__$1,(4),results);
} else {
if((state_val_40091 === (19))){
var state_40090__$1 = state_40090;
var statearr_40121_42515 = state_40090__$1;
(statearr_40121_42515[(2)] = null);

(statearr_40121_42515[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (11))){
var inst_40065 = (state_40090[(2)]);
var state_40090__$1 = (function (){var statearr_40122 = state_40090;
(statearr_40122[(10)] = inst_40065);

return statearr_40122;
})();
var statearr_40123_42520 = state_40090__$1;
(statearr_40123_42520[(2)] = null);

(statearr_40123_42520[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (9))){
var state_40090__$1 = state_40090;
var statearr_40124_42524 = state_40090__$1;
(statearr_40124_42524[(2)] = null);

(statearr_40124_42524[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (5))){
var state_40090__$1 = state_40090;
if(cljs.core.truth_(close_QMARK_)){
var statearr_40125_42525 = state_40090__$1;
(statearr_40125_42525[(1)] = (8));

} else {
var statearr_40126_42526 = state_40090__$1;
(statearr_40126_42526[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (14))){
var inst_40068 = (state_40090[(8)]);
var inst_40068__$1 = (state_40090[(2)]);
var inst_40069 = (inst_40068__$1 == null);
var inst_40070 = cljs.core.not(inst_40069);
var state_40090__$1 = (function (){var statearr_40131 = state_40090;
(statearr_40131[(8)] = inst_40068__$1);

return statearr_40131;
})();
if(inst_40070){
var statearr_40132_42527 = state_40090__$1;
(statearr_40132_42527[(1)] = (15));

} else {
var statearr_40133_42528 = state_40090__$1;
(statearr_40133_42528[(1)] = (16));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (16))){
var state_40090__$1 = state_40090;
var statearr_40143_42529 = state_40090__$1;
(statearr_40143_42529[(2)] = false);

(statearr_40143_42529[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (10))){
var inst_40062 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
var statearr_40160_42535 = state_40090__$1;
(statearr_40160_42535[(2)] = inst_40062);

(statearr_40160_42535[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (18))){
var inst_40073 = (state_40090[(2)]);
var state_40090__$1 = state_40090;
var statearr_40171_42539 = state_40090__$1;
(statearr_40171_42539[(2)] = inst_40073);

(statearr_40171_42539[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40091 === (8))){
var inst_40059 = cljs.core.async.close_BANG_(to);
var state_40090__$1 = state_40090;
var statearr_40172_42541 = state_40090__$1;
(statearr_40172_42541[(2)] = inst_40059);

(statearr_40172_42541[(1)] = (10));


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
});
return (function() {
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_40180 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_40180[(0)] = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__);

(statearr_40180[(1)] = (1));

return statearr_40180;
});
var cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1 = (function (state_40090){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40090);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40181){var ex__39237__auto__ = e40181;
var statearr_40185_42545 = state_40090;
(statearr_40185_42545[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40090[(4)]))){
var statearr_40186_42546 = state_40090;
(statearr_40186_42546[(1)] = cljs.core.first((state_40090[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42550 = state_40090;
state_40090 = G__42550;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__ = function(state_40090){
switch(arguments.length){
case 0:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1.call(this,state_40090);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____0;
cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$pipeline_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$pipeline_STAR__$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40187 = f__39441__auto__();
(statearr_40187[(6)] = c__39440__auto__);

return statearr_40187;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

return c__39440__auto__;
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
var G__40192 = arguments.length;
switch (G__40192) {
case 4:
return cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
case 5:
return cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$5((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]),(arguments[(4)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$4 = (function (n,to,af,from){
return cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$5(n,to,af,from,true);
}));

(cljs.core.async.pipeline_async.cljs$core$IFn$_invoke$arity$5 = (function (n,to,af,from,close_QMARK_){
return cljs.core.async.pipeline_STAR_(n,to,af,from,close_QMARK_,null,new cljs.core.Keyword(null,"async","async",1050769601));
}));

(cljs.core.async.pipeline_async.cljs$lang$maxFixedArity = 5);

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
var G__40195 = arguments.length;
switch (G__40195) {
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
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$4 = (function (n,to,xf,from){
return cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$5(n,to,xf,from,true);
}));

(cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$5 = (function (n,to,xf,from,close_QMARK_){
return cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$6(n,to,xf,from,close_QMARK_,null);
}));

(cljs.core.async.pipeline.cljs$core$IFn$_invoke$arity$6 = (function (n,to,xf,from,close_QMARK_,ex_handler){
return cljs.core.async.pipeline_STAR_(n,to,xf,from,close_QMARK_,ex_handler,new cljs.core.Keyword(null,"compute","compute",1555393130));
}));

(cljs.core.async.pipeline.cljs$lang$maxFixedArity = 6);

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
var G__40200 = arguments.length;
switch (G__40200) {
case 2:
return cljs.core.async.split.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 4:
return cljs.core.async.split.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.split.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.split.cljs$core$IFn$_invoke$arity$4(p,ch,null,null);
}));

(cljs.core.async.split.cljs$core$IFn$_invoke$arity$4 = (function (p,ch,t_buf_or_n,f_buf_or_n){
var tc = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(t_buf_or_n);
var fc = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(f_buf_or_n);
var c__39440__auto___42558 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40226){
var state_val_40227 = (state_40226[(1)]);
if((state_val_40227 === (7))){
var inst_40222 = (state_40226[(2)]);
var state_40226__$1 = state_40226;
var statearr_40228_42559 = state_40226__$1;
(statearr_40228_42559[(2)] = inst_40222);

(statearr_40228_42559[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (1))){
var state_40226__$1 = state_40226;
var statearr_40229_42560 = state_40226__$1;
(statearr_40229_42560[(2)] = null);

(statearr_40229_42560[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (4))){
var inst_40203 = (state_40226[(7)]);
var inst_40203__$1 = (state_40226[(2)]);
var inst_40204 = (inst_40203__$1 == null);
var state_40226__$1 = (function (){var statearr_40231 = state_40226;
(statearr_40231[(7)] = inst_40203__$1);

return statearr_40231;
})();
if(cljs.core.truth_(inst_40204)){
var statearr_40232_42562 = state_40226__$1;
(statearr_40232_42562[(1)] = (5));

} else {
var statearr_40234_42563 = state_40226__$1;
(statearr_40234_42563[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (13))){
var state_40226__$1 = state_40226;
var statearr_40236_42564 = state_40226__$1;
(statearr_40236_42564[(2)] = null);

(statearr_40236_42564[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (6))){
var inst_40203 = (state_40226[(7)]);
var inst_40209 = (p.cljs$core$IFn$_invoke$arity$1 ? p.cljs$core$IFn$_invoke$arity$1(inst_40203) : p.call(null,inst_40203));
var state_40226__$1 = state_40226;
if(cljs.core.truth_(inst_40209)){
var statearr_40237_42565 = state_40226__$1;
(statearr_40237_42565[(1)] = (9));

} else {
var statearr_40238_42566 = state_40226__$1;
(statearr_40238_42566[(1)] = (10));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (3))){
var inst_40224 = (state_40226[(2)]);
var state_40226__$1 = state_40226;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40226__$1,inst_40224);
} else {
if((state_val_40227 === (12))){
var state_40226__$1 = state_40226;
var statearr_40239_42569 = state_40226__$1;
(statearr_40239_42569[(2)] = null);

(statearr_40239_42569[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (2))){
var state_40226__$1 = state_40226;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40226__$1,(4),ch);
} else {
if((state_val_40227 === (11))){
var inst_40203 = (state_40226[(7)]);
var inst_40213 = (state_40226[(2)]);
var state_40226__$1 = state_40226;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_40226__$1,(8),inst_40213,inst_40203);
} else {
if((state_val_40227 === (9))){
var state_40226__$1 = state_40226;
var statearr_40243_42570 = state_40226__$1;
(statearr_40243_42570[(2)] = tc);

(statearr_40243_42570[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (5))){
var inst_40206 = cljs.core.async.close_BANG_(tc);
var inst_40207 = cljs.core.async.close_BANG_(fc);
var state_40226__$1 = (function (){var statearr_40244 = state_40226;
(statearr_40244[(8)] = inst_40206);

return statearr_40244;
})();
var statearr_40245_42571 = state_40226__$1;
(statearr_40245_42571[(2)] = inst_40207);

(statearr_40245_42571[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (14))){
var inst_40220 = (state_40226[(2)]);
var state_40226__$1 = state_40226;
var statearr_40246_42573 = state_40226__$1;
(statearr_40246_42573[(2)] = inst_40220);

(statearr_40246_42573[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (10))){
var state_40226__$1 = state_40226;
var statearr_40247_42574 = state_40226__$1;
(statearr_40247_42574[(2)] = fc);

(statearr_40247_42574[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40227 === (8))){
var inst_40215 = (state_40226[(2)]);
var state_40226__$1 = state_40226;
if(cljs.core.truth_(inst_40215)){
var statearr_40248_42576 = state_40226__$1;
(statearr_40248_42576[(1)] = (12));

} else {
var statearr_40249_42577 = state_40226__$1;
(statearr_40249_42577[(1)] = (13));

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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_40250 = [null,null,null,null,null,null,null,null,null];
(statearr_40250[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_40250[(1)] = (1));

return statearr_40250;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_40226){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40226);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40251){var ex__39237__auto__ = e40251;
var statearr_40252_42578 = state_40226;
(statearr_40252_42578[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40226[(4)]))){
var statearr_40253_42579 = state_40226;
(statearr_40253_42579[(1)] = cljs.core.first((state_40226[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42595 = state_40226;
state_40226 = G__42595;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_40226){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_40226);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40254 = f__39441__auto__();
(statearr_40254[(6)] = c__39440__auto___42558);

return statearr_40254;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [tc,fc], null);
}));

(cljs.core.async.split.cljs$lang$maxFixedArity = 4);

/**
 * f should be a function of 2 arguments. Returns a channel containing
 *   the single result of applying f to init and the first item from the
 *   channel, then applying f to that result and the 2nd item, etc. If
 *   the channel closes without yielding items, returns init and f is not
 *   called. ch must close before reduce produces a result.
 */
cljs.core.async.reduce = (function cljs$core$async$reduce(f,init,ch){
var c__39440__auto__ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40276){
var state_val_40277 = (state_40276[(1)]);
if((state_val_40277 === (7))){
var inst_40272 = (state_40276[(2)]);
var state_40276__$1 = state_40276;
var statearr_40278_42611 = state_40276__$1;
(statearr_40278_42611[(2)] = inst_40272);

(statearr_40278_42611[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (1))){
var inst_40255 = init;
var inst_40256 = inst_40255;
var state_40276__$1 = (function (){var statearr_40279 = state_40276;
(statearr_40279[(7)] = inst_40256);

return statearr_40279;
})();
var statearr_40280_42612 = state_40276__$1;
(statearr_40280_42612[(2)] = null);

(statearr_40280_42612[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (4))){
var inst_40259 = (state_40276[(8)]);
var inst_40259__$1 = (state_40276[(2)]);
var inst_40260 = (inst_40259__$1 == null);
var state_40276__$1 = (function (){var statearr_40284 = state_40276;
(statearr_40284[(8)] = inst_40259__$1);

return statearr_40284;
})();
if(cljs.core.truth_(inst_40260)){
var statearr_40285_42616 = state_40276__$1;
(statearr_40285_42616[(1)] = (5));

} else {
var statearr_40286_42618 = state_40276__$1;
(statearr_40286_42618[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (6))){
var inst_40263 = (state_40276[(9)]);
var inst_40259 = (state_40276[(8)]);
var inst_40256 = (state_40276[(7)]);
var inst_40263__$1 = (f.cljs$core$IFn$_invoke$arity$2 ? f.cljs$core$IFn$_invoke$arity$2(inst_40256,inst_40259) : f.call(null,inst_40256,inst_40259));
var inst_40264 = cljs.core.reduced_QMARK_(inst_40263__$1);
var state_40276__$1 = (function (){var statearr_40293 = state_40276;
(statearr_40293[(9)] = inst_40263__$1);

return statearr_40293;
})();
if(inst_40264){
var statearr_40294_42620 = state_40276__$1;
(statearr_40294_42620[(1)] = (8));

} else {
var statearr_40296_42621 = state_40276__$1;
(statearr_40296_42621[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (3))){
var inst_40274 = (state_40276[(2)]);
var state_40276__$1 = state_40276;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40276__$1,inst_40274);
} else {
if((state_val_40277 === (2))){
var state_40276__$1 = state_40276;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40276__$1,(4),ch);
} else {
if((state_val_40277 === (9))){
var inst_40263 = (state_40276[(9)]);
var inst_40256 = inst_40263;
var state_40276__$1 = (function (){var statearr_40298 = state_40276;
(statearr_40298[(7)] = inst_40256);

return statearr_40298;
})();
var statearr_40300_42627 = state_40276__$1;
(statearr_40300_42627[(2)] = null);

(statearr_40300_42627[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (5))){
var inst_40256 = (state_40276[(7)]);
var state_40276__$1 = state_40276;
var statearr_40302_42631 = state_40276__$1;
(statearr_40302_42631[(2)] = inst_40256);

(statearr_40302_42631[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (10))){
var inst_40270 = (state_40276[(2)]);
var state_40276__$1 = state_40276;
var statearr_40303_42632 = state_40276__$1;
(statearr_40303_42632[(2)] = inst_40270);

(statearr_40303_42632[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40277 === (8))){
var inst_40263 = (state_40276[(9)]);
var inst_40266 = cljs.core.deref(inst_40263);
var state_40276__$1 = state_40276;
var statearr_40304_42639 = state_40276__$1;
(statearr_40304_42639[(2)] = inst_40266);

(statearr_40304_42639[(1)] = (10));


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
});
return (function() {
var cljs$core$async$reduce_$_state_machine__39234__auto__ = null;
var cljs$core$async$reduce_$_state_machine__39234__auto____0 = (function (){
var statearr_40305 = [null,null,null,null,null,null,null,null,null,null];
(statearr_40305[(0)] = cljs$core$async$reduce_$_state_machine__39234__auto__);

(statearr_40305[(1)] = (1));

return statearr_40305;
});
var cljs$core$async$reduce_$_state_machine__39234__auto____1 = (function (state_40276){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40276);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40306){var ex__39237__auto__ = e40306;
var statearr_40307_42640 = state_40276;
(statearr_40307_42640[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40276[(4)]))){
var statearr_40308_42641 = state_40276;
(statearr_40308_42641[(1)] = cljs.core.first((state_40276[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42642 = state_40276;
state_40276 = G__42642;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$reduce_$_state_machine__39234__auto__ = function(state_40276){
switch(arguments.length){
case 0:
return cljs$core$async$reduce_$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$reduce_$_state_machine__39234__auto____1.call(this,state_40276);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$reduce_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$reduce_$_state_machine__39234__auto____0;
cljs$core$async$reduce_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$reduce_$_state_machine__39234__auto____1;
return cljs$core$async$reduce_$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40309 = f__39441__auto__();
(statearr_40309[(6)] = c__39440__auto__);

return statearr_40309;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

return c__39440__auto__;
});
/**
 * async/reduces a channel with a transformation (xform f).
 *   Returns a channel containing the result.  ch must close before
 *   transduce produces a result.
 */
cljs.core.async.transduce = (function cljs$core$async$transduce(xform,f,init,ch){
var f__$1 = (xform.cljs$core$IFn$_invoke$arity$1 ? xform.cljs$core$IFn$_invoke$arity$1(f) : xform.call(null,f));
var c__39440__auto__ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40315){
var state_val_40316 = (state_40315[(1)]);
if((state_val_40316 === (1))){
var inst_40310 = cljs.core.async.reduce(f__$1,init,ch);
var state_40315__$1 = state_40315;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40315__$1,(2),inst_40310);
} else {
if((state_val_40316 === (2))){
var inst_40312 = (state_40315[(2)]);
var inst_40313 = (f__$1.cljs$core$IFn$_invoke$arity$1 ? f__$1.cljs$core$IFn$_invoke$arity$1(inst_40312) : f__$1.call(null,inst_40312));
var state_40315__$1 = state_40315;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40315__$1,inst_40313);
} else {
return null;
}
}
});
return (function() {
var cljs$core$async$transduce_$_state_machine__39234__auto__ = null;
var cljs$core$async$transduce_$_state_machine__39234__auto____0 = (function (){
var statearr_40321 = [null,null,null,null,null,null,null];
(statearr_40321[(0)] = cljs$core$async$transduce_$_state_machine__39234__auto__);

(statearr_40321[(1)] = (1));

return statearr_40321;
});
var cljs$core$async$transduce_$_state_machine__39234__auto____1 = (function (state_40315){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40315);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40322){var ex__39237__auto__ = e40322;
var statearr_40323_42653 = state_40315;
(statearr_40323_42653[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40315[(4)]))){
var statearr_40324_42662 = state_40315;
(statearr_40324_42662[(1)] = cljs.core.first((state_40315[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42663 = state_40315;
state_40315 = G__42663;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$transduce_$_state_machine__39234__auto__ = function(state_40315){
switch(arguments.length){
case 0:
return cljs$core$async$transduce_$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$transduce_$_state_machine__39234__auto____1.call(this,state_40315);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$transduce_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$transduce_$_state_machine__39234__auto____0;
cljs$core$async$transduce_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$transduce_$_state_machine__39234__auto____1;
return cljs$core$async$transduce_$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40325 = f__39441__auto__();
(statearr_40325[(6)] = c__39440__auto__);

return statearr_40325;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

return c__39440__auto__;
});
/**
 * Puts the contents of coll into the supplied channel.
 * 
 *   By default the channel will be closed after the items are copied,
 *   but can be determined by the close? parameter.
 * 
 *   Returns a channel which will close after the items are copied.
 */
cljs.core.async.onto_chan_BANG_ = (function cljs$core$async$onto_chan_BANG_(var_args){
var G__40330 = arguments.length;
switch (G__40330) {
case 2:
return cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$2 = (function (ch,coll){
return cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$3(ch,coll,true);
}));

(cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$3 = (function (ch,coll,close_QMARK_){
var c__39440__auto__ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40358){
var state_val_40359 = (state_40358[(1)]);
if((state_val_40359 === (7))){
var inst_40340 = (state_40358[(2)]);
var state_40358__$1 = state_40358;
var statearr_40360_42671 = state_40358__$1;
(statearr_40360_42671[(2)] = inst_40340);

(statearr_40360_42671[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (1))){
var inst_40334 = cljs.core.seq(coll);
var inst_40335 = inst_40334;
var state_40358__$1 = (function (){var statearr_40361 = state_40358;
(statearr_40361[(7)] = inst_40335);

return statearr_40361;
})();
var statearr_40362_42672 = state_40358__$1;
(statearr_40362_42672[(2)] = null);

(statearr_40362_42672[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (4))){
var inst_40335 = (state_40358[(7)]);
var inst_40338 = cljs.core.first(inst_40335);
var state_40358__$1 = state_40358;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_40358__$1,(7),ch,inst_40338);
} else {
if((state_val_40359 === (13))){
var inst_40352 = (state_40358[(2)]);
var state_40358__$1 = state_40358;
var statearr_40363_42673 = state_40358__$1;
(statearr_40363_42673[(2)] = inst_40352);

(statearr_40363_42673[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (6))){
var inst_40343 = (state_40358[(2)]);
var state_40358__$1 = state_40358;
if(cljs.core.truth_(inst_40343)){
var statearr_40364_42674 = state_40358__$1;
(statearr_40364_42674[(1)] = (8));

} else {
var statearr_40365_42675 = state_40358__$1;
(statearr_40365_42675[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (3))){
var inst_40356 = (state_40358[(2)]);
var state_40358__$1 = state_40358;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40358__$1,inst_40356);
} else {
if((state_val_40359 === (12))){
var state_40358__$1 = state_40358;
var statearr_40370_42676 = state_40358__$1;
(statearr_40370_42676[(2)] = null);

(statearr_40370_42676[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (2))){
var inst_40335 = (state_40358[(7)]);
var state_40358__$1 = state_40358;
if(cljs.core.truth_(inst_40335)){
var statearr_40371_42677 = state_40358__$1;
(statearr_40371_42677[(1)] = (4));

} else {
var statearr_40372_42678 = state_40358__$1;
(statearr_40372_42678[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (11))){
var inst_40349 = cljs.core.async.close_BANG_(ch);
var state_40358__$1 = state_40358;
var statearr_40373_42679 = state_40358__$1;
(statearr_40373_42679[(2)] = inst_40349);

(statearr_40373_42679[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (9))){
var state_40358__$1 = state_40358;
if(cljs.core.truth_(close_QMARK_)){
var statearr_40374_42680 = state_40358__$1;
(statearr_40374_42680[(1)] = (11));

} else {
var statearr_40375_42681 = state_40358__$1;
(statearr_40375_42681[(1)] = (12));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (5))){
var inst_40335 = (state_40358[(7)]);
var state_40358__$1 = state_40358;
var statearr_40376_42682 = state_40358__$1;
(statearr_40376_42682[(2)] = inst_40335);

(statearr_40376_42682[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (10))){
var inst_40354 = (state_40358[(2)]);
var state_40358__$1 = state_40358;
var statearr_40377_42685 = state_40358__$1;
(statearr_40377_42685[(2)] = inst_40354);

(statearr_40377_42685[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40359 === (8))){
var inst_40335 = (state_40358[(7)]);
var inst_40345 = cljs.core.next(inst_40335);
var inst_40335__$1 = inst_40345;
var state_40358__$1 = (function (){var statearr_40378 = state_40358;
(statearr_40378[(7)] = inst_40335__$1);

return statearr_40378;
})();
var statearr_40379_42687 = state_40358__$1;
(statearr_40379_42687[(2)] = null);

(statearr_40379_42687[(1)] = (2));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_40384 = [null,null,null,null,null,null,null,null];
(statearr_40384[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_40384[(1)] = (1));

return statearr_40384;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_40358){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40358);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40385){var ex__39237__auto__ = e40385;
var statearr_40386_42688 = state_40358;
(statearr_40386_42688[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40358[(4)]))){
var statearr_40387_42689 = state_40358;
(statearr_40387_42689[(1)] = cljs.core.first((state_40358[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42690 = state_40358;
state_40358 = G__42690;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_40358){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_40358);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40388 = f__39441__auto__();
(statearr_40388[(6)] = c__39440__auto__);

return statearr_40388;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

return c__39440__auto__;
}));

(cljs.core.async.onto_chan_BANG_.cljs$lang$maxFixedArity = 3);

/**
 * Creates and returns a channel which contains the contents of coll,
 *   closing when exhausted.
 */
cljs.core.async.to_chan_BANG_ = (function cljs$core$async$to_chan_BANG_(coll){
var ch = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(cljs.core.bounded_count((100),coll));
cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$2(ch,coll);

return ch;
});
/**
 * Deprecated - use onto-chan!
 */
cljs.core.async.onto_chan = (function cljs$core$async$onto_chan(var_args){
var G__40398 = arguments.length;
switch (G__40398) {
case 2:
return cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$2 = (function (ch,coll){
return cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$3(ch,coll,true);
}));

(cljs.core.async.onto_chan.cljs$core$IFn$_invoke$arity$3 = (function (ch,coll,close_QMARK_){
return cljs.core.async.onto_chan_BANG_.cljs$core$IFn$_invoke$arity$3(ch,coll,close_QMARK_);
}));

(cljs.core.async.onto_chan.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - use to-chan!
 */
cljs.core.async.to_chan = (function cljs$core$async$to_chan(coll){
return cljs.core.async.to_chan_BANG_(coll);
});

/**
 * @interface
 */
cljs.core.async.Mux = function(){};

var cljs$core$async$Mux$muxch_STAR_$dyn_42701 = (function (_){
var x__4428__auto__ = (((_ == null))?null:_);
var m__4429__auto__ = (cljs.core.async.muxch_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(_) : m__4429__auto__.call(null,_));
} else {
var m__4426__auto__ = (cljs.core.async.muxch_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(_) : m__4426__auto__.call(null,_));
} else {
throw cljs.core.missing_protocol("Mux.muxch*",_);
}
}
});
cljs.core.async.muxch_STAR_ = (function cljs$core$async$muxch_STAR_(_){
if((((!((_ == null)))) && ((!((_.cljs$core$async$Mux$muxch_STAR_$arity$1 == null)))))){
return _.cljs$core$async$Mux$muxch_STAR_$arity$1(_);
} else {
return cljs$core$async$Mux$muxch_STAR_$dyn_42701(_);
}
});


/**
 * @interface
 */
cljs.core.async.Mult = function(){};

var cljs$core$async$Mult$tap_STAR_$dyn_42704 = (function (m,ch,close_QMARK_){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.tap_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(m,ch,close_QMARK_) : m__4429__auto__.call(null,m,ch,close_QMARK_));
} else {
var m__4426__auto__ = (cljs.core.async.tap_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(m,ch,close_QMARK_) : m__4426__auto__.call(null,m,ch,close_QMARK_));
} else {
throw cljs.core.missing_protocol("Mult.tap*",m);
}
}
});
cljs.core.async.tap_STAR_ = (function cljs$core$async$tap_STAR_(m,ch,close_QMARK_){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mult$tap_STAR_$arity$3 == null)))))){
return m.cljs$core$async$Mult$tap_STAR_$arity$3(m,ch,close_QMARK_);
} else {
return cljs$core$async$Mult$tap_STAR_$dyn_42704(m,ch,close_QMARK_);
}
});

var cljs$core$async$Mult$untap_STAR_$dyn_42708 = (function (m,ch){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.untap_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4429__auto__.call(null,m,ch));
} else {
var m__4426__auto__ = (cljs.core.async.untap_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4426__auto__.call(null,m,ch));
} else {
throw cljs.core.missing_protocol("Mult.untap*",m);
}
}
});
cljs.core.async.untap_STAR_ = (function cljs$core$async$untap_STAR_(m,ch){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mult$untap_STAR_$arity$2 == null)))))){
return m.cljs$core$async$Mult$untap_STAR_$arity$2(m,ch);
} else {
return cljs$core$async$Mult$untap_STAR_$dyn_42708(m,ch);
}
});

var cljs$core$async$Mult$untap_all_STAR_$dyn_42710 = (function (m){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.untap_all_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(m) : m__4429__auto__.call(null,m));
} else {
var m__4426__auto__ = (cljs.core.async.untap_all_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(m) : m__4426__auto__.call(null,m));
} else {
throw cljs.core.missing_protocol("Mult.untap-all*",m);
}
}
});
cljs.core.async.untap_all_STAR_ = (function cljs$core$async$untap_all_STAR_(m){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mult$untap_all_STAR_$arity$1 == null)))))){
return m.cljs$core$async$Mult$untap_all_STAR_$arity$1(m);
} else {
return cljs$core$async$Mult$untap_all_STAR_$dyn_42710(m);
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
var cs = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var m = (function (){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async40446 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.Mult}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async40446 = (function (ch,cs,meta40447){
this.ch = ch;
this.cs = cs;
this.meta40447 = meta40447;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_40448,meta40447__$1){
var self__ = this;
var _40448__$1 = this;
return (new cljs.core.async.t_cljs$core$async40446(self__.ch,self__.cs,meta40447__$1));
}));

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_40448){
var self__ = this;
var _40448__$1 = this;
return self__.meta40447;
}));

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mux$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.ch;
}));

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mult$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mult$tap_STAR_$arity$3 = (function (_,ch__$1,close_QMARK_){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$4(self__.cs,cljs.core.assoc,ch__$1,close_QMARK_);

return null;
}));

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mult$untap_STAR_$arity$2 = (function (_,ch__$1){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(self__.cs,cljs.core.dissoc,ch__$1);

return null;
}));

(cljs.core.async.t_cljs$core$async40446.prototype.cljs$core$async$Mult$untap_all_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_(self__.cs,cljs.core.PersistentArrayMap.EMPTY);

return null;
}));

(cljs.core.async.t_cljs$core$async40446.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"cs","cs",-117024463,null),new cljs.core.Symbol(null,"meta40447","meta40447",513715218,null)], null);
}));

(cljs.core.async.t_cljs$core$async40446.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async40446.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async40446");

(cljs.core.async.t_cljs$core$async40446.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async40446");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async40446.
 */
cljs.core.async.__GT_t_cljs$core$async40446 = (function cljs$core$async$mult_$___GT_t_cljs$core$async40446(ch__$1,cs__$1,meta40447){
return (new cljs.core.async.t_cljs$core$async40446(ch__$1,cs__$1,meta40447));
});

}

return (new cljs.core.async.t_cljs$core$async40446(ch,cs,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var dchan = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
var dctr = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(null);
var done = (function (_){
if((cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$2(dctr,cljs.core.dec) === (0))){
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(dchan,true);
} else {
return null;
}
});
var c__39440__auto___42716 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_40622){
var state_val_40623 = (state_40622[(1)]);
if((state_val_40623 === (7))){
var inst_40617 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40627_42717 = state_40622__$1;
(statearr_40627_42717[(2)] = inst_40617);

(statearr_40627_42717[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (20))){
var inst_40502 = (state_40622[(7)]);
var inst_40514 = cljs.core.first(inst_40502);
var inst_40515 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40514,(0),null);
var inst_40516 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40514,(1),null);
var state_40622__$1 = (function (){var statearr_40629 = state_40622;
(statearr_40629[(8)] = inst_40515);

return statearr_40629;
})();
if(cljs.core.truth_(inst_40516)){
var statearr_40632_42718 = state_40622__$1;
(statearr_40632_42718[(1)] = (22));

} else {
var statearr_40633_42719 = state_40622__$1;
(statearr_40633_42719[(1)] = (23));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (27))){
var inst_40558 = (state_40622[(9)]);
var inst_40551 = (state_40622[(10)]);
var inst_40549 = (state_40622[(11)]);
var inst_40466 = (state_40622[(12)]);
var inst_40558__$1 = cljs.core._nth(inst_40549,inst_40551);
var inst_40565 = cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$3(inst_40558__$1,inst_40466,done);
var state_40622__$1 = (function (){var statearr_40641 = state_40622;
(statearr_40641[(9)] = inst_40558__$1);

return statearr_40641;
})();
if(cljs.core.truth_(inst_40565)){
var statearr_40642_42720 = state_40622__$1;
(statearr_40642_42720[(1)] = (30));

} else {
var statearr_40645_42721 = state_40622__$1;
(statearr_40645_42721[(1)] = (31));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (1))){
var state_40622__$1 = state_40622;
var statearr_40647_42722 = state_40622__$1;
(statearr_40647_42722[(2)] = null);

(statearr_40647_42722[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (24))){
var inst_40502 = (state_40622[(7)]);
var inst_40521 = (state_40622[(2)]);
var inst_40522 = cljs.core.next(inst_40502);
var inst_40476 = inst_40522;
var inst_40477 = null;
var inst_40478 = (0);
var inst_40479 = (0);
var state_40622__$1 = (function (){var statearr_40648 = state_40622;
(statearr_40648[(13)] = inst_40476);

(statearr_40648[(14)] = inst_40479);

(statearr_40648[(15)] = inst_40478);

(statearr_40648[(16)] = inst_40477);

(statearr_40648[(17)] = inst_40521);

return statearr_40648;
})();
var statearr_40650_42725 = state_40622__$1;
(statearr_40650_42725[(2)] = null);

(statearr_40650_42725[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (39))){
var state_40622__$1 = state_40622;
var statearr_40657_42726 = state_40622__$1;
(statearr_40657_42726[(2)] = null);

(statearr_40657_42726[(1)] = (41));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (4))){
var inst_40466 = (state_40622[(12)]);
var inst_40466__$1 = (state_40622[(2)]);
var inst_40467 = (inst_40466__$1 == null);
var state_40622__$1 = (function (){var statearr_40659 = state_40622;
(statearr_40659[(12)] = inst_40466__$1);

return statearr_40659;
})();
if(cljs.core.truth_(inst_40467)){
var statearr_40661_42731 = state_40622__$1;
(statearr_40661_42731[(1)] = (5));

} else {
var statearr_40662_42732 = state_40622__$1;
(statearr_40662_42732[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (15))){
var inst_40476 = (state_40622[(13)]);
var inst_40479 = (state_40622[(14)]);
var inst_40478 = (state_40622[(15)]);
var inst_40477 = (state_40622[(16)]);
var inst_40497 = (state_40622[(2)]);
var inst_40498 = (inst_40479 + (1));
var tmp40653 = inst_40476;
var tmp40654 = inst_40478;
var tmp40655 = inst_40477;
var inst_40476__$1 = tmp40653;
var inst_40477__$1 = tmp40655;
var inst_40478__$1 = tmp40654;
var inst_40479__$1 = inst_40498;
var state_40622__$1 = (function (){var statearr_40668 = state_40622;
(statearr_40668[(13)] = inst_40476__$1);

(statearr_40668[(14)] = inst_40479__$1);

(statearr_40668[(15)] = inst_40478__$1);

(statearr_40668[(16)] = inst_40477__$1);

(statearr_40668[(18)] = inst_40497);

return statearr_40668;
})();
var statearr_40671_42735 = state_40622__$1;
(statearr_40671_42735[(2)] = null);

(statearr_40671_42735[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (21))){
var inst_40525 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40676_42736 = state_40622__$1;
(statearr_40676_42736[(2)] = inst_40525);

(statearr_40676_42736[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (31))){
var inst_40558 = (state_40622[(9)]);
var inst_40568 = m.cljs$core$async$Mult$untap_STAR_$arity$2(null,inst_40558);
var state_40622__$1 = state_40622;
var statearr_40681_42738 = state_40622__$1;
(statearr_40681_42738[(2)] = inst_40568);

(statearr_40681_42738[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (32))){
var inst_40548 = (state_40622[(19)]);
var inst_40551 = (state_40622[(10)]);
var inst_40550 = (state_40622[(20)]);
var inst_40549 = (state_40622[(11)]);
var inst_40570 = (state_40622[(2)]);
var inst_40571 = (inst_40551 + (1));
var tmp40672 = inst_40548;
var tmp40673 = inst_40550;
var tmp40674 = inst_40549;
var inst_40548__$1 = tmp40672;
var inst_40549__$1 = tmp40674;
var inst_40550__$1 = tmp40673;
var inst_40551__$1 = inst_40571;
var state_40622__$1 = (function (){var statearr_40682 = state_40622;
(statearr_40682[(19)] = inst_40548__$1);

(statearr_40682[(21)] = inst_40570);

(statearr_40682[(10)] = inst_40551__$1);

(statearr_40682[(20)] = inst_40550__$1);

(statearr_40682[(11)] = inst_40549__$1);

return statearr_40682;
})();
var statearr_40684_42740 = state_40622__$1;
(statearr_40684_42740[(2)] = null);

(statearr_40684_42740[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (40))){
var inst_40588 = (state_40622[(22)]);
var inst_40593 = m.cljs$core$async$Mult$untap_STAR_$arity$2(null,inst_40588);
var state_40622__$1 = state_40622;
var statearr_40699_42741 = state_40622__$1;
(statearr_40699_42741[(2)] = inst_40593);

(statearr_40699_42741[(1)] = (41));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (33))){
var inst_40576 = (state_40622[(23)]);
var inst_40578 = cljs.core.chunked_seq_QMARK_(inst_40576);
var state_40622__$1 = state_40622;
if(inst_40578){
var statearr_40701_42746 = state_40622__$1;
(statearr_40701_42746[(1)] = (36));

} else {
var statearr_40702_42747 = state_40622__$1;
(statearr_40702_42747[(1)] = (37));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (13))){
var inst_40489 = (state_40622[(24)]);
var inst_40494 = cljs.core.async.close_BANG_(inst_40489);
var state_40622__$1 = state_40622;
var statearr_40709_42751 = state_40622__$1;
(statearr_40709_42751[(2)] = inst_40494);

(statearr_40709_42751[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (22))){
var inst_40515 = (state_40622[(8)]);
var inst_40518 = cljs.core.async.close_BANG_(inst_40515);
var state_40622__$1 = state_40622;
var statearr_40713_42752 = state_40622__$1;
(statearr_40713_42752[(2)] = inst_40518);

(statearr_40713_42752[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (36))){
var inst_40576 = (state_40622[(23)]);
var inst_40581 = cljs.core.chunk_first(inst_40576);
var inst_40582 = cljs.core.chunk_rest(inst_40576);
var inst_40583 = cljs.core.count(inst_40581);
var inst_40548 = inst_40582;
var inst_40549 = inst_40581;
var inst_40550 = inst_40583;
var inst_40551 = (0);
var state_40622__$1 = (function (){var statearr_40715 = state_40622;
(statearr_40715[(19)] = inst_40548);

(statearr_40715[(10)] = inst_40551);

(statearr_40715[(20)] = inst_40550);

(statearr_40715[(11)] = inst_40549);

return statearr_40715;
})();
var statearr_40716_42756 = state_40622__$1;
(statearr_40716_42756[(2)] = null);

(statearr_40716_42756[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (41))){
var inst_40576 = (state_40622[(23)]);
var inst_40595 = (state_40622[(2)]);
var inst_40597 = cljs.core.next(inst_40576);
var inst_40548 = inst_40597;
var inst_40549 = null;
var inst_40550 = (0);
var inst_40551 = (0);
var state_40622__$1 = (function (){var statearr_40717 = state_40622;
(statearr_40717[(19)] = inst_40548);

(statearr_40717[(25)] = inst_40595);

(statearr_40717[(10)] = inst_40551);

(statearr_40717[(20)] = inst_40550);

(statearr_40717[(11)] = inst_40549);

return statearr_40717;
})();
var statearr_40719_42760 = state_40622__$1;
(statearr_40719_42760[(2)] = null);

(statearr_40719_42760[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (43))){
var state_40622__$1 = state_40622;
var statearr_40722_42761 = state_40622__$1;
(statearr_40722_42761[(2)] = null);

(statearr_40722_42761[(1)] = (44));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (29))){
var inst_40605 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40725_42765 = state_40622__$1;
(statearr_40725_42765[(2)] = inst_40605);

(statearr_40725_42765[(1)] = (26));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (44))){
var inst_40614 = (state_40622[(2)]);
var state_40622__$1 = (function (){var statearr_40726 = state_40622;
(statearr_40726[(26)] = inst_40614);

return statearr_40726;
})();
var statearr_40728_42767 = state_40622__$1;
(statearr_40728_42767[(2)] = null);

(statearr_40728_42767[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (6))){
var inst_40535 = (state_40622[(27)]);
var inst_40534 = cljs.core.deref(cs);
var inst_40535__$1 = cljs.core.keys(inst_40534);
var inst_40538 = cljs.core.count(inst_40535__$1);
var inst_40539 = cljs.core.reset_BANG_(dctr,inst_40538);
var inst_40547 = cljs.core.seq(inst_40535__$1);
var inst_40548 = inst_40547;
var inst_40549 = null;
var inst_40550 = (0);
var inst_40551 = (0);
var state_40622__$1 = (function (){var statearr_40735 = state_40622;
(statearr_40735[(19)] = inst_40548);

(statearr_40735[(10)] = inst_40551);

(statearr_40735[(20)] = inst_40550);

(statearr_40735[(28)] = inst_40539);

(statearr_40735[(11)] = inst_40549);

(statearr_40735[(27)] = inst_40535__$1);

return statearr_40735;
})();
var statearr_40736_42772 = state_40622__$1;
(statearr_40736_42772[(2)] = null);

(statearr_40736_42772[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (28))){
var inst_40548 = (state_40622[(19)]);
var inst_40576 = (state_40622[(23)]);
var inst_40576__$1 = cljs.core.seq(inst_40548);
var state_40622__$1 = (function (){var statearr_40737 = state_40622;
(statearr_40737[(23)] = inst_40576__$1);

return statearr_40737;
})();
if(inst_40576__$1){
var statearr_40738_42774 = state_40622__$1;
(statearr_40738_42774[(1)] = (33));

} else {
var statearr_40739_42775 = state_40622__$1;
(statearr_40739_42775[(1)] = (34));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (25))){
var inst_40551 = (state_40622[(10)]);
var inst_40550 = (state_40622[(20)]);
var inst_40554 = (inst_40551 < inst_40550);
var inst_40555 = inst_40554;
var state_40622__$1 = state_40622;
if(cljs.core.truth_(inst_40555)){
var statearr_40742_42776 = state_40622__$1;
(statearr_40742_42776[(1)] = (27));

} else {
var statearr_40743_42777 = state_40622__$1;
(statearr_40743_42777[(1)] = (28));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (34))){
var state_40622__$1 = state_40622;
var statearr_40744_42778 = state_40622__$1;
(statearr_40744_42778[(2)] = null);

(statearr_40744_42778[(1)] = (35));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (17))){
var state_40622__$1 = state_40622;
var statearr_40745_42779 = state_40622__$1;
(statearr_40745_42779[(2)] = null);

(statearr_40745_42779[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (3))){
var inst_40619 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
return cljs.core.async.impl.ioc_helpers.return_chan(state_40622__$1,inst_40619);
} else {
if((state_val_40623 === (12))){
var inst_40530 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40750_42781 = state_40622__$1;
(statearr_40750_42781[(2)] = inst_40530);

(statearr_40750_42781[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (2))){
var state_40622__$1 = state_40622;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40622__$1,(4),ch);
} else {
if((state_val_40623 === (23))){
var state_40622__$1 = state_40622;
var statearr_40751_42783 = state_40622__$1;
(statearr_40751_42783[(2)] = null);

(statearr_40751_42783[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (35))){
var inst_40603 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40753_42784 = state_40622__$1;
(statearr_40753_42784[(2)] = inst_40603);

(statearr_40753_42784[(1)] = (29));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (19))){
var inst_40502 = (state_40622[(7)]);
var inst_40506 = cljs.core.chunk_first(inst_40502);
var inst_40507 = cljs.core.chunk_rest(inst_40502);
var inst_40508 = cljs.core.count(inst_40506);
var inst_40476 = inst_40507;
var inst_40477 = inst_40506;
var inst_40478 = inst_40508;
var inst_40479 = (0);
var state_40622__$1 = (function (){var statearr_40755 = state_40622;
(statearr_40755[(13)] = inst_40476);

(statearr_40755[(14)] = inst_40479);

(statearr_40755[(15)] = inst_40478);

(statearr_40755[(16)] = inst_40477);

return statearr_40755;
})();
var statearr_40756_42785 = state_40622__$1;
(statearr_40756_42785[(2)] = null);

(statearr_40756_42785[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (11))){
var inst_40476 = (state_40622[(13)]);
var inst_40502 = (state_40622[(7)]);
var inst_40502__$1 = cljs.core.seq(inst_40476);
var state_40622__$1 = (function (){var statearr_40757 = state_40622;
(statearr_40757[(7)] = inst_40502__$1);

return statearr_40757;
})();
if(inst_40502__$1){
var statearr_40759_42786 = state_40622__$1;
(statearr_40759_42786[(1)] = (16));

} else {
var statearr_40761_42792 = state_40622__$1;
(statearr_40761_42792[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (9))){
var inst_40532 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40764_42794 = state_40622__$1;
(statearr_40764_42794[(2)] = inst_40532);

(statearr_40764_42794[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (5))){
var inst_40474 = cljs.core.deref(cs);
var inst_40475 = cljs.core.seq(inst_40474);
var inst_40476 = inst_40475;
var inst_40477 = null;
var inst_40478 = (0);
var inst_40479 = (0);
var state_40622__$1 = (function (){var statearr_40766 = state_40622;
(statearr_40766[(13)] = inst_40476);

(statearr_40766[(14)] = inst_40479);

(statearr_40766[(15)] = inst_40478);

(statearr_40766[(16)] = inst_40477);

return statearr_40766;
})();
var statearr_40767_42800 = state_40622__$1;
(statearr_40767_42800[(2)] = null);

(statearr_40767_42800[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (14))){
var state_40622__$1 = state_40622;
var statearr_40768_42801 = state_40622__$1;
(statearr_40768_42801[(2)] = null);

(statearr_40768_42801[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (45))){
var inst_40611 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40769_42806 = state_40622__$1;
(statearr_40769_42806[(2)] = inst_40611);

(statearr_40769_42806[(1)] = (44));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (26))){
var inst_40535 = (state_40622[(27)]);
var inst_40607 = (state_40622[(2)]);
var inst_40608 = cljs.core.seq(inst_40535);
var state_40622__$1 = (function (){var statearr_40770 = state_40622;
(statearr_40770[(29)] = inst_40607);

return statearr_40770;
})();
if(inst_40608){
var statearr_40775_42809 = state_40622__$1;
(statearr_40775_42809[(1)] = (42));

} else {
var statearr_40776_42810 = state_40622__$1;
(statearr_40776_42810[(1)] = (43));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (16))){
var inst_40502 = (state_40622[(7)]);
var inst_40504 = cljs.core.chunked_seq_QMARK_(inst_40502);
var state_40622__$1 = state_40622;
if(inst_40504){
var statearr_40777_42811 = state_40622__$1;
(statearr_40777_42811[(1)] = (19));

} else {
var statearr_40795_42812 = state_40622__$1;
(statearr_40795_42812[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (38))){
var inst_40600 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40800_42819 = state_40622__$1;
(statearr_40800_42819[(2)] = inst_40600);

(statearr_40800_42819[(1)] = (35));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (30))){
var state_40622__$1 = state_40622;
var statearr_40801_42820 = state_40622__$1;
(statearr_40801_42820[(2)] = null);

(statearr_40801_42820[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (10))){
var inst_40479 = (state_40622[(14)]);
var inst_40477 = (state_40622[(16)]);
var inst_40488 = cljs.core._nth(inst_40477,inst_40479);
var inst_40489 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40488,(0),null);
var inst_40492 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40488,(1),null);
var state_40622__$1 = (function (){var statearr_40802 = state_40622;
(statearr_40802[(24)] = inst_40489);

return statearr_40802;
})();
if(cljs.core.truth_(inst_40492)){
var statearr_40807_42824 = state_40622__$1;
(statearr_40807_42824[(1)] = (13));

} else {
var statearr_40808_42825 = state_40622__$1;
(statearr_40808_42825[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (18))){
var inst_40528 = (state_40622[(2)]);
var state_40622__$1 = state_40622;
var statearr_40809_42826 = state_40622__$1;
(statearr_40809_42826[(2)] = inst_40528);

(statearr_40809_42826[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (42))){
var state_40622__$1 = state_40622;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_40622__$1,(45),dchan);
} else {
if((state_val_40623 === (37))){
var inst_40576 = (state_40622[(23)]);
var inst_40588 = (state_40622[(22)]);
var inst_40466 = (state_40622[(12)]);
var inst_40588__$1 = cljs.core.first(inst_40576);
var inst_40589 = cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$3(inst_40588__$1,inst_40466,done);
var state_40622__$1 = (function (){var statearr_40812 = state_40622;
(statearr_40812[(22)] = inst_40588__$1);

return statearr_40812;
})();
if(cljs.core.truth_(inst_40589)){
var statearr_40813_42832 = state_40622__$1;
(statearr_40813_42832[(1)] = (39));

} else {
var statearr_40814_42837 = state_40622__$1;
(statearr_40814_42837[(1)] = (40));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_40623 === (8))){
var inst_40479 = (state_40622[(14)]);
var inst_40478 = (state_40622[(15)]);
var inst_40482 = (inst_40479 < inst_40478);
var inst_40483 = inst_40482;
var state_40622__$1 = state_40622;
if(cljs.core.truth_(inst_40483)){
var statearr_40817_42844 = state_40622__$1;
(statearr_40817_42844[(1)] = (10));

} else {
var statearr_40820_42845 = state_40622__$1;
(statearr_40820_42845[(1)] = (11));

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
});
return (function() {
var cljs$core$async$mult_$_state_machine__39234__auto__ = null;
var cljs$core$async$mult_$_state_machine__39234__auto____0 = (function (){
var statearr_40822 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_40822[(0)] = cljs$core$async$mult_$_state_machine__39234__auto__);

(statearr_40822[(1)] = (1));

return statearr_40822;
});
var cljs$core$async$mult_$_state_machine__39234__auto____1 = (function (state_40622){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_40622);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e40823){var ex__39237__auto__ = e40823;
var statearr_40824_42855 = state_40622;
(statearr_40824_42855[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_40622[(4)]))){
var statearr_40825_42857 = state_40622;
(statearr_40825_42857[(1)] = cljs.core.first((state_40622[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__42859 = state_40622;
state_40622 = G__42859;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$mult_$_state_machine__39234__auto__ = function(state_40622){
switch(arguments.length){
case 0:
return cljs$core$async$mult_$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$mult_$_state_machine__39234__auto____1.call(this,state_40622);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mult_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mult_$_state_machine__39234__auto____0;
cljs$core$async$mult_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mult_$_state_machine__39234__auto____1;
return cljs$core$async$mult_$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_40828 = f__39441__auto__();
(statearr_40828[(6)] = c__39440__auto___42716);

return statearr_40828;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return m;
});
/**
 * Copies the mult source onto the supplied channel.
 * 
 *   By default the channel will be closed when the source closes,
 *   but can be determined by the close? parameter.
 */
cljs.core.async.tap = (function cljs$core$async$tap(var_args){
var G__40833 = arguments.length;
switch (G__40833) {
case 2:
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.tap.cljs$core$IFn$_invoke$arity$2 = (function (mult,ch){
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3(mult,ch,true);
}));

(cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3 = (function (mult,ch,close_QMARK_){
cljs.core.async.tap_STAR_(mult,ch,close_QMARK_);

return ch;
}));

(cljs.core.async.tap.cljs$lang$maxFixedArity = 3);

/**
 * Disconnects a target channel from a mult
 */
cljs.core.async.untap = (function cljs$core$async$untap(mult,ch){
return cljs.core.async.untap_STAR_(mult,ch);
});
/**
 * Disconnects all target channels from a mult
 */
cljs.core.async.untap_all = (function cljs$core$async$untap_all(mult){
return cljs.core.async.untap_all_STAR_(mult);
});

/**
 * @interface
 */
cljs.core.async.Mix = function(){};

var cljs$core$async$Mix$admix_STAR_$dyn_42868 = (function (m,ch){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.admix_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4429__auto__.call(null,m,ch));
} else {
var m__4426__auto__ = (cljs.core.async.admix_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4426__auto__.call(null,m,ch));
} else {
throw cljs.core.missing_protocol("Mix.admix*",m);
}
}
});
cljs.core.async.admix_STAR_ = (function cljs$core$async$admix_STAR_(m,ch){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mix$admix_STAR_$arity$2 == null)))))){
return m.cljs$core$async$Mix$admix_STAR_$arity$2(m,ch);
} else {
return cljs$core$async$Mix$admix_STAR_$dyn_42868(m,ch);
}
});

var cljs$core$async$Mix$unmix_STAR_$dyn_42874 = (function (m,ch){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.unmix_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4429__auto__.call(null,m,ch));
} else {
var m__4426__auto__ = (cljs.core.async.unmix_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(m,ch) : m__4426__auto__.call(null,m,ch));
} else {
throw cljs.core.missing_protocol("Mix.unmix*",m);
}
}
});
cljs.core.async.unmix_STAR_ = (function cljs$core$async$unmix_STAR_(m,ch){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mix$unmix_STAR_$arity$2 == null)))))){
return m.cljs$core$async$Mix$unmix_STAR_$arity$2(m,ch);
} else {
return cljs$core$async$Mix$unmix_STAR_$dyn_42874(m,ch);
}
});

var cljs$core$async$Mix$unmix_all_STAR_$dyn_42876 = (function (m){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.unmix_all_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(m) : m__4429__auto__.call(null,m));
} else {
var m__4426__auto__ = (cljs.core.async.unmix_all_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(m) : m__4426__auto__.call(null,m));
} else {
throw cljs.core.missing_protocol("Mix.unmix-all*",m);
}
}
});
cljs.core.async.unmix_all_STAR_ = (function cljs$core$async$unmix_all_STAR_(m){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mix$unmix_all_STAR_$arity$1 == null)))))){
return m.cljs$core$async$Mix$unmix_all_STAR_$arity$1(m);
} else {
return cljs$core$async$Mix$unmix_all_STAR_$dyn_42876(m);
}
});

var cljs$core$async$Mix$toggle_STAR_$dyn_42878 = (function (m,state_map){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.toggle_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(m,state_map) : m__4429__auto__.call(null,m,state_map));
} else {
var m__4426__auto__ = (cljs.core.async.toggle_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(m,state_map) : m__4426__auto__.call(null,m,state_map));
} else {
throw cljs.core.missing_protocol("Mix.toggle*",m);
}
}
});
cljs.core.async.toggle_STAR_ = (function cljs$core$async$toggle_STAR_(m,state_map){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mix$toggle_STAR_$arity$2 == null)))))){
return m.cljs$core$async$Mix$toggle_STAR_$arity$2(m,state_map);
} else {
return cljs$core$async$Mix$toggle_STAR_$dyn_42878(m,state_map);
}
});

var cljs$core$async$Mix$solo_mode_STAR_$dyn_42883 = (function (m,mode){
var x__4428__auto__ = (((m == null))?null:m);
var m__4429__auto__ = (cljs.core.async.solo_mode_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(m,mode) : m__4429__auto__.call(null,m,mode));
} else {
var m__4426__auto__ = (cljs.core.async.solo_mode_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(m,mode) : m__4426__auto__.call(null,m,mode));
} else {
throw cljs.core.missing_protocol("Mix.solo-mode*",m);
}
}
});
cljs.core.async.solo_mode_STAR_ = (function cljs$core$async$solo_mode_STAR_(m,mode){
if((((!((m == null)))) && ((!((m.cljs$core$async$Mix$solo_mode_STAR_$arity$2 == null)))))){
return m.cljs$core$async$Mix$solo_mode_STAR_$arity$2(m,mode);
} else {
return cljs$core$async$Mix$solo_mode_STAR_$dyn_42883(m,mode);
}
});

cljs.core.async.ioc_alts_BANG_ = (function cljs$core$async$ioc_alts_BANG_(var_args){
var args__4742__auto__ = [];
var len__4736__auto___42889 = arguments.length;
var i__4737__auto___42890 = (0);
while(true){
if((i__4737__auto___42890 < len__4736__auto___42889)){
args__4742__auto__.push((arguments[i__4737__auto___42890]));

var G__42893 = (i__4737__auto___42890 + (1));
i__4737__auto___42890 = G__42893;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((3) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((3)),(0),null)):null);
return cljs.core.async.ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),argseq__4743__auto__);
});

(cljs.core.async.ioc_alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = (function (state,cont_block,ports,p__40883){
var map__40884 = p__40883;
var map__40884__$1 = (((((!((map__40884 == null))))?(((((map__40884.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__40884.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__40884):map__40884);
var opts = map__40884__$1;
var statearr_40890_42901 = state;
(statearr_40890_42901[(1)] = cont_block);


var temp__5735__auto__ = cljs.core.async.do_alts((function (val){
var statearr_40891_42906 = state;
(statearr_40891_42906[(2)] = val);


return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state);
}),ports,opts);
if(cljs.core.truth_(temp__5735__auto__)){
var cb = temp__5735__auto__;
var statearr_40892_42914 = state;
(statearr_40892_42914[(2)] = cljs.core.deref(cb));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
return null;
}
}));

(cljs.core.async.ioc_alts_BANG_.cljs$lang$maxFixedArity = (3));

/** @this {Function} */
(cljs.core.async.ioc_alts_BANG_.cljs$lang$applyTo = (function (seq40865){
var G__40866 = cljs.core.first(seq40865);
var seq40865__$1 = cljs.core.next(seq40865);
var G__40867 = cljs.core.first(seq40865__$1);
var seq40865__$2 = cljs.core.next(seq40865__$1);
var G__40868 = cljs.core.first(seq40865__$2);
var seq40865__$3 = cljs.core.next(seq40865__$2);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__40866,G__40867,G__40868,seq40865__$3);
}));

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
var cs = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var solo_modes = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"pause","pause",-2095325672),null,new cljs.core.Keyword(null,"mute","mute",1151223646),null], null), null);
var attrs = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(solo_modes,new cljs.core.Keyword(null,"solo","solo",-316350075));
var solo_mode = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"mute","mute",1151223646));
var change = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(cljs.core.async.sliding_buffer((1)));
var changed = (function (){
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(change,true);
});
var pick = (function (attr,chs){
return cljs.core.reduce_kv((function (ret,c,v){
if(cljs.core.truth_((attr.cljs$core$IFn$_invoke$arity$1 ? attr.cljs$core$IFn$_invoke$arity$1(v) : attr.call(null,v)))){
return cljs.core.conj.cljs$core$IFn$_invoke$arity$2(ret,c);
} else {
return ret;
}
}),cljs.core.PersistentHashSet.EMPTY,chs);
});
var calc_state = (function (){
var chs = cljs.core.deref(cs);
var mode = cljs.core.deref(solo_mode);
var solos = pick(new cljs.core.Keyword(null,"solo","solo",-316350075),chs);
var pauses = pick(new cljs.core.Keyword(null,"pause","pause",-2095325672),chs);
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"solos","solos",1441458643),solos,new cljs.core.Keyword(null,"mutes","mutes",1068806309),pick(new cljs.core.Keyword(null,"mute","mute",1151223646),chs),new cljs.core.Keyword(null,"reads","reads",-1215067361),cljs.core.conj.cljs$core$IFn$_invoke$arity$2(((((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(mode,new cljs.core.Keyword(null,"pause","pause",-2095325672))) && ((!(cljs.core.empty_QMARK_(solos))))))?cljs.core.vec(solos):cljs.core.vec(cljs.core.remove.cljs$core$IFn$_invoke$arity$2(pauses,cljs.core.keys(chs)))),change)], null);
});
var m = (function (){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async40898 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mix}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async40898 = (function (change,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,meta40899){
this.change = change;
this.solo_mode = solo_mode;
this.pick = pick;
this.cs = cs;
this.calc_state = calc_state;
this.out = out;
this.changed = changed;
this.solo_modes = solo_modes;
this.attrs = attrs;
this.meta40899 = meta40899;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_40900,meta40899__$1){
var self__ = this;
var _40900__$1 = this;
return (new cljs.core.async.t_cljs$core$async40898(self__.change,self__.solo_mode,self__.pick,self__.cs,self__.calc_state,self__.out,self__.changed,self__.solo_modes,self__.attrs,meta40899__$1));
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_40900){
var self__ = this;
var _40900__$1 = this;
return self__.meta40899;
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mux$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.out;
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$admix_STAR_$arity$2 = (function (_,ch){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$4(self__.cs,cljs.core.assoc,ch,cljs.core.PersistentArrayMap.EMPTY);

return (self__.changed.cljs$core$IFn$_invoke$arity$0 ? self__.changed.cljs$core$IFn$_invoke$arity$0() : self__.changed.call(null));
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$unmix_STAR_$arity$2 = (function (_,ch){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(self__.cs,cljs.core.dissoc,ch);

return (self__.changed.cljs$core$IFn$_invoke$arity$0 ? self__.changed.cljs$core$IFn$_invoke$arity$0() : self__.changed.call(null));
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$unmix_all_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
cljs.core.reset_BANG_(self__.cs,cljs.core.PersistentArrayMap.EMPTY);

return (self__.changed.cljs$core$IFn$_invoke$arity$0 ? self__.changed.cljs$core$IFn$_invoke$arity$0() : self__.changed.call(null));
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$toggle_STAR_$arity$2 = (function (_,state_map){
var self__ = this;
var ___$1 = this;
cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(self__.cs,cljs.core.partial.cljs$core$IFn$_invoke$arity$2(cljs.core.merge_with,cljs.core.merge),state_map);

return (self__.changed.cljs$core$IFn$_invoke$arity$0 ? self__.changed.cljs$core$IFn$_invoke$arity$0() : self__.changed.call(null));
}));

(cljs.core.async.t_cljs$core$async40898.prototype.cljs$core$async$Mix$solo_mode_STAR_$arity$2 = (function (_,mode){
var self__ = this;
var ___$1 = this;
if(cljs.core.truth_((self__.solo_modes.cljs$core$IFn$_invoke$arity$1 ? self__.solo_modes.cljs$core$IFn$_invoke$arity$1(mode) : self__.solo_modes.call(null,mode)))){
} else {
throw (new Error(["Assert failed: ",["mode must be one of: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(self__.solo_modes)].join(''),"\n","(solo-modes mode)"].join('')));
}

cljs.core.reset_BANG_(self__.solo_mode,mode);

return (self__.changed.cljs$core$IFn$_invoke$arity$0 ? self__.changed.cljs$core$IFn$_invoke$arity$0() : self__.changed.call(null));
}));

(cljs.core.async.t_cljs$core$async40898.getBasis = (function (){
return new cljs.core.PersistentVector(null, 10, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"change","change",477485025,null),new cljs.core.Symbol(null,"solo-mode","solo-mode",2031788074,null),new cljs.core.Symbol(null,"pick","pick",1300068175,null),new cljs.core.Symbol(null,"cs","cs",-117024463,null),new cljs.core.Symbol(null,"calc-state","calc-state",-349968968,null),new cljs.core.Symbol(null,"out","out",729986010,null),new cljs.core.Symbol(null,"changed","changed",-2083710852,null),new cljs.core.Symbol(null,"solo-modes","solo-modes",882180540,null),new cljs.core.Symbol(null,"attrs","attrs",-450137186,null),new cljs.core.Symbol(null,"meta40899","meta40899",1279077642,null)], null);
}));

(cljs.core.async.t_cljs$core$async40898.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async40898.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async40898");

(cljs.core.async.t_cljs$core$async40898.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async40898");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async40898.
 */
cljs.core.async.__GT_t_cljs$core$async40898 = (function cljs$core$async$mix_$___GT_t_cljs$core$async40898(change__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta40899){
return (new cljs.core.async.t_cljs$core$async40898(change__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta40899));
});

}

return (new cljs.core.async.t_cljs$core$async40898(change,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var c__39440__auto___42959 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41057){
var state_val_41058 = (state_41057[(1)]);
if((state_val_41058 === (7))){
var inst_40944 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
var statearr_41068_42962 = state_41057__$1;
(statearr_41068_42962[(2)] = inst_40944);

(statearr_41068_42962[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (20))){
var inst_40956 = (state_41057[(7)]);
var state_41057__$1 = state_41057;
var statearr_41069_42963 = state_41057__$1;
(statearr_41069_42963[(2)] = inst_40956);

(statearr_41069_42963[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (27))){
var state_41057__$1 = state_41057;
var statearr_41075_42966 = state_41057__$1;
(statearr_41075_42966[(2)] = null);

(statearr_41075_42966[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (1))){
var inst_40926 = (state_41057[(8)]);
var inst_40926__$1 = calc_state();
var inst_40929 = (inst_40926__$1 == null);
var inst_40930 = cljs.core.not(inst_40929);
var state_41057__$1 = (function (){var statearr_41076 = state_41057;
(statearr_41076[(8)] = inst_40926__$1);

return statearr_41076;
})();
if(inst_40930){
var statearr_41077_42970 = state_41057__$1;
(statearr_41077_42970[(1)] = (2));

} else {
var statearr_41078_42971 = state_41057__$1;
(statearr_41078_42971[(1)] = (3));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (24))){
var inst_40991 = (state_41057[(9)]);
var inst_41008 = (state_41057[(10)]);
var inst_40981 = (state_41057[(11)]);
var inst_41008__$1 = (inst_40981.cljs$core$IFn$_invoke$arity$1 ? inst_40981.cljs$core$IFn$_invoke$arity$1(inst_40991) : inst_40981.call(null,inst_40991));
var state_41057__$1 = (function (){var statearr_41079 = state_41057;
(statearr_41079[(10)] = inst_41008__$1);

return statearr_41079;
})();
if(cljs.core.truth_(inst_41008__$1)){
var statearr_41080_42980 = state_41057__$1;
(statearr_41080_42980[(1)] = (29));

} else {
var statearr_41081_42981 = state_41057__$1;
(statearr_41081_42981[(1)] = (30));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (4))){
var inst_40947 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_40947)){
var statearr_41083_42983 = state_41057__$1;
(statearr_41083_42983[(1)] = (8));

} else {
var statearr_41084_42985 = state_41057__$1;
(statearr_41084_42985[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (15))){
var inst_40974 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_40974)){
var statearr_41086_42987 = state_41057__$1;
(statearr_41086_42987[(1)] = (19));

} else {
var statearr_41088_42988 = state_41057__$1;
(statearr_41088_42988[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (21))){
var inst_40980 = (state_41057[(12)]);
var inst_40980__$1 = (state_41057[(2)]);
var inst_40981 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40980__$1,new cljs.core.Keyword(null,"solos","solos",1441458643));
var inst_40982 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40980__$1,new cljs.core.Keyword(null,"mutes","mutes",1068806309));
var inst_40983 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40980__$1,new cljs.core.Keyword(null,"reads","reads",-1215067361));
var state_41057__$1 = (function (){var statearr_41090 = state_41057;
(statearr_41090[(13)] = inst_40982);

(statearr_41090[(12)] = inst_40980__$1);

(statearr_41090[(11)] = inst_40981);

return statearr_41090;
})();
return cljs.core.async.ioc_alts_BANG_(state_41057__$1,(22),inst_40983);
} else {
if((state_val_41058 === (31))){
var inst_41020 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_41020)){
var statearr_41092_42989 = state_41057__$1;
(statearr_41092_42989[(1)] = (32));

} else {
var statearr_41093_42990 = state_41057__$1;
(statearr_41093_42990[(1)] = (33));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (32))){
var inst_40990 = (state_41057[(14)]);
var state_41057__$1 = state_41057;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41057__$1,(35),out,inst_40990);
} else {
if((state_val_41058 === (33))){
var inst_40980 = (state_41057[(12)]);
var inst_40956 = inst_40980;
var state_41057__$1 = (function (){var statearr_41097 = state_41057;
(statearr_41097[(7)] = inst_40956);

return statearr_41097;
})();
var statearr_41098_42993 = state_41057__$1;
(statearr_41098_42993[(2)] = null);

(statearr_41098_42993[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (13))){
var inst_40956 = (state_41057[(7)]);
var inst_40963 = inst_40956.cljs$lang$protocol_mask$partition0$;
var inst_40964 = (inst_40963 & (64));
var inst_40965 = inst_40956.cljs$core$ISeq$;
var inst_40966 = (cljs.core.PROTOCOL_SENTINEL === inst_40965);
var inst_40967 = ((inst_40964) || (inst_40966));
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_40967)){
var statearr_41101_43005 = state_41057__$1;
(statearr_41101_43005[(1)] = (16));

} else {
var statearr_41108_43006 = state_41057__$1;
(statearr_41108_43006[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (22))){
var inst_40990 = (state_41057[(14)]);
var inst_40991 = (state_41057[(9)]);
var inst_40988 = (state_41057[(2)]);
var inst_40990__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40988,(0),null);
var inst_40991__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_40988,(1),null);
var inst_40993 = (inst_40990__$1 == null);
var inst_40994 = cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(inst_40991__$1,change);
var inst_40995 = ((inst_40993) || (inst_40994));
var state_41057__$1 = (function (){var statearr_41111 = state_41057;
(statearr_41111[(14)] = inst_40990__$1);

(statearr_41111[(9)] = inst_40991__$1);

return statearr_41111;
})();
if(cljs.core.truth_(inst_40995)){
var statearr_41112_43011 = state_41057__$1;
(statearr_41112_43011[(1)] = (23));

} else {
var statearr_41113_43012 = state_41057__$1;
(statearr_41113_43012[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (36))){
var inst_40980 = (state_41057[(12)]);
var inst_40956 = inst_40980;
var state_41057__$1 = (function (){var statearr_41114 = state_41057;
(statearr_41114[(7)] = inst_40956);

return statearr_41114;
})();
var statearr_41115_43017 = state_41057__$1;
(statearr_41115_43017[(2)] = null);

(statearr_41115_43017[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (29))){
var inst_41008 = (state_41057[(10)]);
var state_41057__$1 = state_41057;
var statearr_41116_43019 = state_41057__$1;
(statearr_41116_43019[(2)] = inst_41008);

(statearr_41116_43019[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (6))){
var state_41057__$1 = state_41057;
var statearr_41117_43021 = state_41057__$1;
(statearr_41117_43021[(2)] = false);

(statearr_41117_43021[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (28))){
var inst_41004 = (state_41057[(2)]);
var inst_41005 = calc_state();
var inst_40956 = inst_41005;
var state_41057__$1 = (function (){var statearr_41118 = state_41057;
(statearr_41118[(15)] = inst_41004);

(statearr_41118[(7)] = inst_40956);

return statearr_41118;
})();
var statearr_41119_43027 = state_41057__$1;
(statearr_41119_43027[(2)] = null);

(statearr_41119_43027[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (25))){
var inst_41034 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
var statearr_41120_43029 = state_41057__$1;
(statearr_41120_43029[(2)] = inst_41034);

(statearr_41120_43029[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (34))){
var inst_41032 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
var statearr_41125_43033 = state_41057__$1;
(statearr_41125_43033[(2)] = inst_41032);

(statearr_41125_43033[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (17))){
var state_41057__$1 = state_41057;
var statearr_41127_43034 = state_41057__$1;
(statearr_41127_43034[(2)] = false);

(statearr_41127_43034[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (3))){
var state_41057__$1 = state_41057;
var statearr_41128_43035 = state_41057__$1;
(statearr_41128_43035[(2)] = false);

(statearr_41128_43035[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (12))){
var inst_41036 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41057__$1,inst_41036);
} else {
if((state_val_41058 === (2))){
var inst_40926 = (state_41057[(8)]);
var inst_40936 = inst_40926.cljs$lang$protocol_mask$partition0$;
var inst_40937 = (inst_40936 & (64));
var inst_40938 = inst_40926.cljs$core$ISeq$;
var inst_40939 = (cljs.core.PROTOCOL_SENTINEL === inst_40938);
var inst_40940 = ((inst_40937) || (inst_40939));
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_40940)){
var statearr_41130_43042 = state_41057__$1;
(statearr_41130_43042[(1)] = (5));

} else {
var statearr_41131_43043 = state_41057__$1;
(statearr_41131_43043[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (23))){
var inst_40990 = (state_41057[(14)]);
var inst_40999 = (inst_40990 == null);
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_40999)){
var statearr_41132_43048 = state_41057__$1;
(statearr_41132_43048[(1)] = (26));

} else {
var statearr_41133_43049 = state_41057__$1;
(statearr_41133_43049[(1)] = (27));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (35))){
var inst_41023 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
if(cljs.core.truth_(inst_41023)){
var statearr_41136_43053 = state_41057__$1;
(statearr_41136_43053[(1)] = (36));

} else {
var statearr_41137_43054 = state_41057__$1;
(statearr_41137_43054[(1)] = (37));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (19))){
var inst_40956 = (state_41057[(7)]);
var inst_40976 = cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,inst_40956);
var state_41057__$1 = state_41057;
var statearr_41139_43061 = state_41057__$1;
(statearr_41139_43061[(2)] = inst_40976);

(statearr_41139_43061[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (11))){
var inst_40956 = (state_41057[(7)]);
var inst_40960 = (inst_40956 == null);
var inst_40961 = cljs.core.not(inst_40960);
var state_41057__$1 = state_41057;
if(inst_40961){
var statearr_41142_43069 = state_41057__$1;
(statearr_41142_43069[(1)] = (13));

} else {
var statearr_41144_43070 = state_41057__$1;
(statearr_41144_43070[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (9))){
var inst_40926 = (state_41057[(8)]);
var state_41057__$1 = state_41057;
var statearr_41145_43071 = state_41057__$1;
(statearr_41145_43071[(2)] = inst_40926);

(statearr_41145_43071[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (5))){
var state_41057__$1 = state_41057;
var statearr_41146_43073 = state_41057__$1;
(statearr_41146_43073[(2)] = true);

(statearr_41146_43073[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (14))){
var state_41057__$1 = state_41057;
var statearr_41147_43081 = state_41057__$1;
(statearr_41147_43081[(2)] = false);

(statearr_41147_43081[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (26))){
var inst_40991 = (state_41057[(9)]);
var inst_41001 = cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(cs,cljs.core.dissoc,inst_40991);
var state_41057__$1 = state_41057;
var statearr_41150_43089 = state_41057__$1;
(statearr_41150_43089[(2)] = inst_41001);

(statearr_41150_43089[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (16))){
var state_41057__$1 = state_41057;
var statearr_41156_43090 = state_41057__$1;
(statearr_41156_43090[(2)] = true);

(statearr_41156_43090[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (38))){
var inst_41028 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
var statearr_41158_43092 = state_41057__$1;
(statearr_41158_43092[(2)] = inst_41028);

(statearr_41158_43092[(1)] = (34));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (30))){
var inst_40982 = (state_41057[(13)]);
var inst_40991 = (state_41057[(9)]);
var inst_40981 = (state_41057[(11)]);
var inst_41011 = cljs.core.empty_QMARK_(inst_40981);
var inst_41016 = (inst_40982.cljs$core$IFn$_invoke$arity$1 ? inst_40982.cljs$core$IFn$_invoke$arity$1(inst_40991) : inst_40982.call(null,inst_40991));
var inst_41017 = cljs.core.not(inst_41016);
var inst_41018 = ((inst_41011) && (inst_41017));
var state_41057__$1 = state_41057;
var statearr_41159_43102 = state_41057__$1;
(statearr_41159_43102[(2)] = inst_41018);

(statearr_41159_43102[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (10))){
var inst_40926 = (state_41057[(8)]);
var inst_40952 = (state_41057[(2)]);
var inst_40953 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40952,new cljs.core.Keyword(null,"solos","solos",1441458643));
var inst_40954 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40952,new cljs.core.Keyword(null,"mutes","mutes",1068806309));
var inst_40955 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_40952,new cljs.core.Keyword(null,"reads","reads",-1215067361));
var inst_40956 = inst_40926;
var state_41057__$1 = (function (){var statearr_41164 = state_41057;
(statearr_41164[(16)] = inst_40955);

(statearr_41164[(17)] = inst_40953);

(statearr_41164[(7)] = inst_40956);

(statearr_41164[(18)] = inst_40954);

return statearr_41164;
})();
var statearr_41165_43112 = state_41057__$1;
(statearr_41165_43112[(2)] = null);

(statearr_41165_43112[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (18))){
var inst_40971 = (state_41057[(2)]);
var state_41057__$1 = state_41057;
var statearr_41169_43119 = state_41057__$1;
(statearr_41169_43119[(2)] = inst_40971);

(statearr_41169_43119[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (37))){
var state_41057__$1 = state_41057;
var statearr_41171_43122 = state_41057__$1;
(statearr_41171_43122[(2)] = null);

(statearr_41171_43122[(1)] = (38));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41058 === (8))){
var inst_40926 = (state_41057[(8)]);
var inst_40949 = cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,inst_40926);
var state_41057__$1 = state_41057;
var statearr_41172_43132 = state_41057__$1;
(statearr_41172_43132[(2)] = inst_40949);

(statearr_41172_43132[(1)] = (10));


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
});
return (function() {
var cljs$core$async$mix_$_state_machine__39234__auto__ = null;
var cljs$core$async$mix_$_state_machine__39234__auto____0 = (function (){
var statearr_41174 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_41174[(0)] = cljs$core$async$mix_$_state_machine__39234__auto__);

(statearr_41174[(1)] = (1));

return statearr_41174;
});
var cljs$core$async$mix_$_state_machine__39234__auto____1 = (function (state_41057){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41057);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41178){var ex__39237__auto__ = e41178;
var statearr_41179_43146 = state_41057;
(statearr_41179_43146[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41057[(4)]))){
var statearr_41182_43150 = state_41057;
(statearr_41182_43150[(1)] = cljs.core.first((state_41057[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43153 = state_41057;
state_41057 = G__43153;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$mix_$_state_machine__39234__auto__ = function(state_41057){
switch(arguments.length){
case 0:
return cljs$core$async$mix_$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$mix_$_state_machine__39234__auto____1.call(this,state_41057);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mix_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mix_$_state_machine__39234__auto____0;
cljs$core$async$mix_$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mix_$_state_machine__39234__auto____1;
return cljs$core$async$mix_$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41183 = f__39441__auto__();
(statearr_41183[(6)] = c__39440__auto___42959);

return statearr_41183;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return m;
});
/**
 * Adds ch as an input to the mix
 */
cljs.core.async.admix = (function cljs$core$async$admix(mix,ch){
return cljs.core.async.admix_STAR_(mix,ch);
});
/**
 * Removes ch as an input to the mix
 */
cljs.core.async.unmix = (function cljs$core$async$unmix(mix,ch){
return cljs.core.async.unmix_STAR_(mix,ch);
});
/**
 * removes all inputs from the mix
 */
cljs.core.async.unmix_all = (function cljs$core$async$unmix_all(mix){
return cljs.core.async.unmix_all_STAR_(mix);
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
return cljs.core.async.toggle_STAR_(mix,state_map);
});
/**
 * Sets the solo mode of the mix. mode must be one of :mute or :pause
 */
cljs.core.async.solo_mode = (function cljs$core$async$solo_mode(mix,mode){
return cljs.core.async.solo_mode_STAR_(mix,mode);
});

/**
 * @interface
 */
cljs.core.async.Pub = function(){};

var cljs$core$async$Pub$sub_STAR_$dyn_43165 = (function (p,v,ch,close_QMARK_){
var x__4428__auto__ = (((p == null))?null:p);
var m__4429__auto__ = (cljs.core.async.sub_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$4 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$4(p,v,ch,close_QMARK_) : m__4429__auto__.call(null,p,v,ch,close_QMARK_));
} else {
var m__4426__auto__ = (cljs.core.async.sub_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$4 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$4(p,v,ch,close_QMARK_) : m__4426__auto__.call(null,p,v,ch,close_QMARK_));
} else {
throw cljs.core.missing_protocol("Pub.sub*",p);
}
}
});
cljs.core.async.sub_STAR_ = (function cljs$core$async$sub_STAR_(p,v,ch,close_QMARK_){
if((((!((p == null)))) && ((!((p.cljs$core$async$Pub$sub_STAR_$arity$4 == null)))))){
return p.cljs$core$async$Pub$sub_STAR_$arity$4(p,v,ch,close_QMARK_);
} else {
return cljs$core$async$Pub$sub_STAR_$dyn_43165(p,v,ch,close_QMARK_);
}
});

var cljs$core$async$Pub$unsub_STAR_$dyn_43172 = (function (p,v,ch){
var x__4428__auto__ = (((p == null))?null:p);
var m__4429__auto__ = (cljs.core.async.unsub_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(p,v,ch) : m__4429__auto__.call(null,p,v,ch));
} else {
var m__4426__auto__ = (cljs.core.async.unsub_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(p,v,ch) : m__4426__auto__.call(null,p,v,ch));
} else {
throw cljs.core.missing_protocol("Pub.unsub*",p);
}
}
});
cljs.core.async.unsub_STAR_ = (function cljs$core$async$unsub_STAR_(p,v,ch){
if((((!((p == null)))) && ((!((p.cljs$core$async$Pub$unsub_STAR_$arity$3 == null)))))){
return p.cljs$core$async$Pub$unsub_STAR_$arity$3(p,v,ch);
} else {
return cljs$core$async$Pub$unsub_STAR_$dyn_43172(p,v,ch);
}
});

var cljs$core$async$Pub$unsub_all_STAR_$dyn_43177 = (function() {
var G__43178 = null;
var G__43178__1 = (function (p){
var x__4428__auto__ = (((p == null))?null:p);
var m__4429__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(p) : m__4429__auto__.call(null,p));
} else {
var m__4426__auto__ = (cljs.core.async.unsub_all_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(p) : m__4426__auto__.call(null,p));
} else {
throw cljs.core.missing_protocol("Pub.unsub-all*",p);
}
}
});
var G__43178__2 = (function (p,v){
var x__4428__auto__ = (((p == null))?null:p);
var m__4429__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(p,v) : m__4429__auto__.call(null,p,v));
} else {
var m__4426__auto__ = (cljs.core.async.unsub_all_STAR_["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(p,v) : m__4426__auto__.call(null,p,v));
} else {
throw cljs.core.missing_protocol("Pub.unsub-all*",p);
}
}
});
G__43178 = function(p,v){
switch(arguments.length){
case 1:
return G__43178__1.call(this,p);
case 2:
return G__43178__2.call(this,p,v);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
G__43178.cljs$core$IFn$_invoke$arity$1 = G__43178__1;
G__43178.cljs$core$IFn$_invoke$arity$2 = G__43178__2;
return G__43178;
})()
;
cljs.core.async.unsub_all_STAR_ = (function cljs$core$async$unsub_all_STAR_(var_args){
var G__41205 = arguments.length;
switch (G__41205) {
case 1:
return cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$1 = (function (p){
if((((!((p == null)))) && ((!((p.cljs$core$async$Pub$unsub_all_STAR_$arity$1 == null)))))){
return p.cljs$core$async$Pub$unsub_all_STAR_$arity$1(p);
} else {
return cljs$core$async$Pub$unsub_all_STAR_$dyn_43177(p);
}
}));

(cljs.core.async.unsub_all_STAR_.cljs$core$IFn$_invoke$arity$2 = (function (p,v){
if((((!((p == null)))) && ((!((p.cljs$core$async$Pub$unsub_all_STAR_$arity$2 == null)))))){
return p.cljs$core$async$Pub$unsub_all_STAR_$arity$2(p,v);
} else {
return cljs$core$async$Pub$unsub_all_STAR_$dyn_43177(p,v);
}
}));

(cljs.core.async.unsub_all_STAR_.cljs$lang$maxFixedArity = 2);


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
var G__41224 = arguments.length;
switch (G__41224) {
case 2:
return cljs.core.async.pub.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.pub.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.pub.cljs$core$IFn$_invoke$arity$2 = (function (ch,topic_fn){
return cljs.core.async.pub.cljs$core$IFn$_invoke$arity$3(ch,topic_fn,cljs.core.constantly(null));
}));

(cljs.core.async.pub.cljs$core$IFn$_invoke$arity$3 = (function (ch,topic_fn,buf_fn){
var mults = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var ensure_mult = (function (topic){
var or__4126__auto__ = cljs.core.get.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(mults),topic);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return cljs.core.get.cljs$core$IFn$_invoke$arity$2(cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$2(mults,(function (p1__41222_SHARP_){
if(cljs.core.truth_((p1__41222_SHARP_.cljs$core$IFn$_invoke$arity$1 ? p1__41222_SHARP_.cljs$core$IFn$_invoke$arity$1(topic) : p1__41222_SHARP_.call(null,topic)))){
return p1__41222_SHARP_;
} else {
return cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(p1__41222_SHARP_,topic,cljs.core.async.mult(cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((buf_fn.cljs$core$IFn$_invoke$arity$1 ? buf_fn.cljs$core$IFn$_invoke$arity$1(topic) : buf_fn.call(null,topic)))));
}
})),topic);
}
});
var p = (function (){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async41239 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.Pub}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.async.Mux}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async41239 = (function (ch,topic_fn,buf_fn,mults,ensure_mult,meta41240){
this.ch = ch;
this.topic_fn = topic_fn;
this.buf_fn = buf_fn;
this.mults = mults;
this.ensure_mult = ensure_mult;
this.meta41240 = meta41240;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_41241,meta41240__$1){
var self__ = this;
var _41241__$1 = this;
return (new cljs.core.async.t_cljs$core$async41239(self__.ch,self__.topic_fn,self__.buf_fn,self__.mults,self__.ensure_mult,meta41240__$1));
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_41241){
var self__ = this;
var _41241__$1 = this;
return self__.meta41240;
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Mux$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.ch;
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Pub$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Pub$sub_STAR_$arity$4 = (function (p,topic,ch__$1,close_QMARK_){
var self__ = this;
var p__$1 = this;
var m = (self__.ensure_mult.cljs$core$IFn$_invoke$arity$1 ? self__.ensure_mult.cljs$core$IFn$_invoke$arity$1(topic) : self__.ensure_mult.call(null,topic));
return cljs.core.async.tap.cljs$core$IFn$_invoke$arity$3(m,ch__$1,close_QMARK_);
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Pub$unsub_STAR_$arity$3 = (function (p,topic,ch__$1){
var self__ = this;
var p__$1 = this;
var temp__5735__auto__ = cljs.core.get.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(self__.mults),topic);
if(cljs.core.truth_(temp__5735__auto__)){
var m = temp__5735__auto__;
return cljs.core.async.untap(m,ch__$1);
} else {
return null;
}
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.reset_BANG_(self__.mults,cljs.core.PersistentArrayMap.EMPTY);
}));

(cljs.core.async.t_cljs$core$async41239.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$2 = (function (_,topic){
var self__ = this;
var ___$1 = this;
return cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(self__.mults,cljs.core.dissoc,topic);
}));

(cljs.core.async.t_cljs$core$async41239.getBasis = (function (){
return new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"topic-fn","topic-fn",-862449736,null),new cljs.core.Symbol(null,"buf-fn","buf-fn",-1200281591,null),new cljs.core.Symbol(null,"mults","mults",-461114485,null),new cljs.core.Symbol(null,"ensure-mult","ensure-mult",1796584816,null),new cljs.core.Symbol(null,"meta41240","meta41240",1132441152,null)], null);
}));

(cljs.core.async.t_cljs$core$async41239.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async41239.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async41239");

(cljs.core.async.t_cljs$core$async41239.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async41239");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async41239.
 */
cljs.core.async.__GT_t_cljs$core$async41239 = (function cljs$core$async$__GT_t_cljs$core$async41239(ch__$1,topic_fn__$1,buf_fn__$1,mults__$1,ensure_mult__$1,meta41240){
return (new cljs.core.async.t_cljs$core$async41239(ch__$1,topic_fn__$1,buf_fn__$1,mults__$1,ensure_mult__$1,meta41240));
});

}

return (new cljs.core.async.t_cljs$core$async41239(ch,topic_fn,buf_fn,mults,ensure_mult,cljs.core.PersistentArrayMap.EMPTY));
})()
;
var c__39440__auto___43217 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41320){
var state_val_41321 = (state_41320[(1)]);
if((state_val_41321 === (7))){
var inst_41315 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41326_43218 = state_41320__$1;
(statearr_41326_43218[(2)] = inst_41315);

(statearr_41326_43218[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (20))){
var state_41320__$1 = state_41320;
var statearr_41328_43219 = state_41320__$1;
(statearr_41328_43219[(2)] = null);

(statearr_41328_43219[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (1))){
var state_41320__$1 = state_41320;
var statearr_41329_43220 = state_41320__$1;
(statearr_41329_43220[(2)] = null);

(statearr_41329_43220[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (24))){
var inst_41298 = (state_41320[(7)]);
var inst_41307 = cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(mults,cljs.core.dissoc,inst_41298);
var state_41320__$1 = state_41320;
var statearr_41330_43221 = state_41320__$1;
(statearr_41330_43221[(2)] = inst_41307);

(statearr_41330_43221[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (4))){
var inst_41249 = (state_41320[(8)]);
var inst_41249__$1 = (state_41320[(2)]);
var inst_41250 = (inst_41249__$1 == null);
var state_41320__$1 = (function (){var statearr_41336 = state_41320;
(statearr_41336[(8)] = inst_41249__$1);

return statearr_41336;
})();
if(cljs.core.truth_(inst_41250)){
var statearr_41337_43222 = state_41320__$1;
(statearr_41337_43222[(1)] = (5));

} else {
var statearr_41338_43223 = state_41320__$1;
(statearr_41338_43223[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (15))){
var inst_41292 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41340_43224 = state_41320__$1;
(statearr_41340_43224[(2)] = inst_41292);

(statearr_41340_43224[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (21))){
var inst_41312 = (state_41320[(2)]);
var state_41320__$1 = (function (){var statearr_41344 = state_41320;
(statearr_41344[(9)] = inst_41312);

return statearr_41344;
})();
var statearr_41345_43225 = state_41320__$1;
(statearr_41345_43225[(2)] = null);

(statearr_41345_43225[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (13))){
var inst_41274 = (state_41320[(10)]);
var inst_41276 = cljs.core.chunked_seq_QMARK_(inst_41274);
var state_41320__$1 = state_41320;
if(inst_41276){
var statearr_41347_43226 = state_41320__$1;
(statearr_41347_43226[(1)] = (16));

} else {
var statearr_41348_43227 = state_41320__$1;
(statearr_41348_43227[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (22))){
var inst_41304 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
if(cljs.core.truth_(inst_41304)){
var statearr_41349_43230 = state_41320__$1;
(statearr_41349_43230[(1)] = (23));

} else {
var statearr_41350_43231 = state_41320__$1;
(statearr_41350_43231[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (6))){
var inst_41298 = (state_41320[(7)]);
var inst_41249 = (state_41320[(8)]);
var inst_41300 = (state_41320[(11)]);
var inst_41298__$1 = (topic_fn.cljs$core$IFn$_invoke$arity$1 ? topic_fn.cljs$core$IFn$_invoke$arity$1(inst_41249) : topic_fn.call(null,inst_41249));
var inst_41299 = cljs.core.deref(mults);
var inst_41300__$1 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(inst_41299,inst_41298__$1);
var state_41320__$1 = (function (){var statearr_41352 = state_41320;
(statearr_41352[(7)] = inst_41298__$1);

(statearr_41352[(11)] = inst_41300__$1);

return statearr_41352;
})();
if(cljs.core.truth_(inst_41300__$1)){
var statearr_41353_43232 = state_41320__$1;
(statearr_41353_43232[(1)] = (19));

} else {
var statearr_41356_43233 = state_41320__$1;
(statearr_41356_43233[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (25))){
var inst_41309 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41358_43234 = state_41320__$1;
(statearr_41358_43234[(2)] = inst_41309);

(statearr_41358_43234[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (17))){
var inst_41274 = (state_41320[(10)]);
var inst_41283 = cljs.core.first(inst_41274);
var inst_41284 = cljs.core.async.muxch_STAR_(inst_41283);
var inst_41285 = cljs.core.async.close_BANG_(inst_41284);
var inst_41286 = cljs.core.next(inst_41274);
var inst_41259 = inst_41286;
var inst_41260 = null;
var inst_41261 = (0);
var inst_41262 = (0);
var state_41320__$1 = (function (){var statearr_41359 = state_41320;
(statearr_41359[(12)] = inst_41260);

(statearr_41359[(13)] = inst_41261);

(statearr_41359[(14)] = inst_41259);

(statearr_41359[(15)] = inst_41262);

(statearr_41359[(16)] = inst_41285);

return statearr_41359;
})();
var statearr_41363_43237 = state_41320__$1;
(statearr_41363_43237[(2)] = null);

(statearr_41363_43237[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (3))){
var inst_41317 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41320__$1,inst_41317);
} else {
if((state_val_41321 === (12))){
var inst_41294 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41364_43241 = state_41320__$1;
(statearr_41364_43241[(2)] = inst_41294);

(statearr_41364_43241[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (2))){
var state_41320__$1 = state_41320;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_41320__$1,(4),ch);
} else {
if((state_val_41321 === (23))){
var state_41320__$1 = state_41320;
var statearr_41366_43242 = state_41320__$1;
(statearr_41366_43242[(2)] = null);

(statearr_41366_43242[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (19))){
var inst_41249 = (state_41320[(8)]);
var inst_41300 = (state_41320[(11)]);
var inst_41302 = cljs.core.async.muxch_STAR_(inst_41300);
var state_41320__$1 = state_41320;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41320__$1,(22),inst_41302,inst_41249);
} else {
if((state_val_41321 === (11))){
var inst_41259 = (state_41320[(14)]);
var inst_41274 = (state_41320[(10)]);
var inst_41274__$1 = cljs.core.seq(inst_41259);
var state_41320__$1 = (function (){var statearr_41375 = state_41320;
(statearr_41375[(10)] = inst_41274__$1);

return statearr_41375;
})();
if(inst_41274__$1){
var statearr_41376_43243 = state_41320__$1;
(statearr_41376_43243[(1)] = (13));

} else {
var statearr_41377_43244 = state_41320__$1;
(statearr_41377_43244[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (9))){
var inst_41296 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41381_43245 = state_41320__$1;
(statearr_41381_43245[(2)] = inst_41296);

(statearr_41381_43245[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (5))){
var inst_41256 = cljs.core.deref(mults);
var inst_41257 = cljs.core.vals(inst_41256);
var inst_41258 = cljs.core.seq(inst_41257);
var inst_41259 = inst_41258;
var inst_41260 = null;
var inst_41261 = (0);
var inst_41262 = (0);
var state_41320__$1 = (function (){var statearr_41383 = state_41320;
(statearr_41383[(12)] = inst_41260);

(statearr_41383[(13)] = inst_41261);

(statearr_41383[(14)] = inst_41259);

(statearr_41383[(15)] = inst_41262);

return statearr_41383;
})();
var statearr_41384_43247 = state_41320__$1;
(statearr_41384_43247[(2)] = null);

(statearr_41384_43247[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (14))){
var state_41320__$1 = state_41320;
var statearr_41389_43248 = state_41320__$1;
(statearr_41389_43248[(2)] = null);

(statearr_41389_43248[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (16))){
var inst_41274 = (state_41320[(10)]);
var inst_41278 = cljs.core.chunk_first(inst_41274);
var inst_41279 = cljs.core.chunk_rest(inst_41274);
var inst_41280 = cljs.core.count(inst_41278);
var inst_41259 = inst_41279;
var inst_41260 = inst_41278;
var inst_41261 = inst_41280;
var inst_41262 = (0);
var state_41320__$1 = (function (){var statearr_41393 = state_41320;
(statearr_41393[(12)] = inst_41260);

(statearr_41393[(13)] = inst_41261);

(statearr_41393[(14)] = inst_41259);

(statearr_41393[(15)] = inst_41262);

return statearr_41393;
})();
var statearr_41394_43253 = state_41320__$1;
(statearr_41394_43253[(2)] = null);

(statearr_41394_43253[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (10))){
var inst_41260 = (state_41320[(12)]);
var inst_41261 = (state_41320[(13)]);
var inst_41259 = (state_41320[(14)]);
var inst_41262 = (state_41320[(15)]);
var inst_41268 = cljs.core._nth(inst_41260,inst_41262);
var inst_41269 = cljs.core.async.muxch_STAR_(inst_41268);
var inst_41270 = cljs.core.async.close_BANG_(inst_41269);
var inst_41271 = (inst_41262 + (1));
var tmp41386 = inst_41260;
var tmp41387 = inst_41261;
var tmp41388 = inst_41259;
var inst_41259__$1 = tmp41388;
var inst_41260__$1 = tmp41386;
var inst_41261__$1 = tmp41387;
var inst_41262__$1 = inst_41271;
var state_41320__$1 = (function (){var statearr_41405 = state_41320;
(statearr_41405[(12)] = inst_41260__$1);

(statearr_41405[(13)] = inst_41261__$1);

(statearr_41405[(14)] = inst_41259__$1);

(statearr_41405[(15)] = inst_41262__$1);

(statearr_41405[(17)] = inst_41270);

return statearr_41405;
})();
var statearr_41406_43269 = state_41320__$1;
(statearr_41406_43269[(2)] = null);

(statearr_41406_43269[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (18))){
var inst_41289 = (state_41320[(2)]);
var state_41320__$1 = state_41320;
var statearr_41410_43272 = state_41320__$1;
(statearr_41410_43272[(2)] = inst_41289);

(statearr_41410_43272[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41321 === (8))){
var inst_41261 = (state_41320[(13)]);
var inst_41262 = (state_41320[(15)]);
var inst_41265 = (inst_41262 < inst_41261);
var inst_41266 = inst_41265;
var state_41320__$1 = state_41320;
if(cljs.core.truth_(inst_41266)){
var statearr_41414_43275 = state_41320__$1;
(statearr_41414_43275[(1)] = (10));

} else {
var statearr_41416_43276 = state_41320__$1;
(statearr_41416_43276[(1)] = (11));

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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_41417 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_41417[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_41417[(1)] = (1));

return statearr_41417;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_41320){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41320);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41418){var ex__39237__auto__ = e41418;
var statearr_41419_43284 = state_41320;
(statearr_41419_43284[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41320[(4)]))){
var statearr_41420_43285 = state_41320;
(statearr_41420_43285[(1)] = cljs.core.first((state_41320[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43286 = state_41320;
state_41320 = G__43286;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_41320){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_41320);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41421 = f__39441__auto__();
(statearr_41421[(6)] = c__39440__auto___43217);

return statearr_41421;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return p;
}));

(cljs.core.async.pub.cljs$lang$maxFixedArity = 3);

/**
 * Subscribes a channel to a topic of a pub.
 * 
 *   By default the channel will be closed when the source closes,
 *   but can be determined by the close? parameter.
 */
cljs.core.async.sub = (function cljs$core$async$sub(var_args){
var G__41425 = arguments.length;
switch (G__41425) {
case 3:
return cljs.core.async.sub.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return cljs.core.async.sub.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.sub.cljs$core$IFn$_invoke$arity$3 = (function (p,topic,ch){
return cljs.core.async.sub.cljs$core$IFn$_invoke$arity$4(p,topic,ch,true);
}));

(cljs.core.async.sub.cljs$core$IFn$_invoke$arity$4 = (function (p,topic,ch,close_QMARK_){
return cljs.core.async.sub_STAR_(p,topic,ch,close_QMARK_);
}));

(cljs.core.async.sub.cljs$lang$maxFixedArity = 4);

/**
 * Unsubscribes a channel from a topic of a pub
 */
cljs.core.async.unsub = (function cljs$core$async$unsub(p,topic,ch){
return cljs.core.async.unsub_STAR_(p,topic,ch);
});
/**
 * Unsubscribes all channels from a pub, or a topic of a pub
 */
cljs.core.async.unsub_all = (function cljs$core$async$unsub_all(var_args){
var G__41436 = arguments.length;
switch (G__41436) {
case 1:
return cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$1 = (function (p){
return cljs.core.async.unsub_all_STAR_(p);
}));

(cljs.core.async.unsub_all.cljs$core$IFn$_invoke$arity$2 = (function (p,topic){
return cljs.core.async.unsub_all_STAR_(p,topic);
}));

(cljs.core.async.unsub_all.cljs$lang$maxFixedArity = 2);

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
var G__41443 = arguments.length;
switch (G__41443) {
case 2:
return cljs.core.async.map.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.map.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.map.cljs$core$IFn$_invoke$arity$2 = (function (f,chs){
return cljs.core.async.map.cljs$core$IFn$_invoke$arity$3(f,chs,null);
}));

(cljs.core.async.map.cljs$core$IFn$_invoke$arity$3 = (function (f,chs,buf_or_n){
var chs__$1 = cljs.core.vec(chs);
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var cnt = cljs.core.count(chs__$1);
var rets = cljs.core.object_array.cljs$core$IFn$_invoke$arity$1(cnt);
var dchan = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
var dctr = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(null);
var done = cljs.core.mapv.cljs$core$IFn$_invoke$arity$2((function (i){
return (function (ret){
(rets[i] = ret);

if((cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$2(dctr,cljs.core.dec) === (0))){
return cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(dchan,rets.slice((0)));
} else {
return null;
}
});
}),cljs.core.range.cljs$core$IFn$_invoke$arity$1(cnt));
var c__39440__auto___43294 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41505){
var state_val_41506 = (state_41505[(1)]);
if((state_val_41506 === (7))){
var state_41505__$1 = state_41505;
var statearr_41512_43295 = state_41505__$1;
(statearr_41512_43295[(2)] = null);

(statearr_41512_43295[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (1))){
var state_41505__$1 = state_41505;
var statearr_41517_43300 = state_41505__$1;
(statearr_41517_43300[(2)] = null);

(statearr_41517_43300[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (4))){
var inst_41457 = (state_41505[(7)]);
var inst_41458 = (state_41505[(8)]);
var inst_41460 = (inst_41458 < inst_41457);
var state_41505__$1 = state_41505;
if(cljs.core.truth_(inst_41460)){
var statearr_41518_43309 = state_41505__$1;
(statearr_41518_43309[(1)] = (6));

} else {
var statearr_41519_43310 = state_41505__$1;
(statearr_41519_43310[(1)] = (7));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (15))){
var inst_41487 = (state_41505[(9)]);
var inst_41494 = cljs.core.apply.cljs$core$IFn$_invoke$arity$2(f,inst_41487);
var state_41505__$1 = state_41505;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41505__$1,(17),out,inst_41494);
} else {
if((state_val_41506 === (13))){
var inst_41487 = (state_41505[(9)]);
var inst_41487__$1 = (state_41505[(2)]);
var inst_41490 = cljs.core.some(cljs.core.nil_QMARK_,inst_41487__$1);
var state_41505__$1 = (function (){var statearr_41520 = state_41505;
(statearr_41520[(9)] = inst_41487__$1);

return statearr_41520;
})();
if(cljs.core.truth_(inst_41490)){
var statearr_41521_43315 = state_41505__$1;
(statearr_41521_43315[(1)] = (14));

} else {
var statearr_41522_43316 = state_41505__$1;
(statearr_41522_43316[(1)] = (15));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (6))){
var state_41505__$1 = state_41505;
var statearr_41523_43318 = state_41505__$1;
(statearr_41523_43318[(2)] = null);

(statearr_41523_43318[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (17))){
var inst_41496 = (state_41505[(2)]);
var state_41505__$1 = (function (){var statearr_41528 = state_41505;
(statearr_41528[(10)] = inst_41496);

return statearr_41528;
})();
var statearr_41529_43320 = state_41505__$1;
(statearr_41529_43320[(2)] = null);

(statearr_41529_43320[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (3))){
var inst_41501 = (state_41505[(2)]);
var state_41505__$1 = state_41505;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41505__$1,inst_41501);
} else {
if((state_val_41506 === (12))){
var _ = (function (){var statearr_41530 = state_41505;
(statearr_41530[(4)] = cljs.core.rest((state_41505[(4)])));

return statearr_41530;
})();
var state_41505__$1 = state_41505;
var ex41525 = (state_41505__$1[(2)]);
var statearr_41531_43322 = state_41505__$1;
(statearr_41531_43322[(5)] = ex41525);


if((ex41525 instanceof Object)){
var statearr_41536_43323 = state_41505__$1;
(statearr_41536_43323[(1)] = (11));

(statearr_41536_43323[(5)] = null);

} else {
throw ex41525;

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (2))){
var inst_41456 = cljs.core.reset_BANG_(dctr,cnt);
var inst_41457 = cnt;
var inst_41458 = (0);
var state_41505__$1 = (function (){var statearr_41540 = state_41505;
(statearr_41540[(11)] = inst_41456);

(statearr_41540[(7)] = inst_41457);

(statearr_41540[(8)] = inst_41458);

return statearr_41540;
})();
var statearr_41541_43324 = state_41505__$1;
(statearr_41541_43324[(2)] = null);

(statearr_41541_43324[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (11))){
var inst_41465 = (state_41505[(2)]);
var inst_41466 = cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$2(dctr,cljs.core.dec);
var state_41505__$1 = (function (){var statearr_41546 = state_41505;
(statearr_41546[(12)] = inst_41465);

return statearr_41546;
})();
var statearr_41548_43326 = state_41505__$1;
(statearr_41548_43326[(2)] = inst_41466);

(statearr_41548_43326[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (9))){
var inst_41458 = (state_41505[(8)]);
var _ = (function (){var statearr_41550 = state_41505;
(statearr_41550[(4)] = cljs.core.cons((12),(state_41505[(4)])));

return statearr_41550;
})();
var inst_41473 = (chs__$1.cljs$core$IFn$_invoke$arity$1 ? chs__$1.cljs$core$IFn$_invoke$arity$1(inst_41458) : chs__$1.call(null,inst_41458));
var inst_41474 = (done.cljs$core$IFn$_invoke$arity$1 ? done.cljs$core$IFn$_invoke$arity$1(inst_41458) : done.call(null,inst_41458));
var inst_41475 = cljs.core.async.take_BANG_.cljs$core$IFn$_invoke$arity$2(inst_41473,inst_41474);
var ___$1 = (function (){var statearr_41551 = state_41505;
(statearr_41551[(4)] = cljs.core.rest((state_41505[(4)])));

return statearr_41551;
})();
var state_41505__$1 = state_41505;
var statearr_41555_43328 = state_41505__$1;
(statearr_41555_43328[(2)] = inst_41475);

(statearr_41555_43328[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (5))){
var inst_41485 = (state_41505[(2)]);
var state_41505__$1 = (function (){var statearr_41557 = state_41505;
(statearr_41557[(13)] = inst_41485);

return statearr_41557;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_41505__$1,(13),dchan);
} else {
if((state_val_41506 === (14))){
var inst_41492 = cljs.core.async.close_BANG_(out);
var state_41505__$1 = state_41505;
var statearr_41559_43330 = state_41505__$1;
(statearr_41559_43330[(2)] = inst_41492);

(statearr_41559_43330[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (16))){
var inst_41499 = (state_41505[(2)]);
var state_41505__$1 = state_41505;
var statearr_41560_43331 = state_41505__$1;
(statearr_41560_43331[(2)] = inst_41499);

(statearr_41560_43331[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (10))){
var inst_41458 = (state_41505[(8)]);
var inst_41478 = (state_41505[(2)]);
var inst_41479 = (inst_41458 + (1));
var inst_41458__$1 = inst_41479;
var state_41505__$1 = (function (){var statearr_41561 = state_41505;
(statearr_41561[(8)] = inst_41458__$1);

(statearr_41561[(14)] = inst_41478);

return statearr_41561;
})();
var statearr_41563_43334 = state_41505__$1;
(statearr_41563_43334[(2)] = null);

(statearr_41563_43334[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41506 === (8))){
var inst_41483 = (state_41505[(2)]);
var state_41505__$1 = state_41505;
var statearr_41566_43335 = state_41505__$1;
(statearr_41566_43335[(2)] = inst_41483);

(statearr_41566_43335[(1)] = (5));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_41576 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_41576[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_41576[(1)] = (1));

return statearr_41576;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_41505){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41505);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41582){var ex__39237__auto__ = e41582;
var statearr_41583_43336 = state_41505;
(statearr_41583_43336[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41505[(4)]))){
var statearr_41585_43338 = state_41505;
(statearr_41585_43338[(1)] = cljs.core.first((state_41505[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43339 = state_41505;
state_41505 = G__43339;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_41505){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_41505);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41588 = f__39441__auto__();
(statearr_41588[(6)] = c__39440__auto___43294);

return statearr_41588;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.map.cljs$lang$maxFixedArity = 3);

/**
 * Takes a collection of source channels and returns a channel which
 *   contains all values taken from them. The returned channel will be
 *   unbuffered by default, or a buf-or-n can be supplied. The channel
 *   will close after all the source channels have closed.
 */
cljs.core.async.merge = (function cljs$core$async$merge(var_args){
var G__41592 = arguments.length;
switch (G__41592) {
case 1:
return cljs.core.async.merge.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.merge.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.merge.cljs$core$IFn$_invoke$arity$1 = (function (chs){
return cljs.core.async.merge.cljs$core$IFn$_invoke$arity$2(chs,null);
}));

(cljs.core.async.merge.cljs$core$IFn$_invoke$arity$2 = (function (chs,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43342 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41629){
var state_val_41630 = (state_41629[(1)]);
if((state_val_41630 === (7))){
var inst_41603 = (state_41629[(7)]);
var inst_41604 = (state_41629[(8)]);
var inst_41603__$1 = (state_41629[(2)]);
var inst_41604__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_41603__$1,(0),null);
var inst_41605 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(inst_41603__$1,(1),null);
var inst_41606 = (inst_41604__$1 == null);
var state_41629__$1 = (function (){var statearr_41631 = state_41629;
(statearr_41631[(7)] = inst_41603__$1);

(statearr_41631[(9)] = inst_41605);

(statearr_41631[(8)] = inst_41604__$1);

return statearr_41631;
})();
if(cljs.core.truth_(inst_41606)){
var statearr_41632_43343 = state_41629__$1;
(statearr_41632_43343[(1)] = (8));

} else {
var statearr_41637_43344 = state_41629__$1;
(statearr_41637_43344[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (1))){
var inst_41593 = cljs.core.vec(chs);
var inst_41594 = inst_41593;
var state_41629__$1 = (function (){var statearr_41638 = state_41629;
(statearr_41638[(10)] = inst_41594);

return statearr_41638;
})();
var statearr_41639_43347 = state_41629__$1;
(statearr_41639_43347[(2)] = null);

(statearr_41639_43347[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (4))){
var inst_41594 = (state_41629[(10)]);
var state_41629__$1 = state_41629;
return cljs.core.async.ioc_alts_BANG_(state_41629__$1,(7),inst_41594);
} else {
if((state_val_41630 === (6))){
var inst_41623 = (state_41629[(2)]);
var state_41629__$1 = state_41629;
var statearr_41645_43348 = state_41629__$1;
(statearr_41645_43348[(2)] = inst_41623);

(statearr_41645_43348[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (3))){
var inst_41625 = (state_41629[(2)]);
var state_41629__$1 = state_41629;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41629__$1,inst_41625);
} else {
if((state_val_41630 === (2))){
var inst_41594 = (state_41629[(10)]);
var inst_41596 = cljs.core.count(inst_41594);
var inst_41597 = (inst_41596 > (0));
var state_41629__$1 = state_41629;
if(cljs.core.truth_(inst_41597)){
var statearr_41647_43351 = state_41629__$1;
(statearr_41647_43351[(1)] = (4));

} else {
var statearr_41648_43353 = state_41629__$1;
(statearr_41648_43353[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (11))){
var inst_41594 = (state_41629[(10)]);
var inst_41616 = (state_41629[(2)]);
var tmp41646 = inst_41594;
var inst_41594__$1 = tmp41646;
var state_41629__$1 = (function (){var statearr_41649 = state_41629;
(statearr_41649[(11)] = inst_41616);

(statearr_41649[(10)] = inst_41594__$1);

return statearr_41649;
})();
var statearr_41650_43361 = state_41629__$1;
(statearr_41650_43361[(2)] = null);

(statearr_41650_43361[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (9))){
var inst_41604 = (state_41629[(8)]);
var state_41629__$1 = state_41629;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41629__$1,(11),out,inst_41604);
} else {
if((state_val_41630 === (5))){
var inst_41621 = cljs.core.async.close_BANG_(out);
var state_41629__$1 = state_41629;
var statearr_41653_43368 = state_41629__$1;
(statearr_41653_43368[(2)] = inst_41621);

(statearr_41653_43368[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (10))){
var inst_41619 = (state_41629[(2)]);
var state_41629__$1 = state_41629;
var statearr_41654_43377 = state_41629__$1;
(statearr_41654_43377[(2)] = inst_41619);

(statearr_41654_43377[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41630 === (8))){
var inst_41603 = (state_41629[(7)]);
var inst_41605 = (state_41629[(9)]);
var inst_41594 = (state_41629[(10)]);
var inst_41604 = (state_41629[(8)]);
var inst_41608 = (function (){var cs = inst_41594;
var vec__41599 = inst_41603;
var v = inst_41604;
var c = inst_41605;
return (function (p1__41589_SHARP_){
return cljs.core.not_EQ_.cljs$core$IFn$_invoke$arity$2(c,p1__41589_SHARP_);
});
})();
var inst_41612 = cljs.core.filterv(inst_41608,inst_41594);
var inst_41594__$1 = inst_41612;
var state_41629__$1 = (function (){var statearr_41655 = state_41629;
(statearr_41655[(10)] = inst_41594__$1);

return statearr_41655;
})();
var statearr_41658_43391 = state_41629__$1;
(statearr_41658_43391[(2)] = null);

(statearr_41658_43391[(1)] = (2));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_41660 = [null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_41660[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_41660[(1)] = (1));

return statearr_41660;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_41629){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41629);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41661){var ex__39237__auto__ = e41661;
var statearr_41662_43396 = state_41629;
(statearr_41662_43396[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41629[(4)]))){
var statearr_41663_43401 = state_41629;
(statearr_41663_43401[(1)] = cljs.core.first((state_41629[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43402 = state_41629;
state_41629 = G__43402;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_41629){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_41629);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41665 = f__39441__auto__();
(statearr_41665[(6)] = c__39440__auto___43342);

return statearr_41665;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.merge.cljs$lang$maxFixedArity = 2);

/**
 * Returns a channel containing the single (collection) result of the
 *   items taken from the channel conjoined to the supplied
 *   collection. ch must close before into produces a result.
 */
cljs.core.async.into = (function cljs$core$async$into(coll,ch){
return cljs.core.async.reduce(cljs.core.conj,coll,ch);
});
/**
 * Returns a channel that will return, at most, n items from ch. After n items
 * have been returned, or ch has been closed, the return chanel will close.
 * 
 *   The output channel is unbuffered by default, unless buf-or-n is given.
 */
cljs.core.async.take = (function cljs$core$async$take(var_args){
var G__41668 = arguments.length;
switch (G__41668) {
case 2:
return cljs.core.async.take.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.take.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.take.cljs$core$IFn$_invoke$arity$2 = (function (n,ch){
return cljs.core.async.take.cljs$core$IFn$_invoke$arity$3(n,ch,null);
}));

(cljs.core.async.take.cljs$core$IFn$_invoke$arity$3 = (function (n,ch,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43427 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41696){
var state_val_41697 = (state_41696[(1)]);
if((state_val_41697 === (7))){
var inst_41678 = (state_41696[(7)]);
var inst_41678__$1 = (state_41696[(2)]);
var inst_41679 = (inst_41678__$1 == null);
var inst_41680 = cljs.core.not(inst_41679);
var state_41696__$1 = (function (){var statearr_41698 = state_41696;
(statearr_41698[(7)] = inst_41678__$1);

return statearr_41698;
})();
if(inst_41680){
var statearr_41699_43432 = state_41696__$1;
(statearr_41699_43432[(1)] = (8));

} else {
var statearr_41700_43433 = state_41696__$1;
(statearr_41700_43433[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (1))){
var inst_41672 = (0);
var state_41696__$1 = (function (){var statearr_41701 = state_41696;
(statearr_41701[(8)] = inst_41672);

return statearr_41701;
})();
var statearr_41702_43438 = state_41696__$1;
(statearr_41702_43438[(2)] = null);

(statearr_41702_43438[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (4))){
var state_41696__$1 = state_41696;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_41696__$1,(7),ch);
} else {
if((state_val_41697 === (6))){
var inst_41691 = (state_41696[(2)]);
var state_41696__$1 = state_41696;
var statearr_41703_43443 = state_41696__$1;
(statearr_41703_43443[(2)] = inst_41691);

(statearr_41703_43443[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (3))){
var inst_41693 = (state_41696[(2)]);
var inst_41694 = cljs.core.async.close_BANG_(out);
var state_41696__$1 = (function (){var statearr_41705 = state_41696;
(statearr_41705[(9)] = inst_41693);

return statearr_41705;
})();
return cljs.core.async.impl.ioc_helpers.return_chan(state_41696__$1,inst_41694);
} else {
if((state_val_41697 === (2))){
var inst_41672 = (state_41696[(8)]);
var inst_41675 = (inst_41672 < n);
var state_41696__$1 = state_41696;
if(cljs.core.truth_(inst_41675)){
var statearr_41706_43444 = state_41696__$1;
(statearr_41706_43444[(1)] = (4));

} else {
var statearr_41707_43449 = state_41696__$1;
(statearr_41707_43449[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (11))){
var inst_41672 = (state_41696[(8)]);
var inst_41683 = (state_41696[(2)]);
var inst_41684 = (inst_41672 + (1));
var inst_41672__$1 = inst_41684;
var state_41696__$1 = (function (){var statearr_41709 = state_41696;
(statearr_41709[(8)] = inst_41672__$1);

(statearr_41709[(10)] = inst_41683);

return statearr_41709;
})();
var statearr_41710_43450 = state_41696__$1;
(statearr_41710_43450[(2)] = null);

(statearr_41710_43450[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (9))){
var state_41696__$1 = state_41696;
var statearr_41711_43452 = state_41696__$1;
(statearr_41711_43452[(2)] = null);

(statearr_41711_43452[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (5))){
var state_41696__$1 = state_41696;
var statearr_41712_43453 = state_41696__$1;
(statearr_41712_43453[(2)] = null);

(statearr_41712_43453[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (10))){
var inst_41688 = (state_41696[(2)]);
var state_41696__$1 = state_41696;
var statearr_41713_43454 = state_41696__$1;
(statearr_41713_43454[(2)] = inst_41688);

(statearr_41713_43454[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41697 === (8))){
var inst_41678 = (state_41696[(7)]);
var state_41696__$1 = state_41696;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41696__$1,(11),out,inst_41678);
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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_41716 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_41716[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_41716[(1)] = (1));

return statearr_41716;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_41696){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41696);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41717){var ex__39237__auto__ = e41717;
var statearr_41718_43458 = state_41696;
(statearr_41718_43458[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41696[(4)]))){
var statearr_41723_43460 = state_41696;
(statearr_41723_43460[(1)] = cljs.core.first((state_41696[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43461 = state_41696;
state_41696 = G__43461;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_41696){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_41696);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41732 = f__39441__auto__();
(statearr_41732[(6)] = c__39440__auto___43427);

return statearr_41732;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.take.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.map_LT_ = (function cljs$core$async$map_LT_(f,ch){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async41738 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async41738 = (function (f,ch,meta41739){
this.f = f;
this.ch = ch;
this.meta41739 = meta41739;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_41740,meta41739__$1){
var self__ = this;
var _41740__$1 = this;
return (new cljs.core.async.t_cljs$core$async41738(self__.f,self__.ch,meta41739__$1));
}));

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_41740){
var self__ = this;
var _41740__$1 = this;
return self__.meta41739;
}));

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$Channel$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_(self__.ch);
}));

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.closed_QMARK_(self__.ch);
}));

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$ReadPort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
var ret = cljs.core.async.impl.protocols.take_BANG_(self__.ch,(function (){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async41741 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Handler}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async41741 = (function (f,ch,meta41739,_,fn1,meta41742){
this.f = f;
this.ch = ch;
this.meta41739 = meta41739;
this._ = _;
this.fn1 = fn1;
this.meta41742 = meta41742;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_41743,meta41742__$1){
var self__ = this;
var _41743__$1 = this;
return (new cljs.core.async.t_cljs$core$async41741(self__.f,self__.ch,self__.meta41739,self__._,self__.fn1,meta41742__$1));
}));

(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_41743){
var self__ = this;
var _41743__$1 = this;
return self__.meta41742;
}));

(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$async$impl$protocols$Handler$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (___$1){
var self__ = this;
var ___$2 = this;
return cljs.core.async.impl.protocols.active_QMARK_(self__.fn1);
}));

(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$async$impl$protocols$Handler$blockable_QMARK_$arity$1 = (function (___$1){
var self__ = this;
var ___$2 = this;
return true;
}));

(cljs.core.async.t_cljs$core$async41741.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (___$1){
var self__ = this;
var ___$2 = this;
var f1 = cljs.core.async.impl.protocols.commit(self__.fn1);
return (function (p1__41737_SHARP_){
var G__41744 = (((p1__41737_SHARP_ == null))?null:(self__.f.cljs$core$IFn$_invoke$arity$1 ? self__.f.cljs$core$IFn$_invoke$arity$1(p1__41737_SHARP_) : self__.f.call(null,p1__41737_SHARP_)));
return (f1.cljs$core$IFn$_invoke$arity$1 ? f1.cljs$core$IFn$_invoke$arity$1(G__41744) : f1.call(null,G__41744));
});
}));

(cljs.core.async.t_cljs$core$async41741.getBasis = (function (){
return new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta41739","meta41739",-704159318,null),cljs.core.with_meta(new cljs.core.Symbol(null,"_","_",-1201019570,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol("cljs.core.async","t_cljs$core$async41738","cljs.core.async/t_cljs$core$async41738",101036131,null)], null)),new cljs.core.Symbol(null,"fn1","fn1",895834444,null),new cljs.core.Symbol(null,"meta41742","meta41742",1893707954,null)], null);
}));

(cljs.core.async.t_cljs$core$async41741.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async41741.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async41741");

(cljs.core.async.t_cljs$core$async41741.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async41741");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async41741.
 */
cljs.core.async.__GT_t_cljs$core$async41741 = (function cljs$core$async$map_LT__$___GT_t_cljs$core$async41741(f__$1,ch__$1,meta41739__$1,___$2,fn1__$1,meta41742){
return (new cljs.core.async.t_cljs$core$async41741(f__$1,ch__$1,meta41739__$1,___$2,fn1__$1,meta41742));
});

}

return (new cljs.core.async.t_cljs$core$async41741(self__.f,self__.ch,self__.meta41739,___$1,fn1,cljs.core.PersistentArrayMap.EMPTY));
})()
);
if(cljs.core.truth_((function (){var and__4115__auto__ = ret;
if(cljs.core.truth_(and__4115__auto__)){
return (!((cljs.core.deref(ret) == null)));
} else {
return and__4115__auto__;
}
})())){
return cljs.core.async.impl.channels.box((function (){var G__41749 = cljs.core.deref(ret);
return (self__.f.cljs$core$IFn$_invoke$arity$1 ? self__.f.cljs$core$IFn$_invoke$arity$1(G__41749) : self__.f.call(null,G__41749));
})());
} else {
return ret;
}
}));

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$WritePort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41738.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.put_BANG_(self__.ch,val,fn1);
}));

(cljs.core.async.t_cljs$core$async41738.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta41739","meta41739",-704159318,null)], null);
}));

(cljs.core.async.t_cljs$core$async41738.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async41738.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async41738");

(cljs.core.async.t_cljs$core$async41738.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async41738");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async41738.
 */
cljs.core.async.__GT_t_cljs$core$async41738 = (function cljs$core$async$map_LT__$___GT_t_cljs$core$async41738(f__$1,ch__$1,meta41739){
return (new cljs.core.async.t_cljs$core$async41738(f__$1,ch__$1,meta41739));
});

}

return (new cljs.core.async.t_cljs$core$async41738(f,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.map_GT_ = (function cljs$core$async$map_GT_(f,ch){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async41754 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async41754 = (function (f,ch,meta41755){
this.f = f;
this.ch = ch;
this.meta41755 = meta41755;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_41756,meta41755__$1){
var self__ = this;
var _41756__$1 = this;
return (new cljs.core.async.t_cljs$core$async41754(self__.f,self__.ch,meta41755__$1));
}));

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_41756){
var self__ = this;
var _41756__$1 = this;
return self__.meta41755;
}));

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$Channel$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_(self__.ch);
}));

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$ReadPort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.take_BANG_(self__.ch,fn1);
}));

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$WritePort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41754.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.put_BANG_(self__.ch,(self__.f.cljs$core$IFn$_invoke$arity$1 ? self__.f.cljs$core$IFn$_invoke$arity$1(val) : self__.f.call(null,val)),fn1);
}));

(cljs.core.async.t_cljs$core$async41754.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"f","f",43394975,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta41755","meta41755",-587306024,null)], null);
}));

(cljs.core.async.t_cljs$core$async41754.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async41754.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async41754");

(cljs.core.async.t_cljs$core$async41754.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async41754");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async41754.
 */
cljs.core.async.__GT_t_cljs$core$async41754 = (function cljs$core$async$map_GT__$___GT_t_cljs$core$async41754(f__$1,ch__$1,meta41755){
return (new cljs.core.async.t_cljs$core$async41754(f__$1,ch__$1,meta41755));
});

}

return (new cljs.core.async.t_cljs$core$async41754(f,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.filter_GT_ = (function cljs$core$async$filter_GT_(p,ch){
if((typeof cljs !== 'undefined') && (typeof cljs.core !== 'undefined') && (typeof cljs.core.async !== 'undefined') && (typeof cljs.core.async.t_cljs$core$async41763 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {cljs.core.async.impl.protocols.Channel}
 * @implements {cljs.core.async.impl.protocols.WritePort}
 * @implements {cljs.core.async.impl.protocols.ReadPort}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
cljs.core.async.t_cljs$core$async41763 = (function (p,ch,meta41764){
this.p = p;
this.ch = ch;
this.meta41764 = meta41764;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_41765,meta41764__$1){
var self__ = this;
var _41765__$1 = this;
return (new cljs.core.async.t_cljs$core$async41763(self__.p,self__.ch,meta41764__$1));
}));

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_41765){
var self__ = this;
var _41765__$1 = this;
return self__.meta41764;
}));

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$Channel$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.close_BANG_(self__.ch);
}));

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.closed_QMARK_(self__.ch);
}));

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$ReadPort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){
var self__ = this;
var ___$1 = this;
return cljs.core.async.impl.protocols.take_BANG_(self__.ch,fn1);
}));

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$WritePort$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.async.t_cljs$core$async41763.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){
var self__ = this;
var ___$1 = this;
if(cljs.core.truth_((self__.p.cljs$core$IFn$_invoke$arity$1 ? self__.p.cljs$core$IFn$_invoke$arity$1(val) : self__.p.call(null,val)))){
return cljs.core.async.impl.protocols.put_BANG_(self__.ch,val,fn1);
} else {
return cljs.core.async.impl.channels.box(cljs.core.not(cljs.core.async.impl.protocols.closed_QMARK_(self__.ch)));
}
}));

(cljs.core.async.t_cljs$core$async41763.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"p","p",1791580836,null),new cljs.core.Symbol(null,"ch","ch",1085813622,null),new cljs.core.Symbol(null,"meta41764","meta41764",-189860048,null)], null);
}));

(cljs.core.async.t_cljs$core$async41763.cljs$lang$type = true);

(cljs.core.async.t_cljs$core$async41763.cljs$lang$ctorStr = "cljs.core.async/t_cljs$core$async41763");

(cljs.core.async.t_cljs$core$async41763.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"cljs.core.async/t_cljs$core$async41763");
}));

/**
 * Positional factory function for cljs.core.async/t_cljs$core$async41763.
 */
cljs.core.async.__GT_t_cljs$core$async41763 = (function cljs$core$async$filter_GT__$___GT_t_cljs$core$async41763(p__$1,ch__$1,meta41764){
return (new cljs.core.async.t_cljs$core$async41763(p__$1,ch__$1,meta41764));
});

}

return (new cljs.core.async.t_cljs$core$async41763(p,ch,cljs.core.PersistentArrayMap.EMPTY));
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.remove_GT_ = (function cljs$core$async$remove_GT_(p,ch){
return cljs.core.async.filter_GT_(cljs.core.complement(p),ch);
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.filter_LT_ = (function cljs$core$async$filter_LT_(var_args){
var G__41780 = arguments.length;
switch (G__41780) {
case 2:
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3(p,ch,null);
}));

(cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3 = (function (p,ch,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43585 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41816){
var state_val_41817 = (state_41816[(1)]);
if((state_val_41817 === (7))){
var inst_41812 = (state_41816[(2)]);
var state_41816__$1 = state_41816;
var statearr_41819_43589 = state_41816__$1;
(statearr_41819_43589[(2)] = inst_41812);

(statearr_41819_43589[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (1))){
var state_41816__$1 = state_41816;
var statearr_41820_43590 = state_41816__$1;
(statearr_41820_43590[(2)] = null);

(statearr_41820_43590[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (4))){
var inst_41794 = (state_41816[(7)]);
var inst_41794__$1 = (state_41816[(2)]);
var inst_41795 = (inst_41794__$1 == null);
var state_41816__$1 = (function (){var statearr_41821 = state_41816;
(statearr_41821[(7)] = inst_41794__$1);

return statearr_41821;
})();
if(cljs.core.truth_(inst_41795)){
var statearr_41822_43601 = state_41816__$1;
(statearr_41822_43601[(1)] = (5));

} else {
var statearr_41823_43602 = state_41816__$1;
(statearr_41823_43602[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (6))){
var inst_41794 = (state_41816[(7)]);
var inst_41803 = (p.cljs$core$IFn$_invoke$arity$1 ? p.cljs$core$IFn$_invoke$arity$1(inst_41794) : p.call(null,inst_41794));
var state_41816__$1 = state_41816;
if(cljs.core.truth_(inst_41803)){
var statearr_41824_43625 = state_41816__$1;
(statearr_41824_43625[(1)] = (8));

} else {
var statearr_41825_43626 = state_41816__$1;
(statearr_41825_43626[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (3))){
var inst_41814 = (state_41816[(2)]);
var state_41816__$1 = state_41816;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41816__$1,inst_41814);
} else {
if((state_val_41817 === (2))){
var state_41816__$1 = state_41816;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_41816__$1,(4),ch);
} else {
if((state_val_41817 === (11))){
var inst_41806 = (state_41816[(2)]);
var state_41816__$1 = state_41816;
var statearr_41830_43645 = state_41816__$1;
(statearr_41830_43645[(2)] = inst_41806);

(statearr_41830_43645[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (9))){
var state_41816__$1 = state_41816;
var statearr_41831_43653 = state_41816__$1;
(statearr_41831_43653[(2)] = null);

(statearr_41831_43653[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (5))){
var inst_41801 = cljs.core.async.close_BANG_(out);
var state_41816__$1 = state_41816;
var statearr_41833_43659 = state_41816__$1;
(statearr_41833_43659[(2)] = inst_41801);

(statearr_41833_43659[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (10))){
var inst_41809 = (state_41816[(2)]);
var state_41816__$1 = (function (){var statearr_41834 = state_41816;
(statearr_41834[(8)] = inst_41809);

return statearr_41834;
})();
var statearr_41835_43668 = state_41816__$1;
(statearr_41835_43668[(2)] = null);

(statearr_41835_43668[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41817 === (8))){
var inst_41794 = (state_41816[(7)]);
var state_41816__$1 = state_41816;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41816__$1,(11),out,inst_41794);
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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_41836 = [null,null,null,null,null,null,null,null,null];
(statearr_41836[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_41836[(1)] = (1));

return statearr_41836;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_41816){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41816);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41837){var ex__39237__auto__ = e41837;
var statearr_41838_43688 = state_41816;
(statearr_41838_43688[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41816[(4)]))){
var statearr_41846_43689 = state_41816;
(statearr_41846_43689[(1)] = cljs.core.first((state_41816[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43690 = state_41816;
state_41816 = G__43690;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_41816){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_41816);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_41849 = f__39441__auto__();
(statearr_41849[(6)] = c__39440__auto___43585);

return statearr_41849;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.filter_LT_.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.remove_LT_ = (function cljs$core$async$remove_LT_(var_args){
var G__41863 = arguments.length;
switch (G__41863) {
case 2:
return cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$2 = (function (p,ch){
return cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$3(p,ch,null);
}));

(cljs.core.async.remove_LT_.cljs$core$IFn$_invoke$arity$3 = (function (p,ch,buf_or_n){
return cljs.core.async.filter_LT_.cljs$core$IFn$_invoke$arity$3(cljs.core.complement(p),ch,buf_or_n);
}));

(cljs.core.async.remove_LT_.cljs$lang$maxFixedArity = 3);

cljs.core.async.mapcat_STAR_ = (function cljs$core$async$mapcat_STAR_(f,in$,out){
var c__39440__auto__ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_41941){
var state_val_41942 = (state_41941[(1)]);
if((state_val_41942 === (7))){
var inst_41937 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
var statearr_41945_43694 = state_41941__$1;
(statearr_41945_43694[(2)] = inst_41937);

(statearr_41945_43694[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (20))){
var inst_41907 = (state_41941[(7)]);
var inst_41918 = (state_41941[(2)]);
var inst_41919 = cljs.core.next(inst_41907);
var inst_41891 = inst_41919;
var inst_41893 = null;
var inst_41895 = (0);
var inst_41896 = (0);
var state_41941__$1 = (function (){var statearr_41949 = state_41941;
(statearr_41949[(8)] = inst_41891);

(statearr_41949[(9)] = inst_41918);

(statearr_41949[(10)] = inst_41893);

(statearr_41949[(11)] = inst_41895);

(statearr_41949[(12)] = inst_41896);

return statearr_41949;
})();
var statearr_41950_43701 = state_41941__$1;
(statearr_41950_43701[(2)] = null);

(statearr_41950_43701[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (1))){
var state_41941__$1 = state_41941;
var statearr_41951_43706 = state_41941__$1;
(statearr_41951_43706[(2)] = null);

(statearr_41951_43706[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (4))){
var inst_41879 = (state_41941[(13)]);
var inst_41879__$1 = (state_41941[(2)]);
var inst_41880 = (inst_41879__$1 == null);
var state_41941__$1 = (function (){var statearr_41952 = state_41941;
(statearr_41952[(13)] = inst_41879__$1);

return statearr_41952;
})();
if(cljs.core.truth_(inst_41880)){
var statearr_41953_43711 = state_41941__$1;
(statearr_41953_43711[(1)] = (5));

} else {
var statearr_41954_43712 = state_41941__$1;
(statearr_41954_43712[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (15))){
var state_41941__$1 = state_41941;
var statearr_41958_43713 = state_41941__$1;
(statearr_41958_43713[(2)] = null);

(statearr_41958_43713[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (21))){
var state_41941__$1 = state_41941;
var statearr_41959_43715 = state_41941__$1;
(statearr_41959_43715[(2)] = null);

(statearr_41959_43715[(1)] = (23));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (13))){
var inst_41891 = (state_41941[(8)]);
var inst_41893 = (state_41941[(10)]);
var inst_41895 = (state_41941[(11)]);
var inst_41896 = (state_41941[(12)]);
var inst_41903 = (state_41941[(2)]);
var inst_41904 = (inst_41896 + (1));
var tmp41955 = inst_41891;
var tmp41956 = inst_41893;
var tmp41957 = inst_41895;
var inst_41891__$1 = tmp41955;
var inst_41893__$1 = tmp41956;
var inst_41895__$1 = tmp41957;
var inst_41896__$1 = inst_41904;
var state_41941__$1 = (function (){var statearr_41960 = state_41941;
(statearr_41960[(8)] = inst_41891__$1);

(statearr_41960[(10)] = inst_41893__$1);

(statearr_41960[(11)] = inst_41895__$1);

(statearr_41960[(14)] = inst_41903);

(statearr_41960[(12)] = inst_41896__$1);

return statearr_41960;
})();
var statearr_41965_43722 = state_41941__$1;
(statearr_41965_43722[(2)] = null);

(statearr_41965_43722[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (22))){
var state_41941__$1 = state_41941;
var statearr_41968_43723 = state_41941__$1;
(statearr_41968_43723[(2)] = null);

(statearr_41968_43723[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (6))){
var inst_41879 = (state_41941[(13)]);
var inst_41889 = (f.cljs$core$IFn$_invoke$arity$1 ? f.cljs$core$IFn$_invoke$arity$1(inst_41879) : f.call(null,inst_41879));
var inst_41890 = cljs.core.seq(inst_41889);
var inst_41891 = inst_41890;
var inst_41893 = null;
var inst_41895 = (0);
var inst_41896 = (0);
var state_41941__$1 = (function (){var statearr_41969 = state_41941;
(statearr_41969[(8)] = inst_41891);

(statearr_41969[(10)] = inst_41893);

(statearr_41969[(11)] = inst_41895);

(statearr_41969[(12)] = inst_41896);

return statearr_41969;
})();
var statearr_41970_43725 = state_41941__$1;
(statearr_41970_43725[(2)] = null);

(statearr_41970_43725[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (17))){
var inst_41907 = (state_41941[(7)]);
var inst_41911 = cljs.core.chunk_first(inst_41907);
var inst_41912 = cljs.core.chunk_rest(inst_41907);
var inst_41913 = cljs.core.count(inst_41911);
var inst_41891 = inst_41912;
var inst_41893 = inst_41911;
var inst_41895 = inst_41913;
var inst_41896 = (0);
var state_41941__$1 = (function (){var statearr_41972 = state_41941;
(statearr_41972[(8)] = inst_41891);

(statearr_41972[(10)] = inst_41893);

(statearr_41972[(11)] = inst_41895);

(statearr_41972[(12)] = inst_41896);

return statearr_41972;
})();
var statearr_41973_43726 = state_41941__$1;
(statearr_41973_43726[(2)] = null);

(statearr_41973_43726[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (3))){
var inst_41939 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
return cljs.core.async.impl.ioc_helpers.return_chan(state_41941__$1,inst_41939);
} else {
if((state_val_41942 === (12))){
var inst_41927 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
var statearr_41975_43728 = state_41941__$1;
(statearr_41975_43728[(2)] = inst_41927);

(statearr_41975_43728[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (2))){
var state_41941__$1 = state_41941;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_41941__$1,(4),in$);
} else {
if((state_val_41942 === (23))){
var inst_41935 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
var statearr_41978_43729 = state_41941__$1;
(statearr_41978_43729[(2)] = inst_41935);

(statearr_41978_43729[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (19))){
var inst_41922 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
var statearr_41979_43730 = state_41941__$1;
(statearr_41979_43730[(2)] = inst_41922);

(statearr_41979_43730[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (11))){
var inst_41891 = (state_41941[(8)]);
var inst_41907 = (state_41941[(7)]);
var inst_41907__$1 = cljs.core.seq(inst_41891);
var state_41941__$1 = (function (){var statearr_41980 = state_41941;
(statearr_41980[(7)] = inst_41907__$1);

return statearr_41980;
})();
if(inst_41907__$1){
var statearr_41981_43735 = state_41941__$1;
(statearr_41981_43735[(1)] = (14));

} else {
var statearr_41985_43736 = state_41941__$1;
(statearr_41985_43736[(1)] = (15));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (9))){
var inst_41929 = (state_41941[(2)]);
var inst_41930 = cljs.core.async.impl.protocols.closed_QMARK_(out);
var state_41941__$1 = (function (){var statearr_41986 = state_41941;
(statearr_41986[(15)] = inst_41929);

return statearr_41986;
})();
if(cljs.core.truth_(inst_41930)){
var statearr_41987_43737 = state_41941__$1;
(statearr_41987_43737[(1)] = (21));

} else {
var statearr_41988_43738 = state_41941__$1;
(statearr_41988_43738[(1)] = (22));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (5))){
var inst_41882 = cljs.core.async.close_BANG_(out);
var state_41941__$1 = state_41941;
var statearr_41989_43739 = state_41941__$1;
(statearr_41989_43739[(2)] = inst_41882);

(statearr_41989_43739[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (14))){
var inst_41907 = (state_41941[(7)]);
var inst_41909 = cljs.core.chunked_seq_QMARK_(inst_41907);
var state_41941__$1 = state_41941;
if(inst_41909){
var statearr_41990_43740 = state_41941__$1;
(statearr_41990_43740[(1)] = (17));

} else {
var statearr_41991_43741 = state_41941__$1;
(statearr_41991_43741[(1)] = (18));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (16))){
var inst_41925 = (state_41941[(2)]);
var state_41941__$1 = state_41941;
var statearr_41993_43742 = state_41941__$1;
(statearr_41993_43742[(2)] = inst_41925);

(statearr_41993_43742[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_41942 === (10))){
var inst_41893 = (state_41941[(10)]);
var inst_41896 = (state_41941[(12)]);
var inst_41901 = cljs.core._nth(inst_41893,inst_41896);
var state_41941__$1 = state_41941;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41941__$1,(13),out,inst_41901);
} else {
if((state_val_41942 === (18))){
var inst_41907 = (state_41941[(7)]);
var inst_41916 = cljs.core.first(inst_41907);
var state_41941__$1 = state_41941;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_41941__$1,(20),out,inst_41916);
} else {
if((state_val_41942 === (8))){
var inst_41895 = (state_41941[(11)]);
var inst_41896 = (state_41941[(12)]);
var inst_41898 = (inst_41896 < inst_41895);
var inst_41899 = inst_41898;
var state_41941__$1 = state_41941;
if(cljs.core.truth_(inst_41899)){
var statearr_41994_43746 = state_41941__$1;
(statearr_41994_43746[(1)] = (10));

} else {
var statearr_41995_43747 = state_41941__$1;
(statearr_41995_43747[(1)] = (11));

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
});
return (function() {
var cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__ = null;
var cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____0 = (function (){
var statearr_41996 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_41996[(0)] = cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__);

(statearr_41996[(1)] = (1));

return statearr_41996;
});
var cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____1 = (function (state_41941){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_41941);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e41997){var ex__39237__auto__ = e41997;
var statearr_41998_43748 = state_41941;
(statearr_41998_43748[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_41941[(4)]))){
var statearr_41999_43749 = state_41941;
(statearr_41999_43749[(1)] = cljs.core.first((state_41941[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43750 = state_41941;
state_41941 = G__43750;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__ = function(state_41941){
switch(arguments.length){
case 0:
return cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____1.call(this,state_41941);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____0;
cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$mapcat_STAR__$_state_machine__39234__auto____1;
return cljs$core$async$mapcat_STAR__$_state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_42000 = f__39441__auto__();
(statearr_42000[(6)] = c__39440__auto__);

return statearr_42000;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

return c__39440__auto__;
});
/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.mapcat_LT_ = (function cljs$core$async$mapcat_LT_(var_args){
var G__42002 = arguments.length;
switch (G__42002) {
case 2:
return cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$2 = (function (f,in$){
return cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$3(f,in$,null);
}));

(cljs.core.async.mapcat_LT_.cljs$core$IFn$_invoke$arity$3 = (function (f,in$,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
cljs.core.async.mapcat_STAR_(f,in$,out);

return out;
}));

(cljs.core.async.mapcat_LT_.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.mapcat_GT_ = (function cljs$core$async$mapcat_GT_(var_args){
var G__42005 = arguments.length;
switch (G__42005) {
case 2:
return cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$2 = (function (f,out){
return cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$3(f,out,null);
}));

(cljs.core.async.mapcat_GT_.cljs$core$IFn$_invoke$arity$3 = (function (f,out,buf_or_n){
var in$ = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
cljs.core.async.mapcat_STAR_(f,in$,out);

return in$;
}));

(cljs.core.async.mapcat_GT_.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.unique = (function cljs$core$async$unique(var_args){
var G__42010 = arguments.length;
switch (G__42010) {
case 1:
return cljs.core.async.unique.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return cljs.core.async.unique.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.unique.cljs$core$IFn$_invoke$arity$1 = (function (ch){
return cljs.core.async.unique.cljs$core$IFn$_invoke$arity$2(ch,null);
}));

(cljs.core.async.unique.cljs$core$IFn$_invoke$arity$2 = (function (ch,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43759 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_42041){
var state_val_42042 = (state_42041[(1)]);
if((state_val_42042 === (7))){
var inst_42036 = (state_42041[(2)]);
var state_42041__$1 = state_42041;
var statearr_42043_43760 = state_42041__$1;
(statearr_42043_43760[(2)] = inst_42036);

(statearr_42043_43760[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (1))){
var inst_42018 = null;
var state_42041__$1 = (function (){var statearr_42044 = state_42041;
(statearr_42044[(7)] = inst_42018);

return statearr_42044;
})();
var statearr_42045_43761 = state_42041__$1;
(statearr_42045_43761[(2)] = null);

(statearr_42045_43761[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (4))){
var inst_42021 = (state_42041[(8)]);
var inst_42021__$1 = (state_42041[(2)]);
var inst_42022 = (inst_42021__$1 == null);
var inst_42023 = cljs.core.not(inst_42022);
var state_42041__$1 = (function (){var statearr_42046 = state_42041;
(statearr_42046[(8)] = inst_42021__$1);

return statearr_42046;
})();
if(inst_42023){
var statearr_42047_43762 = state_42041__$1;
(statearr_42047_43762[(1)] = (5));

} else {
var statearr_42048_43764 = state_42041__$1;
(statearr_42048_43764[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (6))){
var state_42041__$1 = state_42041;
var statearr_42049_43765 = state_42041__$1;
(statearr_42049_43765[(2)] = null);

(statearr_42049_43765[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (3))){
var inst_42038 = (state_42041[(2)]);
var inst_42039 = cljs.core.async.close_BANG_(out);
var state_42041__$1 = (function (){var statearr_42050 = state_42041;
(statearr_42050[(9)] = inst_42038);

return statearr_42050;
})();
return cljs.core.async.impl.ioc_helpers.return_chan(state_42041__$1,inst_42039);
} else {
if((state_val_42042 === (2))){
var state_42041__$1 = state_42041;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_42041__$1,(4),ch);
} else {
if((state_val_42042 === (11))){
var inst_42021 = (state_42041[(8)]);
var inst_42030 = (state_42041[(2)]);
var inst_42018 = inst_42021;
var state_42041__$1 = (function (){var statearr_42051 = state_42041;
(statearr_42051[(10)] = inst_42030);

(statearr_42051[(7)] = inst_42018);

return statearr_42051;
})();
var statearr_42052_43768 = state_42041__$1;
(statearr_42052_43768[(2)] = null);

(statearr_42052_43768[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (9))){
var inst_42021 = (state_42041[(8)]);
var state_42041__$1 = state_42041;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_42041__$1,(11),out,inst_42021);
} else {
if((state_val_42042 === (5))){
var inst_42021 = (state_42041[(8)]);
var inst_42018 = (state_42041[(7)]);
var inst_42025 = cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(inst_42021,inst_42018);
var state_42041__$1 = state_42041;
if(inst_42025){
var statearr_42055_43769 = state_42041__$1;
(statearr_42055_43769[(1)] = (8));

} else {
var statearr_42056_43772 = state_42041__$1;
(statearr_42056_43772[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (10))){
var inst_42033 = (state_42041[(2)]);
var state_42041__$1 = state_42041;
var statearr_42057_43774 = state_42041__$1;
(statearr_42057_43774[(2)] = inst_42033);

(statearr_42057_43774[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42042 === (8))){
var inst_42018 = (state_42041[(7)]);
var tmp42053 = inst_42018;
var inst_42018__$1 = tmp42053;
var state_42041__$1 = (function (){var statearr_42058 = state_42041;
(statearr_42058[(7)] = inst_42018__$1);

return statearr_42058;
})();
var statearr_42059_43775 = state_42041__$1;
(statearr_42059_43775[(2)] = null);

(statearr_42059_43775[(1)] = (2));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_42060 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_42060[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_42060[(1)] = (1));

return statearr_42060;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_42041){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_42041);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e42061){var ex__39237__auto__ = e42061;
var statearr_42062_43778 = state_42041;
(statearr_42062_43778[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_42041[(4)]))){
var statearr_42064_43779 = state_42041;
(statearr_42064_43779[(1)] = cljs.core.first((state_42041[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43780 = state_42041;
state_42041 = G__43780;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_42041){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_42041);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_42066 = f__39441__auto__();
(statearr_42066[(6)] = c__39440__auto___43759);

return statearr_42066;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.unique.cljs$lang$maxFixedArity = 2);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.partition = (function cljs$core$async$partition(var_args){
var G__42068 = arguments.length;
switch (G__42068) {
case 2:
return cljs.core.async.partition.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.partition.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.partition.cljs$core$IFn$_invoke$arity$2 = (function (n,ch){
return cljs.core.async.partition.cljs$core$IFn$_invoke$arity$3(n,ch,null);
}));

(cljs.core.async.partition.cljs$core$IFn$_invoke$arity$3 = (function (n,ch,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43786 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_42112){
var state_val_42113 = (state_42112[(1)]);
if((state_val_42113 === (7))){
var inst_42108 = (state_42112[(2)]);
var state_42112__$1 = state_42112;
var statearr_42114_43787 = state_42112__$1;
(statearr_42114_43787[(2)] = inst_42108);

(statearr_42114_43787[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (1))){
var inst_42075 = (new Array(n));
var inst_42076 = inst_42075;
var inst_42077 = (0);
var state_42112__$1 = (function (){var statearr_42115 = state_42112;
(statearr_42115[(7)] = inst_42076);

(statearr_42115[(8)] = inst_42077);

return statearr_42115;
})();
var statearr_42116_43788 = state_42112__$1;
(statearr_42116_43788[(2)] = null);

(statearr_42116_43788[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (4))){
var inst_42080 = (state_42112[(9)]);
var inst_42080__$1 = (state_42112[(2)]);
var inst_42081 = (inst_42080__$1 == null);
var inst_42082 = cljs.core.not(inst_42081);
var state_42112__$1 = (function (){var statearr_42117 = state_42112;
(statearr_42117[(9)] = inst_42080__$1);

return statearr_42117;
})();
if(inst_42082){
var statearr_42122_43789 = state_42112__$1;
(statearr_42122_43789[(1)] = (5));

} else {
var statearr_42123_43790 = state_42112__$1;
(statearr_42123_43790[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (15))){
var inst_42102 = (state_42112[(2)]);
var state_42112__$1 = state_42112;
var statearr_42124_43791 = state_42112__$1;
(statearr_42124_43791[(2)] = inst_42102);

(statearr_42124_43791[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (13))){
var state_42112__$1 = state_42112;
var statearr_42127_43796 = state_42112__$1;
(statearr_42127_43796[(2)] = null);

(statearr_42127_43796[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (6))){
var inst_42077 = (state_42112[(8)]);
var inst_42098 = (inst_42077 > (0));
var state_42112__$1 = state_42112;
if(cljs.core.truth_(inst_42098)){
var statearr_42130_43800 = state_42112__$1;
(statearr_42130_43800[(1)] = (12));

} else {
var statearr_42131_43801 = state_42112__$1;
(statearr_42131_43801[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (3))){
var inst_42110 = (state_42112[(2)]);
var state_42112__$1 = state_42112;
return cljs.core.async.impl.ioc_helpers.return_chan(state_42112__$1,inst_42110);
} else {
if((state_val_42113 === (12))){
var inst_42076 = (state_42112[(7)]);
var inst_42100 = cljs.core.vec(inst_42076);
var state_42112__$1 = state_42112;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_42112__$1,(15),out,inst_42100);
} else {
if((state_val_42113 === (2))){
var state_42112__$1 = state_42112;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_42112__$1,(4),ch);
} else {
if((state_val_42113 === (11))){
var inst_42092 = (state_42112[(2)]);
var inst_42093 = (new Array(n));
var inst_42076 = inst_42093;
var inst_42077 = (0);
var state_42112__$1 = (function (){var statearr_42140 = state_42112;
(statearr_42140[(10)] = inst_42092);

(statearr_42140[(7)] = inst_42076);

(statearr_42140[(8)] = inst_42077);

return statearr_42140;
})();
var statearr_42141_43816 = state_42112__$1;
(statearr_42141_43816[(2)] = null);

(statearr_42141_43816[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (9))){
var inst_42076 = (state_42112[(7)]);
var inst_42090 = cljs.core.vec(inst_42076);
var state_42112__$1 = state_42112;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_42112__$1,(11),out,inst_42090);
} else {
if((state_val_42113 === (5))){
var inst_42076 = (state_42112[(7)]);
var inst_42080 = (state_42112[(9)]);
var inst_42085 = (state_42112[(11)]);
var inst_42077 = (state_42112[(8)]);
var inst_42084 = (inst_42076[inst_42077] = inst_42080);
var inst_42085__$1 = (inst_42077 + (1));
var inst_42086 = (inst_42085__$1 < n);
var state_42112__$1 = (function (){var statearr_42148 = state_42112;
(statearr_42148[(11)] = inst_42085__$1);

(statearr_42148[(12)] = inst_42084);

return statearr_42148;
})();
if(cljs.core.truth_(inst_42086)){
var statearr_42152_43829 = state_42112__$1;
(statearr_42152_43829[(1)] = (8));

} else {
var statearr_42153_43831 = state_42112__$1;
(statearr_42153_43831[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (14))){
var inst_42105 = (state_42112[(2)]);
var inst_42106 = cljs.core.async.close_BANG_(out);
var state_42112__$1 = (function (){var statearr_42155 = state_42112;
(statearr_42155[(13)] = inst_42105);

return statearr_42155;
})();
var statearr_42156_43836 = state_42112__$1;
(statearr_42156_43836[(2)] = inst_42106);

(statearr_42156_43836[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (10))){
var inst_42096 = (state_42112[(2)]);
var state_42112__$1 = state_42112;
var statearr_42159_43839 = state_42112__$1;
(statearr_42159_43839[(2)] = inst_42096);

(statearr_42159_43839[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42113 === (8))){
var inst_42076 = (state_42112[(7)]);
var inst_42085 = (state_42112[(11)]);
var tmp42154 = inst_42076;
var inst_42076__$1 = tmp42154;
var inst_42077 = inst_42085;
var state_42112__$1 = (function (){var statearr_42163 = state_42112;
(statearr_42163[(7)] = inst_42076__$1);

(statearr_42163[(8)] = inst_42077);

return statearr_42163;
})();
var statearr_42164_43847 = state_42112__$1;
(statearr_42164_43847[(2)] = null);

(statearr_42164_43847[(1)] = (2));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_42165 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_42165[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_42165[(1)] = (1));

return statearr_42165;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_42112){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_42112);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e42166){var ex__39237__auto__ = e42166;
var statearr_42167_43877 = state_42112;
(statearr_42167_43877[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_42112[(4)]))){
var statearr_42168_43882 = state_42112;
(statearr_42168_43882[(1)] = cljs.core.first((state_42112[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43896 = state_42112;
state_42112 = G__43896;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_42112){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_42112);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_42173 = f__39441__auto__();
(statearr_42173[(6)] = c__39440__auto___43786);

return statearr_42173;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.partition.cljs$lang$maxFixedArity = 3);

/**
 * Deprecated - this function will be removed. Use transducer instead
 */
cljs.core.async.partition_by = (function cljs$core$async$partition_by(var_args){
var G__42180 = arguments.length;
switch (G__42180) {
case 2:
return cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$2 = (function (f,ch){
return cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$3(f,ch,null);
}));

(cljs.core.async.partition_by.cljs$core$IFn$_invoke$arity$3 = (function (f,ch,buf_or_n){
var out = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1(buf_or_n);
var c__39440__auto___43929 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_42228){
var state_val_42229 = (state_42228[(1)]);
if((state_val_42229 === (7))){
var inst_42224 = (state_42228[(2)]);
var state_42228__$1 = state_42228;
var statearr_42232_43931 = state_42228__$1;
(statearr_42232_43931[(2)] = inst_42224);

(statearr_42232_43931[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (1))){
var inst_42183 = [];
var inst_42184 = inst_42183;
var inst_42185 = new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",-69252123);
var state_42228__$1 = (function (){var statearr_42233 = state_42228;
(statearr_42233[(7)] = inst_42184);

(statearr_42233[(8)] = inst_42185);

return statearr_42233;
})();
var statearr_42234_43935 = state_42228__$1;
(statearr_42234_43935[(2)] = null);

(statearr_42234_43935[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (4))){
var inst_42190 = (state_42228[(9)]);
var inst_42190__$1 = (state_42228[(2)]);
var inst_42191 = (inst_42190__$1 == null);
var inst_42192 = cljs.core.not(inst_42191);
var state_42228__$1 = (function (){var statearr_42235 = state_42228;
(statearr_42235[(9)] = inst_42190__$1);

return statearr_42235;
})();
if(inst_42192){
var statearr_42237_43937 = state_42228__$1;
(statearr_42237_43937[(1)] = (5));

} else {
var statearr_42238_43938 = state_42228__$1;
(statearr_42238_43938[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (15))){
var inst_42218 = (state_42228[(2)]);
var state_42228__$1 = state_42228;
var statearr_42239_43939 = state_42228__$1;
(statearr_42239_43939[(2)] = inst_42218);

(statearr_42239_43939[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (13))){
var state_42228__$1 = state_42228;
var statearr_42240_43940 = state_42228__$1;
(statearr_42240_43940[(2)] = null);

(statearr_42240_43940[(1)] = (14));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (6))){
var inst_42184 = (state_42228[(7)]);
var inst_42213 = inst_42184.length;
var inst_42214 = (inst_42213 > (0));
var state_42228__$1 = state_42228;
if(cljs.core.truth_(inst_42214)){
var statearr_42241_43941 = state_42228__$1;
(statearr_42241_43941[(1)] = (12));

} else {
var statearr_42242_43942 = state_42228__$1;
(statearr_42242_43942[(1)] = (13));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (3))){
var inst_42226 = (state_42228[(2)]);
var state_42228__$1 = state_42228;
return cljs.core.async.impl.ioc_helpers.return_chan(state_42228__$1,inst_42226);
} else {
if((state_val_42229 === (12))){
var inst_42184 = (state_42228[(7)]);
var inst_42216 = cljs.core.vec(inst_42184);
var state_42228__$1 = state_42228;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_42228__$1,(15),out,inst_42216);
} else {
if((state_val_42229 === (2))){
var state_42228__$1 = state_42228;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_42228__$1,(4),ch);
} else {
if((state_val_42229 === (11))){
var inst_42194 = (state_42228[(10)]);
var inst_42190 = (state_42228[(9)]);
var inst_42206 = (state_42228[(2)]);
var inst_42207 = [];
var inst_42208 = inst_42207.push(inst_42190);
var inst_42184 = inst_42207;
var inst_42185 = inst_42194;
var state_42228__$1 = (function (){var statearr_42252 = state_42228;
(statearr_42252[(7)] = inst_42184);

(statearr_42252[(8)] = inst_42185);

(statearr_42252[(11)] = inst_42206);

(statearr_42252[(12)] = inst_42208);

return statearr_42252;
})();
var statearr_42255_43948 = state_42228__$1;
(statearr_42255_43948[(2)] = null);

(statearr_42255_43948[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (9))){
var inst_42184 = (state_42228[(7)]);
var inst_42204 = cljs.core.vec(inst_42184);
var state_42228__$1 = state_42228;
return cljs.core.async.impl.ioc_helpers.put_BANG_(state_42228__$1,(11),out,inst_42204);
} else {
if((state_val_42229 === (5))){
var inst_42185 = (state_42228[(8)]);
var inst_42194 = (state_42228[(10)]);
var inst_42190 = (state_42228[(9)]);
var inst_42194__$1 = (f.cljs$core$IFn$_invoke$arity$1 ? f.cljs$core$IFn$_invoke$arity$1(inst_42190) : f.call(null,inst_42190));
var inst_42197 = cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(inst_42194__$1,inst_42185);
var inst_42198 = cljs.core.keyword_identical_QMARK_(inst_42185,new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",-69252123));
var inst_42199 = ((inst_42197) || (inst_42198));
var state_42228__$1 = (function (){var statearr_42262 = state_42228;
(statearr_42262[(10)] = inst_42194__$1);

return statearr_42262;
})();
if(cljs.core.truth_(inst_42199)){
var statearr_42263_43957 = state_42228__$1;
(statearr_42263_43957[(1)] = (8));

} else {
var statearr_42265_43962 = state_42228__$1;
(statearr_42265_43962[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (14))){
var inst_42221 = (state_42228[(2)]);
var inst_42222 = cljs.core.async.close_BANG_(out);
var state_42228__$1 = (function (){var statearr_42267 = state_42228;
(statearr_42267[(13)] = inst_42221);

return statearr_42267;
})();
var statearr_42268_43967 = state_42228__$1;
(statearr_42268_43967[(2)] = inst_42222);

(statearr_42268_43967[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (10))){
var inst_42211 = (state_42228[(2)]);
var state_42228__$1 = state_42228;
var statearr_42269_43976 = state_42228__$1;
(statearr_42269_43976[(2)] = inst_42211);

(statearr_42269_43976[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_42229 === (8))){
var inst_42184 = (state_42228[(7)]);
var inst_42194 = (state_42228[(10)]);
var inst_42190 = (state_42228[(9)]);
var inst_42201 = inst_42184.push(inst_42190);
var tmp42266 = inst_42184;
var inst_42184__$1 = tmp42266;
var inst_42185 = inst_42194;
var state_42228__$1 = (function (){var statearr_42270 = state_42228;
(statearr_42270[(14)] = inst_42201);

(statearr_42270[(7)] = inst_42184__$1);

(statearr_42270[(8)] = inst_42185);

return statearr_42270;
})();
var statearr_42271_43988 = state_42228__$1;
(statearr_42271_43988[(2)] = null);

(statearr_42271_43988[(1)] = (2));


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
});
return (function() {
var cljs$core$async$state_machine__39234__auto__ = null;
var cljs$core$async$state_machine__39234__auto____0 = (function (){
var statearr_42272 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_42272[(0)] = cljs$core$async$state_machine__39234__auto__);

(statearr_42272[(1)] = (1));

return statearr_42272;
});
var cljs$core$async$state_machine__39234__auto____1 = (function (state_42228){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_42228);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e42275){var ex__39237__auto__ = e42275;
var statearr_42276_43996 = state_42228;
(statearr_42276_43996[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_42228[(4)]))){
var statearr_42277_43997 = state_42228;
(statearr_42277_43997[(1)] = cljs.core.first((state_42228[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__43998 = state_42228;
state_42228 = G__43998;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
cljs$core$async$state_machine__39234__auto__ = function(state_42228){
switch(arguments.length){
case 0:
return cljs$core$async$state_machine__39234__auto____0.call(this);
case 1:
return cljs$core$async$state_machine__39234__auto____1.call(this,state_42228);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = cljs$core$async$state_machine__39234__auto____0;
cljs$core$async$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = cljs$core$async$state_machine__39234__auto____1;
return cljs$core$async$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_42278 = f__39441__auto__();
(statearr_42278[(6)] = c__39440__auto___43929);

return statearr_42278;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));


return out;
}));

(cljs.core.async.partition_by.cljs$lang$maxFixedArity = 3);


//# sourceMappingURL=cljs.core.async.js.map
