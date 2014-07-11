// Compiled by ClojureScript 0.0-2202
goog.provide('cljs.core.async');
goog.require('cljs.core');
goog.require('cljs.core.async.impl.channels');
goog.require('cljs.core.async.impl.dispatch');
goog.require('cljs.core.async.impl.ioc_helpers');
goog.require('cljs.core.async.impl.protocols');
goog.require('cljs.core.async.impl.channels');
goog.require('cljs.core.async.impl.buffers');
goog.require('cljs.core.async.impl.protocols');
goog.require('cljs.core.async.impl.timers');
goog.require('cljs.core.async.impl.dispatch');
goog.require('cljs.core.async.impl.ioc_helpers');
goog.require('cljs.core.async.impl.buffers');
goog.require('cljs.core.async.impl.timers');
cljs.core.async.fn_handler = (function fn_handler(f){if(typeof cljs.core.async.t15952 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15952 = (function (f,fn_handler,meta15953){
this.f = f;
this.fn_handler = fn_handler;
this.meta15953 = meta15953;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15952.cljs$lang$type = true;
cljs.core.async.t15952.cljs$lang$ctorStr = "cljs.core.async/t15952";
cljs.core.async.t15952.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15952");
});
cljs.core.async.t15952.prototype.cljs$core$async$impl$protocols$Handler$ = true;
cljs.core.async.t15952.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return true;
});
cljs.core.async.t15952.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return self__.f;
});
cljs.core.async.t15952.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_15954){var self__ = this;
var _15954__$1 = this;return self__.meta15953;
});
cljs.core.async.t15952.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_15954,meta15953__$1){var self__ = this;
var _15954__$1 = this;return (new cljs.core.async.t15952(self__.f,self__.fn_handler,meta15953__$1));
});
cljs.core.async.__GT_t15952 = (function __GT_t15952(f__$1,fn_handler__$1,meta15953){return (new cljs.core.async.t15952(f__$1,fn_handler__$1,meta15953));
});
}
return (new cljs.core.async.t15952(f,fn_handler,null));
});
/**
* Returns a fixed buffer of size n. When full, puts will block/park.
*/
cljs.core.async.buffer = (function buffer(n){return cljs.core.async.impl.buffers.fixed_buffer.call(null,n);
});
/**
* Returns a buffer of size n. When full, puts will complete but
* val will be dropped (no transfer).
*/
cljs.core.async.dropping_buffer = (function dropping_buffer(n){return cljs.core.async.impl.buffers.dropping_buffer.call(null,n);
});
/**
* Returns a buffer of size n. When full, puts will complete, and be
* buffered, but oldest elements in buffer will be dropped (not
* transferred).
*/
cljs.core.async.sliding_buffer = (function sliding_buffer(n){return cljs.core.async.impl.buffers.sliding_buffer.call(null,n);
});
/**
* Returns true if a channel created with buff will never block. That is to say,
* puts into this buffer will never cause the buffer to be full.
*/
cljs.core.async.unblocking_buffer_QMARK_ = (function unblocking_buffer_QMARK_(buff){var G__15956 = buff;if(G__15956)
{var bit__8524__auto__ = null;if(cljs.core.truth_((function (){var or__7874__auto__ = bit__8524__auto__;if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{return G__15956.cljs$core$async$impl$protocols$UnblockingBuffer$;
}
})()))
{return true;
} else
{if((!G__15956.cljs$lang$protocol_mask$partition$))
{return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.async.impl.protocols.UnblockingBuffer,G__15956);
} else
{return false;
}
}
} else
{return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.async.impl.protocols.UnblockingBuffer,G__15956);
}
});
/**
* Creates a channel with an optional buffer. If buf-or-n is a number,
* will create and use a fixed buffer of that size.
*/
cljs.core.async.chan = (function() {
var chan = null;
var chan__0 = (function (){return chan.call(null,null);
});
var chan__1 = (function (buf_or_n){var buf_or_n__$1 = ((cljs.core._EQ_.call(null,buf_or_n,0))?null:buf_or_n);return cljs.core.async.impl.channels.chan.call(null,((typeof buf_or_n__$1 === 'number')?cljs.core.async.buffer.call(null,buf_or_n__$1):buf_or_n__$1));
});
chan = function(buf_or_n){
switch(arguments.length){
case 0:
return chan__0.call(this);
case 1:
return chan__1.call(this,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
chan.cljs$core$IFn$_invoke$arity$0 = chan__0;
chan.cljs$core$IFn$_invoke$arity$1 = chan__1;
return chan;
})()
;
/**
* Returns a channel that will close after msecs
*/
cljs.core.async.timeout = (function timeout(msecs){return cljs.core.async.impl.timers.timeout.call(null,msecs);
});
/**
* takes a val from port. Must be called inside a (go ...) block. Will
* return nil if closed. Will park if nothing is available.
* Returns true unless port is already closed
*/
cljs.core.async._LT__BANG_ = (function _LT__BANG_(port){if(null)
{return null;
} else
{throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str("<! used not in (go ...) block"),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,null))].join('')));
}
});
/**
* Asynchronously takes a val from port, passing to fn1. Will pass nil
* if closed. If on-caller? (default true) is true, and value is
* immediately available, will call fn1 on calling thread.
* Returns nil.
*/
cljs.core.async.take_BANG_ = (function() {
var take_BANG_ = null;
var take_BANG___2 = (function (port,fn1){return take_BANG_.call(null,port,fn1,true);
});
var take_BANG___3 = (function (port,fn1,on_caller_QMARK_){var ret = cljs.core.async.impl.protocols.take_BANG_.call(null,port,cljs.core.async.fn_handler.call(null,fn1));if(cljs.core.truth_(ret))
{var val_15957 = cljs.core.deref.call(null,ret);if(cljs.core.truth_(on_caller_QMARK_))
{fn1.call(null,val_15957);
} else
{cljs.core.async.impl.dispatch.run.call(null,((function (val_15957,ret){
return (function (){return fn1.call(null,val_15957);
});})(val_15957,ret))
);
}
} else
{}
return null;
});
take_BANG_ = function(port,fn1,on_caller_QMARK_){
switch(arguments.length){
case 2:
return take_BANG___2.call(this,port,fn1);
case 3:
return take_BANG___3.call(this,port,fn1,on_caller_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
take_BANG_.cljs$core$IFn$_invoke$arity$2 = take_BANG___2;
take_BANG_.cljs$core$IFn$_invoke$arity$3 = take_BANG___3;
return take_BANG_;
})()
;
cljs.core.async.nop = (function nop(_){return null;
});
cljs.core.async.fhnop = cljs.core.async.fn_handler.call(null,cljs.core.async.nop);
/**
* puts a val into port. nil values are not allowed. Must be called
* inside a (go ...) block. Will park if no buffer space is available.
* Returns true unless port is already closed.
*/
cljs.core.async._GT__BANG_ = (function _GT__BANG_(port,val){if(null)
{return null;
} else
{throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str(">! used not in (go ...) block"),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,null))].join('')));
}
});
/**
* Asynchronously puts a val into port, calling fn0 (if supplied) when
* complete. nil values are not allowed. Will throw if closed. If
* on-caller? (default true) is true, and the put is immediately
* accepted, will call fn0 on calling thread.  Returns nil.
*/
cljs.core.async.put_BANG_ = (function() {
var put_BANG_ = null;
var put_BANG___2 = (function (port,val){var temp__4124__auto__ = cljs.core.async.impl.protocols.put_BANG_.call(null,port,val,cljs.core.async.fhnop);if(cljs.core.truth_(temp__4124__auto__))
{var ret = temp__4124__auto__;return cljs.core.deref.call(null,ret);
} else
{return true;
}
});
var put_BANG___3 = (function (port,val,fn1){return put_BANG_.call(null,port,val,fn1,true);
});
var put_BANG___4 = (function (port,val,fn1,on_caller_QMARK_){var temp__4124__auto__ = cljs.core.async.impl.protocols.put_BANG_.call(null,port,val,cljs.core.async.fn_handler.call(null,fn1));if(cljs.core.truth_(temp__4124__auto__))
{var retb = temp__4124__auto__;var ret = cljs.core.deref.call(null,retb);if(cljs.core.truth_(on_caller_QMARK_))
{fn1.call(null,ret);
} else
{cljs.core.async.impl.dispatch.run.call(null,((function (ret,retb,temp__4124__auto__){
return (function (){return fn1.call(null,ret);
});})(ret,retb,temp__4124__auto__))
);
}
return ret;
} else
{return true;
}
});
put_BANG_ = function(port,val,fn1,on_caller_QMARK_){
switch(arguments.length){
case 2:
return put_BANG___2.call(this,port,val);
case 3:
return put_BANG___3.call(this,port,val,fn1);
case 4:
return put_BANG___4.call(this,port,val,fn1,on_caller_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
put_BANG_.cljs$core$IFn$_invoke$arity$2 = put_BANG___2;
put_BANG_.cljs$core$IFn$_invoke$arity$3 = put_BANG___3;
put_BANG_.cljs$core$IFn$_invoke$arity$4 = put_BANG___4;
return put_BANG_;
})()
;
cljs.core.async.close_BANG_ = (function close_BANG_(port){return cljs.core.async.impl.protocols.close_BANG_.call(null,port);
});
cljs.core.async.random_array = (function random_array(n){var a = (new Array(n));var n__8722__auto___15958 = n;var x_15959 = 0;while(true){
if((x_15959 < n__8722__auto___15958))
{(a[x_15959] = 0);
{
var G__15960 = (x_15959 + 1);
x_15959 = G__15960;
continue;
}
} else
{}
break;
}
var i = 1;while(true){
if(cljs.core._EQ_.call(null,i,n))
{return a;
} else
{var j = cljs.core.rand_int.call(null,i);(a[i] = (a[j]));
(a[j] = i);
{
var G__15961 = (i + 1);
i = G__15961;
continue;
}
}
break;
}
});
cljs.core.async.alt_flag = (function alt_flag(){var flag = cljs.core.atom.call(null,true);if(typeof cljs.core.async.t15965 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15965 = (function (flag,alt_flag,meta15966){
this.flag = flag;
this.alt_flag = alt_flag;
this.meta15966 = meta15966;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15965.cljs$lang$type = true;
cljs.core.async.t15965.cljs$lang$ctorStr = "cljs.core.async/t15965";
cljs.core.async.t15965.cljs$lang$ctorPrWriter = ((function (flag){
return (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15965");
});})(flag))
;
cljs.core.async.t15965.prototype.cljs$core$async$impl$protocols$Handler$ = true;
cljs.core.async.t15965.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = ((function (flag){
return (function (_){var self__ = this;
var ___$1 = this;return cljs.core.deref.call(null,self__.flag);
});})(flag))
;
cljs.core.async.t15965.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = ((function (flag){
return (function (_){var self__ = this;
var ___$1 = this;cljs.core.reset_BANG_.call(null,self__.flag,null);
return true;
});})(flag))
;
cljs.core.async.t15965.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (flag){
return (function (_15967){var self__ = this;
var _15967__$1 = this;return self__.meta15966;
});})(flag))
;
cljs.core.async.t15965.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (flag){
return (function (_15967,meta15966__$1){var self__ = this;
var _15967__$1 = this;return (new cljs.core.async.t15965(self__.flag,self__.alt_flag,meta15966__$1));
});})(flag))
;
cljs.core.async.__GT_t15965 = ((function (flag){
return (function __GT_t15965(flag__$1,alt_flag__$1,meta15966){return (new cljs.core.async.t15965(flag__$1,alt_flag__$1,meta15966));
});})(flag))
;
}
return (new cljs.core.async.t15965(flag,alt_flag,null));
});
cljs.core.async.alt_handler = (function alt_handler(flag,cb){if(typeof cljs.core.async.t15971 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15971 = (function (cb,flag,alt_handler,meta15972){
this.cb = cb;
this.flag = flag;
this.alt_handler = alt_handler;
this.meta15972 = meta15972;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15971.cljs$lang$type = true;
cljs.core.async.t15971.cljs$lang$ctorStr = "cljs.core.async/t15971";
cljs.core.async.t15971.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15971");
});
cljs.core.async.t15971.prototype.cljs$core$async$impl$protocols$Handler$ = true;
cljs.core.async.t15971.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.active_QMARK_.call(null,self__.flag);
});
cljs.core.async.t15971.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = (function (_){var self__ = this;
var ___$1 = this;cljs.core.async.impl.protocols.commit.call(null,self__.flag);
return self__.cb;
});
cljs.core.async.t15971.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_15973){var self__ = this;
var _15973__$1 = this;return self__.meta15972;
});
cljs.core.async.t15971.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_15973,meta15972__$1){var self__ = this;
var _15973__$1 = this;return (new cljs.core.async.t15971(self__.cb,self__.flag,self__.alt_handler,meta15972__$1));
});
cljs.core.async.__GT_t15971 = (function __GT_t15971(cb__$1,flag__$1,alt_handler__$1,meta15972){return (new cljs.core.async.t15971(cb__$1,flag__$1,alt_handler__$1,meta15972));
});
}
return (new cljs.core.async.t15971(cb,flag,alt_handler,null));
});
/**
* returns derefable [val port] if immediate, nil if enqueued
*/
cljs.core.async.do_alts = (function do_alts(fret,ports,opts){var flag = cljs.core.async.alt_flag.call(null);var n = cljs.core.count.call(null,ports);var idxs = cljs.core.async.random_array.call(null,n);var priority = new cljs.core.Keyword(null,"priority","priority",4143410454).cljs$core$IFn$_invoke$arity$1(opts);var ret = (function (){var i = 0;while(true){
if((i < n))
{var idx = (cljs.core.truth_(priority)?i:(idxs[i]));var port = cljs.core.nth.call(null,ports,idx);var wport = ((cljs.core.vector_QMARK_.call(null,port))?port.call(null,0):null);var vbox = (cljs.core.truth_(wport)?(function (){var val = port.call(null,1);return cljs.core.async.impl.protocols.put_BANG_.call(null,wport,val,cljs.core.async.alt_handler.call(null,flag,((function (i,val,idx,port,wport,flag,n,idxs,priority){
return (function (p1__15974_SHARP_){return fret.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__15974_SHARP_,wport], null));
});})(i,val,idx,port,wport,flag,n,idxs,priority))
));
})():cljs.core.async.impl.protocols.take_BANG_.call(null,port,cljs.core.async.alt_handler.call(null,flag,((function (i,idx,port,wport,flag,n,idxs,priority){
return (function (p1__15975_SHARP_){return fret.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [p1__15975_SHARP_,port], null));
});})(i,idx,port,wport,flag,n,idxs,priority))
)));if(cljs.core.truth_(vbox))
{return cljs.core.async.impl.channels.box.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.deref.call(null,vbox),(function (){var or__7874__auto__ = wport;if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{return port;
}
})()], null));
} else
{{
var G__15976 = (i + 1);
i = G__15976;
continue;
}
}
} else
{return null;
}
break;
}
})();var or__7874__auto__ = ret;if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{if(cljs.core.contains_QMARK_.call(null,opts,new cljs.core.Keyword(null,"default","default",2558708147)))
{var temp__4126__auto__ = (function (){var and__7862__auto__ = cljs.core.async.impl.protocols.active_QMARK_.call(null,flag);if(cljs.core.truth_(and__7862__auto__))
{return cljs.core.async.impl.protocols.commit.call(null,flag);
} else
{return and__7862__auto__;
}
})();if(cljs.core.truth_(temp__4126__auto__))
{var got = temp__4126__auto__;return cljs.core.async.impl.channels.box.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"default","default",2558708147).cljs$core$IFn$_invoke$arity$1(opts),new cljs.core.Keyword(null,"default","default",2558708147)], null));
} else
{return null;
}
} else
{return null;
}
}
});
/**
* Completes at most one of several channel operations. Must be called
* inside a (go ...) block. ports is a vector of channel endpoints,
* which can be either a channel to take from or a vector of
* [channel-to-put-to val-to-put], in any combination. Takes will be
* made as if by <!, and puts will be made as if by >!. Unless
* the :priority option is true, if more than one port operation is
* ready a non-deterministic choice will be made. If no operation is
* ready and a :default value is supplied, [default-val :default] will
* be returned, otherwise alts! will park until the first operation to
* become ready completes. Returns [val port] of the completed
* operation, where val is the value taken for takes, and a
* boolean (true unless already closed, as per put!) for puts.
* 
* opts are passed as :key val ... Supported options:
* 
* :default val - the value to use if none of the operations are immediately ready
* :priority true - (default nil) when true, the operations will be tried in order.
* 
* Note: there is no guarantee that the port exps or val exprs will be
* used, nor in what order should they be, so they should not be
* depended upon for side effects.
* @param {...*} var_args
*/
cljs.core.async.alts_BANG_ = (function() { 
var alts_BANG___delegate = function (ports,p__15977){var map__15979 = p__15977;var map__15979__$1 = ((cljs.core.seq_QMARK_.call(null,map__15979))?cljs.core.apply.call(null,cljs.core.hash_map,map__15979):map__15979);var opts = map__15979__$1;if(null)
{return null;
} else
{throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str("alts! used not in (go ...) block"),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,null))].join('')));
}
};
var alts_BANG_ = function (ports,var_args){
var p__15977 = null;if (arguments.length > 1) {
  p__15977 = cljs.core.array_seq(Array.prototype.slice.call(arguments, 1),0);} 
return alts_BANG___delegate.call(this,ports,p__15977);};
alts_BANG_.cljs$lang$maxFixedArity = 1;
alts_BANG_.cljs$lang$applyTo = (function (arglist__15980){
var ports = cljs.core.first(arglist__15980);
var p__15977 = cljs.core.rest(arglist__15980);
return alts_BANG___delegate(ports,p__15977);
});
alts_BANG_.cljs$core$IFn$_invoke$arity$variadic = alts_BANG___delegate;
return alts_BANG_;
})()
;
/**
* Takes a function and a source channel, and returns a channel which
* contains the values produced by applying f to each value taken from
* the source channel
*/
cljs.core.async.map_LT_ = (function map_LT_(f,ch){if(typeof cljs.core.async.t15988 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15988 = (function (ch,f,map_LT_,meta15989){
this.ch = ch;
this.f = f;
this.map_LT_ = map_LT_;
this.meta15989 = meta15989;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15988.cljs$lang$type = true;
cljs.core.async.t15988.cljs$lang$ctorStr = "cljs.core.async/t15988";
cljs.core.async.t15988.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15988");
});
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$WritePort$ = true;
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,val,fn1);
});
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){var self__ = this;
var ___$1 = this;var ret = cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,(function (){if(typeof cljs.core.async.t15991 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15991 = (function (fn1,_,meta15989,ch,f,map_LT_,meta15992){
this.fn1 = fn1;
this._ = _;
this.meta15989 = meta15989;
this.ch = ch;
this.f = f;
this.map_LT_ = map_LT_;
this.meta15992 = meta15992;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15991.cljs$lang$type = true;
cljs.core.async.t15991.cljs$lang$ctorStr = "cljs.core.async/t15991";
cljs.core.async.t15991.cljs$lang$ctorPrWriter = ((function (___$1){
return (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15991");
});})(___$1))
;
cljs.core.async.t15991.prototype.cljs$core$async$impl$protocols$Handler$ = true;
cljs.core.async.t15991.prototype.cljs$core$async$impl$protocols$Handler$active_QMARK_$arity$1 = ((function (___$1){
return (function (___$3){var self__ = this;
var ___$4 = this;return cljs.core.async.impl.protocols.active_QMARK_.call(null,self__.fn1);
});})(___$1))
;
cljs.core.async.t15991.prototype.cljs$core$async$impl$protocols$Handler$lock_id$arity$1 = ((function (___$1){
return (function (___$3){var self__ = this;
var ___$4 = this;return cljs.core.async.impl.protocols.lock_id.call(null,self__.fn1);
});})(___$1))
;
cljs.core.async.t15991.prototype.cljs$core$async$impl$protocols$Handler$commit$arity$1 = ((function (___$1){
return (function (___$3){var self__ = this;
var ___$4 = this;var f1 = cljs.core.async.impl.protocols.commit.call(null,self__.fn1);return ((function (f1,___$4,___$1){
return (function (p1__15981_SHARP_){return f1.call(null,(((p1__15981_SHARP_ == null))?null:self__.f.call(null,p1__15981_SHARP_)));
});
;})(f1,___$4,___$1))
});})(___$1))
;
cljs.core.async.t15991.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (___$1){
return (function (_15993){var self__ = this;
var _15993__$1 = this;return self__.meta15992;
});})(___$1))
;
cljs.core.async.t15991.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (___$1){
return (function (_15993,meta15992__$1){var self__ = this;
var _15993__$1 = this;return (new cljs.core.async.t15991(self__.fn1,self__._,self__.meta15989,self__.ch,self__.f,self__.map_LT_,meta15992__$1));
});})(___$1))
;
cljs.core.async.__GT_t15991 = ((function (___$1){
return (function __GT_t15991(fn1__$1,___$2,meta15989__$1,ch__$2,f__$2,map_LT___$2,meta15992){return (new cljs.core.async.t15991(fn1__$1,___$2,meta15989__$1,ch__$2,f__$2,map_LT___$2,meta15992));
});})(___$1))
;
}
return (new cljs.core.async.t15991(fn1,___$1,self__.meta15989,self__.ch,self__.f,self__.map_LT_,null));
})());if(cljs.core.truth_((function (){var and__7862__auto__ = ret;if(cljs.core.truth_(and__7862__auto__))
{return !((cljs.core.deref.call(null,ret) == null));
} else
{return and__7862__auto__;
}
})()))
{return cljs.core.async.impl.channels.box.call(null,self__.f.call(null,cljs.core.deref.call(null,ret)));
} else
{return ret;
}
});
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$Channel$ = true;
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});
cljs.core.async.t15988.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch);
});
cljs.core.async.t15988.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_15990){var self__ = this;
var _15990__$1 = this;return self__.meta15989;
});
cljs.core.async.t15988.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_15990,meta15989__$1){var self__ = this;
var _15990__$1 = this;return (new cljs.core.async.t15988(self__.ch,self__.f,self__.map_LT_,meta15989__$1));
});
cljs.core.async.__GT_t15988 = (function __GT_t15988(ch__$1,f__$1,map_LT___$1,meta15989){return (new cljs.core.async.t15988(ch__$1,f__$1,map_LT___$1,meta15989));
});
}
return (new cljs.core.async.t15988(ch,f,map_LT_,null));
});
/**
* Takes a function and a target channel, and returns a channel which
* applies f to each value before supplying it to the target channel.
*/
cljs.core.async.map_GT_ = (function map_GT_(f,ch){if(typeof cljs.core.async.t15997 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t15997 = (function (ch,f,map_GT_,meta15998){
this.ch = ch;
this.f = f;
this.map_GT_ = map_GT_;
this.meta15998 = meta15998;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t15997.cljs$lang$type = true;
cljs.core.async.t15997.cljs$lang$ctorStr = "cljs.core.async/t15997";
cljs.core.async.t15997.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t15997");
});
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$WritePort$ = true;
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,self__.f.call(null,val),fn1);
});
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,fn1);
});
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$Channel$ = true;
cljs.core.async.t15997.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});
cljs.core.async.t15997.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_15999){var self__ = this;
var _15999__$1 = this;return self__.meta15998;
});
cljs.core.async.t15997.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_15999,meta15998__$1){var self__ = this;
var _15999__$1 = this;return (new cljs.core.async.t15997(self__.ch,self__.f,self__.map_GT_,meta15998__$1));
});
cljs.core.async.__GT_t15997 = (function __GT_t15997(ch__$1,f__$1,map_GT___$1,meta15998){return (new cljs.core.async.t15997(ch__$1,f__$1,map_GT___$1,meta15998));
});
}
return (new cljs.core.async.t15997(ch,f,map_GT_,null));
});
/**
* Takes a predicate and a target channel, and returns a channel which
* supplies only the values for which the predicate returns true to the
* target channel.
*/
cljs.core.async.filter_GT_ = (function filter_GT_(p,ch){if(typeof cljs.core.async.t16003 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t16003 = (function (ch,p,filter_GT_,meta16004){
this.ch = ch;
this.p = p;
this.filter_GT_ = filter_GT_;
this.meta16004 = meta16004;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t16003.cljs$lang$type = true;
cljs.core.async.t16003.cljs$lang$ctorStr = "cljs.core.async/t16003";
cljs.core.async.t16003.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t16003");
});
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$WritePort$ = true;
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$WritePort$put_BANG_$arity$3 = (function (_,val,fn1){var self__ = this;
var ___$1 = this;if(cljs.core.truth_(self__.p.call(null,val)))
{return cljs.core.async.impl.protocols.put_BANG_.call(null,self__.ch,val,fn1);
} else
{return cljs.core.async.impl.channels.box.call(null,cljs.core.not.call(null,cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch)));
}
});
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$ReadPort$ = true;
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$ReadPort$take_BANG_$arity$2 = (function (_,fn1){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.take_BANG_.call(null,self__.ch,fn1);
});
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$Channel$ = true;
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$Channel$close_BANG_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.close_BANG_.call(null,self__.ch);
});
cljs.core.async.t16003.prototype.cljs$core$async$impl$protocols$Channel$closed_QMARK_$arity$1 = (function (_){var self__ = this;
var ___$1 = this;return cljs.core.async.impl.protocols.closed_QMARK_.call(null,self__.ch);
});
cljs.core.async.t16003.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_16005){var self__ = this;
var _16005__$1 = this;return self__.meta16004;
});
cljs.core.async.t16003.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_16005,meta16004__$1){var self__ = this;
var _16005__$1 = this;return (new cljs.core.async.t16003(self__.ch,self__.p,self__.filter_GT_,meta16004__$1));
});
cljs.core.async.__GT_t16003 = (function __GT_t16003(ch__$1,p__$1,filter_GT___$1,meta16004){return (new cljs.core.async.t16003(ch__$1,p__$1,filter_GT___$1,meta16004));
});
}
return (new cljs.core.async.t16003(ch,p,filter_GT_,null));
});
/**
* Takes a predicate and a target channel, and returns a channel which
* supplies only the values for which the predicate returns false to the
* target channel.
*/
cljs.core.async.remove_GT_ = (function remove_GT_(p,ch){return cljs.core.async.filter_GT_.call(null,cljs.core.complement.call(null,p),ch);
});
/**
* Takes a predicate and a source channel, and returns a channel which
* contains only the values taken from the source channel for which the
* predicate returns true. The returned channel will be unbuffered by
* default, or a buf-or-n can be supplied. The channel will close
* when the source channel closes.
*/
cljs.core.async.filter_LT_ = (function() {
var filter_LT_ = null;
var filter_LT___2 = (function (p,ch){return filter_LT_.call(null,p,ch,null);
});
var filter_LT___3 = (function (p,ch,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___16088 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___16088,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___16088,out){
return (function (state_16067){var state_val_16068 = (state_16067[1]);if((state_val_16068 === 7))
{var inst_16063 = (state_16067[2]);var state_16067__$1 = state_16067;var statearr_16069_16089 = state_16067__$1;(statearr_16069_16089[2] = inst_16063);
(statearr_16069_16089[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 1))
{var state_16067__$1 = state_16067;var statearr_16070_16090 = state_16067__$1;(statearr_16070_16090[2] = null);
(statearr_16070_16090[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 4))
{var inst_16049 = (state_16067[7]);var inst_16049__$1 = (state_16067[2]);var inst_16050 = (inst_16049__$1 == null);var state_16067__$1 = (function (){var statearr_16071 = state_16067;(statearr_16071[7] = inst_16049__$1);
return statearr_16071;
})();if(cljs.core.truth_(inst_16050))
{var statearr_16072_16091 = state_16067__$1;(statearr_16072_16091[1] = 5);
} else
{var statearr_16073_16092 = state_16067__$1;(statearr_16073_16092[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 6))
{var inst_16049 = (state_16067[7]);var inst_16054 = p.call(null,inst_16049);var state_16067__$1 = state_16067;if(cljs.core.truth_(inst_16054))
{var statearr_16074_16093 = state_16067__$1;(statearr_16074_16093[1] = 8);
} else
{var statearr_16075_16094 = state_16067__$1;(statearr_16075_16094[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 3))
{var inst_16065 = (state_16067[2]);var state_16067__$1 = state_16067;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16067__$1,inst_16065);
} else
{if((state_val_16068 === 2))
{var state_16067__$1 = state_16067;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_16067__$1,4,ch);
} else
{if((state_val_16068 === 11))
{var inst_16057 = (state_16067[2]);var state_16067__$1 = state_16067;var statearr_16076_16095 = state_16067__$1;(statearr_16076_16095[2] = inst_16057);
(statearr_16076_16095[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 9))
{var state_16067__$1 = state_16067;var statearr_16077_16096 = state_16067__$1;(statearr_16077_16096[2] = null);
(statearr_16077_16096[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 5))
{var inst_16052 = cljs.core.async.close_BANG_.call(null,out);var state_16067__$1 = state_16067;var statearr_16078_16097 = state_16067__$1;(statearr_16078_16097[2] = inst_16052);
(statearr_16078_16097[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 10))
{var inst_16060 = (state_16067[2]);var state_16067__$1 = (function (){var statearr_16079 = state_16067;(statearr_16079[8] = inst_16060);
return statearr_16079;
})();var statearr_16080_16098 = state_16067__$1;(statearr_16080_16098[2] = null);
(statearr_16080_16098[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16068 === 8))
{var inst_16049 = (state_16067[7]);var state_16067__$1 = state_16067;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16067__$1,11,out,inst_16049);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___16088,out))
;return ((function (switch__11563__auto__,c__11627__auto___16088,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16084 = [null,null,null,null,null,null,null,null,null];(statearr_16084[0] = state_machine__11564__auto__);
(statearr_16084[1] = 1);
return statearr_16084;
});
var state_machine__11564__auto____1 = (function (state_16067){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16067);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16085){if((e16085 instanceof Object))
{var ex__11567__auto__ = e16085;var statearr_16086_16099 = state_16067;(statearr_16086_16099[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16067);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16085;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16100 = state_16067;
state_16067 = G__16100;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16067){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16067);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___16088,out))
})();var state__11629__auto__ = (function (){var statearr_16087 = f__11628__auto__.call(null);(statearr_16087[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___16088);
return statearr_16087;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___16088,out))
);
return out;
});
filter_LT_ = function(p,ch,buf_or_n){
switch(arguments.length){
case 2:
return filter_LT___2.call(this,p,ch);
case 3:
return filter_LT___3.call(this,p,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
filter_LT_.cljs$core$IFn$_invoke$arity$2 = filter_LT___2;
filter_LT_.cljs$core$IFn$_invoke$arity$3 = filter_LT___3;
return filter_LT_;
})()
;
/**
* Takes a predicate and a source channel, and returns a channel which
* contains only the values taken from the source channel for which the
* predicate returns false. The returned channel will be unbuffered by
* default, or a buf-or-n can be supplied. The channel will close
* when the source channel closes.
*/
cljs.core.async.remove_LT_ = (function() {
var remove_LT_ = null;
var remove_LT___2 = (function (p,ch){return remove_LT_.call(null,p,ch,null);
});
var remove_LT___3 = (function (p,ch,buf_or_n){return cljs.core.async.filter_LT_.call(null,cljs.core.complement.call(null,p),ch,buf_or_n);
});
remove_LT_ = function(p,ch,buf_or_n){
switch(arguments.length){
case 2:
return remove_LT___2.call(this,p,ch);
case 3:
return remove_LT___3.call(this,p,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
remove_LT_.cljs$core$IFn$_invoke$arity$2 = remove_LT___2;
remove_LT_.cljs$core$IFn$_invoke$arity$3 = remove_LT___3;
return remove_LT_;
})()
;
cljs.core.async.mapcat_STAR_ = (function mapcat_STAR_(f,in$,out){var c__11627__auto__ = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto__){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto__){
return (function (state_16266){var state_val_16267 = (state_16266[1]);if((state_val_16267 === 7))
{var inst_16262 = (state_16266[2]);var state_16266__$1 = state_16266;var statearr_16268_16309 = state_16266__$1;(statearr_16268_16309[2] = inst_16262);
(statearr_16268_16309[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 20))
{var inst_16232 = (state_16266[7]);var inst_16243 = (state_16266[2]);var inst_16244 = cljs.core.next.call(null,inst_16232);var inst_16218 = inst_16244;var inst_16219 = null;var inst_16220 = 0;var inst_16221 = 0;var state_16266__$1 = (function (){var statearr_16269 = state_16266;(statearr_16269[8] = inst_16219);
(statearr_16269[9] = inst_16221);
(statearr_16269[10] = inst_16220);
(statearr_16269[11] = inst_16243);
(statearr_16269[12] = inst_16218);
return statearr_16269;
})();var statearr_16270_16310 = state_16266__$1;(statearr_16270_16310[2] = null);
(statearr_16270_16310[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 1))
{var state_16266__$1 = state_16266;var statearr_16271_16311 = state_16266__$1;(statearr_16271_16311[2] = null);
(statearr_16271_16311[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 4))
{var inst_16207 = (state_16266[13]);var inst_16207__$1 = (state_16266[2]);var inst_16208 = (inst_16207__$1 == null);var state_16266__$1 = (function (){var statearr_16272 = state_16266;(statearr_16272[13] = inst_16207__$1);
return statearr_16272;
})();if(cljs.core.truth_(inst_16208))
{var statearr_16273_16312 = state_16266__$1;(statearr_16273_16312[1] = 5);
} else
{var statearr_16274_16313 = state_16266__$1;(statearr_16274_16313[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 15))
{var state_16266__$1 = state_16266;var statearr_16278_16314 = state_16266__$1;(statearr_16278_16314[2] = null);
(statearr_16278_16314[1] = 16);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 21))
{var state_16266__$1 = state_16266;var statearr_16279_16315 = state_16266__$1;(statearr_16279_16315[2] = null);
(statearr_16279_16315[1] = 23);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 13))
{var inst_16219 = (state_16266[8]);var inst_16221 = (state_16266[9]);var inst_16220 = (state_16266[10]);var inst_16218 = (state_16266[12]);var inst_16228 = (state_16266[2]);var inst_16229 = (inst_16221 + 1);var tmp16275 = inst_16219;var tmp16276 = inst_16220;var tmp16277 = inst_16218;var inst_16218__$1 = tmp16277;var inst_16219__$1 = tmp16275;var inst_16220__$1 = tmp16276;var inst_16221__$1 = inst_16229;var state_16266__$1 = (function (){var statearr_16280 = state_16266;(statearr_16280[8] = inst_16219__$1);
(statearr_16280[9] = inst_16221__$1);
(statearr_16280[10] = inst_16220__$1);
(statearr_16280[14] = inst_16228);
(statearr_16280[12] = inst_16218__$1);
return statearr_16280;
})();var statearr_16281_16316 = state_16266__$1;(statearr_16281_16316[2] = null);
(statearr_16281_16316[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 22))
{var state_16266__$1 = state_16266;var statearr_16282_16317 = state_16266__$1;(statearr_16282_16317[2] = null);
(statearr_16282_16317[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 6))
{var inst_16207 = (state_16266[13]);var inst_16216 = f.call(null,inst_16207);var inst_16217 = cljs.core.seq.call(null,inst_16216);var inst_16218 = inst_16217;var inst_16219 = null;var inst_16220 = 0;var inst_16221 = 0;var state_16266__$1 = (function (){var statearr_16283 = state_16266;(statearr_16283[8] = inst_16219);
(statearr_16283[9] = inst_16221);
(statearr_16283[10] = inst_16220);
(statearr_16283[12] = inst_16218);
return statearr_16283;
})();var statearr_16284_16318 = state_16266__$1;(statearr_16284_16318[2] = null);
(statearr_16284_16318[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 17))
{var inst_16232 = (state_16266[7]);var inst_16236 = cljs.core.chunk_first.call(null,inst_16232);var inst_16237 = cljs.core.chunk_rest.call(null,inst_16232);var inst_16238 = cljs.core.count.call(null,inst_16236);var inst_16218 = inst_16237;var inst_16219 = inst_16236;var inst_16220 = inst_16238;var inst_16221 = 0;var state_16266__$1 = (function (){var statearr_16285 = state_16266;(statearr_16285[8] = inst_16219);
(statearr_16285[9] = inst_16221);
(statearr_16285[10] = inst_16220);
(statearr_16285[12] = inst_16218);
return statearr_16285;
})();var statearr_16286_16319 = state_16266__$1;(statearr_16286_16319[2] = null);
(statearr_16286_16319[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 3))
{var inst_16264 = (state_16266[2]);var state_16266__$1 = state_16266;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16266__$1,inst_16264);
} else
{if((state_val_16267 === 12))
{var inst_16252 = (state_16266[2]);var state_16266__$1 = state_16266;var statearr_16287_16320 = state_16266__$1;(statearr_16287_16320[2] = inst_16252);
(statearr_16287_16320[1] = 9);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 2))
{var state_16266__$1 = state_16266;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_16266__$1,4,in$);
} else
{if((state_val_16267 === 23))
{var inst_16260 = (state_16266[2]);var state_16266__$1 = state_16266;var statearr_16288_16321 = state_16266__$1;(statearr_16288_16321[2] = inst_16260);
(statearr_16288_16321[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 19))
{var inst_16247 = (state_16266[2]);var state_16266__$1 = state_16266;var statearr_16289_16322 = state_16266__$1;(statearr_16289_16322[2] = inst_16247);
(statearr_16289_16322[1] = 16);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 11))
{var inst_16232 = (state_16266[7]);var inst_16218 = (state_16266[12]);var inst_16232__$1 = cljs.core.seq.call(null,inst_16218);var state_16266__$1 = (function (){var statearr_16290 = state_16266;(statearr_16290[7] = inst_16232__$1);
return statearr_16290;
})();if(inst_16232__$1)
{var statearr_16291_16323 = state_16266__$1;(statearr_16291_16323[1] = 14);
} else
{var statearr_16292_16324 = state_16266__$1;(statearr_16292_16324[1] = 15);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 9))
{var inst_16254 = (state_16266[2]);var inst_16255 = cljs.core.async.impl.protocols.closed_QMARK_.call(null,out);var state_16266__$1 = (function (){var statearr_16293 = state_16266;(statearr_16293[15] = inst_16254);
return statearr_16293;
})();if(cljs.core.truth_(inst_16255))
{var statearr_16294_16325 = state_16266__$1;(statearr_16294_16325[1] = 21);
} else
{var statearr_16295_16326 = state_16266__$1;(statearr_16295_16326[1] = 22);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 5))
{var inst_16210 = cljs.core.async.close_BANG_.call(null,out);var state_16266__$1 = state_16266;var statearr_16296_16327 = state_16266__$1;(statearr_16296_16327[2] = inst_16210);
(statearr_16296_16327[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 14))
{var inst_16232 = (state_16266[7]);var inst_16234 = cljs.core.chunked_seq_QMARK_.call(null,inst_16232);var state_16266__$1 = state_16266;if(inst_16234)
{var statearr_16297_16328 = state_16266__$1;(statearr_16297_16328[1] = 17);
} else
{var statearr_16298_16329 = state_16266__$1;(statearr_16298_16329[1] = 18);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 16))
{var inst_16250 = (state_16266[2]);var state_16266__$1 = state_16266;var statearr_16299_16330 = state_16266__$1;(statearr_16299_16330[2] = inst_16250);
(statearr_16299_16330[1] = 12);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16267 === 10))
{var inst_16219 = (state_16266[8]);var inst_16221 = (state_16266[9]);var inst_16226 = cljs.core._nth.call(null,inst_16219,inst_16221);var state_16266__$1 = state_16266;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16266__$1,13,out,inst_16226);
} else
{if((state_val_16267 === 18))
{var inst_16232 = (state_16266[7]);var inst_16241 = cljs.core.first.call(null,inst_16232);var state_16266__$1 = state_16266;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16266__$1,20,out,inst_16241);
} else
{if((state_val_16267 === 8))
{var inst_16221 = (state_16266[9]);var inst_16220 = (state_16266[10]);var inst_16223 = (inst_16221 < inst_16220);var inst_16224 = inst_16223;var state_16266__$1 = state_16266;if(cljs.core.truth_(inst_16224))
{var statearr_16300_16331 = state_16266__$1;(statearr_16300_16331[1] = 10);
} else
{var statearr_16301_16332 = state_16266__$1;(statearr_16301_16332[1] = 11);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto__))
;return ((function (switch__11563__auto__,c__11627__auto__){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16305 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_16305[0] = state_machine__11564__auto__);
(statearr_16305[1] = 1);
return statearr_16305;
});
var state_machine__11564__auto____1 = (function (state_16266){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16266);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16306){if((e16306 instanceof Object))
{var ex__11567__auto__ = e16306;var statearr_16307_16333 = state_16266;(statearr_16307_16333[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16266);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16306;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16334 = state_16266;
state_16266 = G__16334;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16266){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16266);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto__))
})();var state__11629__auto__ = (function (){var statearr_16308 = f__11628__auto__.call(null);(statearr_16308[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto__);
return statearr_16308;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto__))
);
return c__11627__auto__;
});
/**
* Takes a function and a source channel, and returns a channel which
* contains the values in each collection produced by applying f to
* each value taken from the source channel. f must return a
* collection.
* 
* The returned channel will be unbuffered by default, or a buf-or-n
* can be supplied. The channel will close when the source channel
* closes.
*/
cljs.core.async.mapcat_LT_ = (function() {
var mapcat_LT_ = null;
var mapcat_LT___2 = (function (f,in$){return mapcat_LT_.call(null,f,in$,null);
});
var mapcat_LT___3 = (function (f,in$,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);cljs.core.async.mapcat_STAR_.call(null,f,in$,out);
return out;
});
mapcat_LT_ = function(f,in$,buf_or_n){
switch(arguments.length){
case 2:
return mapcat_LT___2.call(this,f,in$);
case 3:
return mapcat_LT___3.call(this,f,in$,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
mapcat_LT_.cljs$core$IFn$_invoke$arity$2 = mapcat_LT___2;
mapcat_LT_.cljs$core$IFn$_invoke$arity$3 = mapcat_LT___3;
return mapcat_LT_;
})()
;
/**
* Takes a function and a target channel, and returns a channel which
* applies f to each value put, then supplies each element of the result
* to the target channel. f must return a collection.
* 
* The returned channel will be unbuffered by default, or a buf-or-n
* can be supplied. The target channel will be closed when the source
* channel closes.
*/
cljs.core.async.mapcat_GT_ = (function() {
var mapcat_GT_ = null;
var mapcat_GT___2 = (function (f,out){return mapcat_GT_.call(null,f,out,null);
});
var mapcat_GT___3 = (function (f,out,buf_or_n){var in$ = cljs.core.async.chan.call(null,buf_or_n);cljs.core.async.mapcat_STAR_.call(null,f,in$,out);
return in$;
});
mapcat_GT_ = function(f,out,buf_or_n){
switch(arguments.length){
case 2:
return mapcat_GT___2.call(this,f,out);
case 3:
return mapcat_GT___3.call(this,f,out,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
mapcat_GT_.cljs$core$IFn$_invoke$arity$2 = mapcat_GT___2;
mapcat_GT_.cljs$core$IFn$_invoke$arity$3 = mapcat_GT___3;
return mapcat_GT_;
})()
;
/**
* Takes elements from the from channel and supplies them to the to
* channel. By default, the to channel will be closed when the from
* channel closes, but can be determined by the close?  parameter. Will
* stop consuming the from channel if the to channel closes
*/
cljs.core.async.pipe = (function() {
var pipe = null;
var pipe__2 = (function (from,to){return pipe.call(null,from,to,true);
});
var pipe__3 = (function (from,to,close_QMARK_){var c__11627__auto___16429 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___16429){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___16429){
return (function (state_16405){var state_val_16406 = (state_16405[1]);if((state_val_16406 === 7))
{var inst_16401 = (state_16405[2]);var state_16405__$1 = state_16405;var statearr_16407_16430 = state_16405__$1;(statearr_16407_16430[2] = inst_16401);
(statearr_16407_16430[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 1))
{var state_16405__$1 = state_16405;var statearr_16408_16431 = state_16405__$1;(statearr_16408_16431[2] = null);
(statearr_16408_16431[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 4))
{var inst_16384 = (state_16405[7]);var inst_16384__$1 = (state_16405[2]);var inst_16385 = (inst_16384__$1 == null);var state_16405__$1 = (function (){var statearr_16409 = state_16405;(statearr_16409[7] = inst_16384__$1);
return statearr_16409;
})();if(cljs.core.truth_(inst_16385))
{var statearr_16410_16432 = state_16405__$1;(statearr_16410_16432[1] = 5);
} else
{var statearr_16411_16433 = state_16405__$1;(statearr_16411_16433[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 13))
{var state_16405__$1 = state_16405;var statearr_16412_16434 = state_16405__$1;(statearr_16412_16434[2] = null);
(statearr_16412_16434[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 6))
{var inst_16384 = (state_16405[7]);var state_16405__$1 = state_16405;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16405__$1,11,to,inst_16384);
} else
{if((state_val_16406 === 3))
{var inst_16403 = (state_16405[2]);var state_16405__$1 = state_16405;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16405__$1,inst_16403);
} else
{if((state_val_16406 === 12))
{var state_16405__$1 = state_16405;var statearr_16413_16435 = state_16405__$1;(statearr_16413_16435[2] = null);
(statearr_16413_16435[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 2))
{var state_16405__$1 = state_16405;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_16405__$1,4,from);
} else
{if((state_val_16406 === 11))
{var inst_16394 = (state_16405[2]);var state_16405__$1 = state_16405;if(cljs.core.truth_(inst_16394))
{var statearr_16414_16436 = state_16405__$1;(statearr_16414_16436[1] = 12);
} else
{var statearr_16415_16437 = state_16405__$1;(statearr_16415_16437[1] = 13);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 9))
{var state_16405__$1 = state_16405;var statearr_16416_16438 = state_16405__$1;(statearr_16416_16438[2] = null);
(statearr_16416_16438[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 5))
{var state_16405__$1 = state_16405;if(cljs.core.truth_(close_QMARK_))
{var statearr_16417_16439 = state_16405__$1;(statearr_16417_16439[1] = 8);
} else
{var statearr_16418_16440 = state_16405__$1;(statearr_16418_16440[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 14))
{var inst_16399 = (state_16405[2]);var state_16405__$1 = state_16405;var statearr_16419_16441 = state_16405__$1;(statearr_16419_16441[2] = inst_16399);
(statearr_16419_16441[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 10))
{var inst_16391 = (state_16405[2]);var state_16405__$1 = state_16405;var statearr_16420_16442 = state_16405__$1;(statearr_16420_16442[2] = inst_16391);
(statearr_16420_16442[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16406 === 8))
{var inst_16388 = cljs.core.async.close_BANG_.call(null,to);var state_16405__$1 = state_16405;var statearr_16421_16443 = state_16405__$1;(statearr_16421_16443[2] = inst_16388);
(statearr_16421_16443[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___16429))
;return ((function (switch__11563__auto__,c__11627__auto___16429){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16425 = [null,null,null,null,null,null,null,null];(statearr_16425[0] = state_machine__11564__auto__);
(statearr_16425[1] = 1);
return statearr_16425;
});
var state_machine__11564__auto____1 = (function (state_16405){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16405);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16426){if((e16426 instanceof Object))
{var ex__11567__auto__ = e16426;var statearr_16427_16444 = state_16405;(statearr_16427_16444[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16405);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16426;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16445 = state_16405;
state_16405 = G__16445;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16405){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16405);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___16429))
})();var state__11629__auto__ = (function (){var statearr_16428 = f__11628__auto__.call(null);(statearr_16428[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___16429);
return statearr_16428;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___16429))
);
return to;
});
pipe = function(from,to,close_QMARK_){
switch(arguments.length){
case 2:
return pipe__2.call(this,from,to);
case 3:
return pipe__3.call(this,from,to,close_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
pipe.cljs$core$IFn$_invoke$arity$2 = pipe__2;
pipe.cljs$core$IFn$_invoke$arity$3 = pipe__3;
return pipe;
})()
;
/**
* Takes a predicate and a source channel and returns a vector of two
* channels, the first of which will contain the values for which the
* predicate returned true, the second those for which it returned
* false.
* 
* The out channels will be unbuffered by default, or two buf-or-ns can
* be supplied. The channels will close after the source channel has
* closed.
*/
cljs.core.async.split = (function() {
var split = null;
var split__2 = (function (p,ch){return split.call(null,p,ch,null,null);
});
var split__4 = (function (p,ch,t_buf_or_n,f_buf_or_n){var tc = cljs.core.async.chan.call(null,t_buf_or_n);var fc = cljs.core.async.chan.call(null,f_buf_or_n);var c__11627__auto___16546 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___16546,tc,fc){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___16546,tc,fc){
return (function (state_16521){var state_val_16522 = (state_16521[1]);if((state_val_16522 === 7))
{var inst_16517 = (state_16521[2]);var state_16521__$1 = state_16521;var statearr_16523_16547 = state_16521__$1;(statearr_16523_16547[2] = inst_16517);
(statearr_16523_16547[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 1))
{var state_16521__$1 = state_16521;var statearr_16524_16548 = state_16521__$1;(statearr_16524_16548[2] = null);
(statearr_16524_16548[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 4))
{var inst_16498 = (state_16521[7]);var inst_16498__$1 = (state_16521[2]);var inst_16499 = (inst_16498__$1 == null);var state_16521__$1 = (function (){var statearr_16525 = state_16521;(statearr_16525[7] = inst_16498__$1);
return statearr_16525;
})();if(cljs.core.truth_(inst_16499))
{var statearr_16526_16549 = state_16521__$1;(statearr_16526_16549[1] = 5);
} else
{var statearr_16527_16550 = state_16521__$1;(statearr_16527_16550[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 13))
{var state_16521__$1 = state_16521;var statearr_16528_16551 = state_16521__$1;(statearr_16528_16551[2] = null);
(statearr_16528_16551[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 6))
{var inst_16498 = (state_16521[7]);var inst_16504 = p.call(null,inst_16498);var state_16521__$1 = state_16521;if(cljs.core.truth_(inst_16504))
{var statearr_16529_16552 = state_16521__$1;(statearr_16529_16552[1] = 9);
} else
{var statearr_16530_16553 = state_16521__$1;(statearr_16530_16553[1] = 10);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 3))
{var inst_16519 = (state_16521[2]);var state_16521__$1 = state_16521;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16521__$1,inst_16519);
} else
{if((state_val_16522 === 12))
{var state_16521__$1 = state_16521;var statearr_16531_16554 = state_16521__$1;(statearr_16531_16554[2] = null);
(statearr_16531_16554[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 2))
{var state_16521__$1 = state_16521;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_16521__$1,4,ch);
} else
{if((state_val_16522 === 11))
{var inst_16498 = (state_16521[7]);var inst_16508 = (state_16521[2]);var state_16521__$1 = state_16521;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16521__$1,8,inst_16508,inst_16498);
} else
{if((state_val_16522 === 9))
{var state_16521__$1 = state_16521;var statearr_16532_16555 = state_16521__$1;(statearr_16532_16555[2] = tc);
(statearr_16532_16555[1] = 11);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 5))
{var inst_16501 = cljs.core.async.close_BANG_.call(null,tc);var inst_16502 = cljs.core.async.close_BANG_.call(null,fc);var state_16521__$1 = (function (){var statearr_16533 = state_16521;(statearr_16533[8] = inst_16501);
return statearr_16533;
})();var statearr_16534_16556 = state_16521__$1;(statearr_16534_16556[2] = inst_16502);
(statearr_16534_16556[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 14))
{var inst_16515 = (state_16521[2]);var state_16521__$1 = state_16521;var statearr_16535_16557 = state_16521__$1;(statearr_16535_16557[2] = inst_16515);
(statearr_16535_16557[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 10))
{var state_16521__$1 = state_16521;var statearr_16536_16558 = state_16521__$1;(statearr_16536_16558[2] = fc);
(statearr_16536_16558[1] = 11);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16522 === 8))
{var inst_16510 = (state_16521[2]);var state_16521__$1 = state_16521;if(cljs.core.truth_(inst_16510))
{var statearr_16537_16559 = state_16521__$1;(statearr_16537_16559[1] = 12);
} else
{var statearr_16538_16560 = state_16521__$1;(statearr_16538_16560[1] = 13);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___16546,tc,fc))
;return ((function (switch__11563__auto__,c__11627__auto___16546,tc,fc){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16542 = [null,null,null,null,null,null,null,null,null];(statearr_16542[0] = state_machine__11564__auto__);
(statearr_16542[1] = 1);
return statearr_16542;
});
var state_machine__11564__auto____1 = (function (state_16521){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16521);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16543){if((e16543 instanceof Object))
{var ex__11567__auto__ = e16543;var statearr_16544_16561 = state_16521;(statearr_16544_16561[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16521);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16543;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16562 = state_16521;
state_16521 = G__16562;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16521){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16521);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___16546,tc,fc))
})();var state__11629__auto__ = (function (){var statearr_16545 = f__11628__auto__.call(null);(statearr_16545[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___16546);
return statearr_16545;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___16546,tc,fc))
);
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [tc,fc], null);
});
split = function(p,ch,t_buf_or_n,f_buf_or_n){
switch(arguments.length){
case 2:
return split__2.call(this,p,ch);
case 4:
return split__4.call(this,p,ch,t_buf_or_n,f_buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
split.cljs$core$IFn$_invoke$arity$2 = split__2;
split.cljs$core$IFn$_invoke$arity$4 = split__4;
return split;
})()
;
/**
* f should be a function of 2 arguments. Returns a channel containing
* the single result of applying f to init and the first item from the
* channel, then applying f to that result and the 2nd item, etc. If
* the channel closes without yielding items, returns init and f is not
* called. ch must close before reduce produces a result.
*/
cljs.core.async.reduce = (function reduce(f,init,ch){var c__11627__auto__ = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto__){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto__){
return (function (state_16609){var state_val_16610 = (state_16609[1]);if((state_val_16610 === 7))
{var inst_16605 = (state_16609[2]);var state_16609__$1 = state_16609;var statearr_16611_16627 = state_16609__$1;(statearr_16611_16627[2] = inst_16605);
(statearr_16611_16627[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16610 === 6))
{var inst_16595 = (state_16609[7]);var inst_16598 = (state_16609[8]);var inst_16602 = f.call(null,inst_16595,inst_16598);var inst_16595__$1 = inst_16602;var state_16609__$1 = (function (){var statearr_16612 = state_16609;(statearr_16612[7] = inst_16595__$1);
return statearr_16612;
})();var statearr_16613_16628 = state_16609__$1;(statearr_16613_16628[2] = null);
(statearr_16613_16628[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16610 === 5))
{var inst_16595 = (state_16609[7]);var state_16609__$1 = state_16609;var statearr_16614_16629 = state_16609__$1;(statearr_16614_16629[2] = inst_16595);
(statearr_16614_16629[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16610 === 4))
{var inst_16598 = (state_16609[8]);var inst_16598__$1 = (state_16609[2]);var inst_16599 = (inst_16598__$1 == null);var state_16609__$1 = (function (){var statearr_16615 = state_16609;(statearr_16615[8] = inst_16598__$1);
return statearr_16615;
})();if(cljs.core.truth_(inst_16599))
{var statearr_16616_16630 = state_16609__$1;(statearr_16616_16630[1] = 5);
} else
{var statearr_16617_16631 = state_16609__$1;(statearr_16617_16631[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16610 === 3))
{var inst_16607 = (state_16609[2]);var state_16609__$1 = state_16609;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16609__$1,inst_16607);
} else
{if((state_val_16610 === 2))
{var state_16609__$1 = state_16609;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_16609__$1,4,ch);
} else
{if((state_val_16610 === 1))
{var inst_16595 = init;var state_16609__$1 = (function (){var statearr_16618 = state_16609;(statearr_16618[7] = inst_16595);
return statearr_16618;
})();var statearr_16619_16632 = state_16609__$1;(statearr_16619_16632[2] = null);
(statearr_16619_16632[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
});})(c__11627__auto__))
;return ((function (switch__11563__auto__,c__11627__auto__){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16623 = [null,null,null,null,null,null,null,null,null];(statearr_16623[0] = state_machine__11564__auto__);
(statearr_16623[1] = 1);
return statearr_16623;
});
var state_machine__11564__auto____1 = (function (state_16609){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16609);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16624){if((e16624 instanceof Object))
{var ex__11567__auto__ = e16624;var statearr_16625_16633 = state_16609;(statearr_16625_16633[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16609);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16624;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16634 = state_16609;
state_16609 = G__16634;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16609){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16609);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto__))
})();var state__11629__auto__ = (function (){var statearr_16626 = f__11628__auto__.call(null);(statearr_16626[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto__);
return statearr_16626;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto__))
);
return c__11627__auto__;
});
/**
* Puts the contents of coll into the supplied channel.
* 
* By default the channel will be closed after the items are copied,
* but can be determined by the close? parameter.
* 
* Returns a channel which will close after the items are copied.
*/
cljs.core.async.onto_chan = (function() {
var onto_chan = null;
var onto_chan__2 = (function (ch,coll){return onto_chan.call(null,ch,coll,true);
});
var onto_chan__3 = (function (ch,coll,close_QMARK_){var c__11627__auto__ = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto__){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto__){
return (function (state_16708){var state_val_16709 = (state_16708[1]);if((state_val_16709 === 7))
{var inst_16690 = (state_16708[2]);var state_16708__$1 = state_16708;var statearr_16710_16733 = state_16708__$1;(statearr_16710_16733[2] = inst_16690);
(statearr_16710_16733[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 1))
{var inst_16684 = cljs.core.seq.call(null,coll);var inst_16685 = inst_16684;var state_16708__$1 = (function (){var statearr_16711 = state_16708;(statearr_16711[7] = inst_16685);
return statearr_16711;
})();var statearr_16712_16734 = state_16708__$1;(statearr_16712_16734[2] = null);
(statearr_16712_16734[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 4))
{var inst_16685 = (state_16708[7]);var inst_16688 = cljs.core.first.call(null,inst_16685);var state_16708__$1 = state_16708;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_16708__$1,7,ch,inst_16688);
} else
{if((state_val_16709 === 13))
{var inst_16702 = (state_16708[2]);var state_16708__$1 = state_16708;var statearr_16713_16735 = state_16708__$1;(statearr_16713_16735[2] = inst_16702);
(statearr_16713_16735[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 6))
{var inst_16693 = (state_16708[2]);var state_16708__$1 = state_16708;if(cljs.core.truth_(inst_16693))
{var statearr_16714_16736 = state_16708__$1;(statearr_16714_16736[1] = 8);
} else
{var statearr_16715_16737 = state_16708__$1;(statearr_16715_16737[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 3))
{var inst_16706 = (state_16708[2]);var state_16708__$1 = state_16708;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_16708__$1,inst_16706);
} else
{if((state_val_16709 === 12))
{var state_16708__$1 = state_16708;var statearr_16716_16738 = state_16708__$1;(statearr_16716_16738[2] = null);
(statearr_16716_16738[1] = 13);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 2))
{var inst_16685 = (state_16708[7]);var state_16708__$1 = state_16708;if(cljs.core.truth_(inst_16685))
{var statearr_16717_16739 = state_16708__$1;(statearr_16717_16739[1] = 4);
} else
{var statearr_16718_16740 = state_16708__$1;(statearr_16718_16740[1] = 5);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 11))
{var inst_16699 = cljs.core.async.close_BANG_.call(null,ch);var state_16708__$1 = state_16708;var statearr_16719_16741 = state_16708__$1;(statearr_16719_16741[2] = inst_16699);
(statearr_16719_16741[1] = 13);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 9))
{var state_16708__$1 = state_16708;if(cljs.core.truth_(close_QMARK_))
{var statearr_16720_16742 = state_16708__$1;(statearr_16720_16742[1] = 11);
} else
{var statearr_16721_16743 = state_16708__$1;(statearr_16721_16743[1] = 12);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 5))
{var inst_16685 = (state_16708[7]);var state_16708__$1 = state_16708;var statearr_16722_16744 = state_16708__$1;(statearr_16722_16744[2] = inst_16685);
(statearr_16722_16744[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 10))
{var inst_16704 = (state_16708[2]);var state_16708__$1 = state_16708;var statearr_16723_16745 = state_16708__$1;(statearr_16723_16745[2] = inst_16704);
(statearr_16723_16745[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_16709 === 8))
{var inst_16685 = (state_16708[7]);var inst_16695 = cljs.core.next.call(null,inst_16685);var inst_16685__$1 = inst_16695;var state_16708__$1 = (function (){var statearr_16724 = state_16708;(statearr_16724[7] = inst_16685__$1);
return statearr_16724;
})();var statearr_16725_16746 = state_16708__$1;(statearr_16725_16746[2] = null);
(statearr_16725_16746[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto__))
;return ((function (switch__11563__auto__,c__11627__auto__){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_16729 = [null,null,null,null,null,null,null,null];(statearr_16729[0] = state_machine__11564__auto__);
(statearr_16729[1] = 1);
return statearr_16729;
});
var state_machine__11564__auto____1 = (function (state_16708){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_16708);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e16730){if((e16730 instanceof Object))
{var ex__11567__auto__ = e16730;var statearr_16731_16747 = state_16708;(statearr_16731_16747[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_16708);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e16730;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__16748 = state_16708;
state_16708 = G__16748;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_16708){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_16708);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto__))
})();var state__11629__auto__ = (function (){var statearr_16732 = f__11628__auto__.call(null);(statearr_16732[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto__);
return statearr_16732;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto__))
);
return c__11627__auto__;
});
onto_chan = function(ch,coll,close_QMARK_){
switch(arguments.length){
case 2:
return onto_chan__2.call(this,ch,coll);
case 3:
return onto_chan__3.call(this,ch,coll,close_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
onto_chan.cljs$core$IFn$_invoke$arity$2 = onto_chan__2;
onto_chan.cljs$core$IFn$_invoke$arity$3 = onto_chan__3;
return onto_chan;
})()
;
/**
* Creates and returns a channel which contains the contents of coll,
* closing when exhausted.
*/
cljs.core.async.to_chan = (function to_chan(coll){var ch = cljs.core.async.chan.call(null,cljs.core.bounded_count.call(null,100,coll));cljs.core.async.onto_chan.call(null,ch,coll);
return ch;
});
cljs.core.async.Mux = (function (){var obj16750 = {};return obj16750;
})();
cljs.core.async.muxch_STAR_ = (function muxch_STAR_(_){if((function (){var and__7862__auto__ = _;if(and__7862__auto__)
{return _.cljs$core$async$Mux$muxch_STAR_$arity$1;
} else
{return and__7862__auto__;
}
})())
{return _.cljs$core$async$Mux$muxch_STAR_$arity$1(_);
} else
{var x__8501__auto__ = (((_ == null))?null:_);return (function (){var or__7874__auto__ = (cljs.core.async.muxch_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.muxch_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mux.muxch*",_);
}
}
})().call(null,_);
}
});
cljs.core.async.Mult = (function (){var obj16752 = {};return obj16752;
})();
cljs.core.async.tap_STAR_ = (function tap_STAR_(m,ch,close_QMARK_){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mult$tap_STAR_$arity$3;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mult$tap_STAR_$arity$3(m,ch,close_QMARK_);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.tap_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.tap_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mult.tap*",m);
}
}
})().call(null,m,ch,close_QMARK_);
}
});
cljs.core.async.untap_STAR_ = (function untap_STAR_(m,ch){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mult$untap_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mult$untap_STAR_$arity$2(m,ch);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.untap_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.untap_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mult.untap*",m);
}
}
})().call(null,m,ch);
}
});
cljs.core.async.untap_all_STAR_ = (function untap_all_STAR_(m){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mult$untap_all_STAR_$arity$1;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mult$untap_all_STAR_$arity$1(m);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.untap_all_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.untap_all_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mult.untap-all*",m);
}
}
})().call(null,m);
}
});
/**
* Creates and returns a mult(iple) of the supplied channel. Channels
* containing copies of the channel can be created with 'tap', and
* detached with 'untap'.
* 
* Each item is distributed to all taps in parallel and synchronously,
* i.e. each tap must accept before the next item is distributed. Use
* buffering/windowing to prevent slow taps from holding up the mult.
* 
* Items received when there are no taps get dropped.
* 
* If a tap puts to a closed channel, it will be removed from the mult.
*/
cljs.core.async.mult = (function mult(ch){var cs = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);var m = (function (){if(typeof cljs.core.async.t16974 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t16974 = (function (cs,ch,mult,meta16975){
this.cs = cs;
this.ch = ch;
this.mult = mult;
this.meta16975 = meta16975;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t16974.cljs$lang$type = true;
cljs.core.async.t16974.cljs$lang$ctorStr = "cljs.core.async/t16974";
cljs.core.async.t16974.cljs$lang$ctorPrWriter = ((function (cs){
return (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t16974");
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$async$Mult$ = true;
cljs.core.async.t16974.prototype.cljs$core$async$Mult$tap_STAR_$arity$3 = ((function (cs){
return (function (_,ch__$2,close_QMARK_){var self__ = this;
var ___$1 = this;cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.assoc,ch__$2,close_QMARK_);
return null;
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$async$Mult$untap_STAR_$arity$2 = ((function (cs){
return (function (_,ch__$2){var self__ = this;
var ___$1 = this;cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.dissoc,ch__$2);
return null;
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$async$Mult$untap_all_STAR_$arity$1 = ((function (cs){
return (function (_){var self__ = this;
var ___$1 = this;cljs.core.reset_BANG_.call(null,self__.cs,cljs.core.PersistentArrayMap.EMPTY);
return null;
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$async$Mux$ = true;
cljs.core.async.t16974.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (cs){
return (function (_){var self__ = this;
var ___$1 = this;return self__.ch;
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (cs){
return (function (_16976){var self__ = this;
var _16976__$1 = this;return self__.meta16975;
});})(cs))
;
cljs.core.async.t16974.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (cs){
return (function (_16976,meta16975__$1){var self__ = this;
var _16976__$1 = this;return (new cljs.core.async.t16974(self__.cs,self__.ch,self__.mult,meta16975__$1));
});})(cs))
;
cljs.core.async.__GT_t16974 = ((function (cs){
return (function __GT_t16974(cs__$1,ch__$1,mult__$1,meta16975){return (new cljs.core.async.t16974(cs__$1,ch__$1,mult__$1,meta16975));
});})(cs))
;
}
return (new cljs.core.async.t16974(cs,ch,mult,null));
})();var dchan = cljs.core.async.chan.call(null,1);var dctr = cljs.core.atom.call(null,null);var done = ((function (cs,m,dchan,dctr){
return (function (_){if((cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec) === 0))
{return cljs.core.async.put_BANG_.call(null,dchan,true);
} else
{return null;
}
});})(cs,m,dchan,dctr))
;var c__11627__auto___17195 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___17195,cs,m,dchan,dctr,done){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___17195,cs,m,dchan,dctr,done){
return (function (state_17107){var state_val_17108 = (state_17107[1]);if((state_val_17108 === 7))
{var inst_17103 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17109_17196 = state_17107__$1;(statearr_17109_17196[2] = inst_17103);
(statearr_17109_17196[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 20))
{var inst_17008 = (state_17107[7]);var inst_17018 = cljs.core.first.call(null,inst_17008);var inst_17019 = cljs.core.nth.call(null,inst_17018,0,null);var inst_17020 = cljs.core.nth.call(null,inst_17018,1,null);var state_17107__$1 = (function (){var statearr_17110 = state_17107;(statearr_17110[8] = inst_17019);
return statearr_17110;
})();if(cljs.core.truth_(inst_17020))
{var statearr_17111_17197 = state_17107__$1;(statearr_17111_17197[1] = 22);
} else
{var statearr_17112_17198 = state_17107__$1;(statearr_17112_17198[1] = 23);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 27))
{var inst_17055 = (state_17107[9]);var inst_16979 = (state_17107[10]);var inst_17050 = (state_17107[11]);var inst_17048 = (state_17107[12]);var inst_17055__$1 = cljs.core._nth.call(null,inst_17048,inst_17050);var inst_17056 = cljs.core.async.put_BANG_.call(null,inst_17055__$1,inst_16979,done);var state_17107__$1 = (function (){var statearr_17113 = state_17107;(statearr_17113[9] = inst_17055__$1);
return statearr_17113;
})();if(cljs.core.truth_(inst_17056))
{var statearr_17114_17199 = state_17107__$1;(statearr_17114_17199[1] = 30);
} else
{var statearr_17115_17200 = state_17107__$1;(statearr_17115_17200[1] = 31);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 1))
{var state_17107__$1 = state_17107;var statearr_17116_17201 = state_17107__$1;(statearr_17116_17201[2] = null);
(statearr_17116_17201[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 24))
{var inst_17008 = (state_17107[7]);var inst_17025 = (state_17107[2]);var inst_17026 = cljs.core.next.call(null,inst_17008);var inst_16988 = inst_17026;var inst_16989 = null;var inst_16990 = 0;var inst_16991 = 0;var state_17107__$1 = (function (){var statearr_17117 = state_17107;(statearr_17117[13] = inst_16989);
(statearr_17117[14] = inst_16988);
(statearr_17117[15] = inst_17025);
(statearr_17117[16] = inst_16990);
(statearr_17117[17] = inst_16991);
return statearr_17117;
})();var statearr_17118_17202 = state_17107__$1;(statearr_17118_17202[2] = null);
(statearr_17118_17202[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 39))
{var state_17107__$1 = state_17107;var statearr_17122_17203 = state_17107__$1;(statearr_17122_17203[2] = null);
(statearr_17122_17203[1] = 41);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 4))
{var inst_16979 = (state_17107[10]);var inst_16979__$1 = (state_17107[2]);var inst_16980 = (inst_16979__$1 == null);var state_17107__$1 = (function (){var statearr_17123 = state_17107;(statearr_17123[10] = inst_16979__$1);
return statearr_17123;
})();if(cljs.core.truth_(inst_16980))
{var statearr_17124_17204 = state_17107__$1;(statearr_17124_17204[1] = 5);
} else
{var statearr_17125_17205 = state_17107__$1;(statearr_17125_17205[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 15))
{var inst_16989 = (state_17107[13]);var inst_16988 = (state_17107[14]);var inst_16990 = (state_17107[16]);var inst_16991 = (state_17107[17]);var inst_17004 = (state_17107[2]);var inst_17005 = (inst_16991 + 1);var tmp17119 = inst_16989;var tmp17120 = inst_16988;var tmp17121 = inst_16990;var inst_16988__$1 = tmp17120;var inst_16989__$1 = tmp17119;var inst_16990__$1 = tmp17121;var inst_16991__$1 = inst_17005;var state_17107__$1 = (function (){var statearr_17126 = state_17107;(statearr_17126[13] = inst_16989__$1);
(statearr_17126[18] = inst_17004);
(statearr_17126[14] = inst_16988__$1);
(statearr_17126[16] = inst_16990__$1);
(statearr_17126[17] = inst_16991__$1);
return statearr_17126;
})();var statearr_17127_17206 = state_17107__$1;(statearr_17127_17206[2] = null);
(statearr_17127_17206[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 21))
{var inst_17029 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17131_17207 = state_17107__$1;(statearr_17131_17207[2] = inst_17029);
(statearr_17131_17207[1] = 18);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 31))
{var inst_17055 = (state_17107[9]);var inst_17059 = cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec);var inst_17060 = cljs.core.async.untap_STAR_.call(null,m,inst_17055);var state_17107__$1 = (function (){var statearr_17132 = state_17107;(statearr_17132[19] = inst_17059);
return statearr_17132;
})();var statearr_17133_17208 = state_17107__$1;(statearr_17133_17208[2] = inst_17060);
(statearr_17133_17208[1] = 32);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 32))
{var inst_17049 = (state_17107[20]);var inst_17047 = (state_17107[21]);var inst_17050 = (state_17107[11]);var inst_17048 = (state_17107[12]);var inst_17062 = (state_17107[2]);var inst_17063 = (inst_17050 + 1);var tmp17128 = inst_17049;var tmp17129 = inst_17047;var tmp17130 = inst_17048;var inst_17047__$1 = tmp17129;var inst_17048__$1 = tmp17130;var inst_17049__$1 = tmp17128;var inst_17050__$1 = inst_17063;var state_17107__$1 = (function (){var statearr_17134 = state_17107;(statearr_17134[22] = inst_17062);
(statearr_17134[20] = inst_17049__$1);
(statearr_17134[21] = inst_17047__$1);
(statearr_17134[11] = inst_17050__$1);
(statearr_17134[12] = inst_17048__$1);
return statearr_17134;
})();var statearr_17135_17209 = state_17107__$1;(statearr_17135_17209[2] = null);
(statearr_17135_17209[1] = 25);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 40))
{var inst_17075 = (state_17107[23]);var inst_17079 = cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec);var inst_17080 = cljs.core.async.untap_STAR_.call(null,m,inst_17075);var state_17107__$1 = (function (){var statearr_17136 = state_17107;(statearr_17136[24] = inst_17079);
return statearr_17136;
})();var statearr_17137_17210 = state_17107__$1;(statearr_17137_17210[2] = inst_17080);
(statearr_17137_17210[1] = 41);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 33))
{var inst_17066 = (state_17107[25]);var inst_17068 = cljs.core.chunked_seq_QMARK_.call(null,inst_17066);var state_17107__$1 = state_17107;if(inst_17068)
{var statearr_17138_17211 = state_17107__$1;(statearr_17138_17211[1] = 36);
} else
{var statearr_17139_17212 = state_17107__$1;(statearr_17139_17212[1] = 37);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 13))
{var inst_16998 = (state_17107[26]);var inst_17001 = cljs.core.async.close_BANG_.call(null,inst_16998);var state_17107__$1 = state_17107;var statearr_17140_17213 = state_17107__$1;(statearr_17140_17213[2] = inst_17001);
(statearr_17140_17213[1] = 15);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 22))
{var inst_17019 = (state_17107[8]);var inst_17022 = cljs.core.async.close_BANG_.call(null,inst_17019);var state_17107__$1 = state_17107;var statearr_17141_17214 = state_17107__$1;(statearr_17141_17214[2] = inst_17022);
(statearr_17141_17214[1] = 24);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 36))
{var inst_17066 = (state_17107[25]);var inst_17070 = cljs.core.chunk_first.call(null,inst_17066);var inst_17071 = cljs.core.chunk_rest.call(null,inst_17066);var inst_17072 = cljs.core.count.call(null,inst_17070);var inst_17047 = inst_17071;var inst_17048 = inst_17070;var inst_17049 = inst_17072;var inst_17050 = 0;var state_17107__$1 = (function (){var statearr_17142 = state_17107;(statearr_17142[20] = inst_17049);
(statearr_17142[21] = inst_17047);
(statearr_17142[11] = inst_17050);
(statearr_17142[12] = inst_17048);
return statearr_17142;
})();var statearr_17143_17215 = state_17107__$1;(statearr_17143_17215[2] = null);
(statearr_17143_17215[1] = 25);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 41))
{var inst_17066 = (state_17107[25]);var inst_17082 = (state_17107[2]);var inst_17083 = cljs.core.next.call(null,inst_17066);var inst_17047 = inst_17083;var inst_17048 = null;var inst_17049 = 0;var inst_17050 = 0;var state_17107__$1 = (function (){var statearr_17144 = state_17107;(statearr_17144[27] = inst_17082);
(statearr_17144[20] = inst_17049);
(statearr_17144[21] = inst_17047);
(statearr_17144[11] = inst_17050);
(statearr_17144[12] = inst_17048);
return statearr_17144;
})();var statearr_17145_17216 = state_17107__$1;(statearr_17145_17216[2] = null);
(statearr_17145_17216[1] = 25);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 43))
{var state_17107__$1 = state_17107;var statearr_17146_17217 = state_17107__$1;(statearr_17146_17217[2] = null);
(statearr_17146_17217[1] = 44);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 29))
{var inst_17091 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17147_17218 = state_17107__$1;(statearr_17147_17218[2] = inst_17091);
(statearr_17147_17218[1] = 26);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 44))
{var inst_17100 = (state_17107[2]);var state_17107__$1 = (function (){var statearr_17148 = state_17107;(statearr_17148[28] = inst_17100);
return statearr_17148;
})();var statearr_17149_17219 = state_17107__$1;(statearr_17149_17219[2] = null);
(statearr_17149_17219[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 6))
{var inst_17039 = (state_17107[29]);var inst_17038 = cljs.core.deref.call(null,cs);var inst_17039__$1 = cljs.core.keys.call(null,inst_17038);var inst_17040 = cljs.core.count.call(null,inst_17039__$1);var inst_17041 = cljs.core.reset_BANG_.call(null,dctr,inst_17040);var inst_17046 = cljs.core.seq.call(null,inst_17039__$1);var inst_17047 = inst_17046;var inst_17048 = null;var inst_17049 = 0;var inst_17050 = 0;var state_17107__$1 = (function (){var statearr_17150 = state_17107;(statearr_17150[30] = inst_17041);
(statearr_17150[20] = inst_17049);
(statearr_17150[29] = inst_17039__$1);
(statearr_17150[21] = inst_17047);
(statearr_17150[11] = inst_17050);
(statearr_17150[12] = inst_17048);
return statearr_17150;
})();var statearr_17151_17220 = state_17107__$1;(statearr_17151_17220[2] = null);
(statearr_17151_17220[1] = 25);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 28))
{var inst_17066 = (state_17107[25]);var inst_17047 = (state_17107[21]);var inst_17066__$1 = cljs.core.seq.call(null,inst_17047);var state_17107__$1 = (function (){var statearr_17152 = state_17107;(statearr_17152[25] = inst_17066__$1);
return statearr_17152;
})();if(inst_17066__$1)
{var statearr_17153_17221 = state_17107__$1;(statearr_17153_17221[1] = 33);
} else
{var statearr_17154_17222 = state_17107__$1;(statearr_17154_17222[1] = 34);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 25))
{var inst_17049 = (state_17107[20]);var inst_17050 = (state_17107[11]);var inst_17052 = (inst_17050 < inst_17049);var inst_17053 = inst_17052;var state_17107__$1 = state_17107;if(cljs.core.truth_(inst_17053))
{var statearr_17155_17223 = state_17107__$1;(statearr_17155_17223[1] = 27);
} else
{var statearr_17156_17224 = state_17107__$1;(statearr_17156_17224[1] = 28);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 34))
{var state_17107__$1 = state_17107;var statearr_17157_17225 = state_17107__$1;(statearr_17157_17225[2] = null);
(statearr_17157_17225[1] = 35);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 17))
{var state_17107__$1 = state_17107;var statearr_17158_17226 = state_17107__$1;(statearr_17158_17226[2] = null);
(statearr_17158_17226[1] = 18);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 3))
{var inst_17105 = (state_17107[2]);var state_17107__$1 = state_17107;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_17107__$1,inst_17105);
} else
{if((state_val_17108 === 12))
{var inst_17034 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17159_17227 = state_17107__$1;(statearr_17159_17227[2] = inst_17034);
(statearr_17159_17227[1] = 9);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 2))
{var state_17107__$1 = state_17107;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_17107__$1,4,ch);
} else
{if((state_val_17108 === 23))
{var state_17107__$1 = state_17107;var statearr_17160_17228 = state_17107__$1;(statearr_17160_17228[2] = null);
(statearr_17160_17228[1] = 24);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 35))
{var inst_17089 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17161_17229 = state_17107__$1;(statearr_17161_17229[2] = inst_17089);
(statearr_17161_17229[1] = 29);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 19))
{var inst_17008 = (state_17107[7]);var inst_17012 = cljs.core.chunk_first.call(null,inst_17008);var inst_17013 = cljs.core.chunk_rest.call(null,inst_17008);var inst_17014 = cljs.core.count.call(null,inst_17012);var inst_16988 = inst_17013;var inst_16989 = inst_17012;var inst_16990 = inst_17014;var inst_16991 = 0;var state_17107__$1 = (function (){var statearr_17162 = state_17107;(statearr_17162[13] = inst_16989);
(statearr_17162[14] = inst_16988);
(statearr_17162[16] = inst_16990);
(statearr_17162[17] = inst_16991);
return statearr_17162;
})();var statearr_17163_17230 = state_17107__$1;(statearr_17163_17230[2] = null);
(statearr_17163_17230[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 11))
{var inst_17008 = (state_17107[7]);var inst_16988 = (state_17107[14]);var inst_17008__$1 = cljs.core.seq.call(null,inst_16988);var state_17107__$1 = (function (){var statearr_17164 = state_17107;(statearr_17164[7] = inst_17008__$1);
return statearr_17164;
})();if(inst_17008__$1)
{var statearr_17165_17231 = state_17107__$1;(statearr_17165_17231[1] = 16);
} else
{var statearr_17166_17232 = state_17107__$1;(statearr_17166_17232[1] = 17);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 9))
{var inst_17036 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17167_17233 = state_17107__$1;(statearr_17167_17233[2] = inst_17036);
(statearr_17167_17233[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 5))
{var inst_16986 = cljs.core.deref.call(null,cs);var inst_16987 = cljs.core.seq.call(null,inst_16986);var inst_16988 = inst_16987;var inst_16989 = null;var inst_16990 = 0;var inst_16991 = 0;var state_17107__$1 = (function (){var statearr_17168 = state_17107;(statearr_17168[13] = inst_16989);
(statearr_17168[14] = inst_16988);
(statearr_17168[16] = inst_16990);
(statearr_17168[17] = inst_16991);
return statearr_17168;
})();var statearr_17169_17234 = state_17107__$1;(statearr_17169_17234[2] = null);
(statearr_17169_17234[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 14))
{var state_17107__$1 = state_17107;var statearr_17170_17235 = state_17107__$1;(statearr_17170_17235[2] = null);
(statearr_17170_17235[1] = 15);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 45))
{var inst_17097 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17171_17236 = state_17107__$1;(statearr_17171_17236[2] = inst_17097);
(statearr_17171_17236[1] = 44);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 26))
{var inst_17039 = (state_17107[29]);var inst_17093 = (state_17107[2]);var inst_17094 = cljs.core.seq.call(null,inst_17039);var state_17107__$1 = (function (){var statearr_17172 = state_17107;(statearr_17172[31] = inst_17093);
return statearr_17172;
})();if(inst_17094)
{var statearr_17173_17237 = state_17107__$1;(statearr_17173_17237[1] = 42);
} else
{var statearr_17174_17238 = state_17107__$1;(statearr_17174_17238[1] = 43);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 16))
{var inst_17008 = (state_17107[7]);var inst_17010 = cljs.core.chunked_seq_QMARK_.call(null,inst_17008);var state_17107__$1 = state_17107;if(inst_17010)
{var statearr_17175_17239 = state_17107__$1;(statearr_17175_17239[1] = 19);
} else
{var statearr_17176_17240 = state_17107__$1;(statearr_17176_17240[1] = 20);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 38))
{var inst_17086 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17177_17241 = state_17107__$1;(statearr_17177_17241[2] = inst_17086);
(statearr_17177_17241[1] = 35);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 30))
{var state_17107__$1 = state_17107;var statearr_17178_17242 = state_17107__$1;(statearr_17178_17242[2] = null);
(statearr_17178_17242[1] = 32);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 10))
{var inst_16989 = (state_17107[13]);var inst_16991 = (state_17107[17]);var inst_16997 = cljs.core._nth.call(null,inst_16989,inst_16991);var inst_16998 = cljs.core.nth.call(null,inst_16997,0,null);var inst_16999 = cljs.core.nth.call(null,inst_16997,1,null);var state_17107__$1 = (function (){var statearr_17179 = state_17107;(statearr_17179[26] = inst_16998);
return statearr_17179;
})();if(cljs.core.truth_(inst_16999))
{var statearr_17180_17243 = state_17107__$1;(statearr_17180_17243[1] = 13);
} else
{var statearr_17181_17244 = state_17107__$1;(statearr_17181_17244[1] = 14);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 18))
{var inst_17032 = (state_17107[2]);var state_17107__$1 = state_17107;var statearr_17182_17245 = state_17107__$1;(statearr_17182_17245[2] = inst_17032);
(statearr_17182_17245[1] = 12);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 42))
{var state_17107__$1 = state_17107;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_17107__$1,45,dchan);
} else
{if((state_val_17108 === 37))
{var inst_17075 = (state_17107[23]);var inst_17066 = (state_17107[25]);var inst_16979 = (state_17107[10]);var inst_17075__$1 = cljs.core.first.call(null,inst_17066);var inst_17076 = cljs.core.async.put_BANG_.call(null,inst_17075__$1,inst_16979,done);var state_17107__$1 = (function (){var statearr_17183 = state_17107;(statearr_17183[23] = inst_17075__$1);
return statearr_17183;
})();if(cljs.core.truth_(inst_17076))
{var statearr_17184_17246 = state_17107__$1;(statearr_17184_17246[1] = 39);
} else
{var statearr_17185_17247 = state_17107__$1;(statearr_17185_17247[1] = 40);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17108 === 8))
{var inst_16990 = (state_17107[16]);var inst_16991 = (state_17107[17]);var inst_16993 = (inst_16991 < inst_16990);var inst_16994 = inst_16993;var state_17107__$1 = state_17107;if(cljs.core.truth_(inst_16994))
{var statearr_17186_17248 = state_17107__$1;(statearr_17186_17248[1] = 10);
} else
{var statearr_17187_17249 = state_17107__$1;(statearr_17187_17249[1] = 11);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___17195,cs,m,dchan,dctr,done))
;return ((function (switch__11563__auto__,c__11627__auto___17195,cs,m,dchan,dctr,done){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_17191 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_17191[0] = state_machine__11564__auto__);
(statearr_17191[1] = 1);
return statearr_17191;
});
var state_machine__11564__auto____1 = (function (state_17107){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_17107);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e17192){if((e17192 instanceof Object))
{var ex__11567__auto__ = e17192;var statearr_17193_17250 = state_17107;(statearr_17193_17250[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17107);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e17192;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__17251 = state_17107;
state_17107 = G__17251;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_17107){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_17107);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___17195,cs,m,dchan,dctr,done))
})();var state__11629__auto__ = (function (){var statearr_17194 = f__11628__auto__.call(null);(statearr_17194[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___17195);
return statearr_17194;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___17195,cs,m,dchan,dctr,done))
);
return m;
});
/**
* Copies the mult source onto the supplied channel.
* 
* By default the channel will be closed when the source closes,
* but can be determined by the close? parameter.
*/
cljs.core.async.tap = (function() {
var tap = null;
var tap__2 = (function (mult,ch){return tap.call(null,mult,ch,true);
});
var tap__3 = (function (mult,ch,close_QMARK_){cljs.core.async.tap_STAR_.call(null,mult,ch,close_QMARK_);
return ch;
});
tap = function(mult,ch,close_QMARK_){
switch(arguments.length){
case 2:
return tap__2.call(this,mult,ch);
case 3:
return tap__3.call(this,mult,ch,close_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
tap.cljs$core$IFn$_invoke$arity$2 = tap__2;
tap.cljs$core$IFn$_invoke$arity$3 = tap__3;
return tap;
})()
;
/**
* Disconnects a target channel from a mult
*/
cljs.core.async.untap = (function untap(mult,ch){return cljs.core.async.untap_STAR_.call(null,mult,ch);
});
/**
* Disconnects all target channels from a mult
*/
cljs.core.async.untap_all = (function untap_all(mult){return cljs.core.async.untap_all_STAR_.call(null,mult);
});
cljs.core.async.Mix = (function (){var obj17253 = {};return obj17253;
})();
cljs.core.async.admix_STAR_ = (function admix_STAR_(m,ch){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mix$admix_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mix$admix_STAR_$arity$2(m,ch);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.admix_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.admix_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mix.admix*",m);
}
}
})().call(null,m,ch);
}
});
cljs.core.async.unmix_STAR_ = (function unmix_STAR_(m,ch){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mix$unmix_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mix$unmix_STAR_$arity$2(m,ch);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.unmix_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.unmix_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mix.unmix*",m);
}
}
})().call(null,m,ch);
}
});
cljs.core.async.unmix_all_STAR_ = (function unmix_all_STAR_(m){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mix$unmix_all_STAR_$arity$1;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mix$unmix_all_STAR_$arity$1(m);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.unmix_all_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.unmix_all_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mix.unmix-all*",m);
}
}
})().call(null,m);
}
});
cljs.core.async.toggle_STAR_ = (function toggle_STAR_(m,state_map){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mix$toggle_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mix$toggle_STAR_$arity$2(m,state_map);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.toggle_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.toggle_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mix.toggle*",m);
}
}
})().call(null,m,state_map);
}
});
cljs.core.async.solo_mode_STAR_ = (function solo_mode_STAR_(m,mode){if((function (){var and__7862__auto__ = m;if(and__7862__auto__)
{return m.cljs$core$async$Mix$solo_mode_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return m.cljs$core$async$Mix$solo_mode_STAR_$arity$2(m,mode);
} else
{var x__8501__auto__ = (((m == null))?null:m);return (function (){var or__7874__auto__ = (cljs.core.async.solo_mode_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.solo_mode_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Mix.solo-mode*",m);
}
}
})().call(null,m,mode);
}
});
/**
* Creates and returns a mix of one or more input channels which will
* be put on the supplied out channel. Input sources can be added to
* the mix with 'admix', and removed with 'unmix'. A mix supports
* soloing, muting and pausing multiple inputs atomically using
* 'toggle', and can solo using either muting or pausing as determined
* by 'solo-mode'.
* 
* Each channel can have zero or more boolean modes set via 'toggle':
* 
* :solo - when true, only this (ond other soloed) channel(s) will appear
* in the mix output channel. :mute and :pause states of soloed
* channels are ignored. If solo-mode is :mute, non-soloed
* channels are muted, if :pause, non-soloed channels are
* paused.
* 
* :mute - muted channels will have their contents consumed but not included in the mix
* :pause - paused channels will not have their contents consumed (and thus also not included in the mix)
*/
cljs.core.async.mix = (function mix(out){var cs = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);var solo_modes = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"pause","pause",1120344424),null,new cljs.core.Keyword(null,"mute","mute",1017267595),null], null), null);var attrs = cljs.core.conj.call(null,solo_modes,new cljs.core.Keyword(null,"solo","solo",1017440337));var solo_mode = cljs.core.atom.call(null,new cljs.core.Keyword(null,"mute","mute",1017267595));var change = cljs.core.async.chan.call(null);var changed = ((function (cs,solo_modes,attrs,solo_mode,change){
return (function (){return cljs.core.async.put_BANG_.call(null,change,true);
});})(cs,solo_modes,attrs,solo_mode,change))
;var pick = ((function (cs,solo_modes,attrs,solo_mode,change,changed){
return (function (attr,chs){return cljs.core.reduce_kv.call(null,((function (cs,solo_modes,attrs,solo_mode,change,changed){
return (function (ret,c,v){if(cljs.core.truth_(attr.call(null,v)))
{return cljs.core.conj.call(null,ret,c);
} else
{return ret;
}
});})(cs,solo_modes,attrs,solo_mode,change,changed))
,cljs.core.PersistentHashSet.EMPTY,chs);
});})(cs,solo_modes,attrs,solo_mode,change,changed))
;var calc_state = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick){
return (function (){var chs = cljs.core.deref.call(null,cs);var mode = cljs.core.deref.call(null,solo_mode);var solos = pick.call(null,new cljs.core.Keyword(null,"solo","solo",1017440337),chs);var pauses = pick.call(null,new cljs.core.Keyword(null,"pause","pause",1120344424),chs);return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"solos","solos",1123523302),solos,new cljs.core.Keyword(null,"mutes","mutes",1118168300),pick.call(null,new cljs.core.Keyword(null,"mute","mute",1017267595),chs),new cljs.core.Keyword(null,"reads","reads",1122290959),cljs.core.conj.call(null,(((cljs.core._EQ_.call(null,mode,new cljs.core.Keyword(null,"pause","pause",1120344424))) && (!(cljs.core.empty_QMARK_.call(null,solos))))?cljs.core.vec.call(null,solos):cljs.core.vec.call(null,cljs.core.remove.call(null,pauses,cljs.core.keys.call(null,chs)))),change)], null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick))
;var m = (function (){if(typeof cljs.core.async.t17373 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t17373 = (function (change,mix,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,meta17374){
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
this.meta17374 = meta17374;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t17373.cljs$lang$type = true;
cljs.core.async.t17373.cljs$lang$ctorStr = "cljs.core.async/t17373";
cljs.core.async.t17373.cljs$lang$ctorPrWriter = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t17373");
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$ = true;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$admix_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,ch){var self__ = this;
var ___$1 = this;cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.assoc,ch,cljs.core.PersistentArrayMap.EMPTY);
return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$unmix_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,ch){var self__ = this;
var ___$1 = this;cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.dissoc,ch);
return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$unmix_all_STAR_$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_){var self__ = this;
var ___$1 = this;cljs.core.reset_BANG_.call(null,self__.cs,cljs.core.PersistentArrayMap.EMPTY);
return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$toggle_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,state_map){var self__ = this;
var ___$1 = this;cljs.core.swap_BANG_.call(null,self__.cs,cljs.core.partial.call(null,cljs.core.merge_with,cljs.core.merge),state_map);
return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mix$solo_mode_STAR_$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_,mode){var self__ = this;
var ___$1 = this;if(cljs.core.truth_(self__.solo_modes.call(null,mode)))
{} else
{throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str([cljs.core.str("mode must be one of: "),cljs.core.str(self__.solo_modes)].join('')),cljs.core.str("\n"),cljs.core.str(cljs.core.pr_str.call(null,cljs.core.list(new cljs.core.Symbol(null,"solo-modes","solo-modes",-1162732933,null),new cljs.core.Symbol(null,"mode","mode",-1637174436,null))))].join('')));
}
cljs.core.reset_BANG_.call(null,self__.solo_mode,mode);
return self__.changed.call(null);
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$async$Mux$ = true;
cljs.core.async.t17373.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_){var self__ = this;
var ___$1 = this;return self__.out;
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_17375){var self__ = this;
var _17375__$1 = this;return self__.meta17374;
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.t17373.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function (_17375,meta17374__$1){var self__ = this;
var _17375__$1 = this;return (new cljs.core.async.t17373(self__.change,self__.mix,self__.solo_mode,self__.pick,self__.cs,self__.calc_state,self__.out,self__.changed,self__.solo_modes,self__.attrs,meta17374__$1));
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
cljs.core.async.__GT_t17373 = ((function (cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state){
return (function __GT_t17373(change__$1,mix__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta17374){return (new cljs.core.async.t17373(change__$1,mix__$1,solo_mode__$1,pick__$1,cs__$1,calc_state__$1,out__$1,changed__$1,solo_modes__$1,attrs__$1,meta17374));
});})(cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state))
;
}
return (new cljs.core.async.t17373(change,mix,solo_mode,pick,cs,calc_state,out,changed,solo_modes,attrs,null));
})();var c__11627__auto___17492 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function (state_17445){var state_val_17446 = (state_17445[1]);if((state_val_17446 === 7))
{var inst_17389 = (state_17445[7]);var inst_17394 = cljs.core.apply.call(null,cljs.core.hash_map,inst_17389);var state_17445__$1 = state_17445;var statearr_17447_17493 = state_17445__$1;(statearr_17447_17493[2] = inst_17394);
(statearr_17447_17493[1] = 9);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 20))
{var inst_17404 = (state_17445[8]);var state_17445__$1 = state_17445;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_17445__$1,23,out,inst_17404);
} else
{if((state_val_17446 === 1))
{var inst_17379 = (state_17445[9]);var inst_17379__$1 = calc_state.call(null);var inst_17380 = cljs.core.seq_QMARK_.call(null,inst_17379__$1);var state_17445__$1 = (function (){var statearr_17448 = state_17445;(statearr_17448[9] = inst_17379__$1);
return statearr_17448;
})();if(inst_17380)
{var statearr_17449_17494 = state_17445__$1;(statearr_17449_17494[1] = 2);
} else
{var statearr_17450_17495 = state_17445__$1;(statearr_17450_17495[1] = 3);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 24))
{var inst_17397 = (state_17445[10]);var inst_17389 = inst_17397;var state_17445__$1 = (function (){var statearr_17451 = state_17445;(statearr_17451[7] = inst_17389);
return statearr_17451;
})();var statearr_17452_17496 = state_17445__$1;(statearr_17452_17496[2] = null);
(statearr_17452_17496[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 4))
{var inst_17379 = (state_17445[9]);var inst_17385 = (state_17445[2]);var inst_17386 = cljs.core.get.call(null,inst_17385,new cljs.core.Keyword(null,"reads","reads",1122290959));var inst_17387 = cljs.core.get.call(null,inst_17385,new cljs.core.Keyword(null,"mutes","mutes",1118168300));var inst_17388 = cljs.core.get.call(null,inst_17385,new cljs.core.Keyword(null,"solos","solos",1123523302));var inst_17389 = inst_17379;var state_17445__$1 = (function (){var statearr_17453 = state_17445;(statearr_17453[11] = inst_17387);
(statearr_17453[12] = inst_17388);
(statearr_17453[7] = inst_17389);
(statearr_17453[13] = inst_17386);
return statearr_17453;
})();var statearr_17454_17497 = state_17445__$1;(statearr_17454_17497[2] = null);
(statearr_17454_17497[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 15))
{var state_17445__$1 = state_17445;var statearr_17455_17498 = state_17445__$1;(statearr_17455_17498[2] = null);
(statearr_17455_17498[1] = 16);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 21))
{var inst_17397 = (state_17445[10]);var inst_17389 = inst_17397;var state_17445__$1 = (function (){var statearr_17456 = state_17445;(statearr_17456[7] = inst_17389);
return statearr_17456;
})();var statearr_17457_17499 = state_17445__$1;(statearr_17457_17499[2] = null);
(statearr_17457_17499[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 13))
{var inst_17441 = (state_17445[2]);var state_17445__$1 = state_17445;var statearr_17458_17500 = state_17445__$1;(statearr_17458_17500[2] = inst_17441);
(statearr_17458_17500[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 22))
{var inst_17439 = (state_17445[2]);var state_17445__$1 = state_17445;var statearr_17459_17501 = state_17445__$1;(statearr_17459_17501[2] = inst_17439);
(statearr_17459_17501[1] = 13);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 6))
{var inst_17443 = (state_17445[2]);var state_17445__$1 = state_17445;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_17445__$1,inst_17443);
} else
{if((state_val_17446 === 25))
{var state_17445__$1 = state_17445;var statearr_17460_17502 = state_17445__$1;(statearr_17460_17502[2] = null);
(statearr_17460_17502[1] = 26);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 17))
{var inst_17419 = (state_17445[14]);var state_17445__$1 = state_17445;var statearr_17461_17503 = state_17445__$1;(statearr_17461_17503[2] = inst_17419);
(statearr_17461_17503[1] = 19);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 3))
{var inst_17379 = (state_17445[9]);var state_17445__$1 = state_17445;var statearr_17462_17504 = state_17445__$1;(statearr_17462_17504[2] = inst_17379);
(statearr_17462_17504[1] = 4);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 12))
{var inst_17400 = (state_17445[15]);var inst_17419 = (state_17445[14]);var inst_17405 = (state_17445[16]);var inst_17419__$1 = inst_17400.call(null,inst_17405);var state_17445__$1 = (function (){var statearr_17463 = state_17445;(statearr_17463[14] = inst_17419__$1);
return statearr_17463;
})();if(cljs.core.truth_(inst_17419__$1))
{var statearr_17464_17505 = state_17445__$1;(statearr_17464_17505[1] = 17);
} else
{var statearr_17465_17506 = state_17445__$1;(statearr_17465_17506[1] = 18);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 2))
{var inst_17379 = (state_17445[9]);var inst_17382 = cljs.core.apply.call(null,cljs.core.hash_map,inst_17379);var state_17445__$1 = state_17445;var statearr_17466_17507 = state_17445__$1;(statearr_17466_17507[2] = inst_17382);
(statearr_17466_17507[1] = 4);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 23))
{var inst_17430 = (state_17445[2]);var state_17445__$1 = state_17445;if(cljs.core.truth_(inst_17430))
{var statearr_17467_17508 = state_17445__$1;(statearr_17467_17508[1] = 24);
} else
{var statearr_17468_17509 = state_17445__$1;(statearr_17468_17509[1] = 25);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 19))
{var inst_17427 = (state_17445[2]);var state_17445__$1 = state_17445;if(cljs.core.truth_(inst_17427))
{var statearr_17469_17510 = state_17445__$1;(statearr_17469_17510[1] = 20);
} else
{var statearr_17470_17511 = state_17445__$1;(statearr_17470_17511[1] = 21);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 11))
{var inst_17404 = (state_17445[8]);var inst_17410 = (inst_17404 == null);var state_17445__$1 = state_17445;if(cljs.core.truth_(inst_17410))
{var statearr_17471_17512 = state_17445__$1;(statearr_17471_17512[1] = 14);
} else
{var statearr_17472_17513 = state_17445__$1;(statearr_17472_17513[1] = 15);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 9))
{var inst_17397 = (state_17445[10]);var inst_17397__$1 = (state_17445[2]);var inst_17398 = cljs.core.get.call(null,inst_17397__$1,new cljs.core.Keyword(null,"reads","reads",1122290959));var inst_17399 = cljs.core.get.call(null,inst_17397__$1,new cljs.core.Keyword(null,"mutes","mutes",1118168300));var inst_17400 = cljs.core.get.call(null,inst_17397__$1,new cljs.core.Keyword(null,"solos","solos",1123523302));var state_17445__$1 = (function (){var statearr_17473 = state_17445;(statearr_17473[10] = inst_17397__$1);
(statearr_17473[15] = inst_17400);
(statearr_17473[17] = inst_17399);
return statearr_17473;
})();return cljs.core.async.impl.ioc_helpers.ioc_alts_BANG_.call(null,state_17445__$1,10,inst_17398);
} else
{if((state_val_17446 === 5))
{var inst_17389 = (state_17445[7]);var inst_17392 = cljs.core.seq_QMARK_.call(null,inst_17389);var state_17445__$1 = state_17445;if(inst_17392)
{var statearr_17474_17514 = state_17445__$1;(statearr_17474_17514[1] = 7);
} else
{var statearr_17475_17515 = state_17445__$1;(statearr_17475_17515[1] = 8);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 14))
{var inst_17405 = (state_17445[16]);var inst_17412 = cljs.core.swap_BANG_.call(null,cs,cljs.core.dissoc,inst_17405);var state_17445__$1 = state_17445;var statearr_17476_17516 = state_17445__$1;(statearr_17476_17516[2] = inst_17412);
(statearr_17476_17516[1] = 16);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 26))
{var inst_17435 = (state_17445[2]);var state_17445__$1 = state_17445;var statearr_17477_17517 = state_17445__$1;(statearr_17477_17517[2] = inst_17435);
(statearr_17477_17517[1] = 22);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 16))
{var inst_17415 = (state_17445[2]);var inst_17416 = calc_state.call(null);var inst_17389 = inst_17416;var state_17445__$1 = (function (){var statearr_17478 = state_17445;(statearr_17478[18] = inst_17415);
(statearr_17478[7] = inst_17389);
return statearr_17478;
})();var statearr_17479_17518 = state_17445__$1;(statearr_17479_17518[2] = null);
(statearr_17479_17518[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 10))
{var inst_17404 = (state_17445[8]);var inst_17405 = (state_17445[16]);var inst_17403 = (state_17445[2]);var inst_17404__$1 = cljs.core.nth.call(null,inst_17403,0,null);var inst_17405__$1 = cljs.core.nth.call(null,inst_17403,1,null);var inst_17406 = (inst_17404__$1 == null);var inst_17407 = cljs.core._EQ_.call(null,inst_17405__$1,change);var inst_17408 = (inst_17406) || (inst_17407);var state_17445__$1 = (function (){var statearr_17480 = state_17445;(statearr_17480[8] = inst_17404__$1);
(statearr_17480[16] = inst_17405__$1);
return statearr_17480;
})();if(cljs.core.truth_(inst_17408))
{var statearr_17481_17519 = state_17445__$1;(statearr_17481_17519[1] = 11);
} else
{var statearr_17482_17520 = state_17445__$1;(statearr_17482_17520[1] = 12);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 18))
{var inst_17400 = (state_17445[15]);var inst_17405 = (state_17445[16]);var inst_17399 = (state_17445[17]);var inst_17422 = cljs.core.empty_QMARK_.call(null,inst_17400);var inst_17423 = inst_17399.call(null,inst_17405);var inst_17424 = cljs.core.not.call(null,inst_17423);var inst_17425 = (inst_17422) && (inst_17424);var state_17445__$1 = state_17445;var statearr_17483_17521 = state_17445__$1;(statearr_17483_17521[2] = inst_17425);
(statearr_17483_17521[1] = 19);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17446 === 8))
{var inst_17389 = (state_17445[7]);var state_17445__$1 = state_17445;var statearr_17484_17522 = state_17445__$1;(statearr_17484_17522[2] = inst_17389);
(statearr_17484_17522[1] = 9);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
;return ((function (switch__11563__auto__,c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_17488 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_17488[0] = state_machine__11564__auto__);
(statearr_17488[1] = 1);
return statearr_17488;
});
var state_machine__11564__auto____1 = (function (state_17445){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_17445);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e17489){if((e17489 instanceof Object))
{var ex__11567__auto__ = e17489;var statearr_17490_17523 = state_17445;(statearr_17490_17523[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17445);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e17489;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__17524 = state_17445;
state_17445 = G__17524;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_17445){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_17445);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
})();var state__11629__auto__ = (function (){var statearr_17491 = f__11628__auto__.call(null);(statearr_17491[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___17492);
return statearr_17491;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___17492,cs,solo_modes,attrs,solo_mode,change,changed,pick,calc_state,m))
);
return m;
});
/**
* Adds ch as an input to the mix
*/
cljs.core.async.admix = (function admix(mix,ch){return cljs.core.async.admix_STAR_.call(null,mix,ch);
});
/**
* Removes ch as an input to the mix
*/
cljs.core.async.unmix = (function unmix(mix,ch){return cljs.core.async.unmix_STAR_.call(null,mix,ch);
});
/**
* removes all inputs from the mix
*/
cljs.core.async.unmix_all = (function unmix_all(mix){return cljs.core.async.unmix_all_STAR_.call(null,mix);
});
/**
* Atomically sets the state(s) of one or more channels in a mix. The
* state map is a map of channels -> channel-state-map. A
* channel-state-map is a map of attrs -> boolean, where attr is one or
* more of :mute, :pause or :solo. Any states supplied are merged with
* the current state.
* 
* Note that channels can be added to a mix via toggle, which can be
* used to add channels in a particular (e.g. paused) state.
*/
cljs.core.async.toggle = (function toggle(mix,state_map){return cljs.core.async.toggle_STAR_.call(null,mix,state_map);
});
/**
* Sets the solo mode of the mix. mode must be one of :mute or :pause
*/
cljs.core.async.solo_mode = (function solo_mode(mix,mode){return cljs.core.async.solo_mode_STAR_.call(null,mix,mode);
});
cljs.core.async.Pub = (function (){var obj17526 = {};return obj17526;
})();
cljs.core.async.sub_STAR_ = (function sub_STAR_(p,v,ch,close_QMARK_){if((function (){var and__7862__auto__ = p;if(and__7862__auto__)
{return p.cljs$core$async$Pub$sub_STAR_$arity$4;
} else
{return and__7862__auto__;
}
})())
{return p.cljs$core$async$Pub$sub_STAR_$arity$4(p,v,ch,close_QMARK_);
} else
{var x__8501__auto__ = (((p == null))?null:p);return (function (){var or__7874__auto__ = (cljs.core.async.sub_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.sub_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Pub.sub*",p);
}
}
})().call(null,p,v,ch,close_QMARK_);
}
});
cljs.core.async.unsub_STAR_ = (function unsub_STAR_(p,v,ch){if((function (){var and__7862__auto__ = p;if(and__7862__auto__)
{return p.cljs$core$async$Pub$unsub_STAR_$arity$3;
} else
{return and__7862__auto__;
}
})())
{return p.cljs$core$async$Pub$unsub_STAR_$arity$3(p,v,ch);
} else
{var x__8501__auto__ = (((p == null))?null:p);return (function (){var or__7874__auto__ = (cljs.core.async.unsub_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.unsub_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Pub.unsub*",p);
}
}
})().call(null,p,v,ch);
}
});
cljs.core.async.unsub_all_STAR_ = (function() {
var unsub_all_STAR_ = null;
var unsub_all_STAR___1 = (function (p){if((function (){var and__7862__auto__ = p;if(and__7862__auto__)
{return p.cljs$core$async$Pub$unsub_all_STAR_$arity$1;
} else
{return and__7862__auto__;
}
})())
{return p.cljs$core$async$Pub$unsub_all_STAR_$arity$1(p);
} else
{var x__8501__auto__ = (((p == null))?null:p);return (function (){var or__7874__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.unsub_all_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Pub.unsub-all*",p);
}
}
})().call(null,p);
}
});
var unsub_all_STAR___2 = (function (p,v){if((function (){var and__7862__auto__ = p;if(and__7862__auto__)
{return p.cljs$core$async$Pub$unsub_all_STAR_$arity$2;
} else
{return and__7862__auto__;
}
})())
{return p.cljs$core$async$Pub$unsub_all_STAR_$arity$2(p,v);
} else
{var x__8501__auto__ = (((p == null))?null:p);return (function (){var or__7874__auto__ = (cljs.core.async.unsub_all_STAR_[goog.typeOf(x__8501__auto__)]);if(or__7874__auto__)
{return or__7874__auto__;
} else
{var or__7874__auto____$1 = (cljs.core.async.unsub_all_STAR_["_"]);if(or__7874__auto____$1)
{return or__7874__auto____$1;
} else
{throw cljs.core.missing_protocol.call(null,"Pub.unsub-all*",p);
}
}
})().call(null,p,v);
}
});
unsub_all_STAR_ = function(p,v){
switch(arguments.length){
case 1:
return unsub_all_STAR___1.call(this,p);
case 2:
return unsub_all_STAR___2.call(this,p,v);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
unsub_all_STAR_.cljs$core$IFn$_invoke$arity$1 = unsub_all_STAR___1;
unsub_all_STAR_.cljs$core$IFn$_invoke$arity$2 = unsub_all_STAR___2;
return unsub_all_STAR_;
})()
;
/**
* Creates and returns a pub(lication) of the supplied channel,
* partitioned into topics by the topic-fn. topic-fn will be applied to
* each value on the channel and the result will determine the 'topic'
* on which that value will be put. Channels can be subscribed to
* receive copies of topics using 'sub', and unsubscribed using
* 'unsub'. Each topic will be handled by an internal mult on a
* dedicated channel. By default these internal channels are
* unbuffered, but a buf-fn can be supplied which, given a topic,
* creates a buffer with desired properties.
* 
* Each item is distributed to all subs in parallel and synchronously,
* i.e. each sub must accept before the next item is distributed. Use
* buffering/windowing to prevent slow subs from holding up the pub.
* 
* Items received when there are no matching subs get dropped.
* 
* Note that if buf-fns are used then each topic is handled
* asynchronously, i.e. if a channel is subscribed to more than one
* topic it should not expect them to be interleaved identically with
* the source.
*/
cljs.core.async.pub = (function() {
var pub = null;
var pub__2 = (function (ch,topic_fn){return pub.call(null,ch,topic_fn,cljs.core.constantly.call(null,null));
});
var pub__3 = (function (ch,topic_fn,buf_fn){var mults = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);var ensure_mult = ((function (mults){
return (function (topic){var or__7874__auto__ = cljs.core.get.call(null,cljs.core.deref.call(null,mults),topic);if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{return cljs.core.get.call(null,cljs.core.swap_BANG_.call(null,mults,((function (or__7874__auto__,mults){
return (function (p1__17527_SHARP_){if(cljs.core.truth_(p1__17527_SHARP_.call(null,topic)))
{return p1__17527_SHARP_;
} else
{return cljs.core.assoc.call(null,p1__17527_SHARP_,topic,cljs.core.async.mult.call(null,cljs.core.async.chan.call(null,buf_fn.call(null,topic))));
}
});})(or__7874__auto__,mults))
),topic);
}
});})(mults))
;var p = (function (){if(typeof cljs.core.async.t17642 !== 'undefined')
{} else
{
/**
* @constructor
*/
cljs.core.async.t17642 = (function (ensure_mult,mults,buf_fn,topic_fn,ch,pub,meta17643){
this.ensure_mult = ensure_mult;
this.mults = mults;
this.buf_fn = buf_fn;
this.topic_fn = topic_fn;
this.ch = ch;
this.pub = pub;
this.meta17643 = meta17643;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
cljs.core.async.t17642.cljs$lang$type = true;
cljs.core.async.t17642.cljs$lang$ctorStr = "cljs.core.async/t17642";
cljs.core.async.t17642.cljs$lang$ctorPrWriter = ((function (mults,ensure_mult){
return (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"cljs.core.async/t17642");
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$async$Pub$ = true;
cljs.core.async.t17642.prototype.cljs$core$async$Pub$sub_STAR_$arity$4 = ((function (mults,ensure_mult){
return (function (p,topic,ch__$2,close_QMARK_){var self__ = this;
var p__$1 = this;var m = self__.ensure_mult.call(null,topic);return cljs.core.async.tap.call(null,m,ch__$2,close_QMARK_);
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$async$Pub$unsub_STAR_$arity$3 = ((function (mults,ensure_mult){
return (function (p,topic,ch__$2){var self__ = this;
var p__$1 = this;var temp__4126__auto__ = cljs.core.get.call(null,cljs.core.deref.call(null,self__.mults),topic);if(cljs.core.truth_(temp__4126__auto__))
{var m = temp__4126__auto__;return cljs.core.async.untap.call(null,m,ch__$2);
} else
{return null;
}
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$1 = ((function (mults,ensure_mult){
return (function (_){var self__ = this;
var ___$1 = this;return cljs.core.reset_BANG_.call(null,self__.mults,cljs.core.PersistentArrayMap.EMPTY);
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$async$Pub$unsub_all_STAR_$arity$2 = ((function (mults,ensure_mult){
return (function (_,topic){var self__ = this;
var ___$1 = this;return cljs.core.swap_BANG_.call(null,self__.mults,cljs.core.dissoc,topic);
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$async$Mux$ = true;
cljs.core.async.t17642.prototype.cljs$core$async$Mux$muxch_STAR_$arity$1 = ((function (mults,ensure_mult){
return (function (_){var self__ = this;
var ___$1 = this;return self__.ch;
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$IMeta$_meta$arity$1 = ((function (mults,ensure_mult){
return (function (_17644){var self__ = this;
var _17644__$1 = this;return self__.meta17643;
});})(mults,ensure_mult))
;
cljs.core.async.t17642.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = ((function (mults,ensure_mult){
return (function (_17644,meta17643__$1){var self__ = this;
var _17644__$1 = this;return (new cljs.core.async.t17642(self__.ensure_mult,self__.mults,self__.buf_fn,self__.topic_fn,self__.ch,self__.pub,meta17643__$1));
});})(mults,ensure_mult))
;
cljs.core.async.__GT_t17642 = ((function (mults,ensure_mult){
return (function __GT_t17642(ensure_mult__$1,mults__$1,buf_fn__$1,topic_fn__$1,ch__$1,pub__$1,meta17643){return (new cljs.core.async.t17642(ensure_mult__$1,mults__$1,buf_fn__$1,topic_fn__$1,ch__$1,pub__$1,meta17643));
});})(mults,ensure_mult))
;
}
return (new cljs.core.async.t17642(ensure_mult,mults,buf_fn,topic_fn,ch,pub,null));
})();var c__11627__auto___17756 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___17756,mults,ensure_mult,p){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___17756,mults,ensure_mult,p){
return (function (state_17712){var state_val_17713 = (state_17712[1]);if((state_val_17713 === 7))
{var inst_17708 = (state_17712[2]);var state_17712__$1 = state_17712;var statearr_17714_17757 = state_17712__$1;(statearr_17714_17757[2] = inst_17708);
(statearr_17714_17757[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 20))
{var state_17712__$1 = state_17712;var statearr_17715_17758 = state_17712__$1;(statearr_17715_17758[2] = null);
(statearr_17715_17758[1] = 22);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 1))
{var state_17712__$1 = state_17712;var statearr_17716_17759 = state_17712__$1;(statearr_17716_17759[2] = null);
(statearr_17716_17759[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 4))
{var inst_17647 = (state_17712[7]);var inst_17647__$1 = (state_17712[2]);var inst_17648 = (inst_17647__$1 == null);var state_17712__$1 = (function (){var statearr_17717 = state_17712;(statearr_17717[7] = inst_17647__$1);
return statearr_17717;
})();if(cljs.core.truth_(inst_17648))
{var statearr_17718_17760 = state_17712__$1;(statearr_17718_17760[1] = 5);
} else
{var statearr_17719_17761 = state_17712__$1;(statearr_17719_17761[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 15))
{var inst_17689 = (state_17712[2]);var state_17712__$1 = state_17712;var statearr_17720_17762 = state_17712__$1;(statearr_17720_17762[2] = inst_17689);
(statearr_17720_17762[1] = 12);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 21))
{var inst_17695 = (state_17712[8]);var inst_17703 = cljs.core.swap_BANG_.call(null,mults,cljs.core.dissoc,inst_17695);var state_17712__$1 = state_17712;var statearr_17721_17763 = state_17712__$1;(statearr_17721_17763[2] = inst_17703);
(statearr_17721_17763[1] = 22);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 13))
{var inst_17671 = (state_17712[9]);var inst_17673 = cljs.core.chunked_seq_QMARK_.call(null,inst_17671);var state_17712__$1 = state_17712;if(inst_17673)
{var statearr_17722_17764 = state_17712__$1;(statearr_17722_17764[1] = 16);
} else
{var statearr_17723_17765 = state_17712__$1;(statearr_17723_17765[1] = 17);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 22))
{var inst_17705 = (state_17712[2]);var state_17712__$1 = (function (){var statearr_17724 = state_17712;(statearr_17724[10] = inst_17705);
return statearr_17724;
})();var statearr_17725_17766 = state_17712__$1;(statearr_17725_17766[2] = null);
(statearr_17725_17766[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 6))
{var inst_17647 = (state_17712[7]);var inst_17695 = (state_17712[8]);var inst_17695__$1 = topic_fn.call(null,inst_17647);var inst_17696 = cljs.core.deref.call(null,mults);var inst_17697 = cljs.core.get.call(null,inst_17696,inst_17695__$1);var inst_17698 = cljs.core.async.muxch_STAR_.call(null,inst_17697);var state_17712__$1 = (function (){var statearr_17726 = state_17712;(statearr_17726[8] = inst_17695__$1);
return statearr_17726;
})();return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_17712__$1,19,inst_17698,inst_17647);
} else
{if((state_val_17713 === 17))
{var inst_17671 = (state_17712[9]);var inst_17680 = cljs.core.first.call(null,inst_17671);var inst_17681 = cljs.core.async.muxch_STAR_.call(null,inst_17680);var inst_17682 = cljs.core.async.close_BANG_.call(null,inst_17681);var inst_17683 = cljs.core.next.call(null,inst_17671);var inst_17657 = inst_17683;var inst_17658 = null;var inst_17659 = 0;var inst_17660 = 0;var state_17712__$1 = (function (){var statearr_17727 = state_17712;(statearr_17727[11] = inst_17682);
(statearr_17727[12] = inst_17659);
(statearr_17727[13] = inst_17658);
(statearr_17727[14] = inst_17657);
(statearr_17727[15] = inst_17660);
return statearr_17727;
})();var statearr_17728_17767 = state_17712__$1;(statearr_17728_17767[2] = null);
(statearr_17728_17767[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 3))
{var inst_17710 = (state_17712[2]);var state_17712__$1 = state_17712;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_17712__$1,inst_17710);
} else
{if((state_val_17713 === 12))
{var inst_17691 = (state_17712[2]);var state_17712__$1 = state_17712;var statearr_17729_17768 = state_17712__$1;(statearr_17729_17768[2] = inst_17691);
(statearr_17729_17768[1] = 9);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 2))
{var state_17712__$1 = state_17712;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_17712__$1,4,ch);
} else
{if((state_val_17713 === 19))
{var inst_17700 = (state_17712[2]);var state_17712__$1 = state_17712;if(cljs.core.truth_(inst_17700))
{var statearr_17730_17769 = state_17712__$1;(statearr_17730_17769[1] = 20);
} else
{var statearr_17731_17770 = state_17712__$1;(statearr_17731_17770[1] = 21);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 11))
{var inst_17671 = (state_17712[9]);var inst_17657 = (state_17712[14]);var inst_17671__$1 = cljs.core.seq.call(null,inst_17657);var state_17712__$1 = (function (){var statearr_17732 = state_17712;(statearr_17732[9] = inst_17671__$1);
return statearr_17732;
})();if(inst_17671__$1)
{var statearr_17733_17771 = state_17712__$1;(statearr_17733_17771[1] = 13);
} else
{var statearr_17734_17772 = state_17712__$1;(statearr_17734_17772[1] = 14);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 9))
{var inst_17693 = (state_17712[2]);var state_17712__$1 = state_17712;var statearr_17735_17773 = state_17712__$1;(statearr_17735_17773[2] = inst_17693);
(statearr_17735_17773[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 5))
{var inst_17654 = cljs.core.deref.call(null,mults);var inst_17655 = cljs.core.vals.call(null,inst_17654);var inst_17656 = cljs.core.seq.call(null,inst_17655);var inst_17657 = inst_17656;var inst_17658 = null;var inst_17659 = 0;var inst_17660 = 0;var state_17712__$1 = (function (){var statearr_17736 = state_17712;(statearr_17736[12] = inst_17659);
(statearr_17736[13] = inst_17658);
(statearr_17736[14] = inst_17657);
(statearr_17736[15] = inst_17660);
return statearr_17736;
})();var statearr_17737_17774 = state_17712__$1;(statearr_17737_17774[2] = null);
(statearr_17737_17774[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 14))
{var state_17712__$1 = state_17712;var statearr_17741_17775 = state_17712__$1;(statearr_17741_17775[2] = null);
(statearr_17741_17775[1] = 15);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 16))
{var inst_17671 = (state_17712[9]);var inst_17675 = cljs.core.chunk_first.call(null,inst_17671);var inst_17676 = cljs.core.chunk_rest.call(null,inst_17671);var inst_17677 = cljs.core.count.call(null,inst_17675);var inst_17657 = inst_17676;var inst_17658 = inst_17675;var inst_17659 = inst_17677;var inst_17660 = 0;var state_17712__$1 = (function (){var statearr_17742 = state_17712;(statearr_17742[12] = inst_17659);
(statearr_17742[13] = inst_17658);
(statearr_17742[14] = inst_17657);
(statearr_17742[15] = inst_17660);
return statearr_17742;
})();var statearr_17743_17776 = state_17712__$1;(statearr_17743_17776[2] = null);
(statearr_17743_17776[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 10))
{var inst_17659 = (state_17712[12]);var inst_17658 = (state_17712[13]);var inst_17657 = (state_17712[14]);var inst_17660 = (state_17712[15]);var inst_17665 = cljs.core._nth.call(null,inst_17658,inst_17660);var inst_17666 = cljs.core.async.muxch_STAR_.call(null,inst_17665);var inst_17667 = cljs.core.async.close_BANG_.call(null,inst_17666);var inst_17668 = (inst_17660 + 1);var tmp17738 = inst_17659;var tmp17739 = inst_17658;var tmp17740 = inst_17657;var inst_17657__$1 = tmp17740;var inst_17658__$1 = tmp17739;var inst_17659__$1 = tmp17738;var inst_17660__$1 = inst_17668;var state_17712__$1 = (function (){var statearr_17744 = state_17712;(statearr_17744[12] = inst_17659__$1);
(statearr_17744[13] = inst_17658__$1);
(statearr_17744[16] = inst_17667);
(statearr_17744[14] = inst_17657__$1);
(statearr_17744[15] = inst_17660__$1);
return statearr_17744;
})();var statearr_17745_17777 = state_17712__$1;(statearr_17745_17777[2] = null);
(statearr_17745_17777[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 18))
{var inst_17686 = (state_17712[2]);var state_17712__$1 = state_17712;var statearr_17746_17778 = state_17712__$1;(statearr_17746_17778[2] = inst_17686);
(statearr_17746_17778[1] = 15);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17713 === 8))
{var inst_17659 = (state_17712[12]);var inst_17660 = (state_17712[15]);var inst_17662 = (inst_17660 < inst_17659);var inst_17663 = inst_17662;var state_17712__$1 = state_17712;if(cljs.core.truth_(inst_17663))
{var statearr_17747_17779 = state_17712__$1;(statearr_17747_17779[1] = 10);
} else
{var statearr_17748_17780 = state_17712__$1;(statearr_17748_17780[1] = 11);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___17756,mults,ensure_mult,p))
;return ((function (switch__11563__auto__,c__11627__auto___17756,mults,ensure_mult,p){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_17752 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_17752[0] = state_machine__11564__auto__);
(statearr_17752[1] = 1);
return statearr_17752;
});
var state_machine__11564__auto____1 = (function (state_17712){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_17712);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e17753){if((e17753 instanceof Object))
{var ex__11567__auto__ = e17753;var statearr_17754_17781 = state_17712;(statearr_17754_17781[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17712);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e17753;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__17782 = state_17712;
state_17712 = G__17782;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_17712){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_17712);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___17756,mults,ensure_mult,p))
})();var state__11629__auto__ = (function (){var statearr_17755 = f__11628__auto__.call(null);(statearr_17755[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___17756);
return statearr_17755;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___17756,mults,ensure_mult,p))
);
return p;
});
pub = function(ch,topic_fn,buf_fn){
switch(arguments.length){
case 2:
return pub__2.call(this,ch,topic_fn);
case 3:
return pub__3.call(this,ch,topic_fn,buf_fn);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
pub.cljs$core$IFn$_invoke$arity$2 = pub__2;
pub.cljs$core$IFn$_invoke$arity$3 = pub__3;
return pub;
})()
;
/**
* Subscribes a channel to a topic of a pub.
* 
* By default the channel will be closed when the source closes,
* but can be determined by the close? parameter.
*/
cljs.core.async.sub = (function() {
var sub = null;
var sub__3 = (function (p,topic,ch){return sub.call(null,p,topic,ch,true);
});
var sub__4 = (function (p,topic,ch,close_QMARK_){return cljs.core.async.sub_STAR_.call(null,p,topic,ch,close_QMARK_);
});
sub = function(p,topic,ch,close_QMARK_){
switch(arguments.length){
case 3:
return sub__3.call(this,p,topic,ch);
case 4:
return sub__4.call(this,p,topic,ch,close_QMARK_);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
sub.cljs$core$IFn$_invoke$arity$3 = sub__3;
sub.cljs$core$IFn$_invoke$arity$4 = sub__4;
return sub;
})()
;
/**
* Unsubscribes a channel from a topic of a pub
*/
cljs.core.async.unsub = (function unsub(p,topic,ch){return cljs.core.async.unsub_STAR_.call(null,p,topic,ch);
});
/**
* Unsubscribes all channels from a pub, or a topic of a pub
*/
cljs.core.async.unsub_all = (function() {
var unsub_all = null;
var unsub_all__1 = (function (p){return cljs.core.async.unsub_all_STAR_.call(null,p);
});
var unsub_all__2 = (function (p,topic){return cljs.core.async.unsub_all_STAR_.call(null,p,topic);
});
unsub_all = function(p,topic){
switch(arguments.length){
case 1:
return unsub_all__1.call(this,p);
case 2:
return unsub_all__2.call(this,p,topic);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
unsub_all.cljs$core$IFn$_invoke$arity$1 = unsub_all__1;
unsub_all.cljs$core$IFn$_invoke$arity$2 = unsub_all__2;
return unsub_all;
})()
;
/**
* Takes a function and a collection of source channels, and returns a
* channel which contains the values produced by applying f to the set
* of first items taken from each source channel, followed by applying
* f to the set of second items from each channel, until any one of the
* channels is closed, at which point the output channel will be
* closed. The returned channel will be unbuffered by default, or a
* buf-or-n can be supplied
*/
cljs.core.async.map = (function() {
var map = null;
var map__2 = (function (f,chs){return map.call(null,f,chs,null);
});
var map__3 = (function (f,chs,buf_or_n){var chs__$1 = cljs.core.vec.call(null,chs);var out = cljs.core.async.chan.call(null,buf_or_n);var cnt = cljs.core.count.call(null,chs__$1);var rets = cljs.core.object_array.call(null,cnt);var dchan = cljs.core.async.chan.call(null,1);var dctr = cljs.core.atom.call(null,null);var done = cljs.core.mapv.call(null,((function (chs__$1,out,cnt,rets,dchan,dctr){
return (function (i){return ((function (chs__$1,out,cnt,rets,dchan,dctr){
return (function (ret){(rets[i] = ret);
if((cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec) === 0))
{return cljs.core.async.put_BANG_.call(null,dchan,rets.slice(0));
} else
{return null;
}
});
;})(chs__$1,out,cnt,rets,dchan,dctr))
});})(chs__$1,out,cnt,rets,dchan,dctr))
,cljs.core.range.call(null,cnt));var c__11627__auto___17919 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function (state_17889){var state_val_17890 = (state_17889[1]);if((state_val_17890 === 7))
{var state_17889__$1 = state_17889;var statearr_17891_17920 = state_17889__$1;(statearr_17891_17920[2] = null);
(statearr_17891_17920[1] = 8);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 1))
{var state_17889__$1 = state_17889;var statearr_17892_17921 = state_17889__$1;(statearr_17892_17921[2] = null);
(statearr_17892_17921[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 4))
{var inst_17853 = (state_17889[7]);var inst_17855 = (inst_17853 < cnt);var state_17889__$1 = state_17889;if(cljs.core.truth_(inst_17855))
{var statearr_17893_17922 = state_17889__$1;(statearr_17893_17922[1] = 6);
} else
{var statearr_17894_17923 = state_17889__$1;(statearr_17894_17923[1] = 7);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 15))
{var inst_17885 = (state_17889[2]);var state_17889__$1 = state_17889;var statearr_17895_17924 = state_17889__$1;(statearr_17895_17924[2] = inst_17885);
(statearr_17895_17924[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 13))
{var inst_17878 = cljs.core.async.close_BANG_.call(null,out);var state_17889__$1 = state_17889;var statearr_17896_17925 = state_17889__$1;(statearr_17896_17925[2] = inst_17878);
(statearr_17896_17925[1] = 15);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 6))
{var state_17889__$1 = state_17889;var statearr_17897_17926 = state_17889__$1;(statearr_17897_17926[2] = null);
(statearr_17897_17926[1] = 11);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 3))
{var inst_17887 = (state_17889[2]);var state_17889__$1 = state_17889;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_17889__$1,inst_17887);
} else
{if((state_val_17890 === 12))
{var inst_17875 = (state_17889[8]);var inst_17875__$1 = (state_17889[2]);var inst_17876 = cljs.core.some.call(null,cljs.core.nil_QMARK_,inst_17875__$1);var state_17889__$1 = (function (){var statearr_17898 = state_17889;(statearr_17898[8] = inst_17875__$1);
return statearr_17898;
})();if(cljs.core.truth_(inst_17876))
{var statearr_17899_17927 = state_17889__$1;(statearr_17899_17927[1] = 13);
} else
{var statearr_17900_17928 = state_17889__$1;(statearr_17900_17928[1] = 14);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 2))
{var inst_17852 = cljs.core.reset_BANG_.call(null,dctr,cnt);var inst_17853 = 0;var state_17889__$1 = (function (){var statearr_17901 = state_17889;(statearr_17901[7] = inst_17853);
(statearr_17901[9] = inst_17852);
return statearr_17901;
})();var statearr_17902_17929 = state_17889__$1;(statearr_17902_17929[2] = null);
(statearr_17902_17929[1] = 4);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 11))
{var inst_17853 = (state_17889[7]);var _ = cljs.core.async.impl.ioc_helpers.add_exception_frame.call(null,state_17889,10,Object,null,9);var inst_17862 = chs__$1.call(null,inst_17853);var inst_17863 = done.call(null,inst_17853);var inst_17864 = cljs.core.async.take_BANG_.call(null,inst_17862,inst_17863);var state_17889__$1 = state_17889;var statearr_17903_17930 = state_17889__$1;(statearr_17903_17930[2] = inst_17864);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17889__$1);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 9))
{var inst_17853 = (state_17889[7]);var inst_17866 = (state_17889[2]);var inst_17867 = (inst_17853 + 1);var inst_17853__$1 = inst_17867;var state_17889__$1 = (function (){var statearr_17904 = state_17889;(statearr_17904[7] = inst_17853__$1);
(statearr_17904[10] = inst_17866);
return statearr_17904;
})();var statearr_17905_17931 = state_17889__$1;(statearr_17905_17931[2] = null);
(statearr_17905_17931[1] = 4);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 5))
{var inst_17873 = (state_17889[2]);var state_17889__$1 = (function (){var statearr_17906 = state_17889;(statearr_17906[11] = inst_17873);
return statearr_17906;
})();return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_17889__$1,12,dchan);
} else
{if((state_val_17890 === 14))
{var inst_17875 = (state_17889[8]);var inst_17880 = cljs.core.apply.call(null,f,inst_17875);var state_17889__$1 = state_17889;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_17889__$1,16,out,inst_17880);
} else
{if((state_val_17890 === 16))
{var inst_17882 = (state_17889[2]);var state_17889__$1 = (function (){var statearr_17907 = state_17889;(statearr_17907[12] = inst_17882);
return statearr_17907;
})();var statearr_17908_17932 = state_17889__$1;(statearr_17908_17932[2] = null);
(statearr_17908_17932[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 10))
{var inst_17857 = (state_17889[2]);var inst_17858 = cljs.core.swap_BANG_.call(null,dctr,cljs.core.dec);var state_17889__$1 = (function (){var statearr_17909 = state_17889;(statearr_17909[13] = inst_17857);
return statearr_17909;
})();var statearr_17910_17933 = state_17889__$1;(statearr_17910_17933[2] = inst_17858);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17889__$1);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_17890 === 8))
{var inst_17871 = (state_17889[2]);var state_17889__$1 = state_17889;var statearr_17911_17934 = state_17889__$1;(statearr_17911_17934[2] = inst_17871);
(statearr_17911_17934[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done))
;return ((function (switch__11563__auto__,c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_17915 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_17915[0] = state_machine__11564__auto__);
(statearr_17915[1] = 1);
return statearr_17915;
});
var state_machine__11564__auto____1 = (function (state_17889){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_17889);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e17916){if((e17916 instanceof Object))
{var ex__11567__auto__ = e17916;var statearr_17917_17935 = state_17889;(statearr_17917_17935[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_17889);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e17916;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__17936 = state_17889;
state_17889 = G__17936;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_17889){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_17889);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done))
})();var state__11629__auto__ = (function (){var statearr_17918 = f__11628__auto__.call(null);(statearr_17918[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___17919);
return statearr_17918;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___17919,chs__$1,out,cnt,rets,dchan,dctr,done))
);
return out;
});
map = function(f,chs,buf_or_n){
switch(arguments.length){
case 2:
return map__2.call(this,f,chs);
case 3:
return map__3.call(this,f,chs,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
map.cljs$core$IFn$_invoke$arity$2 = map__2;
map.cljs$core$IFn$_invoke$arity$3 = map__3;
return map;
})()
;
/**
* Takes a collection of source channels and returns a channel which
* contains all values taken from them. The returned channel will be
* unbuffered by default, or a buf-or-n can be supplied. The channel
* will close after all the source channels have closed.
*/
cljs.core.async.merge = (function() {
var merge = null;
var merge__1 = (function (chs){return merge.call(null,chs,null);
});
var merge__2 = (function (chs,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___18044 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___18044,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___18044,out){
return (function (state_18020){var state_val_18021 = (state_18020[1]);if((state_val_18021 === 7))
{var inst_18000 = (state_18020[7]);var inst_17999 = (state_18020[8]);var inst_17999__$1 = (state_18020[2]);var inst_18000__$1 = cljs.core.nth.call(null,inst_17999__$1,0,null);var inst_18001 = cljs.core.nth.call(null,inst_17999__$1,1,null);var inst_18002 = (inst_18000__$1 == null);var state_18020__$1 = (function (){var statearr_18022 = state_18020;(statearr_18022[9] = inst_18001);
(statearr_18022[7] = inst_18000__$1);
(statearr_18022[8] = inst_17999__$1);
return statearr_18022;
})();if(cljs.core.truth_(inst_18002))
{var statearr_18023_18045 = state_18020__$1;(statearr_18023_18045[1] = 8);
} else
{var statearr_18024_18046 = state_18020__$1;(statearr_18024_18046[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 1))
{var inst_17991 = cljs.core.vec.call(null,chs);var inst_17992 = inst_17991;var state_18020__$1 = (function (){var statearr_18025 = state_18020;(statearr_18025[10] = inst_17992);
return statearr_18025;
})();var statearr_18026_18047 = state_18020__$1;(statearr_18026_18047[2] = null);
(statearr_18026_18047[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 4))
{var inst_17992 = (state_18020[10]);var state_18020__$1 = state_18020;return cljs.core.async.impl.ioc_helpers.ioc_alts_BANG_.call(null,state_18020__$1,7,inst_17992);
} else
{if((state_val_18021 === 6))
{var inst_18016 = (state_18020[2]);var state_18020__$1 = state_18020;var statearr_18027_18048 = state_18020__$1;(statearr_18027_18048[2] = inst_18016);
(statearr_18027_18048[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 3))
{var inst_18018 = (state_18020[2]);var state_18020__$1 = state_18020;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_18020__$1,inst_18018);
} else
{if((state_val_18021 === 2))
{var inst_17992 = (state_18020[10]);var inst_17994 = cljs.core.count.call(null,inst_17992);var inst_17995 = (inst_17994 > 0);var state_18020__$1 = state_18020;if(cljs.core.truth_(inst_17995))
{var statearr_18029_18049 = state_18020__$1;(statearr_18029_18049[1] = 4);
} else
{var statearr_18030_18050 = state_18020__$1;(statearr_18030_18050[1] = 5);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 11))
{var inst_17992 = (state_18020[10]);var inst_18009 = (state_18020[2]);var tmp18028 = inst_17992;var inst_17992__$1 = tmp18028;var state_18020__$1 = (function (){var statearr_18031 = state_18020;(statearr_18031[10] = inst_17992__$1);
(statearr_18031[11] = inst_18009);
return statearr_18031;
})();var statearr_18032_18051 = state_18020__$1;(statearr_18032_18051[2] = null);
(statearr_18032_18051[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 9))
{var inst_18000 = (state_18020[7]);var state_18020__$1 = state_18020;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18020__$1,11,out,inst_18000);
} else
{if((state_val_18021 === 5))
{var inst_18014 = cljs.core.async.close_BANG_.call(null,out);var state_18020__$1 = state_18020;var statearr_18033_18052 = state_18020__$1;(statearr_18033_18052[2] = inst_18014);
(statearr_18033_18052[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 10))
{var inst_18012 = (state_18020[2]);var state_18020__$1 = state_18020;var statearr_18034_18053 = state_18020__$1;(statearr_18034_18053[2] = inst_18012);
(statearr_18034_18053[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18021 === 8))
{var inst_18001 = (state_18020[9]);var inst_17992 = (state_18020[10]);var inst_18000 = (state_18020[7]);var inst_17999 = (state_18020[8]);var inst_18004 = (function (){var c = inst_18001;var v = inst_18000;var vec__17997 = inst_17999;var cs = inst_17992;return ((function (c,v,vec__17997,cs,inst_18001,inst_17992,inst_18000,inst_17999,state_val_18021,c__11627__auto___18044,out){
return (function (p1__17937_SHARP_){return cljs.core.not_EQ_.call(null,c,p1__17937_SHARP_);
});
;})(c,v,vec__17997,cs,inst_18001,inst_17992,inst_18000,inst_17999,state_val_18021,c__11627__auto___18044,out))
})();var inst_18005 = cljs.core.filterv.call(null,inst_18004,inst_17992);var inst_17992__$1 = inst_18005;var state_18020__$1 = (function (){var statearr_18035 = state_18020;(statearr_18035[10] = inst_17992__$1);
return statearr_18035;
})();var statearr_18036_18054 = state_18020__$1;(statearr_18036_18054[2] = null);
(statearr_18036_18054[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___18044,out))
;return ((function (switch__11563__auto__,c__11627__auto___18044,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_18040 = [null,null,null,null,null,null,null,null,null,null,null,null];(statearr_18040[0] = state_machine__11564__auto__);
(statearr_18040[1] = 1);
return statearr_18040;
});
var state_machine__11564__auto____1 = (function (state_18020){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_18020);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e18041){if((e18041 instanceof Object))
{var ex__11567__auto__ = e18041;var statearr_18042_18055 = state_18020;(statearr_18042_18055[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_18020);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18041;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__18056 = state_18020;
state_18020 = G__18056;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_18020){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_18020);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___18044,out))
})();var state__11629__auto__ = (function (){var statearr_18043 = f__11628__auto__.call(null);(statearr_18043[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___18044);
return statearr_18043;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___18044,out))
);
return out;
});
merge = function(chs,buf_or_n){
switch(arguments.length){
case 1:
return merge__1.call(this,chs);
case 2:
return merge__2.call(this,chs,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
merge.cljs$core$IFn$_invoke$arity$1 = merge__1;
merge.cljs$core$IFn$_invoke$arity$2 = merge__2;
return merge;
})()
;
/**
* Returns a channel containing the single (collection) result of the
* items taken from the channel conjoined to the supplied
* collection. ch must close before into produces a result.
*/
cljs.core.async.into = (function into(coll,ch){return cljs.core.async.reduce.call(null,cljs.core.conj,coll,ch);
});
/**
* Returns a channel that will return, at most, n items from ch. After n items
* have been returned, or ch has been closed, the return chanel will close.
* 
* The output channel is unbuffered by default, unless buf-or-n is given.
*/
cljs.core.async.take = (function() {
var take = null;
var take__2 = (function (n,ch){return take.call(null,n,ch,null);
});
var take__3 = (function (n,ch,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___18149 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___18149,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___18149,out){
return (function (state_18126){var state_val_18127 = (state_18126[1]);if((state_val_18127 === 7))
{var inst_18108 = (state_18126[7]);var inst_18108__$1 = (state_18126[2]);var inst_18109 = (inst_18108__$1 == null);var inst_18110 = cljs.core.not.call(null,inst_18109);var state_18126__$1 = (function (){var statearr_18128 = state_18126;(statearr_18128[7] = inst_18108__$1);
return statearr_18128;
})();if(inst_18110)
{var statearr_18129_18150 = state_18126__$1;(statearr_18129_18150[1] = 8);
} else
{var statearr_18130_18151 = state_18126__$1;(statearr_18130_18151[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 1))
{var inst_18103 = 0;var state_18126__$1 = (function (){var statearr_18131 = state_18126;(statearr_18131[8] = inst_18103);
return statearr_18131;
})();var statearr_18132_18152 = state_18126__$1;(statearr_18132_18152[2] = null);
(statearr_18132_18152[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 4))
{var state_18126__$1 = state_18126;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_18126__$1,7,ch);
} else
{if((state_val_18127 === 6))
{var inst_18121 = (state_18126[2]);var state_18126__$1 = state_18126;var statearr_18133_18153 = state_18126__$1;(statearr_18133_18153[2] = inst_18121);
(statearr_18133_18153[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 3))
{var inst_18123 = (state_18126[2]);var inst_18124 = cljs.core.async.close_BANG_.call(null,out);var state_18126__$1 = (function (){var statearr_18134 = state_18126;(statearr_18134[9] = inst_18123);
return statearr_18134;
})();return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_18126__$1,inst_18124);
} else
{if((state_val_18127 === 2))
{var inst_18103 = (state_18126[8]);var inst_18105 = (inst_18103 < n);var state_18126__$1 = state_18126;if(cljs.core.truth_(inst_18105))
{var statearr_18135_18154 = state_18126__$1;(statearr_18135_18154[1] = 4);
} else
{var statearr_18136_18155 = state_18126__$1;(statearr_18136_18155[1] = 5);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 11))
{var inst_18103 = (state_18126[8]);var inst_18113 = (state_18126[2]);var inst_18114 = (inst_18103 + 1);var inst_18103__$1 = inst_18114;var state_18126__$1 = (function (){var statearr_18137 = state_18126;(statearr_18137[10] = inst_18113);
(statearr_18137[8] = inst_18103__$1);
return statearr_18137;
})();var statearr_18138_18156 = state_18126__$1;(statearr_18138_18156[2] = null);
(statearr_18138_18156[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 9))
{var state_18126__$1 = state_18126;var statearr_18139_18157 = state_18126__$1;(statearr_18139_18157[2] = null);
(statearr_18139_18157[1] = 10);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 5))
{var state_18126__$1 = state_18126;var statearr_18140_18158 = state_18126__$1;(statearr_18140_18158[2] = null);
(statearr_18140_18158[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 10))
{var inst_18118 = (state_18126[2]);var state_18126__$1 = state_18126;var statearr_18141_18159 = state_18126__$1;(statearr_18141_18159[2] = inst_18118);
(statearr_18141_18159[1] = 6);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18127 === 8))
{var inst_18108 = (state_18126[7]);var state_18126__$1 = state_18126;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18126__$1,11,out,inst_18108);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___18149,out))
;return ((function (switch__11563__auto__,c__11627__auto___18149,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_18145 = [null,null,null,null,null,null,null,null,null,null,null];(statearr_18145[0] = state_machine__11564__auto__);
(statearr_18145[1] = 1);
return statearr_18145;
});
var state_machine__11564__auto____1 = (function (state_18126){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_18126);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e18146){if((e18146 instanceof Object))
{var ex__11567__auto__ = e18146;var statearr_18147_18160 = state_18126;(statearr_18147_18160[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_18126);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18146;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__18161 = state_18126;
state_18126 = G__18161;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_18126){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_18126);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___18149,out))
})();var state__11629__auto__ = (function (){var statearr_18148 = f__11628__auto__.call(null);(statearr_18148[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___18149);
return statearr_18148;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___18149,out))
);
return out;
});
take = function(n,ch,buf_or_n){
switch(arguments.length){
case 2:
return take__2.call(this,n,ch);
case 3:
return take__3.call(this,n,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
take.cljs$core$IFn$_invoke$arity$2 = take__2;
take.cljs$core$IFn$_invoke$arity$3 = take__3;
return take;
})()
;
/**
* Returns a channel that will contain values from ch. Consecutive duplicate
* values will be dropped.
* 
* The output channel is unbuffered by default, unless buf-or-n is given.
*/
cljs.core.async.unique = (function() {
var unique = null;
var unique__1 = (function (ch){return unique.call(null,ch,null);
});
var unique__2 = (function (ch,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___18258 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___18258,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___18258,out){
return (function (state_18233){var state_val_18234 = (state_18233[1]);if((state_val_18234 === 7))
{var inst_18228 = (state_18233[2]);var state_18233__$1 = state_18233;var statearr_18235_18259 = state_18233__$1;(statearr_18235_18259[2] = inst_18228);
(statearr_18235_18259[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 1))
{var inst_18210 = null;var state_18233__$1 = (function (){var statearr_18236 = state_18233;(statearr_18236[7] = inst_18210);
return statearr_18236;
})();var statearr_18237_18260 = state_18233__$1;(statearr_18237_18260[2] = null);
(statearr_18237_18260[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 4))
{var inst_18213 = (state_18233[8]);var inst_18213__$1 = (state_18233[2]);var inst_18214 = (inst_18213__$1 == null);var inst_18215 = cljs.core.not.call(null,inst_18214);var state_18233__$1 = (function (){var statearr_18238 = state_18233;(statearr_18238[8] = inst_18213__$1);
return statearr_18238;
})();if(inst_18215)
{var statearr_18239_18261 = state_18233__$1;(statearr_18239_18261[1] = 5);
} else
{var statearr_18240_18262 = state_18233__$1;(statearr_18240_18262[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 6))
{var state_18233__$1 = state_18233;var statearr_18241_18263 = state_18233__$1;(statearr_18241_18263[2] = null);
(statearr_18241_18263[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 3))
{var inst_18230 = (state_18233[2]);var inst_18231 = cljs.core.async.close_BANG_.call(null,out);var state_18233__$1 = (function (){var statearr_18242 = state_18233;(statearr_18242[9] = inst_18230);
return statearr_18242;
})();return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_18233__$1,inst_18231);
} else
{if((state_val_18234 === 2))
{var state_18233__$1 = state_18233;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_18233__$1,4,ch);
} else
{if((state_val_18234 === 11))
{var inst_18213 = (state_18233[8]);var inst_18222 = (state_18233[2]);var inst_18210 = inst_18213;var state_18233__$1 = (function (){var statearr_18243 = state_18233;(statearr_18243[7] = inst_18210);
(statearr_18243[10] = inst_18222);
return statearr_18243;
})();var statearr_18244_18264 = state_18233__$1;(statearr_18244_18264[2] = null);
(statearr_18244_18264[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 9))
{var inst_18213 = (state_18233[8]);var state_18233__$1 = state_18233;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18233__$1,11,out,inst_18213);
} else
{if((state_val_18234 === 5))
{var inst_18213 = (state_18233[8]);var inst_18210 = (state_18233[7]);var inst_18217 = cljs.core._EQ_.call(null,inst_18213,inst_18210);var state_18233__$1 = state_18233;if(inst_18217)
{var statearr_18246_18265 = state_18233__$1;(statearr_18246_18265[1] = 8);
} else
{var statearr_18247_18266 = state_18233__$1;(statearr_18247_18266[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 10))
{var inst_18225 = (state_18233[2]);var state_18233__$1 = state_18233;var statearr_18248_18267 = state_18233__$1;(statearr_18248_18267[2] = inst_18225);
(statearr_18248_18267[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18234 === 8))
{var inst_18210 = (state_18233[7]);var tmp18245 = inst_18210;var inst_18210__$1 = tmp18245;var state_18233__$1 = (function (){var statearr_18249 = state_18233;(statearr_18249[7] = inst_18210__$1);
return statearr_18249;
})();var statearr_18250_18268 = state_18233__$1;(statearr_18250_18268[2] = null);
(statearr_18250_18268[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___18258,out))
;return ((function (switch__11563__auto__,c__11627__auto___18258,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_18254 = [null,null,null,null,null,null,null,null,null,null,null];(statearr_18254[0] = state_machine__11564__auto__);
(statearr_18254[1] = 1);
return statearr_18254;
});
var state_machine__11564__auto____1 = (function (state_18233){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_18233);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e18255){if((e18255 instanceof Object))
{var ex__11567__auto__ = e18255;var statearr_18256_18269 = state_18233;(statearr_18256_18269[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_18233);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18255;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__18270 = state_18233;
state_18233 = G__18270;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_18233){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_18233);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___18258,out))
})();var state__11629__auto__ = (function (){var statearr_18257 = f__11628__auto__.call(null);(statearr_18257[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___18258);
return statearr_18257;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___18258,out))
);
return out;
});
unique = function(ch,buf_or_n){
switch(arguments.length){
case 1:
return unique__1.call(this,ch);
case 2:
return unique__2.call(this,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
unique.cljs$core$IFn$_invoke$arity$1 = unique__1;
unique.cljs$core$IFn$_invoke$arity$2 = unique__2;
return unique;
})()
;
/**
* Returns a channel that will contain vectors of n items taken from ch. The
* final vector in the return channel may be smaller than n if ch closed before
* the vector could be completely filled.
* 
* The output channel is unbuffered by default, unless buf-or-n is given
*/
cljs.core.async.partition = (function() {
var partition = null;
var partition__2 = (function (n,ch){return partition.call(null,n,ch,null);
});
var partition__3 = (function (n,ch,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___18405 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___18405,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___18405,out){
return (function (state_18375){var state_val_18376 = (state_18375[1]);if((state_val_18376 === 7))
{var inst_18371 = (state_18375[2]);var state_18375__$1 = state_18375;var statearr_18377_18406 = state_18375__$1;(statearr_18377_18406[2] = inst_18371);
(statearr_18377_18406[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 1))
{var inst_18338 = (new Array(n));var inst_18339 = inst_18338;var inst_18340 = 0;var state_18375__$1 = (function (){var statearr_18378 = state_18375;(statearr_18378[7] = inst_18339);
(statearr_18378[8] = inst_18340);
return statearr_18378;
})();var statearr_18379_18407 = state_18375__$1;(statearr_18379_18407[2] = null);
(statearr_18379_18407[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 4))
{var inst_18343 = (state_18375[9]);var inst_18343__$1 = (state_18375[2]);var inst_18344 = (inst_18343__$1 == null);var inst_18345 = cljs.core.not.call(null,inst_18344);var state_18375__$1 = (function (){var statearr_18380 = state_18375;(statearr_18380[9] = inst_18343__$1);
return statearr_18380;
})();if(inst_18345)
{var statearr_18381_18408 = state_18375__$1;(statearr_18381_18408[1] = 5);
} else
{var statearr_18382_18409 = state_18375__$1;(statearr_18382_18409[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 15))
{var inst_18365 = (state_18375[2]);var state_18375__$1 = state_18375;var statearr_18383_18410 = state_18375__$1;(statearr_18383_18410[2] = inst_18365);
(statearr_18383_18410[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 13))
{var state_18375__$1 = state_18375;var statearr_18384_18411 = state_18375__$1;(statearr_18384_18411[2] = null);
(statearr_18384_18411[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 6))
{var inst_18340 = (state_18375[8]);var inst_18361 = (inst_18340 > 0);var state_18375__$1 = state_18375;if(cljs.core.truth_(inst_18361))
{var statearr_18385_18412 = state_18375__$1;(statearr_18385_18412[1] = 12);
} else
{var statearr_18386_18413 = state_18375__$1;(statearr_18386_18413[1] = 13);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 3))
{var inst_18373 = (state_18375[2]);var state_18375__$1 = state_18375;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_18375__$1,inst_18373);
} else
{if((state_val_18376 === 12))
{var inst_18339 = (state_18375[7]);var inst_18363 = cljs.core.vec.call(null,inst_18339);var state_18375__$1 = state_18375;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18375__$1,15,out,inst_18363);
} else
{if((state_val_18376 === 2))
{var state_18375__$1 = state_18375;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_18375__$1,4,ch);
} else
{if((state_val_18376 === 11))
{var inst_18355 = (state_18375[2]);var inst_18356 = (new Array(n));var inst_18339 = inst_18356;var inst_18340 = 0;var state_18375__$1 = (function (){var statearr_18387 = state_18375;(statearr_18387[10] = inst_18355);
(statearr_18387[7] = inst_18339);
(statearr_18387[8] = inst_18340);
return statearr_18387;
})();var statearr_18388_18414 = state_18375__$1;(statearr_18388_18414[2] = null);
(statearr_18388_18414[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 9))
{var inst_18339 = (state_18375[7]);var inst_18353 = cljs.core.vec.call(null,inst_18339);var state_18375__$1 = state_18375;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18375__$1,11,out,inst_18353);
} else
{if((state_val_18376 === 5))
{var inst_18343 = (state_18375[9]);var inst_18339 = (state_18375[7]);var inst_18340 = (state_18375[8]);var inst_18348 = (state_18375[11]);var inst_18347 = (inst_18339[inst_18340] = inst_18343);var inst_18348__$1 = (inst_18340 + 1);var inst_18349 = (inst_18348__$1 < n);var state_18375__$1 = (function (){var statearr_18389 = state_18375;(statearr_18389[12] = inst_18347);
(statearr_18389[11] = inst_18348__$1);
return statearr_18389;
})();if(cljs.core.truth_(inst_18349))
{var statearr_18390_18415 = state_18375__$1;(statearr_18390_18415[1] = 8);
} else
{var statearr_18391_18416 = state_18375__$1;(statearr_18391_18416[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 14))
{var inst_18368 = (state_18375[2]);var inst_18369 = cljs.core.async.close_BANG_.call(null,out);var state_18375__$1 = (function (){var statearr_18393 = state_18375;(statearr_18393[13] = inst_18368);
return statearr_18393;
})();var statearr_18394_18417 = state_18375__$1;(statearr_18394_18417[2] = inst_18369);
(statearr_18394_18417[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 10))
{var inst_18359 = (state_18375[2]);var state_18375__$1 = state_18375;var statearr_18395_18418 = state_18375__$1;(statearr_18395_18418[2] = inst_18359);
(statearr_18395_18418[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18376 === 8))
{var inst_18339 = (state_18375[7]);var inst_18348 = (state_18375[11]);var tmp18392 = inst_18339;var inst_18339__$1 = tmp18392;var inst_18340 = inst_18348;var state_18375__$1 = (function (){var statearr_18396 = state_18375;(statearr_18396[7] = inst_18339__$1);
(statearr_18396[8] = inst_18340);
return statearr_18396;
})();var statearr_18397_18419 = state_18375__$1;(statearr_18397_18419[2] = null);
(statearr_18397_18419[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___18405,out))
;return ((function (switch__11563__auto__,c__11627__auto___18405,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_18401 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_18401[0] = state_machine__11564__auto__);
(statearr_18401[1] = 1);
return statearr_18401;
});
var state_machine__11564__auto____1 = (function (state_18375){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_18375);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e18402){if((e18402 instanceof Object))
{var ex__11567__auto__ = e18402;var statearr_18403_18420 = state_18375;(statearr_18403_18420[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_18375);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18402;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__18421 = state_18375;
state_18375 = G__18421;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_18375){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_18375);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___18405,out))
})();var state__11629__auto__ = (function (){var statearr_18404 = f__11628__auto__.call(null);(statearr_18404[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___18405);
return statearr_18404;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___18405,out))
);
return out;
});
partition = function(n,ch,buf_or_n){
switch(arguments.length){
case 2:
return partition__2.call(this,n,ch);
case 3:
return partition__3.call(this,n,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
partition.cljs$core$IFn$_invoke$arity$2 = partition__2;
partition.cljs$core$IFn$_invoke$arity$3 = partition__3;
return partition;
})()
;
/**
* Returns a channel that will contain vectors of items taken from ch. New
* vectors will be created whenever (f itm) returns a value that differs from
* the previous item's (f itm).
* 
* The output channel is unbuffered, unless buf-or-n is given
*/
cljs.core.async.partition_by = (function() {
var partition_by = null;
var partition_by__2 = (function (f,ch){return partition_by.call(null,f,ch,null);
});
var partition_by__3 = (function (f,ch,buf_or_n){var out = cljs.core.async.chan.call(null,buf_or_n);var c__11627__auto___18564 = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto___18564,out){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto___18564,out){
return (function (state_18534){var state_val_18535 = (state_18534[1]);if((state_val_18535 === 7))
{var inst_18530 = (state_18534[2]);var state_18534__$1 = state_18534;var statearr_18536_18565 = state_18534__$1;(statearr_18536_18565[2] = inst_18530);
(statearr_18536_18565[1] = 3);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 1))
{var inst_18493 = [];var inst_18494 = inst_18493;var inst_18495 = new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",4382193538);var state_18534__$1 = (function (){var statearr_18537 = state_18534;(statearr_18537[7] = inst_18494);
(statearr_18537[8] = inst_18495);
return statearr_18537;
})();var statearr_18538_18566 = state_18534__$1;(statearr_18538_18566[2] = null);
(statearr_18538_18566[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 4))
{var inst_18498 = (state_18534[9]);var inst_18498__$1 = (state_18534[2]);var inst_18499 = (inst_18498__$1 == null);var inst_18500 = cljs.core.not.call(null,inst_18499);var state_18534__$1 = (function (){var statearr_18539 = state_18534;(statearr_18539[9] = inst_18498__$1);
return statearr_18539;
})();if(inst_18500)
{var statearr_18540_18567 = state_18534__$1;(statearr_18540_18567[1] = 5);
} else
{var statearr_18541_18568 = state_18534__$1;(statearr_18541_18568[1] = 6);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 15))
{var inst_18524 = (state_18534[2]);var state_18534__$1 = state_18534;var statearr_18542_18569 = state_18534__$1;(statearr_18542_18569[2] = inst_18524);
(statearr_18542_18569[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 13))
{var state_18534__$1 = state_18534;var statearr_18543_18570 = state_18534__$1;(statearr_18543_18570[2] = null);
(statearr_18543_18570[1] = 14);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 6))
{var inst_18494 = (state_18534[7]);var inst_18519 = inst_18494.length;var inst_18520 = (inst_18519 > 0);var state_18534__$1 = state_18534;if(cljs.core.truth_(inst_18520))
{var statearr_18544_18571 = state_18534__$1;(statearr_18544_18571[1] = 12);
} else
{var statearr_18545_18572 = state_18534__$1;(statearr_18545_18572[1] = 13);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 3))
{var inst_18532 = (state_18534[2]);var state_18534__$1 = state_18534;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_18534__$1,inst_18532);
} else
{if((state_val_18535 === 12))
{var inst_18494 = (state_18534[7]);var inst_18522 = cljs.core.vec.call(null,inst_18494);var state_18534__$1 = state_18534;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18534__$1,15,out,inst_18522);
} else
{if((state_val_18535 === 2))
{var state_18534__$1 = state_18534;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_18534__$1,4,ch);
} else
{if((state_val_18535 === 11))
{var inst_18502 = (state_18534[10]);var inst_18498 = (state_18534[9]);var inst_18512 = (state_18534[2]);var inst_18513 = [];var inst_18514 = inst_18513.push(inst_18498);var inst_18494 = inst_18513;var inst_18495 = inst_18502;var state_18534__$1 = (function (){var statearr_18546 = state_18534;(statearr_18546[7] = inst_18494);
(statearr_18546[11] = inst_18514);
(statearr_18546[8] = inst_18495);
(statearr_18546[12] = inst_18512);
return statearr_18546;
})();var statearr_18547_18573 = state_18534__$1;(statearr_18547_18573[2] = null);
(statearr_18547_18573[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 9))
{var inst_18494 = (state_18534[7]);var inst_18510 = cljs.core.vec.call(null,inst_18494);var state_18534__$1 = state_18534;return cljs.core.async.impl.ioc_helpers.put_BANG_.call(null,state_18534__$1,11,out,inst_18510);
} else
{if((state_val_18535 === 5))
{var inst_18502 = (state_18534[10]);var inst_18498 = (state_18534[9]);var inst_18495 = (state_18534[8]);var inst_18502__$1 = f.call(null,inst_18498);var inst_18503 = cljs.core._EQ_.call(null,inst_18502__$1,inst_18495);var inst_18504 = cljs.core.keyword_identical_QMARK_.call(null,inst_18495,new cljs.core.Keyword("cljs.core.async","nothing","cljs.core.async/nothing",4382193538));var inst_18505 = (inst_18503) || (inst_18504);var state_18534__$1 = (function (){var statearr_18548 = state_18534;(statearr_18548[10] = inst_18502__$1);
return statearr_18548;
})();if(cljs.core.truth_(inst_18505))
{var statearr_18549_18574 = state_18534__$1;(statearr_18549_18574[1] = 8);
} else
{var statearr_18550_18575 = state_18534__$1;(statearr_18550_18575[1] = 9);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 14))
{var inst_18527 = (state_18534[2]);var inst_18528 = cljs.core.async.close_BANG_.call(null,out);var state_18534__$1 = (function (){var statearr_18552 = state_18534;(statearr_18552[13] = inst_18527);
return statearr_18552;
})();var statearr_18553_18576 = state_18534__$1;(statearr_18553_18576[2] = inst_18528);
(statearr_18553_18576[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 10))
{var inst_18517 = (state_18534[2]);var state_18534__$1 = state_18534;var statearr_18554_18577 = state_18534__$1;(statearr_18554_18577[2] = inst_18517);
(statearr_18554_18577[1] = 7);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_18535 === 8))
{var inst_18502 = (state_18534[10]);var inst_18494 = (state_18534[7]);var inst_18498 = (state_18534[9]);var inst_18507 = inst_18494.push(inst_18498);var tmp18551 = inst_18494;var inst_18494__$1 = tmp18551;var inst_18495 = inst_18502;var state_18534__$1 = (function (){var statearr_18555 = state_18534;(statearr_18555[7] = inst_18494__$1);
(statearr_18555[8] = inst_18495);
(statearr_18555[14] = inst_18507);
return statearr_18555;
})();var statearr_18556_18578 = state_18534__$1;(statearr_18556_18578[2] = null);
(statearr_18556_18578[1] = 2);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{return null;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
});})(c__11627__auto___18564,out))
;return ((function (switch__11563__auto__,c__11627__auto___18564,out){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_18560 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];(statearr_18560[0] = state_machine__11564__auto__);
(statearr_18560[1] = 1);
return statearr_18560;
});
var state_machine__11564__auto____1 = (function (state_18534){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_18534);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e18561){if((e18561 instanceof Object))
{var ex__11567__auto__ = e18561;var statearr_18562_18579 = state_18534;(statearr_18562_18579[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_18534);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e18561;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__18580 = state_18534;
state_18534 = G__18580;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_18534){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_18534);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto___18564,out))
})();var state__11629__auto__ = (function (){var statearr_18563 = f__11628__auto__.call(null);(statearr_18563[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto___18564);
return statearr_18563;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto___18564,out))
);
return out;
});
partition_by = function(f,ch,buf_or_n){
switch(arguments.length){
case 2:
return partition_by__2.call(this,f,ch);
case 3:
return partition_by__3.call(this,f,ch,buf_or_n);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
partition_by.cljs$core$IFn$_invoke$arity$2 = partition_by__2;
partition_by.cljs$core$IFn$_invoke$arity$3 = partition_by__3;
return partition_by;
})()
;
