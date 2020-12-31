goog.provide('shadow.dom');
shadow.dom.transition_supported_QMARK_ = (((typeof window !== 'undefined'))?goog.style.transition.isSupported():null);

/**
 * @interface
 */
shadow.dom.IElement = function(){};

var shadow$dom$IElement$_to_dom$dyn_43290 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (shadow.dom._to_dom[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (shadow.dom._to_dom["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IElement.-to-dom",this$);
}
}
});
shadow.dom._to_dom = (function shadow$dom$_to_dom(this$){
if((((!((this$ == null)))) && ((!((this$.shadow$dom$IElement$_to_dom$arity$1 == null)))))){
return this$.shadow$dom$IElement$_to_dom$arity$1(this$);
} else {
return shadow$dom$IElement$_to_dom$dyn_43290(this$);
}
});


/**
 * @interface
 */
shadow.dom.SVGElement = function(){};

var shadow$dom$SVGElement$_to_svg$dyn_43292 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (shadow.dom._to_svg[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (shadow.dom._to_svg["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("SVGElement.-to-svg",this$);
}
}
});
shadow.dom._to_svg = (function shadow$dom$_to_svg(this$){
if((((!((this$ == null)))) && ((!((this$.shadow$dom$SVGElement$_to_svg$arity$1 == null)))))){
return this$.shadow$dom$SVGElement$_to_svg$arity$1(this$);
} else {
return shadow$dom$SVGElement$_to_svg$dyn_43292(this$);
}
});

shadow.dom.lazy_native_coll_seq = (function shadow$dom$lazy_native_coll_seq(coll,idx){
if((idx < coll.length)){
return (new cljs.core.LazySeq(null,(function (){
return cljs.core.cons((coll[idx]),(function (){var G__42354 = coll;
var G__42355 = (idx + (1));
return (shadow.dom.lazy_native_coll_seq.cljs$core$IFn$_invoke$arity$2 ? shadow.dom.lazy_native_coll_seq.cljs$core$IFn$_invoke$arity$2(G__42354,G__42355) : shadow.dom.lazy_native_coll_seq.call(null,G__42354,G__42355));
})());
}),null,null));
} else {
return null;
}
});

/**
* @constructor
 * @implements {cljs.core.IIndexed}
 * @implements {cljs.core.ICounted}
 * @implements {cljs.core.ISeqable}
 * @implements {cljs.core.IDeref}
 * @implements {shadow.dom.IElement}
*/
shadow.dom.NativeColl = (function (coll){
this.coll = coll;
this.cljs$lang$protocol_mask$partition0$ = 8421394;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(shadow.dom.NativeColl.prototype.cljs$core$IDeref$_deref$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return self__.coll;
}));

(shadow.dom.NativeColl.prototype.cljs$core$IIndexed$_nth$arity$2 = (function (this$,n){
var self__ = this;
var this$__$1 = this;
return (self__.coll[n]);
}));

(shadow.dom.NativeColl.prototype.cljs$core$IIndexed$_nth$arity$3 = (function (this$,n,not_found){
var self__ = this;
var this$__$1 = this;
var or__4126__auto__ = (self__.coll[n]);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return not_found;
}
}));

(shadow.dom.NativeColl.prototype.cljs$core$ICounted$_count$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return self__.coll.length;
}));

(shadow.dom.NativeColl.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return shadow.dom.lazy_native_coll_seq(self__.coll,(0));
}));

(shadow.dom.NativeColl.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(shadow.dom.NativeColl.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return self__.coll;
}));

(shadow.dom.NativeColl.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"coll","coll",-1006698606,null)], null);
}));

(shadow.dom.NativeColl.cljs$lang$type = true);

(shadow.dom.NativeColl.cljs$lang$ctorStr = "shadow.dom/NativeColl");

(shadow.dom.NativeColl.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"shadow.dom/NativeColl");
}));

/**
 * Positional factory function for shadow.dom/NativeColl.
 */
shadow.dom.__GT_NativeColl = (function shadow$dom$__GT_NativeColl(coll){
return (new shadow.dom.NativeColl(coll));
});

shadow.dom.native_coll = (function shadow$dom$native_coll(coll){
return (new shadow.dom.NativeColl(coll));
});
shadow.dom.dom_node = (function shadow$dom$dom_node(el){
if((el == null)){
return null;
} else {
if((((!((el == null))))?((((false) || ((cljs.core.PROTOCOL_SENTINEL === el.shadow$dom$IElement$))))?true:false):false)){
return el.shadow$dom$IElement$_to_dom$arity$1(null);
} else {
if(typeof el === 'string'){
return document.createTextNode(el);
} else {
if(typeof el === 'number'){
return document.createTextNode(cljs.core.str.cljs$core$IFn$_invoke$arity$1(el));
} else {
return el;

}
}
}
}
});
shadow.dom.query_one = (function shadow$dom$query_one(var_args){
var G__42383 = arguments.length;
switch (G__42383) {
case 1:
return shadow.dom.query_one.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.query_one.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.query_one.cljs$core$IFn$_invoke$arity$1 = (function (sel){
return document.querySelector(sel);
}));

(shadow.dom.query_one.cljs$core$IFn$_invoke$arity$2 = (function (sel,root){
return shadow.dom.dom_node(root).querySelector(sel);
}));

(shadow.dom.query_one.cljs$lang$maxFixedArity = 2);

shadow.dom.query = (function shadow$dom$query(var_args){
var G__42396 = arguments.length;
switch (G__42396) {
case 1:
return shadow.dom.query.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.query.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.query.cljs$core$IFn$_invoke$arity$1 = (function (sel){
return (new shadow.dom.NativeColl(document.querySelectorAll(sel)));
}));

(shadow.dom.query.cljs$core$IFn$_invoke$arity$2 = (function (sel,root){
return (new shadow.dom.NativeColl(shadow.dom.dom_node(root).querySelectorAll(sel)));
}));

(shadow.dom.query.cljs$lang$maxFixedArity = 2);

shadow.dom.by_id = (function shadow$dom$by_id(var_args){
var G__42404 = arguments.length;
switch (G__42404) {
case 2:
return shadow.dom.by_id.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 1:
return shadow.dom.by_id.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.by_id.cljs$core$IFn$_invoke$arity$2 = (function (id,el){
return shadow.dom.dom_node(el).getElementById(id);
}));

(shadow.dom.by_id.cljs$core$IFn$_invoke$arity$1 = (function (id){
return document.getElementById(id);
}));

(shadow.dom.by_id.cljs$lang$maxFixedArity = 2);

shadow.dom.build = shadow.dom.dom_node;
shadow.dom.ev_stop = (function shadow$dom$ev_stop(var_args){
var G__42409 = arguments.length;
switch (G__42409) {
case 1:
return shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 4:
return shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$1 = (function (e){
if(cljs.core.truth_(e.stopPropagation)){
e.stopPropagation();

e.preventDefault();
} else {
(e.cancelBubble = true);

(e.returnValue = false);
}

return e;
}));

(shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$2 = (function (e,el){
shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$1(e);

return el;
}));

(shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$4 = (function (e,el,scope,owner){
shadow.dom.ev_stop.cljs$core$IFn$_invoke$arity$1(e);

return el;
}));

(shadow.dom.ev_stop.cljs$lang$maxFixedArity = 4);

/**
 * check wether a parent node (or the document) contains the child
 */
shadow.dom.contains_QMARK_ = (function shadow$dom$contains_QMARK_(var_args){
var G__42425 = arguments.length;
switch (G__42425) {
case 1:
return shadow.dom.contains_QMARK_.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.contains_QMARK_.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.contains_QMARK_.cljs$core$IFn$_invoke$arity$1 = (function (el){
return goog.dom.contains(document,shadow.dom.dom_node(el));
}));

(shadow.dom.contains_QMARK_.cljs$core$IFn$_invoke$arity$2 = (function (parent,el){
return goog.dom.contains(shadow.dom.dom_node(parent),shadow.dom.dom_node(el));
}));

(shadow.dom.contains_QMARK_.cljs$lang$maxFixedArity = 2);

shadow.dom.add_class = (function shadow$dom$add_class(el,cls){
return goog.dom.classlist.add(shadow.dom.dom_node(el),cls);
});
shadow.dom.remove_class = (function shadow$dom$remove_class(el,cls){
return goog.dom.classlist.remove(shadow.dom.dom_node(el),cls);
});
shadow.dom.toggle_class = (function shadow$dom$toggle_class(var_args){
var G__42440 = arguments.length;
switch (G__42440) {
case 2:
return shadow.dom.toggle_class.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return shadow.dom.toggle_class.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.toggle_class.cljs$core$IFn$_invoke$arity$2 = (function (el,cls){
return goog.dom.classlist.toggle(shadow.dom.dom_node(el),cls);
}));

(shadow.dom.toggle_class.cljs$core$IFn$_invoke$arity$3 = (function (el,cls,v){
if(cljs.core.truth_(v)){
return shadow.dom.add_class(el,cls);
} else {
return shadow.dom.remove_class(el,cls);
}
}));

(shadow.dom.toggle_class.cljs$lang$maxFixedArity = 3);

shadow.dom.dom_listen = (cljs.core.truth_((function (){var or__4126__auto__ = (!((typeof document !== 'undefined')));
if(or__4126__auto__){
return or__4126__auto__;
} else {
return document.addEventListener;
}
})())?(function shadow$dom$dom_listen_good(el,ev,handler){
return el.addEventListener(ev,handler,false);
}):(function shadow$dom$dom_listen_ie(el,ev,handler){
try{return el.attachEvent(["on",cljs.core.str.cljs$core$IFn$_invoke$arity$1(ev)].join(''),(function (e){
return (handler.cljs$core$IFn$_invoke$arity$2 ? handler.cljs$core$IFn$_invoke$arity$2(e,el) : handler.call(null,e,el));
}));
}catch (e42450){if((e42450 instanceof Object)){
var e = e42450;
return console.log("didnt support attachEvent",el,e);
} else {
throw e42450;

}
}}));
shadow.dom.dom_listen_remove = (cljs.core.truth_((function (){var or__4126__auto__ = (!((typeof document !== 'undefined')));
if(or__4126__auto__){
return or__4126__auto__;
} else {
return document.removeEventListener;
}
})())?(function shadow$dom$dom_listen_remove_good(el,ev,handler){
return el.removeEventListener(ev,handler,false);
}):(function shadow$dom$dom_listen_remove_ie(el,ev,handler){
return el.detachEvent(["on",cljs.core.str.cljs$core$IFn$_invoke$arity$1(ev)].join(''),handler);
}));
shadow.dom.on_query = (function shadow$dom$on_query(root_el,ev,selector,handler){
var seq__42457 = cljs.core.seq(shadow.dom.query.cljs$core$IFn$_invoke$arity$2(selector,root_el));
var chunk__42458 = null;
var count__42459 = (0);
var i__42460 = (0);
while(true){
if((i__42460 < count__42459)){
var el = chunk__42458.cljs$core$IIndexed$_nth$arity$2(null,i__42460);
var handler_43354__$1 = ((function (seq__42457,chunk__42458,count__42459,i__42460,el){
return (function (e){
return (handler.cljs$core$IFn$_invoke$arity$2 ? handler.cljs$core$IFn$_invoke$arity$2(e,el) : handler.call(null,e,el));
});})(seq__42457,chunk__42458,count__42459,i__42460,el))
;
shadow.dom.dom_listen(el,cljs.core.name(ev),handler_43354__$1);


var G__43357 = seq__42457;
var G__43358 = chunk__42458;
var G__43359 = count__42459;
var G__43360 = (i__42460 + (1));
seq__42457 = G__43357;
chunk__42458 = G__43358;
count__42459 = G__43359;
i__42460 = G__43360;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__42457);
if(temp__5735__auto__){
var seq__42457__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__42457__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__42457__$1);
var G__43364 = cljs.core.chunk_rest(seq__42457__$1);
var G__43365 = c__4556__auto__;
var G__43366 = cljs.core.count(c__4556__auto__);
var G__43367 = (0);
seq__42457 = G__43364;
chunk__42458 = G__43365;
count__42459 = G__43366;
i__42460 = G__43367;
continue;
} else {
var el = cljs.core.first(seq__42457__$1);
var handler_43374__$1 = ((function (seq__42457,chunk__42458,count__42459,i__42460,el,seq__42457__$1,temp__5735__auto__){
return (function (e){
return (handler.cljs$core$IFn$_invoke$arity$2 ? handler.cljs$core$IFn$_invoke$arity$2(e,el) : handler.call(null,e,el));
});})(seq__42457,chunk__42458,count__42459,i__42460,el,seq__42457__$1,temp__5735__auto__))
;
shadow.dom.dom_listen(el,cljs.core.name(ev),handler_43374__$1);


var G__43378 = cljs.core.next(seq__42457__$1);
var G__43379 = null;
var G__43380 = (0);
var G__43381 = (0);
seq__42457 = G__43378;
chunk__42458 = G__43379;
count__42459 = G__43380;
i__42460 = G__43381;
continue;
}
} else {
return null;
}
}
break;
}
});
shadow.dom.on = (function shadow$dom$on(var_args){
var G__42477 = arguments.length;
switch (G__42477) {
case 3:
return shadow.dom.on.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return shadow.dom.on.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.on.cljs$core$IFn$_invoke$arity$3 = (function (el,ev,handler){
return shadow.dom.on.cljs$core$IFn$_invoke$arity$4(el,ev,handler,false);
}));

(shadow.dom.on.cljs$core$IFn$_invoke$arity$4 = (function (el,ev,handler,capture){
if(cljs.core.vector_QMARK_(ev)){
return shadow.dom.on_query(el,cljs.core.first(ev),cljs.core.second(ev),handler);
} else {
var handler__$1 = (function (e){
return (handler.cljs$core$IFn$_invoke$arity$2 ? handler.cljs$core$IFn$_invoke$arity$2(e,el) : handler.call(null,e,el));
});
return shadow.dom.dom_listen(shadow.dom.dom_node(el),cljs.core.name(ev),handler__$1);
}
}));

(shadow.dom.on.cljs$lang$maxFixedArity = 4);

shadow.dom.remove_event_handler = (function shadow$dom$remove_event_handler(el,ev,handler){
return shadow.dom.dom_listen_remove(shadow.dom.dom_node(el),cljs.core.name(ev),handler);
});
shadow.dom.add_event_listeners = (function shadow$dom$add_event_listeners(el,events){
var seq__42483 = cljs.core.seq(events);
var chunk__42484 = null;
var count__42485 = (0);
var i__42486 = (0);
while(true){
if((i__42486 < count__42485)){
var vec__42512 = chunk__42484.cljs$core$IIndexed$_nth$arity$2(null,i__42486);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42512,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42512,(1),null);
shadow.dom.on.cljs$core$IFn$_invoke$arity$3(el,k,v);


var G__43387 = seq__42483;
var G__43388 = chunk__42484;
var G__43389 = count__42485;
var G__43390 = (i__42486 + (1));
seq__42483 = G__43387;
chunk__42484 = G__43388;
count__42485 = G__43389;
i__42486 = G__43390;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__42483);
if(temp__5735__auto__){
var seq__42483__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__42483__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__42483__$1);
var G__43392 = cljs.core.chunk_rest(seq__42483__$1);
var G__43393 = c__4556__auto__;
var G__43394 = cljs.core.count(c__4556__auto__);
var G__43395 = (0);
seq__42483 = G__43392;
chunk__42484 = G__43393;
count__42485 = G__43394;
i__42486 = G__43395;
continue;
} else {
var vec__42521 = cljs.core.first(seq__42483__$1);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42521,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42521,(1),null);
shadow.dom.on.cljs$core$IFn$_invoke$arity$3(el,k,v);


var G__43397 = cljs.core.next(seq__42483__$1);
var G__43398 = null;
var G__43399 = (0);
var G__43400 = (0);
seq__42483 = G__43397;
chunk__42484 = G__43398;
count__42485 = G__43399;
i__42486 = G__43400;
continue;
}
} else {
return null;
}
}
break;
}
});
shadow.dom.set_style = (function shadow$dom$set_style(el,styles){
var dom = shadow.dom.dom_node(el);
var seq__42530 = cljs.core.seq(styles);
var chunk__42531 = null;
var count__42532 = (0);
var i__42533 = (0);
while(true){
if((i__42533 < count__42532)){
var vec__42547 = chunk__42531.cljs$core$IIndexed$_nth$arity$2(null,i__42533);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42547,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42547,(1),null);
goog.style.setStyle(dom,cljs.core.name(k),(((v == null))?"":v));


var G__43403 = seq__42530;
var G__43404 = chunk__42531;
var G__43405 = count__42532;
var G__43406 = (i__42533 + (1));
seq__42530 = G__43403;
chunk__42531 = G__43404;
count__42532 = G__43405;
i__42533 = G__43406;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__42530);
if(temp__5735__auto__){
var seq__42530__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__42530__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__42530__$1);
var G__43410 = cljs.core.chunk_rest(seq__42530__$1);
var G__43411 = c__4556__auto__;
var G__43412 = cljs.core.count(c__4556__auto__);
var G__43413 = (0);
seq__42530 = G__43410;
chunk__42531 = G__43411;
count__42532 = G__43412;
i__42533 = G__43413;
continue;
} else {
var vec__42553 = cljs.core.first(seq__42530__$1);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42553,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42553,(1),null);
goog.style.setStyle(dom,cljs.core.name(k),(((v == null))?"":v));


var G__43414 = cljs.core.next(seq__42530__$1);
var G__43415 = null;
var G__43416 = (0);
var G__43417 = (0);
seq__42530 = G__43414;
chunk__42531 = G__43415;
count__42532 = G__43416;
i__42533 = G__43417;
continue;
}
} else {
return null;
}
}
break;
}
});
shadow.dom.set_attr_STAR_ = (function shadow$dom$set_attr_STAR_(el,key,value){
var G__42556_43419 = key;
var G__42556_43420__$1 = (((G__42556_43419 instanceof cljs.core.Keyword))?G__42556_43419.fqn:null);
switch (G__42556_43420__$1) {
case "id":
(el.id = cljs.core.str.cljs$core$IFn$_invoke$arity$1(value));

break;
case "class":
(el.className = cljs.core.str.cljs$core$IFn$_invoke$arity$1(value));

break;
case "for":
(el.htmlFor = value);

break;
case "cellpadding":
el.setAttribute("cellPadding",value);

break;
case "cellspacing":
el.setAttribute("cellSpacing",value);

break;
case "colspan":
el.setAttribute("colSpan",value);

break;
case "frameborder":
el.setAttribute("frameBorder",value);

break;
case "height":
el.setAttribute("height",value);

break;
case "maxlength":
el.setAttribute("maxLength",value);

break;
case "role":
el.setAttribute("role",value);

break;
case "rowspan":
el.setAttribute("rowSpan",value);

break;
case "type":
el.setAttribute("type",value);

break;
case "usemap":
el.setAttribute("useMap",value);

break;
case "valign":
el.setAttribute("vAlign",value);

break;
case "width":
el.setAttribute("width",value);

break;
case "on":
shadow.dom.add_event_listeners(el,value);

break;
case "style":
if((value == null)){
} else {
if(typeof value === 'string'){
el.setAttribute("style",value);
} else {
if(cljs.core.map_QMARK_(value)){
shadow.dom.set_style(el,value);
} else {
goog.style.setStyle(el,value);

}
}
}

break;
default:
var ks_43426 = cljs.core.name(key);
if(cljs.core.truth_((function (){var or__4126__auto__ = goog.string.startsWith(ks_43426,"data-");
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return goog.string.startsWith(ks_43426,"aria-");
}
})())){
el.setAttribute(ks_43426,value);
} else {
(el[ks_43426] = value);
}

}

return el;
});
shadow.dom.set_attrs = (function shadow$dom$set_attrs(el,attrs){
return cljs.core.reduce_kv((function (el__$1,key,value){
shadow.dom.set_attr_STAR_(el__$1,key,value);

return el__$1;
}),shadow.dom.dom_node(el),attrs);
});
shadow.dom.set_attr = (function shadow$dom$set_attr(el,key,value){
return shadow.dom.set_attr_STAR_(shadow.dom.dom_node(el),key,value);
});
shadow.dom.has_class_QMARK_ = (function shadow$dom$has_class_QMARK_(el,cls){
return goog.dom.classlist.contains(shadow.dom.dom_node(el),cls);
});
shadow.dom.merge_class_string = (function shadow$dom$merge_class_string(current,extra_class){
if(cljs.core.seq(current)){
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(current)," ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(extra_class)].join('');
} else {
return extra_class;
}
});
shadow.dom.parse_tag = (function shadow$dom$parse_tag(spec){
var spec__$1 = cljs.core.name(spec);
var fdot = spec__$1.indexOf(".");
var fhash = spec__$1.indexOf("#");
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((-1),fdot)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((-1),fhash)))){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [spec__$1,null,null], null);
} else {
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((-1),fhash)){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [spec__$1.substring((0),fdot),null,clojure.string.replace(spec__$1.substring((fdot + (1))),/\./," ")], null);
} else {
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((-1),fdot)){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [spec__$1.substring((0),fhash),spec__$1.substring((fhash + (1))),null], null);
} else {
if((fhash > fdot)){
throw ["cant have id after class?",spec__$1].join('');
} else {
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [spec__$1.substring((0),fhash),spec__$1.substring((fhash + (1)),fdot),clojure.string.replace(spec__$1.substring((fdot + (1))),/\./," ")], null);

}
}
}
}
});
shadow.dom.create_dom_node = (function shadow$dom$create_dom_node(tag_def,p__42580){
var map__42581 = p__42580;
var map__42581__$1 = (((((!((map__42581 == null))))?(((((map__42581.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__42581.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__42581):map__42581);
var props = map__42581__$1;
var class$ = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__42581__$1,new cljs.core.Keyword(null,"class","class",-2030961996));
var tag_props = ({});
var vec__42583 = shadow.dom.parse_tag(tag_def);
var tag_name = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42583,(0),null);
var tag_id = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42583,(1),null);
var tag_classes = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42583,(2),null);
if(cljs.core.truth_(tag_id)){
(tag_props["id"] = tag_id);
} else {
}

if(cljs.core.truth_(tag_classes)){
(tag_props["class"] = shadow.dom.merge_class_string(class$,tag_classes));
} else {
}

var G__42586 = goog.dom.createDom(tag_name,tag_props);
shadow.dom.set_attrs(G__42586,cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(props,new cljs.core.Keyword(null,"class","class",-2030961996)));

return G__42586;
});
shadow.dom.append = (function shadow$dom$append(var_args){
var G__42588 = arguments.length;
switch (G__42588) {
case 1:
return shadow.dom.append.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.append.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.append.cljs$core$IFn$_invoke$arity$1 = (function (node){
if(cljs.core.truth_(node)){
var temp__5735__auto__ = shadow.dom.dom_node(node);
if(cljs.core.truth_(temp__5735__auto__)){
var n = temp__5735__auto__;
document.body.appendChild(n);

return n;
} else {
return null;
}
} else {
return null;
}
}));

(shadow.dom.append.cljs$core$IFn$_invoke$arity$2 = (function (el,node){
if(cljs.core.truth_(node)){
var temp__5735__auto__ = shadow.dom.dom_node(node);
if(cljs.core.truth_(temp__5735__auto__)){
var n = temp__5735__auto__;
shadow.dom.dom_node(el).appendChild(n);

return n;
} else {
return null;
}
} else {
return null;
}
}));

(shadow.dom.append.cljs$lang$maxFixedArity = 2);

shadow.dom.destructure_node = (function shadow$dom$destructure_node(create_fn,p__42589){
var vec__42590 = p__42589;
var seq__42591 = cljs.core.seq(vec__42590);
var first__42592 = cljs.core.first(seq__42591);
var seq__42591__$1 = cljs.core.next(seq__42591);
var nn = first__42592;
var first__42592__$1 = cljs.core.first(seq__42591__$1);
var seq__42591__$2 = cljs.core.next(seq__42591__$1);
var np = first__42592__$1;
var nc = seq__42591__$2;
var node = vec__42590;
if((nn instanceof cljs.core.Keyword)){
} else {
throw cljs.core.ex_info.cljs$core$IFn$_invoke$arity$2("invalid dom node",new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"node","node",581201198),node], null));
}

if((((np == null)) && ((nc == null)))){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(function (){var G__42593 = nn;
var G__42594 = cljs.core.PersistentArrayMap.EMPTY;
return (create_fn.cljs$core$IFn$_invoke$arity$2 ? create_fn.cljs$core$IFn$_invoke$arity$2(G__42593,G__42594) : create_fn.call(null,G__42593,G__42594));
})(),cljs.core.List.EMPTY], null);
} else {
if(cljs.core.map_QMARK_(np)){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(create_fn.cljs$core$IFn$_invoke$arity$2 ? create_fn.cljs$core$IFn$_invoke$arity$2(nn,np) : create_fn.call(null,nn,np)),nc], null);
} else {
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(function (){var G__42596 = nn;
var G__42597 = cljs.core.PersistentArrayMap.EMPTY;
return (create_fn.cljs$core$IFn$_invoke$arity$2 ? create_fn.cljs$core$IFn$_invoke$arity$2(G__42596,G__42597) : create_fn.call(null,G__42596,G__42597));
})(),cljs.core.conj.cljs$core$IFn$_invoke$arity$2(nc,np)], null);

}
}
});
shadow.dom.make_dom_node = (function shadow$dom$make_dom_node(structure){
var vec__42598 = shadow.dom.destructure_node(shadow.dom.create_dom_node,structure);
var node = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42598,(0),null);
var node_children = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42598,(1),null);
var seq__42601_43465 = cljs.core.seq(node_children);
var chunk__42602_43466 = null;
var count__42603_43467 = (0);
var i__42604_43468 = (0);
while(true){
if((i__42604_43468 < count__42603_43467)){
var child_struct_43471 = chunk__42602_43466.cljs$core$IIndexed$_nth$arity$2(null,i__42604_43468);
var children_43472 = shadow.dom.dom_node(child_struct_43471);
if(cljs.core.seq_QMARK_(children_43472)){
var seq__42643_43473 = cljs.core.seq(cljs.core.map.cljs$core$IFn$_invoke$arity$2(shadow.dom.dom_node,children_43472));
var chunk__42645_43474 = null;
var count__42646_43475 = (0);
var i__42647_43476 = (0);
while(true){
if((i__42647_43476 < count__42646_43475)){
var child_43477 = chunk__42645_43474.cljs$core$IIndexed$_nth$arity$2(null,i__42647_43476);
if(cljs.core.truth_(child_43477)){
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,child_43477);


var G__43478 = seq__42643_43473;
var G__43479 = chunk__42645_43474;
var G__43480 = count__42646_43475;
var G__43481 = (i__42647_43476 + (1));
seq__42643_43473 = G__43478;
chunk__42645_43474 = G__43479;
count__42646_43475 = G__43480;
i__42647_43476 = G__43481;
continue;
} else {
var G__43482 = seq__42643_43473;
var G__43483 = chunk__42645_43474;
var G__43484 = count__42646_43475;
var G__43485 = (i__42647_43476 + (1));
seq__42643_43473 = G__43482;
chunk__42645_43474 = G__43483;
count__42646_43475 = G__43484;
i__42647_43476 = G__43485;
continue;
}
} else {
var temp__5735__auto___43489 = cljs.core.seq(seq__42643_43473);
if(temp__5735__auto___43489){
var seq__42643_43490__$1 = temp__5735__auto___43489;
if(cljs.core.chunked_seq_QMARK_(seq__42643_43490__$1)){
var c__4556__auto___43493 = cljs.core.chunk_first(seq__42643_43490__$1);
var G__43494 = cljs.core.chunk_rest(seq__42643_43490__$1);
var G__43495 = c__4556__auto___43493;
var G__43496 = cljs.core.count(c__4556__auto___43493);
var G__43497 = (0);
seq__42643_43473 = G__43494;
chunk__42645_43474 = G__43495;
count__42646_43475 = G__43496;
i__42647_43476 = G__43497;
continue;
} else {
var child_43498 = cljs.core.first(seq__42643_43490__$1);
if(cljs.core.truth_(child_43498)){
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,child_43498);


var G__43500 = cljs.core.next(seq__42643_43490__$1);
var G__43501 = null;
var G__43502 = (0);
var G__43503 = (0);
seq__42643_43473 = G__43500;
chunk__42645_43474 = G__43501;
count__42646_43475 = G__43502;
i__42647_43476 = G__43503;
continue;
} else {
var G__43506 = cljs.core.next(seq__42643_43490__$1);
var G__43507 = null;
var G__43508 = (0);
var G__43509 = (0);
seq__42643_43473 = G__43506;
chunk__42645_43474 = G__43507;
count__42646_43475 = G__43508;
i__42647_43476 = G__43509;
continue;
}
}
} else {
}
}
break;
}
} else {
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,children_43472);
}


var G__43511 = seq__42601_43465;
var G__43512 = chunk__42602_43466;
var G__43513 = count__42603_43467;
var G__43514 = (i__42604_43468 + (1));
seq__42601_43465 = G__43511;
chunk__42602_43466 = G__43512;
count__42603_43467 = G__43513;
i__42604_43468 = G__43514;
continue;
} else {
var temp__5735__auto___43515 = cljs.core.seq(seq__42601_43465);
if(temp__5735__auto___43515){
var seq__42601_43516__$1 = temp__5735__auto___43515;
if(cljs.core.chunked_seq_QMARK_(seq__42601_43516__$1)){
var c__4556__auto___43517 = cljs.core.chunk_first(seq__42601_43516__$1);
var G__43518 = cljs.core.chunk_rest(seq__42601_43516__$1);
var G__43519 = c__4556__auto___43517;
var G__43520 = cljs.core.count(c__4556__auto___43517);
var G__43521 = (0);
seq__42601_43465 = G__43518;
chunk__42602_43466 = G__43519;
count__42603_43467 = G__43520;
i__42604_43468 = G__43521;
continue;
} else {
var child_struct_43522 = cljs.core.first(seq__42601_43516__$1);
var children_43523 = shadow.dom.dom_node(child_struct_43522);
if(cljs.core.seq_QMARK_(children_43523)){
var seq__42665_43524 = cljs.core.seq(cljs.core.map.cljs$core$IFn$_invoke$arity$2(shadow.dom.dom_node,children_43523));
var chunk__42667_43525 = null;
var count__42668_43526 = (0);
var i__42669_43527 = (0);
while(true){
if((i__42669_43527 < count__42668_43526)){
var child_43534 = chunk__42667_43525.cljs$core$IIndexed$_nth$arity$2(null,i__42669_43527);
if(cljs.core.truth_(child_43534)){
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,child_43534);


var G__43535 = seq__42665_43524;
var G__43536 = chunk__42667_43525;
var G__43537 = count__42668_43526;
var G__43538 = (i__42669_43527 + (1));
seq__42665_43524 = G__43535;
chunk__42667_43525 = G__43536;
count__42668_43526 = G__43537;
i__42669_43527 = G__43538;
continue;
} else {
var G__43539 = seq__42665_43524;
var G__43540 = chunk__42667_43525;
var G__43541 = count__42668_43526;
var G__43542 = (i__42669_43527 + (1));
seq__42665_43524 = G__43539;
chunk__42667_43525 = G__43540;
count__42668_43526 = G__43541;
i__42669_43527 = G__43542;
continue;
}
} else {
var temp__5735__auto___43543__$1 = cljs.core.seq(seq__42665_43524);
if(temp__5735__auto___43543__$1){
var seq__42665_43544__$1 = temp__5735__auto___43543__$1;
if(cljs.core.chunked_seq_QMARK_(seq__42665_43544__$1)){
var c__4556__auto___43545 = cljs.core.chunk_first(seq__42665_43544__$1);
var G__43546 = cljs.core.chunk_rest(seq__42665_43544__$1);
var G__43547 = c__4556__auto___43545;
var G__43548 = cljs.core.count(c__4556__auto___43545);
var G__43549 = (0);
seq__42665_43524 = G__43546;
chunk__42667_43525 = G__43547;
count__42668_43526 = G__43548;
i__42669_43527 = G__43549;
continue;
} else {
var child_43551 = cljs.core.first(seq__42665_43544__$1);
if(cljs.core.truth_(child_43551)){
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,child_43551);


var G__43552 = cljs.core.next(seq__42665_43544__$1);
var G__43553 = null;
var G__43554 = (0);
var G__43555 = (0);
seq__42665_43524 = G__43552;
chunk__42667_43525 = G__43553;
count__42668_43526 = G__43554;
i__42669_43527 = G__43555;
continue;
} else {
var G__43556 = cljs.core.next(seq__42665_43544__$1);
var G__43557 = null;
var G__43558 = (0);
var G__43559 = (0);
seq__42665_43524 = G__43556;
chunk__42667_43525 = G__43557;
count__42668_43526 = G__43558;
i__42669_43527 = G__43559;
continue;
}
}
} else {
}
}
break;
}
} else {
shadow.dom.append.cljs$core$IFn$_invoke$arity$2(node,children_43523);
}


var G__43562 = cljs.core.next(seq__42601_43516__$1);
var G__43563 = null;
var G__43564 = (0);
var G__43565 = (0);
seq__42601_43465 = G__43562;
chunk__42602_43466 = G__43563;
count__42603_43467 = G__43564;
i__42604_43468 = G__43565;
continue;
}
} else {
}
}
break;
}

return node;
});
(cljs.core.Keyword.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.Keyword.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var this$__$1 = this;
return shadow.dom.make_dom_node(new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [this$__$1], null));
}));

(cljs.core.PersistentVector.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentVector.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var this$__$1 = this;
return shadow.dom.make_dom_node(this$__$1);
}));

(cljs.core.LazySeq.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.LazySeq.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var this$__$1 = this;
return cljs.core.map.cljs$core$IFn$_invoke$arity$2(shadow.dom._to_dom,this$__$1);
}));
if(cljs.core.truth_(((typeof HTMLElement) != 'undefined'))){
(HTMLElement.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(HTMLElement.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var this$__$1 = this;
return this$__$1;
}));
} else {
}
if(cljs.core.truth_(((typeof DocumentFragment) != 'undefined'))){
(DocumentFragment.prototype.shadow$dom$IElement$ = cljs.core.PROTOCOL_SENTINEL);

(DocumentFragment.prototype.shadow$dom$IElement$_to_dom$arity$1 = (function (this$){
var this$__$1 = this;
return this$__$1;
}));
} else {
}
/**
 * clear node children
 */
shadow.dom.reset = (function shadow$dom$reset(node){
return goog.dom.removeChildren(shadow.dom.dom_node(node));
});
shadow.dom.remove = (function shadow$dom$remove(node){
if((((!((node == null))))?(((((node.cljs$lang$protocol_mask$partition0$ & (8388608))) || ((cljs.core.PROTOCOL_SENTINEL === node.cljs$core$ISeqable$))))?true:false):false)){
var seq__42692 = cljs.core.seq(node);
var chunk__42693 = null;
var count__42694 = (0);
var i__42695 = (0);
while(true){
if((i__42695 < count__42694)){
var n = chunk__42693.cljs$core$IIndexed$_nth$arity$2(null,i__42695);
(shadow.dom.remove.cljs$core$IFn$_invoke$arity$1 ? shadow.dom.remove.cljs$core$IFn$_invoke$arity$1(n) : shadow.dom.remove.call(null,n));


var G__43604 = seq__42692;
var G__43605 = chunk__42693;
var G__43606 = count__42694;
var G__43607 = (i__42695 + (1));
seq__42692 = G__43604;
chunk__42693 = G__43605;
count__42694 = G__43606;
i__42695 = G__43607;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__42692);
if(temp__5735__auto__){
var seq__42692__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__42692__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__42692__$1);
var G__43611 = cljs.core.chunk_rest(seq__42692__$1);
var G__43612 = c__4556__auto__;
var G__43613 = cljs.core.count(c__4556__auto__);
var G__43614 = (0);
seq__42692 = G__43611;
chunk__42693 = G__43612;
count__42694 = G__43613;
i__42695 = G__43614;
continue;
} else {
var n = cljs.core.first(seq__42692__$1);
(shadow.dom.remove.cljs$core$IFn$_invoke$arity$1 ? shadow.dom.remove.cljs$core$IFn$_invoke$arity$1(n) : shadow.dom.remove.call(null,n));


var G__43615 = cljs.core.next(seq__42692__$1);
var G__43616 = null;
var G__43617 = (0);
var G__43618 = (0);
seq__42692 = G__43615;
chunk__42693 = G__43616;
count__42694 = G__43617;
i__42695 = G__43618;
continue;
}
} else {
return null;
}
}
break;
}
} else {
return goog.dom.removeNode(node);
}
});
shadow.dom.replace_node = (function shadow$dom$replace_node(old,new$){
return goog.dom.replaceNode(shadow.dom.dom_node(new$),shadow.dom.dom_node(old));
});
shadow.dom.text = (function shadow$dom$text(var_args){
var G__42709 = arguments.length;
switch (G__42709) {
case 2:
return shadow.dom.text.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 1:
return shadow.dom.text.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.text.cljs$core$IFn$_invoke$arity$2 = (function (el,new_text){
return (shadow.dom.dom_node(el).innerText = new_text);
}));

(shadow.dom.text.cljs$core$IFn$_invoke$arity$1 = (function (el){
return shadow.dom.dom_node(el).innerText;
}));

(shadow.dom.text.cljs$lang$maxFixedArity = 2);

shadow.dom.check = (function shadow$dom$check(var_args){
var G__42712 = arguments.length;
switch (G__42712) {
case 1:
return shadow.dom.check.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.check.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.check.cljs$core$IFn$_invoke$arity$1 = (function (el){
return shadow.dom.check.cljs$core$IFn$_invoke$arity$2(el,true);
}));

(shadow.dom.check.cljs$core$IFn$_invoke$arity$2 = (function (el,checked){
return (shadow.dom.dom_node(el).checked = checked);
}));

(shadow.dom.check.cljs$lang$maxFixedArity = 2);

shadow.dom.checked_QMARK_ = (function shadow$dom$checked_QMARK_(el){
return shadow.dom.dom_node(el).checked;
});
shadow.dom.form_elements = (function shadow$dom$form_elements(el){
return (new shadow.dom.NativeColl(shadow.dom.dom_node(el).elements));
});
shadow.dom.children = (function shadow$dom$children(el){
return (new shadow.dom.NativeColl(shadow.dom.dom_node(el).children));
});
shadow.dom.child_nodes = (function shadow$dom$child_nodes(el){
return (new shadow.dom.NativeColl(shadow.dom.dom_node(el).childNodes));
});
shadow.dom.attr = (function shadow$dom$attr(var_args){
var G__42715 = arguments.length;
switch (G__42715) {
case 2:
return shadow.dom.attr.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return shadow.dom.attr.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.attr.cljs$core$IFn$_invoke$arity$2 = (function (el,key){
return shadow.dom.dom_node(el).getAttribute(cljs.core.name(key));
}));

(shadow.dom.attr.cljs$core$IFn$_invoke$arity$3 = (function (el,key,default$){
var or__4126__auto__ = shadow.dom.dom_node(el).getAttribute(cljs.core.name(key));
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return default$;
}
}));

(shadow.dom.attr.cljs$lang$maxFixedArity = 3);

shadow.dom.del_attr = (function shadow$dom$del_attr(el,key){
return shadow.dom.dom_node(el).removeAttribute(cljs.core.name(key));
});
shadow.dom.data = (function shadow$dom$data(el,key){
return shadow.dom.dom_node(el).getAttribute(["data-",cljs.core.name(key)].join(''));
});
shadow.dom.set_data = (function shadow$dom$set_data(el,key,value){
return shadow.dom.dom_node(el).setAttribute(["data-",cljs.core.name(key)].join(''),cljs.core.str.cljs$core$IFn$_invoke$arity$1(value));
});
shadow.dom.set_html = (function shadow$dom$set_html(node,text){
return (shadow.dom.dom_node(node).innerHTML = text);
});
shadow.dom.get_html = (function shadow$dom$get_html(node){
return shadow.dom.dom_node(node).innerHTML;
});
shadow.dom.fragment = (function shadow$dom$fragment(var_args){
var args__4742__auto__ = [];
var len__4736__auto___43629 = arguments.length;
var i__4737__auto___43632 = (0);
while(true){
if((i__4737__auto___43632 < len__4736__auto___43629)){
args__4742__auto__.push((arguments[i__4737__auto___43632]));

var G__43634 = (i__4737__auto___43632 + (1));
i__4737__auto___43632 = G__43634;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((0) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((0)),(0),null)):null);
return shadow.dom.fragment.cljs$core$IFn$_invoke$arity$variadic(argseq__4743__auto__);
});

(shadow.dom.fragment.cljs$core$IFn$_invoke$arity$variadic = (function (nodes){
var fragment = document.createDocumentFragment();
var seq__42727_43635 = cljs.core.seq(nodes);
var chunk__42728_43636 = null;
var count__42729_43637 = (0);
var i__42730_43638 = (0);
while(true){
if((i__42730_43638 < count__42729_43637)){
var node_43639 = chunk__42728_43636.cljs$core$IIndexed$_nth$arity$2(null,i__42730_43638);
fragment.appendChild(shadow.dom._to_dom(node_43639));


var G__43640 = seq__42727_43635;
var G__43641 = chunk__42728_43636;
var G__43642 = count__42729_43637;
var G__43643 = (i__42730_43638 + (1));
seq__42727_43635 = G__43640;
chunk__42728_43636 = G__43641;
count__42729_43637 = G__43642;
i__42730_43638 = G__43643;
continue;
} else {
var temp__5735__auto___43644 = cljs.core.seq(seq__42727_43635);
if(temp__5735__auto___43644){
var seq__42727_43646__$1 = temp__5735__auto___43644;
if(cljs.core.chunked_seq_QMARK_(seq__42727_43646__$1)){
var c__4556__auto___43647 = cljs.core.chunk_first(seq__42727_43646__$1);
var G__43648 = cljs.core.chunk_rest(seq__42727_43646__$1);
var G__43649 = c__4556__auto___43647;
var G__43650 = cljs.core.count(c__4556__auto___43647);
var G__43651 = (0);
seq__42727_43635 = G__43648;
chunk__42728_43636 = G__43649;
count__42729_43637 = G__43650;
i__42730_43638 = G__43651;
continue;
} else {
var node_43652 = cljs.core.first(seq__42727_43646__$1);
fragment.appendChild(shadow.dom._to_dom(node_43652));


var G__43654 = cljs.core.next(seq__42727_43646__$1);
var G__43655 = null;
var G__43656 = (0);
var G__43657 = (0);
seq__42727_43635 = G__43654;
chunk__42728_43636 = G__43655;
count__42729_43637 = G__43656;
i__42730_43638 = G__43657;
continue;
}
} else {
}
}
break;
}

return (new shadow.dom.NativeColl(fragment));
}));

(shadow.dom.fragment.cljs$lang$maxFixedArity = (0));

/** @this {Function} */
(shadow.dom.fragment.cljs$lang$applyTo = (function (seq42723){
var self__4724__auto__ = this;
return self__4724__auto__.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq(seq42723));
}));

/**
 * given a html string, eval all <script> tags and return the html without the scripts
 * don't do this for everything, only content you trust.
 */
shadow.dom.eval_scripts = (function shadow$dom$eval_scripts(s){
var scripts = cljs.core.re_seq(/<script[^>]*?>(.+?)<\/script>/,s);
var seq__42742_43660 = cljs.core.seq(scripts);
var chunk__42743_43661 = null;
var count__42744_43662 = (0);
var i__42745_43663 = (0);
while(true){
if((i__42745_43663 < count__42744_43662)){
var vec__42757_43664 = chunk__42743_43661.cljs$core$IIndexed$_nth$arity$2(null,i__42745_43663);
var script_tag_43665 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42757_43664,(0),null);
var script_body_43666 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42757_43664,(1),null);
eval(script_body_43666);


var G__43669 = seq__42742_43660;
var G__43670 = chunk__42743_43661;
var G__43671 = count__42744_43662;
var G__43672 = (i__42745_43663 + (1));
seq__42742_43660 = G__43669;
chunk__42743_43661 = G__43670;
count__42744_43662 = G__43671;
i__42745_43663 = G__43672;
continue;
} else {
var temp__5735__auto___43674 = cljs.core.seq(seq__42742_43660);
if(temp__5735__auto___43674){
var seq__42742_43675__$1 = temp__5735__auto___43674;
if(cljs.core.chunked_seq_QMARK_(seq__42742_43675__$1)){
var c__4556__auto___43676 = cljs.core.chunk_first(seq__42742_43675__$1);
var G__43677 = cljs.core.chunk_rest(seq__42742_43675__$1);
var G__43678 = c__4556__auto___43676;
var G__43679 = cljs.core.count(c__4556__auto___43676);
var G__43680 = (0);
seq__42742_43660 = G__43677;
chunk__42743_43661 = G__43678;
count__42744_43662 = G__43679;
i__42745_43663 = G__43680;
continue;
} else {
var vec__42762_43681 = cljs.core.first(seq__42742_43675__$1);
var script_tag_43682 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42762_43681,(0),null);
var script_body_43683 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42762_43681,(1),null);
eval(script_body_43683);


var G__43684 = cljs.core.next(seq__42742_43675__$1);
var G__43685 = null;
var G__43686 = (0);
var G__43687 = (0);
seq__42742_43660 = G__43684;
chunk__42743_43661 = G__43685;
count__42744_43662 = G__43686;
i__42745_43663 = G__43687;
continue;
}
} else {
}
}
break;
}

return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (s__$1,p__42766){
var vec__42768 = p__42766;
var script_tag = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42768,(0),null);
var script_body = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42768,(1),null);
return clojure.string.replace(s__$1,script_tag,"");
}),s,scripts);
});
shadow.dom.str__GT_fragment = (function shadow$dom$str__GT_fragment(s){
var el = document.createElement("div");
(el.innerHTML = s);

return (new shadow.dom.NativeColl(goog.dom.childrenToNode_(document,el)));
});
shadow.dom.node_name = (function shadow$dom$node_name(el){
return shadow.dom.dom_node(el).nodeName;
});
shadow.dom.ancestor_by_class = (function shadow$dom$ancestor_by_class(el,cls){
return goog.dom.getAncestorByClass(shadow.dom.dom_node(el),cls);
});
shadow.dom.ancestor_by_tag = (function shadow$dom$ancestor_by_tag(var_args){
var G__42782 = arguments.length;
switch (G__42782) {
case 2:
return shadow.dom.ancestor_by_tag.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return shadow.dom.ancestor_by_tag.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.ancestor_by_tag.cljs$core$IFn$_invoke$arity$2 = (function (el,tag){
return goog.dom.getAncestorByTagNameAndClass(shadow.dom.dom_node(el),cljs.core.name(tag));
}));

(shadow.dom.ancestor_by_tag.cljs$core$IFn$_invoke$arity$3 = (function (el,tag,cls){
return goog.dom.getAncestorByTagNameAndClass(shadow.dom.dom_node(el),cljs.core.name(tag),cljs.core.name(cls));
}));

(shadow.dom.ancestor_by_tag.cljs$lang$maxFixedArity = 3);

shadow.dom.get_value = (function shadow$dom$get_value(dom){
return goog.dom.forms.getValue(shadow.dom.dom_node(dom));
});
shadow.dom.set_value = (function shadow$dom$set_value(dom,value){
return goog.dom.forms.setValue(shadow.dom.dom_node(dom),value);
});
shadow.dom.px = (function shadow$dom$px(value){
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1((value | (0))),"px"].join('');
});
shadow.dom.pct = (function shadow$dom$pct(value){
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(value),"%"].join('');
});
shadow.dom.remove_style_STAR_ = (function shadow$dom$remove_style_STAR_(el,style){
return el.style.removeProperty(cljs.core.name(style));
});
shadow.dom.remove_style = (function shadow$dom$remove_style(el,style){
var el__$1 = shadow.dom.dom_node(el);
return shadow.dom.remove_style_STAR_(el__$1,style);
});
shadow.dom.remove_styles = (function shadow$dom$remove_styles(el,style_keys){
var el__$1 = shadow.dom.dom_node(el);
var seq__42814 = cljs.core.seq(style_keys);
var chunk__42816 = null;
var count__42817 = (0);
var i__42818 = (0);
while(true){
if((i__42818 < count__42817)){
var it = chunk__42816.cljs$core$IIndexed$_nth$arity$2(null,i__42818);
shadow.dom.remove_style_STAR_(el__$1,it);


var G__43697 = seq__42814;
var G__43698 = chunk__42816;
var G__43699 = count__42817;
var G__43700 = (i__42818 + (1));
seq__42814 = G__43697;
chunk__42816 = G__43698;
count__42817 = G__43699;
i__42818 = G__43700;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__42814);
if(temp__5735__auto__){
var seq__42814__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__42814__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__42814__$1);
var G__43702 = cljs.core.chunk_rest(seq__42814__$1);
var G__43703 = c__4556__auto__;
var G__43704 = cljs.core.count(c__4556__auto__);
var G__43705 = (0);
seq__42814 = G__43702;
chunk__42816 = G__43703;
count__42817 = G__43704;
i__42818 = G__43705;
continue;
} else {
var it = cljs.core.first(seq__42814__$1);
shadow.dom.remove_style_STAR_(el__$1,it);


var G__43707 = cljs.core.next(seq__42814__$1);
var G__43708 = null;
var G__43709 = (0);
var G__43710 = (0);
seq__42814 = G__43707;
chunk__42816 = G__43708;
count__42817 = G__43709;
i__42818 = G__43710;
continue;
}
} else {
return null;
}
}
break;
}
});

/**
* @constructor
 * @implements {cljs.core.IRecord}
 * @implements {cljs.core.IKVReduce}
 * @implements {cljs.core.IEquiv}
 * @implements {cljs.core.IHash}
 * @implements {cljs.core.ICollection}
 * @implements {cljs.core.ICounted}
 * @implements {cljs.core.ISeqable}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.ICloneable}
 * @implements {cljs.core.IPrintWithWriter}
 * @implements {cljs.core.IIterable}
 * @implements {cljs.core.IWithMeta}
 * @implements {cljs.core.IAssociative}
 * @implements {cljs.core.IMap}
 * @implements {cljs.core.ILookup}
*/
shadow.dom.Coordinate = (function (x,y,__meta,__extmap,__hash){
this.x = x;
this.y = y;
this.__meta = __meta;
this.__extmap = __extmap;
this.__hash = __hash;
this.cljs$lang$protocol_mask$partition0$ = 2230716170;
this.cljs$lang$protocol_mask$partition1$ = 139264;
});
(shadow.dom.Coordinate.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__4380__auto__,k__4381__auto__){
var self__ = this;
var this__4380__auto____$1 = this;
return this__4380__auto____$1.cljs$core$ILookup$_lookup$arity$3(null,k__4381__auto__,null);
}));

(shadow.dom.Coordinate.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__4382__auto__,k42828,else__4383__auto__){
var self__ = this;
var this__4382__auto____$1 = this;
var G__42846 = k42828;
var G__42846__$1 = (((G__42846 instanceof cljs.core.Keyword))?G__42846.fqn:null);
switch (G__42846__$1) {
case "x":
return self__.x;

break;
case "y":
return self__.y;

break;
default:
return cljs.core.get.cljs$core$IFn$_invoke$arity$3(self__.__extmap,k42828,else__4383__auto__);

}
}));

(shadow.dom.Coordinate.prototype.cljs$core$IKVReduce$_kv_reduce$arity$3 = (function (this__4399__auto__,f__4400__auto__,init__4401__auto__){
var self__ = this;
var this__4399__auto____$1 = this;
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (ret__4402__auto__,p__42850){
var vec__42851 = p__42850;
var k__4403__auto__ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42851,(0),null);
var v__4404__auto__ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42851,(1),null);
return (f__4400__auto__.cljs$core$IFn$_invoke$arity$3 ? f__4400__auto__.cljs$core$IFn$_invoke$arity$3(ret__4402__auto__,k__4403__auto__,v__4404__auto__) : f__4400__auto__.call(null,ret__4402__auto__,k__4403__auto__,v__4404__auto__));
}),init__4401__auto__,this__4399__auto____$1);
}));

(shadow.dom.Coordinate.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__4394__auto__,writer__4395__auto__,opts__4396__auto__){
var self__ = this;
var this__4394__auto____$1 = this;
var pr_pair__4397__auto__ = (function (keyval__4398__auto__){
return cljs.core.pr_sequential_writer(writer__4395__auto__,cljs.core.pr_writer,""," ","",opts__4396__auto__,keyval__4398__auto__);
});
return cljs.core.pr_sequential_writer(writer__4395__auto__,pr_pair__4397__auto__,"#shadow.dom.Coordinate{",", ","}",opts__4396__auto__,cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"x","x",2099068185),self__.x],null)),(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"y","y",-1757859776),self__.y],null))], null),self__.__extmap));
}));

(shadow.dom.Coordinate.prototype.cljs$core$IIterable$_iterator$arity$1 = (function (G__42827){
var self__ = this;
var G__42827__$1 = this;
return (new cljs.core.RecordIter((0),G__42827__$1,2,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"x","x",2099068185),new cljs.core.Keyword(null,"y","y",-1757859776)], null),(cljs.core.truth_(self__.__extmap)?cljs.core._iterator(self__.__extmap):cljs.core.nil_iter())));
}));

(shadow.dom.Coordinate.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__4378__auto__){
var self__ = this;
var this__4378__auto____$1 = this;
return self__.__meta;
}));

(shadow.dom.Coordinate.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__4375__auto__){
var self__ = this;
var this__4375__auto____$1 = this;
return (new shadow.dom.Coordinate(self__.x,self__.y,self__.__meta,self__.__extmap,self__.__hash));
}));

(shadow.dom.Coordinate.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__4384__auto__){
var self__ = this;
var this__4384__auto____$1 = this;
return (2 + cljs.core.count(self__.__extmap));
}));

(shadow.dom.Coordinate.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__4376__auto__){
var self__ = this;
var this__4376__auto____$1 = this;
var h__4238__auto__ = self__.__hash;
if((!((h__4238__auto__ == null)))){
return h__4238__auto__;
} else {
var h__4238__auto____$1 = (function (coll__4377__auto__){
return (145542109 ^ cljs.core.hash_unordered_coll(coll__4377__auto__));
})(this__4376__auto____$1);
(self__.__hash = h__4238__auto____$1);

return h__4238__auto____$1;
}
}));

(shadow.dom.Coordinate.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this42829,other42830){
var self__ = this;
var this42829__$1 = this;
return (((!((other42830 == null)))) && ((this42829__$1.constructor === other42830.constructor)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42829__$1.x,other42830.x)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42829__$1.y,other42830.y)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42829__$1.__extmap,other42830.__extmap)));
}));

(shadow.dom.Coordinate.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__4389__auto__,k__4390__auto__){
var self__ = this;
var this__4389__auto____$1 = this;
if(cljs.core.contains_QMARK_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"y","y",-1757859776),null,new cljs.core.Keyword(null,"x","x",2099068185),null], null), null),k__4390__auto__)){
return cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(cljs.core._with_meta(cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,this__4389__auto____$1),self__.__meta),k__4390__auto__);
} else {
return (new shadow.dom.Coordinate(self__.x,self__.y,self__.__meta,cljs.core.not_empty(cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(self__.__extmap,k__4390__auto__)),null));
}
}));

(shadow.dom.Coordinate.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__4387__auto__,k__4388__auto__,G__42827){
var self__ = this;
var this__4387__auto____$1 = this;
var pred__42865 = cljs.core.keyword_identical_QMARK_;
var expr__42866 = k__4388__auto__;
if(cljs.core.truth_((pred__42865.cljs$core$IFn$_invoke$arity$2 ? pred__42865.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"x","x",2099068185),expr__42866) : pred__42865.call(null,new cljs.core.Keyword(null,"x","x",2099068185),expr__42866)))){
return (new shadow.dom.Coordinate(G__42827,self__.y,self__.__meta,self__.__extmap,null));
} else {
if(cljs.core.truth_((pred__42865.cljs$core$IFn$_invoke$arity$2 ? pred__42865.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"y","y",-1757859776),expr__42866) : pred__42865.call(null,new cljs.core.Keyword(null,"y","y",-1757859776),expr__42866)))){
return (new shadow.dom.Coordinate(self__.x,G__42827,self__.__meta,self__.__extmap,null));
} else {
return (new shadow.dom.Coordinate(self__.x,self__.y,self__.__meta,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.__extmap,k__4388__auto__,G__42827),null));
}
}
}));

(shadow.dom.Coordinate.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__4392__auto__){
var self__ = this;
var this__4392__auto____$1 = this;
return cljs.core.seq(cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.MapEntry(new cljs.core.Keyword(null,"x","x",2099068185),self__.x,null)),(new cljs.core.MapEntry(new cljs.core.Keyword(null,"y","y",-1757859776),self__.y,null))], null),self__.__extmap));
}));

(shadow.dom.Coordinate.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__4379__auto__,G__42827){
var self__ = this;
var this__4379__auto____$1 = this;
return (new shadow.dom.Coordinate(self__.x,self__.y,G__42827,self__.__extmap,self__.__hash));
}));

(shadow.dom.Coordinate.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__4385__auto__,entry__4386__auto__){
var self__ = this;
var this__4385__auto____$1 = this;
if(cljs.core.vector_QMARK_(entry__4386__auto__)){
return this__4385__auto____$1.cljs$core$IAssociative$_assoc$arity$3(null,cljs.core._nth(entry__4386__auto__,(0)),cljs.core._nth(entry__4386__auto__,(1)));
} else {
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3(cljs.core._conj,this__4385__auto____$1,entry__4386__auto__);
}
}));

(shadow.dom.Coordinate.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"x","x",-555367584,null),new cljs.core.Symbol(null,"y","y",-117328249,null)], null);
}));

(shadow.dom.Coordinate.cljs$lang$type = true);

(shadow.dom.Coordinate.cljs$lang$ctorPrSeq = (function (this__4423__auto__){
return (new cljs.core.List(null,"shadow.dom/Coordinate",null,(1),null));
}));

(shadow.dom.Coordinate.cljs$lang$ctorPrWriter = (function (this__4423__auto__,writer__4424__auto__){
return cljs.core._write(writer__4424__auto__,"shadow.dom/Coordinate");
}));

/**
 * Positional factory function for shadow.dom/Coordinate.
 */
shadow.dom.__GT_Coordinate = (function shadow$dom$__GT_Coordinate(x,y){
return (new shadow.dom.Coordinate(x,y,null,null,null));
});

/**
 * Factory function for shadow.dom/Coordinate, taking a map of keywords to field values.
 */
shadow.dom.map__GT_Coordinate = (function shadow$dom$map__GT_Coordinate(G__42831){
var extmap__4419__auto__ = (function (){var G__42877 = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$variadic(G__42831,new cljs.core.Keyword(null,"x","x",2099068185),cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.Keyword(null,"y","y",-1757859776)], 0));
if(cljs.core.record_QMARK_(G__42831)){
return cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,G__42877);
} else {
return G__42877;
}
})();
return (new shadow.dom.Coordinate(new cljs.core.Keyword(null,"x","x",2099068185).cljs$core$IFn$_invoke$arity$1(G__42831),new cljs.core.Keyword(null,"y","y",-1757859776).cljs$core$IFn$_invoke$arity$1(G__42831),null,cljs.core.not_empty(extmap__4419__auto__),null));
});

shadow.dom.get_position = (function shadow$dom$get_position(el){
var pos = goog.style.getPosition(shadow.dom.dom_node(el));
return shadow.dom.__GT_Coordinate(pos.x,pos.y);
});
shadow.dom.get_client_position = (function shadow$dom$get_client_position(el){
var pos = goog.style.getClientPosition(shadow.dom.dom_node(el));
return shadow.dom.__GT_Coordinate(pos.x,pos.y);
});
shadow.dom.get_page_offset = (function shadow$dom$get_page_offset(el){
var pos = goog.style.getPageOffset(shadow.dom.dom_node(el));
return shadow.dom.__GT_Coordinate(pos.x,pos.y);
});

/**
* @constructor
 * @implements {cljs.core.IRecord}
 * @implements {cljs.core.IKVReduce}
 * @implements {cljs.core.IEquiv}
 * @implements {cljs.core.IHash}
 * @implements {cljs.core.ICollection}
 * @implements {cljs.core.ICounted}
 * @implements {cljs.core.ISeqable}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.ICloneable}
 * @implements {cljs.core.IPrintWithWriter}
 * @implements {cljs.core.IIterable}
 * @implements {cljs.core.IWithMeta}
 * @implements {cljs.core.IAssociative}
 * @implements {cljs.core.IMap}
 * @implements {cljs.core.ILookup}
*/
shadow.dom.Size = (function (w,h,__meta,__extmap,__hash){
this.w = w;
this.h = h;
this.__meta = __meta;
this.__extmap = __extmap;
this.__hash = __hash;
this.cljs$lang$protocol_mask$partition0$ = 2230716170;
this.cljs$lang$protocol_mask$partition1$ = 139264;
});
(shadow.dom.Size.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__4380__auto__,k__4381__auto__){
var self__ = this;
var this__4380__auto____$1 = this;
return this__4380__auto____$1.cljs$core$ILookup$_lookup$arity$3(null,k__4381__auto__,null);
}));

(shadow.dom.Size.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__4382__auto__,k42880,else__4383__auto__){
var self__ = this;
var this__4382__auto____$1 = this;
var G__42896 = k42880;
var G__42896__$1 = (((G__42896 instanceof cljs.core.Keyword))?G__42896.fqn:null);
switch (G__42896__$1) {
case "w":
return self__.w;

break;
case "h":
return self__.h;

break;
default:
return cljs.core.get.cljs$core$IFn$_invoke$arity$3(self__.__extmap,k42880,else__4383__auto__);

}
}));

(shadow.dom.Size.prototype.cljs$core$IKVReduce$_kv_reduce$arity$3 = (function (this__4399__auto__,f__4400__auto__,init__4401__auto__){
var self__ = this;
var this__4399__auto____$1 = this;
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (ret__4402__auto__,p__42905){
var vec__42907 = p__42905;
var k__4403__auto__ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42907,(0),null);
var v__4404__auto__ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42907,(1),null);
return (f__4400__auto__.cljs$core$IFn$_invoke$arity$3 ? f__4400__auto__.cljs$core$IFn$_invoke$arity$3(ret__4402__auto__,k__4403__auto__,v__4404__auto__) : f__4400__auto__.call(null,ret__4402__auto__,k__4403__auto__,v__4404__auto__));
}),init__4401__auto__,this__4399__auto____$1);
}));

(shadow.dom.Size.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__4394__auto__,writer__4395__auto__,opts__4396__auto__){
var self__ = this;
var this__4394__auto____$1 = this;
var pr_pair__4397__auto__ = (function (keyval__4398__auto__){
return cljs.core.pr_sequential_writer(writer__4395__auto__,cljs.core.pr_writer,""," ","",opts__4396__auto__,keyval__4398__auto__);
});
return cljs.core.pr_sequential_writer(writer__4395__auto__,pr_pair__4397__auto__,"#shadow.dom.Size{",", ","}",opts__4396__auto__,cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"w","w",354169001),self__.w],null)),(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"h","h",1109658740),self__.h],null))], null),self__.__extmap));
}));

(shadow.dom.Size.prototype.cljs$core$IIterable$_iterator$arity$1 = (function (G__42879){
var self__ = this;
var G__42879__$1 = this;
return (new cljs.core.RecordIter((0),G__42879__$1,2,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"w","w",354169001),new cljs.core.Keyword(null,"h","h",1109658740)], null),(cljs.core.truth_(self__.__extmap)?cljs.core._iterator(self__.__extmap):cljs.core.nil_iter())));
}));

(shadow.dom.Size.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__4378__auto__){
var self__ = this;
var this__4378__auto____$1 = this;
return self__.__meta;
}));

(shadow.dom.Size.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__4375__auto__){
var self__ = this;
var this__4375__auto____$1 = this;
return (new shadow.dom.Size(self__.w,self__.h,self__.__meta,self__.__extmap,self__.__hash));
}));

(shadow.dom.Size.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__4384__auto__){
var self__ = this;
var this__4384__auto____$1 = this;
return (2 + cljs.core.count(self__.__extmap));
}));

(shadow.dom.Size.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__4376__auto__){
var self__ = this;
var this__4376__auto____$1 = this;
var h__4238__auto__ = self__.__hash;
if((!((h__4238__auto__ == null)))){
return h__4238__auto__;
} else {
var h__4238__auto____$1 = (function (coll__4377__auto__){
return (-1228019642 ^ cljs.core.hash_unordered_coll(coll__4377__auto__));
})(this__4376__auto____$1);
(self__.__hash = h__4238__auto____$1);

return h__4238__auto____$1;
}
}));

(shadow.dom.Size.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this42881,other42882){
var self__ = this;
var this42881__$1 = this;
return (((!((other42882 == null)))) && ((this42881__$1.constructor === other42882.constructor)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42881__$1.w,other42882.w)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42881__$1.h,other42882.h)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this42881__$1.__extmap,other42882.__extmap)));
}));

(shadow.dom.Size.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__4389__auto__,k__4390__auto__){
var self__ = this;
var this__4389__auto____$1 = this;
if(cljs.core.contains_QMARK_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"w","w",354169001),null,new cljs.core.Keyword(null,"h","h",1109658740),null], null), null),k__4390__auto__)){
return cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(cljs.core._with_meta(cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,this__4389__auto____$1),self__.__meta),k__4390__auto__);
} else {
return (new shadow.dom.Size(self__.w,self__.h,self__.__meta,cljs.core.not_empty(cljs.core.dissoc.cljs$core$IFn$_invoke$arity$2(self__.__extmap,k__4390__auto__)),null));
}
}));

(shadow.dom.Size.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__4387__auto__,k__4388__auto__,G__42879){
var self__ = this;
var this__4387__auto____$1 = this;
var pred__42934 = cljs.core.keyword_identical_QMARK_;
var expr__42935 = k__4388__auto__;
if(cljs.core.truth_((pred__42934.cljs$core$IFn$_invoke$arity$2 ? pred__42934.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"w","w",354169001),expr__42935) : pred__42934.call(null,new cljs.core.Keyword(null,"w","w",354169001),expr__42935)))){
return (new shadow.dom.Size(G__42879,self__.h,self__.__meta,self__.__extmap,null));
} else {
if(cljs.core.truth_((pred__42934.cljs$core$IFn$_invoke$arity$2 ? pred__42934.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"h","h",1109658740),expr__42935) : pred__42934.call(null,new cljs.core.Keyword(null,"h","h",1109658740),expr__42935)))){
return (new shadow.dom.Size(self__.w,G__42879,self__.__meta,self__.__extmap,null));
} else {
return (new shadow.dom.Size(self__.w,self__.h,self__.__meta,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.__extmap,k__4388__auto__,G__42879),null));
}
}
}));

(shadow.dom.Size.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__4392__auto__){
var self__ = this;
var this__4392__auto____$1 = this;
return cljs.core.seq(cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.MapEntry(new cljs.core.Keyword(null,"w","w",354169001),self__.w,null)),(new cljs.core.MapEntry(new cljs.core.Keyword(null,"h","h",1109658740),self__.h,null))], null),self__.__extmap));
}));

(shadow.dom.Size.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__4379__auto__,G__42879){
var self__ = this;
var this__4379__auto____$1 = this;
return (new shadow.dom.Size(self__.w,self__.h,G__42879,self__.__extmap,self__.__hash));
}));

(shadow.dom.Size.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__4385__auto__,entry__4386__auto__){
var self__ = this;
var this__4385__auto____$1 = this;
if(cljs.core.vector_QMARK_(entry__4386__auto__)){
return this__4385__auto____$1.cljs$core$IAssociative$_assoc$arity$3(null,cljs.core._nth(entry__4386__auto__,(0)),cljs.core._nth(entry__4386__auto__,(1)));
} else {
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3(cljs.core._conj,this__4385__auto____$1,entry__4386__auto__);
}
}));

(shadow.dom.Size.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"w","w",1994700528,null),new cljs.core.Symbol(null,"h","h",-1544777029,null)], null);
}));

(shadow.dom.Size.cljs$lang$type = true);

(shadow.dom.Size.cljs$lang$ctorPrSeq = (function (this__4423__auto__){
return (new cljs.core.List(null,"shadow.dom/Size",null,(1),null));
}));

(shadow.dom.Size.cljs$lang$ctorPrWriter = (function (this__4423__auto__,writer__4424__auto__){
return cljs.core._write(writer__4424__auto__,"shadow.dom/Size");
}));

/**
 * Positional factory function for shadow.dom/Size.
 */
shadow.dom.__GT_Size = (function shadow$dom$__GT_Size(w,h){
return (new shadow.dom.Size(w,h,null,null,null));
});

/**
 * Factory function for shadow.dom/Size, taking a map of keywords to field values.
 */
shadow.dom.map__GT_Size = (function shadow$dom$map__GT_Size(G__42884){
var extmap__4419__auto__ = (function (){var G__42948 = cljs.core.dissoc.cljs$core$IFn$_invoke$arity$variadic(G__42884,new cljs.core.Keyword(null,"w","w",354169001),cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.Keyword(null,"h","h",1109658740)], 0));
if(cljs.core.record_QMARK_(G__42884)){
return cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,G__42948);
} else {
return G__42948;
}
})();
return (new shadow.dom.Size(new cljs.core.Keyword(null,"w","w",354169001).cljs$core$IFn$_invoke$arity$1(G__42884),new cljs.core.Keyword(null,"h","h",1109658740).cljs$core$IFn$_invoke$arity$1(G__42884),null,cljs.core.not_empty(extmap__4419__auto__),null));
});

shadow.dom.size__GT_clj = (function shadow$dom$size__GT_clj(size){
return (new shadow.dom.Size(size.width,size.height,null,null,null));
});
shadow.dom.get_size = (function shadow$dom$get_size(el){
return shadow.dom.size__GT_clj(goog.style.getSize(shadow.dom.dom_node(el)));
});
shadow.dom.get_height = (function shadow$dom$get_height(el){
return shadow.dom.get_size(el).h;
});
shadow.dom.get_viewport_size = (function shadow$dom$get_viewport_size(){
return shadow.dom.size__GT_clj(goog.dom.getViewportSize());
});
shadow.dom.first_child = (function shadow$dom$first_child(el){
return (shadow.dom.dom_node(el).children[(0)]);
});
shadow.dom.select_option_values = (function shadow$dom$select_option_values(el){
var native$ = shadow.dom.dom_node(el);
var opts = (native$["options"]);
var a__4610__auto__ = opts;
var l__4611__auto__ = a__4610__auto__.length;
var i = (0);
var ret = cljs.core.PersistentVector.EMPTY;
while(true){
if((i < l__4611__auto__)){
var G__43770 = (i + (1));
var G__43771 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(ret,(opts[i]["value"]));
i = G__43770;
ret = G__43771;
continue;
} else {
return ret;
}
break;
}
});
shadow.dom.build_url = (function shadow$dom$build_url(path,query_params){
if(cljs.core.empty_QMARK_(query_params)){
return path;
} else {
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(path),"?",clojure.string.join.cljs$core$IFn$_invoke$arity$2("&",cljs.core.map.cljs$core$IFn$_invoke$arity$2((function (p__42976){
var vec__42977 = p__42976;
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42977,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__42977,(1),null);
return [cljs.core.name(k),"=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(encodeURIComponent(cljs.core.str.cljs$core$IFn$_invoke$arity$1(v)))].join('');
}),query_params))].join('');
}
});
shadow.dom.redirect = (function shadow$dom$redirect(var_args){
var G__42984 = arguments.length;
switch (G__42984) {
case 1:
return shadow.dom.redirect.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return shadow.dom.redirect.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.redirect.cljs$core$IFn$_invoke$arity$1 = (function (path){
return shadow.dom.redirect.cljs$core$IFn$_invoke$arity$2(path,cljs.core.PersistentArrayMap.EMPTY);
}));

(shadow.dom.redirect.cljs$core$IFn$_invoke$arity$2 = (function (path,query_params){
return (document["location"]["href"] = shadow.dom.build_url(path,query_params));
}));

(shadow.dom.redirect.cljs$lang$maxFixedArity = 2);

shadow.dom.reload_BANG_ = (function shadow$dom$reload_BANG_(){
return (document.location.href = document.location.href);
});
shadow.dom.tag_name = (function shadow$dom$tag_name(el){
var dom = shadow.dom.dom_node(el);
return dom.tagName;
});
shadow.dom.insert_after = (function shadow$dom$insert_after(ref,new$){
var new_node = shadow.dom.dom_node(new$);
goog.dom.insertSiblingAfter(new_node,shadow.dom.dom_node(ref));

return new_node;
});
shadow.dom.insert_before = (function shadow$dom$insert_before(ref,new$){
var new_node = shadow.dom.dom_node(new$);
goog.dom.insertSiblingBefore(new_node,shadow.dom.dom_node(ref));

return new_node;
});
shadow.dom.insert_first = (function shadow$dom$insert_first(ref,new$){
var temp__5733__auto__ = shadow.dom.dom_node(ref).firstChild;
if(cljs.core.truth_(temp__5733__auto__)){
var child = temp__5733__auto__;
return shadow.dom.insert_before(child,new$);
} else {
return shadow.dom.append.cljs$core$IFn$_invoke$arity$2(ref,new$);
}
});
shadow.dom.index_of = (function shadow$dom$index_of(el){
var el__$1 = shadow.dom.dom_node(el);
var i = (0);
while(true){
var ps = el__$1.previousSibling;
if((ps == null)){
return i;
} else {
var G__43784 = ps;
var G__43785 = (i + (1));
el__$1 = G__43784;
i = G__43785;
continue;
}
break;
}
});
shadow.dom.get_parent = (function shadow$dom$get_parent(el){
return goog.dom.getParentElement(shadow.dom.dom_node(el));
});
shadow.dom.parents = (function shadow$dom$parents(el){
var parent = shadow.dom.get_parent(el);
if(cljs.core.truth_(parent)){
return cljs.core.cons(parent,(new cljs.core.LazySeq(null,(function (){
return (shadow.dom.parents.cljs$core$IFn$_invoke$arity$1 ? shadow.dom.parents.cljs$core$IFn$_invoke$arity$1(parent) : shadow.dom.parents.call(null,parent));
}),null,null)));
} else {
return null;
}
});
shadow.dom.matches = (function shadow$dom$matches(el,sel){
return shadow.dom.dom_node(el).matches(sel);
});
shadow.dom.get_next_sibling = (function shadow$dom$get_next_sibling(el){
return goog.dom.getNextElementSibling(shadow.dom.dom_node(el));
});
shadow.dom.get_previous_sibling = (function shadow$dom$get_previous_sibling(el){
return goog.dom.getPreviousElementSibling(shadow.dom.dom_node(el));
});
shadow.dom.xmlns = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(new cljs.core.PersistentArrayMap(null, 2, ["svg","http://www.w3.org/2000/svg","xlink","http://www.w3.org/1999/xlink"], null));
shadow.dom.create_svg_node = (function shadow$dom$create_svg_node(tag_def,props){
var vec__43050 = shadow.dom.parse_tag(tag_def);
var tag_name = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43050,(0),null);
var tag_id = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43050,(1),null);
var tag_classes = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43050,(2),null);
var el = document.createElementNS("http://www.w3.org/2000/svg",tag_name);
if(cljs.core.truth_(tag_id)){
el.setAttribute("id",tag_id);
} else {
}

if(cljs.core.truth_(tag_classes)){
el.setAttribute("class",shadow.dom.merge_class_string(new cljs.core.Keyword(null,"class","class",-2030961996).cljs$core$IFn$_invoke$arity$1(props),tag_classes));
} else {
}

var seq__43057_43792 = cljs.core.seq(props);
var chunk__43058_43793 = null;
var count__43059_43794 = (0);
var i__43060_43795 = (0);
while(true){
if((i__43060_43795 < count__43059_43794)){
var vec__43085_43797 = chunk__43058_43793.cljs$core$IIndexed$_nth$arity$2(null,i__43060_43795);
var k_43798 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43085_43797,(0),null);
var v_43799 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43085_43797,(1),null);
el.setAttributeNS((function (){var temp__5735__auto__ = cljs.core.namespace(k_43798);
if(cljs.core.truth_(temp__5735__auto__)){
var ns = temp__5735__auto__;
return cljs.core.get.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(shadow.dom.xmlns),ns);
} else {
return null;
}
})(),cljs.core.name(k_43798),v_43799);


var G__43809 = seq__43057_43792;
var G__43810 = chunk__43058_43793;
var G__43811 = count__43059_43794;
var G__43812 = (i__43060_43795 + (1));
seq__43057_43792 = G__43809;
chunk__43058_43793 = G__43810;
count__43059_43794 = G__43811;
i__43060_43795 = G__43812;
continue;
} else {
var temp__5735__auto___43813 = cljs.core.seq(seq__43057_43792);
if(temp__5735__auto___43813){
var seq__43057_43814__$1 = temp__5735__auto___43813;
if(cljs.core.chunked_seq_QMARK_(seq__43057_43814__$1)){
var c__4556__auto___43815 = cljs.core.chunk_first(seq__43057_43814__$1);
var G__43817 = cljs.core.chunk_rest(seq__43057_43814__$1);
var G__43818 = c__4556__auto___43815;
var G__43819 = cljs.core.count(c__4556__auto___43815);
var G__43820 = (0);
seq__43057_43792 = G__43817;
chunk__43058_43793 = G__43818;
count__43059_43794 = G__43819;
i__43060_43795 = G__43820;
continue;
} else {
var vec__43099_43821 = cljs.core.first(seq__43057_43814__$1);
var k_43822 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43099_43821,(0),null);
var v_43823 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43099_43821,(1),null);
el.setAttributeNS((function (){var temp__5735__auto____$1 = cljs.core.namespace(k_43822);
if(cljs.core.truth_(temp__5735__auto____$1)){
var ns = temp__5735__auto____$1;
return cljs.core.get.cljs$core$IFn$_invoke$arity$2(cljs.core.deref(shadow.dom.xmlns),ns);
} else {
return null;
}
})(),cljs.core.name(k_43822),v_43823);


var G__43824 = cljs.core.next(seq__43057_43814__$1);
var G__43825 = null;
var G__43826 = (0);
var G__43827 = (0);
seq__43057_43792 = G__43824;
chunk__43058_43793 = G__43825;
count__43059_43794 = G__43826;
i__43060_43795 = G__43827;
continue;
}
} else {
}
}
break;
}

return el;
});
shadow.dom.svg_node = (function shadow$dom$svg_node(el){
if((el == null)){
return null;
} else {
if((((!((el == null))))?((((false) || ((cljs.core.PROTOCOL_SENTINEL === el.shadow$dom$SVGElement$))))?true:false):false)){
return el.shadow$dom$SVGElement$_to_svg$arity$1(null);
} else {
return el;

}
}
});
shadow.dom.make_svg_node = (function shadow$dom$make_svg_node(structure){
var vec__43123 = shadow.dom.destructure_node(shadow.dom.create_svg_node,structure);
var node = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43123,(0),null);
var node_children = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43123,(1),null);
var seq__43126_43841 = cljs.core.seq(node_children);
var chunk__43128_43842 = null;
var count__43129_43843 = (0);
var i__43130_43844 = (0);
while(true){
if((i__43130_43844 < count__43129_43843)){
var child_struct_43845 = chunk__43128_43842.cljs$core$IIndexed$_nth$arity$2(null,i__43130_43844);
if((!((child_struct_43845 == null)))){
if(typeof child_struct_43845 === 'string'){
var text_43846 = (node["textContent"]);
(node["textContent"] = [cljs.core.str.cljs$core$IFn$_invoke$arity$1(text_43846),child_struct_43845].join(''));
} else {
var children_43848 = shadow.dom.svg_node(child_struct_43845);
if(cljs.core.seq_QMARK_(children_43848)){
var seq__43183_43849 = cljs.core.seq(children_43848);
var chunk__43185_43850 = null;
var count__43186_43851 = (0);
var i__43187_43852 = (0);
while(true){
if((i__43187_43852 < count__43186_43851)){
var child_43853 = chunk__43185_43850.cljs$core$IIndexed$_nth$arity$2(null,i__43187_43852);
if(cljs.core.truth_(child_43853)){
node.appendChild(child_43853);


var G__43854 = seq__43183_43849;
var G__43855 = chunk__43185_43850;
var G__43856 = count__43186_43851;
var G__43857 = (i__43187_43852 + (1));
seq__43183_43849 = G__43854;
chunk__43185_43850 = G__43855;
count__43186_43851 = G__43856;
i__43187_43852 = G__43857;
continue;
} else {
var G__43859 = seq__43183_43849;
var G__43860 = chunk__43185_43850;
var G__43861 = count__43186_43851;
var G__43862 = (i__43187_43852 + (1));
seq__43183_43849 = G__43859;
chunk__43185_43850 = G__43860;
count__43186_43851 = G__43861;
i__43187_43852 = G__43862;
continue;
}
} else {
var temp__5735__auto___43866 = cljs.core.seq(seq__43183_43849);
if(temp__5735__auto___43866){
var seq__43183_43868__$1 = temp__5735__auto___43866;
if(cljs.core.chunked_seq_QMARK_(seq__43183_43868__$1)){
var c__4556__auto___43869 = cljs.core.chunk_first(seq__43183_43868__$1);
var G__43870 = cljs.core.chunk_rest(seq__43183_43868__$1);
var G__43871 = c__4556__auto___43869;
var G__43872 = cljs.core.count(c__4556__auto___43869);
var G__43873 = (0);
seq__43183_43849 = G__43870;
chunk__43185_43850 = G__43871;
count__43186_43851 = G__43872;
i__43187_43852 = G__43873;
continue;
} else {
var child_43874 = cljs.core.first(seq__43183_43868__$1);
if(cljs.core.truth_(child_43874)){
node.appendChild(child_43874);


var G__43878 = cljs.core.next(seq__43183_43868__$1);
var G__43879 = null;
var G__43880 = (0);
var G__43881 = (0);
seq__43183_43849 = G__43878;
chunk__43185_43850 = G__43879;
count__43186_43851 = G__43880;
i__43187_43852 = G__43881;
continue;
} else {
var G__43883 = cljs.core.next(seq__43183_43868__$1);
var G__43884 = null;
var G__43885 = (0);
var G__43886 = (0);
seq__43183_43849 = G__43883;
chunk__43185_43850 = G__43884;
count__43186_43851 = G__43885;
i__43187_43852 = G__43886;
continue;
}
}
} else {
}
}
break;
}
} else {
node.appendChild(children_43848);
}
}


var G__43888 = seq__43126_43841;
var G__43889 = chunk__43128_43842;
var G__43890 = count__43129_43843;
var G__43891 = (i__43130_43844 + (1));
seq__43126_43841 = G__43888;
chunk__43128_43842 = G__43889;
count__43129_43843 = G__43890;
i__43130_43844 = G__43891;
continue;
} else {
var G__43892 = seq__43126_43841;
var G__43893 = chunk__43128_43842;
var G__43894 = count__43129_43843;
var G__43895 = (i__43130_43844 + (1));
seq__43126_43841 = G__43892;
chunk__43128_43842 = G__43893;
count__43129_43843 = G__43894;
i__43130_43844 = G__43895;
continue;
}
} else {
var temp__5735__auto___43897 = cljs.core.seq(seq__43126_43841);
if(temp__5735__auto___43897){
var seq__43126_43898__$1 = temp__5735__auto___43897;
if(cljs.core.chunked_seq_QMARK_(seq__43126_43898__$1)){
var c__4556__auto___43899 = cljs.core.chunk_first(seq__43126_43898__$1);
var G__43900 = cljs.core.chunk_rest(seq__43126_43898__$1);
var G__43901 = c__4556__auto___43899;
var G__43902 = cljs.core.count(c__4556__auto___43899);
var G__43903 = (0);
seq__43126_43841 = G__43900;
chunk__43128_43842 = G__43901;
count__43129_43843 = G__43902;
i__43130_43844 = G__43903;
continue;
} else {
var child_struct_43904 = cljs.core.first(seq__43126_43898__$1);
if((!((child_struct_43904 == null)))){
if(typeof child_struct_43904 === 'string'){
var text_43906 = (node["textContent"]);
(node["textContent"] = [cljs.core.str.cljs$core$IFn$_invoke$arity$1(text_43906),child_struct_43904].join(''));
} else {
var children_43911 = shadow.dom.svg_node(child_struct_43904);
if(cljs.core.seq_QMARK_(children_43911)){
var seq__43209_43912 = cljs.core.seq(children_43911);
var chunk__43211_43913 = null;
var count__43212_43914 = (0);
var i__43213_43915 = (0);
while(true){
if((i__43213_43915 < count__43212_43914)){
var child_43916 = chunk__43211_43913.cljs$core$IIndexed$_nth$arity$2(null,i__43213_43915);
if(cljs.core.truth_(child_43916)){
node.appendChild(child_43916);


var G__43917 = seq__43209_43912;
var G__43918 = chunk__43211_43913;
var G__43919 = count__43212_43914;
var G__43920 = (i__43213_43915 + (1));
seq__43209_43912 = G__43917;
chunk__43211_43913 = G__43918;
count__43212_43914 = G__43919;
i__43213_43915 = G__43920;
continue;
} else {
var G__43921 = seq__43209_43912;
var G__43922 = chunk__43211_43913;
var G__43923 = count__43212_43914;
var G__43924 = (i__43213_43915 + (1));
seq__43209_43912 = G__43921;
chunk__43211_43913 = G__43922;
count__43212_43914 = G__43923;
i__43213_43915 = G__43924;
continue;
}
} else {
var temp__5735__auto___43925__$1 = cljs.core.seq(seq__43209_43912);
if(temp__5735__auto___43925__$1){
var seq__43209_43926__$1 = temp__5735__auto___43925__$1;
if(cljs.core.chunked_seq_QMARK_(seq__43209_43926__$1)){
var c__4556__auto___43927 = cljs.core.chunk_first(seq__43209_43926__$1);
var G__43952 = cljs.core.chunk_rest(seq__43209_43926__$1);
var G__43953 = c__4556__auto___43927;
var G__43954 = cljs.core.count(c__4556__auto___43927);
var G__43955 = (0);
seq__43209_43912 = G__43952;
chunk__43211_43913 = G__43953;
count__43212_43914 = G__43954;
i__43213_43915 = G__43955;
continue;
} else {
var child_43956 = cljs.core.first(seq__43209_43926__$1);
if(cljs.core.truth_(child_43956)){
node.appendChild(child_43956);


var G__43958 = cljs.core.next(seq__43209_43926__$1);
var G__43959 = null;
var G__43960 = (0);
var G__43961 = (0);
seq__43209_43912 = G__43958;
chunk__43211_43913 = G__43959;
count__43212_43914 = G__43960;
i__43213_43915 = G__43961;
continue;
} else {
var G__43963 = cljs.core.next(seq__43209_43926__$1);
var G__43964 = null;
var G__43965 = (0);
var G__43966 = (0);
seq__43209_43912 = G__43963;
chunk__43211_43913 = G__43964;
count__43212_43914 = G__43965;
i__43213_43915 = G__43966;
continue;
}
}
} else {
}
}
break;
}
} else {
node.appendChild(children_43911);
}
}


var G__43968 = cljs.core.next(seq__43126_43898__$1);
var G__43969 = null;
var G__43970 = (0);
var G__43971 = (0);
seq__43126_43841 = G__43968;
chunk__43128_43842 = G__43969;
count__43129_43843 = G__43970;
i__43130_43844 = G__43971;
continue;
} else {
var G__43972 = cljs.core.next(seq__43126_43898__$1);
var G__43973 = null;
var G__43974 = (0);
var G__43975 = (0);
seq__43126_43841 = G__43972;
chunk__43128_43842 = G__43973;
count__43129_43843 = G__43974;
i__43130_43844 = G__43975;
continue;
}
}
} else {
}
}
break;
}

return node;
});
goog.object.set(shadow.dom.SVGElement,"string",true);

goog.object.set(shadow.dom._to_svg,"string",(function (this$){
if((this$ instanceof cljs.core.Keyword)){
return shadow.dom.make_svg_node(new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [this$], null));
} else {
throw cljs.core.ex_info.cljs$core$IFn$_invoke$arity$2("strings cannot be in svgs",new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"this","this",-611633625),this$], null));
}
}));

(cljs.core.PersistentVector.prototype.shadow$dom$SVGElement$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentVector.prototype.shadow$dom$SVGElement$_to_svg$arity$1 = (function (this$){
var this$__$1 = this;
return shadow.dom.make_svg_node(this$__$1);
}));

(cljs.core.LazySeq.prototype.shadow$dom$SVGElement$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.LazySeq.prototype.shadow$dom$SVGElement$_to_svg$arity$1 = (function (this$){
var this$__$1 = this;
return cljs.core.map.cljs$core$IFn$_invoke$arity$2(shadow.dom._to_svg,this$__$1);
}));

goog.object.set(shadow.dom.SVGElement,"null",true);

goog.object.set(shadow.dom._to_svg,"null",(function (_){
return null;
}));
shadow.dom.svg = (function shadow$dom$svg(var_args){
var args__4742__auto__ = [];
var len__4736__auto___43980 = arguments.length;
var i__4737__auto___43981 = (0);
while(true){
if((i__4737__auto___43981 < len__4736__auto___43980)){
args__4742__auto__.push((arguments[i__4737__auto___43981]));

var G__43985 = (i__4737__auto___43981 + (1));
i__4737__auto___43981 = G__43985;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return shadow.dom.svg.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(shadow.dom.svg.cljs$core$IFn$_invoke$arity$variadic = (function (attrs,children){
return shadow.dom._to_svg(cljs.core.vec(cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"svg","svg",856789142),attrs], null),children)));
}));

(shadow.dom.svg.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(shadow.dom.svg.cljs$lang$applyTo = (function (seq43228){
var G__43229 = cljs.core.first(seq43228);
var seq43228__$1 = cljs.core.next(seq43228);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__43229,seq43228__$1);
}));

/**
 * returns a channel for events on el
 * transform-fn should be a (fn [e el] some-val) where some-val will be put on the chan
 * once-or-cleanup handles the removal of the event handler
 * - true: remove after one event
 * - false: never removed
 * - chan: remove on msg/close
 */
shadow.dom.event_chan = (function shadow$dom$event_chan(var_args){
var G__43240 = arguments.length;
switch (G__43240) {
case 2:
return shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
case 4:
return shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$2 = (function (el,event){
return shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$4(el,event,null,false);
}));

(shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$3 = (function (el,event,xf){
return shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$4(el,event,xf,false);
}));

(shadow.dom.event_chan.cljs$core$IFn$_invoke$arity$4 = (function (el,event,xf,once_or_cleanup){
var buf = cljs.core.async.sliding_buffer((1));
var chan = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$2(buf,xf);
var event_fn = (function shadow$dom$event_fn(e){
cljs.core.async.put_BANG_.cljs$core$IFn$_invoke$arity$2(chan,e);

if(once_or_cleanup === true){
shadow.dom.remove_event_handler(el,event,shadow$dom$event_fn);

return cljs.core.async.close_BANG_(chan);
} else {
return null;
}
});
shadow.dom.dom_listen(shadow.dom.dom_node(el),cljs.core.name(event),event_fn);

if(cljs.core.truth_((function (){var and__4115__auto__ = once_or_cleanup;
if(cljs.core.truth_(and__4115__auto__)){
return (!(once_or_cleanup === true));
} else {
return and__4115__auto__;
}
})())){
var c__39440__auto___43995 = cljs.core.async.chan.cljs$core$IFn$_invoke$arity$1((1));
cljs.core.async.impl.dispatch.run((function (){
var f__39441__auto__ = (function (){var switch__39233__auto__ = (function (state_43254){
var state_val_43255 = (state_43254[(1)]);
if((state_val_43255 === (1))){
var state_43254__$1 = state_43254;
return cljs.core.async.impl.ioc_helpers.take_BANG_(state_43254__$1,(2),once_or_cleanup);
} else {
if((state_val_43255 === (2))){
var inst_43250 = (state_43254[(2)]);
var inst_43251 = shadow.dom.remove_event_handler(el,event,event_fn);
var state_43254__$1 = (function (){var statearr_43268 = state_43254;
(statearr_43268[(7)] = inst_43250);

return statearr_43268;
})();
return cljs.core.async.impl.ioc_helpers.return_chan(state_43254__$1,inst_43251);
} else {
return null;
}
}
});
return (function() {
var shadow$dom$state_machine__39234__auto__ = null;
var shadow$dom$state_machine__39234__auto____0 = (function (){
var statearr_43274 = [null,null,null,null,null,null,null,null];
(statearr_43274[(0)] = shadow$dom$state_machine__39234__auto__);

(statearr_43274[(1)] = (1));

return statearr_43274;
});
var shadow$dom$state_machine__39234__auto____1 = (function (state_43254){
while(true){
var ret_value__39235__auto__ = (function (){try{while(true){
var result__39236__auto__ = switch__39233__auto__(state_43254);
if(cljs.core.keyword_identical_QMARK_(result__39236__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__39236__auto__;
}
break;
}
}catch (e43277){var ex__39237__auto__ = e43277;
var statearr_43278_43999 = state_43254;
(statearr_43278_43999[(2)] = ex__39237__auto__);


if(cljs.core.seq((state_43254[(4)]))){
var statearr_43279_44000 = state_43254;
(statearr_43279_44000[(1)] = cljs.core.first((state_43254[(4)])));

} else {
throw ex__39237__auto__;
}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
}})();
if(cljs.core.keyword_identical_QMARK_(ret_value__39235__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__44001 = state_43254;
state_43254 = G__44001;
continue;
} else {
return ret_value__39235__auto__;
}
break;
}
});
shadow$dom$state_machine__39234__auto__ = function(state_43254){
switch(arguments.length){
case 0:
return shadow$dom$state_machine__39234__auto____0.call(this);
case 1:
return shadow$dom$state_machine__39234__auto____1.call(this,state_43254);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
shadow$dom$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$0 = shadow$dom$state_machine__39234__auto____0;
shadow$dom$state_machine__39234__auto__.cljs$core$IFn$_invoke$arity$1 = shadow$dom$state_machine__39234__auto____1;
return shadow$dom$state_machine__39234__auto__;
})()
})();
var state__39442__auto__ = (function (){var statearr_43283 = f__39441__auto__();
(statearr_43283[(6)] = c__39440__auto___43995);

return statearr_43283;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped(state__39442__auto__);
}));

} else {
}

return chan;
}));

(shadow.dom.event_chan.cljs$lang$maxFixedArity = 4);


//# sourceMappingURL=shadow.dom.js.map
