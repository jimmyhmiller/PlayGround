goog.provide('crate.util');
crate.util._STAR_base_url_STAR_ = null;
crate.util.as_str = (function crate$util$as_str(var_args){
var G__46565 = arguments.length;
switch (G__46565) {
case 0:
return crate.util.as_str.cljs$core$IFn$_invoke$arity$0();

break;
case 1:
return crate.util.as_str.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
default:
var args_arr__4757__auto__ = [];
var len__4736__auto___46629 = arguments.length;
var i__4737__auto___46630 = (0);
while(true){
if((i__4737__auto___46630 < len__4736__auto___46629)){
args_arr__4757__auto__.push((arguments[i__4737__auto___46630]));

var G__46631 = (i__4737__auto___46630 + (1));
i__4737__auto___46630 = G__46631;
continue;
} else {
}
break;
}

var argseq__4758__auto__ = (new cljs.core.IndexedSeq(args_arr__4757__auto__.slice((1)),(0),null));
return crate.util.as_str.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4758__auto__);

}
});

(crate.util.as_str.cljs$core$IFn$_invoke$arity$0 = (function (){
return "";
}));

(crate.util.as_str.cljs$core$IFn$_invoke$arity$1 = (function (x){
if((((x instanceof cljs.core.Symbol)) || ((x instanceof cljs.core.Keyword)))){
return cljs.core.name(x);
} else {
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(x);
}
}));

(crate.util.as_str.cljs$core$IFn$_invoke$arity$variadic = (function (x,xs){
return (function (s,more){
while(true){
if(cljs.core.truth_(more)){
var G__46660 = [cljs.core.str.cljs$core$IFn$_invoke$arity$1(s),crate.util.as_str.cljs$core$IFn$_invoke$arity$1(cljs.core.first(more))].join('');
var G__46661 = cljs.core.next(more);
s = G__46660;
more = G__46661;
continue;
} else {
return s;
}
break;
}
})(crate.util.as_str.cljs$core$IFn$_invoke$arity$1(x),xs);
}));

/** @this {Function} */
(crate.util.as_str.cljs$lang$applyTo = (function (seq46563){
var G__46564 = cljs.core.first(seq46563);
var seq46563__$1 = cljs.core.next(seq46563);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46564,seq46563__$1);
}));

(crate.util.as_str.cljs$lang$maxFixedArity = (1));

/**
 * Change special characters into HTML character entities.
 */
crate.util.escape_html = (function crate$util$escape_html(text){
return clojure.string.replace(clojure.string.replace(clojure.string.replace(clojure.string.replace(crate.util.as_str.cljs$core$IFn$_invoke$arity$1(text),"&","&amp;"),"<","&lt;"),">","&gt;"),"\"","&quot;");
});
/**
 * Prepends the base-url to the supplied URI.
 */
crate.util.to_uri = (function crate$util$to_uri(uri){
if(cljs.core.truth_(cljs.core.re_matches(/^\w+:.*/,uri))){
return uri;
} else {
return [cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util._STAR_base_url_STAR_),cljs.core.str.cljs$core$IFn$_invoke$arity$1(uri)].join('');
}
});
crate.util.url_encode_component = (function crate$util$url_encode_component(s){

return encodeURIComponent(crate.util.as_str.cljs$core$IFn$_invoke$arity$1(s));
});
/**
 * Turn a map of parameters into a urlencoded string.
 */
crate.util.url_encode = (function crate$util$url_encode(params){
return clojure.string.join.cljs$core$IFn$_invoke$arity$2("&",(function (){var iter__4529__auto__ = (function crate$util$url_encode_$_iter__46586(s__46587){
return (new cljs.core.LazySeq(null,(function (){
var s__46587__$1 = s__46587;
while(true){
var temp__5735__auto__ = cljs.core.seq(s__46587__$1);
if(temp__5735__auto__){
var s__46587__$2 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(s__46587__$2)){
var c__4527__auto__ = cljs.core.chunk_first(s__46587__$2);
var size__4528__auto__ = cljs.core.count(c__4527__auto__);
var b__46589 = cljs.core.chunk_buffer(size__4528__auto__);
if((function (){var i__46588 = (0);
while(true){
if((i__46588 < size__4528__auto__)){
var vec__46593 = cljs.core._nth(c__4527__auto__,i__46588);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46593,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46593,(1),null);
cljs.core.chunk_append(b__46589,[cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util.url_encode_component(k)),"=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util.url_encode_component(v))].join(''));

var G__46665 = (i__46588 + (1));
i__46588 = G__46665;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons(cljs.core.chunk(b__46589),crate$util$url_encode_$_iter__46586(cljs.core.chunk_rest(s__46587__$2)));
} else {
return cljs.core.chunk_cons(cljs.core.chunk(b__46589),null);
}
} else {
var vec__46602 = cljs.core.first(s__46587__$2);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46602,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46602,(1),null);
return cljs.core.cons([cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util.url_encode_component(k)),"=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util.url_encode_component(v))].join(''),crate$util$url_encode_$_iter__46586(cljs.core.rest(s__46587__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__4529__auto__(params);
})());
});
/**
 * Creates a URL string from a variable list of arguments and an optional
 *   parameter map as the last argument. For example:
 *  (url "/group/" 4 "/products" {:page 9})
 *  => "/group/4/products?page=9"
 */
crate.util.url = (function crate$util$url(var_args){
var args__4742__auto__ = [];
var len__4736__auto___46666 = arguments.length;
var i__4737__auto___46667 = (0);
while(true){
if((i__4737__auto___46667 < len__4736__auto___46666)){
args__4742__auto__.push((arguments[i__4737__auto___46667]));

var G__46668 = (i__4737__auto___46667 + (1));
i__4737__auto___46667 = G__46668;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((0) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((0)),(0),null)):null);
return crate.util.url.cljs$core$IFn$_invoke$arity$variadic(argseq__4743__auto__);
});

(crate.util.url.cljs$core$IFn$_invoke$arity$variadic = (function (args){
var params = cljs.core.last(args);
var args__$1 = cljs.core.butlast(args);
return cljs.core.str.cljs$core$IFn$_invoke$arity$1(crate.util.to_uri([cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.str,args__$1)),cljs.core.str.cljs$core$IFn$_invoke$arity$1(((cljs.core.map_QMARK_(params))?["?",crate.util.url_encode(params)].join(''):params))].join('')));
}));

(crate.util.url.cljs$lang$maxFixedArity = (0));

/** @this {Function} */
(crate.util.url.cljs$lang$applyTo = (function (seq46606){
var self__4724__auto__ = this;
return self__4724__auto__.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq(seq46606));
}));


//# sourceMappingURL=crate.util.js.map
