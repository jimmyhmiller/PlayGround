goog.provide('crate.core');
crate.core.group_id = cljs.core.atom.cljs$core$IFn$_invoke$arity$1((0));
crate.core.raw = (function crate$core$raw(html_str){
return goog.dom.htmlToDocumentFragment(html_str);
});
crate.core.html = (function crate$core$html(var_args){
var args__4742__auto__ = [];
var len__4736__auto___47262 = arguments.length;
var i__4737__auto___47263 = (0);
while(true){
if((i__4737__auto___47263 < len__4736__auto___47262)){
args__4742__auto__.push((arguments[i__4737__auto___47263]));

var G__47264 = (i__4737__auto___47263 + (1));
i__4737__auto___47263 = G__47264;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((0) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((0)),(0),null)):null);
return crate.core.html.cljs$core$IFn$_invoke$arity$variadic(argseq__4743__auto__);
});

(crate.core.html.cljs$core$IFn$_invoke$arity$variadic = (function (tags){
var res = cljs.core.map.cljs$core$IFn$_invoke$arity$2(crate.compiler.elem_factory,tags);
if(cljs.core.truth_(cljs.core.second(res))){
return res;
} else {
return cljs.core.first(res);
}
}));

(crate.core.html.cljs$lang$maxFixedArity = (0));

/** @this {Function} */
(crate.core.html.cljs$lang$applyTo = (function (seq47248){
var self__4724__auto__ = this;
return self__4724__auto__.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq(seq47248));
}));

/**
 * Alias for crate.util/escape-html
 */
crate.core.h = crate.util.escape_html;

//# sourceMappingURL=crate.core.js.map
