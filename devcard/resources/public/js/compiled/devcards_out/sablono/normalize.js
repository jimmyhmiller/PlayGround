// Compiled by ClojureScript 1.7.170 {}
goog.provide('sablono.normalize');
goog.require('cljs.core');
goog.require('clojure.set');
goog.require('clojure.string');
goog.require('sablono.util');
/**
 * Removes all map entries where the value of the entry is empty.
 */
sablono.normalize.compact_map = (function sablono$normalize$compact_map(m){
return cljs.core.reduce.call(null,(function (m__$1,k){
var v = cljs.core.get.call(null,m__$1,k);
if(cljs.core.empty_QMARK_.call(null,v)){
return cljs.core.dissoc.call(null,m__$1,k);
} else {
return m__$1;
}
}),m,cljs.core.keys.call(null,m));
});
sablono.normalize.class_name = (function sablono$normalize$class_name(x){
if(typeof x === 'string'){
return x;
} else {
if((x instanceof cljs.core.Keyword)){
return cljs.core.name.call(null,x);
} else {
return x;

}
}
});
/**
 * Normalize `class` into a set of classes.
 */
sablono.normalize.class$ = (function sablono$normalize$class(class$__$1){
if((class$__$1 == null)){
return null;
} else {
if(cljs.core.list_QMARK_.call(null,class$__$1)){
if((cljs.core.first.call(null,class$__$1) instanceof cljs.core.Symbol)){
return cljs.core.PersistentHashSet.fromArray([class$__$1], true);
} else {
return cljs.core.set.call(null,cljs.core.map.call(null,sablono.normalize.class_name,class$__$1));
}
} else {
if((class$__$1 instanceof cljs.core.Symbol)){
return cljs.core.PersistentHashSet.fromArray([class$__$1], true);
} else {
if(typeof class$__$1 === 'string'){
return cljs.core.PersistentHashSet.fromArray([class$__$1], true);
} else {
if((class$__$1 instanceof cljs.core.Keyword)){
return cljs.core.PersistentHashSet.fromArray([sablono.normalize.class_name.call(null,class$__$1)], true);
} else {
if(((cljs.core.set_QMARK_.call(null,class$__$1)) || (cljs.core.sequential_QMARK_.call(null,class$__$1))) && (cljs.core.every_QMARK_.call(null,(function (p1__22984_SHARP_){
return ((p1__22984_SHARP_ instanceof cljs.core.Keyword)) || (typeof p1__22984_SHARP_ === 'string');
}),class$__$1))){
return cljs.core.apply.call(null,cljs.core.sorted_set,cljs.core.map.call(null,sablono.normalize.class_name,class$__$1));
} else {
if((cljs.core.set_QMARK_.call(null,class$__$1)) || (cljs.core.sequential_QMARK_.call(null,class$__$1))){
return cljs.core.set.call(null,cljs.core.map.call(null,sablono.normalize.class_name,class$__$1));
} else {
return class$__$1;

}
}
}
}
}
}
}
});
/**
 * Normalize the `attrs` of an element.
 */
sablono.normalize.attributes = (function sablono$normalize$attributes(attrs){
var G__22986 = attrs;
var G__22986__$1 = (cljs.core.truth_(new cljs.core.Keyword(null,"class","class",-2030961996).cljs$core$IFn$_invoke$arity$1(attrs))?cljs.core.update_in.call(null,G__22986,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"class","class",-2030961996)], null),sablono.normalize.class$):G__22986);
return G__22986__$1;
});
/**
 * Like clojure.core/merge but concatenate :class entries.
 */
sablono.normalize.merge_with_class = (function sablono$normalize$merge_with_class(var_args){
var args__17831__auto__ = [];
var len__17824__auto___22990 = arguments.length;
var i__17825__auto___22991 = (0);
while(true){
if((i__17825__auto___22991 < len__17824__auto___22990)){
args__17831__auto__.push((arguments[i__17825__auto___22991]));

var G__22992 = (i__17825__auto___22991 + (1));
i__17825__auto___22991 = G__22992;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return sablono.normalize.merge_with_class.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

sablono.normalize.merge_with_class.cljs$core$IFn$_invoke$arity$variadic = (function (maps){
var maps__$1 = cljs.core.map.call(null,sablono.normalize.attributes,maps);
var classes = cljs.core.map.call(null,((function (maps__$1){
return (function (p1__22987_SHARP_){
return cljs.core.into.call(null,cljs.core.PersistentHashSet.EMPTY,p1__22987_SHARP_);
});})(maps__$1))
,cljs.core.map.call(null,new cljs.core.Keyword(null,"class","class",-2030961996),maps__$1));
var classes__$1 = cljs.core.apply.call(null,clojure.set.union,classes);
var G__22989 = cljs.core.apply.call(null,cljs.core.merge,maps__$1);
var G__22989__$1 = ((!(cljs.core.empty_QMARK_.call(null,classes__$1)))?cljs.core.assoc.call(null,G__22989,new cljs.core.Keyword(null,"class","class",-2030961996),classes__$1):G__22989);
return G__22989__$1;
});

sablono.normalize.merge_with_class.cljs$lang$maxFixedArity = (0);

sablono.normalize.merge_with_class.cljs$lang$applyTo = (function (seq22988){
return sablono.normalize.merge_with_class.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq22988));
});
/**
 * Strip the # and . characters from the beginning of `s`.
 */
sablono.normalize.strip_css = (function sablono$normalize$strip_css(s){
if(cljs.core.truth_(s)){
return clojure.string.replace.call(null,s,/^[.#]/,"");
} else {
return null;
}
});
/**
 * Match `s` as a CSS tag and return a vector of tag name, CSS id and
 *   CSS classes.
 */
sablono.normalize.match_tag = (function sablono$normalize$match_tag(s){
var matches = cljs.core.re_seq.call(null,/[#.]?[^#.]+/,cljs.core.name.call(null,s));
var vec__22996 = ((cljs.core.empty_QMARK_.call(null,matches))?(function(){throw cljs.core.ex_info.call(null,[cljs.core.str("Can't match CSS tag: "),cljs.core.str(s)].join(''),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),s], null))})():(cljs.core.truth_(new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["#",null,".",null], null), null).call(null,cljs.core.ffirst.call(null,matches)))?new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, ["div",matches], null):new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.first.call(null,matches),cljs.core.rest.call(null,matches)], null)
));
var tag_name = cljs.core.nth.call(null,vec__22996,(0),null);
var names = cljs.core.nth.call(null,vec__22996,(1),null);
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [tag_name,cljs.core.first.call(null,cljs.core.map.call(null,sablono.normalize.strip_css,cljs.core.filter.call(null,((function (matches,vec__22996,tag_name,names){
return (function (p1__22993_SHARP_){
return cljs.core._EQ_.call(null,"#",cljs.core.first.call(null,p1__22993_SHARP_));
});})(matches,vec__22996,tag_name,names))
,names))),cljs.core.vec.call(null,cljs.core.map.call(null,sablono.normalize.strip_css,cljs.core.filter.call(null,((function (matches,vec__22996,tag_name,names){
return (function (p1__22994_SHARP_){
return cljs.core._EQ_.call(null,".",cljs.core.first.call(null,p1__22994_SHARP_));
});})(matches,vec__22996,tag_name,names))
,names)))], null);
});
/**
 * Normalize the children of a HTML element.
 */
sablono.normalize.children = (function sablono$normalize$children(x){
if((cljs.core.sequential_QMARK_.call(null,x)) && (cljs.core._EQ_.call(null,cljs.core.count.call(null,x),(1)))){
var x__$1 = cljs.core.first.call(null,x);
if(typeof x__$1 === 'string'){
return cljs.core._conj.call(null,cljs.core.List.EMPTY,x__$1);
} else {
if(cljs.core.truth_(sablono.util.element_QMARK_.call(null,x__$1))){
return cljs.core._conj.call(null,cljs.core.List.EMPTY,x__$1);
} else {
if(cljs.core.sequential_QMARK_.call(null,x__$1)){
return cljs.core.seq.call(null,x__$1);
} else {
return x__$1;

}
}
}
} else {
return x;
}
});
/**
 * Ensure an element vector is of the form [tag-name attrs content].
 */
sablono.normalize.element = (function sablono$normalize$element(p__22997){
var vec__23000 = p__22997;
var tag = cljs.core.nth.call(null,vec__23000,(0),null);
var content = cljs.core.nthnext.call(null,vec__23000,(1));
if(!(((tag instanceof cljs.core.Keyword)) || ((tag instanceof cljs.core.Symbol)) || (typeof tag === 'string'))){
throw cljs.core.ex_info.call(null,[cljs.core.str(tag),cljs.core.str(" is not a valid element name.")].join(''),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),tag,new cljs.core.Keyword(null,"content","content",15833224),content], null));
} else {
}

var vec__23001 = sablono.normalize.match_tag.call(null,tag);
var tag__$1 = cljs.core.nth.call(null,vec__23001,(0),null);
var id = cljs.core.nth.call(null,vec__23001,(1),null);
var class$ = cljs.core.nth.call(null,vec__23001,(2),null);
var tag_attrs = sablono.normalize.compact_map.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"id","id",-1388402092),id,new cljs.core.Keyword(null,"class","class",-2030961996),class$], null));
var map_attrs = cljs.core.first.call(null,content);
if(cljs.core.map_QMARK_.call(null,map_attrs)){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [tag__$1,sablono.normalize.merge_with_class.call(null,tag_attrs,map_attrs),sablono.normalize.children.call(null,cljs.core.next.call(null,content))], null);
} else {
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [tag__$1,sablono.normalize.attributes.call(null,tag_attrs),sablono.normalize.children.call(null,content)], null);
}
});

//# sourceMappingURL=normalize.js.map?rel=1454621291121