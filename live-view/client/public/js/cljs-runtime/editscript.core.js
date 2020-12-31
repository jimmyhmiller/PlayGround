goog.provide('editscript.core');
/**
 * Create an editscript to represent the transformations needed to turn a
 *   Clojure data structure `a` into another Clojure data structure `b`.
 * 
 *   This function accepts any nested Clojure data structures. In Clojure, those
 *   implement `IPersistentVector`, `IPersistentMap`, `IPersistentList`,
 *   and `IPersistentSet` will be treated as collections. The same are true for
 *   the corresponding deftypes in Clojurescript, such as `PersistentVector`,
 *   `PersistentMap`, and so on. Anything else are treated as atomic values.
 * 
 *   The editscript is represented as a vector of basic operations: add `:+`,
 *   delete `:-`, and replace `:r`. Each operation also include a path to the
 *   location of the operation, which is similar to the path vector in `update-in`.
 *   However, editscript path works for all above four collection types, not just
 *   associative ones. For `:+` and `:r`, a new value is also required.
 * 
 * 
 *   The following options are supported in the option map:
 * 
 *   * `:algo`  chooses the diff algorithm. The value can be `:a-star` (default) or `:quick`; `:a-star` algorithm minimize the size of the resulting editscript, `:quick` algorithm is much faster, but does not producing diff with minimal size.
 * 
 *   * `:str-diff?` determines if to perform string diff, string diff may reduce the result size for small changes in long strings, but will incur a slight computation cost. The value is a boolean: `true` or `false` (default) 
 */
editscript.core.diff = (function editscript$core$diff(var_args){
var G__47322 = arguments.length;
switch (G__47322) {
case 2:
return editscript.core.diff.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return editscript.core.diff.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(editscript.core.diff.cljs$core$IFn$_invoke$arity$2 = (function (a,b){
return editscript.core.diff.cljs$core$IFn$_invoke$arity$3(a,b,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"algo","algo",1472048382),new cljs.core.Keyword(null,"a-star","a-star",-171330865),new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760),false], null));
}));

(editscript.core.diff.cljs$core$IFn$_invoke$arity$3 = (function (a,b,p__47323){
var map__47324 = p__47323;
var map__47324__$1 = (((((!((map__47324 == null))))?(((((map__47324.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__47324.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__47324):map__47324);
var algo = cljs.core.get.cljs$core$IFn$_invoke$arity$3(map__47324__$1,new cljs.core.Keyword(null,"algo","algo",1472048382),new cljs.core.Keyword(null,"a-star","a-star",-171330865));
var str_diff_QMARK_ = cljs.core.get.cljs$core$IFn$_invoke$arity$3(map__47324__$1,new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760),false);
return editscript.util.common.diff_algo.cljs$core$IFn$_invoke$arity$3(a,b,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"algo","algo",1472048382),algo,new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760),str_diff_QMARK_], null));
}));

(editscript.core.diff.cljs$lang$maxFixedArity = 3);

/**
 * Apply the editscript `script` on `a` to produce `b`, assuming the
 *   script is the results of running  `(diff a b)`, such that
 *   `(= b (patch a (diff a b)))` is true
 */
editscript.core.patch = (function editscript$core$patch(a,script){
if((script instanceof editscript.edit.EditScript)){
} else {
throw (new Error("Assert failed: (instance? editscript.edit.EditScript script)"));
}

return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (p1__47326_SHARP_,p2__47327_SHARP_){
return editscript.patch.patch_STAR_(p1__47326_SHARP_,p2__47327_SHARP_);
}),a,editscript.edit.get_edits(script));
});

//# sourceMappingURL=editscript.core.js.map
