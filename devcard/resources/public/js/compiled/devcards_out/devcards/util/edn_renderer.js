// Compiled by ClojureScript 1.7.170 {}
goog.provide('devcards.util.edn_renderer');
goog.require('cljs.core');
goog.require('sablono.core');
goog.require('devcards.util.utils');
if(typeof devcards.util.edn_renderer._STAR_key_counter_STAR_ !== 'undefined'){
} else {
devcards.util.edn_renderer._STAR_key_counter_STAR_ = null;
}
devcards.util.edn_renderer.get_key = (function devcards$util$edn_renderer$get_key(){
cljs.core.swap_BANG_.call(null,devcards.util.edn_renderer._STAR_key_counter_STAR_,cljs.core.inc);

return [cljs.core.str("k-"),cljs.core.str(cljs.core.deref.call(null,devcards.util.edn_renderer._STAR_key_counter_STAR_))].join('');
});
devcards.util.edn_renderer.html;
devcards.util.edn_renderer.literal_QMARK_ = (function devcards$util$edn_renderer$literal_QMARK_(x){
return (!(cljs.core.seq_QMARK_.call(null,x))) && (!(cljs.core.coll_QMARK_.call(null,x)));
});
devcards.util.edn_renderer.separator_STAR_ = (function devcards$util$edn_renderer$separator_STAR_(s){
return React.createElement("span",{"key": devcards.util.edn_renderer.get_key.call(null), "className": "seperator"},sablono.interpreter.interpret.call(null,s));
});
devcards.util.edn_renderer.clearfix_separator_STAR_ = (function devcards$util$edn_renderer$clearfix_separator_STAR_(s){
return React.createElement("span",{"key": devcards.util.edn_renderer.get_key.call(null)},sablono.interpreter.interpret.call(null,devcards.util.edn_renderer.separator_STAR_.call(null,s)),React.createElement("span",{"className": "clearfix"}));
});
devcards.util.edn_renderer.separate_fn = (function devcards$util$edn_renderer$separate_fn(coll){
if(!(cljs.core.every_QMARK_.call(null,devcards.util.edn_renderer.literal_QMARK_,coll))){
return devcards.util.edn_renderer.clearfix_separator_STAR_;
} else {
return devcards.util.edn_renderer.separator_STAR_;
}
});
devcards.util.edn_renderer.interpose_separator = (function devcards$util$edn_renderer$interpose_separator(rct_coll,s,sep_fn){
return cljs.core.to_array.call(null,cljs.core.cons.call(null,cljs.core.first.call(null,rct_coll),cljs.core.interleave.call(null,cljs.core.repeatedly.call(null,(function (){
return sep_fn.call(null,s);
})),cljs.core.rest.call(null,rct_coll))));
});
devcards.util.edn_renderer.literal = (function devcards$util$edn_renderer$literal(class$,x){
return React.createElement("span",{"className": class$, "key": devcards.util.edn_renderer.get_key.call(null)},sablono.interpreter.interpret.call(null,devcards.util.utils.pprint_str.call(null,x)));
});
devcards.util.edn_renderer.html_val = (function devcards$util$edn_renderer$html_val(index,v){
return React.createElement("span",{"key": index},sablono.interpreter.interpret.call(null,devcards.util.edn_renderer.html.call(null,v)));
});
devcards.util.edn_renderer.join_html = (function devcards$util$edn_renderer$join_html(separator,coll){
return devcards.util.edn_renderer.interpose_separator.call(null,cljs.core.into.call(null,cljs.core.PersistentVector.EMPTY,cljs.core.map_indexed.call(null,devcards.util.edn_renderer.html_val,coll)),separator,devcards.util.edn_renderer.separate_fn.call(null,coll));
});
devcards.util.edn_renderer.html_keyval = (function devcards$util$edn_renderer$html_keyval(p__23726){
var vec__23728 = p__23726;
var k = cljs.core.nth.call(null,vec__23728,(0),null);
var v = cljs.core.nth.call(null,vec__23728,(1),null);
return React.createElement("span",{"key": cljs.core.prn_str.call(null,k), "className": "keyval"},sablono.interpreter.interpret.call(null,devcards.util.edn_renderer.html.call(null,k)),sablono.interpreter.interpret.call(null,devcards.util.edn_renderer.html.call(null,v)));
});
devcards.util.edn_renderer.html_keyvals = (function devcards$util$edn_renderer$html_keyvals(coll){
return devcards.util.edn_renderer.interpose_separator.call(null,cljs.core.mapv.call(null,devcards.util.edn_renderer.html_keyval,coll)," ",devcards.util.edn_renderer.separate_fn.call(null,cljs.core.vals.call(null,coll)));
});
devcards.util.edn_renderer.open_close = (function devcards$util$edn_renderer$open_close(class_str,opener,closer,rct_coll){
return React.createElement("span",{"className": class_str, "key": [cljs.core.str(cljs.core.hash.call(null,rct_coll))].join('')},React.createElement("span",{"key": (1), "className": "opener"},sablono.interpreter.interpret.call(null,opener)),React.createElement("span",{"key": (2), "className": "contents"},sablono.interpreter.interpret.call(null,rct_coll)),React.createElement("span",{"key": (3), "className": "closer"},sablono.interpreter.interpret.call(null,closer)));
});
devcards.util.edn_renderer.html_collection = (function devcards$util$edn_renderer$html_collection(class$,opener,closer,coll){
return devcards.util.edn_renderer.open_close.call(null,[cljs.core.str("collection "),cljs.core.str(class$)].join(''),opener,closer,devcards.util.edn_renderer.join_html.call(null," ",coll));
});
devcards.util.edn_renderer.html_map = (function devcards$util$edn_renderer$html_map(coll){
return devcards.util.edn_renderer.open_close.call(null,"collection map","{","}",devcards.util.edn_renderer.html_keyvals.call(null,coll));
});
devcards.util.edn_renderer.html_string = (function devcards$util$edn_renderer$html_string(s){
return devcards.util.edn_renderer.open_close.call(null,"string","\"","\"",s);
});
devcards.util.edn_renderer.html = (function devcards$util$edn_renderer$html(x){
if(typeof x === 'number'){
return devcards.util.edn_renderer.literal.call(null,"number",x);
} else {
if((x instanceof cljs.core.Keyword)){
return devcards.util.edn_renderer.literal.call(null,"keyword",x);
} else {
if((x instanceof cljs.core.Symbol)){
return devcards.util.edn_renderer.literal.call(null,"symbol",x);
} else {
if(typeof x === 'string'){
return devcards.util.edn_renderer.html_string.call(null,x);
} else {
if(cljs.core.map_QMARK_.call(null,x)){
return devcards.util.edn_renderer.html_map.call(null,x);
} else {
if(cljs.core.set_QMARK_.call(null,x)){
return devcards.util.edn_renderer.html_collection.call(null,"set","#{","}",x);
} else {
if(cljs.core.vector_QMARK_.call(null,x)){
return devcards.util.edn_renderer.html_collection.call(null,"vector","[","]",x);
} else {
if(cljs.core.seq_QMARK_.call(null,x)){
return devcards.util.edn_renderer.html_collection.call(null,"seq","(",")",x);
} else {
return devcards.util.edn_renderer.literal.call(null,"literal",x);

}
}
}
}
}
}
}
}
});
devcards.util.edn_renderer.html_edn = (function devcards$util$edn_renderer$html_edn(e){
var _STAR_key_counter_STAR_23731 = devcards.util.edn_renderer._STAR_key_counter_STAR_;
devcards.util.edn_renderer._STAR_key_counter_STAR_ = cljs.core.atom.call(null,(0));

try{var attrs23732 = devcards.util.edn_renderer.html.call(null,e);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs23732))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["com-rigsomelight-devcards-typog",null,"com-rigsomelight-rendered-edn",null], null), null)], null),attrs23732)):{"className": "com-rigsomelight-devcards-typog com-rigsomelight-rendered-edn"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs23732))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs23732)], null))));
}finally {devcards.util.edn_renderer._STAR_key_counter_STAR_ = _STAR_key_counter_STAR_23731;
}});

//# sourceMappingURL=edn_renderer.js.map?rel=1454621291661