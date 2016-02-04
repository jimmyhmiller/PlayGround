// Compiled by ClojureScript 1.7.170 {}
goog.provide('devcards.core');
goog.require('cljs.core');
goog.require('devcards.util.edn_renderer');
goog.require('devcards.util.utils');
goog.require('devcards.system');
goog.require('cljs.core.async');
goog.require('cljs.test');
goog.require('devcards.util.markdown');
goog.require('sablono.core');
goog.require('clojure.string');
cljs.core.enable_console_print_BANG_.call(null);
if(typeof devcards.core.devcard_event_chan !== 'undefined'){
} else {
devcards.core.devcard_event_chan = cljs.core.async.chan.call(null);
}
/**
 * Make a react Symbol the same way as React 0.14
 */
devcards.core.react_element_type_symbol = (function (){var or__16766__auto__ = (function (){var and__16754__auto__ = typeof Symbol !== 'undefined';
if(and__16754__auto__){
var and__16754__auto____$1 = cljs.core.fn_QMARK_.call(null,Symbol);
if(and__16754__auto____$1){
var and__16754__auto____$2 = (Symbol["for"]);
if(cljs.core.truth_(and__16754__auto____$2)){
return (Symbol["for"]).call(null,"react.element");
} else {
return and__16754__auto____$2;
}
} else {
return and__16754__auto____$1;
}
} else {
return and__16754__auto__;
}
})();
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return (60103);
}
})();
/**
 * This event doesn't need to be fired for the system to run. It will just render
 * a little faster on reload if it is fired. Figwheel isn't required to run devcards.
 */
devcards.core.register_figwheel_listeners_BANG_ = (function devcards$core$register_figwheel_listeners_BANG_(){
if(typeof devcards.core.register_listeners_fig !== 'undefined'){
return null;
} else {
devcards.core.register_listeners_fig = (function (){
document.body.addEventListener("figwheel.js-reload",(function (p1__19577_SHARP_){
return cljs.core.async.put_BANG_.call(null,devcards.core.devcard_event_chan,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"jsreload","jsreload",331693051),p1__19577_SHARP_.detail], null));
}));

return true;
})()
;
}
});
devcards.core.assert_options_map = (function devcards$core$assert_options_map(m){
if(!(((m == null)) || (cljs.core.map_QMARK_.call(null,m)))){
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"propagated-errors","propagated-errors",1359777293),new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"label","label",1718410804),new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"message","message",-406056002),"should be a Map or nil.",new cljs.core.Keyword(null,"value","value",305978217),m], null)], null)], null);
} else {
return m;
}
});
devcards.core.start_devcard_ui_BANG__STAR_ = (function devcards$core$start_devcard_ui_BANG__STAR_(var_args){
var args19578 = [];
var len__17824__auto___19581 = arguments.length;
var i__17825__auto___19582 = (0);
while(true){
if((i__17825__auto___19582 < len__17824__auto___19581)){
args19578.push((arguments[i__17825__auto___19582]));

var G__19583 = (i__17825__auto___19582 + (1));
i__17825__auto___19582 = G__19583;
continue;
} else {
}
break;
}

var G__19580 = args19578.length;
switch (G__19580) {
case 0:
return devcards.core.start_devcard_ui_BANG__STAR_.cljs$core$IFn$_invoke$arity$0();

break;
case 1:
return devcards.core.start_devcard_ui_BANG__STAR_.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19578.length)].join('')));

}
});

devcards.core.start_devcard_ui_BANG__STAR_.cljs$core$IFn$_invoke$arity$0 = (function (){
return devcards.core.start_devcard_ui_BANG__STAR_.call(null,cljs.core.PersistentArrayMap.EMPTY);
});

devcards.core.start_devcard_ui_BANG__STAR_.cljs$core$IFn$_invoke$arity$1 = (function (options){
if((cljs.core.map_QMARK_.call(null,options)) && (cljs.core.map_QMARK_.call(null,new cljs.core.Keyword(null,"default-card-options","default-card-options",1708667352).cljs$core$IFn$_invoke$arity$1(options)))){
cljs.core.swap_BANG_.call(null,devcards.system.app_state,cljs.core.update_in,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"base-card-options","base-card-options",141017756)], null),(function (opts){
return cljs.core.merge.call(null,opts,new cljs.core.Keyword(null,"default-card-options","default-card-options",1708667352).cljs$core$IFn$_invoke$arity$1(options));
}));
} else {
}

devcards.system.start_ui.call(null,devcards.core.devcard_event_chan);

return devcards.core.register_figwheel_listeners_BANG_.call(null);
});

devcards.core.start_devcard_ui_BANG__STAR_.cljs$lang$maxFixedArity = 1;
devcards.core.card_QMARK_ = (function devcards$core$card_QMARK_(c){
var and__16754__auto__ = cljs.core.map_QMARK_.call(null,c);
if(and__16754__auto__){
var map__19591 = c;
var map__19591__$1 = ((((!((map__19591 == null)))?((((map__19591.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19591.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19591):map__19591);
var path = cljs.core.get.call(null,map__19591__$1,new cljs.core.Keyword(null,"path","path",-188191168));
var func = cljs.core.get.call(null,map__19591__$1,new cljs.core.Keyword(null,"func","func",-238706040));
cljs.core.vector_QMARK_.call(null,path);

cljs.core.not_empty.call(null,path);

cljs.core.every_QMARK_.call(null,cljs.core.keyword_QMARK_,path);

return cljs.core.fn_QMARK_.call(null,func);
} else {
return and__16754__auto__;
}
});
devcards.core.register_card = (function devcards$core$register_card(c){
if(cljs.core.truth_(devcards.core.card_QMARK_.call(null,c))){
} else {
throw (new Error([cljs.core.str("Assert failed: "),cljs.core.str(cljs.core.pr_str.call(null,cljs.core.list(new cljs.core.Symbol(null,"card?","card?",2082377665,null),new cljs.core.Symbol(null,"c","c",-122660552,null))))].join('')));
}


return cljs.core.async.put_BANG_.call(null,devcards.core.devcard_event_chan,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"register-card","register-card",-1375971588),c], null));
});
devcards.core.react_raw = (function devcards$core$react_raw(raw_html_str){

return React.DOM.div(cljs.core.clj__GT_js.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),[cljs.core.str(cljs.core.hash.call(null,raw_html_str))].join(''),new cljs.core.Keyword(null,"dangerouslySetInnerHTML","dangerouslySetInnerHTML",-554971138),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"__html","__html",674048345),raw_html_str], null)], null)));
});
devcards.core.get_props;

devcards.core.ref__GT_node;
devcards.core.get_hljs = (function devcards$core$get_hljs(){
return (goog.global["hljs"]);
});
devcards.core.highlight_node = (function devcards$core$highlight_node(this$){
var temp__4425__auto__ = devcards.core.ref__GT_node.call(null,this$,"code-ref");
if(cljs.core.truth_(temp__4425__auto__)){
var node = temp__4425__auto__;
var temp__4425__auto____$1 = devcards.core.get_hljs.call(null);
if(cljs.core.truth_(temp__4425__auto____$1)){
var hljs = temp__4425__auto____$1;
var temp__4425__auto____$2 = (hljs["highlightBlock"]);
if(cljs.core.truth_(temp__4425__auto____$2)){
var highlight_block = temp__4425__auto____$2;
return highlight_block.call(null,node);
} else {
return null;
}
} else {
return null;
}
} else {
return null;
}
});
var base__19397__auto___19597 = {"componentDidMount": (function (){
var this$ = this;
return devcards.core.highlight_node.call(null,this$);
}), "componentDidUpdate": (function (){
var this$ = this;
return devcards.core.highlight_node.call(null,this$);
}), "render": (function (){
var this$ = this;
return React.createElement("pre",{"className": (cljs.core.truth_(devcards.core.get_hljs.call(null))?"com-rigsomelight-devcards-code-highlighting":""), "key": cljs.core.hash.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"code","code",1586293142)))},React.createElement("code",{"className": (function (){var or__16766__auto__ = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"lang","lang",-1819677104));
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return "";
}
})(), "ref": "code-ref"},sablono.interpreter.interpret.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"code","code",1586293142)))));
})};
if(typeof devcards.core.CodeHighlight !== 'undefined'){
} else {
devcards.core.CodeHighlight = React.createClass(base__19397__auto___19597);
}

var seq__19593_19598 = cljs.core.seq.call(null,cljs.core.map.call(null,cljs.core.name,cljs.core.list(new cljs.core.Symbol("cljs-react-reload.core","shouldComponentUpdate","cljs-react-reload.core/shouldComponentUpdate",-526191550,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillReceiveProps","cljs-react-reload.core/componentWillReceiveProps",-1087108864,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillMount","cljs-react-reload.core/componentWillMount",-1529759893,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidMount","cljs-react-reload.core/componentDidMount",-2035273110,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUpdate","cljs-react-reload.core/componentWillUpdate",-453323386,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidUpdate","cljs-react-reload.core/componentDidUpdate",-6660227,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUnmount","cljs-react-reload.core/componentWillUnmount",-1549767430,null),new cljs.core.Symbol("cljs-react-reload.core","render","cljs-react-reload.core/render",298414516,null))));
var chunk__19594_19599 = null;
var count__19595_19600 = (0);
var i__19596_19601 = (0);
while(true){
if((i__19596_19601 < count__19595_19600)){
var property__19398__auto___19602 = cljs.core._nth.call(null,chunk__19594_19599,i__19596_19601);
if(cljs.core.truth_((base__19397__auto___19597[property__19398__auto___19602]))){
(devcards.core.CodeHighlight.prototype[property__19398__auto___19602] = (base__19397__auto___19597[property__19398__auto___19602]));
} else {
}

var G__19603 = seq__19593_19598;
var G__19604 = chunk__19594_19599;
var G__19605 = count__19595_19600;
var G__19606 = (i__19596_19601 + (1));
seq__19593_19598 = G__19603;
chunk__19594_19599 = G__19604;
count__19595_19600 = G__19605;
i__19596_19601 = G__19606;
continue;
} else {
var temp__4425__auto___19607 = cljs.core.seq.call(null,seq__19593_19598);
if(temp__4425__auto___19607){
var seq__19593_19608__$1 = temp__4425__auto___19607;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__19593_19608__$1)){
var c__17569__auto___19609 = cljs.core.chunk_first.call(null,seq__19593_19608__$1);
var G__19610 = cljs.core.chunk_rest.call(null,seq__19593_19608__$1);
var G__19611 = c__17569__auto___19609;
var G__19612 = cljs.core.count.call(null,c__17569__auto___19609);
var G__19613 = (0);
seq__19593_19598 = G__19610;
chunk__19594_19599 = G__19611;
count__19595_19600 = G__19612;
i__19596_19601 = G__19613;
continue;
} else {
var property__19398__auto___19614 = cljs.core.first.call(null,seq__19593_19608__$1);
if(cljs.core.truth_((base__19397__auto___19597[property__19398__auto___19614]))){
(devcards.core.CodeHighlight.prototype[property__19398__auto___19614] = (base__19397__auto___19597[property__19398__auto___19614]));
} else {
}

var G__19615 = cljs.core.next.call(null,seq__19593_19608__$1);
var G__19616 = null;
var G__19617 = (0);
var G__19618 = (0);
seq__19593_19598 = G__19615;
chunk__19594_19599 = G__19616;
count__19595_19600 = G__19617;
i__19596_19601 = G__19618;
continue;
}
} else {
}
}
break;
}
devcards.core.code_highlight = (function devcards$core$code_highlight(code_str,lang){
return React.createElement(devcards.core.CodeHighlight,{"code": code_str, "lang": lang});
});
if(typeof devcards.core.markdown_block__GT_react !== 'undefined'){
} else {
devcards.core.markdown_block__GT_react = (function (){var method_table__17679__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var prefer_table__17680__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var method_cache__17681__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var cached_hierarchy__17682__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var hierarchy__17683__auto__ = cljs.core.get.call(null,cljs.core.PersistentArrayMap.EMPTY,new cljs.core.Keyword(null,"hierarchy","hierarchy",-1053470341),cljs.core.get_global_hierarchy.call(null));
return (new cljs.core.MultiFn(cljs.core.symbol.call(null,"devcards.core","markdown-block->react"),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"default","default",-1987822328),hierarchy__17683__auto__,method_table__17679__auto__,prefer_table__17680__auto__,method_cache__17681__auto__,cached_hierarchy__17682__auto__));
})();
}
cljs.core._add_method.call(null,devcards.core.markdown_block__GT_react,new cljs.core.Keyword(null,"default","default",-1987822328),(function (p__19619){
var map__19620 = p__19619;
var map__19620__$1 = ((((!((map__19620 == null)))?((((map__19620.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19620.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19620):map__19620);
var content = cljs.core.get.call(null,map__19620__$1,new cljs.core.Keyword(null,"content","content",15833224));
return devcards.core.react_raw.call(null,devcards.util.markdown.markdown_to_html.call(null,content));
}));
cljs.core._add_method.call(null,devcards.core.markdown_block__GT_react,new cljs.core.Keyword(null,"code-block","code-block",-2113425141),(function (p__19622){
var map__19623 = p__19622;
var map__19623__$1 = ((((!((map__19623 == null)))?((((map__19623.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19623.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19623):map__19623);
var block = map__19623__$1;
var content = cljs.core.get.call(null,map__19623__$1,new cljs.core.Keyword(null,"content","content",15833224));
return React.createElement(devcards.core.CodeHighlight,{"code": new cljs.core.Keyword(null,"content","content",15833224).cljs$core$IFn$_invoke$arity$1(block), "lang": new cljs.core.Keyword(null,"lang","lang",-1819677104).cljs$core$IFn$_invoke$arity$1(block)});
}));
devcards.core.react_element_QMARK_;
devcards.core.markdown__GT_react = (function devcards$core$markdown__GT_react(var_args){
var args__17831__auto__ = [];
var len__17824__auto___19627 = arguments.length;
var i__17825__auto___19628 = (0);
while(true){
if((i__17825__auto___19628 < len__17824__auto___19627)){
args__17831__auto__.push((arguments[i__17825__auto___19628]));

var G__19629 = (i__17825__auto___19628 + (1));
i__17825__auto___19628 = G__19629;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return devcards.core.markdown__GT_react.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

devcards.core.markdown__GT_react.cljs$core$IFn$_invoke$arity$variadic = (function (strs){
var strs__$1 = cljs.core.map.call(null,(function (x){
if(typeof x === 'string'){
return x;
} else {
if(cljs.core.truth_(devcards.core.react_element_QMARK_.call(null,x))){
return null;
} else {
return [cljs.core.str("```clojure\n"),cljs.core.str(devcards.util.utils.pprint_code.call(null,x)),cljs.core.str("```\n")].join('');
}
}
}),strs);
if(cljs.core.every_QMARK_.call(null,cljs.core.string_QMARK_,strs__$1)){
var blocks = cljs.core.mapcat.call(null,devcards.util.markdown.parse_out_blocks,strs__$1);
var attrs19626 = cljs.core.map_indexed.call(null,((function (blocks,strs__$1){
return (function (i,data){
return React.createElement("div",{"key": i},sablono.interpreter.interpret.call(null,devcards.core.markdown_block__GT_react.call(null,data)));
});})(blocks,strs__$1))
,blocks);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19626))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["com-rigsomelight-devcards-markdown",null,"com-rigsomelight-devcards-typog",null], null), null)], null),attrs19626)):{"className": "com-rigsomelight-devcards-markdown com-rigsomelight-devcards-typog"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19626))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19626)], null))));
} else {
var message = "Devcards Error: Didn't pass a seq of strings to less-sensitive-markdown.\n You are probably trying to pass react to markdown instead of strings. (defcard-doc (doc ...)) won't work.";
console.error(message);

return React.createElement("div",{"style": {"color": "#a94442"}},sablono.interpreter.interpret.call(null,message));
}
});

devcards.core.markdown__GT_react.cljs$lang$maxFixedArity = (0);

devcards.core.markdown__GT_react.cljs$lang$applyTo = (function (seq19625){
return devcards.core.markdown__GT_react.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq19625));
});
devcards.core.naked_card = (function devcards$core$naked_card(children,card){
var classname = cljs.core.get_in.call(null,card,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"classname","classname",777390796)], null));
var padding_QMARK_ = cljs.core.get_in.call(null,card,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"padding","padding",1660304693)], null));
return React.createElement("div",{"key": [cljs.core.str(cljs.core.hash.call(null,card)),cljs.core.str("2")].join(''), "className": (function (){var G__19631 = devcards.system.devcards_rendered_card_class;
var G__19631__$1 = (cljs.core.truth_(padding_QMARK_)?[cljs.core.str(G__19631),cljs.core.str(" com-rigsomelight-devcards-devcard-padding")].join(''):G__19631);
var G__19631__$2 = (cljs.core.truth_(cljs.core.not_empty.call(null,classname))?[cljs.core.str(G__19631__$1),cljs.core.str(" "),cljs.core.str(classname)].join(''):G__19631__$1);
return G__19631__$2;
})()},sablono.interpreter.interpret.call(null,children));
});
devcards.core.frame = (function devcards$core$frame(var_args){
var args19632 = [];
var len__17824__auto___19640 = arguments.length;
var i__17825__auto___19641 = (0);
while(true){
if((i__17825__auto___19641 < len__17824__auto___19640)){
args19632.push((arguments[i__17825__auto___19641]));

var G__19642 = (i__17825__auto___19641 + (1));
i__17825__auto___19641 = G__19642;
continue;
} else {
}
break;
}

var G__19634 = args19632.length;
switch (G__19634) {
case 1:
return devcards.core.frame.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return devcards.core.frame.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args19632.length)].join('')));

}
});

devcards.core.frame.cljs$core$IFn$_invoke$arity$1 = (function (children){
return devcards.core.frame.call(null,children,cljs.core.PersistentArrayMap.EMPTY);
});

devcards.core.frame.cljs$core$IFn$_invoke$arity$2 = (function (children,card){
var map__19635 = card;
var map__19635__$1 = ((((!((map__19635 == null)))?((((map__19635.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19635.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19635):map__19635);
var path = cljs.core.get.call(null,map__19635__$1,new cljs.core.Keyword(null,"path","path",-188191168));
var options = cljs.core.get.call(null,map__19635__$1,new cljs.core.Keyword(null,"options","options",99638489));
if(cljs.core.not.call(null,new cljs.core.Keyword(null,"hidden","hidden",-312506092).cljs$core$IFn$_invoke$arity$1(options))){
if(new cljs.core.Keyword(null,"heading","heading",-1312171873).cljs$core$IFn$_invoke$arity$1(options) === false){
return React.createElement("div",{"key": cljs.core.prn_str.call(null,path), "className": sablono.util.join_classes.call(null,cljs.core.PersistentHashSet.fromArray([[cljs.core.str("com-rigsomelight-devcards-card-base-no-pad "),cljs.core.str((cljs.core.truth_(new cljs.core.Keyword(null,"hide-border","hide-border",1463657151).cljs$core$IFn$_invoke$arity$1(options))?" com-rigsomelight-devcards-card-hide-border":null))].join('')], true))},sablono.interpreter.interpret.call(null,devcards.core.naked_card.call(null,children,card)));
} else {
return React.createElement("div",{"key": cljs.core.prn_str.call(null,path), "className": "com-rigsomelight-devcards-base com-rigsomelight-devcards-card-base-no-pad"},React.createElement("div",{"key": [cljs.core.str(cljs.core.hash.call(null,card)),cljs.core.str("1")].join(''), "className": "com-rigsomelight-devcards-panel-heading com-rigsomelight-devcards-typog"},(cljs.core.truth_(path)?sablono.interpreter.interpret.call(null,React.createElement("a",{"href": "#", "onClick": devcards.system.prevent__GT_.call(null,((function (map__19635,map__19635__$1,path,options){
return (function (){
return devcards.system.set_current_path_BANG_.call(null,devcards.system.app_state,path);
});})(map__19635,map__19635__$1,path,options))
)},sablono.interpreter.interpret.call(null,cljs.core.name.call(null,cljs.core.last.call(null,path)))," ")):sablono.interpreter.interpret.call(null,(function (){var attrs19637 = new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(card);
return cljs.core.apply.call(null,React.createElement,"span",((cljs.core.map_QMARK_.call(null,attrs19637))?sablono.interpreter.attributes.call(null,attrs19637):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19637))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19637)], null))));
})()))),sablono.interpreter.interpret.call(null,devcards.core.naked_card.call(null,children,card)));
}
} else {
return React.createElement("span",null);
}
});

devcards.core.frame.cljs$lang$maxFixedArity = 2;

/**
 * @interface
 */
devcards.core.IDevcardOptions = function(){};

devcards.core._devcard_options = (function devcards$core$_devcard_options(this$,devcard_opts){
if((!((this$ == null))) && (!((this$.devcards$core$IDevcardOptions$_devcard_options$arity$2 == null)))){
return this$.devcards$core$IDevcardOptions$_devcard_options$arity$2(this$,devcard_opts);
} else {
var x__17421__auto__ = (((this$ == null))?null:this$);
var m__17422__auto__ = (devcards.core._devcard_options[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,this$,devcard_opts);
} else {
var m__17422__auto____$1 = (devcards.core._devcard_options["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,this$,devcard_opts);
} else {
throw cljs.core.missing_protocol.call(null,"IDevcardOptions.-devcard-options",this$);
}
}
}
});


/**
 * @interface
 */
devcards.core.IDevcard = function(){};

devcards.core._devcard = (function devcards$core$_devcard(this$,devcard_opts){
if((!((this$ == null))) && (!((this$.devcards$core$IDevcard$_devcard$arity$2 == null)))){
return this$.devcards$core$IDevcard$_devcard$arity$2(this$,devcard_opts);
} else {
var x__17421__auto__ = (((this$ == null))?null:this$);
var m__17422__auto__ = (devcards.core._devcard[goog.typeOf(x__17421__auto__)]);
if(!((m__17422__auto__ == null))){
return m__17422__auto__.call(null,this$,devcard_opts);
} else {
var m__17422__auto____$1 = (devcards.core._devcard["_"]);
if(!((m__17422__auto____$1 == null))){
return m__17422__auto____$1.call(null,this$,devcard_opts);
} else {
throw cljs.core.missing_protocol.call(null,"IDevcard.-devcard",this$);
}
}
}
});

devcards.core.hist_recorder_STAR_;
devcards.core.ref__GT_node = (function devcards$core$ref__GT_node(this$,ref){
var temp__4425__auto__ = (this$.refs[ref]);
if(cljs.core.truth_(temp__4425__auto__)){
var comp = temp__4425__auto__;
return ReactDOM.findDOMNode(comp);
} else {
return null;
}
});
devcards.core.get_props = (function devcards$core$get_props(this$,k){
return (this$.props[cljs.core.name.call(null,k)]);
});
devcards.core.get_state = (function devcards$core$get_state(this$,k){
if(cljs.core.truth_(this$.state)){
return (this$.state[cljs.core.name.call(null,k)]);
} else {
return null;
}
});
var base__19397__auto___19649 = {"shouldComponentUpdate": (function (next_props,b){
var this$ = this;
var update_QMARK_ = cljs.core._EQ_.call(null,(next_props["change_count"]),devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"change_count","change_count",-533812109)));
return update_QMARK_;
}), "render": (function (){
var this$ = this;
var attrs19644 = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"children_thunk","children_thunk",-1161306645));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19644))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["com-rigsomelight-dont-update",null], null), null)], null),attrs19644)):{"className": "com-rigsomelight-dont-update"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19644))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19644)], null))));
})};
if(typeof devcards.core.DontUpdate !== 'undefined'){
} else {
devcards.core.DontUpdate = React.createClass(base__19397__auto___19649);
}

var seq__19645_19650 = cljs.core.seq.call(null,cljs.core.map.call(null,cljs.core.name,cljs.core.list(new cljs.core.Symbol("cljs-react-reload.core","shouldComponentUpdate","cljs-react-reload.core/shouldComponentUpdate",-526191550,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillReceiveProps","cljs-react-reload.core/componentWillReceiveProps",-1087108864,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillMount","cljs-react-reload.core/componentWillMount",-1529759893,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidMount","cljs-react-reload.core/componentDidMount",-2035273110,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUpdate","cljs-react-reload.core/componentWillUpdate",-453323386,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidUpdate","cljs-react-reload.core/componentDidUpdate",-6660227,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUnmount","cljs-react-reload.core/componentWillUnmount",-1549767430,null),new cljs.core.Symbol("cljs-react-reload.core","render","cljs-react-reload.core/render",298414516,null))));
var chunk__19646_19651 = null;
var count__19647_19652 = (0);
var i__19648_19653 = (0);
while(true){
if((i__19648_19653 < count__19647_19652)){
var property__19398__auto___19654 = cljs.core._nth.call(null,chunk__19646_19651,i__19648_19653);
if(cljs.core.truth_((base__19397__auto___19649[property__19398__auto___19654]))){
(devcards.core.DontUpdate.prototype[property__19398__auto___19654] = (base__19397__auto___19649[property__19398__auto___19654]));
} else {
}

var G__19655 = seq__19645_19650;
var G__19656 = chunk__19646_19651;
var G__19657 = count__19647_19652;
var G__19658 = (i__19648_19653 + (1));
seq__19645_19650 = G__19655;
chunk__19646_19651 = G__19656;
count__19647_19652 = G__19657;
i__19648_19653 = G__19658;
continue;
} else {
var temp__4425__auto___19659 = cljs.core.seq.call(null,seq__19645_19650);
if(temp__4425__auto___19659){
var seq__19645_19660__$1 = temp__4425__auto___19659;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__19645_19660__$1)){
var c__17569__auto___19661 = cljs.core.chunk_first.call(null,seq__19645_19660__$1);
var G__19662 = cljs.core.chunk_rest.call(null,seq__19645_19660__$1);
var G__19663 = c__17569__auto___19661;
var G__19664 = cljs.core.count.call(null,c__17569__auto___19661);
var G__19665 = (0);
seq__19645_19650 = G__19662;
chunk__19646_19651 = G__19663;
count__19647_19652 = G__19664;
i__19648_19653 = G__19665;
continue;
} else {
var property__19398__auto___19666 = cljs.core.first.call(null,seq__19645_19660__$1);
if(cljs.core.truth_((base__19397__auto___19649[property__19398__auto___19666]))){
(devcards.core.DontUpdate.prototype[property__19398__auto___19666] = (base__19397__auto___19649[property__19398__auto___19666]));
} else {
}

var G__19667 = cljs.core.next.call(null,seq__19645_19660__$1);
var G__19668 = null;
var G__19669 = (0);
var G__19670 = (0);
seq__19645_19650 = G__19667;
chunk__19646_19651 = G__19668;
count__19647_19652 = G__19669;
i__19648_19653 = G__19670;
continue;
}
} else {
}
}
break;
}
devcards.core.dont_update = (function devcards$core$dont_update(change_count,children_thunk){
return React.createElement(devcards.core.DontUpdate,{"change_count": change_count, "children_thunk": children_thunk});
});
devcards.core.wrangle_inital_data = (function devcards$core$wrangle_inital_data(this$){
var data = (function (){var or__16766__auto__ = new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804).cljs$core$IFn$_invoke$arity$1(devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"card","card",-1430355152)));
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core.PersistentArrayMap.EMPTY;
}
})();
if(((!((data == null)))?((((data.cljs$lang$protocol_mask$partition1$ & (16384))) || (data.cljs$core$IAtom$))?true:(((!data.cljs$lang$protocol_mask$partition1$))?cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IAtom,data):false)):cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IAtom,data))){
return data;
} else {
return cljs.core.atom.call(null,data);
}
});
devcards.core.get_data_atom = (cljs.core.truth_(devcards.util.utils.html_env_QMARK_.call(null))?(function (this$){
return devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
}):(function (this$){
return devcards.core.wrangle_inital_data.call(null,this$);
}));
devcards.core.atom_like_QMARK_;
var base__19397__auto___19678 = {"getInitialState": (function (){
return {"unique_id": cljs.core.gensym.call(null,new cljs.core.Symbol(null,"devcards-base-","devcards-base-",-1457268595,null)), "state_change_count": (0)};
}), "componentDidUpdate": (function (_,___$1){
var this$ = this;
var atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
var card = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"card","card",-1430355152));
var options = new cljs.core.Keyword(null,"options","options",99638489).cljs$core$IFn$_invoke$arity$1(card);
if(cljs.core.truth_(new cljs.core.Keyword(null,"static-state","static-state",-1049492012).cljs$core$IFn$_invoke$arity$1(options))){
var initial_data = new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804).cljs$core$IFn$_invoke$arity$1(card);
var data = (cljs.core.truth_(devcards.core.atom_like_QMARK_.call(null,initial_data))?cljs.core.deref.call(null,initial_data):initial_data);
if(cljs.core.not_EQ_.call(null,cljs.core.deref.call(null,atom),data)){
return cljs.core.reset_BANG_.call(null,atom,data);
} else {
return null;
}
} else {
return null;
}
}), "componentWillMount": (cljs.core.truth_(devcards.util.utils.html_env_QMARK_.call(null))?(function (){
var this$ = this;
return this$.setState((function (){var or__16766__auto__ = (function (){var and__16754__auto__ = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
if(cljs.core.truth_(and__16754__auto__)){
return this$.state;
} else {
return and__16754__auto__;
}
})();
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return {"data_atom": devcards.core.wrangle_inital_data.call(null,this$)};
}
})());
}):(function (){
return null;
})), "componentWillUnmount": (function (){
var this$ = this;
var data_atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
var id = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329));
if(cljs.core.truth_((function (){var and__16754__auto__ = data_atom;
if(cljs.core.truth_(and__16754__auto__)){
return id;
} else {
return and__16754__auto__;
}
})())){
return cljs.core.remove_watch.call(null,data_atom,id);
} else {
return null;
}
}), "componentDidMount": (cljs.core.truth_(devcards.util.utils.html_env_QMARK_.call(null))?(function (){
var this$ = this;
var temp__4425__auto__ = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
if(cljs.core.truth_(temp__4425__auto__)){
var data_atom = temp__4425__auto__;
var temp__4425__auto____$1 = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329));
if(cljs.core.truth_(temp__4425__auto____$1)){
var id = temp__4425__auto____$1;
return cljs.core.add_watch.call(null,data_atom,id,((function (id,temp__4425__auto____$1,data_atom,temp__4425__auto__,this$){
return (function (_,___$1,___$2,___$3){
return this$.setState({"state_change_count": (devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"state_change_count","state_change_count",-135095612)) + (1))});
});})(id,temp__4425__auto____$1,data_atom,temp__4425__auto__,this$))
);
} else {
return null;
}
} else {
return null;
}
}):(function (){
return null;
})), "render": (function (){
var this$ = this;
var data_atom = devcards.core.get_data_atom.call(null,this$);
var card = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"card","card",-1430355152));
var change_count = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"state_change_count","state_change_count",-135095612));
var options = new cljs.core.Keyword(null,"options","options",99638489).cljs$core$IFn$_invoke$arity$1(card);
var main_obj_SINGLEQUOTE_ = (function (){var m = new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742).cljs$core$IFn$_invoke$arity$1(card);
if(cljs.core.fn_QMARK_.call(null,m)){
return m.call(null,data_atom,this$);
} else {
return m;
}
})();
var main_obj = (((!((main_obj_SINGLEQUOTE_ == null))) && (cljs.core.not.call(null,devcards.core.react_element_QMARK_.call(null,main_obj_SINGLEQUOTE_))))?devcards.core.code_highlight.call(null,devcards.util.utils.pprint_code.call(null,main_obj_SINGLEQUOTE_),"clojure"):main_obj_SINGLEQUOTE_);
var main = ((new cljs.core.Keyword(null,"watch-atom","watch-atom",-2134031308).cljs$core$IFn$_invoke$arity$1(options) === false)?devcards.core.dont_update.call(null,change_count,main_obj):main_obj);
var hist_ctl = (cljs.core.truth_(new cljs.core.Keyword(null,"history","history",-247395220).cljs$core$IFn$_invoke$arity$1(options))?devcards.core.hist_recorder_STAR_.call(null,data_atom):null);
var document = (function (){var temp__4425__auto__ = new cljs.core.Keyword(null,"documentation","documentation",1889593999).cljs$core$IFn$_invoke$arity$1(card);
if(cljs.core.truth_(temp__4425__auto__)){
var docu = temp__4425__auto__;
return devcards.core.markdown__GT_react.call(null,docu);
} else {
return null;
}
})();
var edn = (cljs.core.truth_(new cljs.core.Keyword(null,"inspect-data","inspect-data",640452006).cljs$core$IFn$_invoke$arity$1(options))?(function (){var attrs19673 = devcards.util.edn_renderer.html_edn.call(null,cljs.core.deref.call(null,data_atom));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19673))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["com-rigsomelight-devcards-padding-top-border",null], null), null)], null),attrs19673)):{"className": "com-rigsomelight-devcards-padding-top-border"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19673))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19673)], null))));
})():null);
var card__$1 = (((typeof main_obj === 'string') || ((main_obj == null)))?cljs.core.assoc_in.call(null,card,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"hide-border","hide-border",1463657151)], null),true):card);
var children = cljs.core.keep_indexed.call(null,((function (data_atom,card,change_count,options,main_obj_SINGLEQUOTE_,main_obj,main,hist_ctl,document,edn,card__$1,this$){
return (function (i,child){
return React.createElement("div",{"key": i},sablono.interpreter.interpret.call(null,child));
});})(data_atom,card,change_count,options,main_obj_SINGLEQUOTE_,main_obj,main,hist_ctl,document,edn,card__$1,this$))
,cljs.core._conj.call(null,cljs.core._conj.call(null,cljs.core._conj.call(null,cljs.core._conj.call(null,cljs.core.List.EMPTY,edn),hist_ctl),main),document));
if(cljs.core.truth_(new cljs.core.Keyword(null,"frame","frame",-1711082588).cljs$core$IFn$_invoke$arity$1(options))){
return devcards.core.frame.call(null,children,card__$1);
} else {
return React.createElement("div",{"className": "com-rigsomelight-devcards-frameless"},sablono.interpreter.interpret.call(null,children));
}
})};
if(typeof devcards.core.DevcardBase !== 'undefined'){
} else {
devcards.core.DevcardBase = React.createClass(base__19397__auto___19678);
}

var seq__19674_19679 = cljs.core.seq.call(null,cljs.core.map.call(null,cljs.core.name,cljs.core.list(new cljs.core.Symbol("cljs-react-reload.core","shouldComponentUpdate","cljs-react-reload.core/shouldComponentUpdate",-526191550,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillReceiveProps","cljs-react-reload.core/componentWillReceiveProps",-1087108864,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillMount","cljs-react-reload.core/componentWillMount",-1529759893,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidMount","cljs-react-reload.core/componentDidMount",-2035273110,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUpdate","cljs-react-reload.core/componentWillUpdate",-453323386,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidUpdate","cljs-react-reload.core/componentDidUpdate",-6660227,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUnmount","cljs-react-reload.core/componentWillUnmount",-1549767430,null),new cljs.core.Symbol("cljs-react-reload.core","render","cljs-react-reload.core/render",298414516,null))));
var chunk__19675_19680 = null;
var count__19676_19681 = (0);
var i__19677_19682 = (0);
while(true){
if((i__19677_19682 < count__19676_19681)){
var property__19398__auto___19683 = cljs.core._nth.call(null,chunk__19675_19680,i__19677_19682);
if(cljs.core.truth_((base__19397__auto___19678[property__19398__auto___19683]))){
(devcards.core.DevcardBase.prototype[property__19398__auto___19683] = (base__19397__auto___19678[property__19398__auto___19683]));
} else {
}

var G__19684 = seq__19674_19679;
var G__19685 = chunk__19675_19680;
var G__19686 = count__19676_19681;
var G__19687 = (i__19677_19682 + (1));
seq__19674_19679 = G__19684;
chunk__19675_19680 = G__19685;
count__19676_19681 = G__19686;
i__19677_19682 = G__19687;
continue;
} else {
var temp__4425__auto___19688 = cljs.core.seq.call(null,seq__19674_19679);
if(temp__4425__auto___19688){
var seq__19674_19689__$1 = temp__4425__auto___19688;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__19674_19689__$1)){
var c__17569__auto___19690 = cljs.core.chunk_first.call(null,seq__19674_19689__$1);
var G__19691 = cljs.core.chunk_rest.call(null,seq__19674_19689__$1);
var G__19692 = c__17569__auto___19690;
var G__19693 = cljs.core.count.call(null,c__17569__auto___19690);
var G__19694 = (0);
seq__19674_19679 = G__19691;
chunk__19675_19680 = G__19692;
count__19676_19681 = G__19693;
i__19677_19682 = G__19694;
continue;
} else {
var property__19398__auto___19695 = cljs.core.first.call(null,seq__19674_19689__$1);
if(cljs.core.truth_((base__19397__auto___19678[property__19398__auto___19695]))){
(devcards.core.DevcardBase.prototype[property__19398__auto___19695] = (base__19397__auto___19678[property__19398__auto___19695]));
} else {
}

var G__19696 = cljs.core.next.call(null,seq__19674_19689__$1);
var G__19697 = null;
var G__19698 = (0);
var G__19699 = (0);
seq__19674_19679 = G__19696;
chunk__19675_19680 = G__19697;
count__19676_19681 = G__19698;
i__19677_19682 = G__19699;
continue;
}
} else {
}
}
break;
}
devcards.core.render_into_dom = (cljs.core.truth_(devcards.util.utils.html_env_QMARK_.call(null))?(function (this$){
var temp__4425__auto__ = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"node_fn","node_fn",1182818791));
if(cljs.core.truth_(temp__4425__auto__)){
var node_fn = temp__4425__auto__;
var temp__4425__auto____$1 = devcards.core.ref__GT_node.call(null,this$,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329)));
if(cljs.core.truth_(temp__4425__auto____$1)){
var node = temp__4425__auto____$1;
return node_fn.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504)),node);
} else {
return null;
}
} else {
return null;
}
}):cljs.core.identity);
var base__19397__auto___19704 = {"getInitialState": (function (){
return {"unique_id": [cljs.core.str(cljs.core.gensym.call(null,new cljs.core.Symbol(null,"devcards-dom-component-","devcards-dom-component-",-730322144,null)))].join('')};
}), "componentDidUpdate": (function (prevP,prevS){
var this$ = this;
if(cljs.core.truth_((function (){var and__16754__auto__ = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"node_fn","node_fn",1182818791));
if(cljs.core.truth_(and__16754__auto__)){
return cljs.core.not_EQ_.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"node_fn","node_fn",1182818791)),(prevP["node_fn"]));
} else {
return and__16754__auto__;
}
})())){
return devcards.core.render_into_dom.call(null,this$);
} else {
return null;
}
}), "componentWillUnmount": (function (){
var this$ = this;
var temp__4425__auto__ = devcards.core.ref__GT_node.call(null,this$,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329)));
if(cljs.core.truth_(temp__4425__auto__)){
var node = temp__4425__auto__;
return ReactDOM.unmountComponentAtNode(node);
} else {
return null;
}
}), "componentDidMount": (function (){
var this$ = this;
return devcards.core.render_into_dom.call(null,this$);
}), "render": (cljs.core.truth_(devcards.util.utils.html_env_QMARK_.call(null))?(function (){
var this$ = this;
return React.DOM.div({"className": "com-rigsomelight-devcards-dom-node", "ref": devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329))},"Card has not mounted DOM node.");
}):(function (){
return React.DOM.div("Card has not mounted DOM node.");
}))};
if(typeof devcards.core.DomComponent !== 'undefined'){
} else {
devcards.core.DomComponent = React.createClass(base__19397__auto___19704);
}

var seq__19700_19705 = cljs.core.seq.call(null,cljs.core.map.call(null,cljs.core.name,cljs.core.list(new cljs.core.Symbol("cljs-react-reload.core","shouldComponentUpdate","cljs-react-reload.core/shouldComponentUpdate",-526191550,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillReceiveProps","cljs-react-reload.core/componentWillReceiveProps",-1087108864,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillMount","cljs-react-reload.core/componentWillMount",-1529759893,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidMount","cljs-react-reload.core/componentDidMount",-2035273110,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUpdate","cljs-react-reload.core/componentWillUpdate",-453323386,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidUpdate","cljs-react-reload.core/componentDidUpdate",-6660227,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUnmount","cljs-react-reload.core/componentWillUnmount",-1549767430,null),new cljs.core.Symbol("cljs-react-reload.core","render","cljs-react-reload.core/render",298414516,null))));
var chunk__19701_19706 = null;
var count__19702_19707 = (0);
var i__19703_19708 = (0);
while(true){
if((i__19703_19708 < count__19702_19707)){
var property__19398__auto___19709 = cljs.core._nth.call(null,chunk__19701_19706,i__19703_19708);
if(cljs.core.truth_((base__19397__auto___19704[property__19398__auto___19709]))){
(devcards.core.DomComponent.prototype[property__19398__auto___19709] = (base__19397__auto___19704[property__19398__auto___19709]));
} else {
}

var G__19710 = seq__19700_19705;
var G__19711 = chunk__19701_19706;
var G__19712 = count__19702_19707;
var G__19713 = (i__19703_19708 + (1));
seq__19700_19705 = G__19710;
chunk__19701_19706 = G__19711;
count__19702_19707 = G__19712;
i__19703_19708 = G__19713;
continue;
} else {
var temp__4425__auto___19714 = cljs.core.seq.call(null,seq__19700_19705);
if(temp__4425__auto___19714){
var seq__19700_19715__$1 = temp__4425__auto___19714;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__19700_19715__$1)){
var c__17569__auto___19716 = cljs.core.chunk_first.call(null,seq__19700_19715__$1);
var G__19717 = cljs.core.chunk_rest.call(null,seq__19700_19715__$1);
var G__19718 = c__17569__auto___19716;
var G__19719 = cljs.core.count.call(null,c__17569__auto___19716);
var G__19720 = (0);
seq__19700_19705 = G__19717;
chunk__19701_19706 = G__19718;
count__19702_19707 = G__19719;
i__19703_19708 = G__19720;
continue;
} else {
var property__19398__auto___19721 = cljs.core.first.call(null,seq__19700_19715__$1);
if(cljs.core.truth_((base__19397__auto___19704[property__19398__auto___19721]))){
(devcards.core.DomComponent.prototype[property__19398__auto___19721] = (base__19397__auto___19704[property__19398__auto___19721]));
} else {
}

var G__19722 = cljs.core.next.call(null,seq__19700_19715__$1);
var G__19723 = null;
var G__19724 = (0);
var G__19725 = (0);
seq__19700_19705 = G__19722;
chunk__19701_19706 = G__19723;
count__19702_19707 = G__19724;
i__19703_19708 = G__19725;
continue;
}
} else {
}
}
break;
}
devcards.core.booler_QMARK_ = (function devcards$core$booler_QMARK_(key,opts){
var x = cljs.core.get.call(null,opts,key);
var or__16766__auto__ = x === true;
if(or__16766__auto__){
return or__16766__auto__;
} else {
var or__16766__auto____$1 = x === false;
if(or__16766__auto____$1){
return or__16766__auto____$1;
} else {
var or__16766__auto____$2 = (x == null);
if(or__16766__auto____$2){
return or__16766__auto____$2;
} else {
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"label","label",1718410804),key,new cljs.core.Keyword(null,"message","message",-406056002),"should be boolean or nil",new cljs.core.Keyword(null,"value","value",305978217),x], null);
}
}
}
});
devcards.core.stringer_QMARK_ = (function devcards$core$stringer_QMARK_(key,opts){
var x = cljs.core.get.call(null,opts,key);
var or__16766__auto__ = typeof x === 'string';
if(or__16766__auto__){
return or__16766__auto__;
} else {
var or__16766__auto____$1 = (x == null);
if(or__16766__auto____$1){
return or__16766__auto____$1;
} else {
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"label","label",1718410804),key,new cljs.core.Keyword(null,"message","message",-406056002),"should be string or nil",new cljs.core.Keyword(null,"value","value",305978217),x], null);
}
}
});
devcards.core.react_element_QMARK_ = (function devcards$core$react_element_QMARK_(main_obj){
var or__16766__auto__ = (main_obj["_isReactElement"]);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core._EQ_.call(null,devcards.core.react_element_type_symbol,(main_obj["$$typeof"]));
}
});
devcards.core.validate_card_options = (function devcards$core$validate_card_options(opts){
if(cljs.core.map_QMARK_.call(null,opts)){
var propagated_errors = cljs.core.get_in.call(null,opts,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"propagated-errors","propagated-errors",1359777293)], null));
return cljs.core.filter.call(null,((function (propagated_errors){
return (function (p1__19726_SHARP_){
return !(p1__19726_SHARP_ === true);
});})(propagated_errors))
,(function (){var map__19735 = opts;
var map__19735__$1 = ((((!((map__19735 == null)))?((((map__19735.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19735.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19735):map__19735);
var name = cljs.core.get.call(null,map__19735__$1,new cljs.core.Keyword(null,"name","name",1843675177));
var main_obj = cljs.core.get.call(null,map__19735__$1,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742));
var initial_data = cljs.core.get.call(null,map__19735__$1,new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804));
var options = cljs.core.get.call(null,map__19735__$1,new cljs.core.Keyword(null,"options","options",99638489));
return cljs.core.concat.call(null,propagated_errors,new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [(function (){var or__16766__auto__ = cljs.core.map_QMARK_.call(null,options);
if(or__16766__auto__){
return or__16766__auto__;
} else {
var or__16766__auto____$1 = (options == null);
if(or__16766__auto____$1){
return or__16766__auto____$1;
} else {
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"label","label",1718410804),new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.Keyword(null,"message","message",-406056002),"should be a Map or nil",new cljs.core.Keyword(null,"value","value",305978217),options], null);
}
}
})(),devcards.core.stringer_QMARK_.call(null,new cljs.core.Keyword(null,"name","name",1843675177),opts),devcards.core.stringer_QMARK_.call(null,new cljs.core.Keyword(null,"documentation","documentation",1889593999),opts),(function (){var or__16766__auto__ = (initial_data == null);
if(or__16766__auto__){
return or__16766__auto__;
} else {
var or__16766__auto____$1 = cljs.core.vector_QMARK_.call(null,initial_data);
if(or__16766__auto____$1){
return or__16766__auto____$1;
} else {
var or__16766__auto____$2 = cljs.core.map_QMARK_.call(null,initial_data);
if(or__16766__auto____$2){
return or__16766__auto____$2;
} else {
var or__16766__auto____$3 = ((!((initial_data == null)))?((((initial_data.cljs$lang$protocol_mask$partition1$ & (16384))) || (initial_data.cljs$core$IAtom$))?true:(((!initial_data.cljs$lang$protocol_mask$partition1$))?cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IAtom,initial_data):false)):cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IAtom,initial_data));
if(or__16766__auto____$3){
return or__16766__auto____$3;
} else {
return new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"label","label",1718410804),new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804),new cljs.core.Keyword(null,"message","message",-406056002),"should be an Atom or a Map or nil.",new cljs.core.Keyword(null,"value","value",305978217),initial_data], null);
}
}
}
}
})()], null),cljs.core.mapv.call(null,((function (map__19735,map__19735__$1,name,main_obj,initial_data,options,propagated_errors){
return (function (p1__19727_SHARP_){
return devcards.core.booler_QMARK_.call(null,p1__19727_SHARP_,new cljs.core.Keyword(null,"options","options",99638489).cljs$core$IFn$_invoke$arity$1(opts));
});})(map__19735,map__19735__$1,name,main_obj,initial_data,options,propagated_errors))
,new cljs.core.PersistentVector(null, 7, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"frame","frame",-1711082588),new cljs.core.Keyword(null,"heading","heading",-1312171873),new cljs.core.Keyword(null,"padding","padding",1660304693),new cljs.core.Keyword(null,"inspect-data","inspect-data",640452006),new cljs.core.Keyword(null,"watch-atom","watch-atom",-2134031308),new cljs.core.Keyword(null,"history","history",-247395220),new cljs.core.Keyword(null,"static-state","static-state",-1049492012)], null)));
})());
} else {
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"message","message",-406056002),"Card should be a Map.",new cljs.core.Keyword(null,"value","value",305978217),opts], null)], null);
}
});
devcards.core.error_line = (function devcards$core$error_line(e){
return React.createElement("div",{"style": {"color": "#a94442", "display": "flex", "margin": "0.5em 0px"}},sablono.interpreter.interpret.call(null,React.createElement("code",{"style": {"flex": "1 100px", "marginRight": "10px"}},sablono.interpreter.interpret.call(null,(cljs.core.truth_(new cljs.core.Keyword(null,"label","label",1718410804).cljs$core$IFn$_invoke$arity$1(e))?cljs.core.pr_str.call(null,new cljs.core.Keyword(null,"label","label",1718410804).cljs$core$IFn$_invoke$arity$1(e)):null)))),React.createElement("span",{"style": {"flex": "3 100px", "marginRight": "10px"}},sablono.interpreter.interpret.call(null,new cljs.core.Keyword(null,"message","message",-406056002).cljs$core$IFn$_invoke$arity$1(e))),React.createElement("span",{"style": {"flex": "1 100px"}}," Received: ",(function (){var attrs19743 = cljs.core.pr_str.call(null,new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(e));
return cljs.core.apply.call(null,React.createElement,"code",((cljs.core.map_QMARK_.call(null,attrs19743))?sablono.interpreter.attributes.call(null,attrs19743):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19743))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19743)], null))));
})()));
});
devcards.core.render_errors = (function devcards$core$render_errors(opts,errors){
return React.createElement("div",{"className": "com-rigsomelight-devcards-card-base-no-pad"},(function (){var attrs19747 = [cljs.core.str((((cljs.core.map_QMARK_.call(null,opts)) && (typeof new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(opts) === 'string'))?[cljs.core.str(new cljs.core.Keyword(null,"name","name",1843675177).cljs$core$IFn$_invoke$arity$1(opts)),cljs.core.str(": ")].join(''):null)),cljs.core.str("Devcard received bad options")].join('');
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19747))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["com-rigsomelight-devcards-fail",null,"com-rigsomelight-devcards-panel-heading",null], null), null)], null),attrs19747)):{"className": "com-rigsomelight-devcards-fail com-rigsomelight-devcards-panel-heading"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19747))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19747)], null))));
})(),sablono.interpreter.interpret.call(null,devcards.core.naked_card.call(null,React.createElement("div",null,(function (){var attrs19748 = cljs.core.map.call(null,devcards.core.error_line,errors);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19748))?sablono.interpreter.attributes.call(null,attrs19748):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19748))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19748)], null))));
})(),sablono.interpreter.interpret.call(null,((cljs.core.map_QMARK_.call(null,opts))?(function (){var attrs19749 = devcards.util.edn_renderer.html_edn.call(null,cljs.core.update_in.call(null,opts,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489)], null),cljs.core.dissoc,new cljs.core.Keyword(null,"propagated-errors","propagated-errors",1359777293)));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19749))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["com-rigsomelight-devcards-padding-top-border",null], null), null)], null),attrs19749)):{"className": "com-rigsomelight-devcards-padding-top-border"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19749))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19749)], null))));
})():null))),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"options","options",99638489),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"padding","padding",1660304693),true], null)], null))));
});
devcards.core.add_environment_defaults = (function devcards$core$add_environment_defaults(card_options){
return cljs.core.update_in.call(null,card_options,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"options","options",99638489)], null),(function (p1__19750_SHARP_){
return cljs.core.merge.call(null,new cljs.core.Keyword(null,"base-card-options","base-card-options",141017756).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,devcards.system.app_state)),p1__19750_SHARP_);
}));
});
devcards.core.card_with_errors = (function devcards$core$card_with_errors(card_options){
var errors = devcards.core.validate_card_options.call(null,card_options);
if(cljs.core.truth_(cljs.core.not_empty.call(null,errors))){
return devcards.core.render_errors.call(null,card_options,errors);
} else {
return React.createElement(devcards.core.DevcardBase,{"card": devcards.core.add_environment_defaults.call(null,card_options)});
}
});

/**
* @constructor
 * @implements {cljs.core.IRecord}
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
 * @implements {devcards.core.IDevcardOptions}
*/
devcards.core.IdentiyOptions = (function (obj,__meta,__extmap,__hash){
this.obj = obj;
this.__meta = __meta;
this.__extmap = __extmap;
this.__hash = __hash;
this.cljs$lang$protocol_mask$partition0$ = 2229667594;
this.cljs$lang$protocol_mask$partition1$ = 8192;
})
devcards.core.IdentiyOptions.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__17380__auto__,k__17381__auto__){
var self__ = this;
var this__17380__auto____$1 = this;
return cljs.core._lookup.call(null,this__17380__auto____$1,k__17381__auto__,null);
});

devcards.core.IdentiyOptions.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__17382__auto__,k19752,else__17383__auto__){
var self__ = this;
var this__17382__auto____$1 = this;
var G__19754 = (((k19752 instanceof cljs.core.Keyword))?k19752.fqn:null);
switch (G__19754) {
case "obj":
return self__.obj;

break;
default:
return cljs.core.get.call(null,self__.__extmap,k19752,else__17383__auto__);

}
});

devcards.core.IdentiyOptions.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__17394__auto__,writer__17395__auto__,opts__17396__auto__){
var self__ = this;
var this__17394__auto____$1 = this;
var pr_pair__17397__auto__ = ((function (this__17394__auto____$1){
return (function (keyval__17398__auto__){
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,cljs.core.pr_writer,""," ","",opts__17396__auto__,keyval__17398__auto__);
});})(this__17394__auto____$1))
;
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,pr_pair__17397__auto__,"#devcards.core.IdentiyOptions{",", ","}",opts__17396__auto__,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.IdentiyOptions.prototype.cljs$core$IIterable$ = true;

devcards.core.IdentiyOptions.prototype.cljs$core$IIterable$_iterator$arity$1 = (function (G__19751){
var self__ = this;
var G__19751__$1 = this;
return (new cljs.core.RecordIter((0),G__19751__$1,1,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"obj","obj",981763962)], null),cljs.core._iterator.call(null,self__.__extmap)));
});

devcards.core.IdentiyOptions.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__17378__auto__){
var self__ = this;
var this__17378__auto____$1 = this;
return self__.__meta;
});

devcards.core.IdentiyOptions.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__17374__auto__){
var self__ = this;
var this__17374__auto____$1 = this;
return (new devcards.core.IdentiyOptions(self__.obj,self__.__meta,self__.__extmap,self__.__hash));
});

devcards.core.IdentiyOptions.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__17384__auto__){
var self__ = this;
var this__17384__auto____$1 = this;
return (1 + cljs.core.count.call(null,self__.__extmap));
});

devcards.core.IdentiyOptions.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__17375__auto__){
var self__ = this;
var this__17375__auto____$1 = this;
var h__17201__auto__ = self__.__hash;
if(!((h__17201__auto__ == null))){
return h__17201__auto__;
} else {
var h__17201__auto____$1 = cljs.core.hash_imap.call(null,this__17375__auto____$1);
self__.__hash = h__17201__auto____$1;

return h__17201__auto____$1;
}
});

devcards.core.IdentiyOptions.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this__17376__auto__,other__17377__auto__){
var self__ = this;
var this__17376__auto____$1 = this;
if(cljs.core.truth_((function (){var and__16754__auto__ = other__17377__auto__;
if(cljs.core.truth_(and__16754__auto__)){
var and__16754__auto____$1 = (this__17376__auto____$1.constructor === other__17377__auto__.constructor);
if(and__16754__auto____$1){
return cljs.core.equiv_map.call(null,this__17376__auto____$1,other__17377__auto__);
} else {
return and__16754__auto____$1;
}
} else {
return and__16754__auto__;
}
})())){
return true;
} else {
return false;
}
});

devcards.core.IdentiyOptions.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__17389__auto__,k__17390__auto__){
var self__ = this;
var this__17389__auto____$1 = this;
if(cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"obj","obj",981763962),null], null), null),k__17390__auto__)){
return cljs.core.dissoc.call(null,cljs.core.with_meta.call(null,cljs.core.into.call(null,cljs.core.PersistentArrayMap.EMPTY,this__17389__auto____$1),self__.__meta),k__17390__auto__);
} else {
return (new devcards.core.IdentiyOptions(self__.obj,self__.__meta,cljs.core.not_empty.call(null,cljs.core.dissoc.call(null,self__.__extmap,k__17390__auto__)),null));
}
});

devcards.core.IdentiyOptions.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__17387__auto__,k__17388__auto__,G__19751){
var self__ = this;
var this__17387__auto____$1 = this;
var pred__19755 = cljs.core.keyword_identical_QMARK_;
var expr__19756 = k__17388__auto__;
if(cljs.core.truth_(pred__19755.call(null,new cljs.core.Keyword(null,"obj","obj",981763962),expr__19756))){
return (new devcards.core.IdentiyOptions(G__19751,self__.__meta,self__.__extmap,null));
} else {
return (new devcards.core.IdentiyOptions(self__.obj,self__.__meta,cljs.core.assoc.call(null,self__.__extmap,k__17388__auto__,G__19751),null));
}
});

devcards.core.IdentiyOptions.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__17392__auto__){
var self__ = this;
var this__17392__auto____$1 = this;
return cljs.core.seq.call(null,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.IdentiyOptions.prototype.devcards$core$IDevcardOptions$ = true;

devcards.core.IdentiyOptions.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,opts){
var self__ = this;
var this$__$1 = this;
return opts;
});

devcards.core.IdentiyOptions.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__17379__auto__,G__19751){
var self__ = this;
var this__17379__auto____$1 = this;
return (new devcards.core.IdentiyOptions(self__.obj,G__19751,self__.__extmap,self__.__hash));
});

devcards.core.IdentiyOptions.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__17385__auto__,entry__17386__auto__){
var self__ = this;
var this__17385__auto____$1 = this;
if(cljs.core.vector_QMARK_.call(null,entry__17386__auto__)){
return cljs.core._assoc.call(null,this__17385__auto____$1,cljs.core._nth.call(null,entry__17386__auto__,(0)),cljs.core._nth.call(null,entry__17386__auto__,(1)));
} else {
return cljs.core.reduce.call(null,cljs.core._conj,this__17385__auto____$1,entry__17386__auto__);
}
});

devcards.core.IdentiyOptions.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"obj","obj",-1672671807,null)], null);
});

devcards.core.IdentiyOptions.cljs$lang$type = true;

devcards.core.IdentiyOptions.cljs$lang$ctorPrSeq = (function (this__17414__auto__){
return cljs.core._conj.call(null,cljs.core.List.EMPTY,"devcards.core/IdentiyOptions");
});

devcards.core.IdentiyOptions.cljs$lang$ctorPrWriter = (function (this__17414__auto__,writer__17415__auto__){
return cljs.core._write.call(null,writer__17415__auto__,"devcards.core/IdentiyOptions");
});

devcards.core.__GT_IdentiyOptions = (function devcards$core$__GT_IdentiyOptions(obj){
return (new devcards.core.IdentiyOptions(obj,null,null,null));
});

devcards.core.map__GT_IdentiyOptions = (function devcards$core$map__GT_IdentiyOptions(G__19753){
return (new devcards.core.IdentiyOptions(new cljs.core.Keyword(null,"obj","obj",981763962).cljs$core$IFn$_invoke$arity$1(G__19753),null,cljs.core.dissoc.call(null,G__19753,new cljs.core.Keyword(null,"obj","obj",981763962)),null));
});

devcards.core.atom_like_options = (function devcards$core$atom_like_options(main_obj,p__19759){
var map__19762 = p__19759;
var map__19762__$1 = ((((!((map__19762 == null)))?((((map__19762.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19762.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19762):map__19762);
var devcard_opts = map__19762__$1;
var options = cljs.core.get.call(null,map__19762__$1,new cljs.core.Keyword(null,"options","options",99638489));
return cljs.core.assoc.call(null,devcard_opts,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742),((function (map__19762,map__19762__$1,devcard_opts,options){
return (function (data_atom,_){
return devcards.util.edn_renderer.html_edn.call(null,cljs.core.deref.call(null,data_atom));
});})(map__19762,map__19762__$1,devcard_opts,options))
,new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804),main_obj,new cljs.core.Keyword(null,"options","options",99638489),cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"history","history",-247395220),true], null),devcards.core.assert_options_map.call(null,options)));
});

/**
* @constructor
 * @implements {cljs.core.IRecord}
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
 * @implements {devcards.core.IDevcardOptions}
*/
devcards.core.AtomLikeOptions = (function (obj,__meta,__extmap,__hash){
this.obj = obj;
this.__meta = __meta;
this.__extmap = __extmap;
this.__hash = __hash;
this.cljs$lang$protocol_mask$partition0$ = 2229667594;
this.cljs$lang$protocol_mask$partition1$ = 8192;
})
devcards.core.AtomLikeOptions.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__17380__auto__,k__17381__auto__){
var self__ = this;
var this__17380__auto____$1 = this;
return cljs.core._lookup.call(null,this__17380__auto____$1,k__17381__auto__,null);
});

devcards.core.AtomLikeOptions.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__17382__auto__,k19765,else__17383__auto__){
var self__ = this;
var this__17382__auto____$1 = this;
var G__19767 = (((k19765 instanceof cljs.core.Keyword))?k19765.fqn:null);
switch (G__19767) {
case "obj":
return self__.obj;

break;
default:
return cljs.core.get.call(null,self__.__extmap,k19765,else__17383__auto__);

}
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__17394__auto__,writer__17395__auto__,opts__17396__auto__){
var self__ = this;
var this__17394__auto____$1 = this;
var pr_pair__17397__auto__ = ((function (this__17394__auto____$1){
return (function (keyval__17398__auto__){
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,cljs.core.pr_writer,""," ","",opts__17396__auto__,keyval__17398__auto__);
});})(this__17394__auto____$1))
;
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,pr_pair__17397__auto__,"#devcards.core.AtomLikeOptions{",", ","}",opts__17396__auto__,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IIterable$ = true;

devcards.core.AtomLikeOptions.prototype.cljs$core$IIterable$_iterator$arity$1 = (function (G__19764){
var self__ = this;
var G__19764__$1 = this;
return (new cljs.core.RecordIter((0),G__19764__$1,1,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"obj","obj",981763962)], null),cljs.core._iterator.call(null,self__.__extmap)));
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__17378__auto__){
var self__ = this;
var this__17378__auto____$1 = this;
return self__.__meta;
});

devcards.core.AtomLikeOptions.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__17374__auto__){
var self__ = this;
var this__17374__auto____$1 = this;
return (new devcards.core.AtomLikeOptions(self__.obj,self__.__meta,self__.__extmap,self__.__hash));
});

devcards.core.AtomLikeOptions.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__17384__auto__){
var self__ = this;
var this__17384__auto____$1 = this;
return (1 + cljs.core.count.call(null,self__.__extmap));
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__17375__auto__){
var self__ = this;
var this__17375__auto____$1 = this;
var h__17201__auto__ = self__.__hash;
if(!((h__17201__auto__ == null))){
return h__17201__auto__;
} else {
var h__17201__auto____$1 = cljs.core.hash_imap.call(null,this__17375__auto____$1);
self__.__hash = h__17201__auto____$1;

return h__17201__auto____$1;
}
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this__17376__auto__,other__17377__auto__){
var self__ = this;
var this__17376__auto____$1 = this;
if(cljs.core.truth_((function (){var and__16754__auto__ = other__17377__auto__;
if(cljs.core.truth_(and__16754__auto__)){
var and__16754__auto____$1 = (this__17376__auto____$1.constructor === other__17377__auto__.constructor);
if(and__16754__auto____$1){
return cljs.core.equiv_map.call(null,this__17376__auto____$1,other__17377__auto__);
} else {
return and__16754__auto____$1;
}
} else {
return and__16754__auto__;
}
})())){
return true;
} else {
return false;
}
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__17389__auto__,k__17390__auto__){
var self__ = this;
var this__17389__auto____$1 = this;
if(cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"obj","obj",981763962),null], null), null),k__17390__auto__)){
return cljs.core.dissoc.call(null,cljs.core.with_meta.call(null,cljs.core.into.call(null,cljs.core.PersistentArrayMap.EMPTY,this__17389__auto____$1),self__.__meta),k__17390__auto__);
} else {
return (new devcards.core.AtomLikeOptions(self__.obj,self__.__meta,cljs.core.not_empty.call(null,cljs.core.dissoc.call(null,self__.__extmap,k__17390__auto__)),null));
}
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__17387__auto__,k__17388__auto__,G__19764){
var self__ = this;
var this__17387__auto____$1 = this;
var pred__19768 = cljs.core.keyword_identical_QMARK_;
var expr__19769 = k__17388__auto__;
if(cljs.core.truth_(pred__19768.call(null,new cljs.core.Keyword(null,"obj","obj",981763962),expr__19769))){
return (new devcards.core.AtomLikeOptions(G__19764,self__.__meta,self__.__extmap,null));
} else {
return (new devcards.core.AtomLikeOptions(self__.obj,self__.__meta,cljs.core.assoc.call(null,self__.__extmap,k__17388__auto__,G__19764),null));
}
});

devcards.core.AtomLikeOptions.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__17392__auto__){
var self__ = this;
var this__17392__auto____$1 = this;
return cljs.core.seq.call(null,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.AtomLikeOptions.prototype.devcards$core$IDevcardOptions$ = true;

devcards.core.AtomLikeOptions.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,opts){
var self__ = this;
var this$__$1 = this;
return devcards.core.atom_like_options.call(null,self__.obj,opts);
});

devcards.core.AtomLikeOptions.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__17379__auto__,G__19764){
var self__ = this;
var this__17379__auto____$1 = this;
return (new devcards.core.AtomLikeOptions(self__.obj,G__19764,self__.__extmap,self__.__hash));
});

devcards.core.AtomLikeOptions.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__17385__auto__,entry__17386__auto__){
var self__ = this;
var this__17385__auto____$1 = this;
if(cljs.core.vector_QMARK_.call(null,entry__17386__auto__)){
return cljs.core._assoc.call(null,this__17385__auto____$1,cljs.core._nth.call(null,entry__17386__auto__,(0)),cljs.core._nth.call(null,entry__17386__auto__,(1)));
} else {
return cljs.core.reduce.call(null,cljs.core._conj,this__17385__auto____$1,entry__17386__auto__);
}
});

devcards.core.AtomLikeOptions.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"obj","obj",-1672671807,null)], null);
});

devcards.core.AtomLikeOptions.cljs$lang$type = true;

devcards.core.AtomLikeOptions.cljs$lang$ctorPrSeq = (function (this__17414__auto__){
return cljs.core._conj.call(null,cljs.core.List.EMPTY,"devcards.core/AtomLikeOptions");
});

devcards.core.AtomLikeOptions.cljs$lang$ctorPrWriter = (function (this__17414__auto__,writer__17415__auto__){
return cljs.core._write.call(null,writer__17415__auto__,"devcards.core/AtomLikeOptions");
});

devcards.core.__GT_AtomLikeOptions = (function devcards$core$__GT_AtomLikeOptions(obj){
return (new devcards.core.AtomLikeOptions(obj,null,null,null));
});

devcards.core.map__GT_AtomLikeOptions = (function devcards$core$map__GT_AtomLikeOptions(G__19766){
return (new devcards.core.AtomLikeOptions(new cljs.core.Keyword(null,"obj","obj",981763962).cljs$core$IFn$_invoke$arity$1(G__19766),null,cljs.core.dissoc.call(null,G__19766,new cljs.core.Keyword(null,"obj","obj",981763962)),null));
});

devcards.core.edn_like_options = (function devcards$core$edn_like_options(main_obj,devcard_opts){
return cljs.core.assoc.call(null,devcard_opts,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742),devcards.util.edn_renderer.html_edn.call(null,((((!((main_obj == null)))?((((main_obj.cljs$lang$protocol_mask$partition0$ & (32768))) || (main_obj.cljs$core$IDeref$))?true:(((!main_obj.cljs$lang$protocol_mask$partition0$))?cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,main_obj):false)):cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,main_obj)))?cljs.core.deref.call(null,main_obj):main_obj)));
});

/**
* @constructor
 * @implements {cljs.core.IRecord}
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
 * @implements {devcards.core.IDevcardOptions}
*/
devcards.core.EdnLikeOptions = (function (obj,__meta,__extmap,__hash){
this.obj = obj;
this.__meta = __meta;
this.__extmap = __extmap;
this.__hash = __hash;
this.cljs$lang$protocol_mask$partition0$ = 2229667594;
this.cljs$lang$protocol_mask$partition1$ = 8192;
})
devcards.core.EdnLikeOptions.prototype.cljs$core$ILookup$_lookup$arity$2 = (function (this__17380__auto__,k__17381__auto__){
var self__ = this;
var this__17380__auto____$1 = this;
return cljs.core._lookup.call(null,this__17380__auto____$1,k__17381__auto__,null);
});

devcards.core.EdnLikeOptions.prototype.cljs$core$ILookup$_lookup$arity$3 = (function (this__17382__auto__,k19775,else__17383__auto__){
var self__ = this;
var this__17382__auto____$1 = this;
var G__19777 = (((k19775 instanceof cljs.core.Keyword))?k19775.fqn:null);
switch (G__19777) {
case "obj":
return self__.obj;

break;
default:
return cljs.core.get.call(null,self__.__extmap,k19775,else__17383__auto__);

}
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (this__17394__auto__,writer__17395__auto__,opts__17396__auto__){
var self__ = this;
var this__17394__auto____$1 = this;
var pr_pair__17397__auto__ = ((function (this__17394__auto____$1){
return (function (keyval__17398__auto__){
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,cljs.core.pr_writer,""," ","",opts__17396__auto__,keyval__17398__auto__);
});})(this__17394__auto____$1))
;
return cljs.core.pr_sequential_writer.call(null,writer__17395__auto__,pr_pair__17397__auto__,"#devcards.core.EdnLikeOptions{",", ","}",opts__17396__auto__,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IIterable$ = true;

devcards.core.EdnLikeOptions.prototype.cljs$core$IIterable$_iterator$arity$1 = (function (G__19774){
var self__ = this;
var G__19774__$1 = this;
return (new cljs.core.RecordIter((0),G__19774__$1,1,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"obj","obj",981763962)], null),cljs.core._iterator.call(null,self__.__extmap)));
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IMeta$_meta$arity$1 = (function (this__17378__auto__){
var self__ = this;
var this__17378__auto____$1 = this;
return self__.__meta;
});

devcards.core.EdnLikeOptions.prototype.cljs$core$ICloneable$_clone$arity$1 = (function (this__17374__auto__){
var self__ = this;
var this__17374__auto____$1 = this;
return (new devcards.core.EdnLikeOptions(self__.obj,self__.__meta,self__.__extmap,self__.__hash));
});

devcards.core.EdnLikeOptions.prototype.cljs$core$ICounted$_count$arity$1 = (function (this__17384__auto__){
var self__ = this;
var this__17384__auto____$1 = this;
return (1 + cljs.core.count.call(null,self__.__extmap));
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IHash$_hash$arity$1 = (function (this__17375__auto__){
var self__ = this;
var this__17375__auto____$1 = this;
var h__17201__auto__ = self__.__hash;
if(!((h__17201__auto__ == null))){
return h__17201__auto__;
} else {
var h__17201__auto____$1 = cljs.core.hash_imap.call(null,this__17375__auto____$1);
self__.__hash = h__17201__auto____$1;

return h__17201__auto____$1;
}
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this__17376__auto__,other__17377__auto__){
var self__ = this;
var this__17376__auto____$1 = this;
if(cljs.core.truth_((function (){var and__16754__auto__ = other__17377__auto__;
if(cljs.core.truth_(and__16754__auto__)){
var and__16754__auto____$1 = (this__17376__auto____$1.constructor === other__17377__auto__.constructor);
if(and__16754__auto____$1){
return cljs.core.equiv_map.call(null,this__17376__auto____$1,other__17377__auto__);
} else {
return and__16754__auto____$1;
}
} else {
return and__16754__auto__;
}
})())){
return true;
} else {
return false;
}
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IMap$_dissoc$arity$2 = (function (this__17389__auto__,k__17390__auto__){
var self__ = this;
var this__17389__auto____$1 = this;
if(cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"obj","obj",981763962),null], null), null),k__17390__auto__)){
return cljs.core.dissoc.call(null,cljs.core.with_meta.call(null,cljs.core.into.call(null,cljs.core.PersistentArrayMap.EMPTY,this__17389__auto____$1),self__.__meta),k__17390__auto__);
} else {
return (new devcards.core.EdnLikeOptions(self__.obj,self__.__meta,cljs.core.not_empty.call(null,cljs.core.dissoc.call(null,self__.__extmap,k__17390__auto__)),null));
}
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IAssociative$_assoc$arity$3 = (function (this__17387__auto__,k__17388__auto__,G__19774){
var self__ = this;
var this__17387__auto____$1 = this;
var pred__19778 = cljs.core.keyword_identical_QMARK_;
var expr__19779 = k__17388__auto__;
if(cljs.core.truth_(pred__19778.call(null,new cljs.core.Keyword(null,"obj","obj",981763962),expr__19779))){
return (new devcards.core.EdnLikeOptions(G__19774,self__.__meta,self__.__extmap,null));
} else {
return (new devcards.core.EdnLikeOptions(self__.obj,self__.__meta,cljs.core.assoc.call(null,self__.__extmap,k__17388__auto__,G__19774),null));
}
});

devcards.core.EdnLikeOptions.prototype.cljs$core$ISeqable$_seq$arity$1 = (function (this__17392__auto__){
var self__ = this;
var this__17392__auto____$1 = this;
return cljs.core.seq.call(null,cljs.core.concat.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [(new cljs.core.PersistentVector(null,2,(5),cljs.core.PersistentVector.EMPTY_NODE,[new cljs.core.Keyword(null,"obj","obj",981763962),self__.obj],null))], null),self__.__extmap));
});

devcards.core.EdnLikeOptions.prototype.devcards$core$IDevcardOptions$ = true;

devcards.core.EdnLikeOptions.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var self__ = this;
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,self__.obj,devcard_opts);
});

devcards.core.EdnLikeOptions.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (this__17379__auto__,G__19774){
var self__ = this;
var this__17379__auto____$1 = this;
return (new devcards.core.EdnLikeOptions(self__.obj,G__19774,self__.__extmap,self__.__hash));
});

devcards.core.EdnLikeOptions.prototype.cljs$core$ICollection$_conj$arity$2 = (function (this__17385__auto__,entry__17386__auto__){
var self__ = this;
var this__17385__auto____$1 = this;
if(cljs.core.vector_QMARK_.call(null,entry__17386__auto__)){
return cljs.core._assoc.call(null,this__17385__auto____$1,cljs.core._nth.call(null,entry__17386__auto__,(0)),cljs.core._nth.call(null,entry__17386__auto__,(1)));
} else {
return cljs.core.reduce.call(null,cljs.core._conj,this__17385__auto____$1,entry__17386__auto__);
}
});

devcards.core.EdnLikeOptions.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"obj","obj",-1672671807,null)], null);
});

devcards.core.EdnLikeOptions.cljs$lang$type = true;

devcards.core.EdnLikeOptions.cljs$lang$ctorPrSeq = (function (this__17414__auto__){
return cljs.core._conj.call(null,cljs.core.List.EMPTY,"devcards.core/EdnLikeOptions");
});

devcards.core.EdnLikeOptions.cljs$lang$ctorPrWriter = (function (this__17414__auto__,writer__17415__auto__){
return cljs.core._write.call(null,writer__17415__auto__,"devcards.core/EdnLikeOptions");
});

devcards.core.__GT_EdnLikeOptions = (function devcards$core$__GT_EdnLikeOptions(obj){
return (new devcards.core.EdnLikeOptions(obj,null,null,null));
});

devcards.core.map__GT_EdnLikeOptions = (function devcards$core$map__GT_EdnLikeOptions(G__19776){
return (new devcards.core.EdnLikeOptions(new cljs.core.Keyword(null,"obj","obj",981763962).cljs$core$IFn$_invoke$arity$1(G__19776),null,cljs.core.dissoc.call(null,G__19776,new cljs.core.Keyword(null,"obj","obj",981763962)),null));
});

devcards.core.atom_like_QMARK_ = (function devcards$core$atom_like_QMARK_(x){
var and__16754__auto__ = ((!((x == null)))?((((x.cljs$lang$protocol_mask$partition1$ & (2))) || (x.cljs$core$IWatchable$))?true:(((!x.cljs$lang$protocol_mask$partition1$))?cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IWatchable,x):false)):cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IWatchable,x));
if(and__16754__auto__){
if(!((x == null))){
if(((x.cljs$lang$protocol_mask$partition0$ & (32768))) || (x.cljs$core$IDeref$)){
return true;
} else {
if((!x.cljs$lang$protocol_mask$partition0$)){
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,x);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,x);
}
} else {
return and__16754__auto__;
}
});
devcards.core.edn_like_QMARK_ = (function devcards$core$edn_like_QMARK_(x){
if(!((x == null))){
if(((x.cljs$lang$protocol_mask$partition0$ & (32768))) || (x.cljs$core$IDeref$)){
return true;
} else {
if((!x.cljs$lang$protocol_mask$partition0$)){
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,x);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_.call(null,cljs.core.IDeref,x);
}
});
devcards.core.coerce_to_devcards_options = (function devcards$core$coerce_to_devcards_options(main_obj){
if(((!((main_obj == null)))?(((false) || (main_obj.devcards$core$IDevcardOptions$))?true:(((!main_obj.cljs$lang$protocol_mask$partition$))?cljs.core.native_satisfies_QMARK_.call(null,devcards.core.IDevcardOptions,main_obj):false)):cljs.core.native_satisfies_QMARK_.call(null,devcards.core.IDevcardOptions,main_obj))){
return main_obj;
} else {
if(cljs.core.truth_(devcards.core.atom_like_QMARK_.call(null,main_obj))){
return (new devcards.core.AtomLikeOptions(main_obj,null,null,null));
} else {
if(cljs.core.truth_(devcards.core.edn_like_QMARK_.call(null,main_obj))){
return (new devcards.core.EdnLikeOptions(main_obj,null,null,null));
} else {
return (new devcards.core.IdentiyOptions(main_obj,null,null,null));

}
}
}
});
devcards.core.card_base = (function devcards$core$card_base(opts){
var opts__$1 = cljs.core.assoc.call(null,opts,new cljs.core.Keyword(null,"path","path",-188191168),new cljs.core.Keyword(null,"path","path",-188191168).cljs$core$IFn$_invoke$arity$1(devcards.system._STAR_devcard_data_STAR_));
if((function (){var G__19793 = new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742).cljs$core$IFn$_invoke$arity$1(opts__$1);
if(!((G__19793 == null))){
if((false) || (G__19793.devcards$core$IDevcard$)){
return true;
} else {
if((!G__19793.cljs$lang$protocol_mask$partition$)){
return cljs.core.native_satisfies_QMARK_.call(null,devcards.core.IDevcard,G__19793);
} else {
return false;
}
}
} else {
return cljs.core.native_satisfies_QMARK_.call(null,devcards.core.IDevcard,G__19793);
}
})()){
return devcards.core._devcard.call(null,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742).cljs$core$IFn$_invoke$arity$1(opts__$1),opts__$1);
} else {
return devcards.core.card_with_errors.call(null,devcards.core._devcard_options.call(null,devcards.core.coerce_to_devcards_options.call(null,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742).cljs$core$IFn$_invoke$arity$1(opts__$1)),opts__$1));
}
});
devcards.core.dom_node_STAR_ = (function devcards$core$dom_node_STAR_(node_fn){
return (function (data_atom,owner){
return React.createElement(devcards.core.DomComponent,{"node_fn": node_fn, "data_atom": data_atom});
});
});
(devcards.core.IDevcardOptions["string"] = true);

(devcards.core._devcard_options["string"] = (function (this$,devcard_opts){
return cljs.core.update_in.call(null,devcard_opts,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742)], null),devcards.core.markdown__GT_react);
}));
cljs.core.PersistentArrayMap.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.PersistentArrayMap.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.PersistentVector.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.PersistentVector.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.PersistentHashSet.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.PersistentHashSet.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.List.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.List.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.LazySeq.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.LazySeq.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.Cons.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.Cons.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.EmptyList.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.EmptyList.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.edn_like_options.call(null,this$__$1,devcard_opts);
});
cljs.core.Atom.prototype.devcards$core$IDevcardOptions$ = true;

cljs.core.Atom.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this$,devcard_opts){
var this$__$1 = this;
return devcards.core.atom_like_options.call(null,this$__$1,devcard_opts);
});
devcards.core.can_go_back = (function devcards$core$can_go_back(this$){
var map__19796 = cljs.core.deref.call(null,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013)));
var map__19796__$1 = ((((!((map__19796 == null)))?((((map__19796.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19796.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19796):map__19796);
var history = cljs.core.get.call(null,map__19796__$1,new cljs.core.Keyword(null,"history","history",-247395220));
var pointer = cljs.core.get.call(null,map__19796__$1,new cljs.core.Keyword(null,"pointer","pointer",85071187));
return ((pointer + (1)) < cljs.core.count.call(null,history));
});
devcards.core.can_go_forward = (function devcards$core$can_go_forward(this$){
return (new cljs.core.Keyword(null,"pointer","pointer",85071187).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013)))) > (0));
});
devcards.core.in_time_machine_QMARK_ = (function devcards$core$in_time_machine_QMARK_(this$){
return !((new cljs.core.Keyword(null,"pointer","pointer",85071187).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013)))) === (0)));
});
devcards.core.back_in_history_BANG_ = (function devcards$core$back_in_history_BANG_(this$){
var history_atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013));
var map__19800 = cljs.core.deref.call(null,history_atom);
var map__19800__$1 = ((((!((map__19800 == null)))?((((map__19800.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19800.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19800):map__19800);
var history = cljs.core.get.call(null,map__19800__$1,new cljs.core.Keyword(null,"history","history",-247395220));
var pointer = cljs.core.get.call(null,map__19800__$1,new cljs.core.Keyword(null,"pointer","pointer",85071187));
if(cljs.core.truth_(devcards.core.can_go_back.call(null,this$))){
cljs.core.swap_BANG_.call(null,history_atom,cljs.core.assoc,new cljs.core.Keyword(null,"pointer","pointer",85071187),(pointer + (1)),new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927),true);

cljs.core.reset_BANG_.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504)),cljs.core.nth.call(null,history,(pointer + (1))));

return this$.forceUpdate();
} else {
return null;
}
});
devcards.core.forward_in_history_BANG_ = (function devcards$core$forward_in_history_BANG_(this$){
var history_atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013));
var map__19804 = cljs.core.deref.call(null,history_atom);
var map__19804__$1 = ((((!((map__19804 == null)))?((((map__19804.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19804.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19804):map__19804);
var history = cljs.core.get.call(null,map__19804__$1,new cljs.core.Keyword(null,"history","history",-247395220));
var pointer = cljs.core.get.call(null,map__19804__$1,new cljs.core.Keyword(null,"pointer","pointer",85071187));
if(cljs.core.truth_(devcards.core.can_go_forward.call(null,this$))){
cljs.core.swap_BANG_.call(null,history_atom,cljs.core.assoc,new cljs.core.Keyword(null,"pointer","pointer",85071187),(pointer - (1)),new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927),true);

cljs.core.reset_BANG_.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504)),cljs.core.nth.call(null,history,(pointer - (1))));

return this$.forceUpdate();
} else {
return null;
}
});
devcards.core.continue_on_BANG_ = (function devcards$core$continue_on_BANG_(this$){
var history_atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013));
var map__19808 = cljs.core.deref.call(null,history_atom);
var map__19808__$1 = ((((!((map__19808 == null)))?((((map__19808.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19808.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19808):map__19808);
var history = cljs.core.get.call(null,map__19808__$1,new cljs.core.Keyword(null,"history","history",-247395220));
if(cljs.core.truth_(devcards.core.can_go_forward.call(null,this$))){
cljs.core.swap_BANG_.call(null,history_atom,cljs.core.assoc,new cljs.core.Keyword(null,"pointer","pointer",85071187),(0),new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927),true);

cljs.core.reset_BANG_.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504)),cljs.core.first.call(null,history));

return this$.forceUpdate();
} else {
return null;
}
});
devcards.core.HistoryComponent = React.createClass({"getInitialState": (function (){
return {"unique_id": [cljs.core.str(cljs.core.gensym.call(null,new cljs.core.Symbol(null,"devcards-history-runner-","devcards-history-runner-",-1709703043,null)))].join(''), "history_atom": cljs.core.atom.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"history","history",-247395220),cljs.core.List.EMPTY,new cljs.core.Keyword(null,"pointer","pointer",85071187),(0)], null))};
}), "componentWillMount": (function (){
var this$ = this;
return cljs.core.swap_BANG_.call(null,devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013)),cljs.core.assoc_in,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"history","history",-247395220)], null),cljs.core._conj.call(null,cljs.core.List.EMPTY,cljs.core.deref.call(null,devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504)))));
}), "componentDidMount": (function (){
var this$ = this;
var data_atom = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
var id = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"unique_id","unique_id",-796578329));
var history_atom = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"history_atom","history_atom",-533227013));
if(cljs.core.truth_((function (){var and__16754__auto__ = data_atom;
if(cljs.core.truth_(and__16754__auto__)){
return id;
} else {
return and__16754__auto__;
}
})())){
return cljs.core.add_watch.call(null,data_atom,id,((function (data_atom,id,history_atom,this$){
return (function (_,___$1,___$2,n){
if(cljs.core.truth_(devcards.core.in_time_machine_QMARK_.call(null,this$))){
return cljs.core.swap_BANG_.call(null,history_atom,((function (data_atom,id,history_atom,this$){
return (function (p__19810){
var map__19811 = p__19810;
var map__19811__$1 = ((((!((map__19811 == null)))?((((map__19811.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19811.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19811):map__19811);
var ha = map__19811__$1;
var pointer = cljs.core.get.call(null,map__19811__$1,new cljs.core.Keyword(null,"pointer","pointer",85071187));
var history = cljs.core.get.call(null,map__19811__$1,new cljs.core.Keyword(null,"history","history",-247395220));
var ignore_click = cljs.core.get.call(null,map__19811__$1,new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927));
if(cljs.core.truth_(ignore_click)){
return cljs.core.assoc.call(null,ha,new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927),false);
} else {
return cljs.core.assoc.call(null,ha,new cljs.core.Keyword(null,"history","history",-247395220),(function (){var abridged_hist = cljs.core.drop.call(null,pointer,history);
if(cljs.core.not_EQ_.call(null,n,cljs.core.first.call(null,abridged_hist))){
return cljs.core.cons.call(null,n,abridged_hist);
} else {
return abridged_hist;
}
})(),new cljs.core.Keyword(null,"pointer","pointer",85071187),(0));
}
});})(data_atom,id,history_atom,this$))
);
} else {
return cljs.core.swap_BANG_.call(null,history_atom,cljs.core.assoc,new cljs.core.Keyword(null,"history","history",-247395220),(function (){var hist = new cljs.core.Keyword(null,"history","history",-247395220).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,history_atom));
if(cljs.core.not_EQ_.call(null,n,cljs.core.first.call(null,hist))){
return cljs.core.cons.call(null,n,hist);
} else {
return hist;
}
})(),new cljs.core.Keyword(null,"ignore-click","ignore-click",-875855927),false);
}
});})(data_atom,id,history_atom,this$))
);
} else {
return null;
}
}), "render": (function (){
var this$ = this;
if(cljs.core.truth_((function (){var or__16766__auto__ = devcards.core.can_go_back.call(null,this$);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return devcards.core.can_go_forward.call(null,this$);
}
})())){
return React.createElement("div",{"style": {"display": (cljs.core.truth_((function (){var or__16766__auto__ = devcards.core.can_go_back.call(null,this$);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return devcards.core.can_go_forward.call(null,this$);
}
})())?"block":"none")}, "className": "com-rigsomelight-devcards-history-control-bar"},sablono.interpreter.interpret.call(null,(function (){var action = ((function (this$){
return (function (e){
e.preventDefault();

return devcards.core.back_in_history_BANG_.call(null,this$);
});})(this$))
;
return React.createElement("button",{"style": {"visibility": (cljs.core.truth_(devcards.core.can_go_back.call(null,this$))?"visible":"hidden")}, "href": "#", "onClick": action, "onTouchEnd": action},React.createElement("span",{"className": "com-rigsomelight-devcards-history-control-left"},""));
})()),sablono.interpreter.interpret.call(null,(function (){var action = ((function (this$){
return (function (e){
e.preventDefault();

var data_atom = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"data_atom","data_atom",257894504));
return cljs.core.reset_BANG_.call(null,data_atom,cljs.core.deref.call(null,data_atom));
});})(this$))
;
return React.createElement("button",{"style": {"visibility": (cljs.core.truth_(devcards.core.can_go_forward.call(null,this$))?"visible":"hidden")}, "onClick": action, "onTouchEnd": action},React.createElement("span",{"className": "com-rigsomelight-devcards-history-stop"},""));
})()),sablono.interpreter.interpret.call(null,(function (){var action = ((function (this$){
return (function (e){
e.preventDefault();

return devcards.core.forward_in_history_BANG_.call(null,this$);
});})(this$))
;
return React.createElement("button",{"style": {"visibility": (cljs.core.truth_(devcards.core.can_go_forward.call(null,this$))?"visible":"hidden")}, "onClick": action, "onTouchEnd": action},React.createElement("span",{"className": "com-rigsomelight-devcards-history-control-right"},""));
})()),sablono.interpreter.interpret.call(null,(function (){var listener = ((function (this$){
return (function (e){
e.preventDefault();

return devcards.core.continue_on_BANG_.call(null,this$);
});})(this$))
;
return React.createElement("button",{"style": {"visibility": (cljs.core.truth_(devcards.core.can_go_forward.call(null,this$))?"visible":"hidden")}, "onClick": listener, "onTouchEnd": listener},React.createElement("span",{"className": "com-rigsomelight-devcards-history-control-small-arrow"}),React.createElement("span",{"className": "com-rigsomelight-devcards-history-control-small-arrow"}),React.createElement("span",{"className": "com-rigsomelight-devcards-history-control-block"}));
})()));
} else {
return null;
}
})});
devcards.core.hist_recorder_STAR_ = (function devcards$core$hist_recorder_STAR_(data_atom){
return React.createElement(devcards.core.HistoryComponent,{"data_atom": data_atom});
});
devcards.core.collect_test = (function devcards$core$collect_test(m){
return cljs.test.update_current_env_BANG_.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"_devcards_collect_tests","_devcards_collect_tests",-1114031206)], null),cljs.core.conj,cljs.core.merge.call(null,cljs.core.select_keys.call(null,cljs.test.get_current_env.call(null),new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523)], null)),m));
});
cljs.core._add_method.call(null,cljs.test.report,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805),new cljs.core.Keyword(null,"pass","pass",1574159993)], null),(function (m){
cljs.test.inc_report_counter_BANG_.call(null,new cljs.core.Keyword(null,"pass","pass",1574159993));

devcards.core.collect_test.call(null,m);

return m;
}));
cljs.core._add_method.call(null,cljs.test.report,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805),new cljs.core.Keyword(null,"fail","fail",1706214930)], null),(function (m){
cljs.test.inc_report_counter_BANG_.call(null,new cljs.core.Keyword(null,"fail","fail",1706214930));

devcards.core.collect_test.call(null,m);

return m;
}));
cljs.core._add_method.call(null,cljs.test.report,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805),new cljs.core.Keyword(null,"error","error",-978969032)], null),(function (m){
cljs.test.inc_report_counter_BANG_.call(null,new cljs.core.Keyword(null,"error","error",-978969032));

devcards.core.collect_test.call(null,m);

return m;
}));
cljs.core._add_method.call(null,cljs.test.report,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805),new cljs.core.Keyword(null,"test-doc","test-doc",1730699463)], null),(function (m){
devcards.core.collect_test.call(null,m);

return m;
}));
devcards.core.run_test_block = (function devcards$core$run_test_block(f){
var _STAR_current_env_STAR_19826 = cljs.test._STAR_current_env_STAR_;
cljs.test._STAR_current_env_STAR_ = cljs.core.assoc.call(null,cljs.test.empty_env.call(null),new cljs.core.Keyword(null,"reporter","reporter",-805360621),new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805));

try{f.call(null);

return cljs.test.get_current_env.call(null);
}finally {cljs.test._STAR_current_env_STAR_ = _STAR_current_env_STAR_19826;
}});
if(typeof devcards.core.test_render !== 'undefined'){
} else {
devcards.core.test_render = (function (){var method_table__17679__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var prefer_table__17680__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var method_cache__17681__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var cached_hierarchy__17682__auto__ = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
var hierarchy__17683__auto__ = cljs.core.get.call(null,cljs.core.PersistentArrayMap.EMPTY,new cljs.core.Keyword(null,"hierarchy","hierarchy",-1053470341),cljs.core.get_global_hierarchy.call(null));
return (new cljs.core.MultiFn(cljs.core.symbol.call(null,"devcards.core","test-render"),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"default","default",-1987822328),hierarchy__17683__auto__,method_table__17679__auto__,prefer_table__17680__auto__,method_cache__17681__auto__,cached_hierarchy__17682__auto__));
})();
}
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"default","default",-1987822328),(function (m){
var attrs19827 = cljs.core.prn_str.call(null,m);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19827))?sablono.interpreter.attributes.call(null,attrs19827):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19827))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19827)], null))));
}));
devcards.core.display_message = (function devcards$core$display_message(p__19828,body){
var map__19832 = p__19828;
var map__19832__$1 = ((((!((map__19832 == null)))?((((map__19832.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19832.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19832):map__19832);
var message = cljs.core.get.call(null,map__19832__$1,new cljs.core.Keyword(null,"message","message",-406056002));
if(cljs.core.truth_(message)){
return React.createElement("div",null,(function (){var attrs19834 = message;
return cljs.core.apply.call(null,React.createElement,"span",((cljs.core.map_QMARK_.call(null,attrs19834))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["com-rigsomelight-devcards-test-message",null], null), null)], null),attrs19834)):{"className": "com-rigsomelight-devcards-test-message"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19834))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19834)], null))));
})(),sablono.interpreter.interpret.call(null,body));
} else {
return body;
}
});
devcards.core.render_pass_fail = (function devcards$core$render_pass_fail(p__19835){
var map__19843 = p__19835;
var map__19843__$1 = ((((!((map__19843 == null)))?((((map__19843.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19843.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19843):map__19843);
var m = map__19843__$1;
var expected = cljs.core.get.call(null,map__19843__$1,new cljs.core.Keyword(null,"expected","expected",1583670997));
var actual = cljs.core.get.call(null,map__19843__$1,new cljs.core.Keyword(null,"actual","actual",107306363));
var type = cljs.core.get.call(null,map__19843__$1,new cljs.core.Keyword(null,"type","type",1174270348));
return devcards.core.display_message.call(null,m,(function (){var attrs19845 = React.createElement(devcards.core.CodeHighlight,{"code": devcards.util.utils.pprint_code.call(null,expected), "lang": "clojure"});
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19845))?sablono.interpreter.attributes.call(null,attrs19845):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19845))?new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,((cljs.core._EQ_.call(null,type,new cljs.core.Keyword(null,"fail","fail",1706214930)))?React.createElement("div",{"style": {"marginTop": "5px"}},React.createElement("div",{"style": {"position": "absolute", "fontSize": "0.9em"}},"\u25B6"),React.createElement("div",{"style": {"marginLeft": "20px"}},sablono.interpreter.interpret.call(null,React.createElement(devcards.core.CodeHighlight,{"code": devcards.util.utils.pprint_code.call(null,actual), "lang": "clojure"})))):null))], null):new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19845),sablono.interpreter.interpret.call(null,((cljs.core._EQ_.call(null,type,new cljs.core.Keyword(null,"fail","fail",1706214930)))?React.createElement("div",{"style": {"marginTop": "5px"}},React.createElement("div",{"style": {"position": "absolute", "fontSize": "0.9em"}},"\u25B6"),React.createElement("div",{"style": {"marginLeft": "20px"}},sablono.interpreter.interpret.call(null,React.createElement(devcards.core.CodeHighlight,{"code": devcards.util.utils.pprint_code.call(null,actual), "lang": "clojure"})))):null))], null))));
})());
});
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"pass","pass",1574159993),(function (m){
return devcards.core.render_pass_fail.call(null,m);
}));
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"fail","fail",1706214930),(function (m){
return devcards.core.render_pass_fail.call(null,m);
}));
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"error","error",-978969032),(function (m){
return devcards.core.display_message.call(null,m,React.createElement("div",null,React.createElement("strong",null,"Error: "),(function (){var attrs19852 = [cljs.core.str(new cljs.core.Keyword(null,"actual","actual",107306363).cljs$core$IFn$_invoke$arity$1(m))].join('');
return cljs.core.apply.call(null,React.createElement,"code",((cljs.core.map_QMARK_.call(null,attrs19852))?sablono.interpreter.attributes.call(null,attrs19852):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19852))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19852)], null))));
})()));
}));
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"test-doc","test-doc",1730699463),(function (m){
var attrs19853 = devcards.core.markdown__GT_react.call(null,new cljs.core.Keyword(null,"documentation","documentation",1889593999).cljs$core$IFn$_invoke$arity$1(m));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19853))?sablono.interpreter.attributes.call(null,attrs19853):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19853))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19853)], null))));
}));
cljs.core._add_method.call(null,devcards.core.test_render,new cljs.core.Keyword(null,"context","context",-830191113),(function (p__19854){
var map__19855 = p__19854;
var map__19855__$1 = ((((!((map__19855 == null)))?((((map__19855.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19855.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19855):map__19855);
var testing_contexts = cljs.core.get.call(null,map__19855__$1,new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523));
var attrs19857 = cljs.core.interpose.call(null," / ",cljs.core.concat.call(null,cljs.core.map_indexed.call(null,((function (map__19855,map__19855__$1,testing_contexts){
return (function (i,t){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"span","span",1394872991),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),i,new cljs.core.Keyword(null,"style","style",-496642736),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"color","color",1011675173),"#bbb"], null)], null),t," "], null);
});})(map__19855,map__19855__$1,testing_contexts))
,cljs.core.reverse.call(null,cljs.core.rest.call(null,testing_contexts))),cljs.core._conj.call(null,cljs.core.List.EMPTY,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"span","span",1394872991),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"key","key",-1516042587),(-1)], null),cljs.core.first.call(null,testing_contexts)], null))));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19857))?sablono.interpreter.attributes.call(null,attrs19857):null),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19857))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19857)], null))));
}));
devcards.core.test_doc = (function devcards$core$test_doc(s){
return cljs.test.report.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"test-doc","test-doc",1730699463),new cljs.core.Keyword(null,"documentation","documentation",1889593999),s], null));
});
devcards.core.test_renderer = (function devcards$core$test_renderer(t){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"div","div",1057191632),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),cljs.core.pr_str.call(null,t),new cljs.core.Keyword(null,"className","className",-1983287057),[cljs.core.str("com-rigsomelight-devcards-test-line com-rigsomelight-devcards-"),cljs.core.str(cljs.core.name.call(null,new cljs.core.Keyword(null,"type","type",1174270348).cljs$core$IFn$_invoke$arity$1(t)))].join('')], null),devcards.core.test_render.call(null,t)], null);
});
devcards.core.layout_tests = (function devcards$core$layout_tests(tests){
var attrs19862 = new cljs.core.Keyword(null,"html-list","html-list",-2067090601).cljs$core$IFn$_invoke$arity$1(cljs.core.reduce.call(null,(function (p__19863,t){
var map__19864 = p__19863;
var map__19864__$1 = ((((!((map__19864 == null)))?((((map__19864.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19864.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19864):map__19864);
var last_context = cljs.core.get.call(null,map__19864__$1,new cljs.core.Keyword(null,"last-context","last-context",-820617548));
var html_list = cljs.core.get.call(null,map__19864__$1,new cljs.core.Keyword(null,"html-list","html-list",-2067090601));
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"last-context","last-context",-820617548),new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523).cljs$core$IFn$_invoke$arity$1(t),new cljs.core.Keyword(null,"html-list","html-list",-2067090601),(function (){var res = cljs.core._conj.call(null,cljs.core.List.EMPTY,devcards.core.test_renderer.call(null,t));
var res__$1 = ((cljs.core._EQ_.call(null,last_context,new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523).cljs$core$IFn$_invoke$arity$1(t)))?res:(cljs.core.truth_(cljs.core.not_empty.call(null,new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523).cljs$core$IFn$_invoke$arity$1(t)))?cljs.core.cons.call(null,devcards.core.test_renderer.call(null,cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"context","context",-830191113)], null),cljs.core.select_keys.call(null,t,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"testing-contexts","testing-contexts",-1485646523)], null)))),res):res));
return cljs.core.concat.call(null,html_list,res__$1);
})()], null);
}),cljs.core.PersistentArrayMap.EMPTY,cljs.core.reverse.call(null,tests)));
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs19862))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["com-rigsomelight-devcards-test-card",null], null), null)], null),attrs19862)):{"className": "com-rigsomelight-devcards-test-card"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs19862))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs19862)], null))));
});
devcards.core.render_tests = (function devcards$core$render_tests(this$,path,test_summary){
var error_QMARK_ = new cljs.core.Keyword(null,"error","error",-978969032).cljs$core$IFn$_invoke$arity$1(test_summary);
var tests = new cljs.core.Keyword(null,"_devcards_collect_tests","_devcards_collect_tests",-1114031206).cljs$core$IFn$_invoke$arity$1(test_summary);
var some_tests = cljs.core.filter.call(null,((function (error_QMARK_,tests){
return (function (p__19878){
var map__19879 = p__19878;
var map__19879__$1 = ((((!((map__19879 == null)))?((((map__19879.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19879.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19879):map__19879);
var type = cljs.core.get.call(null,map__19879__$1,new cljs.core.Keyword(null,"type","type",1174270348));
return cljs.core.not_EQ_.call(null,type,new cljs.core.Keyword(null,"test-doc","test-doc",1730699463));
});})(error_QMARK_,tests))
,new cljs.core.Keyword(null,"_devcards_collect_tests","_devcards_collect_tests",-1114031206).cljs$core$IFn$_invoke$arity$1(test_summary));
var total_tests = cljs.core.count.call(null,some_tests);
var map__19877 = new cljs.core.Keyword(null,"report-counters","report-counters",-1702609242).cljs$core$IFn$_invoke$arity$1(test_summary);
var map__19877__$1 = ((((!((map__19877 == null)))?((((map__19877.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19877.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19877):map__19877);
var fail = cljs.core.get.call(null,map__19877__$1,new cljs.core.Keyword(null,"fail","fail",1706214930));
var pass = cljs.core.get.call(null,map__19877__$1,new cljs.core.Keyword(null,"pass","pass",1574159993));
var error = cljs.core.get.call(null,map__19877__$1,new cljs.core.Keyword(null,"error","error",-978969032));
var error__$1 = (cljs.core.truth_(error_QMARK_)?(error + (1)):error);
return React.createElement("div",{"className": "com-rigsomelight-devcards-base com-rigsomelight-devcards-card-base-no-pad com-rigsomelight-devcards-typog"},React.createElement("div",{"className": "com-rigsomelight-devcards-panel-heading"},React.createElement("a",{"href": "#", "onClick": devcards.system.prevent__GT_.call(null,((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (){
return devcards.system.set_current_path_BANG_.call(null,devcards.system.app_state,path);
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
)},sablono.interpreter.interpret.call(null,(cljs.core.truth_(path)?[cljs.core.str(cljs.core.name.call(null,cljs.core.last.call(null,path)))].join(''):null))),React.createElement("button",{"style": {"float": "right", "margin": "3px 3px"}, "onClick": devcards.system.prevent__GT_.call(null,((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (){
return this$.setState({"filter": cljs.core.identity});
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
), "className": "com-rigsomelight-devcards-badge"},sablono.interpreter.interpret.call(null,total_tests)),sablono.interpreter.interpret.call(null,((((fail + error__$1) === (0)))?null:React.createElement("button",{"style": {"float": "right", "backgroundColor": "#F7918E", "color": "#fff", "margin": "3px 3px"}, "onClick": devcards.system.prevent__GT_.call(null,((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (){
return this$.setState({"filter": ((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (p__19882){
var map__19883 = p__19882;
var map__19883__$1 = ((((!((map__19883 == null)))?((((map__19883.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19883.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19883):map__19883);
var type = cljs.core.get.call(null,map__19883__$1,new cljs.core.Keyword(null,"type","type",1174270348));
return new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"fail","fail",1706214930),null,new cljs.core.Keyword(null,"error","error",-978969032),null], null), null).call(null,type);
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
});
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
), "className": "com-rigsomelight-devcards-badge"},sablono.interpreter.interpret.call(null,[cljs.core.str((fail + error__$1))].join(''))))),sablono.interpreter.interpret.call(null,((((pass == null)) || ((pass === (0))))?null:React.createElement("button",{"style": {"float": "right", "backgroundColor": "#92C648", "color": "#fff", "margin": "3px 3px"}, "onClick": devcards.system.prevent__GT_.call(null,((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (){
return this$.setState({"filter": ((function (error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1){
return (function (p__19885){
var map__19886 = p__19885;
var map__19886__$1 = ((((!((map__19886 == null)))?((((map__19886.cljs$lang$protocol_mask$partition0$ & (64))) || (map__19886.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__19886):map__19886);
var type = cljs.core.get.call(null,map__19886__$1,new cljs.core.Keyword(null,"type","type",1174270348));
return cljs.core._EQ_.call(null,type,new cljs.core.Keyword(null,"pass","pass",1574159993));
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
});
});})(error_QMARK_,tests,some_tests,total_tests,map__19877,map__19877__$1,fail,pass,error,error__$1))
), "className": "com-rigsomelight-devcards-badge"},sablono.interpreter.interpret.call(null,pass))))),React.createElement("div",{"className": devcards.system.devcards_rendered_card_class},sablono.interpreter.interpret.call(null,devcards.core.layout_tests.call(null,cljs.core.filter.call(null,(function (){var or__16766__auto__ = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"filter","filter",-948537934));
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core.identity;
}
})(),tests)))));
});
devcards.core.test_timeout = (800);
if(typeof devcards.core.test_channel !== 'undefined'){
} else {
devcards.core.test_channel = cljs.core.async.chan.call(null);
}
devcards.core.run_card_tests = (function devcards$core$run_card_tests(test_thunks){
var out = cljs.core.async.chan.call(null);
var test_env = cljs.core.assoc.call(null,cljs.test.empty_env.call(null),new cljs.core.Keyword(null,"reporter","reporter",-805360621),new cljs.core.Keyword(null,"_devcards_test_card_reporter","_devcards_test_card_reporter",-1561437805));
cljs.test.set_env_BANG_.call(null,test_env);

var tests = cljs.core.concat.call(null,test_thunks,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [((function (out,test_env){
return (function (){
cljs.core.async.put_BANG_.call(null,out,cljs.test.get_current_env.call(null));

return cljs.core.async.close_BANG_.call(null,out);
});})(out,test_env))
], null));
cljs.core.prn.call(null,"Running tests!!");

cljs.test.run_block.call(null,tests);

return out;
});
if(typeof devcards.core.test_loop !== 'undefined'){
} else {
devcards.core.test_loop = (function (){var c__18883__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18883__auto__){
return (function (){
var f__18884__auto__ = (function (){var switch__18862__auto__ = ((function (c__18883__auto__){
return (function (state_19974){
var state_val_19975 = (state_19974[(1)]);
if((state_val_19975 === (7))){
var state_19974__$1 = state_19974;
var statearr_19976_20025 = state_19974__$1;
(statearr_19976_20025[(2)] = false);

(statearr_19976_20025[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (20))){
var inst_19915 = (state_19974[(7)]);
var inst_19934 = cljs.core.apply.call(null,cljs.core.hash_map,inst_19915);
var state_19974__$1 = state_19974;
var statearr_19977_20026 = state_19974__$1;
(statearr_19977_20026[(2)] = inst_19934);

(statearr_19977_20026[(1)] = (22));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (27))){
var inst_19949 = (state_19974[(8)]);
var inst_19939 = (state_19974[(9)]);
var inst_19953 = inst_19939.call(null,inst_19949);
var state_19974__$1 = state_19974;
var statearr_19978_20027 = state_19974__$1;
(statearr_19978_20027[(2)] = inst_19953);

(statearr_19978_20027[(1)] = (29));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (1))){
var state_19974__$1 = state_19974;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19974__$1,(2),devcards.core.test_channel);
} else {
if((state_val_19975 === (24))){
var state_19974__$1 = state_19974;
var statearr_19979_20028 = state_19974__$1;
(statearr_19979_20028[(2)] = null);

(statearr_19979_20028[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (4))){
var state_19974__$1 = state_19974;
var statearr_19980_20029 = state_19974__$1;
(statearr_19980_20029[(2)] = false);

(statearr_19980_20029[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (15))){
var state_19974__$1 = state_19974;
var statearr_19981_20030 = state_19974__$1;
(statearr_19981_20030[(2)] = false);

(statearr_19981_20030[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (21))){
var inst_19915 = (state_19974[(7)]);
var state_19974__$1 = state_19974;
var statearr_19982_20031 = state_19974__$1;
(statearr_19982_20031[(2)] = inst_19915);

(statearr_19982_20031[(1)] = (22));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (13))){
var inst_19972 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_19974__$1,inst_19972);
} else {
if((state_val_19975 === (22))){
var inst_19938 = (state_19974[(10)]);
var inst_19937 = (state_19974[(2)]);
var inst_19938__$1 = cljs.core.get.call(null,inst_19937,new cljs.core.Keyword(null,"tests","tests",-1041085625));
var inst_19939 = cljs.core.get.call(null,inst_19937,new cljs.core.Keyword(null,"callback","callback",-705136228));
var state_19974__$1 = (function (){var statearr_19983 = state_19974;
(statearr_19983[(10)] = inst_19938__$1);

(statearr_19983[(9)] = inst_19939);

return statearr_19983;
})();
if(cljs.core.truth_(inst_19938__$1)){
var statearr_19984_20032 = state_19974__$1;
(statearr_19984_20032[(1)] = (23));

} else {
var statearr_19985_20033 = state_19974__$1;
(statearr_19985_20033[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (29))){
var inst_19963 = (state_19974[(2)]);
var inst_19964 = cljs.test.clear_env_BANG_.call(null);
var state_19974__$1 = (function (){var statearr_19986 = state_19974;
(statearr_19986[(11)] = inst_19964);

(statearr_19986[(12)] = inst_19963);

return statearr_19986;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_19974__$1,(30),devcards.core.test_channel);
} else {
if((state_val_19975 === (6))){
var state_19974__$1 = state_19974;
var statearr_19987_20034 = state_19974__$1;
(statearr_19987_20034[(2)] = true);

(statearr_19987_20034[(1)] = (8));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (28))){
var inst_19939 = (state_19974[(9)]);
var inst_19955 = [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"actual","actual",107306363)];
var inst_19956 = [new cljs.core.Keyword(null,"error","error",-978969032),"Tests timed out. Please check Dev Console for Exceptions"];
var inst_19957 = cljs.core.PersistentHashMap.fromArrays(inst_19955,inst_19956);
var inst_19958 = devcards.core.collect_test.call(null,inst_19957);
var inst_19959 = cljs.test.get_current_env.call(null);
var inst_19960 = cljs.core.assoc.call(null,inst_19959,new cljs.core.Keyword(null,"error","error",-978969032),"Execution timed out!");
var inst_19961 = inst_19939.call(null,inst_19960);
var state_19974__$1 = (function (){var statearr_19988 = state_19974;
(statearr_19988[(13)] = inst_19958);

return statearr_19988;
})();
var statearr_19989_20035 = state_19974__$1;
(statearr_19989_20035[(2)] = inst_19961);

(statearr_19989_20035[(1)] = (29));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (25))){
var inst_19970 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
var statearr_19990_20036 = state_19974__$1;
(statearr_19990_20036[(2)] = inst_19970);

(statearr_19990_20036[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (17))){
var state_19974__$1 = state_19974;
var statearr_19991_20037 = state_19974__$1;
(statearr_19991_20037[(2)] = true);

(statearr_19991_20037[(1)] = (19));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (3))){
var inst_19892 = (state_19974[(14)]);
var inst_19897 = inst_19892.cljs$lang$protocol_mask$partition0$;
var inst_19898 = (inst_19897 & (64));
var inst_19899 = inst_19892.cljs$core$ISeq$;
var inst_19900 = (inst_19898) || (inst_19899);
var state_19974__$1 = state_19974;
if(cljs.core.truth_(inst_19900)){
var statearr_19992_20038 = state_19974__$1;
(statearr_19992_20038[(1)] = (6));

} else {
var statearr_19993_20039 = state_19974__$1;
(statearr_19993_20039[(1)] = (7));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (12))){
var inst_19915 = (state_19974[(7)]);
var inst_19919 = (inst_19915 == null);
var inst_19920 = cljs.core.not.call(null,inst_19919);
var state_19974__$1 = state_19974;
if(inst_19920){
var statearr_19994_20040 = state_19974__$1;
(statearr_19994_20040[(1)] = (14));

} else {
var statearr_19995_20041 = state_19974__$1;
(statearr_19995_20041[(1)] = (15));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (2))){
var inst_19892 = (state_19974[(14)]);
var inst_19892__$1 = (state_19974[(2)]);
var inst_19894 = (inst_19892__$1 == null);
var inst_19895 = cljs.core.not.call(null,inst_19894);
var state_19974__$1 = (function (){var statearr_19996 = state_19974;
(statearr_19996[(14)] = inst_19892__$1);

return statearr_19996;
})();
if(inst_19895){
var statearr_19997_20042 = state_19974__$1;
(statearr_19997_20042[(1)] = (3));

} else {
var statearr_19998_20043 = state_19974__$1;
(statearr_19998_20043[(1)] = (4));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (23))){
var inst_19938 = (state_19974[(10)]);
var inst_19942 = (state_19974[(15)]);
var inst_19942__$1 = cljs.core.async.timeout.call(null,devcards.core.test_timeout);
var inst_19943 = cljs.core.PersistentVector.EMPTY_NODE;
var inst_19944 = devcards.core.run_card_tests.call(null,inst_19938);
var inst_19945 = [inst_19944,inst_19942__$1];
var inst_19946 = (new cljs.core.PersistentVector(null,2,(5),inst_19943,inst_19945,null));
var state_19974__$1 = (function (){var statearr_19999 = state_19974;
(statearr_19999[(15)] = inst_19942__$1);

return statearr_19999;
})();
return cljs.core.async.ioc_alts_BANG_.call(null,state_19974__$1,(26),inst_19946);
} else {
if((state_val_19975 === (19))){
var inst_19929 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
var statearr_20000_20044 = state_19974__$1;
(statearr_20000_20044[(2)] = inst_19929);

(statearr_20000_20044[(1)] = (16));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (11))){
var inst_19892 = (state_19974[(14)]);
var inst_19912 = (state_19974[(2)]);
var inst_19913 = cljs.core.get.call(null,inst_19912,new cljs.core.Keyword(null,"tests","tests",-1041085625));
var inst_19914 = cljs.core.get.call(null,inst_19912,new cljs.core.Keyword(null,"callback","callback",-705136228));
var inst_19915 = inst_19892;
var state_19974__$1 = (function (){var statearr_20001 = state_19974;
(statearr_20001[(16)] = inst_19913);

(statearr_20001[(17)] = inst_19914);

(statearr_20001[(7)] = inst_19915);

return statearr_20001;
})();
var statearr_20002_20045 = state_19974__$1;
(statearr_20002_20045[(2)] = null);

(statearr_20002_20045[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (9))){
var inst_19892 = (state_19974[(14)]);
var inst_19909 = cljs.core.apply.call(null,cljs.core.hash_map,inst_19892);
var state_19974__$1 = state_19974;
var statearr_20003_20046 = state_19974__$1;
(statearr_20003_20046[(2)] = inst_19909);

(statearr_20003_20046[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (5))){
var inst_19907 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
if(cljs.core.truth_(inst_19907)){
var statearr_20004_20047 = state_19974__$1;
(statearr_20004_20047[(1)] = (9));

} else {
var statearr_20005_20048 = state_19974__$1;
(statearr_20005_20048[(1)] = (10));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (14))){
var inst_19915 = (state_19974[(7)]);
var inst_19922 = inst_19915.cljs$lang$protocol_mask$partition0$;
var inst_19923 = (inst_19922 & (64));
var inst_19924 = inst_19915.cljs$core$ISeq$;
var inst_19925 = (inst_19923) || (inst_19924);
var state_19974__$1 = state_19974;
if(cljs.core.truth_(inst_19925)){
var statearr_20006_20049 = state_19974__$1;
(statearr_20006_20049[(1)] = (17));

} else {
var statearr_20007_20050 = state_19974__$1;
(statearr_20007_20050[(1)] = (18));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (26))){
var inst_19942 = (state_19974[(15)]);
var inst_19948 = (state_19974[(2)]);
var inst_19949 = cljs.core.nth.call(null,inst_19948,(0),null);
var inst_19950 = cljs.core.nth.call(null,inst_19948,(1),null);
var inst_19951 = cljs.core.not_EQ_.call(null,inst_19950,inst_19942);
var state_19974__$1 = (function (){var statearr_20008 = state_19974;
(statearr_20008[(8)] = inst_19949);

return statearr_20008;
})();
if(inst_19951){
var statearr_20009_20051 = state_19974__$1;
(statearr_20009_20051[(1)] = (27));

} else {
var statearr_20010_20052 = state_19974__$1;
(statearr_20010_20052[(1)] = (28));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (16))){
var inst_19932 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
if(cljs.core.truth_(inst_19932)){
var statearr_20011_20053 = state_19974__$1;
(statearr_20011_20053[(1)] = (20));

} else {
var statearr_20012_20054 = state_19974__$1;
(statearr_20012_20054[(1)] = (21));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (30))){
var inst_19966 = (state_19974[(2)]);
var inst_19915 = inst_19966;
var state_19974__$1 = (function (){var statearr_20013 = state_19974;
(statearr_20013[(7)] = inst_19915);

return statearr_20013;
})();
var statearr_20014_20055 = state_19974__$1;
(statearr_20014_20055[(2)] = null);

(statearr_20014_20055[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (10))){
var inst_19892 = (state_19974[(14)]);
var state_19974__$1 = state_19974;
var statearr_20015_20056 = state_19974__$1;
(statearr_20015_20056[(2)] = inst_19892);

(statearr_20015_20056[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (18))){
var state_19974__$1 = state_19974;
var statearr_20016_20057 = state_19974__$1;
(statearr_20016_20057[(2)] = false);

(statearr_20016_20057[(1)] = (19));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_19975 === (8))){
var inst_19904 = (state_19974[(2)]);
var state_19974__$1 = state_19974;
var statearr_20017_20058 = state_19974__$1;
(statearr_20017_20058[(2)] = inst_19904);

(statearr_20017_20058[(1)] = (5));


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
});})(c__18883__auto__))
;
return ((function (switch__18862__auto__,c__18883__auto__){
return (function() {
var devcards$core$state_machine__18863__auto__ = null;
var devcards$core$state_machine__18863__auto____0 = (function (){
var statearr_20021 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_20021[(0)] = devcards$core$state_machine__18863__auto__);

(statearr_20021[(1)] = (1));

return statearr_20021;
});
var devcards$core$state_machine__18863__auto____1 = (function (state_19974){
while(true){
var ret_value__18864__auto__ = (function (){try{while(true){
var result__18865__auto__ = switch__18862__auto__.call(null,state_19974);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18865__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18865__auto__;
}
break;
}
}catch (e20022){if((e20022 instanceof Object)){
var ex__18866__auto__ = e20022;
var statearr_20023_20059 = state_19974;
(statearr_20023_20059[(5)] = ex__18866__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_19974);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20022;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18864__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20060 = state_19974;
state_19974 = G__20060;
continue;
} else {
return ret_value__18864__auto__;
}
break;
}
});
devcards$core$state_machine__18863__auto__ = function(state_19974){
switch(arguments.length){
case 0:
return devcards$core$state_machine__18863__auto____0.call(this);
case 1:
return devcards$core$state_machine__18863__auto____1.call(this,state_19974);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
devcards$core$state_machine__18863__auto__.cljs$core$IFn$_invoke$arity$0 = devcards$core$state_machine__18863__auto____0;
devcards$core$state_machine__18863__auto__.cljs$core$IFn$_invoke$arity$1 = devcards$core$state_machine__18863__auto____1;
return devcards$core$state_machine__18863__auto__;
})()
;})(switch__18862__auto__,c__18883__auto__))
})();
var state__18885__auto__ = (function (){var statearr_20024 = f__18884__auto__.call(null);
(statearr_20024[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18883__auto__);

return statearr_20024;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18885__auto__);
});})(c__18883__auto__))
);

return c__18883__auto__;
})();
}
devcards.core.test_card_test_run = (function devcards$core$test_card_test_run(this$,tests){
return cljs.core.async.put_BANG_.call(null,devcards.core.test_channel,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tests","tests",-1041085625),tests,new cljs.core.Keyword(null,"callback","callback",-705136228),(function (results){
return this$.setState({"test_results": results});
})], null));
});
var base__19397__auto___20065 = {"componentWillMount": (function (){
var this$ = this;
var temp__4425__auto__ = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"test_thunks","test_thunks",304669805));
if(cljs.core.truth_(temp__4425__auto__)){
var test_thunks = temp__4425__auto__;
return devcards.core.test_card_test_run.call(null,this$,test_thunks);
} else {
return null;
}
}), "componentWillReceiveProps": (function (next_props){
var this$ = this;
var temp__4425__auto__ = (next_props[cljs.core.name.call(null,new cljs.core.Keyword(null,"test_thunks","test_thunks",304669805))]);
if(cljs.core.truth_(temp__4425__auto__)){
var test_thunks = temp__4425__auto__;
return devcards.core.test_card_test_run.call(null,this$,test_thunks);
} else {
return null;
}
}), "render": (function (){
var this$ = this;
var test_summary = devcards.core.get_state.call(null,this$,new cljs.core.Keyword(null,"test_results","test_results",1062111317));
var path = devcards.core.get_props.call(null,this$,new cljs.core.Keyword(null,"path","path",-188191168));
return devcards.core.render_tests.call(null,this$,path,test_summary);
})};
if(typeof devcards.core.TestDevcard !== 'undefined'){
} else {
devcards.core.TestDevcard = React.createClass(base__19397__auto___20065);
}

var seq__20061_20066 = cljs.core.seq.call(null,cljs.core.map.call(null,cljs.core.name,cljs.core.list(new cljs.core.Symbol("cljs-react-reload.core","shouldComponentUpdate","cljs-react-reload.core/shouldComponentUpdate",-526191550,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillReceiveProps","cljs-react-reload.core/componentWillReceiveProps",-1087108864,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillMount","cljs-react-reload.core/componentWillMount",-1529759893,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidMount","cljs-react-reload.core/componentDidMount",-2035273110,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUpdate","cljs-react-reload.core/componentWillUpdate",-453323386,null),new cljs.core.Symbol("cljs-react-reload.core","componentDidUpdate","cljs-react-reload.core/componentDidUpdate",-6660227,null),new cljs.core.Symbol("cljs-react-reload.core","componentWillUnmount","cljs-react-reload.core/componentWillUnmount",-1549767430,null),new cljs.core.Symbol("cljs-react-reload.core","render","cljs-react-reload.core/render",298414516,null))));
var chunk__20062_20067 = null;
var count__20063_20068 = (0);
var i__20064_20069 = (0);
while(true){
if((i__20064_20069 < count__20063_20068)){
var property__19398__auto___20070 = cljs.core._nth.call(null,chunk__20062_20067,i__20064_20069);
if(cljs.core.truth_((base__19397__auto___20065[property__19398__auto___20070]))){
(devcards.core.TestDevcard.prototype[property__19398__auto___20070] = (base__19397__auto___20065[property__19398__auto___20070]));
} else {
}

var G__20071 = seq__20061_20066;
var G__20072 = chunk__20062_20067;
var G__20073 = count__20063_20068;
var G__20074 = (i__20064_20069 + (1));
seq__20061_20066 = G__20071;
chunk__20062_20067 = G__20072;
count__20063_20068 = G__20073;
i__20064_20069 = G__20074;
continue;
} else {
var temp__4425__auto___20075 = cljs.core.seq.call(null,seq__20061_20066);
if(temp__4425__auto___20075){
var seq__20061_20076__$1 = temp__4425__auto___20075;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__20061_20076__$1)){
var c__17569__auto___20077 = cljs.core.chunk_first.call(null,seq__20061_20076__$1);
var G__20078 = cljs.core.chunk_rest.call(null,seq__20061_20076__$1);
var G__20079 = c__17569__auto___20077;
var G__20080 = cljs.core.count.call(null,c__17569__auto___20077);
var G__20081 = (0);
seq__20061_20066 = G__20078;
chunk__20062_20067 = G__20079;
count__20063_20068 = G__20080;
i__20064_20069 = G__20081;
continue;
} else {
var property__19398__auto___20082 = cljs.core.first.call(null,seq__20061_20076__$1);
if(cljs.core.truth_((base__19397__auto___20065[property__19398__auto___20082]))){
(devcards.core.TestDevcard.prototype[property__19398__auto___20082] = (base__19397__auto___20065[property__19398__auto___20082]));
} else {
}

var G__20083 = cljs.core.next.call(null,seq__20061_20076__$1);
var G__20084 = null;
var G__20085 = (0);
var G__20086 = (0);
seq__20061_20066 = G__20083;
chunk__20062_20067 = G__20084;
count__20063_20068 = G__20085;
i__20064_20069 = G__20086;
continue;
}
} else {
}
}
break;
}
devcards.core.test_card = (function devcards$core$test_card(var_args){
var args__17831__auto__ = [];
var len__17824__auto___20091 = arguments.length;
var i__17825__auto___20092 = (0);
while(true){
if((i__17825__auto___20092 < len__17824__auto___20091)){
args__17831__auto__.push((arguments[i__17825__auto___20092]));

var G__20093 = (i__17825__auto___20092 + (1));
i__17825__auto___20092 = G__20093;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return devcards.core.test_card.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

devcards.core.test_card.cljs$core$IFn$_invoke$arity$variadic = (function (test_thunks){
if(typeof devcards.core.t_devcards$core20088 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {devcards.core.IDevcard}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
devcards.core.t_devcards$core20088 = (function (test_thunks,meta20089){
this.test_thunks = test_thunks;
this.meta20089 = meta20089;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
devcards.core.t_devcards$core20088.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_20090,meta20089__$1){
var self__ = this;
var _20090__$1 = this;
return (new devcards.core.t_devcards$core20088(self__.test_thunks,meta20089__$1));
});

devcards.core.t_devcards$core20088.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_20090){
var self__ = this;
var _20090__$1 = this;
return self__.meta20089;
});

devcards.core.t_devcards$core20088.prototype.devcards$core$IDevcard$ = true;

devcards.core.t_devcards$core20088.prototype.devcards$core$IDevcard$_devcard$arity$2 = (function (this$,devcard_opts){
var self__ = this;
var this$__$1 = this;
var path = new cljs.core.Keyword(null,"path","path",-188191168).cljs$core$IFn$_invoke$arity$1(devcards.system._STAR_devcard_data_STAR_);
return React.createElement(devcards.core.TestDevcard,{"test_thunks": self__.test_thunks, "path": path});
});

devcards.core.t_devcards$core20088.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"test-thunks","test-thunks",2032684042,null),new cljs.core.Symbol(null,"meta20089","meta20089",911479202,null)], null);
});

devcards.core.t_devcards$core20088.cljs$lang$type = true;

devcards.core.t_devcards$core20088.cljs$lang$ctorStr = "devcards.core/t_devcards$core20088";

devcards.core.t_devcards$core20088.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"devcards.core/t_devcards$core20088");
});

devcards.core.__GT_t_devcards$core20088 = (function devcards$core$__GT_t_devcards$core20088(test_thunks__$1,meta20089){
return (new devcards.core.t_devcards$core20088(test_thunks__$1,meta20089));
});

}

return (new devcards.core.t_devcards$core20088(test_thunks,cljs.core.PersistentArrayMap.EMPTY));
});

devcards.core.test_card.cljs$lang$maxFixedArity = (0);

devcards.core.test_card.cljs$lang$applyTo = (function (seq20087){
return devcards.core.test_card.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq20087));
});
devcards.core.get_front_matter = (function devcards$core$get_front_matter(munged_namespace){
return cljs.core.reduce.call(null,cljs.core.aget,goog.global,cljs.core.concat.call(null,clojure.string.split.call(null,cljs.core.name.call(null,munged_namespace),"."),new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, ["front_matter"], null)));
});
devcards.core.get_cards_for_ns = (function devcards$core$get_cards_for_ns(ns_symbol){
var temp__4425__auto__ = new cljs.core.Keyword(null,"cards","cards",169174038).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,devcards.system.app_state));
if(cljs.core.truth_(temp__4425__auto__)){
var cards = temp__4425__auto__;
var temp__4425__auto____$1 = cljs.core.get_in.call(null,cards,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.keyword.call(null,ns_symbol)], null));
if(cljs.core.truth_(temp__4425__auto____$1)){
var card = temp__4425__auto____$1;
return card;
} else {
return null;
}
} else {
return null;
}
});
devcards.core.load_data_from_channel_BANG_ = (function devcards$core$load_data_from_channel_BANG_(){
return devcards.system.load_data_from_channel_BANG_.call(null,devcards.core.devcard_event_chan);
});
goog.exportSymbol('devcards.core.load_data_from_channel_BANG_', devcards.core.load_data_from_channel_BANG_);
devcards.core.merge_front_matter_options_BANG_ = (function devcards$core$merge_front_matter_options_BANG_(ns_symbol){
var temp__4425__auto__ = new cljs.core.Keyword(null,"base-card-options","base-card-options",141017756).cljs$core$IFn$_invoke$arity$1(devcards.core.get_front_matter.call(null,cljs.core.name.call(null,ns_symbol)));
if(cljs.core.truth_(temp__4425__auto__)){
var base_card_options = temp__4425__auto__;
cljs.core.println.call(null,"Adding base card options!",cljs.core.prn_str.call(null,base_card_options));

return cljs.core.swap_BANG_.call(null,devcards.system.app_state,cljs.core.update_in,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"base-card-options","base-card-options",141017756)], null),((function (base_card_options,temp__4425__auto__){
return (function (opts){
return cljs.core.merge.call(null,opts,base_card_options);
});})(base_card_options,temp__4425__auto__))
);
} else {
return null;
}
});
goog.exportSymbol('devcards.core.merge_front_matter_options_BANG_', devcards.core.merge_front_matter_options_BANG_);
devcards.core.render_namespace_to_string = (function devcards$core$render_namespace_to_string(ns_symbol){
var temp__4425__auto__ = devcards.core.get_cards_for_ns.call(null,ns_symbol);
if(cljs.core.truth_(temp__4425__auto__)){
var card = temp__4425__auto__;
devcards.core.merge_front_matter_options_BANG_.call(null,ns_symbol);

return [cljs.core.str("<div id=\"com-rigsomelight-devcards-main\">"),cljs.core.str(React.renderToString((function (){var attrs20095 = devcards.system.render_cards.call(null,devcards.system.display_cards.call(null,card),devcards.system.app_state);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs20095))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["com-rigsomelight-devcards-base",null,"com-rigsomelight-devcards-string-render",null], null), null)], null),attrs20095)):{"className": "com-rigsomelight-devcards-base com-rigsomelight-devcards-string-render"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs20095))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs20095)], null))));
})())),cljs.core.str("</div>")].join('');
} else {
return null;
}
});
goog.exportSymbol('devcards.core.render_namespace_to_string', devcards.core.render_namespace_to_string);
devcards.core.render_ns = (function devcards$core$render_ns(ns_symbol,app_state){
var temp__4425__auto__ = devcards.core.get_cards_for_ns.call(null,ns_symbol);
if(cljs.core.truth_(temp__4425__auto__)){
var card = temp__4425__auto__;
return React.render((function (){var attrs20097 = devcards.system.render_cards.call(null,devcards.system.display_cards.call(null,card),app_state);
return cljs.core.apply.call(null,React.createElement,"div",((cljs.core.map_QMARK_.call(null,attrs20097))?sablono.interpreter.attributes.call(null,sablono.normalize.merge_with_class.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"class","class",-2030961996),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["com-rigsomelight-devcards-base",null,"com-rigsomelight-devcards-string-render",null], null), null)], null),attrs20097)):{"className": "com-rigsomelight-devcards-base com-rigsomelight-devcards-string-render"}),cljs.core.remove.call(null,cljs.core.nil_QMARK_,((cljs.core.map_QMARK_.call(null,attrs20097))?cljs.core.PersistentVector.EMPTY:new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [sablono.interpreter.interpret.call(null,attrs20097)], null))));
})(),devcards.system.devcards_app_node.call(null));
} else {
return null;
}
});
devcards.core.mount_namespace = (function devcards$core$mount_namespace(ns_symbol){
devcards.core.merge_front_matter_options_BANG_.call(null,ns_symbol);

var c__18883__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18883__auto__){
return (function (){
var f__18884__auto__ = (function (){var switch__18862__auto__ = ((function (c__18883__auto__){
return (function (state_20127){
var state_val_20128 = (state_20127[(1)]);
if((state_val_20128 === (1))){
var inst_20118 = devcards.core.load_data_from_channel_BANG_.call(null);
var state_20127__$1 = state_20127;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20127__$1,(2),inst_20118);
} else {
if((state_val_20128 === (2))){
var inst_20120 = (state_20127[(2)]);
var inst_20121 = cljs.core.async.timeout.call(null,(100));
var state_20127__$1 = (function (){var statearr_20129 = state_20127;
(statearr_20129[(7)] = inst_20120);

return statearr_20129;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_20127__$1,(3),inst_20121);
} else {
if((state_val_20128 === (3))){
var inst_20123 = (state_20127[(2)]);
var inst_20124 = (function (){return ((function (inst_20123,state_val_20128,c__18883__auto__){
return (function (){
return devcards.core.render_ns.call(null,ns_symbol,devcards.system.app_state);
});
;})(inst_20123,state_val_20128,c__18883__auto__))
})();
var inst_20125 = setTimeout(inst_20124,(0));
var state_20127__$1 = (function (){var statearr_20130 = state_20127;
(statearr_20130[(8)] = inst_20123);

return statearr_20130;
})();
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_20127__$1,inst_20125);
} else {
return null;
}
}
}
});})(c__18883__auto__))
;
return ((function (switch__18862__auto__,c__18883__auto__){
return (function() {
var devcards$core$mount_namespace_$_state_machine__18863__auto__ = null;
var devcards$core$mount_namespace_$_state_machine__18863__auto____0 = (function (){
var statearr_20134 = [null,null,null,null,null,null,null,null,null];
(statearr_20134[(0)] = devcards$core$mount_namespace_$_state_machine__18863__auto__);

(statearr_20134[(1)] = (1));

return statearr_20134;
});
var devcards$core$mount_namespace_$_state_machine__18863__auto____1 = (function (state_20127){
while(true){
var ret_value__18864__auto__ = (function (){try{while(true){
var result__18865__auto__ = switch__18862__auto__.call(null,state_20127);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18865__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18865__auto__;
}
break;
}
}catch (e20135){if((e20135 instanceof Object)){
var ex__18866__auto__ = e20135;
var statearr_20136_20138 = state_20127;
(statearr_20136_20138[(5)] = ex__18866__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_20127);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e20135;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18864__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__20139 = state_20127;
state_20127 = G__20139;
continue;
} else {
return ret_value__18864__auto__;
}
break;
}
});
devcards$core$mount_namespace_$_state_machine__18863__auto__ = function(state_20127){
switch(arguments.length){
case 0:
return devcards$core$mount_namespace_$_state_machine__18863__auto____0.call(this);
case 1:
return devcards$core$mount_namespace_$_state_machine__18863__auto____1.call(this,state_20127);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
devcards$core$mount_namespace_$_state_machine__18863__auto__.cljs$core$IFn$_invoke$arity$0 = devcards$core$mount_namespace_$_state_machine__18863__auto____0;
devcards$core$mount_namespace_$_state_machine__18863__auto__.cljs$core$IFn$_invoke$arity$1 = devcards$core$mount_namespace_$_state_machine__18863__auto____1;
return devcards$core$mount_namespace_$_state_machine__18863__auto__;
})()
;})(switch__18862__auto__,c__18883__auto__))
})();
var state__18885__auto__ = (function (){var statearr_20137 = f__18884__auto__.call(null);
(statearr_20137[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18883__auto__);

return statearr_20137;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18885__auto__);
});})(c__18883__auto__))
);

return c__18883__auto__;
});
goog.exportSymbol('devcards.core.mount_namespace', devcards.core.mount_namespace);
devcards.core.mount_namespace_live = (function devcards$core$mount_namespace_live(ns_symbol){
devcards.core.merge_front_matter_options_BANG_.call(null,ns_symbol);

return devcards.system.start_ui_with_renderer.call(null,devcards.core.devcard_event_chan,cljs.core.partial.call(null,devcards.core.render_ns,ns_symbol));
});
goog.exportSymbol('devcards.core.mount_namespace_live', devcards.core.mount_namespace_live);

//# sourceMappingURL=core.js.map?rel=1454621603891