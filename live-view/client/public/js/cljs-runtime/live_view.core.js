goog.provide('live_view.core');
var module$node_modules$morphdom$dist$morphdom=shadow.js.require("module$node_modules$morphdom$dist$morphdom", {});
live_view.core.apply_morphdom_patch = (function live_view$core$apply_morphdom_patch(node,current_hiccup,patch){
var new_hiccup = editscript.core.patch(current_hiccup,editscript.edit.edits__GT_script(patch));
module$node_modules$morphdom$dist$morphdom(node,crate.core.html.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new_hiccup], 0)),({"onBeforeElUpdated": (function (from,to){
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(to.tagName,"INPUT")) && (cljs.core.not(to.attributes.value)))){
return false;
} else {
return true;

}
})}));

return new_hiccup;
});
live_view.core.create_renderer = (function live_view$core$create_renderer(dom_node){
var dom_node__$1 = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(dom_node);
var virtual_dom = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(null);
return (function (data){
var current_vdom = cljs.core.deref(virtual_dom);
var G__48681 = new cljs.core.Keyword(null,"type","type",1174270348).cljs$core$IFn$_invoke$arity$1(data);
var G__48681__$1 = (((G__48681 instanceof cljs.core.Keyword))?G__48681.fqn:null);
switch (G__48681__$1) {
case "patch":
if(cljs.core.truth_(current_vdom)){
return cljs.core.reset_BANG_(virtual_dom,live_view.core.apply_morphdom_patch(cljs.core.deref(dom_node__$1),current_vdom,new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data)));
} else {
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"error","error",-978969032),new cljs.core.Keyword(null,"reason","reason",-2070751759),new cljs.core.Keyword(null,"no-state","no-state",-1096309128)], null);
}

break;
case "init":
cljs.core.reset_BANG_(virtual_dom,new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data));

var node = crate.core.html.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data)], 0));
cljs.core.deref(dom_node__$1).replaceWith(node);

return cljs.core.reset_BANG_(dom_node__$1,node);

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__48681__$1)].join('')));

}
});
});
live_view.core.init = (function live_view$core$init(){
var renderer = live_view.core.create_renderer(document.body);
var ws = (new WebSocket("ws://localhost:50505/loc/"));
(ws.onopen = (function (){
return ws.send("init");
}));

return (ws.onmessage = (function (e){
var reader = cognitect.transit.reader.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"json","json",1279968570));
var payload = cognitect.transit.read(reader,e.data);
cljs.core.prn.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([payload], 0));

return renderer(payload);
}));
});

//# sourceMappingURL=live_view.core.js.map
