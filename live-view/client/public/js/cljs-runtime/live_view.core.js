goog.provide('live_view.core');
var module$node_modules$morphdom$dist$morphdom=shadow.js.require("module$node_modules$morphdom$dist$morphdom", {});
live_view.core.hipo_options = (function live_view$core$hipo_options(ws){
return new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attribute-handlers","attribute-handlers",855454691),new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attr","attr",-604132353),"onchange"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (node,a,b,p__43099){
var vec__43100 = p__43099;
var action = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43100,(0),null);
var payload = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43100,(1),null);
return node.addEventListener("input",(function (e){
var writer = cognitect.transit.writer.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"json","json",1279968570));
return ws.send(cognitect.transit.write(writer,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [action,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3((function (){var or__4126__auto__ = payload;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return cljs.core.PersistentArrayMap.EMPTY;
}
})(),new cljs.core.Keyword(null,"value","value",305978217),e.target.value)], null)));
}));
})], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attr","attr",-604132353),"onsubmit"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (node,a,b,val){
return node.addEventListener("submit",(function (e){
e.preventDefault();

var writer = cognitect.transit.writer.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"json","json",1279968570));
return ws.send(cognitect.transit.write(writer,val));
}));
})], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attr","attr",-604132353),"onclick"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (node,a,b,val){
return node.addEventListener("click",(function (e){
e.preventDefault();

var writer = cognitect.transit.writer.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"json","json",1279968570));
return ws.send(cognitect.transit.write(writer,val));
}));
})], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attr","attr",-604132353),"style"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (node,_,___$1,styles){
var seq__43103 = cljs.core.seq(styles);
var chunk__43104 = null;
var count__43105 = (0);
var i__43106 = (0);
while(true){
if((i__43106 < count__43105)){
var vec__43113 = chunk__43104.cljs$core$IIndexed$_nth$arity$2(null,i__43106);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43113,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43113,(1),null);
goog.object.set(node.style,cljs.core.name(k),v);


var G__43121 = seq__43103;
var G__43122 = chunk__43104;
var G__43123 = count__43105;
var G__43124 = (i__43106 + (1));
seq__43103 = G__43121;
chunk__43104 = G__43122;
count__43105 = G__43123;
i__43106 = G__43124;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__43103);
if(temp__5735__auto__){
var seq__43103__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__43103__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__43103__$1);
var G__43125 = cljs.core.chunk_rest(seq__43103__$1);
var G__43126 = c__4556__auto__;
var G__43127 = cljs.core.count(c__4556__auto__);
var G__43128 = (0);
seq__43103 = G__43125;
chunk__43104 = G__43126;
count__43105 = G__43127;
i__43106 = G__43128;
continue;
} else {
var vec__43116 = cljs.core.first(seq__43103__$1);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43116,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__43116,(1),null);
goog.object.set(node.style,cljs.core.name(k),v);


var G__43129 = cljs.core.next(seq__43103__$1);
var G__43130 = null;
var G__43131 = (0);
var G__43132 = (0);
seq__43103 = G__43129;
chunk__43104 = G__43130;
count__43105 = G__43131;
i__43106 = G__43132;
continue;
}
} else {
return null;
}
}
break;
}
})], null)], null)], null);
});
live_view.core.apply_patch = (function live_view$core$apply_patch(node,current_hiccup,patch){
var new_hiccup = editscript.core.patch(current_hiccup,editscript.edit.edits__GT_script(patch));
hipo.core.reconciliate_BANG_.cljs$core$IFn$_invoke$arity$2(node,new_hiccup);

return new_hiccup;
});
live_view.core.create_renderer = (function live_view$core$create_renderer(dom_node,ws){
var dom_node__$1 = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(dom_node);
var virtual_dom = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(null);
return (function (data){
var current_vdom = cljs.core.deref(virtual_dom);
var G__43119 = new cljs.core.Keyword(null,"type","type",1174270348).cljs$core$IFn$_invoke$arity$1(data);
var G__43119__$1 = (((G__43119 instanceof cljs.core.Keyword))?G__43119.fqn:null);
switch (G__43119__$1) {
case "patch":
if(cljs.core.truth_(current_vdom)){
return cljs.core.reset_BANG_(virtual_dom,live_view.core.apply_patch(cljs.core.deref(dom_node__$1),current_vdom,new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data)));
} else {
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"error","error",-978969032),new cljs.core.Keyword(null,"reason","reason",-2070751759),new cljs.core.Keyword(null,"no-state","no-state",-1096309128)], null);
}

break;
case "init":
cljs.core.reset_BANG_(virtual_dom,new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data));

var node = (function (){var v43120 = new cljs.core.Keyword(null,"value","value",305978217).cljs$core$IFn$_invoke$arity$1(data);
var el__37721__auto__ = hipo.interpreter.create(v43120,live_view.core.hipo_options(ws));
hipo.core.set_hiccup_BANG_(el__37721__auto__,v43120);

return el__37721__auto__;
})();
cljs.core.deref(dom_node__$1).replaceWith(node);

return cljs.core.reset_BANG_(dom_node__$1,node);

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__43119__$1)].join('')));

}
});
});
live_view.core.init = (function live_view$core$init(){
var ws = (new WebSocket("ws://localhost:50505/loc/"));
var renderer = live_view.core.create_renderer(document.body,ws);
(ws.onopen = (function (){
return ws.send("init");
}));

return (ws.onmessage = (function (e){
var reader = cognitect.transit.reader.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"json","json",1279968570));
var payload = cognitect.transit.read(reader,e.data);
return renderer(payload);
}));
});

//# sourceMappingURL=live_view.core.js.map
