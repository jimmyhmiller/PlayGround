goog.provide('crate.compiler');
crate.compiler.xmlns = new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"xhtml","xhtml",1912943770),"http://www.w3.org/1999/xhtml",new cljs.core.Keyword(null,"svg","svg",856789142),"http://www.w3.org/2000/svg"], null);


crate.compiler.group_id = cljs.core.atom.cljs$core$IFn$_invoke$arity$1((0));
crate.compiler.bindings = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentVector.EMPTY);
crate.compiler.capture_binding = (function crate$compiler$capture_binding(tag,b){
return cljs.core.swap_BANG_.cljs$core$IFn$_invoke$arity$3(crate.compiler.bindings,cljs.core.conj,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [tag,b], null));
});

/**
 * @interface
 */
crate.compiler.Element = function(){};

var crate$compiler$Element$_elem$dyn_47220 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (crate.compiler._elem[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (crate.compiler._elem["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("Element.-elem",this$);
}
}
});
crate.compiler._elem = (function crate$compiler$_elem(this$){
if((((!((this$ == null)))) && ((!((this$.crate$compiler$Element$_elem$arity$1 == null)))))){
return this$.crate$compiler$Element$_elem$arity$1(this$);
} else {
return crate$compiler$Element$_elem$dyn_47220(this$);
}
});

crate.compiler.as_content = (function crate$compiler$as_content(parent,content){
var seq__46925 = cljs.core.seq(content);
var chunk__46926 = null;
var count__46927 = (0);
var i__46928 = (0);
while(true){
if((i__46928 < count__46927)){
var c = chunk__46926.cljs$core$IIndexed$_nth$arity$2(null,i__46928);
var child_47227 = (((((!((c == null))))?((((false) || ((cljs.core.PROTOCOL_SENTINEL === c.crate$compiler$Element$))))?true:(((!c.cljs$lang$protocol_mask$partition$))?cljs.core.native_satisfies_QMARK_(crate.compiler.Element,c):false)):cljs.core.native_satisfies_QMARK_(crate.compiler.Element,c)))?crate.compiler._elem(c):(((c == null))?null:((cljs.core.map_QMARK_(c))?(function(){throw "Maps cannot be used as content"})():((typeof c === 'string')?goog.dom.createTextNode(c):((cljs.core.vector_QMARK_(c))?(crate.compiler.elem_factory.cljs$core$IFn$_invoke$arity$1 ? crate.compiler.elem_factory.cljs$core$IFn$_invoke$arity$1(c) : crate.compiler.elem_factory.call(null,c)):((cljs.core.seq_QMARK_(c))?(crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(parent,c) : crate.compiler.as_content.call(null,parent,c)):((crate.binding.binding_coll_QMARK_(c))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"coll","coll",1647737163),c);

var G__46991 = parent;
var G__46992 = new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [crate.binding.value(c)], null);
return (crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(G__46991,G__46992) : crate.compiler.as_content.call(null,G__46991,G__46992));
})()
:((crate.binding.binding_QMARK_(c))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"text","text",-1790561697),c);

var G__46993 = parent;
var G__46994 = new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [crate.binding.value(c)], null);
return (crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(G__46993,G__46994) : crate.compiler.as_content.call(null,G__46993,G__46994));
})()
:(cljs.core.truth_(c.nodeName)?c:(cljs.core.truth_(c.get)?c.get((0)):goog.dom.createTextNode(cljs.core.str.cljs$core$IFn$_invoke$arity$1(c))
))))))))));
if(cljs.core.truth_(child_47227)){
goog.dom.appendChild(parent,child_47227);
} else {
}


var G__47231 = seq__46925;
var G__47232 = chunk__46926;
var G__47233 = count__46927;
var G__47234 = (i__46928 + (1));
seq__46925 = G__47231;
chunk__46926 = G__47232;
count__46927 = G__47233;
i__46928 = G__47234;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46925);
if(temp__5735__auto__){
var seq__46925__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46925__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46925__$1);
var G__47235 = cljs.core.chunk_rest(seq__46925__$1);
var G__47236 = c__4556__auto__;
var G__47237 = cljs.core.count(c__4556__auto__);
var G__47238 = (0);
seq__46925 = G__47235;
chunk__46926 = G__47236;
count__46927 = G__47237;
i__46928 = G__47238;
continue;
} else {
var c = cljs.core.first(seq__46925__$1);
var child_47240 = (((((!((c == null))))?((((false) || ((cljs.core.PROTOCOL_SENTINEL === c.crate$compiler$Element$))))?true:(((!c.cljs$lang$protocol_mask$partition$))?cljs.core.native_satisfies_QMARK_(crate.compiler.Element,c):false)):cljs.core.native_satisfies_QMARK_(crate.compiler.Element,c)))?crate.compiler._elem(c):(((c == null))?null:((cljs.core.map_QMARK_(c))?(function(){throw "Maps cannot be used as content"})():((typeof c === 'string')?goog.dom.createTextNode(c):((cljs.core.vector_QMARK_(c))?(crate.compiler.elem_factory.cljs$core$IFn$_invoke$arity$1 ? crate.compiler.elem_factory.cljs$core$IFn$_invoke$arity$1(c) : crate.compiler.elem_factory.call(null,c)):((cljs.core.seq_QMARK_(c))?(crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(parent,c) : crate.compiler.as_content.call(null,parent,c)):((crate.binding.binding_coll_QMARK_(c))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"coll","coll",1647737163),c);

var G__47019 = parent;
var G__47020 = new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [crate.binding.value(c)], null);
return (crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(G__47019,G__47020) : crate.compiler.as_content.call(null,G__47019,G__47020));
})()
:((crate.binding.binding_QMARK_(c))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"text","text",-1790561697),c);

var G__47021 = parent;
var G__47022 = new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [crate.binding.value(c)], null);
return (crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.as_content.cljs$core$IFn$_invoke$arity$2(G__47021,G__47022) : crate.compiler.as_content.call(null,G__47021,G__47022));
})()
:(cljs.core.truth_(c.nodeName)?c:(cljs.core.truth_(c.get)?c.get((0)):goog.dom.createTextNode(cljs.core.str.cljs$core$IFn$_invoke$arity$1(c))
))))))))));
if(cljs.core.truth_(child_47240)){
goog.dom.appendChild(parent,child_47240);
} else {
}


var G__47242 = cljs.core.next(seq__46925__$1);
var G__47243 = null;
var G__47244 = (0);
var G__47245 = (0);
seq__46925 = G__47242;
chunk__46926 = G__47243;
count__46927 = G__47244;
i__46928 = G__47245;
continue;
}
} else {
return null;
}
}
break;
}
});
if((typeof crate !== 'undefined') && (typeof crate.compiler !== 'undefined') && (typeof crate.compiler.dom_binding !== 'undefined')){
} else {
crate.compiler.dom_binding = (function (){var method_table__4619__auto__ = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var prefer_table__4620__auto__ = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var method_cache__4621__auto__ = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var cached_hierarchy__4622__auto__ = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentArrayMap.EMPTY);
var hierarchy__4623__auto__ = cljs.core.get.cljs$core$IFn$_invoke$arity$3(cljs.core.PersistentArrayMap.EMPTY,new cljs.core.Keyword(null,"hierarchy","hierarchy",-1053470341),(function (){var fexpr__47030 = cljs.core.get_global_hierarchy;
return (fexpr__47030.cljs$core$IFn$_invoke$arity$0 ? fexpr__47030.cljs$core$IFn$_invoke$arity$0() : fexpr__47030.call(null));
})());
return (new cljs.core.MultiFn(cljs.core.symbol.cljs$core$IFn$_invoke$arity$2("crate.compiler","dom-binding"),(function (type,_,___$1){
return type;
}),new cljs.core.Keyword(null,"default","default",-1987822328),hierarchy__4623__auto__,method_table__4619__auto__,prefer_table__4620__auto__,method_cache__4621__auto__,cached_hierarchy__4622__auto__));
})();
}
crate.compiler.dom_binding.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"text","text",-1790561697),(function (_,b,elem){
return crate.binding.on_change(b,(function (v){
goog.dom.removeChildren(elem);

return crate.compiler.as_content(elem,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [v], null));
}));
}));
crate.compiler.dom_binding.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"attr","attr",-604132353),(function (_,p__47031,elem){
var vec__47032 = p__47031;
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47032,(0),null);
var b = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47032,(1),null);
return crate.binding.on_change(b,(function (v){
return (crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3 ? crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3(elem,k,v) : crate.compiler.dom_attr.call(null,elem,k,v));
}));
}));
crate.compiler.dom_binding.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"style","style",-496642736),(function (_,p__47041,elem){
var vec__47042 = p__47041;
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47042,(0),null);
var b = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47042,(1),null);
return crate.binding.on_change(b,(function (v){
if(cljs.core.truth_(k)){
return (crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3 ? crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3(elem,k,v) : crate.compiler.dom_style.call(null,elem,k,v));
} else {
return (crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2 ? crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2(elem,v) : crate.compiler.dom_style.call(null,elem,v));
}
}));
}));
crate.compiler.dom_add = (function crate$compiler$dom_add(bc,parent,elem,v){
var temp__5733__auto__ = crate.binding.opt(bc,new cljs.core.Keyword(null,"add","add",235287739));
if(cljs.core.truth_(temp__5733__auto__)){
var adder = temp__5733__auto__;
return (adder.cljs$core$IFn$_invoke$arity$3 ? adder.cljs$core$IFn$_invoke$arity$3(parent,elem,v) : adder.call(null,parent,elem,v));
} else {
return goog.dom.appendChild(parent,elem);
}
});
crate.compiler.dom_remove = (function crate$compiler$dom_remove(bc,elem){
var temp__5733__auto__ = crate.binding.opt(bc,new cljs.core.Keyword(null,"remove","remove",-131428414));
if(cljs.core.truth_(temp__5733__auto__)){
var remover = temp__5733__auto__;
return (remover.cljs$core$IFn$_invoke$arity$1 ? remover.cljs$core$IFn$_invoke$arity$1(elem) : remover.call(null,elem));
} else {
return goog.dom.removeNode(elem);
}
});
crate.compiler.dom_binding.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"coll","coll",1647737163),(function (_,bc,parent){
return crate.binding.on_change(bc,(function (type,elem,v){
var pred__47061 = cljs.core._EQ_;
var expr__47062 = type;
if(cljs.core.truth_((pred__47061.cljs$core$IFn$_invoke$arity$2 ? pred__47061.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"add","add",235287739),expr__47062) : pred__47061.call(null,new cljs.core.Keyword(null,"add","add",235287739),expr__47062)))){
return crate.compiler.dom_add(bc,parent,elem,v);
} else {
if(cljs.core.truth_((pred__47061.cljs$core$IFn$_invoke$arity$2 ? pred__47061.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"remove","remove",-131428414),expr__47062) : pred__47061.call(null,new cljs.core.Keyword(null,"remove","remove",-131428414),expr__47062)))){
return crate.compiler.dom_remove(bc,elem);
} else {
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(expr__47062)].join('')));
}
}
}));
}));
crate.compiler.handle_bindings = (function crate$compiler$handle_bindings(bs,elem){
var seq__47067 = cljs.core.seq(bs);
var chunk__47068 = null;
var count__47069 = (0);
var i__47070 = (0);
while(true){
if((i__47070 < count__47069)){
var vec__47080 = chunk__47068.cljs$core$IIndexed$_nth$arity$2(null,i__47070);
var type = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47080,(0),null);
var b = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47080,(1),null);
crate.compiler.dom_binding.cljs$core$IFn$_invoke$arity$3(type,b,elem);


var G__47249 = seq__47067;
var G__47250 = chunk__47068;
var G__47251 = count__47069;
var G__47252 = (i__47070 + (1));
seq__47067 = G__47249;
chunk__47068 = G__47250;
count__47069 = G__47251;
i__47070 = G__47252;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__47067);
if(temp__5735__auto__){
var seq__47067__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__47067__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__47067__$1);
var G__47253 = cljs.core.chunk_rest(seq__47067__$1);
var G__47254 = c__4556__auto__;
var G__47255 = cljs.core.count(c__4556__auto__);
var G__47256 = (0);
seq__47067 = G__47253;
chunk__47068 = G__47254;
count__47069 = G__47255;
i__47070 = G__47256;
continue;
} else {
var vec__47084 = cljs.core.first(seq__47067__$1);
var type = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47084,(0),null);
var b = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47084,(1),null);
crate.compiler.dom_binding.cljs$core$IFn$_invoke$arity$3(type,b,elem);


var G__47257 = cljs.core.next(seq__47067__$1);
var G__47258 = null;
var G__47259 = (0);
var G__47260 = (0);
seq__47067 = G__47257;
chunk__47068 = G__47258;
count__47069 = G__47259;
i__47070 = G__47260;
continue;
}
} else {
return null;
}
}
break;
}
});
crate.compiler.dom_style = (function crate$compiler$dom_style(var_args){
var G__47088 = arguments.length;
switch (G__47088) {
case 2:
return crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2 = (function (elem,v){
if(typeof v === 'string'){
elem.setAttribute("style",v);
} else {
if(cljs.core.map_QMARK_(v)){
var seq__47092_47265 = cljs.core.seq(v);
var chunk__47093_47266 = null;
var count__47094_47267 = (0);
var i__47095_47268 = (0);
while(true){
if((i__47095_47268 < count__47094_47267)){
var vec__47106_47270 = chunk__47093_47266.cljs$core$IIndexed$_nth$arity$2(null,i__47095_47268);
var k_47271 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47106_47270,(0),null);
var v_47272__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47106_47270,(1),null);
crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3(elem,k_47271,v_47272__$1);


var G__47273 = seq__47092_47265;
var G__47274 = chunk__47093_47266;
var G__47275 = count__47094_47267;
var G__47276 = (i__47095_47268 + (1));
seq__47092_47265 = G__47273;
chunk__47093_47266 = G__47274;
count__47094_47267 = G__47275;
i__47095_47268 = G__47276;
continue;
} else {
var temp__5735__auto___47277 = cljs.core.seq(seq__47092_47265);
if(temp__5735__auto___47277){
var seq__47092_47278__$1 = temp__5735__auto___47277;
if(cljs.core.chunked_seq_QMARK_(seq__47092_47278__$1)){
var c__4556__auto___47279 = cljs.core.chunk_first(seq__47092_47278__$1);
var G__47280 = cljs.core.chunk_rest(seq__47092_47278__$1);
var G__47281 = c__4556__auto___47279;
var G__47282 = cljs.core.count(c__4556__auto___47279);
var G__47283 = (0);
seq__47092_47265 = G__47280;
chunk__47093_47266 = G__47281;
count__47094_47267 = G__47282;
i__47095_47268 = G__47283;
continue;
} else {
var vec__47114_47285 = cljs.core.first(seq__47092_47278__$1);
var k_47286 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47114_47285,(0),null);
var v_47287__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47114_47285,(1),null);
crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3(elem,k_47286,v_47287__$1);


var G__47288 = cljs.core.next(seq__47092_47278__$1);
var G__47289 = null;
var G__47290 = (0);
var G__47291 = (0);
seq__47092_47265 = G__47288;
chunk__47093_47266 = G__47289;
count__47094_47267 = G__47290;
i__47095_47268 = G__47291;
continue;
}
} else {
}
}
break;
}
} else {
if(crate.binding.binding_QMARK_(v)){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"style","style",-496642736),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [null,v], null));

crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2(elem,crate.binding.value(v));
} else {
}
}
}

return elem;
}));

(crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$3 = (function (elem,k,v){
var v__$1 = ((crate.binding.binding_QMARK_(v))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"style","style",-496642736),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [k,v], null));

return crate.binding.value(v);
})()
:v);
return goog.style.setStyle(elem,cljs.core.name(k),v__$1);
}));

(crate.compiler.dom_style.cljs$lang$maxFixedArity = 3);

crate.compiler.dom_attr = (function crate$compiler$dom_attr(var_args){
var G__47147 = arguments.length;
switch (G__47147) {
case 2:
return crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$2 = (function (elem,attrs){
if(cljs.core.truth_(elem)){
if((!(cljs.core.map_QMARK_(attrs)))){
return elem.getAttribute(cljs.core.name(attrs));
} else {
var seq__47151_47295 = cljs.core.seq(attrs);
var chunk__47152_47296 = null;
var count__47153_47297 = (0);
var i__47154_47298 = (0);
while(true){
if((i__47154_47298 < count__47153_47297)){
var vec__47164_47299 = chunk__47152_47296.cljs$core$IIndexed$_nth$arity$2(null,i__47154_47298);
var k_47300 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47164_47299,(0),null);
var v_47301 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47164_47299,(1),null);
crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3(elem,k_47300,v_47301);


var G__47302 = seq__47151_47295;
var G__47303 = chunk__47152_47296;
var G__47304 = count__47153_47297;
var G__47305 = (i__47154_47298 + (1));
seq__47151_47295 = G__47302;
chunk__47152_47296 = G__47303;
count__47153_47297 = G__47304;
i__47154_47298 = G__47305;
continue;
} else {
var temp__5735__auto___47306 = cljs.core.seq(seq__47151_47295);
if(temp__5735__auto___47306){
var seq__47151_47307__$1 = temp__5735__auto___47306;
if(cljs.core.chunked_seq_QMARK_(seq__47151_47307__$1)){
var c__4556__auto___47308 = cljs.core.chunk_first(seq__47151_47307__$1);
var G__47309 = cljs.core.chunk_rest(seq__47151_47307__$1);
var G__47310 = c__4556__auto___47308;
var G__47311 = cljs.core.count(c__4556__auto___47308);
var G__47312 = (0);
seq__47151_47295 = G__47309;
chunk__47152_47296 = G__47310;
count__47153_47297 = G__47311;
i__47154_47298 = G__47312;
continue;
} else {
var vec__47172_47313 = cljs.core.first(seq__47151_47307__$1);
var k_47314 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47172_47313,(0),null);
var v_47315 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47172_47313,(1),null);
crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3(elem,k_47314,v_47315);


var G__47316 = cljs.core.next(seq__47151_47307__$1);
var G__47317 = null;
var G__47318 = (0);
var G__47319 = (0);
seq__47151_47295 = G__47316;
chunk__47152_47296 = G__47317;
count__47153_47297 = G__47318;
i__47154_47298 = G__47319;
continue;
}
} else {
}
}
break;
}

return elem;
}
} else {
return null;
}
}));

(crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$3 = (function (elem,k,v){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(k,new cljs.core.Keyword(null,"style","style",-496642736))){
crate.compiler.dom_style.cljs$core$IFn$_invoke$arity$2(elem,v);
} else {
var v_47320__$1 = ((crate.binding.binding_QMARK_(v))?(function (){
crate.compiler.capture_binding(new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [k,v], null));

return crate.binding.value(v);
})()
:v);
elem.setAttribute(cljs.core.name(k),v_47320__$1);
}

return elem;
}));

(crate.compiler.dom_attr.cljs$lang$maxFixedArity = 3);

/**
 * Regular expression that parses a CSS-style id and class from a tag name.
 */
crate.compiler.re_tag = /([^\s\.#]+)(?:#([^\s\.#]+))?(?:\.([^\s#]+))?/;
crate.compiler.normalize_map_attrs = (function crate$compiler$normalize_map_attrs(map_attrs){
return cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,cljs.core.map.cljs$core$IFn$_invoke$arity$2((function (p__47175){
var vec__47176 = p__47175;
var n = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47176,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47176,(1),null);
if(v === true){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [n,cljs.core.name(n)], null);
} else {
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [n,v], null);
}
}),cljs.core.filter.cljs$core$IFn$_invoke$arity$2(cljs.core.comp.cljs$core$IFn$_invoke$arity$2(cljs.core.boolean$,cljs.core.second),map_attrs)));
});
/**
 * Ensure a tag vector is of the form [tag-name attrs content].
 */
crate.compiler.normalize_element = (function crate$compiler$normalize_element(p__47184){
var vec__47186 = p__47184;
var seq__47187 = cljs.core.seq(vec__47186);
var first__47188 = cljs.core.first(seq__47187);
var seq__47187__$1 = cljs.core.next(seq__47187);
var tag = first__47188;
var content = seq__47187__$1;
if((!((((tag instanceof cljs.core.Keyword)) || ((tag instanceof cljs.core.Symbol)) || (typeof tag === 'string'))))){
throw [cljs.core.str.cljs$core$IFn$_invoke$arity$1(tag)," is not a valid tag name."].join('');
} else {
}

var vec__47195 = cljs.core.re_matches(crate.compiler.re_tag,cljs.core.name(tag));
var _ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47195,(0),null);
var tag__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47195,(1),null);
var id = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47195,(2),null);
var class$ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47195,(3),null);
var vec__47198 = (function (){var vec__47201 = clojure.string.split.cljs$core$IFn$_invoke$arity$2(tag__$1,/:/);
var nsp = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47201,(0),null);
var t = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47201,(1),null);
var ns_xmlns = (function (){var G__47205 = cljs.core.keyword.cljs$core$IFn$_invoke$arity$1(nsp);
return (crate.compiler.xmlns.cljs$core$IFn$_invoke$arity$1 ? crate.compiler.xmlns.cljs$core$IFn$_invoke$arity$1(G__47205) : crate.compiler.xmlns.call(null,G__47205));
})();
if(cljs.core.truth_(t)){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [(function (){var or__4126__auto__ = ns_xmlns;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return nsp;
}
})(),t], null);
} else {
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"xhtml","xhtml",1912943770).cljs$core$IFn$_invoke$arity$1(crate.compiler.xmlns),nsp], null);
}
})();
var nsp = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47198,(0),null);
var tag__$2 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47198,(1),null);
var tag_attrs = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentArrayMap.EMPTY,cljs.core.filter.cljs$core$IFn$_invoke$arity$2((function (p1__47182_SHARP_){
return (!((cljs.core.second(p1__47182_SHARP_) == null)));
}),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"id","id",-1388402092),(function (){var or__4126__auto__ = id;
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return null;
}
})(),new cljs.core.Keyword(null,"class","class",-2030961996),(cljs.core.truth_(class$)?clojure.string.replace(class$,/\./," "):null)], null)));
var map_attrs = cljs.core.first(content);
if(cljs.core.map_QMARK_(map_attrs)){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [nsp,tag__$2,cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([tag_attrs,crate.compiler.normalize_map_attrs(map_attrs)], 0)),cljs.core.next(content)], null);
} else {
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [nsp,tag__$2,tag_attrs,content], null);
}
});
crate.compiler.parse_content = (function crate$compiler$parse_content(elem,content){
var attrs = cljs.core.first(content);
if(cljs.core.map_QMARK_(attrs)){
crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$2(elem,attrs);

return cljs.core.rest(content);
} else {
return content;
}
});
crate.compiler.create_elem = (cljs.core.truth_(document.createElementNS)?(function (nsp,tag){
return document.createElementNS(nsp,tag);
}):(function (_,tag){
return document.createElement(tag);
}));
crate.compiler.elem_factory = (function crate$compiler$elem_factory(tag_def){
var bindings_orig_val__47207 = crate.compiler.bindings;
var bindings_temp_val__47208 = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(cljs.core.PersistentVector.EMPTY);
(crate.compiler.bindings = bindings_temp_val__47208);

try{var vec__47209 = crate.compiler.normalize_element(tag_def);
var nsp = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47209,(0),null);
var tag = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47209,(1),null);
var attrs = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47209,(2),null);
var content = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47209,(3),null);
var elem = crate.compiler.create_elem(nsp,tag);
crate.compiler.dom_attr.cljs$core$IFn$_invoke$arity$2(elem,attrs);

crate.compiler.as_content(elem,content);

crate.compiler.handle_bindings(cljs.core.deref(crate.compiler.bindings),elem);

return elem;
}finally {(crate.compiler.bindings = bindings_orig_val__47207);
}});
/**
 * Add an optional attribute argument to a function that returns a vector tag.
 */
crate.compiler.add_optional_attrs = (function crate$compiler$add_optional_attrs(func){
return (function() { 
var G__47329__delegate = function (args){
if(cljs.core.map_QMARK_(cljs.core.first(args))){
var vec__47216 = cljs.core.apply.cljs$core$IFn$_invoke$arity$2(func,cljs.core.rest(args));
var seq__47217 = cljs.core.seq(vec__47216);
var first__47218 = cljs.core.first(seq__47217);
var seq__47217__$1 = cljs.core.next(seq__47217);
var tag = first__47218;
var body = seq__47217__$1;
if(cljs.core.map_QMARK_(cljs.core.first(body))){
return cljs.core.apply.cljs$core$IFn$_invoke$arity$4(cljs.core.vector,tag,cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([cljs.core.first(body),cljs.core.first(args)], 0)),cljs.core.rest(body));
} else {
return cljs.core.apply.cljs$core$IFn$_invoke$arity$4(cljs.core.vector,tag,cljs.core.first(args),body);
}
} else {
return cljs.core.apply.cljs$core$IFn$_invoke$arity$2(func,args);
}
};
var G__47329 = function (var_args){
var args = null;
if (arguments.length > 0) {
var G__47330__i = 0, G__47330__a = new Array(arguments.length -  0);
while (G__47330__i < G__47330__a.length) {G__47330__a[G__47330__i] = arguments[G__47330__i + 0]; ++G__47330__i;}
  args = new cljs.core.IndexedSeq(G__47330__a,0,null);
} 
return G__47329__delegate.call(this,args);};
G__47329.cljs$lang$maxFixedArity = 0;
G__47329.cljs$lang$applyTo = (function (arglist__47331){
var args = cljs.core.seq(arglist__47331);
return G__47329__delegate(args);
});
G__47329.cljs$core$IFn$_invoke$arity$variadic = G__47329__delegate;
return G__47329;
})()
;
});

//# sourceMappingURL=crate.compiler.js.map
