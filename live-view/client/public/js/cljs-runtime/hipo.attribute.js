goog.provide('hipo.attribute');
hipo.attribute.style_handler = new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"attr","attr",-604132353),"style"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (p1__37203_SHARP_,p2__37204_SHARP_,p3__37205_SHARP_,p4__37202_SHARP_){
var seq__37207 = cljs.core.seq(p4__37202_SHARP_);
var chunk__37208 = null;
var count__37209 = (0);
var i__37210 = (0);
while(true){
if((i__37210 < count__37209)){
var vec__37219 = chunk__37208.cljs$core$IIndexed$_nth$arity$2(null,i__37210);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37219,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37219,(1),null);
(p1__37203_SHARP_["style"][cljs.core.name(k)] = v);


var G__37248 = seq__37207;
var G__37249 = chunk__37208;
var G__37250 = count__37209;
var G__37251 = (i__37210 + (1));
seq__37207 = G__37248;
chunk__37208 = G__37249;
count__37209 = G__37250;
i__37210 = G__37251;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__37207);
if(temp__5735__auto__){
var seq__37207__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__37207__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__37207__$1);
var G__37252 = cljs.core.chunk_rest(seq__37207__$1);
var G__37253 = c__4556__auto__;
var G__37254 = cljs.core.count(c__4556__auto__);
var G__37255 = (0);
seq__37207 = G__37252;
chunk__37208 = G__37253;
count__37209 = G__37254;
i__37210 = G__37255;
continue;
} else {
var vec__37224 = cljs.core.first(seq__37207__$1);
var k = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37224,(0),null);
var v = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__37224,(1),null);
(p1__37203_SHARP_["style"][cljs.core.name(k)] = v);


var G__37256 = cljs.core.next(seq__37207__$1);
var G__37257 = null;
var G__37258 = (0);
var G__37259 = (0);
seq__37207 = G__37256;
chunk__37208 = G__37257;
count__37209 = G__37258;
i__37210 = G__37259;
continue;
}
} else {
return null;
}
}
break;
}
})], null);
hipo.attribute.property_name__GT_js_property_name = (function hipo$attribute$property_name__GT_js_property_name(n){
return n.replace("-","_");
});
hipo.attribute.set_property_value = (function hipo$attribute$set_property_value(el,k,v){
return (el[hipo.attribute.property_name__GT_js_property_name(cljs.core.name(k))] = v);
});
hipo.attribute.set_attribute_BANG_ = (function hipo$attribute$set_attribute_BANG_(el,k,v,m){
var temp__5733__auto__ = (((k instanceof cljs.core.Keyword))?hipo.hiccup.key__GT_namespace(cljs.core.namespace(k),m):null);
if(cljs.core.truth_(temp__5733__auto__)){
var nns = temp__5733__auto__;
return el.setAttributeNS(nns,cljs.core.name(k),v);
} else {
return el.setAttribute(cljs.core.name(k),v);
}
});
hipo.attribute.remove_attribute_BANG_ = (function hipo$attribute$remove_attribute_BANG_(el,k,m){
var temp__5733__auto__ = (((k instanceof cljs.core.Keyword))?hipo.hiccup.key__GT_namespace(cljs.core.namespace(k),m):null);
if(cljs.core.truth_(temp__5733__auto__)){
var nns = temp__5733__auto__;
return el.removeAttributeNS(nns,cljs.core.name(k));
} else {
return el.removeAttribute(cljs.core.name(k));
}
});
hipo.attribute.default_handler_fns = new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"prop","prop",-515168332),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (p1__37228_SHARP_,p2__37229_SHARP_,p3__37231_SHARP_,p4__37230_SHARP_){
return hipo.attribute.set_property_value(p1__37228_SHARP_,p2__37229_SHARP_,p4__37230_SHARP_);
})], null),new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (p1__37233_SHARP_,p2__37234_SHARP_,p3__37236_SHARP_,p4__37232_SHARP_,p5__37235_SHARP_){
if(cljs.core.truth_(p4__37232_SHARP_)){
return hipo.attribute.set_attribute_BANG_(p1__37233_SHARP_,p2__37234_SHARP_,p4__37232_SHARP_,p5__37235_SHARP_);
} else {
return hipo.attribute.remove_attribute_BANG_(p1__37233_SHARP_,p2__37234_SHARP_,p5__37235_SHARP_);
}
})], null)], null);
hipo.attribute.default_handlers = new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"ns","ns",441598760),"svg",new cljs.core.Keyword(null,"attr","attr",-604132353),"class"], null),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"attr","attr",-604132353)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),"input",new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["value",null,"checked",null], null), null)], null),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"prop","prop",-515168332)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),"input",new cljs.core.Keyword(null,"attr","attr",-604132353),"autofocus"], null),new cljs.core.Keyword(null,"fn","fn",-1175266204),(function (p1__37239_SHARP_,p2__37240_SHARP_,p3__37241_SHARP_,p4__37238_SHARP_){
if(cljs.core.truth_(p4__37238_SHARP_)){
p1__37239_SHARP_.focus();

return p1__37239_SHARP_.setAttribute(p2__37240_SHARP_,p4__37238_SHARP_);
} else {
return null;
}
})], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),"option",new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["selected",null], null), null)], null),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"prop","prop",-515168332)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),"select",new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, ["value",null,"selectIndex",null], null), null)], null),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"prop","prop",-515168332)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"target","target",253001721),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),"textarea",new cljs.core.Keyword(null,"attr","attr",-604132353),new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["value",null], null), null)], null),new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"prop","prop",-515168332)], null)], null);
hipo.attribute.matches_QMARK_ = (function hipo$attribute$matches_QMARK_(expr,s){
if(cljs.core.truth_(expr)){
if(cljs.core.set_QMARK_(expr)){
return cljs.core.contains_QMARK_(expr,s);
} else {
return cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(s,expr);

}
} else {
return true;
}
});
hipo.attribute.target_matches_QMARK_ = (function hipo$attribute$target_matches_QMARK_(m,ns,tag,attr){
return ((hipo.attribute.matches_QMARK_(new cljs.core.Keyword(null,"ns","ns",441598760).cljs$core$IFn$_invoke$arity$1(m),ns)) && (hipo.attribute.matches_QMARK_(new cljs.core.Keyword(null,"tag","tag",-1290361223).cljs$core$IFn$_invoke$arity$1(m),tag)) && (hipo.attribute.matches_QMARK_(new cljs.core.Keyword(null,"attr","attr",-604132353).cljs$core$IFn$_invoke$arity$1(m),attr)));
});
hipo.attribute.handler = (function hipo$attribute$handler(m,ns,tag,attr){
var v = cljs.core.concat.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"attribute-handlers","attribute-handlers",855454691).cljs$core$IFn$_invoke$arity$1(m),hipo.attribute.default_handlers);
var h = cljs.core.some((function (p1__37246_SHARP_){
var t = new cljs.core.Keyword(null,"target","target",253001721).cljs$core$IFn$_invoke$arity$1(p1__37246_SHARP_);
if(hipo.attribute.target_matches_QMARK_(t,ns,tag,cljs.core.name(attr))){
return p1__37246_SHARP_;
} else {
return null;
}
}),v);
if(cljs.core.contains_QMARK_(h,new cljs.core.Keyword(null,"type","type",1174270348))){
var fexpr__37247 = new cljs.core.Keyword(null,"type","type",1174270348).cljs$core$IFn$_invoke$arity$1(h);
return (fexpr__37247.cljs$core$IFn$_invoke$arity$1 ? fexpr__37247.cljs$core$IFn$_invoke$arity$1(hipo.attribute.default_handler_fns) : fexpr__37247.call(null,hipo.attribute.default_handler_fns));
} else {
return h;
}
});
hipo.attribute.default_set_value_BANG_ = (function hipo$attribute$default_set_value_BANG_(el,attr,ov,nv,m){
if(((hipo.hiccup.literal_QMARK_(ov)) || (hipo.hiccup.literal_QMARK_(nv)))){
if(cljs.core.truth_(nv)){
return hipo.attribute.set_attribute_BANG_(el,attr,nv,m);
} else {
return hipo.attribute.remove_attribute_BANG_(el,attr,m);
}
} else {
return (el[attr] = hipo.attribute.set_property_value(el,attr,nv));
}
});
hipo.attribute.set_value_BANG_ = (function hipo$attribute$set_value_BANG_(el,m,ns,tag,attr,ov,nv){
var h = hipo.attribute.handler(m,ns,tag,attr);
var f = (function (){var or__4126__auto__ = new cljs.core.Keyword(null,"fn","fn",-1175266204).cljs$core$IFn$_invoke$arity$1(h);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return hipo.attribute.default_set_value_BANG_;
}
})();
return (f.cljs$core$IFn$_invoke$arity$5 ? f.cljs$core$IFn$_invoke$arity$5(el,attr,ov,nv,m) : f.call(null,el,attr,ov,nv,m));
});

//# sourceMappingURL=hipo.attribute.js.map
