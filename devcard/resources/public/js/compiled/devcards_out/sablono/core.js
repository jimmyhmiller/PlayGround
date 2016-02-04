// Compiled by ClojureScript 1.7.170 {}
goog.provide('sablono.core');
goog.require('cljs.core');
goog.require('goog.dom');
goog.require('goog.string');
goog.require('sablono.normalize');
goog.require('sablono.util');
goog.require('sablono.interpreter');
goog.require('cljsjs.react');
goog.require('cljsjs.react.dom.server');
goog.require('cljsjs.react.dom');
goog.require('clojure.string');
/**
 * Add an optional attribute argument to a function that returns a element vector.
 */
sablono.core.wrap_attrs = (function sablono$core$wrap_attrs(func){
return (function() { 
var G__23456__delegate = function (args){
if(cljs.core.map_QMARK_.call(null,cljs.core.first.call(null,args))){
var vec__23455 = cljs.core.apply.call(null,func,cljs.core.rest.call(null,args));
var tag = cljs.core.nth.call(null,vec__23455,(0),null);
var body = cljs.core.nthnext.call(null,vec__23455,(1));
if(cljs.core.map_QMARK_.call(null,cljs.core.first.call(null,body))){
return cljs.core.apply.call(null,cljs.core.vector,tag,cljs.core.merge.call(null,cljs.core.first.call(null,body),cljs.core.first.call(null,args)),cljs.core.rest.call(null,body));
} else {
return cljs.core.apply.call(null,cljs.core.vector,tag,cljs.core.first.call(null,args),body);
}
} else {
return cljs.core.apply.call(null,func,args);
}
};
var G__23456 = function (var_args){
var args = null;
if (arguments.length > 0) {
var G__23457__i = 0, G__23457__a = new Array(arguments.length -  0);
while (G__23457__i < G__23457__a.length) {G__23457__a[G__23457__i] = arguments[G__23457__i + 0]; ++G__23457__i;}
  args = new cljs.core.IndexedSeq(G__23457__a,0);
} 
return G__23456__delegate.call(this,args);};
G__23456.cljs$lang$maxFixedArity = 0;
G__23456.cljs$lang$applyTo = (function (arglist__23458){
var args = cljs.core.seq(arglist__23458);
return G__23456__delegate(args);
});
G__23456.cljs$core$IFn$_invoke$arity$variadic = G__23456__delegate;
return G__23456;
})()
;
});
sablono.core.update_arglists = (function sablono$core$update_arglists(arglists){
var iter__17538__auto__ = (function sablono$core$update_arglists_$_iter__23463(s__23464){
return (new cljs.core.LazySeq(null,(function (){
var s__23464__$1 = s__23464;
while(true){
var temp__4425__auto__ = cljs.core.seq.call(null,s__23464__$1);
if(temp__4425__auto__){
var s__23464__$2 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,s__23464__$2)){
var c__17536__auto__ = cljs.core.chunk_first.call(null,s__23464__$2);
var size__17537__auto__ = cljs.core.count.call(null,c__17536__auto__);
var b__23466 = cljs.core.chunk_buffer.call(null,size__17537__auto__);
if((function (){var i__23465 = (0);
while(true){
if((i__23465 < size__17537__auto__)){
var args = cljs.core._nth.call(null,c__17536__auto__,i__23465);
cljs.core.chunk_append.call(null,b__23466,cljs.core.vec.call(null,cljs.core.cons.call(null,new cljs.core.Symbol(null,"attr-map?","attr-map?",116307443,null),args)));

var G__23467 = (i__23465 + (1));
i__23465 = G__23467;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23466),sablono$core$update_arglists_$_iter__23463.call(null,cljs.core.chunk_rest.call(null,s__23464__$2)));
} else {
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23466),null);
}
} else {
var args = cljs.core.first.call(null,s__23464__$2);
return cljs.core.cons.call(null,cljs.core.vec.call(null,cljs.core.cons.call(null,new cljs.core.Symbol(null,"attr-map?","attr-map?",116307443,null),args)),sablono$core$update_arglists_$_iter__23463.call(null,cljs.core.rest.call(null,s__23464__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__17538__auto__.call(null,arglists);
});
/**
 * Render `element` as HTML string.
 */
sablono.core.render = (function sablono$core$render(element){
if(cljs.core.truth_(element)){
return ReactDOMServer.renderToString(element);
} else {
return null;
}
});
/**
 * Render `element` as HTML string, without React internal attributes.
 */
sablono.core.render_static = (function sablono$core$render_static(element){
if(cljs.core.truth_(element)){
return ReactDOMServer.renderToStaticMarkup(element);
} else {
return null;
}
});
/**
 * Include a list of external stylesheet files.
 */
sablono.core.include_css = (function sablono$core$include_css(var_args){
var args__17831__auto__ = [];
var len__17824__auto___23473 = arguments.length;
var i__17825__auto___23474 = (0);
while(true){
if((i__17825__auto___23474 < len__17824__auto___23473)){
args__17831__auto__.push((arguments[i__17825__auto___23474]));

var G__23475 = (i__17825__auto___23474 + (1));
i__17825__auto___23474 = G__23475;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return sablono.core.include_css.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

sablono.core.include_css.cljs$core$IFn$_invoke$arity$variadic = (function (styles){
var iter__17538__auto__ = (function sablono$core$iter__23469(s__23470){
return (new cljs.core.LazySeq(null,(function (){
var s__23470__$1 = s__23470;
while(true){
var temp__4425__auto__ = cljs.core.seq.call(null,s__23470__$1);
if(temp__4425__auto__){
var s__23470__$2 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,s__23470__$2)){
var c__17536__auto__ = cljs.core.chunk_first.call(null,s__23470__$2);
var size__17537__auto__ = cljs.core.count.call(null,c__17536__auto__);
var b__23472 = cljs.core.chunk_buffer.call(null,size__17537__auto__);
if((function (){var i__23471 = (0);
while(true){
if((i__23471 < size__17537__auto__)){
var style = cljs.core._nth.call(null,c__17536__auto__,i__23471);
cljs.core.chunk_append.call(null,b__23472,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"link","link",-1769163468),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"type","type",1174270348),"text/css",new cljs.core.Keyword(null,"href","href",-793805698),sablono.util.as_str.call(null,style),new cljs.core.Keyword(null,"rel","rel",1378823488),"stylesheet"], null)], null));

var G__23476 = (i__23471 + (1));
i__23471 = G__23476;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23472),sablono$core$iter__23469.call(null,cljs.core.chunk_rest.call(null,s__23470__$2)));
} else {
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23472),null);
}
} else {
var style = cljs.core.first.call(null,s__23470__$2);
return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"link","link",-1769163468),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"type","type",1174270348),"text/css",new cljs.core.Keyword(null,"href","href",-793805698),sablono.util.as_str.call(null,style),new cljs.core.Keyword(null,"rel","rel",1378823488),"stylesheet"], null)], null),sablono$core$iter__23469.call(null,cljs.core.rest.call(null,s__23470__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__17538__auto__.call(null,styles);
});

sablono.core.include_css.cljs$lang$maxFixedArity = (0);

sablono.core.include_css.cljs$lang$applyTo = (function (seq23468){
return sablono.core.include_css.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq23468));
});
/**
 * Include the JavaScript library at `src`.
 */
sablono.core.include_js = (function sablono$core$include_js(src){
return goog.dom.appendChild(goog.dom.getDocument().body,goog.dom.createDom("script",{"src": src}));
});
/**
 * Include Facebook's React JavaScript library.
 */
sablono.core.include_react = (function sablono$core$include_react(){
return sablono.core.include_js.call(null,"http://fb.me/react-0.12.2.js");
});
/**
 * Wraps some content in a HTML hyperlink with the supplied URL.
 */
sablono.core.link_to23477 = (function sablono$core$link_to23477(var_args){
var args__17831__auto__ = [];
var len__17824__auto___23480 = arguments.length;
var i__17825__auto___23481 = (0);
while(true){
if((i__17825__auto___23481 < len__17824__auto___23480)){
args__17831__auto__.push((arguments[i__17825__auto___23481]));

var G__23482 = (i__17825__auto___23481 + (1));
i__17825__auto___23481 = G__23482;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((1) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((1)),(0))):null);
return sablono.core.link_to23477.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__17832__auto__);
});

sablono.core.link_to23477.cljs$core$IFn$_invoke$arity$variadic = (function (url,content){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"a","a",-2123407586),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"href","href",-793805698),sablono.util.as_str.call(null,url)], null),content], null);
});

sablono.core.link_to23477.cljs$lang$maxFixedArity = (1);

sablono.core.link_to23477.cljs$lang$applyTo = (function (seq23478){
var G__23479 = cljs.core.first.call(null,seq23478);
var seq23478__$1 = cljs.core.next.call(null,seq23478);
return sablono.core.link_to23477.cljs$core$IFn$_invoke$arity$variadic(G__23479,seq23478__$1);
});

sablono.core.link_to = sablono.core.wrap_attrs.call(null,sablono.core.link_to23477);
/**
 * Wraps some content in a HTML hyperlink with the supplied e-mail
 *   address. If no content provided use the e-mail address as content.
 */
sablono.core.mail_to23483 = (function sablono$core$mail_to23483(var_args){
var args__17831__auto__ = [];
var len__17824__auto___23488 = arguments.length;
var i__17825__auto___23489 = (0);
while(true){
if((i__17825__auto___23489 < len__17824__auto___23488)){
args__17831__auto__.push((arguments[i__17825__auto___23489]));

var G__23490 = (i__17825__auto___23489 + (1));
i__17825__auto___23489 = G__23490;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((1) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((1)),(0))):null);
return sablono.core.mail_to23483.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__17832__auto__);
});

sablono.core.mail_to23483.cljs$core$IFn$_invoke$arity$variadic = (function (e_mail,p__23486){
var vec__23487 = p__23486;
var content = cljs.core.nth.call(null,vec__23487,(0),null);
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"a","a",-2123407586),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"href","href",-793805698),[cljs.core.str("mailto:"),cljs.core.str(e_mail)].join('')], null),(function (){var or__16766__auto__ = content;
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return e_mail;
}
})()], null);
});

sablono.core.mail_to23483.cljs$lang$maxFixedArity = (1);

sablono.core.mail_to23483.cljs$lang$applyTo = (function (seq23484){
var G__23485 = cljs.core.first.call(null,seq23484);
var seq23484__$1 = cljs.core.next.call(null,seq23484);
return sablono.core.mail_to23483.cljs$core$IFn$_invoke$arity$variadic(G__23485,seq23484__$1);
});

sablono.core.mail_to = sablono.core.wrap_attrs.call(null,sablono.core.mail_to23483);
/**
 * Wrap a collection in an unordered list.
 */
sablono.core.unordered_list23491 = (function sablono$core$unordered_list23491(coll){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"ul","ul",-1349521403),(function (){var iter__17538__auto__ = (function sablono$core$unordered_list23491_$_iter__23496(s__23497){
return (new cljs.core.LazySeq(null,(function (){
var s__23497__$1 = s__23497;
while(true){
var temp__4425__auto__ = cljs.core.seq.call(null,s__23497__$1);
if(temp__4425__auto__){
var s__23497__$2 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,s__23497__$2)){
var c__17536__auto__ = cljs.core.chunk_first.call(null,s__23497__$2);
var size__17537__auto__ = cljs.core.count.call(null,c__17536__auto__);
var b__23499 = cljs.core.chunk_buffer.call(null,size__17537__auto__);
if((function (){var i__23498 = (0);
while(true){
if((i__23498 < size__17537__auto__)){
var x = cljs.core._nth.call(null,c__17536__auto__,i__23498);
cljs.core.chunk_append.call(null,b__23499,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",723558921),x], null));

var G__23500 = (i__23498 + (1));
i__23498 = G__23500;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23499),sablono$core$unordered_list23491_$_iter__23496.call(null,cljs.core.chunk_rest.call(null,s__23497__$2)));
} else {
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23499),null);
}
} else {
var x = cljs.core.first.call(null,s__23497__$2);
return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",723558921),x], null),sablono$core$unordered_list23491_$_iter__23496.call(null,cljs.core.rest.call(null,s__23497__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__17538__auto__.call(null,coll);
})()], null);
});

sablono.core.unordered_list = sablono.core.wrap_attrs.call(null,sablono.core.unordered_list23491);
/**
 * Wrap a collection in an ordered list.
 */
sablono.core.ordered_list23501 = (function sablono$core$ordered_list23501(coll){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"ol","ol",932524051),(function (){var iter__17538__auto__ = (function sablono$core$ordered_list23501_$_iter__23506(s__23507){
return (new cljs.core.LazySeq(null,(function (){
var s__23507__$1 = s__23507;
while(true){
var temp__4425__auto__ = cljs.core.seq.call(null,s__23507__$1);
if(temp__4425__auto__){
var s__23507__$2 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,s__23507__$2)){
var c__17536__auto__ = cljs.core.chunk_first.call(null,s__23507__$2);
var size__17537__auto__ = cljs.core.count.call(null,c__17536__auto__);
var b__23509 = cljs.core.chunk_buffer.call(null,size__17537__auto__);
if((function (){var i__23508 = (0);
while(true){
if((i__23508 < size__17537__auto__)){
var x = cljs.core._nth.call(null,c__17536__auto__,i__23508);
cljs.core.chunk_append.call(null,b__23509,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",723558921),x], null));

var G__23510 = (i__23508 + (1));
i__23508 = G__23510;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23509),sablono$core$ordered_list23501_$_iter__23506.call(null,cljs.core.chunk_rest.call(null,s__23507__$2)));
} else {
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23509),null);
}
} else {
var x = cljs.core.first.call(null,s__23507__$2);
return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",723558921),x], null),sablono$core$ordered_list23501_$_iter__23506.call(null,cljs.core.rest.call(null,s__23507__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__17538__auto__.call(null,coll);
})()], null);
});

sablono.core.ordered_list = sablono.core.wrap_attrs.call(null,sablono.core.ordered_list23501);
/**
 * Create an image element.
 */
sablono.core.image23511 = (function sablono$core$image23511(var_args){
var args23512 = [];
var len__17824__auto___23515 = arguments.length;
var i__17825__auto___23516 = (0);
while(true){
if((i__17825__auto___23516 < len__17824__auto___23515)){
args23512.push((arguments[i__17825__auto___23516]));

var G__23517 = (i__17825__auto___23516 + (1));
i__17825__auto___23516 = G__23517;
continue;
} else {
}
break;
}

var G__23514 = args23512.length;
switch (G__23514) {
case 1:
return sablono.core.image23511.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.image23511.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23512.length)].join('')));

}
});

sablono.core.image23511.cljs$core$IFn$_invoke$arity$1 = (function (src){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"img","img",1442687358),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"src","src",-1651076051),sablono.util.as_str.call(null,src)], null)], null);
});

sablono.core.image23511.cljs$core$IFn$_invoke$arity$2 = (function (src,alt){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"img","img",1442687358),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"src","src",-1651076051),sablono.util.as_str.call(null,src),new cljs.core.Keyword(null,"alt","alt",-3214426),alt], null)], null);
});

sablono.core.image23511.cljs$lang$maxFixedArity = 2;

sablono.core.image = sablono.core.wrap_attrs.call(null,sablono.core.image23511);
sablono.core._STAR_group_STAR_ = cljs.core.PersistentVector.EMPTY;
/**
 * Create a field name from the supplied argument the current field group.
 */
sablono.core.make_name = (function sablono$core$make_name(name){
return cljs.core.reduce.call(null,(function (p1__23519_SHARP_,p2__23520_SHARP_){
return [cljs.core.str(p1__23519_SHARP_),cljs.core.str("["),cljs.core.str(p2__23520_SHARP_),cljs.core.str("]")].join('');
}),cljs.core.conj.call(null,sablono.core._STAR_group_STAR_,sablono.util.as_str.call(null,name)));
});
/**
 * Create a field id from the supplied argument and current field group.
 */
sablono.core.make_id = (function sablono$core$make_id(name){
return cljs.core.reduce.call(null,(function (p1__23521_SHARP_,p2__23522_SHARP_){
return [cljs.core.str(p1__23521_SHARP_),cljs.core.str("-"),cljs.core.str(p2__23522_SHARP_)].join('');
}),cljs.core.conj.call(null,sablono.core._STAR_group_STAR_,sablono.util.as_str.call(null,name)));
});
/**
 * Creates a new <input> element.
 */
sablono.core.input_field_STAR_ = (function sablono$core$input_field_STAR_(type,name,value){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",556931961),new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"type","type",1174270348),type,new cljs.core.Keyword(null,"name","name",1843675177),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",-1388402092),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",305978217),value], null)], null);
});
/**
 * Creates a color input field.
 */
sablono.core.color_field23523 = (function sablono$core$color_field23523(var_args){
var args23524 = [];
var len__17824__auto___23591 = arguments.length;
var i__17825__auto___23592 = (0);
while(true){
if((i__17825__auto___23592 < len__17824__auto___23591)){
args23524.push((arguments[i__17825__auto___23592]));

var G__23593 = (i__17825__auto___23592 + (1));
i__17825__auto___23592 = G__23593;
continue;
} else {
}
break;
}

var G__23526 = args23524.length;
switch (G__23526) {
case 1:
return sablono.core.color_field23523.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.color_field23523.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23524.length)].join('')));

}
});

sablono.core.color_field23523.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.color_field23523.call(null,name__23444__auto__,null);
});

sablono.core.color_field23523.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"color","color",-1642760596,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.color_field23523.cljs$lang$maxFixedArity = 2;

sablono.core.color_field = sablono.core.wrap_attrs.call(null,sablono.core.color_field23523);

/**
 * Creates a date input field.
 */
sablono.core.date_field23527 = (function sablono$core$date_field23527(var_args){
var args23528 = [];
var len__17824__auto___23595 = arguments.length;
var i__17825__auto___23596 = (0);
while(true){
if((i__17825__auto___23596 < len__17824__auto___23595)){
args23528.push((arguments[i__17825__auto___23596]));

var G__23597 = (i__17825__auto___23596 + (1));
i__17825__auto___23596 = G__23597;
continue;
} else {
}
break;
}

var G__23530 = args23528.length;
switch (G__23530) {
case 1:
return sablono.core.date_field23527.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.date_field23527.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23528.length)].join('')));

}
});

sablono.core.date_field23527.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.date_field23527.call(null,name__23444__auto__,null);
});

sablono.core.date_field23527.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"date","date",177097065,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.date_field23527.cljs$lang$maxFixedArity = 2;

sablono.core.date_field = sablono.core.wrap_attrs.call(null,sablono.core.date_field23527);

/**
 * Creates a datetime input field.
 */
sablono.core.datetime_field23531 = (function sablono$core$datetime_field23531(var_args){
var args23532 = [];
var len__17824__auto___23599 = arguments.length;
var i__17825__auto___23600 = (0);
while(true){
if((i__17825__auto___23600 < len__17824__auto___23599)){
args23532.push((arguments[i__17825__auto___23600]));

var G__23601 = (i__17825__auto___23600 + (1));
i__17825__auto___23600 = G__23601;
continue;
} else {
}
break;
}

var G__23534 = args23532.length;
switch (G__23534) {
case 1:
return sablono.core.datetime_field23531.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.datetime_field23531.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23532.length)].join('')));

}
});

sablono.core.datetime_field23531.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.datetime_field23531.call(null,name__23444__auto__,null);
});

sablono.core.datetime_field23531.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"datetime","datetime",2135207229,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.datetime_field23531.cljs$lang$maxFixedArity = 2;

sablono.core.datetime_field = sablono.core.wrap_attrs.call(null,sablono.core.datetime_field23531);

/**
 * Creates a datetime-local input field.
 */
sablono.core.datetime_local_field23535 = (function sablono$core$datetime_local_field23535(var_args){
var args23536 = [];
var len__17824__auto___23603 = arguments.length;
var i__17825__auto___23604 = (0);
while(true){
if((i__17825__auto___23604 < len__17824__auto___23603)){
args23536.push((arguments[i__17825__auto___23604]));

var G__23605 = (i__17825__auto___23604 + (1));
i__17825__auto___23604 = G__23605;
continue;
} else {
}
break;
}

var G__23538 = args23536.length;
switch (G__23538) {
case 1:
return sablono.core.datetime_local_field23535.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.datetime_local_field23535.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23536.length)].join('')));

}
});

sablono.core.datetime_local_field23535.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.datetime_local_field23535.call(null,name__23444__auto__,null);
});

sablono.core.datetime_local_field23535.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"datetime-local","datetime-local",-507312697,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.datetime_local_field23535.cljs$lang$maxFixedArity = 2;

sablono.core.datetime_local_field = sablono.core.wrap_attrs.call(null,sablono.core.datetime_local_field23535);

/**
 * Creates a email input field.
 */
sablono.core.email_field23539 = (function sablono$core$email_field23539(var_args){
var args23540 = [];
var len__17824__auto___23607 = arguments.length;
var i__17825__auto___23608 = (0);
while(true){
if((i__17825__auto___23608 < len__17824__auto___23607)){
args23540.push((arguments[i__17825__auto___23608]));

var G__23609 = (i__17825__auto___23608 + (1));
i__17825__auto___23608 = G__23609;
continue;
} else {
}
break;
}

var G__23542 = args23540.length;
switch (G__23542) {
case 1:
return sablono.core.email_field23539.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.email_field23539.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23540.length)].join('')));

}
});

sablono.core.email_field23539.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.email_field23539.call(null,name__23444__auto__,null);
});

sablono.core.email_field23539.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"email","email",-1238619063,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.email_field23539.cljs$lang$maxFixedArity = 2;

sablono.core.email_field = sablono.core.wrap_attrs.call(null,sablono.core.email_field23539);

/**
 * Creates a file input field.
 */
sablono.core.file_field23543 = (function sablono$core$file_field23543(var_args){
var args23544 = [];
var len__17824__auto___23611 = arguments.length;
var i__17825__auto___23612 = (0);
while(true){
if((i__17825__auto___23612 < len__17824__auto___23611)){
args23544.push((arguments[i__17825__auto___23612]));

var G__23613 = (i__17825__auto___23612 + (1));
i__17825__auto___23612 = G__23613;
continue;
} else {
}
break;
}

var G__23546 = args23544.length;
switch (G__23546) {
case 1:
return sablono.core.file_field23543.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.file_field23543.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23544.length)].join('')));

}
});

sablono.core.file_field23543.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.file_field23543.call(null,name__23444__auto__,null);
});

sablono.core.file_field23543.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"file","file",370885649,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.file_field23543.cljs$lang$maxFixedArity = 2;

sablono.core.file_field = sablono.core.wrap_attrs.call(null,sablono.core.file_field23543);

/**
 * Creates a hidden input field.
 */
sablono.core.hidden_field23547 = (function sablono$core$hidden_field23547(var_args){
var args23548 = [];
var len__17824__auto___23615 = arguments.length;
var i__17825__auto___23616 = (0);
while(true){
if((i__17825__auto___23616 < len__17824__auto___23615)){
args23548.push((arguments[i__17825__auto___23616]));

var G__23617 = (i__17825__auto___23616 + (1));
i__17825__auto___23616 = G__23617;
continue;
} else {
}
break;
}

var G__23550 = args23548.length;
switch (G__23550) {
case 1:
return sablono.core.hidden_field23547.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.hidden_field23547.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23548.length)].join('')));

}
});

sablono.core.hidden_field23547.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.hidden_field23547.call(null,name__23444__auto__,null);
});

sablono.core.hidden_field23547.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"hidden","hidden",1328025435,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.hidden_field23547.cljs$lang$maxFixedArity = 2;

sablono.core.hidden_field = sablono.core.wrap_attrs.call(null,sablono.core.hidden_field23547);

/**
 * Creates a month input field.
 */
sablono.core.month_field23551 = (function sablono$core$month_field23551(var_args){
var args23552 = [];
var len__17824__auto___23619 = arguments.length;
var i__17825__auto___23620 = (0);
while(true){
if((i__17825__auto___23620 < len__17824__auto___23619)){
args23552.push((arguments[i__17825__auto___23620]));

var G__23621 = (i__17825__auto___23620 + (1));
i__17825__auto___23620 = G__23621;
continue;
} else {
}
break;
}

var G__23554 = args23552.length;
switch (G__23554) {
case 1:
return sablono.core.month_field23551.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.month_field23551.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23552.length)].join('')));

}
});

sablono.core.month_field23551.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.month_field23551.call(null,name__23444__auto__,null);
});

sablono.core.month_field23551.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"month","month",-319717006,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.month_field23551.cljs$lang$maxFixedArity = 2;

sablono.core.month_field = sablono.core.wrap_attrs.call(null,sablono.core.month_field23551);

/**
 * Creates a number input field.
 */
sablono.core.number_field23555 = (function sablono$core$number_field23555(var_args){
var args23556 = [];
var len__17824__auto___23623 = arguments.length;
var i__17825__auto___23624 = (0);
while(true){
if((i__17825__auto___23624 < len__17824__auto___23623)){
args23556.push((arguments[i__17825__auto___23624]));

var G__23625 = (i__17825__auto___23624 + (1));
i__17825__auto___23624 = G__23625;
continue;
} else {
}
break;
}

var G__23558 = args23556.length;
switch (G__23558) {
case 1:
return sablono.core.number_field23555.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.number_field23555.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23556.length)].join('')));

}
});

sablono.core.number_field23555.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.number_field23555.call(null,name__23444__auto__,null);
});

sablono.core.number_field23555.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"number","number",-1084057331,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.number_field23555.cljs$lang$maxFixedArity = 2;

sablono.core.number_field = sablono.core.wrap_attrs.call(null,sablono.core.number_field23555);

/**
 * Creates a password input field.
 */
sablono.core.password_field23559 = (function sablono$core$password_field23559(var_args){
var args23560 = [];
var len__17824__auto___23627 = arguments.length;
var i__17825__auto___23628 = (0);
while(true){
if((i__17825__auto___23628 < len__17824__auto___23627)){
args23560.push((arguments[i__17825__auto___23628]));

var G__23629 = (i__17825__auto___23628 + (1));
i__17825__auto___23628 = G__23629;
continue;
} else {
}
break;
}

var G__23562 = args23560.length;
switch (G__23562) {
case 1:
return sablono.core.password_field23559.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.password_field23559.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23560.length)].join('')));

}
});

sablono.core.password_field23559.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.password_field23559.call(null,name__23444__auto__,null);
});

sablono.core.password_field23559.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"password","password",2057553998,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.password_field23559.cljs$lang$maxFixedArity = 2;

sablono.core.password_field = sablono.core.wrap_attrs.call(null,sablono.core.password_field23559);

/**
 * Creates a range input field.
 */
sablono.core.range_field23563 = (function sablono$core$range_field23563(var_args){
var args23564 = [];
var len__17824__auto___23631 = arguments.length;
var i__17825__auto___23632 = (0);
while(true){
if((i__17825__auto___23632 < len__17824__auto___23631)){
args23564.push((arguments[i__17825__auto___23632]));

var G__23633 = (i__17825__auto___23632 + (1));
i__17825__auto___23632 = G__23633;
continue;
} else {
}
break;
}

var G__23566 = args23564.length;
switch (G__23566) {
case 1:
return sablono.core.range_field23563.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.range_field23563.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23564.length)].join('')));

}
});

sablono.core.range_field23563.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.range_field23563.call(null,name__23444__auto__,null);
});

sablono.core.range_field23563.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"range","range",-1014743483,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.range_field23563.cljs$lang$maxFixedArity = 2;

sablono.core.range_field = sablono.core.wrap_attrs.call(null,sablono.core.range_field23563);

/**
 * Creates a search input field.
 */
sablono.core.search_field23567 = (function sablono$core$search_field23567(var_args){
var args23568 = [];
var len__17824__auto___23635 = arguments.length;
var i__17825__auto___23636 = (0);
while(true){
if((i__17825__auto___23636 < len__17824__auto___23635)){
args23568.push((arguments[i__17825__auto___23636]));

var G__23637 = (i__17825__auto___23636 + (1));
i__17825__auto___23636 = G__23637;
continue;
} else {
}
break;
}

var G__23570 = args23568.length;
switch (G__23570) {
case 1:
return sablono.core.search_field23567.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.search_field23567.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23568.length)].join('')));

}
});

sablono.core.search_field23567.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.search_field23567.call(null,name__23444__auto__,null);
});

sablono.core.search_field23567.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"search","search",-1089495947,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.search_field23567.cljs$lang$maxFixedArity = 2;

sablono.core.search_field = sablono.core.wrap_attrs.call(null,sablono.core.search_field23567);

/**
 * Creates a tel input field.
 */
sablono.core.tel_field23571 = (function sablono$core$tel_field23571(var_args){
var args23572 = [];
var len__17824__auto___23639 = arguments.length;
var i__17825__auto___23640 = (0);
while(true){
if((i__17825__auto___23640 < len__17824__auto___23639)){
args23572.push((arguments[i__17825__auto___23640]));

var G__23641 = (i__17825__auto___23640 + (1));
i__17825__auto___23640 = G__23641;
continue;
} else {
}
break;
}

var G__23574 = args23572.length;
switch (G__23574) {
case 1:
return sablono.core.tel_field23571.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.tel_field23571.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23572.length)].join('')));

}
});

sablono.core.tel_field23571.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.tel_field23571.call(null,name__23444__auto__,null);
});

sablono.core.tel_field23571.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"tel","tel",1864669686,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.tel_field23571.cljs$lang$maxFixedArity = 2;

sablono.core.tel_field = sablono.core.wrap_attrs.call(null,sablono.core.tel_field23571);

/**
 * Creates a text input field.
 */
sablono.core.text_field23575 = (function sablono$core$text_field23575(var_args){
var args23576 = [];
var len__17824__auto___23643 = arguments.length;
var i__17825__auto___23644 = (0);
while(true){
if((i__17825__auto___23644 < len__17824__auto___23643)){
args23576.push((arguments[i__17825__auto___23644]));

var G__23645 = (i__17825__auto___23644 + (1));
i__17825__auto___23644 = G__23645;
continue;
} else {
}
break;
}

var G__23578 = args23576.length;
switch (G__23578) {
case 1:
return sablono.core.text_field23575.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.text_field23575.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23576.length)].join('')));

}
});

sablono.core.text_field23575.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.text_field23575.call(null,name__23444__auto__,null);
});

sablono.core.text_field23575.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"text","text",-150030170,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.text_field23575.cljs$lang$maxFixedArity = 2;

sablono.core.text_field = sablono.core.wrap_attrs.call(null,sablono.core.text_field23575);

/**
 * Creates a time input field.
 */
sablono.core.time_field23579 = (function sablono$core$time_field23579(var_args){
var args23580 = [];
var len__17824__auto___23647 = arguments.length;
var i__17825__auto___23648 = (0);
while(true){
if((i__17825__auto___23648 < len__17824__auto___23647)){
args23580.push((arguments[i__17825__auto___23648]));

var G__23649 = (i__17825__auto___23648 + (1));
i__17825__auto___23648 = G__23649;
continue;
} else {
}
break;
}

var G__23582 = args23580.length;
switch (G__23582) {
case 1:
return sablono.core.time_field23579.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.time_field23579.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23580.length)].join('')));

}
});

sablono.core.time_field23579.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.time_field23579.call(null,name__23444__auto__,null);
});

sablono.core.time_field23579.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"time","time",-1268547887,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.time_field23579.cljs$lang$maxFixedArity = 2;

sablono.core.time_field = sablono.core.wrap_attrs.call(null,sablono.core.time_field23579);

/**
 * Creates a url input field.
 */
sablono.core.url_field23583 = (function sablono$core$url_field23583(var_args){
var args23584 = [];
var len__17824__auto___23651 = arguments.length;
var i__17825__auto___23652 = (0);
while(true){
if((i__17825__auto___23652 < len__17824__auto___23651)){
args23584.push((arguments[i__17825__auto___23652]));

var G__23653 = (i__17825__auto___23652 + (1));
i__17825__auto___23652 = G__23653;
continue;
} else {
}
break;
}

var G__23586 = args23584.length;
switch (G__23586) {
case 1:
return sablono.core.url_field23583.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.url_field23583.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23584.length)].join('')));

}
});

sablono.core.url_field23583.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.url_field23583.call(null,name__23444__auto__,null);
});

sablono.core.url_field23583.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"url","url",1916828573,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.url_field23583.cljs$lang$maxFixedArity = 2;

sablono.core.url_field = sablono.core.wrap_attrs.call(null,sablono.core.url_field23583);

/**
 * Creates a week input field.
 */
sablono.core.week_field23587 = (function sablono$core$week_field23587(var_args){
var args23588 = [];
var len__17824__auto___23655 = arguments.length;
var i__17825__auto___23656 = (0);
while(true){
if((i__17825__auto___23656 < len__17824__auto___23655)){
args23588.push((arguments[i__17825__auto___23656]));

var G__23657 = (i__17825__auto___23656 + (1));
i__17825__auto___23656 = G__23657;
continue;
} else {
}
break;
}

var G__23590 = args23588.length;
switch (G__23590) {
case 1:
return sablono.core.week_field23587.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.week_field23587.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23588.length)].join('')));

}
});

sablono.core.week_field23587.cljs$core$IFn$_invoke$arity$1 = (function (name__23444__auto__){
return sablono.core.week_field23587.call(null,name__23444__auto__,null);
});

sablono.core.week_field23587.cljs$core$IFn$_invoke$arity$2 = (function (name__23444__auto__,value__23445__auto__){
return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"week","week",314058249,null))].join(''),name__23444__auto__,value__23445__auto__);
});

sablono.core.week_field23587.cljs$lang$maxFixedArity = 2;

sablono.core.week_field = sablono.core.wrap_attrs.call(null,sablono.core.week_field23587);
sablono.core.file_upload = sablono.core.file_field;
/**
 * Creates a check box.
 */
sablono.core.check_box23659 = (function sablono$core$check_box23659(var_args){
var args23660 = [];
var len__17824__auto___23663 = arguments.length;
var i__17825__auto___23664 = (0);
while(true){
if((i__17825__auto___23664 < len__17824__auto___23663)){
args23660.push((arguments[i__17825__auto___23664]));

var G__23665 = (i__17825__auto___23664 + (1));
i__17825__auto___23664 = G__23665;
continue;
} else {
}
break;
}

var G__23662 = args23660.length;
switch (G__23662) {
case 1:
return sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23660.length)].join('')));

}
});

sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$1 = (function (name){
return sablono.core.check_box23659.call(null,name,null);
});

sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$2 = (function (name,checked_QMARK_){
return sablono.core.check_box23659.call(null,name,checked_QMARK_,"true");
});

sablono.core.check_box23659.cljs$core$IFn$_invoke$arity$3 = (function (name,checked_QMARK_,value){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",556931961),new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"type","type",1174270348),"checkbox",new cljs.core.Keyword(null,"name","name",1843675177),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",-1388402092),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",305978217),value,new cljs.core.Keyword(null,"checked","checked",-50955819),checked_QMARK_], null)], null);
});

sablono.core.check_box23659.cljs$lang$maxFixedArity = 3;

sablono.core.check_box = sablono.core.wrap_attrs.call(null,sablono.core.check_box23659);
/**
 * Creates a radio button.
 */
sablono.core.radio_button23667 = (function sablono$core$radio_button23667(var_args){
var args23668 = [];
var len__17824__auto___23671 = arguments.length;
var i__17825__auto___23672 = (0);
while(true){
if((i__17825__auto___23672 < len__17824__auto___23671)){
args23668.push((arguments[i__17825__auto___23672]));

var G__23673 = (i__17825__auto___23672 + (1));
i__17825__auto___23672 = G__23673;
continue;
} else {
}
break;
}

var G__23670 = args23668.length;
switch (G__23670) {
case 1:
return sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23668.length)].join('')));

}
});

sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$1 = (function (group){
return sablono.core.radio_button23667.call(null,group,null);
});

sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$2 = (function (group,checked_QMARK_){
return sablono.core.radio_button23667.call(null,group,checked_QMARK_,"true");
});

sablono.core.radio_button23667.cljs$core$IFn$_invoke$arity$3 = (function (group,checked_QMARK_,value){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",556931961),new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"type","type",1174270348),"radio",new cljs.core.Keyword(null,"name","name",1843675177),sablono.core.make_name.call(null,group),new cljs.core.Keyword(null,"id","id",-1388402092),sablono.core.make_id.call(null,[cljs.core.str(sablono.util.as_str.call(null,group)),cljs.core.str("-"),cljs.core.str(sablono.util.as_str.call(null,value))].join('')),new cljs.core.Keyword(null,"value","value",305978217),value,new cljs.core.Keyword(null,"checked","checked",-50955819),checked_QMARK_], null)], null);
});

sablono.core.radio_button23667.cljs$lang$maxFixedArity = 3;

sablono.core.radio_button = sablono.core.wrap_attrs.call(null,sablono.core.radio_button23667);
sablono.core.hash_key = (function sablono$core$hash_key(x){
return goog.string.hashCode(cljs.core.pr_str.call(null,x));
});
/**
 * Creates a seq of option tags from a collection.
 */
sablono.core.select_options23675 = (function sablono$core$select_options23675(coll){
var iter__17538__auto__ = (function sablono$core$select_options23675_$_iter__23684(s__23685){
return (new cljs.core.LazySeq(null,(function (){
var s__23685__$1 = s__23685;
while(true){
var temp__4425__auto__ = cljs.core.seq.call(null,s__23685__$1);
if(temp__4425__auto__){
var s__23685__$2 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,s__23685__$2)){
var c__17536__auto__ = cljs.core.chunk_first.call(null,s__23685__$2);
var size__17537__auto__ = cljs.core.count.call(null,c__17536__auto__);
var b__23687 = cljs.core.chunk_buffer.call(null,size__17537__auto__);
if((function (){var i__23686 = (0);
while(true){
if((i__23686 < size__17537__auto__)){
var x = cljs.core._nth.call(null,c__17536__auto__,i__23686);
cljs.core.chunk_append.call(null,b__23687,((cljs.core.sequential_QMARK_.call(null,x))?(function (){var vec__23690 = x;
var text = cljs.core.nth.call(null,vec__23690,(0),null);
var val = cljs.core.nth.call(null,vec__23690,(1),null);
var disabled_QMARK_ = cljs.core.nth.call(null,vec__23690,(2),null);
var disabled_QMARK___$1 = cljs.core.boolean$.call(null,disabled_QMARK_);
if(cljs.core.sequential_QMARK_.call(null,val)){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"optgroup","optgroup",1738282218),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,text),new cljs.core.Keyword(null,"label","label",1718410804),text], null),sablono$core$select_options23675.call(null,val)], null);
} else {
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",65132272),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"disabled","disabled",-1529784218),disabled_QMARK___$1,new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,val),new cljs.core.Keyword(null,"value","value",305978217),val], null),text], null);
}
})():new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",65132272),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,x),new cljs.core.Keyword(null,"value","value",305978217),x], null),x], null)));

var G__23692 = (i__23686 + (1));
i__23686 = G__23692;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23687),sablono$core$select_options23675_$_iter__23684.call(null,cljs.core.chunk_rest.call(null,s__23685__$2)));
} else {
return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__23687),null);
}
} else {
var x = cljs.core.first.call(null,s__23685__$2);
return cljs.core.cons.call(null,((cljs.core.sequential_QMARK_.call(null,x))?(function (){var vec__23691 = x;
var text = cljs.core.nth.call(null,vec__23691,(0),null);
var val = cljs.core.nth.call(null,vec__23691,(1),null);
var disabled_QMARK_ = cljs.core.nth.call(null,vec__23691,(2),null);
var disabled_QMARK___$1 = cljs.core.boolean$.call(null,disabled_QMARK_);
if(cljs.core.sequential_QMARK_.call(null,val)){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"optgroup","optgroup",1738282218),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,text),new cljs.core.Keyword(null,"label","label",1718410804),text], null),sablono$core$select_options23675.call(null,val)], null);
} else {
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",65132272),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"disabled","disabled",-1529784218),disabled_QMARK___$1,new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,val),new cljs.core.Keyword(null,"value","value",305978217),val], null),text], null);
}
})():new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",65132272),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",-1516042587),sablono.core.hash_key.call(null,x),new cljs.core.Keyword(null,"value","value",305978217),x], null),x], null)),sablono$core$select_options23675_$_iter__23684.call(null,cljs.core.rest.call(null,s__23685__$2)));
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__17538__auto__.call(null,coll);
});

sablono.core.select_options = sablono.core.wrap_attrs.call(null,sablono.core.select_options23675);
/**
 * Creates a drop-down box using the <select> tag.
 */
sablono.core.drop_down23693 = (function sablono$core$drop_down23693(var_args){
var args23694 = [];
var len__17824__auto___23697 = arguments.length;
var i__17825__auto___23698 = (0);
while(true){
if((i__17825__auto___23698 < len__17824__auto___23697)){
args23694.push((arguments[i__17825__auto___23698]));

var G__23699 = (i__17825__auto___23698 + (1));
i__17825__auto___23698 = G__23699;
continue;
} else {
}
break;
}

var G__23696 = args23694.length;
switch (G__23696) {
case 2:
return sablono.core.drop_down23693.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return sablono.core.drop_down23693.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23694.length)].join('')));

}
});

sablono.core.drop_down23693.cljs$core$IFn$_invoke$arity$2 = (function (name,options){
return sablono.core.drop_down23693.call(null,name,options,null);
});

sablono.core.drop_down23693.cljs$core$IFn$_invoke$arity$3 = (function (name,options,selected){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"select","select",1147833503),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"name","name",1843675177),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",-1388402092),sablono.core.make_id.call(null,name)], null),sablono.core.select_options.call(null,options,selected)], null);
});

sablono.core.drop_down23693.cljs$lang$maxFixedArity = 3;

sablono.core.drop_down = sablono.core.wrap_attrs.call(null,sablono.core.drop_down23693);
/**
 * Creates a text area element.
 */
sablono.core.text_area23701 = (function sablono$core$text_area23701(var_args){
var args23702 = [];
var len__17824__auto___23705 = arguments.length;
var i__17825__auto___23706 = (0);
while(true){
if((i__17825__auto___23706 < len__17824__auto___23705)){
args23702.push((arguments[i__17825__auto___23706]));

var G__23707 = (i__17825__auto___23706 + (1));
i__17825__auto___23706 = G__23707;
continue;
} else {
}
break;
}

var G__23704 = args23702.length;
switch (G__23704) {
case 1:
return sablono.core.text_area23701.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return sablono.core.text_area23701.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args23702.length)].join('')));

}
});

sablono.core.text_area23701.cljs$core$IFn$_invoke$arity$1 = (function (name){
return sablono.core.text_area23701.call(null,name,null);
});

sablono.core.text_area23701.cljs$core$IFn$_invoke$arity$2 = (function (name,value){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"textarea","textarea",-650375824),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"name","name",1843675177),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",-1388402092),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",305978217),value], null)], null);
});

sablono.core.text_area23701.cljs$lang$maxFixedArity = 2;

sablono.core.text_area = sablono.core.wrap_attrs.call(null,sablono.core.text_area23701);
/**
 * Creates a label for an input field with the supplied name.
 */
sablono.core.label23709 = (function sablono$core$label23709(name,text){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"label","label",1718410804),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"htmlFor","htmlFor",-1050291720),sablono.core.make_id.call(null,name)], null),text], null);
});

sablono.core.label = sablono.core.wrap_attrs.call(null,sablono.core.label23709);
/**
 * Creates a submit button.
 */
sablono.core.submit_button23710 = (function sablono$core$submit_button23710(text){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",556931961),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),"submit",new cljs.core.Keyword(null,"value","value",305978217),text], null)], null);
});

sablono.core.submit_button = sablono.core.wrap_attrs.call(null,sablono.core.submit_button23710);
/**
 * Creates a form reset button.
 */
sablono.core.reset_button23711 = (function sablono$core$reset_button23711(text){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",556931961),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),"reset",new cljs.core.Keyword(null,"value","value",305978217),text], null)], null);
});

sablono.core.reset_button = sablono.core.wrap_attrs.call(null,sablono.core.reset_button23711);
/**
 * Create a form that points to a particular method and route.
 *   e.g. (form-to [:put "/post"]
 *       ...)
 */
sablono.core.form_to23712 = (function sablono$core$form_to23712(var_args){
var args__17831__auto__ = [];
var len__17824__auto___23717 = arguments.length;
var i__17825__auto___23718 = (0);
while(true){
if((i__17825__auto___23718 < len__17824__auto___23717)){
args__17831__auto__.push((arguments[i__17825__auto___23718]));

var G__23719 = (i__17825__auto___23718 + (1));
i__17825__auto___23718 = G__23719;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((1) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((1)),(0))):null);
return sablono.core.form_to23712.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__17832__auto__);
});

sablono.core.form_to23712.cljs$core$IFn$_invoke$arity$variadic = (function (p__23715,body){
var vec__23716 = p__23715;
var method = cljs.core.nth.call(null,vec__23716,(0),null);
var action = cljs.core.nth.call(null,vec__23716,(1),null);
var method_str = clojure.string.upper_case.call(null,cljs.core.name.call(null,method));
var action_uri = sablono.util.to_uri.call(null,action);
return cljs.core.vec.call(null,cljs.core.concat.call(null,((cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"get","get",1683182755),null,new cljs.core.Keyword(null,"post","post",269697687),null], null), null),method))?new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"form","form",-1624062471),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"method","method",55703592),method_str,new cljs.core.Keyword(null,"action","action",-811238024),action_uri], null)], null):new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"form","form",-1624062471),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"method","method",55703592),"POST",new cljs.core.Keyword(null,"action","action",-811238024),action_uri], null),sablono.core.hidden_field.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"key","key",-1516042587),(3735928559)], null),"_method",method_str)], null)),body));
});

sablono.core.form_to23712.cljs$lang$maxFixedArity = (1);

sablono.core.form_to23712.cljs$lang$applyTo = (function (seq23713){
var G__23714 = cljs.core.first.call(null,seq23713);
var seq23713__$1 = cljs.core.next.call(null,seq23713);
return sablono.core.form_to23712.cljs$core$IFn$_invoke$arity$variadic(G__23714,seq23713__$1);
});

sablono.core.form_to = sablono.core.wrap_attrs.call(null,sablono.core.form_to23712);

//# sourceMappingURL=core.js.map?rel=1454621291575