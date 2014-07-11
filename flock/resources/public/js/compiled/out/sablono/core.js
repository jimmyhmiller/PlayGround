// Compiled by ClojureScript 0.0-2202
goog.provide('sablono.core');
goog.require('cljs.core');
goog.require('clojure.walk');
goog.require('clojure.string');
goog.require('sablono.util');
goog.require('goog.dom');
goog.require('goog.dom');
goog.require('sablono.interpreter');
goog.require('sablono.interpreter');
goog.require('sablono.util');
goog.require('clojure.walk');
goog.require('clojure.string');
/**
* Add an optional attribute argument to a function that returns a element vector.
*/
sablono.core.wrap_attrs = (function wrap_attrs(func){return (function() { 
var G__15507__delegate = function (args){if(cljs.core.map_QMARK_.call(null,cljs.core.first.call(null,args)))
{var vec__15506 = cljs.core.apply.call(null,func,cljs.core.rest.call(null,args));var tag = cljs.core.nth.call(null,vec__15506,0,null);var body = cljs.core.nthnext.call(null,vec__15506,1);if(cljs.core.map_QMARK_.call(null,cljs.core.first.call(null,body)))
{return cljs.core.apply.call(null,cljs.core.vector,tag,cljs.core.merge.call(null,cljs.core.first.call(null,body),cljs.core.first.call(null,args)),cljs.core.rest.call(null,body));
} else
{return cljs.core.apply.call(null,cljs.core.vector,tag,cljs.core.first.call(null,args),body);
}
} else
{return cljs.core.apply.call(null,func,args);
}
};
var G__15507 = function (var_args){
var args = null;if (arguments.length > 0) {
  args = cljs.core.array_seq(Array.prototype.slice.call(arguments, 0),0);} 
return G__15507__delegate.call(this,args);};
G__15507.cljs$lang$maxFixedArity = 0;
G__15507.cljs$lang$applyTo = (function (arglist__15508){
var args = cljs.core.seq(arglist__15508);
return G__15507__delegate(args);
});
G__15507.cljs$core$IFn$_invoke$arity$variadic = G__15507__delegate;
return G__15507;
})()
;
});
sablono.core.update_arglists = (function update_arglists(arglists){var iter__8591__auto__ = (function iter__15513(s__15514){return (new cljs.core.LazySeq(null,(function (){var s__15514__$1 = s__15514;while(true){
var temp__4126__auto__ = cljs.core.seq.call(null,s__15514__$1);if(temp__4126__auto__)
{var s__15514__$2 = temp__4126__auto__;if(cljs.core.chunked_seq_QMARK_.call(null,s__15514__$2))
{var c__8589__auto__ = cljs.core.chunk_first.call(null,s__15514__$2);var size__8590__auto__ = cljs.core.count.call(null,c__8589__auto__);var b__15516 = cljs.core.chunk_buffer.call(null,size__8590__auto__);if((function (){var i__15515 = 0;while(true){
if((i__15515 < size__8590__auto__))
{var args = cljs.core._nth.call(null,c__8589__auto__,i__15515);cljs.core.chunk_append.call(null,b__15516,cljs.core.vec.call(null,cljs.core.cons.call(null,new cljs.core.Symbol(null,"attr-map?","attr-map?",-1682549128,null),args)));
{
var G__15517 = (i__15515 + 1);
i__15515 = G__15517;
continue;
}
} else
{return true;
}
break;
}
})())
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15516),iter__15513.call(null,cljs.core.chunk_rest.call(null,s__15514__$2)));
} else
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15516),null);
}
} else
{var args = cljs.core.first.call(null,s__15514__$2);return cljs.core.cons.call(null,cljs.core.vec.call(null,cljs.core.cons.call(null,new cljs.core.Symbol(null,"attr-map?","attr-map?",-1682549128,null),args)),iter__15513.call(null,cljs.core.rest.call(null,s__15514__$2)));
}
} else
{return null;
}
break;
}
}),null,null));
});return iter__8591__auto__.call(null,arglists);
});
/**
* Render the React `component` as an HTML string.
*/
sablono.core.render = (function render(component){return React.renderComponentToString(component);
});
/**
* Include a list of external stylesheet files.
* @param {...*} var_args
*/
sablono.core.include_css = (function() { 
var include_css__delegate = function (styles){var iter__8591__auto__ = (function iter__15522(s__15523){return (new cljs.core.LazySeq(null,(function (){var s__15523__$1 = s__15523;while(true){
var temp__4126__auto__ = cljs.core.seq.call(null,s__15523__$1);if(temp__4126__auto__)
{var s__15523__$2 = temp__4126__auto__;if(cljs.core.chunked_seq_QMARK_.call(null,s__15523__$2))
{var c__8589__auto__ = cljs.core.chunk_first.call(null,s__15523__$2);var size__8590__auto__ = cljs.core.count.call(null,c__8589__auto__);var b__15525 = cljs.core.chunk_buffer.call(null,size__8590__auto__);if((function (){var i__15524 = 0;while(true){
if((i__15524 < size__8590__auto__))
{var style = cljs.core._nth.call(null,c__8589__auto__,i__15524);cljs.core.chunk_append.call(null,b__15525,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"link","link",1017226092),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"type","type",1017479852),"text/css",new cljs.core.Keyword(null,"href","href",1017115293),sablono.util.as_str.call(null,style),new cljs.core.Keyword(null,"rel","rel",1014017035),"stylesheet"], null)], null));
{
var G__15526 = (i__15524 + 1);
i__15524 = G__15526;
continue;
}
} else
{return true;
}
break;
}
})())
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15525),iter__15522.call(null,cljs.core.chunk_rest.call(null,s__15523__$2)));
} else
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15525),null);
}
} else
{var style = cljs.core.first.call(null,s__15523__$2);return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"link","link",1017226092),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"type","type",1017479852),"text/css",new cljs.core.Keyword(null,"href","href",1017115293),sablono.util.as_str.call(null,style),new cljs.core.Keyword(null,"rel","rel",1014017035),"stylesheet"], null)], null),iter__15522.call(null,cljs.core.rest.call(null,s__15523__$2)));
}
} else
{return null;
}
break;
}
}),null,null));
});return iter__8591__auto__.call(null,styles);
};
var include_css = function (var_args){
var styles = null;if (arguments.length > 0) {
  styles = cljs.core.array_seq(Array.prototype.slice.call(arguments, 0),0);} 
return include_css__delegate.call(this,styles);};
include_css.cljs$lang$maxFixedArity = 0;
include_css.cljs$lang$applyTo = (function (arglist__15527){
var styles = cljs.core.seq(arglist__15527);
return include_css__delegate(styles);
});
include_css.cljs$core$IFn$_invoke$arity$variadic = include_css__delegate;
return include_css;
})()
;
/**
* Include the JavaScript library at `src`.
*/
sablono.core.include_js = (function include_js(src){return goog.dom.appendChild(goog.dom.getDocument().body,goog.dom.createDom("script",{"src": src}));
});
/**
* Include Facebook's React JavaScript library.
*/
sablono.core.include_react = (function include_react(){return sablono.core.include_js.call(null,"http://fb.me/react-0.9.0.js");
});
/**
* Wraps some content in a HTML hyperlink with the supplied URL.
* @param {...*} var_args
*/
sablono.core.link_to15528 = (function() { 
var link_to15528__delegate = function (url,content){return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"a","a",1013904339),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"href","href",1017115293),sablono.util.as_str.call(null,url)], null),content], null);
};
var link_to15528 = function (url,var_args){
var content = null;if (arguments.length > 1) {
  content = cljs.core.array_seq(Array.prototype.slice.call(arguments, 1),0);} 
return link_to15528__delegate.call(this,url,content);};
link_to15528.cljs$lang$maxFixedArity = 1;
link_to15528.cljs$lang$applyTo = (function (arglist__15529){
var url = cljs.core.first(arglist__15529);
var content = cljs.core.rest(arglist__15529);
return link_to15528__delegate(url,content);
});
link_to15528.cljs$core$IFn$_invoke$arity$variadic = link_to15528__delegate;
return link_to15528;
})()
;
sablono.core.link_to = sablono.core.wrap_attrs.call(null,sablono.core.link_to15528);
/**
* Wraps some content in a HTML hyperlink with the supplied e-mail
* address. If no content provided use the e-mail address as content.
* @param {...*} var_args
*/
sablono.core.mail_to15530 = (function() { 
var mail_to15530__delegate = function (e_mail,p__15531){var vec__15533 = p__15531;var content = cljs.core.nth.call(null,vec__15533,0,null);return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"a","a",1013904339),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"href","href",1017115293),[cljs.core.str("mailto:"),cljs.core.str(e_mail)].join('')], null),(function (){var or__7874__auto__ = content;if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{return e_mail;
}
})()], null);
};
var mail_to15530 = function (e_mail,var_args){
var p__15531 = null;if (arguments.length > 1) {
  p__15531 = cljs.core.array_seq(Array.prototype.slice.call(arguments, 1),0);} 
return mail_to15530__delegate.call(this,e_mail,p__15531);};
mail_to15530.cljs$lang$maxFixedArity = 1;
mail_to15530.cljs$lang$applyTo = (function (arglist__15534){
var e_mail = cljs.core.first(arglist__15534);
var p__15531 = cljs.core.rest(arglist__15534);
return mail_to15530__delegate(e_mail,p__15531);
});
mail_to15530.cljs$core$IFn$_invoke$arity$variadic = mail_to15530__delegate;
return mail_to15530;
})()
;
sablono.core.mail_to = sablono.core.wrap_attrs.call(null,sablono.core.mail_to15530);
/**
* Wrap a collection in an unordered list.
*/
sablono.core.unordered_list15535 = (function unordered_list15535(coll){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"ul","ul",1013907977),(function (){var iter__8591__auto__ = (function iter__15540(s__15541){return (new cljs.core.LazySeq(null,(function (){var s__15541__$1 = s__15541;while(true){
var temp__4126__auto__ = cljs.core.seq.call(null,s__15541__$1);if(temp__4126__auto__)
{var s__15541__$2 = temp__4126__auto__;if(cljs.core.chunked_seq_QMARK_.call(null,s__15541__$2))
{var c__8589__auto__ = cljs.core.chunk_first.call(null,s__15541__$2);var size__8590__auto__ = cljs.core.count.call(null,c__8589__auto__);var b__15543 = cljs.core.chunk_buffer.call(null,size__8590__auto__);if((function (){var i__15542 = 0;while(true){
if((i__15542 < size__8590__auto__))
{var x = cljs.core._nth.call(null,c__8589__auto__,i__15542);cljs.core.chunk_append.call(null,b__15543,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",1013907695),x], null));
{
var G__15544 = (i__15542 + 1);
i__15542 = G__15544;
continue;
}
} else
{return true;
}
break;
}
})())
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15543),iter__15540.call(null,cljs.core.chunk_rest.call(null,s__15541__$2)));
} else
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15543),null);
}
} else
{var x = cljs.core.first.call(null,s__15541__$2);return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",1013907695),x], null),iter__15540.call(null,cljs.core.rest.call(null,s__15541__$2)));
}
} else
{return null;
}
break;
}
}),null,null));
});return iter__8591__auto__.call(null,coll);
})()], null);
});
sablono.core.unordered_list = sablono.core.wrap_attrs.call(null,sablono.core.unordered_list15535);
/**
* Wrap a collection in an ordered list.
*/
sablono.core.ordered_list15545 = (function ordered_list15545(coll){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"ol","ol",1013907791),(function (){var iter__8591__auto__ = (function iter__15550(s__15551){return (new cljs.core.LazySeq(null,(function (){var s__15551__$1 = s__15551;while(true){
var temp__4126__auto__ = cljs.core.seq.call(null,s__15551__$1);if(temp__4126__auto__)
{var s__15551__$2 = temp__4126__auto__;if(cljs.core.chunked_seq_QMARK_.call(null,s__15551__$2))
{var c__8589__auto__ = cljs.core.chunk_first.call(null,s__15551__$2);var size__8590__auto__ = cljs.core.count.call(null,c__8589__auto__);var b__15553 = cljs.core.chunk_buffer.call(null,size__8590__auto__);if((function (){var i__15552 = 0;while(true){
if((i__15552 < size__8590__auto__))
{var x = cljs.core._nth.call(null,c__8589__auto__,i__15552);cljs.core.chunk_append.call(null,b__15553,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",1013907695),x], null));
{
var G__15554 = (i__15552 + 1);
i__15552 = G__15554;
continue;
}
} else
{return true;
}
break;
}
})())
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15553),iter__15550.call(null,cljs.core.chunk_rest.call(null,s__15551__$2)));
} else
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15553),null);
}
} else
{var x = cljs.core.first.call(null,s__15551__$2);return cljs.core.cons.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"li","li",1013907695),x], null),iter__15550.call(null,cljs.core.rest.call(null,s__15551__$2)));
}
} else
{return null;
}
break;
}
}),null,null));
});return iter__8591__auto__.call(null,coll);
})()], null);
});
sablono.core.ordered_list = sablono.core.wrap_attrs.call(null,sablono.core.ordered_list15545);
/**
* Create an image element.
*/
sablono.core.image15555 = (function() {
var image15555 = null;
var image15555__1 = (function (src){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"img","img",1014008629),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"src","src",1014018390),sablono.util.as_str.call(null,src)], null)], null);
});
var image15555__2 = (function (src,alt){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"img","img",1014008629),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"src","src",1014018390),sablono.util.as_str.call(null,src),new cljs.core.Keyword(null,"alt","alt",1014000923),alt], null)], null);
});
image15555 = function(src,alt){
switch(arguments.length){
case 1:
return image15555__1.call(this,src);
case 2:
return image15555__2.call(this,src,alt);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
image15555.cljs$core$IFn$_invoke$arity$1 = image15555__1;
image15555.cljs$core$IFn$_invoke$arity$2 = image15555__2;
return image15555;
})()
;
sablono.core.image = sablono.core.wrap_attrs.call(null,sablono.core.image15555);
sablono.core._STAR_group_STAR_ = cljs.core.PersistentVector.EMPTY;
/**
* Create a field name from the supplied argument the current field group.
*/
sablono.core.make_name = (function make_name(name){return cljs.core.reduce.call(null,(function (p1__15556_SHARP_,p2__15557_SHARP_){return [cljs.core.str(p1__15556_SHARP_),cljs.core.str("["),cljs.core.str(p2__15557_SHARP_),cljs.core.str("]")].join('');
}),cljs.core.conj.call(null,sablono.core._STAR_group_STAR_,sablono.util.as_str.call(null,name)));
});
/**
* Create a field id from the supplied argument and current field group.
*/
sablono.core.make_id = (function make_id(name){return cljs.core.reduce.call(null,(function (p1__15558_SHARP_,p2__15559_SHARP_){return [cljs.core.str(p1__15558_SHARP_),cljs.core.str("-"),cljs.core.str(p2__15559_SHARP_)].join('');
}),cljs.core.conj.call(null,sablono.core._STAR_group_STAR_,sablono.util.as_str.call(null,name)));
});
/**
* Creates a new <input> element.
*/
sablono.core.input_field_STAR_ = (function input_field_STAR_(type,name,value){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",1114262332),new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"type","type",1017479852),type,new cljs.core.Keyword(null,"name","name",1017277949),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",1013907597),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",1125876963),value], null)], null);
});
/**
* Creates a color input field.
*/
sablono.core.color_field15560 = (function() {
var color_field15560 = null;
var color_field15560__1 = (function (name__9334__auto__){return color_field15560.call(null,name__9334__auto__,null);
});
var color_field15560__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"color","color",-1545688804,null))].join(''),name__9334__auto__,value__9335__auto__);
});
color_field15560 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return color_field15560__1.call(this,name__9334__auto__);
case 2:
return color_field15560__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
color_field15560.cljs$core$IFn$_invoke$arity$1 = color_field15560__1;
color_field15560.cljs$core$IFn$_invoke$arity$2 = color_field15560__2;
return color_field15560;
})()
;
sablono.core.color_field = sablono.core.wrap_attrs.call(null,sablono.core.color_field15560);
/**
* Creates a date input field.
*/
sablono.core.date_field15561 = (function() {
var date_field15561 = null;
var date_field15561__1 = (function (name__9334__auto__){return date_field15561.call(null,name__9334__auto__,null);
});
var date_field15561__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"date","date",-1637455513,null))].join(''),name__9334__auto__,value__9335__auto__);
});
date_field15561 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return date_field15561__1.call(this,name__9334__auto__);
case 2:
return date_field15561__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
date_field15561.cljs$core$IFn$_invoke$arity$1 = date_field15561__1;
date_field15561.cljs$core$IFn$_invoke$arity$2 = date_field15561__2;
return date_field15561;
})()
;
sablono.core.date_field = sablono.core.wrap_attrs.call(null,sablono.core.date_field15561);
/**
* Creates a datetime input field.
*/
sablono.core.datetime_field15562 = (function() {
var datetime_field15562 = null;
var datetime_field15562__1 = (function (name__9334__auto__){return datetime_field15562.call(null,name__9334__auto__,null);
});
var datetime_field15562__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"datetime","datetime",153171252,null))].join(''),name__9334__auto__,value__9335__auto__);
});
datetime_field15562 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return datetime_field15562__1.call(this,name__9334__auto__);
case 2:
return datetime_field15562__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
datetime_field15562.cljs$core$IFn$_invoke$arity$1 = datetime_field15562__1;
datetime_field15562.cljs$core$IFn$_invoke$arity$2 = datetime_field15562__2;
return datetime_field15562;
})()
;
sablono.core.datetime_field = sablono.core.wrap_attrs.call(null,sablono.core.datetime_field15562);
/**
* Creates a datetime-local input field.
*/
sablono.core.datetime_local_field15563 = (function() {
var datetime_local_field15563 = null;
var datetime_local_field15563__1 = (function (name__9334__auto__){return datetime_local_field15563.call(null,name__9334__auto__,null);
});
var datetime_local_field15563__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"datetime-local","datetime-local",1631019090,null))].join(''),name__9334__auto__,value__9335__auto__);
});
datetime_local_field15563 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return datetime_local_field15563__1.call(this,name__9334__auto__);
case 2:
return datetime_local_field15563__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
datetime_local_field15563.cljs$core$IFn$_invoke$arity$1 = datetime_local_field15563__1;
datetime_local_field15563.cljs$core$IFn$_invoke$arity$2 = datetime_local_field15563__2;
return datetime_local_field15563;
})()
;
sablono.core.datetime_local_field = sablono.core.wrap_attrs.call(null,sablono.core.datetime_local_field15563);
/**
* Creates a email input field.
*/
sablono.core.email_field15564 = (function() {
var email_field15564 = null;
var email_field15564__1 = (function (name__9334__auto__){return email_field15564.call(null,name__9334__auto__,null);
});
var email_field15564__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"email","email",-1543912107,null))].join(''),name__9334__auto__,value__9335__auto__);
});
email_field15564 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return email_field15564__1.call(this,name__9334__auto__);
case 2:
return email_field15564__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
email_field15564.cljs$core$IFn$_invoke$arity$1 = email_field15564__1;
email_field15564.cljs$core$IFn$_invoke$arity$2 = email_field15564__2;
return email_field15564;
})()
;
sablono.core.email_field = sablono.core.wrap_attrs.call(null,sablono.core.email_field15564);
/**
* Creates a file input field.
*/
sablono.core.file_field15565 = (function() {
var file_field15565 = null;
var file_field15565__1 = (function (name__9334__auto__){return file_field15565.call(null,name__9334__auto__,null);
});
var file_field15565__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"file","file",-1637388491,null))].join(''),name__9334__auto__,value__9335__auto__);
});
file_field15565 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return file_field15565__1.call(this,name__9334__auto__);
case 2:
return file_field15565__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
file_field15565.cljs$core$IFn$_invoke$arity$1 = file_field15565__1;
file_field15565.cljs$core$IFn$_invoke$arity$2 = file_field15565__2;
return file_field15565;
})()
;
sablono.core.file_field = sablono.core.wrap_attrs.call(null,sablono.core.file_field15565);
/**
* Creates a hidden input field.
*/
sablono.core.hidden_field15566 = (function() {
var hidden_field15566 = null;
var hidden_field15566__1 = (function (name__9334__auto__){return hidden_field15566.call(null,name__9334__auto__,null);
});
var hidden_field15566__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"hidden","hidden",1436948323,null))].join(''),name__9334__auto__,value__9335__auto__);
});
hidden_field15566 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return hidden_field15566__1.call(this,name__9334__auto__);
case 2:
return hidden_field15566__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
hidden_field15566.cljs$core$IFn$_invoke$arity$1 = hidden_field15566__1;
hidden_field15566.cljs$core$IFn$_invoke$arity$2 = hidden_field15566__2;
return hidden_field15566;
})()
;
sablono.core.hidden_field = sablono.core.wrap_attrs.call(null,sablono.core.hidden_field15566);
/**
* Creates a month input field.
*/
sablono.core.month_field15567 = (function() {
var month_field15567 = null;
var month_field15567__1 = (function (name__9334__auto__){return month_field15567.call(null,name__9334__auto__,null);
});
var month_field15567__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"month","month",-1536451527,null))].join(''),name__9334__auto__,value__9335__auto__);
});
month_field15567 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return month_field15567__1.call(this,name__9334__auto__);
case 2:
return month_field15567__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
month_field15567.cljs$core$IFn$_invoke$arity$1 = month_field15567__1;
month_field15567.cljs$core$IFn$_invoke$arity$2 = month_field15567__2;
return month_field15567;
})()
;
sablono.core.month_field = sablono.core.wrap_attrs.call(null,sablono.core.month_field15567);
/**
* Creates a number input field.
*/
sablono.core.number_field15568 = (function() {
var number_field15568 = null;
var number_field15568__1 = (function (name__9334__auto__){return number_field15568.call(null,name__9334__auto__,null);
});
var number_field15568__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"number","number",1620071682,null))].join(''),name__9334__auto__,value__9335__auto__);
});
number_field15568 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return number_field15568__1.call(this,name__9334__auto__);
case 2:
return number_field15568__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
number_field15568.cljs$core$IFn$_invoke$arity$1 = number_field15568__1;
number_field15568.cljs$core$IFn$_invoke$arity$2 = number_field15568__2;
return number_field15568;
})()
;
sablono.core.number_field = sablono.core.wrap_attrs.call(null,sablono.core.number_field15568);
/**
* Creates a password input field.
*/
sablono.core.password_field15569 = (function() {
var password_field15569 = null;
var password_field15569__1 = (function (name__9334__auto__){return password_field15569.call(null,name__9334__auto__,null);
});
var password_field15569__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"password","password",-423545772,null))].join(''),name__9334__auto__,value__9335__auto__);
});
password_field15569 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return password_field15569__1.call(this,name__9334__auto__);
case 2:
return password_field15569__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
password_field15569.cljs$core$IFn$_invoke$arity$1 = password_field15569__1;
password_field15569.cljs$core$IFn$_invoke$arity$2 = password_field15569__2;
return password_field15569;
})()
;
sablono.core.password_field = sablono.core.wrap_attrs.call(null,sablono.core.password_field15569);
/**
* Creates a range input field.
*/
sablono.core.range_field15570 = (function() {
var range_field15570 = null;
var range_field15570__1 = (function (name__9334__auto__){return range_field15570.call(null,name__9334__auto__,null);
});
var range_field15570__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"range","range",-1532251402,null))].join(''),name__9334__auto__,value__9335__auto__);
});
range_field15570 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return range_field15570__1.call(this,name__9334__auto__);
case 2:
return range_field15570__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
range_field15570.cljs$core$IFn$_invoke$arity$1 = range_field15570__1;
range_field15570.cljs$core$IFn$_invoke$arity$2 = range_field15570__2;
return range_field15570;
})()
;
sablono.core.range_field = sablono.core.wrap_attrs.call(null,sablono.core.range_field15570);
/**
* Creates a search input field.
*/
sablono.core.search_field15571 = (function() {
var search_field15571 = null;
var search_field15571__1 = (function (name__9334__auto__){return search_field15571.call(null,name__9334__auto__,null);
});
var search_field15571__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"search","search",1748098913,null))].join(''),name__9334__auto__,value__9335__auto__);
});
search_field15571 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return search_field15571__1.call(this,name__9334__auto__);
case 2:
return search_field15571__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
search_field15571.cljs$core$IFn$_invoke$arity$1 = search_field15571__1;
search_field15571.cljs$core$IFn$_invoke$arity$2 = search_field15571__2;
return search_field15571;
})()
;
sablono.core.search_field = sablono.core.wrap_attrs.call(null,sablono.core.search_field15571);
/**
* Creates a tel input field.
*/
sablono.core.tel_field15572 = (function() {
var tel_field15572 = null;
var tel_field15572__1 = (function (name__9334__auto__){return tel_field15572.call(null,name__9334__auto__,null);
});
var tel_field15572__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"tel","tel",-1640416812,null))].join(''),name__9334__auto__,value__9335__auto__);
});
tel_field15572 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return tel_field15572__1.call(this,name__9334__auto__);
case 2:
return tel_field15572__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
tel_field15572.cljs$core$IFn$_invoke$arity$1 = tel_field15572__1;
tel_field15572.cljs$core$IFn$_invoke$arity$2 = tel_field15572__2;
return tel_field15572;
})()
;
sablono.core.tel_field = sablono.core.wrap_attrs.call(null,sablono.core.tel_field15572);
/**
* Creates a text input field.
*/
sablono.core.text_field15573 = (function() {
var text_field15573 = null;
var text_field15573__1 = (function (name__9334__auto__){return text_field15573.call(null,name__9334__auto__,null);
});
var text_field15573__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"text","text",-1636974874,null))].join(''),name__9334__auto__,value__9335__auto__);
});
text_field15573 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return text_field15573__1.call(this,name__9334__auto__);
case 2:
return text_field15573__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
text_field15573.cljs$core$IFn$_invoke$arity$1 = text_field15573__1;
text_field15573.cljs$core$IFn$_invoke$arity$2 = text_field15573__2;
return text_field15573;
})()
;
sablono.core.text_field = sablono.core.wrap_attrs.call(null,sablono.core.text_field15573);
/**
* Creates a time input field.
*/
sablono.core.time_field15574 = (function() {
var time_field15574 = null;
var time_field15574__1 = (function (name__9334__auto__){return time_field15574.call(null,name__9334__auto__,null);
});
var time_field15574__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"time","time",-1636971386,null))].join(''),name__9334__auto__,value__9335__auto__);
});
time_field15574 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return time_field15574__1.call(this,name__9334__auto__);
case 2:
return time_field15574__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
time_field15574.cljs$core$IFn$_invoke$arity$1 = time_field15574__1;
time_field15574.cljs$core$IFn$_invoke$arity$2 = time_field15574__2;
return time_field15574;
})()
;
sablono.core.time_field = sablono.core.wrap_attrs.call(null,sablono.core.time_field15574);
/**
* Creates a url input field.
*/
sablono.core.url_field15575 = (function() {
var url_field15575 = null;
var url_field15575__1 = (function (name__9334__auto__){return url_field15575.call(null,name__9334__auto__,null);
});
var url_field15575__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"url","url",-1640415448,null))].join(''),name__9334__auto__,value__9335__auto__);
});
url_field15575 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return url_field15575__1.call(this,name__9334__auto__);
case 2:
return url_field15575__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
url_field15575.cljs$core$IFn$_invoke$arity$1 = url_field15575__1;
url_field15575.cljs$core$IFn$_invoke$arity$2 = url_field15575__2;
return url_field15575;
})()
;
sablono.core.url_field = sablono.core.wrap_attrs.call(null,sablono.core.url_field15575);
/**
* Creates a week input field.
*/
sablono.core.week_field15576 = (function() {
var week_field15576 = null;
var week_field15576__1 = (function (name__9334__auto__){return week_field15576.call(null,name__9334__auto__,null);
});
var week_field15576__2 = (function (name__9334__auto__,value__9335__auto__){return sablono.core.input_field_STAR_.call(null,[cljs.core.str(new cljs.core.Symbol(null,"week","week",-1636886099,null))].join(''),name__9334__auto__,value__9335__auto__);
});
week_field15576 = function(name__9334__auto__,value__9335__auto__){
switch(arguments.length){
case 1:
return week_field15576__1.call(this,name__9334__auto__);
case 2:
return week_field15576__2.call(this,name__9334__auto__,value__9335__auto__);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
week_field15576.cljs$core$IFn$_invoke$arity$1 = week_field15576__1;
week_field15576.cljs$core$IFn$_invoke$arity$2 = week_field15576__2;
return week_field15576;
})()
;
sablono.core.week_field = sablono.core.wrap_attrs.call(null,sablono.core.week_field15576);
sablono.core.file_upload = sablono.core.file_field;
/**
* Creates a check box.
*/
sablono.core.check_box15577 = (function() {
var check_box15577 = null;
var check_box15577__1 = (function (name){return check_box15577.call(null,name,null);
});
var check_box15577__2 = (function (name,checked_QMARK_){return check_box15577.call(null,name,checked_QMARK_,"true");
});
var check_box15577__3 = (function (name,checked_QMARK_,value){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",1114262332),new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"type","type",1017479852),"checkbox",new cljs.core.Keyword(null,"name","name",1017277949),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",1013907597),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",1125876963),value,new cljs.core.Keyword(null,"checked","checked",1756218137),checked_QMARK_], null)], null);
});
check_box15577 = function(name,checked_QMARK_,value){
switch(arguments.length){
case 1:
return check_box15577__1.call(this,name);
case 2:
return check_box15577__2.call(this,name,checked_QMARK_);
case 3:
return check_box15577__3.call(this,name,checked_QMARK_,value);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
check_box15577.cljs$core$IFn$_invoke$arity$1 = check_box15577__1;
check_box15577.cljs$core$IFn$_invoke$arity$2 = check_box15577__2;
check_box15577.cljs$core$IFn$_invoke$arity$3 = check_box15577__3;
return check_box15577;
})()
;
sablono.core.check_box = sablono.core.wrap_attrs.call(null,sablono.core.check_box15577);
/**
* Creates a radio button.
*/
sablono.core.radio_button15578 = (function() {
var radio_button15578 = null;
var radio_button15578__1 = (function (group){return radio_button15578.call(null,group,null);
});
var radio_button15578__2 = (function (group,checked_QMARK_){return radio_button15578.call(null,group,checked_QMARK_,"true");
});
var radio_button15578__3 = (function (group,checked_QMARK_,value){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",1114262332),new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"type","type",1017479852),"radio",new cljs.core.Keyword(null,"name","name",1017277949),sablono.core.make_name.call(null,group),new cljs.core.Keyword(null,"id","id",1013907597),sablono.core.make_id.call(null,[cljs.core.str(sablono.util.as_str.call(null,group)),cljs.core.str("-"),cljs.core.str(sablono.util.as_str.call(null,value))].join('')),new cljs.core.Keyword(null,"value","value",1125876963),value,new cljs.core.Keyword(null,"checked","checked",1756218137),checked_QMARK_], null)], null);
});
radio_button15578 = function(group,checked_QMARK_,value){
switch(arguments.length){
case 1:
return radio_button15578__1.call(this,group);
case 2:
return radio_button15578__2.call(this,group,checked_QMARK_);
case 3:
return radio_button15578__3.call(this,group,checked_QMARK_,value);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
radio_button15578.cljs$core$IFn$_invoke$arity$1 = radio_button15578__1;
radio_button15578.cljs$core$IFn$_invoke$arity$2 = radio_button15578__2;
radio_button15578.cljs$core$IFn$_invoke$arity$3 = radio_button15578__3;
return radio_button15578;
})()
;
sablono.core.radio_button = sablono.core.wrap_attrs.call(null,sablono.core.radio_button15578);
/**
* Creates a seq of option tags from a collection.
*/
sablono.core.select_options15579 = (function() {
var select_options15579 = null;
var select_options15579__1 = (function (coll){return select_options15579.call(null,coll,null);
});
var select_options15579__2 = (function (coll,selected){var iter__8591__auto__ = (function iter__15588(s__15589){return (new cljs.core.LazySeq(null,(function (){var s__15589__$1 = s__15589;while(true){
var temp__4126__auto__ = cljs.core.seq.call(null,s__15589__$1);if(temp__4126__auto__)
{var s__15589__$2 = temp__4126__auto__;if(cljs.core.chunked_seq_QMARK_.call(null,s__15589__$2))
{var c__8589__auto__ = cljs.core.chunk_first.call(null,s__15589__$2);var size__8590__auto__ = cljs.core.count.call(null,c__8589__auto__);var b__15591 = cljs.core.chunk_buffer.call(null,size__8590__auto__);if((function (){var i__15590 = 0;while(true){
if((i__15590 < size__8590__auto__))
{var x = cljs.core._nth.call(null,c__8589__auto__,i__15590);cljs.core.chunk_append.call(null,b__15591,((cljs.core.sequential_QMARK_.call(null,x))?(function (){var vec__15594 = x;var text = cljs.core.nth.call(null,vec__15594,0,null);var val = cljs.core.nth.call(null,vec__15594,1,null);var disabled_QMARK_ = cljs.core.nth.call(null,vec__15594,2,null);var disabled_QMARK___$1 = cljs.core.boolean$.call(null,disabled_QMARK_);if(cljs.core.sequential_QMARK_.call(null,val))
{return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"optgroup","optgroup",933131038),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"label","label",1116631654),text], null),select_options15579.call(null,val,selected)], null);
} else
{return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",4298734567),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"value","value",1125876963),val,new cljs.core.Keyword(null,"selected","selected",2205476365),cljs.core._EQ_.call(null,val,selected),new cljs.core.Keyword(null,"disabled","disabled",1284845038),disabled_QMARK___$1], null),text], null);
}
})():new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",4298734567),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"selected","selected",2205476365),cljs.core._EQ_.call(null,x,selected)], null),x], null)));
{
var G__15596 = (i__15590 + 1);
i__15590 = G__15596;
continue;
}
} else
{return true;
}
break;
}
})())
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15591),iter__15588.call(null,cljs.core.chunk_rest.call(null,s__15589__$2)));
} else
{return cljs.core.chunk_cons.call(null,cljs.core.chunk.call(null,b__15591),null);
}
} else
{var x = cljs.core.first.call(null,s__15589__$2);return cljs.core.cons.call(null,((cljs.core.sequential_QMARK_.call(null,x))?(function (){var vec__15595 = x;var text = cljs.core.nth.call(null,vec__15595,0,null);var val = cljs.core.nth.call(null,vec__15595,1,null);var disabled_QMARK_ = cljs.core.nth.call(null,vec__15595,2,null);var disabled_QMARK___$1 = cljs.core.boolean$.call(null,disabled_QMARK_);if(cljs.core.sequential_QMARK_.call(null,val))
{return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"optgroup","optgroup",933131038),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"label","label",1116631654),text], null),select_options15579.call(null,val,selected)], null);
} else
{return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",4298734567),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"value","value",1125876963),val,new cljs.core.Keyword(null,"selected","selected",2205476365),cljs.core._EQ_.call(null,val,selected),new cljs.core.Keyword(null,"disabled","disabled",1284845038),disabled_QMARK___$1], null),text], null);
}
})():new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"option","option",4298734567),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"selected","selected",2205476365),cljs.core._EQ_.call(null,x,selected)], null),x], null)),iter__15588.call(null,cljs.core.rest.call(null,s__15589__$2)));
}
} else
{return null;
}
break;
}
}),null,null));
});return iter__8591__auto__.call(null,coll);
});
select_options15579 = function(coll,selected){
switch(arguments.length){
case 1:
return select_options15579__1.call(this,coll);
case 2:
return select_options15579__2.call(this,coll,selected);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
select_options15579.cljs$core$IFn$_invoke$arity$1 = select_options15579__1;
select_options15579.cljs$core$IFn$_invoke$arity$2 = select_options15579__2;
return select_options15579;
})()
;
sablono.core.select_options = sablono.core.wrap_attrs.call(null,sablono.core.select_options15579);
/**
* Creates a drop-down box using the <select> tag.
*/
sablono.core.drop_down15597 = (function() {
var drop_down15597 = null;
var drop_down15597__2 = (function (name,options){return drop_down15597.call(null,name,options,null);
});
var drop_down15597__3 = (function (name,options,selected){return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"select","select",4402849902),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"name","name",1017277949),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",1013907597),sablono.core.make_id.call(null,name)], null),sablono.core.select_options.call(null,options,selected)], null);
});
drop_down15597 = function(name,options,selected){
switch(arguments.length){
case 2:
return drop_down15597__2.call(this,name,options);
case 3:
return drop_down15597__3.call(this,name,options,selected);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
drop_down15597.cljs$core$IFn$_invoke$arity$2 = drop_down15597__2;
drop_down15597.cljs$core$IFn$_invoke$arity$3 = drop_down15597__3;
return drop_down15597;
})()
;
sablono.core.drop_down = sablono.core.wrap_attrs.call(null,sablono.core.drop_down15597);
/**
* Creates a text area element.
*/
sablono.core.text_area15598 = (function() {
var text_area15598 = null;
var text_area15598__1 = (function (name){return text_area15598.call(null,name,null);
});
var text_area15598__2 = (function (name,value){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"textarea","textarea",4305627820),new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"name","name",1017277949),sablono.core.make_name.call(null,name),new cljs.core.Keyword(null,"id","id",1013907597),sablono.core.make_id.call(null,name),new cljs.core.Keyword(null,"value","value",1125876963),value], null)], null);
});
text_area15598 = function(name,value){
switch(arguments.length){
case 1:
return text_area15598__1.call(this,name);
case 2:
return text_area15598__2.call(this,name,value);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
text_area15598.cljs$core$IFn$_invoke$arity$1 = text_area15598__1;
text_area15598.cljs$core$IFn$_invoke$arity$2 = text_area15598__2;
return text_area15598;
})()
;
sablono.core.text_area = sablono.core.wrap_attrs.call(null,sablono.core.text_area15598);
/**
* Creates a label for an input field with the supplied name.
*/
sablono.core.label15599 = (function label15599(name,text){return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"label","label",1116631654),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"htmlFor","htmlFor",2249940112),sablono.core.make_id.call(null,name)], null),text], null);
});
sablono.core.label = sablono.core.wrap_attrs.call(null,sablono.core.label15599);
/**
* Creates a submit button.
*/
sablono.core.submit_button15600 = (function submit_button15600(text){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",1114262332),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1017479852),"submit",new cljs.core.Keyword(null,"value","value",1125876963),text], null)], null);
});
sablono.core.submit_button = sablono.core.wrap_attrs.call(null,sablono.core.submit_button15600);
/**
* Creates a form reset button.
*/
sablono.core.reset_button15601 = (function reset_button15601(text){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"input","input",1114262332),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1017479852),"reset",new cljs.core.Keyword(null,"value","value",1125876963),text], null)], null);
});
sablono.core.reset_button = sablono.core.wrap_attrs.call(null,sablono.core.reset_button15601);
/**
* Create a form that points to a particular method and route.
* e.g. (form-to [:put "/post"]
* ...)
* @param {...*} var_args
*/
sablono.core.form_to15602 = (function() { 
var form_to15602__delegate = function (p__15603,body){var vec__15605 = p__15603;var method = cljs.core.nth.call(null,vec__15605,0,null);var action = cljs.core.nth.call(null,vec__15605,1,null);var method_str = clojure.string.upper_case.call(null,cljs.core.name.call(null,method));var action_uri = sablono.util.to_uri.call(null,action);return cljs.core.vec.call(null,cljs.core.concat.call(null,((cljs.core.contains_QMARK_.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"get","get",1014006472),null,new cljs.core.Keyword(null,"post","post",1017351186),null], null), null),method))?new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"form","form",1017053238),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"method","method",4231316563),method_str,new cljs.core.Keyword(null,"action","action",3885920680),action_uri], null)], null):new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"form","form",1017053238),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"method","method",4231316563),"POST",new cljs.core.Keyword(null,"action","action",3885920680),action_uri], null),sablono.core.hidden_field.call(null,"_method",method_str)], null)),body));
};
var form_to15602 = function (p__15603,var_args){
var body = null;if (arguments.length > 1) {
  body = cljs.core.array_seq(Array.prototype.slice.call(arguments, 1),0);} 
return form_to15602__delegate.call(this,p__15603,body);};
form_to15602.cljs$lang$maxFixedArity = 1;
form_to15602.cljs$lang$applyTo = (function (arglist__15606){
var p__15603 = cljs.core.first(arglist__15606);
var body = cljs.core.rest(arglist__15606);
return form_to15602__delegate(p__15603,body);
});
form_to15602.cljs$core$IFn$_invoke$arity$variadic = form_to15602__delegate;
return form_to15602;
})()
;
sablono.core.form_to = sablono.core.wrap_attrs.call(null,sablono.core.form_to15602);
