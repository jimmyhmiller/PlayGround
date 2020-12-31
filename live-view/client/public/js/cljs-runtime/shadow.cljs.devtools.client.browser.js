goog.provide('shadow.cljs.devtools.client.browser');
shadow.cljs.devtools.client.browser.devtools_msg = (function shadow$cljs$devtools$client$browser$devtools_msg(var_args){
var args__4742__auto__ = [];
var len__4736__auto___46721 = arguments.length;
var i__4737__auto___46722 = (0);
while(true){
if((i__4737__auto___46722 < len__4736__auto___46721)){
args__4742__auto__.push((arguments[i__4737__auto___46722]));

var G__46723 = (i__4737__auto___46722 + (1));
i__4737__auto___46722 = G__46723;
continue;
} else {
}
break;
}

var argseq__4743__auto__ = ((((1) < args__4742__auto__.length))?(new cljs.core.IndexedSeq(args__4742__auto__.slice((1)),(0),null)):null);
return shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic((arguments[(0)]),argseq__4743__auto__);
});

(shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic = (function (msg,args){
if(shadow.cljs.devtools.client.env.log){
if(cljs.core.seq(shadow.cljs.devtools.client.env.log_style)){
return console.log.apply(console,cljs.core.into_array.cljs$core$IFn$_invoke$arity$1(cljs.core.into.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [["%cshadow-cljs: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(msg)].join(''),shadow.cljs.devtools.client.env.log_style], null),args)));
} else {
return console.log.apply(console,cljs.core.into_array.cljs$core$IFn$_invoke$arity$1(cljs.core.into.cljs$core$IFn$_invoke$arity$2(new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [["shadow-cljs: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(msg)].join('')], null),args)));
}
} else {
return null;
}
}));

(shadow.cljs.devtools.client.browser.devtools_msg.cljs$lang$maxFixedArity = (1));

/** @this {Function} */
(shadow.cljs.devtools.client.browser.devtools_msg.cljs$lang$applyTo = (function (seq46285){
var G__46286 = cljs.core.first(seq46285);
var seq46285__$1 = cljs.core.next(seq46285);
var self__4723__auto__ = this;
return self__4723__auto__.cljs$core$IFn$_invoke$arity$variadic(G__46286,seq46285__$1);
}));

shadow.cljs.devtools.client.browser.script_eval = (function shadow$cljs$devtools$client$browser$script_eval(code){
return goog.globalEval(code);
});
shadow.cljs.devtools.client.browser.do_js_load = (function shadow$cljs$devtools$client$browser$do_js_load(sources){
var seq__46307 = cljs.core.seq(sources);
var chunk__46309 = null;
var count__46310 = (0);
var i__46311 = (0);
while(true){
if((i__46311 < count__46310)){
var map__46374 = chunk__46309.cljs$core$IIndexed$_nth$arity$2(null,i__46311);
var map__46374__$1 = (((((!((map__46374 == null))))?(((((map__46374.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46374.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46374):map__46374);
var src = map__46374__$1;
var resource_id = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46374__$1,new cljs.core.Keyword(null,"resource-id","resource-id",-1308422582));
var output_name = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46374__$1,new cljs.core.Keyword(null,"output-name","output-name",-1769107767));
var resource_name = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46374__$1,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100));
var js = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46374__$1,new cljs.core.Keyword(null,"js","js",1768080579));
$CLJS.SHADOW_ENV.setLoaded(output_name);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load JS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([resource_name], 0));

shadow.cljs.devtools.client.env.before_load_src(src);

try{shadow.cljs.devtools.client.browser.script_eval([cljs.core.str.cljs$core$IFn$_invoke$arity$1(js),"\n//# sourceURL=",cljs.core.str.cljs$core$IFn$_invoke$arity$1($CLJS.SHADOW_ENV.scriptBase),cljs.core.str.cljs$core$IFn$_invoke$arity$1(output_name)].join(''));
}catch (e46379){var e_46727 = e46379;
if(shadow.cljs.devtools.client.env.log){
console.error(["Failed to load ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name)].join(''),e_46727);
} else {
}

throw (new Error(["Failed to load ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name),": ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(e_46727.message)].join('')));
}

var G__46731 = seq__46307;
var G__46732 = chunk__46309;
var G__46733 = count__46310;
var G__46734 = (i__46311 + (1));
seq__46307 = G__46731;
chunk__46309 = G__46732;
count__46310 = G__46733;
i__46311 = G__46734;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46307);
if(temp__5735__auto__){
var seq__46307__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46307__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46307__$1);
var G__46735 = cljs.core.chunk_rest(seq__46307__$1);
var G__46736 = c__4556__auto__;
var G__46737 = cljs.core.count(c__4556__auto__);
var G__46738 = (0);
seq__46307 = G__46735;
chunk__46309 = G__46736;
count__46310 = G__46737;
i__46311 = G__46738;
continue;
} else {
var map__46380 = cljs.core.first(seq__46307__$1);
var map__46380__$1 = (((((!((map__46380 == null))))?(((((map__46380.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46380.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46380):map__46380);
var src = map__46380__$1;
var resource_id = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46380__$1,new cljs.core.Keyword(null,"resource-id","resource-id",-1308422582));
var output_name = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46380__$1,new cljs.core.Keyword(null,"output-name","output-name",-1769107767));
var resource_name = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46380__$1,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100));
var js = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46380__$1,new cljs.core.Keyword(null,"js","js",1768080579));
$CLJS.SHADOW_ENV.setLoaded(output_name);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load JS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([resource_name], 0));

shadow.cljs.devtools.client.env.before_load_src(src);

try{shadow.cljs.devtools.client.browser.script_eval([cljs.core.str.cljs$core$IFn$_invoke$arity$1(js),"\n//# sourceURL=",cljs.core.str.cljs$core$IFn$_invoke$arity$1($CLJS.SHADOW_ENV.scriptBase),cljs.core.str.cljs$core$IFn$_invoke$arity$1(output_name)].join(''));
}catch (e46383){var e_46740 = e46383;
if(shadow.cljs.devtools.client.env.log){
console.error(["Failed to load ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name)].join(''),e_46740);
} else {
}

throw (new Error(["Failed to load ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name),": ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(e_46740.message)].join('')));
}

var G__46744 = cljs.core.next(seq__46307__$1);
var G__46745 = null;
var G__46746 = (0);
var G__46747 = (0);
seq__46307 = G__46744;
chunk__46309 = G__46745;
count__46310 = G__46746;
i__46311 = G__46747;
continue;
}
} else {
return null;
}
}
break;
}
});
shadow.cljs.devtools.client.browser.do_js_reload = (function shadow$cljs$devtools$client$browser$do_js_reload(msg,sources,complete_fn,failure_fn){
return shadow.cljs.devtools.client.env.do_js_reload.cljs$core$IFn$_invoke$arity$4(cljs.core.assoc.cljs$core$IFn$_invoke$arity$variadic(msg,new cljs.core.Keyword(null,"log-missing-fn","log-missing-fn",732676765),(function (fn_sym){
return null;
}),cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([new cljs.core.Keyword(null,"log-call-async","log-call-async",183826192),(function (fn_sym){
return shadow.cljs.devtools.client.browser.devtools_msg(["call async ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(fn_sym)].join(''));
}),new cljs.core.Keyword(null,"log-call","log-call",412404391),(function (fn_sym){
return shadow.cljs.devtools.client.browser.devtools_msg(["call ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(fn_sym)].join(''));
})], 0)),(function (){
return shadow.cljs.devtools.client.browser.do_js_load(sources);
}),complete_fn,failure_fn);
});
/**
 * when (require '["some-str" :as x]) is done at the REPL we need to manually call the shadow.js.require for it
 * since the file only adds the shadow$provide. only need to do this for shadow-js.
 */
shadow.cljs.devtools.client.browser.do_js_requires = (function shadow$cljs$devtools$client$browser$do_js_requires(js_requires){
var seq__46386 = cljs.core.seq(js_requires);
var chunk__46387 = null;
var count__46388 = (0);
var i__46389 = (0);
while(true){
if((i__46389 < count__46388)){
var js_ns = chunk__46387.cljs$core$IIndexed$_nth$arity$2(null,i__46389);
var require_str_46748 = ["var ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(js_ns)," = shadow.js.require(\"",cljs.core.str.cljs$core$IFn$_invoke$arity$1(js_ns),"\");"].join('');
shadow.cljs.devtools.client.browser.script_eval(require_str_46748);


var G__46749 = seq__46386;
var G__46750 = chunk__46387;
var G__46751 = count__46388;
var G__46752 = (i__46389 + (1));
seq__46386 = G__46749;
chunk__46387 = G__46750;
count__46388 = G__46751;
i__46389 = G__46752;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46386);
if(temp__5735__auto__){
var seq__46386__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46386__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46386__$1);
var G__46753 = cljs.core.chunk_rest(seq__46386__$1);
var G__46754 = c__4556__auto__;
var G__46755 = cljs.core.count(c__4556__auto__);
var G__46756 = (0);
seq__46386 = G__46753;
chunk__46387 = G__46754;
count__46388 = G__46755;
i__46389 = G__46756;
continue;
} else {
var js_ns = cljs.core.first(seq__46386__$1);
var require_str_46757 = ["var ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(js_ns)," = shadow.js.require(\"",cljs.core.str.cljs$core$IFn$_invoke$arity$1(js_ns),"\");"].join('');
shadow.cljs.devtools.client.browser.script_eval(require_str_46757);


var G__46758 = cljs.core.next(seq__46386__$1);
var G__46759 = null;
var G__46760 = (0);
var G__46761 = (0);
seq__46386 = G__46758;
chunk__46387 = G__46759;
count__46388 = G__46760;
i__46389 = G__46761;
continue;
}
} else {
return null;
}
}
break;
}
});
shadow.cljs.devtools.client.browser.handle_build_complete = (function shadow$cljs$devtools$client$browser$handle_build_complete(runtime,p__46398){
var map__46400 = p__46398;
var map__46400__$1 = (((((!((map__46400 == null))))?(((((map__46400.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46400.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46400):map__46400);
var msg = map__46400__$1;
var info = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46400__$1,new cljs.core.Keyword(null,"info","info",-317069002));
var reload_info = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46400__$1,new cljs.core.Keyword(null,"reload-info","reload-info",1648088086));
var warnings = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentVector.EMPTY,cljs.core.distinct.cljs$core$IFn$_invoke$arity$1((function (){var iter__4529__auto__ = (function shadow$cljs$devtools$client$browser$handle_build_complete_$_iter__46402(s__46403){
return (new cljs.core.LazySeq(null,(function (){
var s__46403__$1 = s__46403;
while(true){
var temp__5735__auto__ = cljs.core.seq(s__46403__$1);
if(temp__5735__auto__){
var xs__6292__auto__ = temp__5735__auto__;
var map__46409 = cljs.core.first(xs__6292__auto__);
var map__46409__$1 = (((((!((map__46409 == null))))?(((((map__46409.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46409.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46409):map__46409);
var src = map__46409__$1;
var resource_name = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46409__$1,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100));
var warnings = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46409__$1,new cljs.core.Keyword(null,"warnings","warnings",-735437651));
if(cljs.core.not(new cljs.core.Keyword(null,"from-jar","from-jar",1050932827).cljs$core$IFn$_invoke$arity$1(src))){
var iterys__4525__auto__ = ((function (s__46403__$1,map__46409,map__46409__$1,src,resource_name,warnings,xs__6292__auto__,temp__5735__auto__,map__46400,map__46400__$1,msg,info,reload_info){
return (function shadow$cljs$devtools$client$browser$handle_build_complete_$_iter__46402_$_iter__46404(s__46405){
return (new cljs.core.LazySeq(null,((function (s__46403__$1,map__46409,map__46409__$1,src,resource_name,warnings,xs__6292__auto__,temp__5735__auto__,map__46400,map__46400__$1,msg,info,reload_info){
return (function (){
var s__46405__$1 = s__46405;
while(true){
var temp__5735__auto____$1 = cljs.core.seq(s__46405__$1);
if(temp__5735__auto____$1){
var s__46405__$2 = temp__5735__auto____$1;
if(cljs.core.chunked_seq_QMARK_(s__46405__$2)){
var c__4527__auto__ = cljs.core.chunk_first(s__46405__$2);
var size__4528__auto__ = cljs.core.count(c__4527__auto__);
var b__46407 = cljs.core.chunk_buffer(size__4528__auto__);
if((function (){var i__46406 = (0);
while(true){
if((i__46406 < size__4528__auto__)){
var warning = cljs.core._nth(c__4527__auto__,i__46406);
cljs.core.chunk_append(b__46407,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(warning,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100),resource_name));

var G__46765 = (i__46406 + (1));
i__46406 = G__46765;
continue;
} else {
return true;
}
break;
}
})()){
return cljs.core.chunk_cons(cljs.core.chunk(b__46407),shadow$cljs$devtools$client$browser$handle_build_complete_$_iter__46402_$_iter__46404(cljs.core.chunk_rest(s__46405__$2)));
} else {
return cljs.core.chunk_cons(cljs.core.chunk(b__46407),null);
}
} else {
var warning = cljs.core.first(s__46405__$2);
return cljs.core.cons(cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(warning,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100),resource_name),shadow$cljs$devtools$client$browser$handle_build_complete_$_iter__46402_$_iter__46404(cljs.core.rest(s__46405__$2)));
}
} else {
return null;
}
break;
}
});})(s__46403__$1,map__46409,map__46409__$1,src,resource_name,warnings,xs__6292__auto__,temp__5735__auto__,map__46400,map__46400__$1,msg,info,reload_info))
,null,null));
});})(s__46403__$1,map__46409,map__46409__$1,src,resource_name,warnings,xs__6292__auto__,temp__5735__auto__,map__46400,map__46400__$1,msg,info,reload_info))
;
var fs__4526__auto__ = cljs.core.seq(iterys__4525__auto__(warnings));
if(fs__4526__auto__){
return cljs.core.concat.cljs$core$IFn$_invoke$arity$2(fs__4526__auto__,shadow$cljs$devtools$client$browser$handle_build_complete_$_iter__46402(cljs.core.rest(s__46403__$1)));
} else {
var G__46766 = cljs.core.rest(s__46403__$1);
s__46403__$1 = G__46766;
continue;
}
} else {
var G__46767 = cljs.core.rest(s__46403__$1);
s__46403__$1 = G__46767;
continue;
}
} else {
return null;
}
break;
}
}),null,null));
});
return iter__4529__auto__(new cljs.core.Keyword(null,"sources","sources",-321166424).cljs$core$IFn$_invoke$arity$1(info));
})()));
if(shadow.cljs.devtools.client.env.log){
var seq__46417_46768 = cljs.core.seq(warnings);
var chunk__46418_46769 = null;
var count__46419_46770 = (0);
var i__46420_46771 = (0);
while(true){
if((i__46420_46771 < count__46419_46770)){
var map__46436_46774 = chunk__46418_46769.cljs$core$IIndexed$_nth$arity$2(null,i__46420_46771);
var map__46436_46775__$1 = (((((!((map__46436_46774 == null))))?(((((map__46436_46774.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46436_46774.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46436_46774):map__46436_46774);
var w_46776 = map__46436_46775__$1;
var msg_46777__$1 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46436_46775__$1,new cljs.core.Keyword(null,"msg","msg",-1386103444));
var line_46778 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46436_46775__$1,new cljs.core.Keyword(null,"line","line",212345235));
var column_46779 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46436_46775__$1,new cljs.core.Keyword(null,"column","column",2078222095));
var resource_name_46780 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46436_46775__$1,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100));
console.warn(["BUILD-WARNING in ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name_46780)," at [",cljs.core.str.cljs$core$IFn$_invoke$arity$1(line_46778),":",cljs.core.str.cljs$core$IFn$_invoke$arity$1(column_46779),"]\n\t",cljs.core.str.cljs$core$IFn$_invoke$arity$1(msg_46777__$1)].join(''));


var G__46781 = seq__46417_46768;
var G__46782 = chunk__46418_46769;
var G__46783 = count__46419_46770;
var G__46784 = (i__46420_46771 + (1));
seq__46417_46768 = G__46781;
chunk__46418_46769 = G__46782;
count__46419_46770 = G__46783;
i__46420_46771 = G__46784;
continue;
} else {
var temp__5735__auto___46785 = cljs.core.seq(seq__46417_46768);
if(temp__5735__auto___46785){
var seq__46417_46786__$1 = temp__5735__auto___46785;
if(cljs.core.chunked_seq_QMARK_(seq__46417_46786__$1)){
var c__4556__auto___46788 = cljs.core.chunk_first(seq__46417_46786__$1);
var G__46789 = cljs.core.chunk_rest(seq__46417_46786__$1);
var G__46790 = c__4556__auto___46788;
var G__46791 = cljs.core.count(c__4556__auto___46788);
var G__46792 = (0);
seq__46417_46768 = G__46789;
chunk__46418_46769 = G__46790;
count__46419_46770 = G__46791;
i__46420_46771 = G__46792;
continue;
} else {
var map__46446_46796 = cljs.core.first(seq__46417_46786__$1);
var map__46446_46797__$1 = (((((!((map__46446_46796 == null))))?(((((map__46446_46796.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46446_46796.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46446_46796):map__46446_46796);
var w_46798 = map__46446_46797__$1;
var msg_46799__$1 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46446_46797__$1,new cljs.core.Keyword(null,"msg","msg",-1386103444));
var line_46800 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46446_46797__$1,new cljs.core.Keyword(null,"line","line",212345235));
var column_46801 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46446_46797__$1,new cljs.core.Keyword(null,"column","column",2078222095));
var resource_name_46802 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46446_46797__$1,new cljs.core.Keyword(null,"resource-name","resource-name",2001617100));
console.warn(["BUILD-WARNING in ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(resource_name_46802)," at [",cljs.core.str.cljs$core$IFn$_invoke$arity$1(line_46800),":",cljs.core.str.cljs$core$IFn$_invoke$arity$1(column_46801),"]\n\t",cljs.core.str.cljs$core$IFn$_invoke$arity$1(msg_46799__$1)].join(''));


var G__46803 = cljs.core.next(seq__46417_46786__$1);
var G__46804 = null;
var G__46805 = (0);
var G__46806 = (0);
seq__46417_46768 = G__46803;
chunk__46418_46769 = G__46804;
count__46419_46770 = G__46805;
i__46420_46771 = G__46806;
continue;
}
} else {
}
}
break;
}
} else {
}

if((!(shadow.cljs.devtools.client.env.autoload))){
return shadow.cljs.devtools.client.hud.load_end_success();
} else {
if(((cljs.core.empty_QMARK_(warnings)) || (shadow.cljs.devtools.client.env.ignore_warnings))){
var sources_to_get = shadow.cljs.devtools.client.env.filter_reload_sources(info,reload_info);
if(cljs.core.not(cljs.core.seq(sources_to_get))){
return shadow.cljs.devtools.client.hud.load_end_success();
} else {
if(cljs.core.seq(cljs.core.get_in.cljs$core$IFn$_invoke$arity$2(msg,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"reload-info","reload-info",1648088086),new cljs.core.Keyword(null,"after-load","after-load",-1278503285)], null)))){
} else {
shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("reloading code but no :after-load hooks are configured!",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2(["https://shadow-cljs.github.io/docs/UsersGuide.html#_lifecycle_hooks"], 0));
}

return shadow.cljs.devtools.client.shared.load_sources(runtime,sources_to_get,(function (p1__46397_SHARP_){
return shadow.cljs.devtools.client.browser.do_js_reload(msg,p1__46397_SHARP_,shadow.cljs.devtools.client.hud.load_end_success,shadow.cljs.devtools.client.hud.load_failure);
}));
}
} else {
return null;
}
}
});
shadow.cljs.devtools.client.browser.page_load_uri = (cljs.core.truth_(goog.global.document)?goog.Uri.parse(document.location.href):null);
shadow.cljs.devtools.client.browser.match_paths = (function shadow$cljs$devtools$client$browser$match_paths(old,new$){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2("file",shadow.cljs.devtools.client.browser.page_load_uri.getScheme())){
var rel_new = cljs.core.subs.cljs$core$IFn$_invoke$arity$2(new$,(1));
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(old,rel_new)) || (clojure.string.starts_with_QMARK_(old,[rel_new,"?"].join(''))))){
return rel_new;
} else {
return null;
}
} else {
var node_uri = goog.Uri.parse(old);
var node_uri_resolved = shadow.cljs.devtools.client.browser.page_load_uri.resolve(node_uri);
var node_abs = node_uri_resolved.getPath();
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$1(shadow.cljs.devtools.client.browser.page_load_uri.hasSameDomainAs(node_uri))) || (cljs.core.not(node_uri.hasDomain())))){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(node_abs,new$)){
return new$;
} else {
return false;
}
} else {
return false;
}
}
});
shadow.cljs.devtools.client.browser.handle_asset_update = (function shadow$cljs$devtools$client$browser$handle_asset_update(p__46449){
var map__46450 = p__46449;
var map__46450__$1 = (((((!((map__46450 == null))))?(((((map__46450.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46450.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46450):map__46450);
var msg = map__46450__$1;
var updates = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46450__$1,new cljs.core.Keyword(null,"updates","updates",2013983452));
var seq__46456 = cljs.core.seq(updates);
var chunk__46458 = null;
var count__46459 = (0);
var i__46460 = (0);
while(true){
if((i__46460 < count__46459)){
var path = chunk__46458.cljs$core$IIndexed$_nth$arity$2(null,i__46460);
if(clojure.string.ends_with_QMARK_(path,"css")){
var seq__46566_46815 = cljs.core.seq(cljs.core.array_seq.cljs$core$IFn$_invoke$arity$1(document.querySelectorAll("link[rel=\"stylesheet\"]")));
var chunk__46570_46816 = null;
var count__46571_46817 = (0);
var i__46572_46818 = (0);
while(true){
if((i__46572_46818 < count__46571_46817)){
var node_46824 = chunk__46570_46816.cljs$core$IIndexed$_nth$arity$2(null,i__46572_46818);
if(cljs.core.not(node_46824.shadow$old)){
var path_match_46825 = shadow.cljs.devtools.client.browser.match_paths(node_46824.getAttribute("href"),path);
if(cljs.core.truth_(path_match_46825)){
var new_link_46829 = (function (){var G__46590 = node_46824.cloneNode(true);
G__46590.setAttribute("href",[cljs.core.str.cljs$core$IFn$_invoke$arity$1(path_match_46825),"?r=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.rand.cljs$core$IFn$_invoke$arity$0())].join(''));

return G__46590;
})();
(node_46824.shadow$old = true);

(new_link_46829.onload = ((function (seq__46566_46815,chunk__46570_46816,count__46571_46817,i__46572_46818,seq__46456,chunk__46458,count__46459,i__46460,new_link_46829,path_match_46825,node_46824,path,map__46450,map__46450__$1,msg,updates){
return (function (e){
return goog.dom.removeNode(node_46824);
});})(seq__46566_46815,chunk__46570_46816,count__46571_46817,i__46572_46818,seq__46456,chunk__46458,count__46459,i__46460,new_link_46829,path_match_46825,node_46824,path,map__46450,map__46450__$1,msg,updates))
);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load CSS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([path_match_46825], 0));

goog.dom.insertSiblingAfter(new_link_46829,node_46824);


var G__46835 = seq__46566_46815;
var G__46836 = chunk__46570_46816;
var G__46837 = count__46571_46817;
var G__46838 = (i__46572_46818 + (1));
seq__46566_46815 = G__46835;
chunk__46570_46816 = G__46836;
count__46571_46817 = G__46837;
i__46572_46818 = G__46838;
continue;
} else {
var G__46839 = seq__46566_46815;
var G__46840 = chunk__46570_46816;
var G__46841 = count__46571_46817;
var G__46842 = (i__46572_46818 + (1));
seq__46566_46815 = G__46839;
chunk__46570_46816 = G__46840;
count__46571_46817 = G__46841;
i__46572_46818 = G__46842;
continue;
}
} else {
var G__46843 = seq__46566_46815;
var G__46844 = chunk__46570_46816;
var G__46845 = count__46571_46817;
var G__46846 = (i__46572_46818 + (1));
seq__46566_46815 = G__46843;
chunk__46570_46816 = G__46844;
count__46571_46817 = G__46845;
i__46572_46818 = G__46846;
continue;
}
} else {
var temp__5735__auto___46853 = cljs.core.seq(seq__46566_46815);
if(temp__5735__auto___46853){
var seq__46566_46854__$1 = temp__5735__auto___46853;
if(cljs.core.chunked_seq_QMARK_(seq__46566_46854__$1)){
var c__4556__auto___46855 = cljs.core.chunk_first(seq__46566_46854__$1);
var G__46856 = cljs.core.chunk_rest(seq__46566_46854__$1);
var G__46857 = c__4556__auto___46855;
var G__46858 = cljs.core.count(c__4556__auto___46855);
var G__46859 = (0);
seq__46566_46815 = G__46856;
chunk__46570_46816 = G__46857;
count__46571_46817 = G__46858;
i__46572_46818 = G__46859;
continue;
} else {
var node_46860 = cljs.core.first(seq__46566_46854__$1);
if(cljs.core.not(node_46860.shadow$old)){
var path_match_46861 = shadow.cljs.devtools.client.browser.match_paths(node_46860.getAttribute("href"),path);
if(cljs.core.truth_(path_match_46861)){
var new_link_46862 = (function (){var G__46605 = node_46860.cloneNode(true);
G__46605.setAttribute("href",[cljs.core.str.cljs$core$IFn$_invoke$arity$1(path_match_46861),"?r=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.rand.cljs$core$IFn$_invoke$arity$0())].join(''));

return G__46605;
})();
(node_46860.shadow$old = true);

(new_link_46862.onload = ((function (seq__46566_46815,chunk__46570_46816,count__46571_46817,i__46572_46818,seq__46456,chunk__46458,count__46459,i__46460,new_link_46862,path_match_46861,node_46860,seq__46566_46854__$1,temp__5735__auto___46853,path,map__46450,map__46450__$1,msg,updates){
return (function (e){
return goog.dom.removeNode(node_46860);
});})(seq__46566_46815,chunk__46570_46816,count__46571_46817,i__46572_46818,seq__46456,chunk__46458,count__46459,i__46460,new_link_46862,path_match_46861,node_46860,seq__46566_46854__$1,temp__5735__auto___46853,path,map__46450,map__46450__$1,msg,updates))
);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load CSS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([path_match_46861], 0));

goog.dom.insertSiblingAfter(new_link_46862,node_46860);


var G__46863 = cljs.core.next(seq__46566_46854__$1);
var G__46864 = null;
var G__46865 = (0);
var G__46866 = (0);
seq__46566_46815 = G__46863;
chunk__46570_46816 = G__46864;
count__46571_46817 = G__46865;
i__46572_46818 = G__46866;
continue;
} else {
var G__46867 = cljs.core.next(seq__46566_46854__$1);
var G__46868 = null;
var G__46869 = (0);
var G__46870 = (0);
seq__46566_46815 = G__46867;
chunk__46570_46816 = G__46868;
count__46571_46817 = G__46869;
i__46572_46818 = G__46870;
continue;
}
} else {
var G__46871 = cljs.core.next(seq__46566_46854__$1);
var G__46872 = null;
var G__46873 = (0);
var G__46874 = (0);
seq__46566_46815 = G__46871;
chunk__46570_46816 = G__46872;
count__46571_46817 = G__46873;
i__46572_46818 = G__46874;
continue;
}
}
} else {
}
}
break;
}


var G__46881 = seq__46456;
var G__46882 = chunk__46458;
var G__46883 = count__46459;
var G__46884 = (i__46460 + (1));
seq__46456 = G__46881;
chunk__46458 = G__46882;
count__46459 = G__46883;
i__46460 = G__46884;
continue;
} else {
var G__46886 = seq__46456;
var G__46887 = chunk__46458;
var G__46888 = count__46459;
var G__46889 = (i__46460 + (1));
seq__46456 = G__46886;
chunk__46458 = G__46887;
count__46459 = G__46888;
i__46460 = G__46889;
continue;
}
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46456);
if(temp__5735__auto__){
var seq__46456__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46456__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46456__$1);
var G__46890 = cljs.core.chunk_rest(seq__46456__$1);
var G__46891 = c__4556__auto__;
var G__46892 = cljs.core.count(c__4556__auto__);
var G__46893 = (0);
seq__46456 = G__46890;
chunk__46458 = G__46891;
count__46459 = G__46892;
i__46460 = G__46893;
continue;
} else {
var path = cljs.core.first(seq__46456__$1);
if(clojure.string.ends_with_QMARK_(path,"css")){
var seq__46615_46896 = cljs.core.seq(cljs.core.array_seq.cljs$core$IFn$_invoke$arity$1(document.querySelectorAll("link[rel=\"stylesheet\"]")));
var chunk__46635_46897 = null;
var count__46636_46898 = (0);
var i__46637_46899 = (0);
while(true){
if((i__46637_46899 < count__46636_46898)){
var node_46901 = chunk__46635_46897.cljs$core$IIndexed$_nth$arity$2(null,i__46637_46899);
if(cljs.core.not(node_46901.shadow$old)){
var path_match_46902 = shadow.cljs.devtools.client.browser.match_paths(node_46901.getAttribute("href"),path);
if(cljs.core.truth_(path_match_46902)){
var new_link_46903 = (function (){var G__46658 = node_46901.cloneNode(true);
G__46658.setAttribute("href",[cljs.core.str.cljs$core$IFn$_invoke$arity$1(path_match_46902),"?r=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.rand.cljs$core$IFn$_invoke$arity$0())].join(''));

return G__46658;
})();
(node_46901.shadow$old = true);

(new_link_46903.onload = ((function (seq__46615_46896,chunk__46635_46897,count__46636_46898,i__46637_46899,seq__46456,chunk__46458,count__46459,i__46460,new_link_46903,path_match_46902,node_46901,path,seq__46456__$1,temp__5735__auto__,map__46450,map__46450__$1,msg,updates){
return (function (e){
return goog.dom.removeNode(node_46901);
});})(seq__46615_46896,chunk__46635_46897,count__46636_46898,i__46637_46899,seq__46456,chunk__46458,count__46459,i__46460,new_link_46903,path_match_46902,node_46901,path,seq__46456__$1,temp__5735__auto__,map__46450,map__46450__$1,msg,updates))
);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load CSS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([path_match_46902], 0));

goog.dom.insertSiblingAfter(new_link_46903,node_46901);


var G__46905 = seq__46615_46896;
var G__46906 = chunk__46635_46897;
var G__46907 = count__46636_46898;
var G__46908 = (i__46637_46899 + (1));
seq__46615_46896 = G__46905;
chunk__46635_46897 = G__46906;
count__46636_46898 = G__46907;
i__46637_46899 = G__46908;
continue;
} else {
var G__46909 = seq__46615_46896;
var G__46910 = chunk__46635_46897;
var G__46911 = count__46636_46898;
var G__46912 = (i__46637_46899 + (1));
seq__46615_46896 = G__46909;
chunk__46635_46897 = G__46910;
count__46636_46898 = G__46911;
i__46637_46899 = G__46912;
continue;
}
} else {
var G__46913 = seq__46615_46896;
var G__46914 = chunk__46635_46897;
var G__46915 = count__46636_46898;
var G__46916 = (i__46637_46899 + (1));
seq__46615_46896 = G__46913;
chunk__46635_46897 = G__46914;
count__46636_46898 = G__46915;
i__46637_46899 = G__46916;
continue;
}
} else {
var temp__5735__auto___46917__$1 = cljs.core.seq(seq__46615_46896);
if(temp__5735__auto___46917__$1){
var seq__46615_46918__$1 = temp__5735__auto___46917__$1;
if(cljs.core.chunked_seq_QMARK_(seq__46615_46918__$1)){
var c__4556__auto___46919 = cljs.core.chunk_first(seq__46615_46918__$1);
var G__46920 = cljs.core.chunk_rest(seq__46615_46918__$1);
var G__46921 = c__4556__auto___46919;
var G__46922 = cljs.core.count(c__4556__auto___46919);
var G__46923 = (0);
seq__46615_46896 = G__46920;
chunk__46635_46897 = G__46921;
count__46636_46898 = G__46922;
i__46637_46899 = G__46923;
continue;
} else {
var node_46924 = cljs.core.first(seq__46615_46918__$1);
if(cljs.core.not(node_46924.shadow$old)){
var path_match_46929 = shadow.cljs.devtools.client.browser.match_paths(node_46924.getAttribute("href"),path);
if(cljs.core.truth_(path_match_46929)){
var new_link_46930 = (function (){var G__46659 = node_46924.cloneNode(true);
G__46659.setAttribute("href",[cljs.core.str.cljs$core$IFn$_invoke$arity$1(path_match_46929),"?r=",cljs.core.str.cljs$core$IFn$_invoke$arity$1(cljs.core.rand.cljs$core$IFn$_invoke$arity$0())].join(''));

return G__46659;
})();
(node_46924.shadow$old = true);

(new_link_46930.onload = ((function (seq__46615_46896,chunk__46635_46897,count__46636_46898,i__46637_46899,seq__46456,chunk__46458,count__46459,i__46460,new_link_46930,path_match_46929,node_46924,seq__46615_46918__$1,temp__5735__auto___46917__$1,path,seq__46456__$1,temp__5735__auto__,map__46450,map__46450__$1,msg,updates){
return (function (e){
return goog.dom.removeNode(node_46924);
});})(seq__46615_46896,chunk__46635_46897,count__46636_46898,i__46637_46899,seq__46456,chunk__46458,count__46459,i__46460,new_link_46930,path_match_46929,node_46924,seq__46615_46918__$1,temp__5735__auto___46917__$1,path,seq__46456__$1,temp__5735__auto__,map__46450,map__46450__$1,msg,updates))
);

shadow.cljs.devtools.client.browser.devtools_msg.cljs$core$IFn$_invoke$arity$variadic("load CSS",cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([path_match_46929], 0));

goog.dom.insertSiblingAfter(new_link_46930,node_46924);


var G__46933 = cljs.core.next(seq__46615_46918__$1);
var G__46934 = null;
var G__46935 = (0);
var G__46936 = (0);
seq__46615_46896 = G__46933;
chunk__46635_46897 = G__46934;
count__46636_46898 = G__46935;
i__46637_46899 = G__46936;
continue;
} else {
var G__46943 = cljs.core.next(seq__46615_46918__$1);
var G__46944 = null;
var G__46945 = (0);
var G__46946 = (0);
seq__46615_46896 = G__46943;
chunk__46635_46897 = G__46944;
count__46636_46898 = G__46945;
i__46637_46899 = G__46946;
continue;
}
} else {
var G__46947 = cljs.core.next(seq__46615_46918__$1);
var G__46948 = null;
var G__46949 = (0);
var G__46950 = (0);
seq__46615_46896 = G__46947;
chunk__46635_46897 = G__46948;
count__46636_46898 = G__46949;
i__46637_46899 = G__46950;
continue;
}
}
} else {
}
}
break;
}


var G__46953 = cljs.core.next(seq__46456__$1);
var G__46954 = null;
var G__46955 = (0);
var G__46956 = (0);
seq__46456 = G__46953;
chunk__46458 = G__46954;
count__46459 = G__46955;
i__46460 = G__46956;
continue;
} else {
var G__46959 = cljs.core.next(seq__46456__$1);
var G__46960 = null;
var G__46961 = (0);
var G__46962 = (0);
seq__46456 = G__46959;
chunk__46458 = G__46960;
count__46459 = G__46961;
i__46460 = G__46962;
continue;
}
}
} else {
return null;
}
}
break;
}
});
shadow.cljs.devtools.client.browser.global_eval = (function shadow$cljs$devtools$client$browser$global_eval(js){
if(cljs.core.not_EQ_.cljs$core$IFn$_invoke$arity$2("undefined",typeof(module))){
return eval(js);
} else {
return (0,eval)(js);;
}
});
shadow.cljs.devtools.client.browser.repl_init = (function shadow$cljs$devtools$client$browser$repl_init(runtime,p__46662){
var map__46663 = p__46662;
var map__46663__$1 = (((((!((map__46663 == null))))?(((((map__46663.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46663.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46663):map__46663);
var repl_state = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46663__$1,new cljs.core.Keyword(null,"repl-state","repl-state",-1733780387));
return shadow.cljs.devtools.client.shared.load_sources(runtime,cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentVector.EMPTY,cljs.core.remove.cljs$core$IFn$_invoke$arity$2(shadow.cljs.devtools.client.env.src_is_loaded_QMARK_,new cljs.core.Keyword(null,"repl-sources","repl-sources",723867535).cljs$core$IFn$_invoke$arity$1(repl_state))),(function (sources){
shadow.cljs.devtools.client.browser.do_js_load(sources);

return shadow.cljs.devtools.client.browser.devtools_msg("ready!");
}));
});
shadow.cljs.devtools.client.browser.runtime_info = (((typeof SHADOW_CONFIG !== 'undefined'))?shadow.json.to_clj.cljs$core$IFn$_invoke$arity$1(SHADOW_CONFIG):null);
shadow.cljs.devtools.client.browser.client_info = cljs.core.merge.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([shadow.cljs.devtools.client.browser.runtime_info,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"host","host",-1558485167),(cljs.core.truth_(goog.global.document)?new cljs.core.Keyword(null,"browser","browser",828191719):new cljs.core.Keyword(null,"browser-worker","browser-worker",1638998282)),new cljs.core.Keyword(null,"user-agent","user-agent",1220426212),[(cljs.core.truth_(goog.userAgent.OPERA)?"Opera":(cljs.core.truth_(goog.userAgent.product.CHROME)?"Chrome":(cljs.core.truth_(goog.userAgent.IE)?"MSIE":(cljs.core.truth_(goog.userAgent.EDGE)?"Edge":(cljs.core.truth_(goog.userAgent.GECKO)?"Firefox":(cljs.core.truth_(goog.userAgent.SAFARI)?"Safari":(cljs.core.truth_(goog.userAgent.WEBKIT)?"Webkit":null)))))))," ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(goog.userAgent.VERSION)," [",cljs.core.str.cljs$core$IFn$_invoke$arity$1(goog.userAgent.PLATFORM),"]"].join(''),new cljs.core.Keyword(null,"dom","dom",-1236537922),(!((goog.global.document == null)))], null)], 0));
if((typeof shadow !== 'undefined') && (typeof shadow.cljs !== 'undefined') && (typeof shadow.cljs.devtools !== 'undefined') && (typeof shadow.cljs.devtools.client !== 'undefined') && (typeof shadow.cljs.devtools.client.browser !== 'undefined') && (typeof shadow.cljs.devtools.client.browser.ws_was_welcome_ref !== 'undefined')){
} else {
shadow.cljs.devtools.client.browser.ws_was_welcome_ref = cljs.core.atom.cljs$core$IFn$_invoke$arity$1(false);
}
if(((shadow.cljs.devtools.client.env.enabled) && ((shadow.cljs.devtools.client.env.worker_client_id > (0))))){
(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$remote$runtime$api$IEvalJS$ = cljs.core.PROTOCOL_SENTINEL);

(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$remote$runtime$api$IEvalJS$_js_eval$arity$2 = (function (this$,code){
var this$__$1 = this;
return shadow.cljs.devtools.client.browser.global_eval(code);
}));

(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$cljs$devtools$client$shared$IHostSpecific$ = cljs.core.PROTOCOL_SENTINEL);

(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$cljs$devtools$client$shared$IHostSpecific$do_invoke$arity$2 = (function (this$,p__46673){
var map__46674 = p__46673;
var map__46674__$1 = (((((!((map__46674 == null))))?(((((map__46674.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46674.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46674):map__46674);
var _ = map__46674__$1;
var js = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46674__$1,new cljs.core.Keyword(null,"js","js",1768080579));
var this$__$1 = this;
return shadow.cljs.devtools.client.browser.global_eval(js);
}));

(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$cljs$devtools$client$shared$IHostSpecific$do_repl_init$arity$4 = (function (runtime,p__46677,done,error){
var map__46678 = p__46677;
var map__46678__$1 = (((((!((map__46678 == null))))?(((((map__46678.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46678.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46678):map__46678);
var repl_sources = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46678__$1,new cljs.core.Keyword(null,"repl-sources","repl-sources",723867535));
var runtime__$1 = this;
return shadow.cljs.devtools.client.shared.load_sources(runtime__$1,cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentVector.EMPTY,cljs.core.remove.cljs$core$IFn$_invoke$arity$2(shadow.cljs.devtools.client.env.src_is_loaded_QMARK_,repl_sources)),(function (sources){
shadow.cljs.devtools.client.browser.do_js_load(sources);

return (done.cljs$core$IFn$_invoke$arity$0 ? done.cljs$core$IFn$_invoke$arity$0() : done.call(null));
}));
}));

(shadow.cljs.devtools.client.shared.Runtime.prototype.shadow$cljs$devtools$client$shared$IHostSpecific$do_repl_require$arity$4 = (function (runtime,p__46681,done,error){
var map__46682 = p__46681;
var map__46682__$1 = (((((!((map__46682 == null))))?(((((map__46682.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46682.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46682):map__46682);
var msg = map__46682__$1;
var sources = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46682__$1,new cljs.core.Keyword(null,"sources","sources",-321166424));
var reload_namespaces = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46682__$1,new cljs.core.Keyword(null,"reload-namespaces","reload-namespaces",250210134));
var js_requires = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46682__$1,new cljs.core.Keyword(null,"js-requires","js-requires",-1311472051));
var runtime__$1 = this;
var sources_to_load = cljs.core.into.cljs$core$IFn$_invoke$arity$2(cljs.core.PersistentVector.EMPTY,cljs.core.remove.cljs$core$IFn$_invoke$arity$2((function (p__46685){
var map__46686 = p__46685;
var map__46686__$1 = (((((!((map__46686 == null))))?(((((map__46686.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46686.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46686):map__46686);
var src = map__46686__$1;
var provides = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46686__$1,new cljs.core.Keyword(null,"provides","provides",-1634397992));
var and__4115__auto__ = shadow.cljs.devtools.client.env.src_is_loaded_QMARK_(src);
if(cljs.core.truth_(and__4115__auto__)){
return cljs.core.not(cljs.core.some(reload_namespaces,provides));
} else {
return and__4115__auto__;
}
}),sources));
if(cljs.core.not(cljs.core.seq(sources_to_load))){
var G__46688 = cljs.core.PersistentVector.EMPTY;
return (done.cljs$core$IFn$_invoke$arity$1 ? done.cljs$core$IFn$_invoke$arity$1(G__46688) : done.call(null,G__46688));
} else {
return shadow.remote.runtime.shared.call.cljs$core$IFn$_invoke$arity$3(runtime__$1,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"op","op",-1882987955),new cljs.core.Keyword(null,"cljs-load-sources","cljs-load-sources",-1458295962),new cljs.core.Keyword(null,"to","to",192099007),shadow.cljs.devtools.client.env.worker_client_id,new cljs.core.Keyword(null,"sources","sources",-321166424),cljs.core.into.cljs$core$IFn$_invoke$arity$3(cljs.core.PersistentVector.EMPTY,cljs.core.map.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"resource-id","resource-id",-1308422582)),sources_to_load)], null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"cljs-sources","cljs-sources",31121610),(function (p__46689){
var map__46692 = p__46689;
var map__46692__$1 = (((((!((map__46692 == null))))?(((((map__46692.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46692.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46692):map__46692);
var msg__$1 = map__46692__$1;
var sources__$1 = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46692__$1,new cljs.core.Keyword(null,"sources","sources",-321166424));
try{shadow.cljs.devtools.client.browser.do_js_load(sources__$1);

if(cljs.core.seq(js_requires)){
shadow.cljs.devtools.client.browser.do_js_requires(js_requires);
} else {
}

return (done.cljs$core$IFn$_invoke$arity$1 ? done.cljs$core$IFn$_invoke$arity$1(sources_to_load) : done.call(null,sources_to_load));
}catch (e46694){var ex = e46694;
return (error.cljs$core$IFn$_invoke$arity$1 ? error.cljs$core$IFn$_invoke$arity$1(ex) : error.call(null,ex));
}})], null));
}
}));

shadow.cljs.devtools.client.shared.add_plugin_BANG_(new cljs.core.Keyword("shadow.cljs.devtools.client.browser","client","shadow.cljs.devtools.client.browser/client",-1461019282),cljs.core.PersistentHashSet.EMPTY,(function (p__46695){
var map__46696 = p__46695;
var map__46696__$1 = (((((!((map__46696 == null))))?(((((map__46696.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46696.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46696):map__46696);
var env = map__46696__$1;
var runtime = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46696__$1,new cljs.core.Keyword(null,"runtime","runtime",-1331573996));
var svc = new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"runtime","runtime",-1331573996),runtime], null);
shadow.remote.runtime.api.add_extension(runtime,new cljs.core.Keyword("shadow.cljs.devtools.client.browser","client","shadow.cljs.devtools.client.browser/client",-1461019282),new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"on-welcome","on-welcome",1895317125),(function (){
cljs.core.reset_BANG_(shadow.cljs.devtools.client.browser.ws_was_welcome_ref,true);

shadow.cljs.devtools.client.hud.connection_error_clear_BANG_();

shadow.cljs.devtools.client.env.patch_goog_BANG_();

return shadow.cljs.devtools.client.browser.devtools_msg(["#",cljs.core.str.cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"client-id","client-id",-464622140).cljs$core$IFn$_invoke$arity$1(cljs.core.deref(new cljs.core.Keyword(null,"state-ref","state-ref",2127874952).cljs$core$IFn$_invoke$arity$1(runtime))))," ready!"].join(''));
}),new cljs.core.Keyword(null,"on-disconnect","on-disconnect",-809021814),(function (e){
if(cljs.core.truth_(cljs.core.deref(shadow.cljs.devtools.client.browser.ws_was_welcome_ref))){
shadow.cljs.devtools.client.hud.connection_error("The Websocket connection was closed!");

return cljs.core.reset_BANG_(shadow.cljs.devtools.client.browser.ws_was_welcome_ref,false);
} else {
return null;
}
}),new cljs.core.Keyword(null,"on-reconnect","on-reconnect",1239988702),(function (e){
return shadow.cljs.devtools.client.hud.connection_error("Reconnecting ...");
}),new cljs.core.Keyword(null,"ops","ops",1237330063),new cljs.core.PersistentArrayMap(null, 8, [new cljs.core.Keyword(null,"access-denied","access-denied",959449406),(function (msg){
cljs.core.reset_BANG_(shadow.cljs.devtools.client.browser.ws_was_welcome_ref,false);

return shadow.cljs.devtools.client.hud.connection_error(["Stale Output! Your loaded JS was not produced by the running shadow-cljs instance."," Is the watch for this build running?"].join(''));
}),new cljs.core.Keyword(null,"cljs-runtime-init","cljs-runtime-init",1305890232),(function (msg){
return shadow.cljs.devtools.client.browser.repl_init(runtime,msg);
}),new cljs.core.Keyword(null,"cljs-asset-update","cljs-asset-update",1224093028),(function (p__46702){
var map__46703 = p__46702;
var map__46703__$1 = (((((!((map__46703 == null))))?(((((map__46703.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46703.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46703):map__46703);
var msg = map__46703__$1;
var updates = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46703__$1,new cljs.core.Keyword(null,"updates","updates",2013983452));
return shadow.cljs.devtools.client.browser.handle_asset_update(msg);
}),new cljs.core.Keyword(null,"cljs-build-configure","cljs-build-configure",-2089891268),(function (msg){
return null;
}),new cljs.core.Keyword(null,"cljs-build-start","cljs-build-start",-725781241),(function (msg){
shadow.cljs.devtools.client.hud.hud_hide();

shadow.cljs.devtools.client.hud.load_start();

return shadow.cljs.devtools.client.env.run_custom_notify_BANG_(cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(msg,new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"build-start","build-start",-959649480)));
}),new cljs.core.Keyword(null,"cljs-build-complete","cljs-build-complete",273626153),(function (msg){
var msg__$1 = shadow.cljs.devtools.client.env.add_warnings_to_info(msg);
shadow.cljs.devtools.client.hud.hud_warnings(msg__$1);

shadow.cljs.devtools.client.browser.handle_build_complete(runtime,msg__$1);

return shadow.cljs.devtools.client.env.run_custom_notify_BANG_(cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(msg__$1,new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"build-complete","build-complete",-501868472)));
}),new cljs.core.Keyword(null,"cljs-build-failure","cljs-build-failure",1718154990),(function (msg){
shadow.cljs.devtools.client.hud.load_end();

shadow.cljs.devtools.client.hud.hud_error(msg);

return shadow.cljs.devtools.client.env.run_custom_notify_BANG_(cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(msg,new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"build-failure","build-failure",-2107487466)));
}),new cljs.core.Keyword("shadow.cljs.devtools.client.env","worker-notify","shadow.cljs.devtools.client.env/worker-notify",-1456820670),(function (p__46709){
var map__46710 = p__46709;
var map__46710__$1 = (((((!((map__46710 == null))))?(((((map__46710.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46710.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46710):map__46710);
var event_op = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46710__$1,new cljs.core.Keyword(null,"event-op","event-op",200358057));
var client_id = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46710__$1,new cljs.core.Keyword(null,"client-id","client-id",-464622140));
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"client-disconnect","client-disconnect",640227957),event_op)) && (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(client_id,shadow.cljs.devtools.client.env.worker_client_id)))){
shadow.cljs.devtools.client.hud.connection_error_clear_BANG_();

return shadow.cljs.devtools.client.hud.connection_error("The watch for this build was stopped!");
} else {
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"client-connect","client-connect",-1113973888),event_op)){
shadow.cljs.devtools.client.hud.connection_error_clear_BANG_();

return shadow.cljs.devtools.client.hud.connection_error("The watch for this build was restarted. Reload required!");
} else {
return null;
}
}
})], null)], null));

return svc;
}),(function (p__46712){
var map__46713 = p__46712;
var map__46713__$1 = (((((!((map__46713 == null))))?(((((map__46713.cljs$lang$protocol_mask$partition0$ & (64))) || ((cljs.core.PROTOCOL_SENTINEL === map__46713.cljs$core$ISeq$))))?true:false):false))?cljs.core.apply.cljs$core$IFn$_invoke$arity$2(cljs.core.hash_map,map__46713):map__46713);
var svc = map__46713__$1;
var runtime = cljs.core.get.cljs$core$IFn$_invoke$arity$2(map__46713__$1,new cljs.core.Keyword(null,"runtime","runtime",-1331573996));
return shadow.remote.runtime.api.del_extension(runtime,new cljs.core.Keyword("shadow.cljs.devtools.client.browser","client","shadow.cljs.devtools.client.browser/client",-1461019282));
}));

shadow.cljs.devtools.client.shared.init_runtime_BANG_(shadow.cljs.devtools.client.browser.client_info,shadow.cljs.devtools.client.websocket.start,shadow.cljs.devtools.client.websocket.send,shadow.cljs.devtools.client.websocket.stop);
} else {
}

//# sourceMappingURL=shadow.cljs.devtools.client.browser.js.map
