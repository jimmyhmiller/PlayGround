// Compiled by ClojureScript 1.7.170 {}
goog.provide('figwheel.client');
goog.require('cljs.core');
goog.require('goog.userAgent.product');
goog.require('goog.Uri');
goog.require('cljs.core.async');
goog.require('figwheel.client.socket');
goog.require('figwheel.client.file_reloading');
goog.require('clojure.string');
goog.require('figwheel.client.utils');
goog.require('cljs.repl');
goog.require('figwheel.client.heads_up');
figwheel.client.figwheel_repl_print = (function figwheel$client$figwheel_repl_print(args){
figwheel.client.socket.send_BANG_.call(null,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"figwheel-event","figwheel-event",519570592),"callback",new cljs.core.Keyword(null,"callback-name","callback-name",336964714),"figwheel-repl-print",new cljs.core.Keyword(null,"content","content",15833224),args], null));

return args;
});
figwheel.client.autoload_QMARK_ = (cljs.core.truth_(figwheel.client.utils.html_env_QMARK_.call(null))?(function (){
var pred__26778 = cljs.core._EQ_;
var expr__26779 = (function (){var or__16766__auto__ = (function (){try{return localStorage.getItem("figwheel_autoload");
}catch (e26782){if((e26782 instanceof Error)){
var e = e26782;
return false;
} else {
throw e26782;

}
}})();
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return "true";
}
})();
if(cljs.core.truth_(pred__26778.call(null,"true",expr__26779))){
return true;
} else {
if(cljs.core.truth_(pred__26778.call(null,"false",expr__26779))){
return false;
} else {
throw (new Error([cljs.core.str("No matching clause: "),cljs.core.str(expr__26779)].join('')));
}
}
}):(function (){
return true;
}));
figwheel.client.toggle_autoload = (function figwheel$client$toggle_autoload(){
if(cljs.core.truth_(figwheel.client.utils.html_env_QMARK_.call(null))){
try{localStorage.setItem("figwheel_autoload",cljs.core.not.call(null,figwheel.client.autoload_QMARK_.call(null)));

return figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),[cljs.core.str("Figwheel autoloading "),cljs.core.str((cljs.core.truth_(figwheel.client.autoload_QMARK_.call(null))?"ON":"OFF"))].join(''));
}catch (e26784){if((e26784 instanceof Error)){
var e = e26784;
return figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),[cljs.core.str("Unable to access localStorage")].join(''));
} else {
throw e26784;

}
}} else {
return null;
}
});
goog.exportSymbol('figwheel.client.toggle_autoload', figwheel.client.toggle_autoload);
figwheel.client.console_print = (function figwheel$client$console_print(args){
console.log.apply(console,cljs.core.into_array.call(null,args));

return args;
});
figwheel.client.repl_print_fn = (function figwheel$client$repl_print_fn(var_args){
var args__17831__auto__ = [];
var len__17824__auto___26786 = arguments.length;
var i__17825__auto___26787 = (0);
while(true){
if((i__17825__auto___26787 < len__17824__auto___26786)){
args__17831__auto__.push((arguments[i__17825__auto___26787]));

var G__26788 = (i__17825__auto___26787 + (1));
i__17825__auto___26787 = G__26788;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return figwheel.client.repl_print_fn.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

figwheel.client.repl_print_fn.cljs$core$IFn$_invoke$arity$variadic = (function (args){
figwheel.client.figwheel_repl_print.call(null,figwheel.client.console_print.call(null,args));

return null;
});

figwheel.client.repl_print_fn.cljs$lang$maxFixedArity = (0);

figwheel.client.repl_print_fn.cljs$lang$applyTo = (function (seq26785){
return figwheel.client.repl_print_fn.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq26785));
});
figwheel.client.enable_repl_print_BANG_ = (function figwheel$client$enable_repl_print_BANG_(){
cljs.core._STAR_print_newline_STAR_ = false;

return cljs.core._STAR_print_fn_STAR_ = figwheel.client.repl_print_fn;
});
figwheel.client.get_essential_messages = (function figwheel$client$get_essential_messages(ed){
if(cljs.core.truth_(ed)){
return cljs.core.cons.call(null,cljs.core.select_keys.call(null,ed,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"message","message",-406056002),new cljs.core.Keyword(null,"class","class",-2030961996)], null)),figwheel$client$get_essential_messages.call(null,new cljs.core.Keyword(null,"cause","cause",231901252).cljs$core$IFn$_invoke$arity$1(ed)));
} else {
return null;
}
});
figwheel.client.error_msg_format = (function figwheel$client$error_msg_format(p__26789){
var map__26792 = p__26789;
var map__26792__$1 = ((((!((map__26792 == null)))?((((map__26792.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26792.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26792):map__26792);
var message = cljs.core.get.call(null,map__26792__$1,new cljs.core.Keyword(null,"message","message",-406056002));
var class$ = cljs.core.get.call(null,map__26792__$1,new cljs.core.Keyword(null,"class","class",-2030961996));
return [cljs.core.str(class$),cljs.core.str(" : "),cljs.core.str(message)].join('');
});
figwheel.client.format_messages = cljs.core.comp.call(null,cljs.core.partial.call(null,cljs.core.map,figwheel.client.error_msg_format),figwheel.client.get_essential_messages);
figwheel.client.focus_msgs = (function figwheel$client$focus_msgs(name_set,msg_hist){
return cljs.core.cons.call(null,cljs.core.first.call(null,msg_hist),cljs.core.filter.call(null,cljs.core.comp.call(null,name_set,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863)),cljs.core.rest.call(null,msg_hist)));
});
figwheel.client.reload_file_QMARK__STAR_ = (function figwheel$client$reload_file_QMARK__STAR_(msg_name,opts){
var or__16766__auto__ = new cljs.core.Keyword(null,"load-warninged-code","load-warninged-code",-2030345223).cljs$core$IFn$_invoke$arity$1(opts);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core.not_EQ_.call(null,msg_name,new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356));
}
});
figwheel.client.reload_file_state_QMARK_ = (function figwheel$client$reload_file_state_QMARK_(msg_names,opts){
var and__16754__auto__ = cljs.core._EQ_.call(null,cljs.core.first.call(null,msg_names),new cljs.core.Keyword(null,"files-changed","files-changed",-1418200563));
if(and__16754__auto__){
return figwheel.client.reload_file_QMARK__STAR_.call(null,cljs.core.second.call(null,msg_names),opts);
} else {
return and__16754__auto__;
}
});
figwheel.client.block_reload_file_state_QMARK_ = (function figwheel$client$block_reload_file_state_QMARK_(msg_names,opts){
return (cljs.core._EQ_.call(null,cljs.core.first.call(null,msg_names),new cljs.core.Keyword(null,"files-changed","files-changed",-1418200563))) && (cljs.core.not.call(null,figwheel.client.reload_file_QMARK__STAR_.call(null,cljs.core.second.call(null,msg_names),opts)));
});
figwheel.client.warning_append_state_QMARK_ = (function figwheel$client$warning_append_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356)], null),cljs.core.take.call(null,(2),msg_names));
});
figwheel.client.warning_state_QMARK_ = (function figwheel$client$warning_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),cljs.core.first.call(null,msg_names));
});
figwheel.client.rewarning_state_QMARK_ = (function figwheel$client$rewarning_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),new cljs.core.Keyword(null,"files-changed","files-changed",-1418200563),new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356)], null),cljs.core.take.call(null,(3),msg_names));
});
figwheel.client.compile_fail_state_QMARK_ = (function figwheel$client$compile_fail_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"compile-failed","compile-failed",-477639289),cljs.core.first.call(null,msg_names));
});
figwheel.client.compile_refail_state_QMARK_ = (function figwheel$client$compile_refail_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"compile-failed","compile-failed",-477639289),new cljs.core.Keyword(null,"compile-failed","compile-failed",-477639289)], null),cljs.core.take.call(null,(2),msg_names));
});
figwheel.client.css_loaded_state_QMARK_ = (function figwheel$client$css_loaded_state_QMARK_(msg_names){
return cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"css-files-changed","css-files-changed",720773874),cljs.core.first.call(null,msg_names));
});
figwheel.client.file_reloader_plugin = (function figwheel$client$file_reloader_plugin(opts){
var ch = cljs.core.async.chan.call(null);
var c__18933__auto___26954 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___26954,ch){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___26954,ch){
return (function (state_26923){
var state_val_26924 = (state_26923[(1)]);
if((state_val_26924 === (7))){
var inst_26919 = (state_26923[(2)]);
var state_26923__$1 = state_26923;
var statearr_26925_26955 = state_26923__$1;
(statearr_26925_26955[(2)] = inst_26919);

(statearr_26925_26955[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (1))){
var state_26923__$1 = state_26923;
var statearr_26926_26956 = state_26923__$1;
(statearr_26926_26956[(2)] = null);

(statearr_26926_26956[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (4))){
var inst_26876 = (state_26923[(7)]);
var inst_26876__$1 = (state_26923[(2)]);
var state_26923__$1 = (function (){var statearr_26927 = state_26923;
(statearr_26927[(7)] = inst_26876__$1);

return statearr_26927;
})();
if(cljs.core.truth_(inst_26876__$1)){
var statearr_26928_26957 = state_26923__$1;
(statearr_26928_26957[(1)] = (5));

} else {
var statearr_26929_26958 = state_26923__$1;
(statearr_26929_26958[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (15))){
var inst_26883 = (state_26923[(8)]);
var inst_26898 = new cljs.core.Keyword(null,"files","files",-472457450).cljs$core$IFn$_invoke$arity$1(inst_26883);
var inst_26899 = cljs.core.first.call(null,inst_26898);
var inst_26900 = new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(inst_26899);
var inst_26901 = [cljs.core.str("Figwheel: Not loading code with warnings - "),cljs.core.str(inst_26900)].join('');
var inst_26902 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"warn","warn",-436710552),inst_26901);
var state_26923__$1 = state_26923;
var statearr_26930_26959 = state_26923__$1;
(statearr_26930_26959[(2)] = inst_26902);

(statearr_26930_26959[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (13))){
var inst_26907 = (state_26923[(2)]);
var state_26923__$1 = state_26923;
var statearr_26931_26960 = state_26923__$1;
(statearr_26931_26960[(2)] = inst_26907);

(statearr_26931_26960[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (6))){
var state_26923__$1 = state_26923;
var statearr_26932_26961 = state_26923__$1;
(statearr_26932_26961[(2)] = null);

(statearr_26932_26961[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (17))){
var inst_26905 = (state_26923[(2)]);
var state_26923__$1 = state_26923;
var statearr_26933_26962 = state_26923__$1;
(statearr_26933_26962[(2)] = inst_26905);

(statearr_26933_26962[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (3))){
var inst_26921 = (state_26923[(2)]);
var state_26923__$1 = state_26923;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_26923__$1,inst_26921);
} else {
if((state_val_26924 === (12))){
var inst_26882 = (state_26923[(9)]);
var inst_26896 = figwheel.client.block_reload_file_state_QMARK_.call(null,inst_26882,opts);
var state_26923__$1 = state_26923;
if(cljs.core.truth_(inst_26896)){
var statearr_26934_26963 = state_26923__$1;
(statearr_26934_26963[(1)] = (15));

} else {
var statearr_26935_26964 = state_26923__$1;
(statearr_26935_26964[(1)] = (16));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (2))){
var state_26923__$1 = state_26923;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_26923__$1,(4),ch);
} else {
if((state_val_26924 === (11))){
var inst_26883 = (state_26923[(8)]);
var inst_26888 = cljs.core.PersistentVector.EMPTY_NODE;
var inst_26889 = figwheel.client.file_reloading.reload_js_files.call(null,opts,inst_26883);
var inst_26890 = cljs.core.async.timeout.call(null,(1000));
var inst_26891 = [inst_26889,inst_26890];
var inst_26892 = (new cljs.core.PersistentVector(null,2,(5),inst_26888,inst_26891,null));
var state_26923__$1 = state_26923;
return cljs.core.async.ioc_alts_BANG_.call(null,state_26923__$1,(14),inst_26892);
} else {
if((state_val_26924 === (9))){
var inst_26883 = (state_26923[(8)]);
var inst_26909 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"warn","warn",-436710552),"Figwheel: code autoloading is OFF");
var inst_26910 = new cljs.core.Keyword(null,"files","files",-472457450).cljs$core$IFn$_invoke$arity$1(inst_26883);
var inst_26911 = cljs.core.map.call(null,new cljs.core.Keyword(null,"file","file",-1269645878),inst_26910);
var inst_26912 = [cljs.core.str("Not loading: "),cljs.core.str(inst_26911)].join('');
var inst_26913 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),inst_26912);
var state_26923__$1 = (function (){var statearr_26936 = state_26923;
(statearr_26936[(10)] = inst_26909);

return statearr_26936;
})();
var statearr_26937_26965 = state_26923__$1;
(statearr_26937_26965[(2)] = inst_26913);

(statearr_26937_26965[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (5))){
var inst_26876 = (state_26923[(7)]);
var inst_26878 = [new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),null,new cljs.core.Keyword(null,"files-changed","files-changed",-1418200563),null];
var inst_26879 = (new cljs.core.PersistentArrayMap(null,2,inst_26878,null));
var inst_26880 = (new cljs.core.PersistentHashSet(null,inst_26879,null));
var inst_26881 = figwheel.client.focus_msgs.call(null,inst_26880,inst_26876);
var inst_26882 = cljs.core.map.call(null,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863),inst_26881);
var inst_26883 = cljs.core.first.call(null,inst_26881);
var inst_26884 = figwheel.client.autoload_QMARK_.call(null);
var state_26923__$1 = (function (){var statearr_26938 = state_26923;
(statearr_26938[(8)] = inst_26883);

(statearr_26938[(9)] = inst_26882);

return statearr_26938;
})();
if(cljs.core.truth_(inst_26884)){
var statearr_26939_26966 = state_26923__$1;
(statearr_26939_26966[(1)] = (8));

} else {
var statearr_26940_26967 = state_26923__$1;
(statearr_26940_26967[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (14))){
var inst_26894 = (state_26923[(2)]);
var state_26923__$1 = state_26923;
var statearr_26941_26968 = state_26923__$1;
(statearr_26941_26968[(2)] = inst_26894);

(statearr_26941_26968[(1)] = (13));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (16))){
var state_26923__$1 = state_26923;
var statearr_26942_26969 = state_26923__$1;
(statearr_26942_26969[(2)] = null);

(statearr_26942_26969[(1)] = (17));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (10))){
var inst_26915 = (state_26923[(2)]);
var state_26923__$1 = (function (){var statearr_26943 = state_26923;
(statearr_26943[(11)] = inst_26915);

return statearr_26943;
})();
var statearr_26944_26970 = state_26923__$1;
(statearr_26944_26970[(2)] = null);

(statearr_26944_26970[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26924 === (8))){
var inst_26882 = (state_26923[(9)]);
var inst_26886 = figwheel.client.reload_file_state_QMARK_.call(null,inst_26882,opts);
var state_26923__$1 = state_26923;
if(cljs.core.truth_(inst_26886)){
var statearr_26945_26971 = state_26923__$1;
(statearr_26945_26971[(1)] = (11));

} else {
var statearr_26946_26972 = state_26923__$1;
(statearr_26946_26972[(1)] = (12));

}

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
});})(c__18933__auto___26954,ch))
;
return ((function (switch__18821__auto__,c__18933__auto___26954,ch){
return (function() {
var figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__ = null;
var figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____0 = (function (){
var statearr_26950 = [null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_26950[(0)] = figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__);

(statearr_26950[(1)] = (1));

return statearr_26950;
});
var figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____1 = (function (state_26923){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_26923);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e26951){if((e26951 instanceof Object)){
var ex__18825__auto__ = e26951;
var statearr_26952_26973 = state_26923;
(statearr_26952_26973[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_26923);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e26951;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__26974 = state_26923;
state_26923 = G__26974;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__ = function(state_26923){
switch(arguments.length){
case 0:
return figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____1.call(this,state_26923);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____0;
figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$file_reloader_plugin_$_state_machine__18822__auto____1;
return figwheel$client$file_reloader_plugin_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___26954,ch))
})();
var state__18935__auto__ = (function (){var statearr_26953 = f__18934__auto__.call(null);
(statearr_26953[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___26954);

return statearr_26953;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___26954,ch))
);


return ((function (ch){
return (function (msg_hist){
cljs.core.async.put_BANG_.call(null,ch,msg_hist);

return msg_hist;
});
;})(ch))
});
figwheel.client.truncate_stack_trace = (function figwheel$client$truncate_stack_trace(stack_str){
return cljs.core.take_while.call(null,(function (p1__26975_SHARP_){
return cljs.core.not.call(null,cljs.core.re_matches.call(null,/.*eval_javascript_STAR__STAR_.*/,p1__26975_SHARP_));
}),clojure.string.split_lines.call(null,stack_str));
});
figwheel.client.get_ua_product = (function figwheel$client$get_ua_product(){
if(cljs.core.truth_(figwheel.client.utils.node_env_QMARK_.call(null))){
return new cljs.core.Keyword(null,"chrome","chrome",1718738387);
} else {
if(cljs.core.truth_(goog.userAgent.product.SAFARI)){
return new cljs.core.Keyword(null,"safari","safari",497115653);
} else {
if(cljs.core.truth_(goog.userAgent.product.CHROME)){
return new cljs.core.Keyword(null,"chrome","chrome",1718738387);
} else {
if(cljs.core.truth_(goog.userAgent.product.FIREFOX)){
return new cljs.core.Keyword(null,"firefox","firefox",1283768880);
} else {
if(cljs.core.truth_(goog.userAgent.product.IE)){
return new cljs.core.Keyword(null,"ie","ie",2038473780);
} else {
return null;
}
}
}
}
}
});
var base_path_26982 = figwheel.client.utils.base_url_path.call(null);
figwheel.client.eval_javascript_STAR__STAR_ = ((function (base_path_26982){
return (function figwheel$client$eval_javascript_STAR__STAR_(code,opts,result_handler){
try{var _STAR_print_fn_STAR_26980 = cljs.core._STAR_print_fn_STAR_;
var _STAR_print_newline_STAR_26981 = cljs.core._STAR_print_newline_STAR_;
cljs.core._STAR_print_fn_STAR_ = figwheel.client.repl_print_fn;

cljs.core._STAR_print_newline_STAR_ = false;

try{return result_handler.call(null,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"status","status",-1997798413),new cljs.core.Keyword(null,"success","success",1890645906),new cljs.core.Keyword(null,"ua-product","ua-product",938384227),figwheel.client.get_ua_product.call(null),new cljs.core.Keyword(null,"value","value",305978217),figwheel.client.utils.eval_helper.call(null,code,opts)], null));
}finally {cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR_26981;

cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR_26980;
}}catch (e26979){if((e26979 instanceof Error)){
var e = e26979;
return result_handler.call(null,new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"status","status",-1997798413),new cljs.core.Keyword(null,"exception","exception",-335277064),new cljs.core.Keyword(null,"value","value",305978217),cljs.core.pr_str.call(null,e),new cljs.core.Keyword(null,"ua-product","ua-product",938384227),figwheel.client.get_ua_product.call(null),new cljs.core.Keyword(null,"stacktrace","stacktrace",-95588394),clojure.string.join.call(null,"\n",figwheel.client.truncate_stack_trace.call(null,e.stack)),new cljs.core.Keyword(null,"base-path","base-path",495760020),base_path_26982], null));
} else {
var e = e26979;
return result_handler.call(null,new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"status","status",-1997798413),new cljs.core.Keyword(null,"exception","exception",-335277064),new cljs.core.Keyword(null,"ua-product","ua-product",938384227),figwheel.client.get_ua_product.call(null),new cljs.core.Keyword(null,"value","value",305978217),cljs.core.pr_str.call(null,e),new cljs.core.Keyword(null,"stacktrace","stacktrace",-95588394),"No stacktrace available."], null));

}
}});})(base_path_26982))
;
/**
 * The REPL can disconnect and reconnect lets ensure cljs.user exists at least.
 */
figwheel.client.ensure_cljs_user = (function figwheel$client$ensure_cljs_user(){
if(cljs.core.truth_(cljs.user)){
return null;
} else {
return cljs.user = {};
}
});
figwheel.client.repl_plugin = (function figwheel$client$repl_plugin(p__26983){
var map__26990 = p__26983;
var map__26990__$1 = ((((!((map__26990 == null)))?((((map__26990.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26990.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26990):map__26990);
var opts = map__26990__$1;
var build_id = cljs.core.get.call(null,map__26990__$1,new cljs.core.Keyword(null,"build-id","build-id",1642831089));
return ((function (map__26990,map__26990__$1,opts,build_id){
return (function (p__26992){
var vec__26993 = p__26992;
var map__26994 = cljs.core.nth.call(null,vec__26993,(0),null);
var map__26994__$1 = ((((!((map__26994 == null)))?((((map__26994.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26994.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26994):map__26994);
var msg = map__26994__$1;
var msg_name = cljs.core.get.call(null,map__26994__$1,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863));
var _ = cljs.core.nthnext.call(null,vec__26993,(1));
if(cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"repl-eval","repl-eval",-1784727398),msg_name)){
figwheel.client.ensure_cljs_user.call(null);

return figwheel.client.eval_javascript_STAR__STAR_.call(null,new cljs.core.Keyword(null,"code","code",1586293142).cljs$core$IFn$_invoke$arity$1(msg),opts,((function (vec__26993,map__26994,map__26994__$1,msg,msg_name,_,map__26990,map__26990__$1,opts,build_id){
return (function (res){
return figwheel.client.socket.send_BANG_.call(null,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"figwheel-event","figwheel-event",519570592),"callback",new cljs.core.Keyword(null,"callback-name","callback-name",336964714),new cljs.core.Keyword(null,"callback-name","callback-name",336964714).cljs$core$IFn$_invoke$arity$1(msg),new cljs.core.Keyword(null,"content","content",15833224),res], null));
});})(vec__26993,map__26994,map__26994__$1,msg,msg_name,_,map__26990,map__26990__$1,opts,build_id))
);
} else {
return null;
}
});
;})(map__26990,map__26990__$1,opts,build_id))
});
figwheel.client.css_reloader_plugin = (function figwheel$client$css_reloader_plugin(opts){
return (function (p__27000){
var vec__27001 = p__27000;
var map__27002 = cljs.core.nth.call(null,vec__27001,(0),null);
var map__27002__$1 = ((((!((map__27002 == null)))?((((map__27002.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27002.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27002):map__27002);
var msg = map__27002__$1;
var msg_name = cljs.core.get.call(null,map__27002__$1,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863));
var _ = cljs.core.nthnext.call(null,vec__27001,(1));
if(cljs.core._EQ_.call(null,msg_name,new cljs.core.Keyword(null,"css-files-changed","css-files-changed",720773874))){
return figwheel.client.file_reloading.reload_css_files.call(null,opts,msg);
} else {
return null;
}
});
});
figwheel.client.compile_fail_warning_plugin = (function figwheel$client$compile_fail_warning_plugin(p__27004){
var map__27014 = p__27004;
var map__27014__$1 = ((((!((map__27014 == null)))?((((map__27014.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27014.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27014):map__27014);
var on_compile_warning = cljs.core.get.call(null,map__27014__$1,new cljs.core.Keyword(null,"on-compile-warning","on-compile-warning",-1195585947));
var on_compile_fail = cljs.core.get.call(null,map__27014__$1,new cljs.core.Keyword(null,"on-compile-fail","on-compile-fail",728013036));
return ((function (map__27014,map__27014__$1,on_compile_warning,on_compile_fail){
return (function (p__27016){
var vec__27017 = p__27016;
var map__27018 = cljs.core.nth.call(null,vec__27017,(0),null);
var map__27018__$1 = ((((!((map__27018 == null)))?((((map__27018.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27018.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27018):map__27018);
var msg = map__27018__$1;
var msg_name = cljs.core.get.call(null,map__27018__$1,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863));
var _ = cljs.core.nthnext.call(null,vec__27017,(1));
var pred__27020 = cljs.core._EQ_;
var expr__27021 = msg_name;
if(cljs.core.truth_(pred__27020.call(null,new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),expr__27021))){
return on_compile_warning.call(null,msg);
} else {
if(cljs.core.truth_(pred__27020.call(null,new cljs.core.Keyword(null,"compile-failed","compile-failed",-477639289),expr__27021))){
return on_compile_fail.call(null,msg);
} else {
return null;
}
}
});
;})(map__27014,map__27014__$1,on_compile_warning,on_compile_fail))
});
figwheel.client.heads_up_plugin_msg_handler = (function figwheel$client$heads_up_plugin_msg_handler(opts,msg_hist_SINGLEQUOTE_){
var msg_hist = figwheel.client.focus_msgs.call(null,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"compile-failed","compile-failed",-477639289),null,new cljs.core.Keyword(null,"compile-warning","compile-warning",43425356),null,new cljs.core.Keyword(null,"files-changed","files-changed",-1418200563),null], null), null),msg_hist_SINGLEQUOTE_);
var msg_names = cljs.core.map.call(null,new cljs.core.Keyword(null,"msg-name","msg-name",-353709863),msg_hist);
var msg = cljs.core.first.call(null,msg_hist);
var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__,msg_hist,msg_names,msg){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__,msg_hist,msg_names,msg){
return (function (state_27237){
var state_val_27238 = (state_27237[(1)]);
if((state_val_27238 === (7))){
var inst_27161 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27161)){
var statearr_27239_27285 = state_27237__$1;
(statearr_27239_27285[(1)] = (8));

} else {
var statearr_27240_27286 = state_27237__$1;
(statearr_27240_27286[(1)] = (9));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (20))){
var inst_27231 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27241_27287 = state_27237__$1;
(statearr_27241_27287[(2)] = inst_27231);

(statearr_27241_27287[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (27))){
var inst_27227 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27242_27288 = state_27237__$1;
(statearr_27242_27288[(2)] = inst_27227);

(statearr_27242_27288[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (1))){
var inst_27154 = figwheel.client.reload_file_state_QMARK_.call(null,msg_names,opts);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27154)){
var statearr_27243_27289 = state_27237__$1;
(statearr_27243_27289[(1)] = (2));

} else {
var statearr_27244_27290 = state_27237__$1;
(statearr_27244_27290[(1)] = (3));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (24))){
var inst_27229 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27245_27291 = state_27237__$1;
(statearr_27245_27291[(2)] = inst_27229);

(statearr_27245_27291[(1)] = (20));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (4))){
var inst_27235 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_27237__$1,inst_27235);
} else {
if((state_val_27238 === (15))){
var inst_27233 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27246_27292 = state_27237__$1;
(statearr_27246_27292[(2)] = inst_27233);

(statearr_27246_27292[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (21))){
var inst_27192 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27247_27293 = state_27237__$1;
(statearr_27247_27293[(2)] = inst_27192);

(statearr_27247_27293[(1)] = (20));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (31))){
var inst_27216 = figwheel.client.css_loaded_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27216)){
var statearr_27248_27294 = state_27237__$1;
(statearr_27248_27294[(1)] = (34));

} else {
var statearr_27249_27295 = state_27237__$1;
(statearr_27249_27295[(1)] = (35));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (32))){
var inst_27225 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27250_27296 = state_27237__$1;
(statearr_27250_27296[(2)] = inst_27225);

(statearr_27250_27296[(1)] = (27));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (33))){
var inst_27214 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27251_27297 = state_27237__$1;
(statearr_27251_27297[(2)] = inst_27214);

(statearr_27251_27297[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (13))){
var inst_27175 = figwheel.client.heads_up.clear.call(null);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(16),inst_27175);
} else {
if((state_val_27238 === (22))){
var inst_27196 = new cljs.core.Keyword(null,"message","message",-406056002).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27197 = figwheel.client.heads_up.append_message.call(null,inst_27196);
var state_27237__$1 = state_27237;
var statearr_27252_27298 = state_27237__$1;
(statearr_27252_27298[(2)] = inst_27197);

(statearr_27252_27298[(1)] = (24));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (36))){
var inst_27223 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27253_27299 = state_27237__$1;
(statearr_27253_27299[(2)] = inst_27223);

(statearr_27253_27299[(1)] = (32));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (29))){
var inst_27207 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27254_27300 = state_27237__$1;
(statearr_27254_27300[(2)] = inst_27207);

(statearr_27254_27300[(1)] = (27));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (6))){
var inst_27156 = (state_27237[(7)]);
var state_27237__$1 = state_27237;
var statearr_27255_27301 = state_27237__$1;
(statearr_27255_27301[(2)] = inst_27156);

(statearr_27255_27301[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (28))){
var inst_27203 = (state_27237[(2)]);
var inst_27204 = new cljs.core.Keyword(null,"message","message",-406056002).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27205 = figwheel.client.heads_up.display_warning.call(null,inst_27204);
var state_27237__$1 = (function (){var statearr_27256 = state_27237;
(statearr_27256[(8)] = inst_27203);

return statearr_27256;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(29),inst_27205);
} else {
if((state_val_27238 === (25))){
var inst_27201 = figwheel.client.heads_up.clear.call(null);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(28),inst_27201);
} else {
if((state_val_27238 === (34))){
var inst_27218 = figwheel.client.heads_up.flash_loaded.call(null);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(37),inst_27218);
} else {
if((state_val_27238 === (17))){
var inst_27183 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27257_27302 = state_27237__$1;
(statearr_27257_27302[(2)] = inst_27183);

(statearr_27257_27302[(1)] = (15));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (3))){
var inst_27173 = figwheel.client.compile_refail_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27173)){
var statearr_27258_27303 = state_27237__$1;
(statearr_27258_27303[(1)] = (13));

} else {
var statearr_27259_27304 = state_27237__$1;
(statearr_27259_27304[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (12))){
var inst_27169 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27260_27305 = state_27237__$1;
(statearr_27260_27305[(2)] = inst_27169);

(statearr_27260_27305[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (2))){
var inst_27156 = (state_27237[(7)]);
var inst_27156__$1 = figwheel.client.autoload_QMARK_.call(null);
var state_27237__$1 = (function (){var statearr_27261 = state_27237;
(statearr_27261[(7)] = inst_27156__$1);

return statearr_27261;
})();
if(cljs.core.truth_(inst_27156__$1)){
var statearr_27262_27306 = state_27237__$1;
(statearr_27262_27306[(1)] = (5));

} else {
var statearr_27263_27307 = state_27237__$1;
(statearr_27263_27307[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (23))){
var inst_27199 = figwheel.client.rewarning_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27199)){
var statearr_27264_27308 = state_27237__$1;
(statearr_27264_27308[(1)] = (25));

} else {
var statearr_27265_27309 = state_27237__$1;
(statearr_27265_27309[(1)] = (26));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (35))){
var state_27237__$1 = state_27237;
var statearr_27266_27310 = state_27237__$1;
(statearr_27266_27310[(2)] = null);

(statearr_27266_27310[(1)] = (36));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (19))){
var inst_27194 = figwheel.client.warning_append_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27194)){
var statearr_27267_27311 = state_27237__$1;
(statearr_27267_27311[(1)] = (22));

} else {
var statearr_27268_27312 = state_27237__$1;
(statearr_27268_27312[(1)] = (23));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (11))){
var inst_27165 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27269_27313 = state_27237__$1;
(statearr_27269_27313[(2)] = inst_27165);

(statearr_27269_27313[(1)] = (10));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (9))){
var inst_27167 = figwheel.client.heads_up.clear.call(null);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(12),inst_27167);
} else {
if((state_val_27238 === (5))){
var inst_27158 = new cljs.core.Keyword(null,"autoload","autoload",-354122500).cljs$core$IFn$_invoke$arity$1(opts);
var state_27237__$1 = state_27237;
var statearr_27270_27314 = state_27237__$1;
(statearr_27270_27314[(2)] = inst_27158);

(statearr_27270_27314[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (14))){
var inst_27185 = figwheel.client.compile_fail_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27185)){
var statearr_27271_27315 = state_27237__$1;
(statearr_27271_27315[(1)] = (18));

} else {
var statearr_27272_27316 = state_27237__$1;
(statearr_27272_27316[(1)] = (19));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (26))){
var inst_27209 = figwheel.client.warning_state_QMARK_.call(null,msg_names);
var state_27237__$1 = state_27237;
if(cljs.core.truth_(inst_27209)){
var statearr_27273_27317 = state_27237__$1;
(statearr_27273_27317[(1)] = (30));

} else {
var statearr_27274_27318 = state_27237__$1;
(statearr_27274_27318[(1)] = (31));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (16))){
var inst_27177 = (state_27237[(2)]);
var inst_27178 = new cljs.core.Keyword(null,"exception-data","exception-data",-512474886).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27179 = figwheel.client.format_messages.call(null,inst_27178);
var inst_27180 = new cljs.core.Keyword(null,"cause","cause",231901252).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27181 = figwheel.client.heads_up.display_error.call(null,inst_27179,inst_27180);
var state_27237__$1 = (function (){var statearr_27275 = state_27237;
(statearr_27275[(9)] = inst_27177);

return statearr_27275;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(17),inst_27181);
} else {
if((state_val_27238 === (30))){
var inst_27211 = new cljs.core.Keyword(null,"message","message",-406056002).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27212 = figwheel.client.heads_up.display_warning.call(null,inst_27211);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(33),inst_27212);
} else {
if((state_val_27238 === (10))){
var inst_27171 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27276_27319 = state_27237__$1;
(statearr_27276_27319[(2)] = inst_27171);

(statearr_27276_27319[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (18))){
var inst_27187 = new cljs.core.Keyword(null,"exception-data","exception-data",-512474886).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27188 = figwheel.client.format_messages.call(null,inst_27187);
var inst_27189 = new cljs.core.Keyword(null,"cause","cause",231901252).cljs$core$IFn$_invoke$arity$1(msg);
var inst_27190 = figwheel.client.heads_up.display_error.call(null,inst_27188,inst_27189);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(21),inst_27190);
} else {
if((state_val_27238 === (37))){
var inst_27220 = (state_27237[(2)]);
var state_27237__$1 = state_27237;
var statearr_27277_27320 = state_27237__$1;
(statearr_27277_27320[(2)] = inst_27220);

(statearr_27277_27320[(1)] = (36));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27238 === (8))){
var inst_27163 = figwheel.client.heads_up.flash_loaded.call(null);
var state_27237__$1 = state_27237;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27237__$1,(11),inst_27163);
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
}
}
}
}
}
}
}
});})(c__18933__auto__,msg_hist,msg_names,msg))
;
return ((function (switch__18821__auto__,c__18933__auto__,msg_hist,msg_names,msg){
return (function() {
var figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__ = null;
var figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____0 = (function (){
var statearr_27281 = [null,null,null,null,null,null,null,null,null,null];
(statearr_27281[(0)] = figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__);

(statearr_27281[(1)] = (1));

return statearr_27281;
});
var figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____1 = (function (state_27237){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_27237);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e27282){if((e27282 instanceof Object)){
var ex__18825__auto__ = e27282;
var statearr_27283_27321 = state_27237;
(statearr_27283_27321[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_27237);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e27282;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__27322 = state_27237;
state_27237 = G__27322;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__ = function(state_27237){
switch(arguments.length){
case 0:
return figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____1.call(this,state_27237);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____0;
figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto____1;
return figwheel$client$heads_up_plugin_msg_handler_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__,msg_hist,msg_names,msg))
})();
var state__18935__auto__ = (function (){var statearr_27284 = f__18934__auto__.call(null);
(statearr_27284[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_27284;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__,msg_hist,msg_names,msg))
);

return c__18933__auto__;
});
figwheel.client.heads_up_plugin = (function figwheel$client$heads_up_plugin(opts){
var ch = cljs.core.async.chan.call(null);
figwheel.client.heads_up_config_options_STAR__STAR_ = opts;

var c__18933__auto___27385 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___27385,ch){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___27385,ch){
return (function (state_27368){
var state_val_27369 = (state_27368[(1)]);
if((state_val_27369 === (1))){
var state_27368__$1 = state_27368;
var statearr_27370_27386 = state_27368__$1;
(statearr_27370_27386[(2)] = null);

(statearr_27370_27386[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27369 === (2))){
var state_27368__$1 = state_27368;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27368__$1,(4),ch);
} else {
if((state_val_27369 === (3))){
var inst_27366 = (state_27368[(2)]);
var state_27368__$1 = state_27368;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_27368__$1,inst_27366);
} else {
if((state_val_27369 === (4))){
var inst_27356 = (state_27368[(7)]);
var inst_27356__$1 = (state_27368[(2)]);
var state_27368__$1 = (function (){var statearr_27371 = state_27368;
(statearr_27371[(7)] = inst_27356__$1);

return statearr_27371;
})();
if(cljs.core.truth_(inst_27356__$1)){
var statearr_27372_27387 = state_27368__$1;
(statearr_27372_27387[(1)] = (5));

} else {
var statearr_27373_27388 = state_27368__$1;
(statearr_27373_27388[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27369 === (5))){
var inst_27356 = (state_27368[(7)]);
var inst_27358 = figwheel.client.heads_up_plugin_msg_handler.call(null,opts,inst_27356);
var state_27368__$1 = state_27368;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27368__$1,(8),inst_27358);
} else {
if((state_val_27369 === (6))){
var state_27368__$1 = state_27368;
var statearr_27374_27389 = state_27368__$1;
(statearr_27374_27389[(2)] = null);

(statearr_27374_27389[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27369 === (7))){
var inst_27364 = (state_27368[(2)]);
var state_27368__$1 = state_27368;
var statearr_27375_27390 = state_27368__$1;
(statearr_27375_27390[(2)] = inst_27364);

(statearr_27375_27390[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_27369 === (8))){
var inst_27360 = (state_27368[(2)]);
var state_27368__$1 = (function (){var statearr_27376 = state_27368;
(statearr_27376[(8)] = inst_27360);

return statearr_27376;
})();
var statearr_27377_27391 = state_27368__$1;
(statearr_27377_27391[(2)] = null);

(statearr_27377_27391[(1)] = (2));


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
});})(c__18933__auto___27385,ch))
;
return ((function (switch__18821__auto__,c__18933__auto___27385,ch){
return (function() {
var figwheel$client$heads_up_plugin_$_state_machine__18822__auto__ = null;
var figwheel$client$heads_up_plugin_$_state_machine__18822__auto____0 = (function (){
var statearr_27381 = [null,null,null,null,null,null,null,null,null];
(statearr_27381[(0)] = figwheel$client$heads_up_plugin_$_state_machine__18822__auto__);

(statearr_27381[(1)] = (1));

return statearr_27381;
});
var figwheel$client$heads_up_plugin_$_state_machine__18822__auto____1 = (function (state_27368){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_27368);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e27382){if((e27382 instanceof Object)){
var ex__18825__auto__ = e27382;
var statearr_27383_27392 = state_27368;
(statearr_27383_27392[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_27368);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e27382;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__27393 = state_27368;
state_27368 = G__27393;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$heads_up_plugin_$_state_machine__18822__auto__ = function(state_27368){
switch(arguments.length){
case 0:
return figwheel$client$heads_up_plugin_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$heads_up_plugin_$_state_machine__18822__auto____1.call(this,state_27368);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$heads_up_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$heads_up_plugin_$_state_machine__18822__auto____0;
figwheel$client$heads_up_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$heads_up_plugin_$_state_machine__18822__auto____1;
return figwheel$client$heads_up_plugin_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___27385,ch))
})();
var state__18935__auto__ = (function (){var statearr_27384 = f__18934__auto__.call(null);
(statearr_27384[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___27385);

return statearr_27384;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___27385,ch))
);


figwheel.client.heads_up.ensure_container.call(null);

return ((function (ch){
return (function (msg_hist){
cljs.core.async.put_BANG_.call(null,ch,msg_hist);

return msg_hist;
});
;})(ch))
});
figwheel.client.enforce_project_plugin = (function figwheel$client$enforce_project_plugin(opts){
return (function (msg_hist){
if(((1) < cljs.core.count.call(null,cljs.core.set.call(null,cljs.core.keep.call(null,new cljs.core.Keyword(null,"project-id","project-id",206449307),cljs.core.take.call(null,(5),msg_hist)))))){
figwheel.client.socket.close_BANG_.call(null);

console.error("Figwheel: message received from different project. Shutting socket down.");

if(cljs.core.truth_(new cljs.core.Keyword(null,"heads-up-display","heads-up-display",-896577202).cljs$core$IFn$_invoke$arity$1(opts))){
var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__){
return (function (state_27414){
var state_val_27415 = (state_27414[(1)]);
if((state_val_27415 === (1))){
var inst_27409 = cljs.core.async.timeout.call(null,(3000));
var state_27414__$1 = state_27414;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_27414__$1,(2),inst_27409);
} else {
if((state_val_27415 === (2))){
var inst_27411 = (state_27414[(2)]);
var inst_27412 = figwheel.client.heads_up.display_system_warning.call(null,"Connection from different project","Shutting connection down!!!!!");
var state_27414__$1 = (function (){var statearr_27416 = state_27414;
(statearr_27416[(7)] = inst_27411);

return statearr_27416;
})();
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_27414__$1,inst_27412);
} else {
return null;
}
}
});})(c__18933__auto__))
;
return ((function (switch__18821__auto__,c__18933__auto__){
return (function() {
var figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__ = null;
var figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____0 = (function (){
var statearr_27420 = [null,null,null,null,null,null,null,null];
(statearr_27420[(0)] = figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__);

(statearr_27420[(1)] = (1));

return statearr_27420;
});
var figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____1 = (function (state_27414){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_27414);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e27421){if((e27421 instanceof Object)){
var ex__18825__auto__ = e27421;
var statearr_27422_27424 = state_27414;
(statearr_27422_27424[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_27414);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e27421;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__27425 = state_27414;
state_27414 = G__27425;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__ = function(state_27414){
switch(arguments.length){
case 0:
return figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____1.call(this,state_27414);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____0;
figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$enforce_project_plugin_$_state_machine__18822__auto____1;
return figwheel$client$enforce_project_plugin_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__))
})();
var state__18935__auto__ = (function (){var statearr_27423 = f__18934__auto__.call(null);
(statearr_27423[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_27423;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__))
);

return c__18933__auto__;
} else {
return null;
}
} else {
return null;
}
});
});
figwheel.client.default_on_jsload = cljs.core.identity;
figwheel.client.default_on_compile_fail = (function figwheel$client$default_on_compile_fail(p__27426){
var map__27433 = p__27426;
var map__27433__$1 = ((((!((map__27433 == null)))?((((map__27433.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27433.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27433):map__27433);
var ed = map__27433__$1;
var formatted_exception = cljs.core.get.call(null,map__27433__$1,new cljs.core.Keyword(null,"formatted-exception","formatted-exception",-116489026));
var exception_data = cljs.core.get.call(null,map__27433__$1,new cljs.core.Keyword(null,"exception-data","exception-data",-512474886));
var cause = cljs.core.get.call(null,map__27433__$1,new cljs.core.Keyword(null,"cause","cause",231901252));
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: Compile Exception");

var seq__27435_27439 = cljs.core.seq.call(null,figwheel.client.format_messages.call(null,exception_data));
var chunk__27436_27440 = null;
var count__27437_27441 = (0);
var i__27438_27442 = (0);
while(true){
if((i__27438_27442 < count__27437_27441)){
var msg_27443 = cljs.core._nth.call(null,chunk__27436_27440,i__27438_27442);
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),msg_27443);

var G__27444 = seq__27435_27439;
var G__27445 = chunk__27436_27440;
var G__27446 = count__27437_27441;
var G__27447 = (i__27438_27442 + (1));
seq__27435_27439 = G__27444;
chunk__27436_27440 = G__27445;
count__27437_27441 = G__27446;
i__27438_27442 = G__27447;
continue;
} else {
var temp__4425__auto___27448 = cljs.core.seq.call(null,seq__27435_27439);
if(temp__4425__auto___27448){
var seq__27435_27449__$1 = temp__4425__auto___27448;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__27435_27449__$1)){
var c__17569__auto___27450 = cljs.core.chunk_first.call(null,seq__27435_27449__$1);
var G__27451 = cljs.core.chunk_rest.call(null,seq__27435_27449__$1);
var G__27452 = c__17569__auto___27450;
var G__27453 = cljs.core.count.call(null,c__17569__auto___27450);
var G__27454 = (0);
seq__27435_27439 = G__27451;
chunk__27436_27440 = G__27452;
count__27437_27441 = G__27453;
i__27438_27442 = G__27454;
continue;
} else {
var msg_27455 = cljs.core.first.call(null,seq__27435_27449__$1);
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),msg_27455);

var G__27456 = cljs.core.next.call(null,seq__27435_27449__$1);
var G__27457 = null;
var G__27458 = (0);
var G__27459 = (0);
seq__27435_27439 = G__27456;
chunk__27436_27440 = G__27457;
count__27437_27441 = G__27458;
i__27438_27442 = G__27459;
continue;
}
} else {
}
}
break;
}

if(cljs.core.truth_(cause)){
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),[cljs.core.str("Error on file "),cljs.core.str(new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(cause)),cljs.core.str(", line "),cljs.core.str(new cljs.core.Keyword(null,"line","line",212345235).cljs$core$IFn$_invoke$arity$1(cause)),cljs.core.str(", column "),cljs.core.str(new cljs.core.Keyword(null,"column","column",2078222095).cljs$core$IFn$_invoke$arity$1(cause))].join(''));
} else {
}

return ed;
});
figwheel.client.default_on_compile_warning = (function figwheel$client$default_on_compile_warning(p__27460){
var map__27463 = p__27460;
var map__27463__$1 = ((((!((map__27463 == null)))?((((map__27463.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27463.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27463):map__27463);
var w = map__27463__$1;
var message = cljs.core.get.call(null,map__27463__$1,new cljs.core.Keyword(null,"message","message",-406056002));
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"warn","warn",-436710552),[cljs.core.str("Figwheel: Compile Warning - "),cljs.core.str(message)].join(''));

return w;
});
figwheel.client.default_before_load = (function figwheel$client$default_before_load(files){
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: notified of file changes");

return files;
});
figwheel.client.default_on_cssload = (function figwheel$client$default_on_cssload(files){
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: loaded CSS files");

figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"info","info",-317069002),cljs.core.pr_str.call(null,cljs.core.map.call(null,new cljs.core.Keyword(null,"file","file",-1269645878),files)));

return files;
});
if(typeof figwheel.client.config_defaults !== 'undefined'){
} else {
figwheel.client.config_defaults = cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"on-compile-warning","on-compile-warning",-1195585947),new cljs.core.Keyword(null,"on-jsload","on-jsload",-395756602),new cljs.core.Keyword(null,"reload-dependents","reload-dependents",-956865430),new cljs.core.Keyword(null,"on-compile-fail","on-compile-fail",728013036),new cljs.core.Keyword(null,"debug","debug",-1608172596),new cljs.core.Keyword(null,"heads-up-display","heads-up-display",-896577202),new cljs.core.Keyword(null,"websocket-url","websocket-url",-490444938),new cljs.core.Keyword(null,"before-jsload","before-jsload",-847513128),new cljs.core.Keyword(null,"load-warninged-code","load-warninged-code",-2030345223),new cljs.core.Keyword(null,"eval-fn","eval-fn",-1111644294),new cljs.core.Keyword(null,"retry-count","retry-count",1936122875),new cljs.core.Keyword(null,"autoload","autoload",-354122500),new cljs.core.Keyword(null,"on-cssload","on-cssload",1825432318)],[figwheel.client.default_on_compile_warning,figwheel.client.default_on_jsload,true,figwheel.client.default_on_compile_fail,false,true,[cljs.core.str("ws://"),cljs.core.str((cljs.core.truth_(figwheel.client.utils.html_env_QMARK_.call(null))?location.host:"localhost:3449")),cljs.core.str("/figwheel-ws")].join(''),figwheel.client.default_before_load,false,false,(100),true,figwheel.client.default_on_cssload]);
}
figwheel.client.handle_deprecated_jsload_callback = (function figwheel$client$handle_deprecated_jsload_callback(config){
if(cljs.core.truth_(new cljs.core.Keyword(null,"jsload-callback","jsload-callback",-1949628369).cljs$core$IFn$_invoke$arity$1(config))){
return cljs.core.dissoc.call(null,cljs.core.assoc.call(null,config,new cljs.core.Keyword(null,"on-jsload","on-jsload",-395756602),new cljs.core.Keyword(null,"jsload-callback","jsload-callback",-1949628369).cljs$core$IFn$_invoke$arity$1(config)),new cljs.core.Keyword(null,"jsload-callback","jsload-callback",-1949628369));
} else {
return config;
}
});
figwheel.client.base_plugins = (function figwheel$client$base_plugins(system_options){
var base = new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"enforce-project-plugin","enforce-project-plugin",959402899),figwheel.client.enforce_project_plugin,new cljs.core.Keyword(null,"file-reloader-plugin","file-reloader-plugin",-1792964733),figwheel.client.file_reloader_plugin,new cljs.core.Keyword(null,"comp-fail-warning-plugin","comp-fail-warning-plugin",634311),figwheel.client.compile_fail_warning_plugin,new cljs.core.Keyword(null,"css-reloader-plugin","css-reloader-plugin",2002032904),figwheel.client.css_reloader_plugin,new cljs.core.Keyword(null,"repl-plugin","repl-plugin",-1138952371),figwheel.client.repl_plugin], null);
var base__$1 = ((cljs.core.not.call(null,figwheel.client.utils.html_env_QMARK_.call(null)))?cljs.core.select_keys.call(null,base,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"file-reloader-plugin","file-reloader-plugin",-1792964733),new cljs.core.Keyword(null,"comp-fail-warning-plugin","comp-fail-warning-plugin",634311),new cljs.core.Keyword(null,"repl-plugin","repl-plugin",-1138952371)], null)):base);
var base__$2 = ((new cljs.core.Keyword(null,"autoload","autoload",-354122500).cljs$core$IFn$_invoke$arity$1(system_options) === false)?cljs.core.dissoc.call(null,base__$1,new cljs.core.Keyword(null,"file-reloader-plugin","file-reloader-plugin",-1792964733)):base__$1);
if(cljs.core.truth_((function (){var and__16754__auto__ = new cljs.core.Keyword(null,"heads-up-display","heads-up-display",-896577202).cljs$core$IFn$_invoke$arity$1(system_options);
if(cljs.core.truth_(and__16754__auto__)){
return figwheel.client.utils.html_env_QMARK_.call(null);
} else {
return and__16754__auto__;
}
})())){
return cljs.core.assoc.call(null,base__$2,new cljs.core.Keyword(null,"heads-up-display-plugin","heads-up-display-plugin",1745207501),figwheel.client.heads_up_plugin);
} else {
return base__$2;
}
});
figwheel.client.add_message_watch = (function figwheel$client$add_message_watch(key,callback){
return cljs.core.add_watch.call(null,figwheel.client.socket.message_history_atom,key,(function (_,___$1,___$2,msg_hist){
return callback.call(null,cljs.core.first.call(null,msg_hist));
}));
});
figwheel.client.add_plugins = (function figwheel$client$add_plugins(plugins,system_options){
var seq__27471 = cljs.core.seq.call(null,plugins);
var chunk__27472 = null;
var count__27473 = (0);
var i__27474 = (0);
while(true){
if((i__27474 < count__27473)){
var vec__27475 = cljs.core._nth.call(null,chunk__27472,i__27474);
var k = cljs.core.nth.call(null,vec__27475,(0),null);
var plugin = cljs.core.nth.call(null,vec__27475,(1),null);
if(cljs.core.truth_(plugin)){
var pl_27477 = plugin.call(null,system_options);
cljs.core.add_watch.call(null,figwheel.client.socket.message_history_atom,k,((function (seq__27471,chunk__27472,count__27473,i__27474,pl_27477,vec__27475,k,plugin){
return (function (_,___$1,___$2,msg_hist){
return pl_27477.call(null,msg_hist);
});})(seq__27471,chunk__27472,count__27473,i__27474,pl_27477,vec__27475,k,plugin))
);
} else {
}

var G__27478 = seq__27471;
var G__27479 = chunk__27472;
var G__27480 = count__27473;
var G__27481 = (i__27474 + (1));
seq__27471 = G__27478;
chunk__27472 = G__27479;
count__27473 = G__27480;
i__27474 = G__27481;
continue;
} else {
var temp__4425__auto__ = cljs.core.seq.call(null,seq__27471);
if(temp__4425__auto__){
var seq__27471__$1 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__27471__$1)){
var c__17569__auto__ = cljs.core.chunk_first.call(null,seq__27471__$1);
var G__27482 = cljs.core.chunk_rest.call(null,seq__27471__$1);
var G__27483 = c__17569__auto__;
var G__27484 = cljs.core.count.call(null,c__17569__auto__);
var G__27485 = (0);
seq__27471 = G__27482;
chunk__27472 = G__27483;
count__27473 = G__27484;
i__27474 = G__27485;
continue;
} else {
var vec__27476 = cljs.core.first.call(null,seq__27471__$1);
var k = cljs.core.nth.call(null,vec__27476,(0),null);
var plugin = cljs.core.nth.call(null,vec__27476,(1),null);
if(cljs.core.truth_(plugin)){
var pl_27486 = plugin.call(null,system_options);
cljs.core.add_watch.call(null,figwheel.client.socket.message_history_atom,k,((function (seq__27471,chunk__27472,count__27473,i__27474,pl_27486,vec__27476,k,plugin,seq__27471__$1,temp__4425__auto__){
return (function (_,___$1,___$2,msg_hist){
return pl_27486.call(null,msg_hist);
});})(seq__27471,chunk__27472,count__27473,i__27474,pl_27486,vec__27476,k,plugin,seq__27471__$1,temp__4425__auto__))
);
} else {
}

var G__27487 = cljs.core.next.call(null,seq__27471__$1);
var G__27488 = null;
var G__27489 = (0);
var G__27490 = (0);
seq__27471 = G__27487;
chunk__27472 = G__27488;
count__27473 = G__27489;
i__27474 = G__27490;
continue;
}
} else {
return null;
}
}
break;
}
});
figwheel.client.start = (function figwheel$client$start(var_args){
var args27491 = [];
var len__17824__auto___27494 = arguments.length;
var i__17825__auto___27495 = (0);
while(true){
if((i__17825__auto___27495 < len__17824__auto___27494)){
args27491.push((arguments[i__17825__auto___27495]));

var G__27496 = (i__17825__auto___27495 + (1));
i__17825__auto___27495 = G__27496;
continue;
} else {
}
break;
}

var G__27493 = args27491.length;
switch (G__27493) {
case 1:
return figwheel.client.start.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 0:
return figwheel.client.start.cljs$core$IFn$_invoke$arity$0();

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args27491.length)].join('')));

}
});

figwheel.client.start.cljs$core$IFn$_invoke$arity$1 = (function (opts){
if((goog.dependencies_ == null)){
return null;
} else {
if(typeof figwheel.client.__figwheel_start_once__ !== 'undefined'){
return null;
} else {
figwheel.client.__figwheel_start_once__ = setTimeout((function (){
var plugins_SINGLEQUOTE_ = new cljs.core.Keyword(null,"plugins","plugins",1900073717).cljs$core$IFn$_invoke$arity$1(opts);
var merge_plugins = new cljs.core.Keyword(null,"merge-plugins","merge-plugins",-1193912370).cljs$core$IFn$_invoke$arity$1(opts);
var system_options = figwheel.client.handle_deprecated_jsload_callback.call(null,cljs.core.merge.call(null,figwheel.client.config_defaults,cljs.core.dissoc.call(null,opts,new cljs.core.Keyword(null,"plugins","plugins",1900073717),new cljs.core.Keyword(null,"merge-plugins","merge-plugins",-1193912370))));
var plugins = (cljs.core.truth_(plugins_SINGLEQUOTE_)?plugins_SINGLEQUOTE_:cljs.core.merge.call(null,figwheel.client.base_plugins.call(null,system_options),merge_plugins));
figwheel.client.utils._STAR_print_debug_STAR_ = new cljs.core.Keyword(null,"debug","debug",-1608172596).cljs$core$IFn$_invoke$arity$1(opts);

figwheel.client.add_plugins.call(null,plugins,system_options);

figwheel.client.file_reloading.patch_goog_base.call(null);

return figwheel.client.socket.open.call(null,system_options);
}));
}
}
});

figwheel.client.start.cljs$core$IFn$_invoke$arity$0 = (function (){
return figwheel.client.start.call(null,cljs.core.PersistentArrayMap.EMPTY);
});

figwheel.client.start.cljs$lang$maxFixedArity = 1;
figwheel.client.watch_and_reload_with_opts = figwheel.client.start;
figwheel.client.watch_and_reload = (function figwheel$client$watch_and_reload(var_args){
var args__17831__auto__ = [];
var len__17824__auto___27502 = arguments.length;
var i__17825__auto___27503 = (0);
while(true){
if((i__17825__auto___27503 < len__17824__auto___27502)){
args__17831__auto__.push((arguments[i__17825__auto___27503]));

var G__27504 = (i__17825__auto___27503 + (1));
i__17825__auto___27503 = G__27504;
continue;
} else {
}
break;
}

var argseq__17832__auto__ = ((((0) < args__17831__auto__.length))?(new cljs.core.IndexedSeq(args__17831__auto__.slice((0)),(0))):null);
return figwheel.client.watch_and_reload.cljs$core$IFn$_invoke$arity$variadic(argseq__17832__auto__);
});

figwheel.client.watch_and_reload.cljs$core$IFn$_invoke$arity$variadic = (function (p__27499){
var map__27500 = p__27499;
var map__27500__$1 = ((((!((map__27500 == null)))?((((map__27500.cljs$lang$protocol_mask$partition0$ & (64))) || (map__27500.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__27500):map__27500);
var opts = map__27500__$1;
return figwheel.client.start.call(null,opts);
});

figwheel.client.watch_and_reload.cljs$lang$maxFixedArity = (0);

figwheel.client.watch_and_reload.cljs$lang$applyTo = (function (seq27498){
return figwheel.client.watch_and_reload.cljs$core$IFn$_invoke$arity$variadic(cljs.core.seq.call(null,seq27498));
});

//# sourceMappingURL=client.js.map?rel=1454621294367