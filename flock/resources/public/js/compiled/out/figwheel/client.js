// Compiled by ClojureScript 0.0-2202
goog.provide('figwheel.client');
goog.require('cljs.core');
goog.require('cljs.reader');
goog.require('cljs.core.async');
goog.require('clojure.string');
goog.require('clojure.string');
goog.require('cljs.core.async');
goog.require('cljs.core.async');
goog.require('cljs.reader');
goog.require('goog.net.jsloader');
goog.require('goog.net.jsloader');
figwheel.client.log_style = "color:rgb(0,128,0);";
/**
* @param {...*} var_args
*/
figwheel.client.log = (function() { 
var log__delegate = function (p__15681,args){var map__15683 = p__15681;var map__15683__$1 = ((cljs.core.seq_QMARK_.call(null,map__15683))?cljs.core.apply.call(null,cljs.core.hash_map,map__15683):map__15683);var debug = cljs.core.get.call(null,map__15683__$1,new cljs.core.Keyword(null,"debug","debug",1109363141));if(cljs.core.truth_(debug))
{return console.log(cljs.core.to_array.call(null,args));
} else
{return null;
}
};
var log = function (p__15681,var_args){
var args = null;if (arguments.length > 1) {
  args = cljs.core.array_seq(Array.prototype.slice.call(arguments, 1),0);} 
return log__delegate.call(this,p__15681,args);};
log.cljs$lang$maxFixedArity = 1;
log.cljs$lang$applyTo = (function (arglist__15684){
var p__15681 = cljs.core.first(arglist__15684);
var args = cljs.core.rest(arglist__15684);
return log__delegate(p__15681,args);
});
log.cljs$core$IFn$_invoke$arity$variadic = log__delegate;
return log;
})()
;
figwheel.client.add_cache_buster = (function add_cache_buster(url){return [cljs.core.str(url),cljs.core.str("?rel="),cljs.core.str((new Date()).getTime())].join('');
});
figwheel.client.js_reload = (function js_reload(p__15685,callback){var map__15687 = p__15685;var map__15687__$1 = ((cljs.core.seq_QMARK_.call(null,map__15687))?cljs.core.apply.call(null,cljs.core.hash_map,map__15687):map__15687);var msg = map__15687__$1;var dependency_file = cljs.core.get.call(null,map__15687__$1,new cljs.core.Keyword(null,"dependency-file","dependency-file",2750516784));var namespace = cljs.core.get.call(null,map__15687__$1,new cljs.core.Keyword(null,"namespace","namespace",2266122445));var file = cljs.core.get.call(null,map__15687__$1,new cljs.core.Keyword(null,"file","file",1017047278));if(cljs.core.truth_((function (){var or__7874__auto__ = dependency_file;if(cljs.core.truth_(or__7874__auto__))
{return or__7874__auto__;
} else
{return goog.isProvided_(namespace);
}
})()))
{return goog.net.jsloader.load(figwheel.client.add_cache_buster.call(null,file)).addCallback(((function (map__15687,map__15687__$1,msg,dependency_file,namespace,file){
return (function (){return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [file], null));
});})(map__15687,map__15687__$1,msg,dependency_file,namespace,file))
);
} else
{return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [false], null));
}
});
figwheel.client.reload_js_file = (function reload_js_file(file_msg){var out = cljs.core.async.chan.call(null);figwheel.client.js_reload.call(null,file_msg,((function (out){
return (function (url){cljs.core.async.put_BANG_.call(null,out,url);
return cljs.core.async.close_BANG_.call(null,out);
});})(out))
);
return out;
});
figwheel.client.load_all_js_files = (function load_all_js_files(files){return cljs.core.async.into.call(null,cljs.core.PersistentVector.EMPTY,cljs.core.async.filter_LT_.call(null,cljs.core.identity,cljs.core.async.merge.call(null,cljs.core.mapv.call(null,figwheel.client.reload_js_file,files))));
});
figwheel.client.reload_js_files = (function reload_js_files(p__15688,callback){var map__15724 = p__15688;var map__15724__$1 = ((cljs.core.seq_QMARK_.call(null,map__15724))?cljs.core.apply.call(null,cljs.core.hash_map,map__15724):map__15724);var files = cljs.core.get.call(null,map__15724__$1,new cljs.core.Keyword(null,"files","files",1111338473));var c__11627__auto__ = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto__,map__15724,map__15724__$1,files){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto__,map__15724,map__15724__$1,files){
return (function (state_15743){var state_val_15744 = (state_15743[1]);if((state_val_15744 === 6))
{var inst_15727 = (state_15743[7]);var inst_15735 = (state_15743[2]);var inst_15736 = [inst_15727];var inst_15737 = (new cljs.core.PersistentVector(null,1,5,cljs.core.PersistentVector.EMPTY_NODE,inst_15736,null));var inst_15738 = cljs.core.apply.call(null,callback,inst_15737);var state_15743__$1 = (function (){var statearr_15745 = state_15743;(statearr_15745[8] = inst_15735);
return statearr_15745;
})();var statearr_15746_15759 = state_15743__$1;(statearr_15746_15759[2] = inst_15738);
(statearr_15746_15759[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_15744 === 5))
{var inst_15741 = (state_15743[2]);var state_15743__$1 = state_15743;return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_15743__$1,inst_15741);
} else
{if((state_val_15744 === 4))
{var state_15743__$1 = state_15743;var statearr_15747_15760 = state_15743__$1;(statearr_15747_15760[2] = null);
(statearr_15747_15760[1] = 5);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_15744 === 3))
{var inst_15727 = (state_15743[7]);var inst_15730 = console.log("%cFigwheel: loading these files",figwheel.client.log_style);var inst_15731 = cljs.core.clj__GT_js.call(null,inst_15727);var inst_15732 = console.log(inst_15731);var inst_15733 = cljs.core.async.timeout.call(null,10);var state_15743__$1 = (function (){var statearr_15748 = state_15743;(statearr_15748[9] = inst_15732);
(statearr_15748[10] = inst_15730);
return statearr_15748;
})();return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_15743__$1,6,inst_15733);
} else
{if((state_val_15744 === 2))
{var inst_15727 = (state_15743[7]);var inst_15727__$1 = (state_15743[2]);var inst_15728 = cljs.core.not_empty.call(null,inst_15727__$1);var state_15743__$1 = (function (){var statearr_15749 = state_15743;(statearr_15749[7] = inst_15727__$1);
return statearr_15749;
})();if(cljs.core.truth_(inst_15728))
{var statearr_15750_15761 = state_15743__$1;(statearr_15750_15761[1] = 3);
} else
{var statearr_15751_15762 = state_15743__$1;(statearr_15751_15762[1] = 4);
}
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if((state_val_15744 === 1))
{var inst_15725 = figwheel.client.load_all_js_files.call(null,files);var state_15743__$1 = state_15743;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_15743__$1,2,inst_15725);
} else
{return null;
}
}
}
}
}
}
});})(c__11627__auto__,map__15724,map__15724__$1,files))
;return ((function (switch__11563__auto__,c__11627__auto__,map__15724,map__15724__$1,files){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_15755 = [null,null,null,null,null,null,null,null,null,null,null];(statearr_15755[0] = state_machine__11564__auto__);
(statearr_15755[1] = 1);
return statearr_15755;
});
var state_machine__11564__auto____1 = (function (state_15743){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_15743);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e15756){if((e15756 instanceof Object))
{var ex__11567__auto__ = e15756;var statearr_15757_15763 = state_15743;(statearr_15757_15763[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_15743);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e15756;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__15764 = state_15743;
state_15743 = G__15764;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_15743){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_15743);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto__,map__15724,map__15724__$1,files))
})();var state__11629__auto__ = (function (){var statearr_15758 = f__11628__auto__.call(null);(statearr_15758[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto__);
return statearr_15758;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto__,map__15724,map__15724__$1,files))
);
return c__11627__auto__;
});
figwheel.client.current_links = (function current_links(){return Array.prototype.slice.call(document.getElementsByTagName("link"));
});
figwheel.client.matches_file_QMARK_ = (function matches_file_QMARK_(css_path,link_href){return cljs.core._EQ_.call(null,css_path,clojure.string.replace_first.call(null,clojure.string.replace_first.call(null,cljs.core.first.call(null,clojure.string.split.call(null,link_href,/\?/)),[cljs.core.str(location.protocol),cljs.core.str("//")].join(''),""),location.host,""));
});
figwheel.client.get_correct_link = (function get_correct_link(css_path){return cljs.core.some.call(null,(function (l){if(figwheel.client.matches_file_QMARK_.call(null,css_path,l.href))
{return l;
} else
{return null;
}
}),figwheel.client.current_links.call(null));
});
figwheel.client.clone_link = (function clone_link(link,url){var clone = document.createElement("link");clone.rel = "stylesheet";
clone.media = link.media;
clone.disabled = link.disabled;
clone.href = figwheel.client.add_cache_buster.call(null,url);
return clone;
});
figwheel.client.create_link = (function create_link(url){var link = document.createElement("link");link.rel = "stylesheet";
link.href = figwheel.client.add_cache_buster.call(null,url);
return link;
});
figwheel.client.add_link_to_doc = (function() {
var add_link_to_doc = null;
var add_link_to_doc__1 = (function (new_link){return (document.getElementsByTagName("head")[0]).appendChild(new_link);
});
var add_link_to_doc__2 = (function (orig_link,klone){var parent = orig_link.parentNode;if(cljs.core._EQ_.call(null,orig_link,parent.lastChild))
{parent.appendChild(klone);
} else
{parent.insertBefore(klone,orig_link.nextSibling);
}
var c__11627__auto__ = cljs.core.async.chan.call(null,1);cljs.core.async.impl.dispatch.run.call(null,((function (c__11627__auto__,parent){
return (function (){var f__11628__auto__ = (function (){var switch__11563__auto__ = ((function (c__11627__auto__,parent){
return (function (state_15785){var state_val_15786 = (state_15785[1]);if((state_val_15786 === 2))
{var inst_15782 = (state_15785[2]);var inst_15783 = parent.removeChild(orig_link);var state_15785__$1 = (function (){var statearr_15787 = state_15785;(statearr_15787[7] = inst_15782);
return statearr_15787;
})();return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_15785__$1,inst_15783);
} else
{if((state_val_15786 === 1))
{var inst_15780 = cljs.core.async.timeout.call(null,200);var state_15785__$1 = state_15785;return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_15785__$1,2,inst_15780);
} else
{return null;
}
}
});})(c__11627__auto__,parent))
;return ((function (switch__11563__auto__,c__11627__auto__,parent){
return (function() {
var state_machine__11564__auto__ = null;
var state_machine__11564__auto____0 = (function (){var statearr_15791 = [null,null,null,null,null,null,null,null];(statearr_15791[0] = state_machine__11564__auto__);
(statearr_15791[1] = 1);
return statearr_15791;
});
var state_machine__11564__auto____1 = (function (state_15785){while(true){
var ret_value__11565__auto__ = (function (){try{while(true){
var result__11566__auto__ = switch__11563__auto__.call(null,state_15785);if(cljs.core.keyword_identical_QMARK_.call(null,result__11566__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
continue;
}
} else
{return result__11566__auto__;
}
break;
}
}catch (e15792){if((e15792 instanceof Object))
{var ex__11567__auto__ = e15792;var statearr_15793_15795 = state_15785;(statearr_15793_15795[5] = ex__11567__auto__);
cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_15785);
return new cljs.core.Keyword(null,"recur","recur",1122293407);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{throw e15792;
} else
{return null;
}
}
}})();if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__11565__auto__,new cljs.core.Keyword(null,"recur","recur",1122293407)))
{{
var G__15796 = state_15785;
state_15785 = G__15796;
continue;
}
} else
{return ret_value__11565__auto__;
}
break;
}
});
state_machine__11564__auto__ = function(state_15785){
switch(arguments.length){
case 0:
return state_machine__11564__auto____0.call(this);
case 1:
return state_machine__11564__auto____1.call(this,state_15785);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$0 = state_machine__11564__auto____0;
state_machine__11564__auto__.cljs$core$IFn$_invoke$arity$1 = state_machine__11564__auto____1;
return state_machine__11564__auto__;
})()
;})(switch__11563__auto__,c__11627__auto__,parent))
})();var state__11629__auto__ = (function (){var statearr_15794 = f__11628__auto__.call(null);(statearr_15794[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__11627__auto__);
return statearr_15794;
})();return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__11629__auto__);
});})(c__11627__auto__,parent))
);
return c__11627__auto__;
});
add_link_to_doc = function(orig_link,klone){
switch(arguments.length){
case 1:
return add_link_to_doc__1.call(this,orig_link);
case 2:
return add_link_to_doc__2.call(this,orig_link,klone);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
add_link_to_doc.cljs$core$IFn$_invoke$arity$1 = add_link_to_doc__1;
add_link_to_doc.cljs$core$IFn$_invoke$arity$2 = add_link_to_doc__2;
return add_link_to_doc;
})()
;
figwheel.client.reload_css_file = (function reload_css_file(p__15797){var map__15799 = p__15797;var map__15799__$1 = ((cljs.core.seq_QMARK_.call(null,map__15799))?cljs.core.apply.call(null,cljs.core.hash_map,map__15799):map__15799);var file = cljs.core.get.call(null,map__15799__$1,new cljs.core.Keyword(null,"file","file",1017047278));var temp__4124__auto__ = figwheel.client.get_correct_link.call(null,file);if(cljs.core.truth_(temp__4124__auto__))
{var link = temp__4124__auto__;return figwheel.client.add_link_to_doc.call(null,link,figwheel.client.clone_link.call(null,link,file));
} else
{return figwheel.client.add_link_to_doc.call(null,figwheel.client.create_link.call(null,file));
}
});
figwheel.client.reload_css_files = (function reload_css_files(files_msg,jsload_callback){var seq__15804_15808 = cljs.core.seq.call(null,new cljs.core.Keyword(null,"files","files",1111338473).cljs$core$IFn$_invoke$arity$1(files_msg));var chunk__15805_15809 = null;var count__15806_15810 = 0;var i__15807_15811 = 0;while(true){
if((i__15807_15811 < count__15806_15810))
{var f_15812 = cljs.core._nth.call(null,chunk__15805_15809,i__15807_15811);figwheel.client.reload_css_file.call(null,f_15812);
{
var G__15813 = seq__15804_15808;
var G__15814 = chunk__15805_15809;
var G__15815 = count__15806_15810;
var G__15816 = (i__15807_15811 + 1);
seq__15804_15808 = G__15813;
chunk__15805_15809 = G__15814;
count__15806_15810 = G__15815;
i__15807_15811 = G__15816;
continue;
}
} else
{var temp__4126__auto___15817 = cljs.core.seq.call(null,seq__15804_15808);if(temp__4126__auto___15817)
{var seq__15804_15818__$1 = temp__4126__auto___15817;if(cljs.core.chunked_seq_QMARK_.call(null,seq__15804_15818__$1))
{var c__8622__auto___15819 = cljs.core.chunk_first.call(null,seq__15804_15818__$1);{
var G__15820 = cljs.core.chunk_rest.call(null,seq__15804_15818__$1);
var G__15821 = c__8622__auto___15819;
var G__15822 = cljs.core.count.call(null,c__8622__auto___15819);
var G__15823 = 0;
seq__15804_15808 = G__15820;
chunk__15805_15809 = G__15821;
count__15806_15810 = G__15822;
i__15807_15811 = G__15823;
continue;
}
} else
{var f_15824 = cljs.core.first.call(null,seq__15804_15818__$1);figwheel.client.reload_css_file.call(null,f_15824);
{
var G__15825 = cljs.core.next.call(null,seq__15804_15818__$1);
var G__15826 = null;
var G__15827 = 0;
var G__15828 = 0;
seq__15804_15808 = G__15825;
chunk__15805_15809 = G__15826;
count__15806_15810 = G__15827;
i__15807_15811 = G__15828;
continue;
}
}
} else
{}
}
break;
}
console.log("%cFigwheel: loaded CSS files",figwheel.client.log_style);
return console.log(cljs.core.clj__GT_js.call(null,cljs.core.map.call(null,new cljs.core.Keyword(null,"file","file",1017047278),new cljs.core.Keyword(null,"files","files",1111338473).cljs$core$IFn$_invoke$arity$1(files_msg))));
});
figwheel.client.figwheel_closure_import_script = (function figwheel_closure_import_script(src){if(cljs.core.truth_(goog.inHtmlDocument_()))
{goog.net.jsloader.load(figwheel.client.add_cache_buster.call(null,src));
return true;
} else
{return false;
}
});
figwheel.client.patch_goog_base = (function patch_goog_base(){goog.provide = goog.exportPath_;
return goog.global.CLOSURE_IMPORT_SCRIPT = figwheel.client.figwheel_closure_import_script;
});
figwheel.client.watch_and_reload_STAR_ = (function watch_and_reload_STAR_(p__15829){var map__15834 = p__15829;var map__15834__$1 = ((cljs.core.seq_QMARK_.call(null,map__15834))?cljs.core.apply.call(null,cljs.core.hash_map,map__15834):map__15834);var opts = map__15834__$1;var jsload_callback = cljs.core.get.call(null,map__15834__$1,new cljs.core.Keyword(null,"jsload-callback","jsload-callback",3126035989));var websocket_url = cljs.core.get.call(null,map__15834__$1,new cljs.core.Keyword(null,"websocket-url","websocket-url",633671131));var retry_count = cljs.core.get.call(null,map__15834__$1,new cljs.core.Keyword(null,"retry-count","retry-count",2949373212));console.log("%cFigwheel: trying to open cljs reload socket",figwheel.client.log_style);
var socket = (new WebSocket(websocket_url));socket.onmessage = ((function (socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (msg_str){var msg = cljs.reader.read_string.call(null,msg_str.data);var pred__15835 = cljs.core._EQ_;var expr__15836 = new cljs.core.Keyword(null,"msg-name","msg-name",3979112649).cljs$core$IFn$_invoke$arity$1(msg);if(cljs.core.truth_(pred__15835.call(null,new cljs.core.Keyword(null,"files-changed","files-changed",2807270608),expr__15836)))
{return figwheel.client.reload_js_files.call(null,msg,jsload_callback);
} else
{if(cljs.core.truth_(pred__15835.call(null,new cljs.core.Keyword(null,"css-files-changed","css-files-changed",1058553478),expr__15836)))
{return figwheel.client.reload_css_files.call(null,msg,jsload_callback);
} else
{return null;
}
}
});})(socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count))
;
socket.onopen = ((function (socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (x){figwheel.client.patch_goog_base.call(null);
return console.log("%cFigwheel: socket connection established",figwheel.client.log_style);
});})(socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count))
;
socket.onclose = ((function (socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (x){figwheel.client.log.call(null,opts,"Figwheel: socket closed or failed to open");
if((retry_count > 0))
{return window.setTimeout(((function (socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (){return watch_and_reload_STAR_.call(null,cljs.core.assoc.call(null,opts,new cljs.core.Keyword(null,"retry-count","retry-count",2949373212),(retry_count - 1)));
});})(socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count))
,2000);
} else
{return null;
}
});})(socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count))
;
return socket.onerror = ((function (socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (x){return figwheel.client.log.call(null,opts,"Figwheel: socket error ");
});})(socket,map__15834,map__15834__$1,opts,jsload_callback,websocket_url,retry_count))
;
});
/**
* @param {...*} var_args
*/
figwheel.client.watch_and_reload = (function() { 
var watch_and_reload__delegate = function (p__15838){var map__15842 = p__15838;var map__15842__$1 = ((cljs.core.seq_QMARK_.call(null,map__15842))?cljs.core.apply.call(null,cljs.core.hash_map,map__15842):map__15842);var opts = map__15842__$1;var jsload_callback = cljs.core.get.call(null,map__15842__$1,new cljs.core.Keyword(null,"jsload-callback","jsload-callback",3126035989));var websocket_url = cljs.core.get.call(null,map__15842__$1,new cljs.core.Keyword(null,"websocket-url","websocket-url",633671131));var retry_count = cljs.core.get.call(null,map__15842__$1,new cljs.core.Keyword(null,"retry-count","retry-count",2949373212));if(cljs.core.truth_(figwheel.client.hasOwnProperty("watch_and_reload_singleton")))
{return null;
} else
{figwheel.client.watch_and_reload_singleton = figwheel.client.watch_and_reload_STAR_.call(null,cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"retry-count","retry-count",2949373212),100,new cljs.core.Keyword(null,"jsload-callback","jsload-callback",3126035989),((function (map__15842,map__15842__$1,opts,jsload_callback,websocket_url,retry_count){
return (function (url){return document.querySelector("body").dispatchEvent((new CustomEvent("figwheel.js-reload",(function (){var obj15844 = {"detail":url};return obj15844;
})())));
});})(map__15842,map__15842__$1,opts,jsload_callback,websocket_url,retry_count))
,new cljs.core.Keyword(null,"websocket-url","websocket-url",633671131),[cljs.core.str("ws:"),cljs.core.str(location.host),cljs.core.str("/figwheel-ws")].join('')], null),opts));
}
};
var watch_and_reload = function (var_args){
var p__15838 = null;if (arguments.length > 0) {
  p__15838 = cljs.core.array_seq(Array.prototype.slice.call(arguments, 0),0);} 
return watch_and_reload__delegate.call(this,p__15838);};
watch_and_reload.cljs$lang$maxFixedArity = 0;
watch_and_reload.cljs$lang$applyTo = (function (arglist__15845){
var p__15838 = cljs.core.seq(arglist__15845);
return watch_and_reload__delegate(p__15838);
});
watch_and_reload.cljs$core$IFn$_invoke$arity$variadic = watch_and_reload__delegate;
return watch_and_reload;
})()
;
