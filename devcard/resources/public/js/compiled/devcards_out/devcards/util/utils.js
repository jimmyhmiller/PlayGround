// Compiled by ClojureScript 1.7.170 {}
goog.provide('devcards.util.utils');
goog.require('cljs.core');
goog.require('cljs.pprint');
devcards.util.utils.html_env_QMARK_ = (function devcards$util$utils$html_env_QMARK_(){
var temp__4423__auto__ = goog.global.document;
if(cljs.core.truth_(temp__4423__auto__)){
var doc = temp__4423__auto__;
return (doc["write"]);
} else {
return null;
}
});
devcards.util.utils.node_env_QMARK_ = (function devcards$util$utils$node_env_QMARK_(){
return !((goog.nodeGlobalRequire == null));
});
devcards.util.utils.pprint_str = (function devcards$util$utils$pprint_str(obj){
try{var sb__17740__auto__ = (new goog.string.StringBuffer());
var _STAR_print_newline_STAR_22905_22907 = cljs.core._STAR_print_newline_STAR_;
var _STAR_print_fn_STAR_22906_22908 = cljs.core._STAR_print_fn_STAR_;
cljs.core._STAR_print_newline_STAR_ = true;

cljs.core._STAR_print_fn_STAR_ = ((function (_STAR_print_newline_STAR_22905_22907,_STAR_print_fn_STAR_22906_22908,sb__17740__auto__){
return (function (x__17741__auto__){
return sb__17740__auto__.append(x__17741__auto__);
});})(_STAR_print_newline_STAR_22905_22907,_STAR_print_fn_STAR_22906_22908,sb__17740__auto__))
;

try{cljs.pprint.pprint.call(null,obj);
}finally {cljs.core._STAR_print_fn_STAR_ = _STAR_print_fn_STAR_22906_22908;

cljs.core._STAR_print_newline_STAR_ = _STAR_print_newline_STAR_22905_22907;
}
return [cljs.core.str(sb__17740__auto__)].join('');
}catch (e22903){if((e22903 instanceof Error)){
var e1 = e22903;
try{return obj.toString();
}catch (e22904){if((e22904 instanceof Error)){
var e2 = e22904;
return [cljs.core.str("<<Un-printable Type>>")].join('');
} else {
throw e22904;

}
}} else {
throw e22903;

}
}});
devcards.util.utils.pprint_code = (function devcards$util$utils$pprint_code(code){
var _STAR_print_pprint_dispatch_STAR_22910 = cljs.pprint._STAR_print_pprint_dispatch_STAR_;
cljs.pprint._STAR_print_pprint_dispatch_STAR_ = cljs.pprint.code_dispatch;

try{return devcards.util.utils.pprint_str.call(null,code);
}finally {cljs.pprint._STAR_print_pprint_dispatch_STAR_ = _STAR_print_pprint_dispatch_STAR_22910;
}});

//# sourceMappingURL=utils.js.map?rel=1454621290920