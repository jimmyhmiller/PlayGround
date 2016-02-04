// Compiled by ClojureScript 1.7.170 {}
goog.provide('devcard.core');
goog.require('cljs.core');
goog.require('om.core');
goog.require('sablono.core');
cljs.core.enable_console_print_BANG_.call(null);
devcard.core.widget = (function devcard$core$widget(data,owner){
if(typeof devcard.core.t_devcard$core22240 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {om.core.IRender}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
devcard.core.t_devcard$core22240 = (function (widget,data,owner,meta22241){
this.widget = widget;
this.data = data;
this.owner = owner;
this.meta22241 = meta22241;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
devcard.core.t_devcard$core22240.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_22242,meta22241__$1){
var self__ = this;
var _22242__$1 = this;
return (new devcard.core.t_devcard$core22240(self__.widget,self__.data,self__.owner,meta22241__$1));
});

devcard.core.t_devcard$core22240.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_22242){
var self__ = this;
var _22242__$1 = this;
return self__.meta22241;
});

devcard.core.t_devcard$core22240.prototype.om$core$IRender$ = true;

devcard.core.t_devcard$core22240.prototype.om$core$IRender$render$arity$1 = (function (this__21623__auto__){
var self__ = this;
var this__21623__auto____$1 = this;
return React.createElement("h2",null,"This is an om card, ",sablono.interpreter.interpret.call(null,new cljs.core.Keyword(null,"text","text",-1790561697).cljs$core$IFn$_invoke$arity$1(self__.data)));
});

devcard.core.t_devcard$core22240.getBasis = (function (){
return new cljs.core.PersistentVector(null, 4, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"widget","widget",786562584,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"arglists","arglists",1661989754),cljs.core.list(new cljs.core.Symbol(null,"quote","quote",1377916282,null),cljs.core.list(new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"data","data",1407862150,null),new cljs.core.Symbol(null,"owner","owner",1247919588,null)], null)))], null)),new cljs.core.Symbol(null,"data","data",1407862150,null),new cljs.core.Symbol(null,"owner","owner",1247919588,null),new cljs.core.Symbol(null,"meta22241","meta22241",-1107105213,null)], null);
});

devcard.core.t_devcard$core22240.cljs$lang$type = true;

devcard.core.t_devcard$core22240.cljs$lang$ctorStr = "devcard.core/t_devcard$core22240";

devcard.core.t_devcard$core22240.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"devcard.core/t_devcard$core22240");
});

devcard.core.__GT_t_devcard$core22240 = (function devcard$core$widget_$___GT_t_devcard$core22240(widget__$1,data__$1,owner__$1,meta22241){
return (new devcard.core.t_devcard$core22240(widget__$1,data__$1,owner__$1,meta22241));
});

}

return (new devcard.core.t_devcard$core22240(devcard$core$widget,data,owner,null));
});
devcards.core.register_card.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"path","path",-188191168),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"devcard.core","devcard.core",1086767139),new cljs.core.Keyword(null,"omcard-ex","omcard-ex",1187696134)], null),new cljs.core.Keyword(null,"func","func",-238706040),(function (){
return devcards.core.card_base.call(null,new cljs.core.PersistentArrayMap(null, 5, [new cljs.core.Keyword(null,"name","name",1843675177),"omcard-ex",new cljs.core.Keyword(null,"documentation","documentation",1889593999),null,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742),(function (){
if(typeof devcard.core.t_devcard$core22243 !== 'undefined'){
} else {

/**
* @constructor
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
 * @implements {devcards.core.IDevcardOptions}
*/
devcard.core.t_devcard$core22243 = (function (meta22244){
this.meta22244 = meta22244;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
})
devcard.core.t_devcard$core22243.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_22245,meta22244__$1){
var self__ = this;
var _22245__$1 = this;
return (new devcard.core.t_devcard$core22243(meta22244__$1));
});

devcard.core.t_devcard$core22243.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_22245){
var self__ = this;
var _22245__$1 = this;
return self__.meta22244;
});

devcard.core.t_devcard$core22243.prototype.devcards$core$IDevcardOptions$ = true;

devcard.core.t_devcard$core22243.prototype.devcards$core$IDevcardOptions$_devcard_options$arity$2 = (function (this__20223__auto__,devcard_opts__20224__auto__){
var self__ = this;
var this__20223__auto____$1 = this;
return cljs.core.assoc.call(null,devcard_opts__20224__auto__,new cljs.core.Keyword(null,"main-obj","main-obj",-1544409742),devcards.core.dom_node_STAR_.call(null,((function (this__20223__auto____$1){
return (function (data_atom__20256__auto__,node__20257__auto__){
return om.core.root.call(null,devcard.core.widget,data_atom__20256__auto__,cljs.core.merge.call(null,cljs.core.PersistentArrayMap.EMPTY,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"target","target",253001721),node__20257__auto__], null)));
});})(this__20223__auto____$1))
),new cljs.core.Keyword(null,"options","options",99638489),cljs.core.merge.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"watch-atom","watch-atom",-2134031308),true], null),devcards.core.assert_options_map.call(null,new cljs.core.Keyword(null,"options","options",99638489).cljs$core$IFn$_invoke$arity$1(devcard_opts__20224__auto__))));
});

devcard.core.t_devcard$core22243.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"meta22244","meta22244",-188399662,null)], null);
});

devcard.core.t_devcard$core22243.cljs$lang$type = true;

devcard.core.t_devcard$core22243.cljs$lang$ctorStr = "devcard.core/t_devcard$core22243";

devcard.core.t_devcard$core22243.cljs$lang$ctorPrWriter = (function (this__17364__auto__,writer__17365__auto__,opt__17366__auto__){
return cljs.core._write.call(null,writer__17365__auto__,"devcard.core/t_devcard$core22243");
});

devcard.core.__GT_t_devcard$core22243 = (function devcard$core$__GT_t_devcard$core22243(meta22244){
return (new devcard.core.t_devcard$core22243(meta22244));
});

}

return (new devcard.core.t_devcard$core22243(null));
})()
,new cljs.core.Keyword(null,"initial-data","initial-data",-1315709804),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"text","text",-1790561697),"yep"], null),new cljs.core.Keyword(null,"options","options",99638489),cljs.core.PersistentArrayMap.EMPTY], null));
})], null));
devcard.core.main = (function devcard$core$main(){
var temp__4423__auto__ = document.getElementById("main-app-area");
if(cljs.core.truth_(temp__4423__auto__)){
var node = temp__4423__auto__;
return React.render(React.createElement("div",null,"This is working"),node);
} else {
return null;
}
});
devcard.core.main.call(null);

//# sourceMappingURL=core.js.map?rel=1454621589237