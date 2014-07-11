// Compiled by ClojureScript 0.0-2202
goog.provide('flock.core');
goog.require('cljs.core');
goog.require('sablono.core');
goog.require('sablono.core');
goog.require('om.dom');
goog.require('om.dom');
goog.require('om.core');
goog.require('om.core');
goog.require('figwheel.client');
goog.require('figwheel.client');
cljs.core.enable_console_print_BANG_.call(null);
if(typeof flock.core.pi !== 'undefined')
{} else
{flock.core.pi = Math.PI;
}
if(typeof flock.core.MAXACC !== 'undefined')
{} else
{flock.core.MAXACC = 10;
}
if(typeof flock.core.MAXDISTANCE !== 'undefined')
{} else
{flock.core.MAXDISTANCE = 100;
}
if(typeof flock.core.MAXSTEER !== 'undefined')
{} else
{flock.core.MAXSTEER = (flock.core.pi / 100);
}
if(typeof flock.core.SCALE !== 'undefined')
{} else
{flock.core.SCALE = 1;
}
flock.core.square = (function square(x){return (x * x);
});
flock.core.y_comp = (function y_comp(p__22661){var map__22663 = p__22661;var map__22663__$1 = ((cljs.core.seq_QMARK_.call(null,map__22663))?cljs.core.apply.call(null,cljs.core.hash_map,map__22663):map__22663);var O = cljs.core.get.call(null,map__22663__$1,new cljs.core.Keyword(null,"O","O",1013904321));var v = cljs.core.get.call(null,map__22663__$1,new cljs.core.Keyword(null,"v","v",1013904360));var com = (v * Math.cos.call(null,O));return (-1 * com);
});
flock.core.x_comp = (function x_comp(p__22664){var map__22666 = p__22664;var map__22666__$1 = ((cljs.core.seq_QMARK_.call(null,map__22666))?cljs.core.apply.call(null,cljs.core.hash_map,map__22666):map__22666);var O = cljs.core.get.call(null,map__22666__$1,new cljs.core.Keyword(null,"O","O",1013904321));var v = cljs.core.get.call(null,map__22666__$1,new cljs.core.Keyword(null,"v","v",1013904360));var com = (v * Math.sin.call(null,O));return com;
});
flock.core.move = (function move(x,vx,a,t){return ((x + (vx * t)) + ((.5 * a) * flock.core.square.call(null,t)));
});
flock.core.move_x = (function move_x(p__22667,t){var map__22669 = p__22667;var map__22669__$1 = ((cljs.core.seq_QMARK_.call(null,map__22669))?cljs.core.apply.call(null,cljs.core.hash_map,map__22669):map__22669);var v = cljs.core.get.call(null,map__22669__$1,new cljs.core.Keyword(null,"v","v",1013904360));var x = cljs.core.get.call(null,map__22669__$1,new cljs.core.Keyword(null,"x","x",1013904362));var vx = flock.core.x_comp.call(null,v);return flock.core.move.call(null,x,vx,0,t);
});
flock.core.move_y = (function move_y(p__22670,t){var map__22672 = p__22670;var map__22672__$1 = ((cljs.core.seq_QMARK_.call(null,map__22672))?cljs.core.apply.call(null,cljs.core.hash_map,map__22672):map__22672);var v = cljs.core.get.call(null,map__22672__$1,new cljs.core.Keyword(null,"v","v",1013904360));var y = cljs.core.get.call(null,map__22672__$1,new cljs.core.Keyword(null,"y","y",1013904363));var vy = flock.core.y_comp.call(null,v);return flock.core.move.call(null,y,vy,0,t);
});
flock.core.avg = (function avg(coll){return (cljs.core.reduce.call(null,cljs.core._PLUS_,coll) / cljs.core.count.call(null,coll));
});
flock.core.direction = (function direction(object){return new cljs.core.Keyword(null,"O","O",1013904321).cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"v","v",1013904360).cljs$core$IFn$_invoke$arity$1(object));
});
flock.core.avg_heading = (function avg_heading(coll){return flock.core.avg.call(null,cljs.core.map.call(null,flock.core.direction,coll));
});
flock.core.distance = (function distance(object1,object2){return Math.sqrt.call(null,(flock.core.square.call(null,(new cljs.core.Keyword(null,"x","x",1013904362).cljs$core$IFn$_invoke$arity$1(object1) - new cljs.core.Keyword(null,"x","x",1013904362).cljs$core$IFn$_invoke$arity$1(object2))) + flock.core.square.call(null,(new cljs.core.Keyword(null,"y","y",1013904363).cljs$core$IFn$_invoke$arity$1(object1) - new cljs.core.Keyword(null,"y","y",1013904363).cljs$core$IFn$_invoke$arity$1(object2)))));
});
flock.core.neighbors = (function neighbors(object,coll){return cljs.core.filter.call(null,(function (p1__22673_SHARP_){return ((flock.core.distance.call(null,object,p1__22673_SHARP_) < flock.core.MAXDISTANCE)) && (cljs.core.not_EQ_.call(null,flock.core.distance.call(null,object,p1__22673_SHARP_),0));
}),coll);
});
flock.core.steer = (function steer(object,coll){var direction = flock.core.direction.call(null,object);var heading = flock.core.avg_heading.call(null,flock.core.neighbors.call(null,object,coll));var distance = Math.abs.call(null,(direction - heading));if((distance < flock.core.MAXSTEER))
{return distance;
} else
{if((direction < heading))
{return flock.core.MAXSTEER;
} else
{if((direction > heading))
{return (-1 * flock.core.MAXSTEER);
} else
{if(new cljs.core.Keyword(null,"else","else",1017020587))
{return 0;
} else
{return null;
}
}
}
}
});
flock.core.step = (function step(object,coll,t){var x = flock.core.move_x.call(null,object,t);var y = flock.core.move_y.call(null,object,t);var v = new cljs.core.Keyword(null,"v","v",1013904360).cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"v","v",1013904360).cljs$core$IFn$_invoke$arity$1(object));var O = cljs.core.mod.call(null,(new cljs.core.Keyword(null,"O","O",1013904321).cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"v","v",1013904360).cljs$core$IFn$_invoke$arity$1(object)) + flock.core.steer.call(null,object,coll)),(2 * flock.core.pi));return cljs.core.assoc.call(null,object,new cljs.core.Keyword(null,"x","x",1013904362),x,new cljs.core.Keyword(null,"y","y",1013904363),y,new cljs.core.Keyword(null,"v","v",1013904360),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"v","v",1013904360),v,new cljs.core.Keyword(null,"O","O",1013904321),O], null));
});
flock.core.triangle = (function triangle(x,y,direction,key){return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"div","div",1014003715),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"key","key",1014010321),key,new cljs.core.Keyword(null,"style","style",1123684643),cljs.core.PersistentHashMap.fromArrays([new cljs.core.Keyword(null,"border-right","border-right",1319660717),new cljs.core.Keyword(null,"transform","transform",2066570974),new cljs.core.Keyword(null,"font-size","font-size",3722789425),new cljs.core.Keyword(null,"top","top",1014019271),new cljs.core.Keyword(null,"width","width",1127031096),new cljs.core.Keyword(null,"border-left","border-left",1716321402),new cljs.core.Keyword(null,"position","position",1761709211),new cljs.core.Keyword(null,"height","height",4087841945),new cljs.core.Keyword(null,"border-bottom","border-bottom",1450293854),new cljs.core.Keyword(null,"left","left",1017222009)],["5px solid transparent",[cljs.core.str("rotate("),cljs.core.str(direction),cljs.core.str("rad) scaleY("),cljs.core.str(3),cljs.core.str(")")].join(''),5,(y * flock.core.SCALE),0,"5px solid transparent","absolute",0,"5px solid black",(x * flock.core.SCALE)])], null)], null);
});
flock.core.main_loop = (function main_loop(app){var objects = new cljs.core.Keyword(null,"objects","objects",3649222790).cljs$core$IFn$_invoke$arity$1(cljs.core.deref.call(null,app));return om.core.update_BANG_.call(null,app,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"objects","objects",3649222790)], null),cljs.core.map.call(null,((function (objects){
return (function (p1__22674_SHARP_){return flock.core.step.call(null,p1__22674_SHARP_,objects,0.032);
});})(objects))
,objects));
});
flock.core.flock = (function flock__$1(app,owner){if(typeof flock.core.t22683 !== 'undefined')
{} else
{
/**
* @constructor
*/
flock.core.t22683 = (function (owner,app,flock,meta22684){
this.owner = owner;
this.app = app;
this.flock = flock;
this.meta22684 = meta22684;
this.cljs$lang$protocol_mask$partition1$ = 0;
this.cljs$lang$protocol_mask$partition0$ = 393216;
})
flock.core.t22683.cljs$lang$type = true;
flock.core.t22683.cljs$lang$ctorStr = "flock.core/t22683";
flock.core.t22683.cljs$lang$ctorPrWriter = (function (this__8441__auto__,writer__8442__auto__,opt__8443__auto__){return cljs.core._write.call(null,writer__8442__auto__,"flock.core/t22683");
});
flock.core.t22683.prototype.om$core$IRender$ = true;
flock.core.t22683.prototype.om$core$IRender$render$arity$1 = (function (this$){var self__ = this;
var this$__$1 = this;var attrs22686 = flock.core.triangle.call(null,20,20,flock.core.pi);if(cljs.core.map_QMARK_.call(null,attrs22686))
{return React.DOM.div(sablono.interpreter.attributes.call(null,attrs22686),sablono.interpreter.interpret.call(null,cljs.core.map.call(null,((function (attrs22686,this$__$1){
return (function (p__22687){var map__22688 = p__22687;var map__22688__$1 = ((cljs.core.seq_QMARK_.call(null,map__22688))?cljs.core.apply.call(null,cljs.core.hash_map,map__22688):map__22688);var key = cljs.core.get.call(null,map__22688__$1,new cljs.core.Keyword(null,"key","key",1014010321));var v = cljs.core.get.call(null,map__22688__$1,new cljs.core.Keyword(null,"v","v",1013904360));var y = cljs.core.get.call(null,map__22688__$1,new cljs.core.Keyword(null,"y","y",1013904363));var x = cljs.core.get.call(null,map__22688__$1,new cljs.core.Keyword(null,"x","x",1013904362));return flock.core.triangle.call(null,cljs.core.mod.call(null,x,new cljs.core.Keyword(null,"width","width",1127031096).cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"browser","browser",1164844698).cljs$core$IFn$_invoke$arity$1(self__.app))),y,new cljs.core.Keyword(null,"O","O",1013904321).cljs$core$IFn$_invoke$arity$1(v),key);
});})(attrs22686,this$__$1))
,new cljs.core.Keyword(null,"objects","objects",3649222790).cljs$core$IFn$_invoke$arity$1(self__.app))));
} else
{return React.DOM.div(null,sablono.interpreter.interpret.call(null,attrs22686),sablono.interpreter.interpret.call(null,cljs.core.map.call(null,((function (attrs22686,this$__$1){
return (function (p__22689){var map__22690 = p__22689;var map__22690__$1 = ((cljs.core.seq_QMARK_.call(null,map__22690))?cljs.core.apply.call(null,cljs.core.hash_map,map__22690):map__22690);var key = cljs.core.get.call(null,map__22690__$1,new cljs.core.Keyword(null,"key","key",1014010321));var v = cljs.core.get.call(null,map__22690__$1,new cljs.core.Keyword(null,"v","v",1013904360));var y = cljs.core.get.call(null,map__22690__$1,new cljs.core.Keyword(null,"y","y",1013904363));var x = cljs.core.get.call(null,map__22690__$1,new cljs.core.Keyword(null,"x","x",1013904362));return flock.core.triangle.call(null,cljs.core.mod.call(null,x,new cljs.core.Keyword(null,"width","width",1127031096).cljs$core$IFn$_invoke$arity$1(new cljs.core.Keyword(null,"browser","browser",1164844698).cljs$core$IFn$_invoke$arity$1(self__.app))),y,new cljs.core.Keyword(null,"O","O",1013904321).cljs$core$IFn$_invoke$arity$1(v),key);
});})(attrs22686,this$__$1))
,new cljs.core.Keyword(null,"objects","objects",3649222790).cljs$core$IFn$_invoke$arity$1(self__.app))));
}
});
flock.core.t22683.prototype.om$core$IWillMount$ = true;
flock.core.t22683.prototype.om$core$IWillMount$will_mount$arity$1 = (function (this$){var self__ = this;
var this$__$1 = this;window.addEventListener("resize",((function (this$__$1){
return (function (){return om.core.update_BANG_.call(null,self__.app,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"browser","browser",1164844698)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"width","width",1127031096),(window["innerWidth"]),new cljs.core.Keyword(null,"height","height",4087841945),(window["innerHeight"])], null));
});})(this$__$1))
);
om.core.update_BANG_.call(null,self__.app,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"browser","browser",1164844698)], null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"width","width",1127031096),(window["innerWidth"]),new cljs.core.Keyword(null,"height","height",4087841945),(window["innerHeight"])], null));
return window.setInterval(((function (this$__$1){
return (function (){return flock.core.main_loop.call(null,self__.app);
});})(this$__$1))
,110);
});
flock.core.t22683.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_22685){var self__ = this;
var _22685__$1 = this;return self__.meta22684;
});
flock.core.t22683.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_22685,meta22684__$1){var self__ = this;
var _22685__$1 = this;return (new flock.core.t22683(self__.owner,self__.app,self__.flock,meta22684__$1));
});
flock.core.__GT_t22683 = (function __GT_t22683(owner__$1,app__$1,flock__$2,meta22684){return (new flock.core.t22683(owner__$1,app__$1,flock__$2,meta22684));
});
}
return (new flock.core.t22683(owner,app,flock__$1,null));
});
if(typeof flock.core.app_state !== 'undefined')
{} else
{flock.core.app_state = cljs.core.atom.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"objects","objects",3649222790),cljs.core.map.call(null,(function (d){return new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"x","x",1013904362),300,new cljs.core.Keyword(null,"y","y",1013904363),300,new cljs.core.Keyword(null,"v","v",1013904360),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"v","v",1013904360),50,new cljs.core.Keyword(null,"O","O",1013904321),d], null),new cljs.core.Keyword(null,"key","key",1014010321),d], null);
}),cljs.core.range.call(null,0,(2 * flock.core.pi),0.5)),new cljs.core.Keyword(null,"browser","browser",1164844698),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"width","width",1127031096),0,new cljs.core.Keyword(null,"height","height",4087841945),0], null)], null));
}
om.core.root.call(null,flock.core.flock,flock.core.app_state,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"target","target",4427965699),document.getElementById("app")], null));
figwheel.client.watch_and_reload.call(null,new cljs.core.Keyword(null,"jsload-callback","jsload-callback",3126035989),(function (){return null;
}));
