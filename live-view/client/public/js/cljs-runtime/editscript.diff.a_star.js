goog.provide('editscript.diff.a_star');

/**
 * @interface
 */
editscript.diff.a_star.INode = function(){};

var editscript$diff$a_star$INode$get_path$dyn_47269 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_path[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_path["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-path",this$);
}
}
});
/**
 * Get the path to the node from root
 */
editscript.diff.a_star.get_path = (function editscript$diff$a_star$get_path(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_path$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_path$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_path$dyn_47269(this$);
}
});

var editscript$diff$a_star$INode$get_value$dyn_47284 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_value[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_value["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-value",this$);
}
}
});
/**
 * Get the actual data
 */
editscript.diff.a_star.get_value = (function editscript$diff$a_star$get_value(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_value$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_value$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_value$dyn_47284(this$);
}
});

var editscript$diff$a_star$INode$get_children$dyn_47292 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_children[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_children["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-children",this$);
}
}
});
/**
 * Get all children node in a map
 */
editscript.diff.a_star.get_children = (function editscript$diff$a_star$get_children(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_children$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_children$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_children$dyn_47292(this$);
}
});

var editscript$diff$a_star$INode$add_child$dyn_47294 = (function (this$,node){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.add_child[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,node) : m__4429__auto__.call(null,this$,node));
} else {
var m__4426__auto__ = (editscript.diff.a_star.add_child["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,node) : m__4426__auto__.call(null,this$,node));
} else {
throw cljs.core.missing_protocol("INode.add-child",this$);
}
}
});
/**
 * Add a child node
 */
editscript.diff.a_star.add_child = (function editscript$diff$a_star$add_child(this$,node){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$add_child$arity$2 == null)))))){
return this$.editscript$diff$a_star$INode$add_child$arity$2(this$,node);
} else {
return editscript$diff$a_star$INode$add_child$dyn_47294(this$,node);
}
});

var editscript$diff$a_star$INode$get_key$dyn_47332 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_key[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_key["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-key",this$);
}
}
});
/**
 * Get the key of this node
 */
editscript.diff.a_star.get_key = (function editscript$diff$a_star$get_key(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_key$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_key$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_key$dyn_47332(this$);
}
});

var editscript$diff$a_star$INode$get_parent$dyn_47334 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_parent[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_parent["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-parent",this$);
}
}
});
/**
 * Get the parent node
 */
editscript.diff.a_star.get_parent = (function editscript$diff$a_star$get_parent(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_parent$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_parent$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_parent$dyn_47334(this$);
}
});

var editscript$diff$a_star$INode$get_first$dyn_47335 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_first[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_first["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-first",this$);
}
}
});
/**
 * Get the first child node
 */
editscript.diff.a_star.get_first = (function editscript$diff$a_star$get_first(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_first$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_first$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_first$dyn_47335(this$);
}
});

var editscript$diff$a_star$INode$get_last$dyn_47337 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_last[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_last["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-last",this$);
}
}
});
/**
 * Get the last child node
 */
editscript.diff.a_star.get_last = (function editscript$diff$a_star$get_last(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_last$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_last$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_last$dyn_47337(this$);
}
});

var editscript$diff$a_star$INode$get_next$dyn_47338 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_next[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_next["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-next",this$);
}
}
});
/**
 * Get the next sibling node
 */
editscript.diff.a_star.get_next = (function editscript$diff$a_star$get_next(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_next$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_next$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_next$dyn_47338(this$);
}
});

var editscript$diff$a_star$INode$set_next$dyn_47339 = (function (this$,node){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_next[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,node) : m__4429__auto__.call(null,this$,node));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_next["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,node) : m__4426__auto__.call(null,this$,node));
} else {
throw cljs.core.missing_protocol("INode.set-next",this$);
}
}
});
/**
 * Set the next sibling node
 */
editscript.diff.a_star.set_next = (function editscript$diff$a_star$set_next(this$,node){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$set_next$arity$2 == null)))))){
return this$.editscript$diff$a_star$INode$set_next$arity$2(this$,node);
} else {
return editscript$diff$a_star$INode$set_next$dyn_47339(this$,node);
}
});

var editscript$diff$a_star$INode$set_order$dyn_47340 = (function (this$,o){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_order[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,o) : m__4429__auto__.call(null,this$,o));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_order["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,o) : m__4426__auto__.call(null,this$,o));
} else {
throw cljs.core.missing_protocol("INode.set-order",this$);
}
}
});
/**
 * Set the traversal order of this node
 */
editscript.diff.a_star.set_order = (function editscript$diff$a_star$set_order(this$,o){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$set_order$arity$2 == null)))))){
return this$.editscript$diff$a_star$INode$set_order$arity$2(this$,o);
} else {
return editscript$diff$a_star$INode$set_order$dyn_47340(this$,o);
}
});

var editscript$diff$a_star$INode$get_order$dyn_47341 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_order[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_order["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-order",this$);
}
}
});
/**
 * Get the order of this node in traversal
 */
editscript.diff.a_star.get_order = (function editscript$diff$a_star$get_order(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_order$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_order$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_order$dyn_47341(this$);
}
});

var editscript$diff$a_star$INode$get_size$dyn_47342 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_size[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_size["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("INode.get-size",this$);
}
}
});
/**
 * Get the size of sub-tree, used to estimate cost
 */
editscript.diff.a_star.get_size = (function editscript$diff$a_star$get_size(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$get_size$arity$1 == null)))))){
return this$.editscript$diff$a_star$INode$get_size$arity$1(this$);
} else {
return editscript$diff$a_star$INode$get_size$dyn_47342(this$);
}
});

var editscript$diff$a_star$INode$set_size$dyn_47343 = (function (this$,s){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_size[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,s) : m__4429__auto__.call(null,this$,s));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_size["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,s) : m__4426__auto__.call(null,this$,s));
} else {
throw cljs.core.missing_protocol("INode.set-size",this$);
}
}
});
/**
 * Set the size of sub-tree
 */
editscript.diff.a_star.set_size = (function editscript$diff$a_star$set_size(this$,s){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$INode$set_size$arity$2 == null)))))){
return this$.editscript$diff$a_star$INode$set_size$arity$2(this$,s);
} else {
return editscript$diff$a_star$INode$set_size$dyn_47343(this$,s);
}
});


/**
* @constructor
 * @implements {editscript.diff.a_star.INode}
*/
editscript.diff.a_star.Node = (function (path,value,parent,children,first,last,next,index,order,size){
this.path = path;
this.value = value;
this.parent = parent;
this.children = children;
this.first = first;
this.last = last;
this.next = next;
this.index = index;
this.order = order;
this.size = size;
});
(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_key$arity$1 = (function (this$){
var self__ = this;
var this$__$1 = this;
return cljs.core.peek(this$__$1.editscript$diff$a_star$INode$get_path$arity$1(null));
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$set_size$arity$2 = (function (this$,s){
var self__ = this;
var this$__$1 = this;
(self__.size = cljs.core.long$(s));

return this$__$1;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_size$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.size;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$set_next$arity$2 = (function (_,n){
var self__ = this;
var ___$1 = this;
return (self__.next = n);
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_parent$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.parent;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_first$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.first;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_last$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.last;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_path$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.path;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_value$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.value;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_next$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.next;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_children$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.children;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$set_order$arity$2 = (function (this$,o){
var self__ = this;
var this$__$1 = this;
(self__.order = cljs.core.long$(o));

return this$__$1;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$get_order$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.order;
}));

(editscript.diff.a_star.Node.prototype.editscript$diff$a_star$INode$add_child$arity$2 = (function (_,node){
var self__ = this;
var ___$1 = this;
(self__.children = cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(self__.children,editscript.diff.a_star.get_key(node),node));

if(cljs.core.truth_(self__.last)){
editscript.diff.a_star.set_next(self__.last,node);
} else {
}

if(cljs.core.truth_(self__.first)){
} else {
(self__.first = node);
}

(self__.last = node);

return node;
}));

(editscript.diff.a_star.Node.getBasis = (function (){
return new cljs.core.PersistentVector(null, 10, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"path","path",1452340359,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"PersistentVector","PersistentVector",-837570443,null)], null)),new cljs.core.Symbol(null,"value","value",1946509744,null),new cljs.core.Symbol(null,"parent","parent",761652748,null),cljs.core.with_meta(new cljs.core.Symbol(null,"children","children",699969545,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"first","first",996428481,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"last","last",-1548700637,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"next","next",1522830042,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"index","index",108845612,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"order","order",385854271,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"size","size",-1555742762,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null))], null);
}));

(editscript.diff.a_star.Node.cljs$lang$type = true);

(editscript.diff.a_star.Node.cljs$lang$ctorStr = "editscript.diff.a-star/Node");

(editscript.diff.a_star.Node.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.diff.a-star/Node");
}));

/**
 * Positional factory function for editscript.diff.a-star/Node.
 */
editscript.diff.a_star.__GT_Node = (function editscript$diff$a_star$__GT_Node(path,value,parent,children,first,last,next,index,order,size){
return (new editscript.diff.a_star.Node(path,value,parent,children,first,last,next,index,order,size));
});

editscript.diff.a_star.index_STAR_ = (function editscript$diff$a_star$index_STAR_(order,path,data,parent){
return null;
});
/**
 * map and vector are associative
 */
editscript.diff.a_star.associative_children = (function editscript$diff$a_star$associative_children(order,path,data,parent){
return cljs.core.reduce_kv((function (_,k,v){
var G__46582 = order;
var G__46583 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,k);
var G__46584 = v;
var G__46585 = parent;
return (editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4 ? editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4(G__46582,G__46583,G__46584,G__46585) : editscript.diff.a_star.index_STAR_.call(null,G__46582,G__46583,G__46584,G__46585));
}),null,data);
});
/**
 * set is a map of keys to themselves
 */
editscript.diff.a_star.set_children = (function editscript$diff$a_star$set_children(order,path,data,parent){
var seq__46592 = cljs.core.seq(data);
var chunk__46596 = null;
var count__46598 = (0);
var i__46599 = (0);
while(true){
if((i__46599 < count__46598)){
var x = chunk__46596.cljs$core$IIndexed$_nth$arity$2(null,i__46599);
var G__46619_47344 = order;
var G__46620_47345 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,x);
var G__46621_47346 = x;
var G__46622_47347 = parent;
(editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4 ? editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4(G__46619_47344,G__46620_47345,G__46621_47346,G__46622_47347) : editscript.diff.a_star.index_STAR_.call(null,G__46619_47344,G__46620_47345,G__46621_47346,G__46622_47347));


var G__47348 = seq__46592;
var G__47349 = chunk__46596;
var G__47350 = count__46598;
var G__47351 = (i__46599 + (1));
seq__46592 = G__47348;
chunk__46596 = G__47349;
count__46598 = G__47350;
i__46599 = G__47351;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__46592);
if(temp__5735__auto__){
var seq__46592__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__46592__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__46592__$1);
var G__47352 = cljs.core.chunk_rest(seq__46592__$1);
var G__47353 = c__4556__auto__;
var G__47354 = cljs.core.count(c__4556__auto__);
var G__47355 = (0);
seq__46592 = G__47352;
chunk__46596 = G__47353;
count__46598 = G__47354;
i__46599 = G__47355;
continue;
} else {
var x = cljs.core.first(seq__46592__$1);
var G__46624_47356 = order;
var G__46625_47357 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,x);
var G__46626_47358 = x;
var G__46627_47359 = parent;
(editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4 ? editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4(G__46624_47356,G__46625_47357,G__46626_47358,G__46627_47359) : editscript.diff.a_star.index_STAR_.call(null,G__46624_47356,G__46625_47357,G__46626_47358,G__46627_47359));


var G__47360 = cljs.core.next(seq__46592__$1);
var G__47361 = null;
var G__47362 = (0);
var G__47363 = (0);
seq__46592 = G__47360;
chunk__46596 = G__47361;
count__46598 = G__47362;
i__46599 = G__47363;
continue;
}
} else {
return null;
}
}
break;
}
});
/**
 * add index as key
 */
editscript.diff.a_star.list_children = (function editscript$diff$a_star$list_children(order,path,data,parent){
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (i,x){
var G__46641_47364 = order;
var G__46642_47365 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,i);
var G__46643_47366 = x;
var G__46644_47367 = parent;
(editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4 ? editscript.diff.a_star.index_STAR_.cljs$core$IFn$_invoke$arity$4(G__46641_47364,G__46642_47365,G__46643_47366,G__46644_47367) : editscript.diff.a_star.index_STAR_.call(null,G__46641_47364,G__46642_47365,G__46643_47366,G__46644_47367));

return (i + (1));
}),(0),data);
});
editscript.diff.a_star.inc_order = (function editscript$diff$a_star$inc_order(order){
return cljs.core._vreset_BANG_(order,(function (o){
return (o + (1));
})(cljs.core._deref(order)));
});
editscript.diff.a_star.index_collection = (function editscript$diff$a_star$index_collection(type,order,path,data,parent){
var node = editscript.diff.a_star.__GT_Node(path,data,parent,cljs.core.PersistentArrayMap.EMPTY,null,null,null,(0),(0),(1));
editscript.diff.a_star.add_child(parent,node);

var G__46655_47368 = type;
var G__46655_47369__$1 = (((G__46655_47368 instanceof cljs.core.Keyword))?G__46655_47368.fqn:null);
switch (G__46655_47369__$1) {
case "map":
case "vec":
editscript.diff.a_star.associative_children(order,path,data,node);

break;
case "set":
editscript.diff.a_star.set_children(order,path,data,node);

break;
case "lst":
editscript.diff.a_star.list_children(order,path,data,node);

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__46655_47369__$1)].join('')));

}

var cs_47371 = cljs.core.reduce.cljs$core$IFn$_invoke$arity$2(cljs.core._PLUS_,cljs.core.map.cljs$core$IFn$_invoke$arity$2(editscript.diff.a_star.get_size,cljs.core.vals(node.editscript$diff$a_star$INode$get_children$arity$1(null))));
var G__46657_47372 = node;
G__46657_47372.editscript$diff$a_star$INode$set_order$arity$2(null,cljs.core.deref(order));

G__46657_47372.editscript$diff$a_star$INode$set_size$arity$2(null,(node.editscript$diff$a_star$INode$get_size$arity$1(null) + cs_47371));


editscript.diff.a_star.inc_order(order);

return node;
});
editscript.diff.a_star.index_value = (function editscript$diff$a_star$index_value(order,path,data,parent){
var node = editscript.diff.a_star.__GT_Node(path,data,parent,null,null,null,null,(0),cljs.core.deref(order),(1));
editscript.diff.a_star.add_child(parent,node);

editscript.diff.a_star.inc_order(order);

return node;
});
editscript.diff.a_star.index_STAR_ = (function editscript$diff$a_star$index_STAR_(order,path,data,parent){
var type = editscript.edit.get_type(data);
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(type,new cljs.core.Keyword(null,"val","val",128701612))) || (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(type,new cljs.core.Keyword(null,"str","str",1089608819))))){
return editscript.diff.a_star.index_value(order,path,data,parent);
} else {
return editscript.diff.a_star.index_collection(type,order,path,data,parent);
}
});
/**
 * Traverse data to build an indexing tree of Nodes,
 *   compute path, sizes of sub-trees, siblings, etc. for each Node.
 *   This takes little time
 */
editscript.diff.a_star.index = (function editscript$diff$a_star$index(data){
var order = cljs.core.volatile_BANG_((0));
return editscript.diff.a_star.index_STAR_(order,cljs.core.PersistentVector.EMPTY,data,editscript.diff.a_star.__GT_Node(cljs.core.PersistentVector.EMPTY,new cljs.core.Keyword("editscript.diff.a-star","dummy","editscript.diff.a-star/dummy",978832816),null,cljs.core.PersistentArrayMap.EMPTY,null,null,null,(0),(-1),(0)));
});
editscript.diff.a_star.coord_hash = (function editscript$diff$a_star$coord_hash(a,b){
return editscript.util.common.szudzik(editscript.diff.a_star.get_order(a),editscript.diff.a_star.get_order(b));
});

/**
* @constructor
 * @implements {cljs.core.IEquiv}
 * @implements {cljs.core.IHash}
 * @implements {cljs.core.IComparable}
*/
editscript.diff.a_star.Coord = (function (a,b){
this.a = a;
this.b = b;
this.cljs$lang$protocol_mask$partition0$ = 6291456;
this.cljs$lang$protocol_mask$partition1$ = 2048;
});
(editscript.diff.a_star.Coord.prototype.cljs$core$IHash$_hash$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return editscript.diff.a_star.coord_hash(self__.a,self__.b);
}));

(editscript.diff.a_star.Coord.prototype.cljs$core$IEquiv$_equiv$arity$2 = (function (this$,that){
var self__ = this;
var this$__$1 = this;
return cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(this$__$1.cljs$core$IHash$_hash$arity$1(null),cljs.core._hash(that));
}));

(editscript.diff.a_star.Coord.prototype.cljs$core$IComparable$_compare$arity$2 = (function (this$,that){
var self__ = this;
var this$__$1 = this;
return (this$__$1.cljs$core$IHash$_hash$arity$1(null) - cljs.core._hash(that));
}));

(editscript.diff.a_star.Coord.getBasis = (function (){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"a","a",-482876059,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"Node","Node",446032991,null)], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"b","b",-1172211299,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"Node","Node",446032991,null)], null))], null);
}));

(editscript.diff.a_star.Coord.cljs$lang$type = true);

(editscript.diff.a_star.Coord.cljs$lang$ctorStr = "editscript.diff.a-star/Coord");

(editscript.diff.a_star.Coord.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.diff.a-star/Coord");
}));

/**
 * Positional factory function for editscript.diff.a-star/Coord.
 */
editscript.diff.a_star.__GT_Coord = (function editscript$diff$a_star$__GT_Coord(a,b){
return (new editscript.diff.a_star.Coord(a,b));
});

editscript.diff.a_star.get_coord = (function editscript$diff$a_star$get_coord(coord){
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [coord.a,coord.b], null);
});

/**
 * @interface
 */
editscript.diff.a_star.IStep = function(){};

var editscript$diff$a_star$IStep$operator$dyn_47373 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.operator[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.operator["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IStep.operator",this$);
}
}
});
/**
 * Operator to try
 */
editscript.diff.a_star.operator = (function editscript$diff$a_star$operator(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IStep$operator$arity$1 == null)))))){
return this$.editscript$diff$a_star$IStep$operator$arity$1(this$);
} else {
return editscript$diff$a_star$IStep$operator$dyn_47373(this$);
}
});

var editscript$diff$a_star$IStep$current$dyn_47374 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.current[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.current["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IStep.current",this$);
}
}
});
/**
 * Starting pair of nodes
 */
editscript.diff.a_star.current = (function editscript$diff$a_star$current(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IStep$current$arity$1 == null)))))){
return this$.editscript$diff$a_star$IStep$current$arity$1(this$);
} else {
return editscript$diff$a_star$IStep$current$dyn_47374(this$);
}
});

var editscript$diff$a_star$IStep$neighbor$dyn_47375 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.neighbor[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.neighbor["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IStep.neighbor",this$);
}
}
});
/**
 * Destination pair of nodes
 */
editscript.diff.a_star.neighbor = (function editscript$diff$a_star$neighbor(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IStep$neighbor$arity$1 == null)))))){
return this$.editscript$diff$a_star$IStep$neighbor$arity$1(this$);
} else {
return editscript$diff$a_star$IStep$neighbor$dyn_47375(this$);
}
});


/**
* @constructor
 * @implements {editscript.diff.a_star.IStep}
*/
editscript.diff.a_star.Step = (function (op,cur,nbr){
this.op = op;
this.cur = cur;
this.nbr = nbr;
});
(editscript.diff.a_star.Step.prototype.editscript$diff$a_star$IStep$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.diff.a_star.Step.prototype.editscript$diff$a_star$IStep$operator$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.op;
}));

(editscript.diff.a_star.Step.prototype.editscript$diff$a_star$IStep$current$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.cur;
}));

(editscript.diff.a_star.Step.prototype.editscript$diff$a_star$IStep$neighbor$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.nbr;
}));

(editscript.diff.a_star.Step.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"op","op",-242456428,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"Keyword","Keyword",-850065993,null)], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"cur","cur",-1501245170,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"Coord","Coord",462878922,null)], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"nbr","nbr",-1618248259,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"Coord","Coord",462878922,null)], null))], null);
}));

(editscript.diff.a_star.Step.cljs$lang$type = true);

(editscript.diff.a_star.Step.cljs$lang$ctorStr = "editscript.diff.a-star/Step");

(editscript.diff.a_star.Step.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.diff.a-star/Step");
}));

/**
 * Positional factory function for editscript.diff.a-star/Step.
 */
editscript.diff.a_star.__GT_Step = (function editscript$diff$a_star$__GT_Step(op,cur,nbr){
return (new editscript.diff.a_star.Step(op,cur,nbr));
});

editscript.diff.a_star.get_step = (function editscript$diff$a_star$get_step(step){
return cljs.core.juxt.cljs$core$IFn$_invoke$arity$3(editscript.diff.a_star.operator,editscript.diff.a_star.current,editscript.diff.a_star.neighbor)(step);
});

/**
 * @interface
 */
editscript.diff.a_star.IState = function(){};

var editscript$diff$a_star$IState$get_came$dyn_47376 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_came[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_came["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IState.get-came",this$);
}
}
});
/**
 * Get the local succession map
 */
editscript.diff.a_star.get_came = (function editscript$diff$a_star$get_came(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$get_came$arity$1 == null)))))){
return this$.editscript$diff$a_star$IState$get_came$arity$1(this$);
} else {
return editscript$diff$a_star$IState$get_came$dyn_47376(this$);
}
});

var editscript$diff$a_star$IState$set_came$dyn_47377 = (function (this$,came){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_came[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,came) : m__4429__auto__.call(null,this$,came));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_came["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,came) : m__4426__auto__.call(null,this$,came));
} else {
throw cljs.core.missing_protocol("IState.set-came",this$);
}
}
});
/**
 * Set the local succession map
 */
editscript.diff.a_star.set_came = (function editscript$diff$a_star$set_came(this$,came){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$set_came$arity$2 == null)))))){
return this$.editscript$diff$a_star$IState$set_came$arity$2(this$,came);
} else {
return editscript$diff$a_star$IState$set_came$dyn_47377(this$,came);
}
});

var editscript$diff$a_star$IState$get_open$dyn_47378 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_open[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_open["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IState.get-open",this$);
}
}
});
/**
 * Get the open priority queue
 */
editscript.diff.a_star.get_open = (function editscript$diff$a_star$get_open(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$get_open$arity$1 == null)))))){
return this$.editscript$diff$a_star$IState$get_open$arity$1(this$);
} else {
return editscript$diff$a_star$IState$get_open$dyn_47378(this$);
}
});

var editscript$diff$a_star$IState$set_open$dyn_47379 = (function (this$,open){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_open[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,open) : m__4429__auto__.call(null,this$,open));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_open["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,open) : m__4426__auto__.call(null,this$,open));
} else {
throw cljs.core.missing_protocol("IState.set-open",this$);
}
}
});
/**
 * Set the open priority queue
 */
editscript.diff.a_star.set_open = (function editscript$diff$a_star$set_open(this$,open){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$set_open$arity$2 == null)))))){
return this$.editscript$diff$a_star$IState$set_open$arity$2(this$,open);
} else {
return editscript$diff$a_star$IState$set_open$dyn_47379(this$,open);
}
});

var editscript$diff$a_star$IState$get_g$dyn_47380 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.get_g[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.diff.a_star.get_g["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IState.get-g",this$);
}
}
});
/**
 * Get the g cost map
 */
editscript.diff.a_star.get_g = (function editscript$diff$a_star$get_g(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$get_g$arity$1 == null)))))){
return this$.editscript$diff$a_star$IState$get_g$arity$1(this$);
} else {
return editscript$diff$a_star$IState$get_g$dyn_47380(this$);
}
});

var editscript$diff$a_star$IState$set_g$dyn_47381 = (function (this$,g){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.diff.a_star.set_g[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,g) : m__4429__auto__.call(null,this$,g));
} else {
var m__4426__auto__ = (editscript.diff.a_star.set_g["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,g) : m__4426__auto__.call(null,this$,g));
} else {
throw cljs.core.missing_protocol("IState.set-g",this$);
}
}
});
/**
 * Set the g cost map
 */
editscript.diff.a_star.set_g = (function editscript$diff$a_star$set_g(this$,g){
if((((!((this$ == null)))) && ((!((this$.editscript$diff$a_star$IState$set_g$arity$2 == null)))))){
return this$.editscript$diff$a_star$IState$set_g$arity$2(this$,g);
} else {
return editscript$diff$a_star$IState$set_g$dyn_47381(this$,g);
}
});


/**
* @constructor
 * @implements {editscript.diff.a_star.IState}
*/
editscript.diff.a_star.State = (function (came,open,g){
this.came = came;
this.open = open;
this.g = g;
});
(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$get_came$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.came;
}));

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$set_came$arity$2 = (function (this$,c){
var self__ = this;
var this$__$1 = this;
(self__.came = c);

return this$__$1;
}));

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$get_open$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.open;
}));

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$set_open$arity$2 = (function (this$,o){
var self__ = this;
var this$__$1 = this;
(self__.open = o);

return this$__$1;
}));

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$get_g$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.g;
}));

(editscript.diff.a_star.State.prototype.editscript$diff$a_star$IState$set_g$arity$2 = (function (this$,got){
var self__ = this;
var this$__$1 = this;
(self__.g = got);

return this$__$1;
}));

(editscript.diff.a_star.State.getBasis = (function (){
return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"came","came",-1609486569,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"open","open",-123064921,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"g","g",-916345864,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null))], null);
}));

(editscript.diff.a_star.State.cljs$lang$type = true);

(editscript.diff.a_star.State.cljs$lang$ctorStr = "editscript.diff.a-star/State");

(editscript.diff.a_star.State.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.diff.a-star/State");
}));

/**
 * Positional factory function for editscript.diff.a-star/State.
 */
editscript.diff.a_star.__GT_State = (function editscript$diff$a_star$__GT_State(came,open,g){
return (new editscript.diff.a_star.State(came,open,g));
});

editscript.diff.a_star.get_state = (function editscript$diff$a_star$get_state(state){
return cljs.core.juxt.cljs$core$IFn$_invoke$arity$3(editscript.diff.a_star.get_came,editscript.diff.a_star.get_open,editscript.diff.a_star.get_g)(state);
});
editscript.diff.a_star.access_g = (function editscript$diff$a_star$access_g(g,cur){
return cljs.core.get.cljs$core$IFn$_invoke$arity$3(g,cur,goog.math.Long.getMaxValue());
});
editscript.diff.a_star.diff_STAR_ = (function editscript$diff$a_star$diff_STAR_(ra,rb,came){
return null;
});
editscript.diff.a_star.compute_cost = (function editscript$diff$a_star$compute_cost(cur,came,g,op){
var gc = editscript.diff.a_star.access_g(g,cur);
var G__46807 = op;
var G__46807__$1 = (((G__46807 instanceof cljs.core.Keyword))?G__46807.fqn:null);
switch (G__46807__$1) {
case "=":
return gc;

break;
case "-":
return (gc + (1));

break;
case "a":
case "i":
var sb = editscript.diff.a_star.get_size(cur.b);
return (gc + (sb + (1)));

break;
case "r":
return (gc + (function (){var G__46808 = cur.a;
var G__46809 = cur.b;
var G__46810 = came;
return (editscript.diff.a_star.diff_STAR_.cljs$core$IFn$_invoke$arity$3 ? editscript.diff.a_star.diff_STAR_.cljs$core$IFn$_invoke$arity$3(G__46808,G__46809,G__46810) : editscript.diff.a_star.diff_STAR_.call(null,G__46808,G__46809,G__46810));
})());

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__46807__$1)].join('')));

}
});
/**
 * A simplistic but optimistic estimate of the cost to reach goal when at (x y).
 * 
 *   For nested structure, multiple deletion may be merged into one.
 *   Also, because addition/replacement requires new value to be present in
 *   editscript, whereas deletion does not, we assign estimate differently. 
 */
editscript.diff.a_star.heuristic = (function editscript$diff$a_star$heuristic(type,cur,end,p__46823){
var vec__46826 = p__46823;
var gx = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46826,(0),null);
var gy = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46826,(1),null);
var G__46830 = type;
var G__46830__$1 = (((G__46830 instanceof cljs.core.Keyword))?G__46830.fqn:null);
switch (G__46830__$1) {
case "map":
case "set":
return (0);

break;
case "vec":
case "lst":
var vec__46847 = editscript.diff.a_star.get_coord(cur);
var na = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46847,(0),null);
var nb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46847,(1),null);
var vec__46850 = editscript.diff.a_star.get_coord(end);
var ra = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46850,(0),null);
var rb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46850,(1),null);
var x = (((ra === na))?gx:editscript.diff.a_star.get_order(na));
var y = (((rb === nb))?gy:editscript.diff.a_star.get_order(nb));
var dy = (gy - y);
var dx = (gx - x);
if((dx === (0))){
return dy;
} else {
if((dy === (0))){
return (1);
} else {
if((dx > dy)){
return (4);
} else {
if((dx < dy)){
return ((dy - dx) + (3));
} else {
return (2);

}
}
}
}

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__46830__$1)].join('')));

}
});
editscript.diff.a_star.explore = (function editscript$diff$a_star$explore(type,end,came,goal,state,step){
var vec__46875 = editscript.diff.a_star.get_state(state);
var came_SINGLEQUOTE_ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46875,(0),null);
var open = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46875,(1),null);
var g = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46875,(2),null);
var vec__46878 = editscript.diff.a_star.get_step(step);
var op = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46878,(0),null);
var cur = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46878,(1),null);
var nbr = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46878,(2),null);
var tmp_g = editscript.diff.a_star.compute_cost(cur,came,g,op);
if((tmp_g >= editscript.diff.a_star.access_g(g,nbr))){
return state;
} else {
var G__46904 = state;
editscript.diff.a_star.set_came(G__46904,cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(came_SINGLEQUOTE_,nbr,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cur,op], null)));

editscript.diff.a_star.set_open(G__46904,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(open,nbr,(tmp_g + editscript.diff.a_star.heuristic(type,nbr,end,goal))));

editscript.diff.a_star.set_g(G__46904,cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(g,nbr,tmp_g));

return G__46904;
}
});
editscript.diff.a_star.next_node = (function editscript$diff$a_star$next_node(na,ra){
var or__4126__auto__ = editscript.diff.a_star.get_next(na);
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return ra;
}
});
editscript.diff.a_star.vec_frontier = (function editscript$diff$a_star$vec_frontier(end,cur){
var vec__46937 = editscript.diff.a_star.get_coord(end);
var ra = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46937,(0),null);
var rb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46937,(1),null);
var vec__46940 = editscript.diff.a_star.get_coord(cur);
var na = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46940,(0),null);
var nb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46940,(1),null);
var a_EQ_b = cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(editscript.diff.a_star.get_value(na),editscript.diff.a_star.get_value(nb));
var x_EQ_gx = (na === ra);
var x_LT_gx = (!(x_EQ_gx));
var y_LT_gy = (!((nb === rb)));
var na_SINGLEQUOTE_ = editscript.diff.a_star.next_node(na,ra);
var nb_SINGLEQUOTE_ = editscript.diff.a_star.next_node(nb,rb);
if(((x_LT_gx) && (y_LT_gy) && (a_EQ_b))){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"=","=",1152933628),cur,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb_SINGLEQUOTE_))], null);
} else {
var G__46963 = cljs.core.PersistentVector.EMPTY;
var G__46963__$1 = ((x_LT_gx)?cljs.core.conj.cljs$core$IFn$_invoke$arity$2(G__46963,editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"-","-",-2112348439),cur,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb))):G__46963);
var G__46963__$2 = ((((x_LT_gx) && (y_LT_gy)))?cljs.core.conj.cljs$core$IFn$_invoke$arity$2(G__46963__$1,editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"r","r",-471384190),cur,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb_SINGLEQUOTE_))):G__46963__$1);
var G__46963__$3 = ((((x_EQ_gx) && (y_LT_gy)))?cljs.core.conj.cljs$core$IFn$_invoke$arity$2(G__46963__$2,editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"a","a",-2123407586),cur,editscript.diff.a_star.__GT_Coord(na,nb_SINGLEQUOTE_))):G__46963__$2);
if(((x_LT_gx) && (y_LT_gy))){
return cljs.core.conj.cljs$core$IFn$_invoke$arity$2(G__46963__$3,editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"i","i",-1386841315),cur,editscript.diff.a_star.__GT_Coord(na,nb_SINGLEQUOTE_)));
} else {
return G__46963__$3;
}
}
});
editscript.diff.a_star.map_frontier = (function editscript$diff$a_star$map_frontier(init,end,cur){
var vec__46972 = editscript.diff.a_star.get_coord(end);
var ra = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46972,(0),null);
var rb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46972,(1),null);
var vec__46975 = editscript.diff.a_star.get_coord(cur);
var na = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46975,(0),null);
var nb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__46975,(1),null);
var ka = editscript.diff.a_star.get_key(na);
var kb = editscript.diff.a_star.get_key(nb);
if((na === ra)){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.__GT_Step(((cljs.core.contains_QMARK_(editscript.diff.a_star.get_value(ra),kb))?new cljs.core.Keyword(null,"=","=",1152933628):new cljs.core.Keyword(null,"a","a",-2123407586)),cur,editscript.diff.a_star.__GT_Coord(ra,editscript.diff.a_star.next_node(nb,rb)))], null);
} else {
var va = editscript.diff.a_star.get_value(na);
var vb = editscript.diff.a_star.get_value(nb);
var mb = editscript.diff.a_star.get_value(rb);
var na_SINGLEQUOTE_ = editscript.diff.a_star.next_node(na,ra);
var cb = editscript.diff.a_star.get_children(rb);
if((na_SINGLEQUOTE_ === ra)){
var startb = editscript.diff.a_star.__GT_Coord(ra,init.b);
var enda = editscript.diff.a_star.__GT_Coord(na,(cb.cljs$core$IFn$_invoke$arity$1 ? cb.cljs$core$IFn$_invoke$arity$1(ka) : cb.call(null,ka)));
if(cljs.core.contains_QMARK_(mb,ka)){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(ka,kb)){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.__GT_Step(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(va,vb))?new cljs.core.Keyword(null,"=","=",1152933628):new cljs.core.Keyword(null,"r","r",-471384190)),cur,startb)], null);
} else {
return new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"=","=",1152933628),cur,enda),editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"r","r",-471384190),enda,startb)], null);
}
} else {
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"-","-",-2112348439),cur,startb)], null);
}
} else {
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [((cljs.core.contains_QMARK_(mb,ka))?((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(ka,kb))?editscript.diff.a_star.__GT_Step(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(va,vb))?new cljs.core.Keyword(null,"=","=",1152933628):new cljs.core.Keyword(null,"r","r",-471384190)),cur,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,(function (){var or__4126__auto__ = (function (){var G__46990 = editscript.diff.a_star.get_key(na_SINGLEQUOTE_);
return (cb.cljs$core$IFn$_invoke$arity$1 ? cb.cljs$core$IFn$_invoke$arity$1(G__46990) : cb.call(null,G__46990));
})();
if(cljs.core.truth_(or__4126__auto__)){
return or__4126__auto__;
} else {
return nb;
}
})())):editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"=","=",1152933628),cur,editscript.diff.a_star.__GT_Coord(na,(cb.cljs$core$IFn$_invoke$arity$1 ? cb.cljs$core$IFn$_invoke$arity$1(ka) : cb.call(null,ka))))):editscript.diff.a_star.__GT_Step(new cljs.core.Keyword(null,"-","-",-2112348439),cur,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb)))], null);
}
}
});
editscript.diff.a_star.frontier = (function editscript$diff$a_star$frontier(type,init,end,cur){
var G__47004 = type;
var G__47004__$1 = (((G__47004 instanceof cljs.core.Keyword))?G__47004.fqn:null);
switch (G__47004__$1) {
case "vec":
case "lst":
return editscript.diff.a_star.vec_frontier(end,cur);

break;
case "map":
case "set":
return editscript.diff.a_star.map_frontier(init,end,cur);

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__47004__$1)].join('')));

}
});
editscript.diff.a_star.A_STAR_ = (function editscript$diff$a_star$A_STAR_(type,ra,rb,came){
var end = editscript.diff.a_star.__GT_Coord(ra,rb);
var init = editscript.diff.a_star.__GT_Coord(editscript.diff.a_star.get_first(ra),editscript.diff.a_star.get_first(rb));
var goal = new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [editscript.diff.a_star.get_order(ra),editscript.diff.a_star.get_order(rb)], null);
var state = editscript.diff.a_star.__GT_State(cljs.core.transient$(cljs.core.PersistentArrayMap.EMPTY),editscript.util.pairing.priority_map.cljs$core$IFn$_invoke$arity$variadic(cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([init,editscript.diff.a_star.heuristic(type,init,end,goal)], 0)),cljs.core.transient$(cljs.core.PersistentArrayMap.createAsIfByAssoc([init,(0)])));
while(true){
var vec__47035 = editscript.diff.a_star.get_state(state);
var came_SINGLEQUOTE_ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47035,(0),null);
var open = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47035,(1),null);
var _ = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47035,(2),null);
if(cljs.core.empty_QMARK_(open)){
throw cljs.core.ex_info.cljs$core$IFn$_invoke$arity$2("A* diff fails to find a solution",new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"ra","ra",-382035026),ra,new cljs.core.Keyword(null,"rb","rb",1673817808),rb], null));
} else {
var vec__47038 = cljs.core.peek(open);
var cur = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47038,(0),null);
var cost = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47038,(1),null);
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(cur,end)){
cljs.core._vreset_BANG_(came,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(cljs.core._deref(came),end,cljs.core.persistent_BANG_(came_SINGLEQUOTE_)));

return cost;
} else {
var G__47385 = cljs.core.reduce.cljs$core$IFn$_invoke$arity$3(cljs.core.partial.cljs$core$IFn$_invoke$arity$variadic(editscript.diff.a_star.explore,type,end,came,cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([goal], 0)),editscript.diff.a_star.set_open(state,cljs.core.pop(open)),editscript.diff.a_star.frontier(type,init,end,cur));
state = G__47385;
continue;
}
}
break;
}
});
editscript.diff.a_star.vec_fn = (function editscript$diff$a_star$vec_fn(node){
var v = editscript.diff.a_star.get_value(node);
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"vec","vec",-657847931),editscript.edit.get_type(v))){
return v;
} else {
return cljs.core.vec(v);
}
});
editscript.diff.a_star.use_quick = (function editscript$diff$a_star$use_quick(ra,rb,came){
var G__47060 = editscript.util.common.vec_edits(editscript.diff.a_star.vec_fn(ra),editscript.diff.a_star.vec_fn(rb));
var vec__47064 = G__47060;
var seq__47065 = cljs.core.seq(vec__47064);
var first__47066 = cljs.core.first(seq__47065);
var seq__47065__$1 = cljs.core.next(seq__47065);
var op = first__47066;
var ops = seq__47065__$1;
var na = editscript.diff.a_star.get_first(ra);
var nb = editscript.diff.a_star.get_first(rb);
var m = cljs.core.transient$(cljs.core.PersistentArrayMap.EMPTY);
var cost = (0);
var G__47060__$1 = G__47060;
var na__$1 = na;
var nb__$1 = nb;
var m__$1 = m;
var cost__$1 = cost;
while(true){
var vec__47089 = G__47060__$1;
var seq__47090 = cljs.core.seq(vec__47089);
var first__47091 = cljs.core.first(seq__47090);
var seq__47090__$1 = cljs.core.next(seq__47090);
var op__$1 = first__47091;
var ops__$1 = seq__47090__$1;
var na__$2 = na__$1;
var nb__$2 = nb__$1;
var m__$2 = m__$1;
var cost__$2 = cost__$1;
if(cljs.core.truth_(op__$1)){
var na_SINGLEQUOTE_ = editscript.diff.a_star.next_node(na__$2,ra);
var nb_SINGLEQUOTE_ = editscript.diff.a_star.next_node(nb__$2,rb);
var cur = editscript.diff.a_star.__GT_Coord(na__$2,nb__$2);
var sb = editscript.diff.a_star.get_size(nb__$2);
if(cljs.core.integer_QMARK_(op__$1)){
var G__47386 = (((op__$1 > (1)))?cljs.core.vec(cljs.core.sequence.cljs$core$IFn$_invoke$arity$1(cljs.core.seq(cljs.core.concat.cljs$core$IFn$_invoke$arity$2((new cljs.core.List(null,(op__$1 - (1)),null,(1),null)),ops__$1)))):ops__$1);
var G__47387 = na_SINGLEQUOTE_;
var G__47388 = nb_SINGLEQUOTE_;
var G__47389 = cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(m__$2,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb_SINGLEQUOTE_),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cur,new cljs.core.Keyword(null,"=","=",1152933628)], null));
var G__47390 = cljs.core.long$(cost__$2);
G__47060__$1 = G__47386;
na__$1 = G__47387;
nb__$1 = G__47388;
m__$1 = G__47389;
cost__$1 = G__47390;
continue;
} else {
var G__47102 = op__$1;
var G__47102__$1 = (((G__47102 instanceof cljs.core.Keyword))?G__47102.fqn:null);
switch (G__47102__$1) {
case "-":
var G__47392 = ops__$1;
var G__47393 = na_SINGLEQUOTE_;
var G__47394 = nb__$2;
var G__47395 = cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(m__$2,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb__$2),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cur,op__$1], null));
var G__47396 = (cljs.core.long$(cost__$2) + (1));
G__47060__$1 = G__47392;
na__$1 = G__47393;
nb__$1 = G__47394;
m__$1 = G__47395;
cost__$1 = G__47396;
continue;

break;
case "+":
var G__47397 = ops__$1;
var G__47398 = na__$2;
var G__47399 = nb_SINGLEQUOTE_;
var G__47400 = cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(m__$2,editscript.diff.a_star.__GT_Coord(na__$2,nb_SINGLEQUOTE_),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cur,(((na__$2 === ra))?new cljs.core.Keyword(null,"a","a",-2123407586):new cljs.core.Keyword(null,"i","i",-1386841315))], null));
var G__47401 = ((cljs.core.long$(cost__$2) + (1)) + cljs.core.long$(sb));
G__47060__$1 = G__47397;
na__$1 = G__47398;
nb__$1 = G__47399;
m__$1 = G__47400;
cost__$1 = G__47401;
continue;

break;
case "r":
var G__47402 = ops__$1;
var G__47403 = na_SINGLEQUOTE_;
var G__47404 = nb_SINGLEQUOTE_;
var G__47405 = cljs.core.assoc_BANG_.cljs$core$IFn$_invoke$arity$3(m__$2,editscript.diff.a_star.__GT_Coord(na_SINGLEQUOTE_,nb_SINGLEQUOTE_),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [cur,op__$1], null));
var G__47406 = ((cljs.core.long$(cost__$2) + (1)) + cljs.core.long$(sb));
G__47060__$1 = G__47402;
na__$1 = G__47403;
nb__$1 = G__47404;
m__$1 = G__47405;
cost__$1 = G__47406;
continue;

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__47102__$1)].join('')));

}
}
} else {
var root = editscript.diff.a_star.__GT_Coord(ra,rb);
cljs.core._vreset_BANG_(came,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(cljs.core._deref(came),root,cljs.core.persistent_BANG_(m__$2)));

return cost__$2;
}
break;
}
});
editscript.diff.a_star.diff_STAR_ = (function editscript$diff$a_star$diff_STAR_(ra,rb,came){
var sa = editscript.diff.a_star.get_size(ra);
var sb = editscript.diff.a_star.get_size(rb);
var va = editscript.diff.a_star.get_value(ra);
var vb = editscript.diff.a_star.get_value(rb);
var typea = editscript.edit.get_type(va);
var update = (function (){
return cljs.core._vreset_BANG_(came,cljs.core.assoc.cljs$core$IFn$_invoke$arity$3(cljs.core._deref(came),editscript.diff.a_star.__GT_Coord(ra,rb),cljs.core.PersistentArrayMap.EMPTY));
});
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$variadic((1),sa,cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([sb], 0))){
update();

if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(va,vb)){
return (0);
} else {
return (2);
}
} else {
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((1),sa)) || (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((1),sb)))){
update();

return (sb + (1));
} else {
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(typea,editscript.edit.get_type(vb))){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(va,vb)){
update();

return (0);
} else {
var a = (cljs.core.truth_((function (){var and__4115__auto__ = (function (){var fexpr__47160 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"lst","lst",269745987),null,new cljs.core.Keyword(null,"vec","vec",-657847931),null], null), null);
return (fexpr__47160.cljs$core$IFn$_invoke$arity$1 ? fexpr__47160.cljs$core$IFn$_invoke$arity$1(typea) : fexpr__47160.call(null,typea));
})();
if(cljs.core.truth_(and__4115__auto__)){
var cc_PLUS_1 = (function (p1__47128_SHARP_){
return (cljs.core.count(editscript.diff.a_star.get_children(p1__47128_SHARP_)) + (1));
});
return ((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(sa,cc_PLUS_1(ra))) || (cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(sb,cc_PLUS_1(rb))));
} else {
return and__4115__auto__;
}
})())?editscript.diff.a_star.use_quick(ra,rb,came):editscript.diff.a_star.A_STAR_(typea,ra,rb,came));
var r = (sb + (1));
if((r < a)){
update();

return r;
} else {
return a;
}
}
} else {
update();

return (sb + (1));

}
}
}
});
editscript.diff.a_star.index_key_QMARK_ = (function editscript$diff$a_star$index_key_QMARK_(node){
var G__47170 = editscript.edit.get_type(editscript.diff.a_star.get_value(node));
var fexpr__47169 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"lst","lst",269745987),null,new cljs.core.Keyword(null,"vec","vec",-657847931),null], null), null);
return (fexpr__47169.cljs$core$IFn$_invoke$arity$1 ? fexpr__47169.cljs$core$IFn$_invoke$arity$1(G__47170) : fexpr__47169.call(null,G__47170));
});
editscript.diff.a_star.adjust_delete_insert = (function editscript$diff$a_star$adjust_delete_insert(trie,op,root,path){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(op,new cljs.core.Keyword(null,"=","=",1152933628))){
return path;
} else {
var newp = cljs.core.PersistentVector.EMPTY;
var prev = cljs.core.PersistentVector.EMPTY;
var node = root;
var left = path;
while(true){
if(cljs.core.seq(left)){
var vec__47191 = left;
var seq__47192 = cljs.core.seq(vec__47191);
var first__47193 = cljs.core.first(seq__47192);
var seq__47192__$1 = cljs.core.next(seq__47192);
var k = first__47193;
var ks = seq__47192__$1;
var d = cljs.core.get_in.cljs$core$IFn$_invoke$arity$3(cljs.core.deref(trie),cljs.core.conj.cljs$core$IFn$_invoke$arity$2(prev,new cljs.core.Keyword(null,"delta","delta",108939957)),(0));
var G__47407 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(newp,(cljs.core.truth_(editscript.diff.a_star.index_key_QMARK_(node))?(k + d):k));
var G__47408 = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(prev,k);
var G__47409 = (function (){var fexpr__47204 = editscript.diff.a_star.get_children(node);
return (fexpr__47204.cljs$core$IFn$_invoke$arity$1 ? fexpr__47204.cljs$core$IFn$_invoke$arity$1(k) : fexpr__47204.call(null,k));
})();
var G__47410 = ks;
newp = G__47407;
prev = G__47408;
node = G__47409;
left = G__47410;
continue;
} else {
if(cljs.core.truth_(editscript.diff.a_star.index_key_QMARK_(editscript.diff.a_star.get_parent(node)))){
var seen = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(((cljs.core.seq(path))?cljs.core.pop(path):path),new cljs.core.Keyword(null,"delta","delta",108939957));
var d = cljs.core.get_in.cljs$core$IFn$_invoke$arity$3(cljs.core.deref(trie),seen,(0));
cljs.core._vreset_BANG_(trie,cljs.core.assoc_in(cljs.core._deref(trie),seen,(function (){var G__47206 = op;
var G__47206__$1 = (((G__47206 instanceof cljs.core.Keyword))?G__47206.fqn:null);
switch (G__47206__$1) {
case "-":
return (d - (1));

break;
case "i":
return (d + (1));

break;
default:
return d;

}
})()));

return newp;
} else {
return newp;
}
}
break;
}
}
});
editscript.diff.a_star.adjust_append = (function editscript$diff$a_star$adjust_append(trie,op,na,nb,path,path_SINGLEQUOTE_){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(op,new cljs.core.Keyword(null,"a","a",-2123407586))){
if(cljs.core.truth_(editscript.diff.a_star.index_key_QMARK_(na))){
return cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path_SINGLEQUOTE_,(function (){var seen = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path,new cljs.core.Keyword(null,"delta","delta",108939957));
var d = cljs.core.get_in.cljs$core$IFn$_invoke$arity$3(cljs.core.deref(trie),seen,(0));
cljs.core._vreset_BANG_(trie,cljs.core.assoc_in(cljs.core._deref(trie),seen,(d + (1))));

return (d + cljs.core.count(editscript.diff.a_star.get_children(na)));
})());
} else {
return cljs.core.conj.cljs$core$IFn$_invoke$arity$2(path_SINGLEQUOTE_,editscript.diff.a_star.get_key(nb));
}
} else {
return path_SINGLEQUOTE_;
}
});
editscript.diff.a_star.convert_path = (function editscript$diff$a_star$convert_path(trie,op,roota,na,nb,path){
return editscript.diff.a_star.adjust_append(trie,op,na,nb,path,editscript.diff.a_star.adjust_delete_insert(trie,op,roota,path));
});
editscript.diff.a_star.write_script = (function editscript$diff$a_star$write_script(steps,roota,script,opts){
return cljs.core.reduce.cljs$core$IFn$_invoke$arity$3((function (trie,p__47212){
var vec__47213 = p__47212;
var op = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47213,(0),null);
var na = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47213,(1),null);
var nb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47213,(2),null);
var path = editscript.diff.a_star.convert_path(trie,op,roota,na,nb,editscript.diff.a_star.get_path(na));
var va = editscript.diff.a_star.get_value(na);
var vb = editscript.diff.a_star.get_value(nb);
var G__47219_47412 = op;
var G__47219_47413__$1 = (((G__47219_47412 instanceof cljs.core.Keyword))?G__47219_47412.fqn:null);
switch (G__47219_47413__$1) {
case "-":
editscript.edit.delete_data(script,path);

break;
case "r":
if(cljs.core.truth_(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$variadic(new cljs.core.Keyword(null,"str","str",1089608819),editscript.edit.get_type(va),cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([editscript.edit.get_type(vb)], 0)))?new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760).cljs$core$IFn$_invoke$arity$1(opts):false))){
editscript.util.common.diff_str(script,path,va,vb,opts);
} else {
editscript.edit.replace_data(script,path,vb);
}

break;
case "a":
case "i":
editscript.edit.add_data(script,path,vb);

break;
default:

}

return trie;
}),cljs.core.volatile_BANG_(new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"delta","delta",108939957),(0)], null)),steps);
});
editscript.diff.a_star.trace_STAR_ = (function editscript$diff$a_star$trace_STAR_(came,cur,steps){
var temp__5733__auto__ = (came.cljs$core$IFn$_invoke$arity$1 ? came.cljs$core$IFn$_invoke$arity$1(cur) : came.call(null,cur));
if(cljs.core.truth_(temp__5733__auto__)){
var m = temp__5733__auto__;
if(cljs.core.seq(m)){
var v = (m.cljs$core$IFn$_invoke$arity$1 ? m.cljs$core$IFn$_invoke$arity$1(cur) : m.call(null,cur));
while(true){
if(cljs.core.truth_(v)){
var vec__47221 = v;
var prev = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47221,(0),null);
var op = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47221,(1),null);
var vec__47224 = editscript.diff.a_star.get_coord(prev);
var na = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47224,(0),null);
var nb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47224,(1),null);
if(cljs.core.truth_((function (){var and__4115__auto__ = (came.cljs$core$IFn$_invoke$arity$1 ? came.cljs$core$IFn$_invoke$arity$1(prev) : came.call(null,prev));
if(cljs.core.truth_(and__4115__auto__)){
return cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(op,new cljs.core.Keyword(null,"r","r",-471384190));
} else {
return and__4115__auto__;
}
})())){
(editscript.diff.a_star.trace_STAR_.cljs$core$IFn$_invoke$arity$3 ? editscript.diff.a_star.trace_STAR_.cljs$core$IFn$_invoke$arity$3(came,prev,steps) : editscript.diff.a_star.trace_STAR_.call(null,came,prev,steps));
} else {
cljs.core._vreset_BANG_(steps,cljs.core.conj.cljs$core$IFn$_invoke$arity$2(cljs.core._deref(steps),new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [op,na,nb], null)));
}

var G__47415 = (m.cljs$core$IFn$_invoke$arity$1 ? m.cljs$core$IFn$_invoke$arity$1(prev) : m.call(null,prev));
v = G__47415;
continue;
} else {
return steps;
}
break;
}
} else {
var vec__47228 = editscript.diff.a_star.get_coord(cur);
var ra = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47228,(0),null);
var rb = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__47228,(1),null);
cljs.core._vreset_BANG_(steps,cljs.core.conj.cljs$core$IFn$_invoke$arity$2(cljs.core._deref(steps),new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(editscript.diff.a_star.get_value(ra),editscript.diff.a_star.get_value(rb)))?new cljs.core.Keyword(null,"=","=",1152933628):new cljs.core.Keyword(null,"r","r",-471384190)),ra,rb], null)));

return steps;
}
} else {
return steps;
}
});
editscript.diff.a_star.trace = (function editscript$diff$a_star$trace(var_args){
var G__47241 = arguments.length;
switch (G__47241) {
case 2:
return editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 4:
return editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$4((arguments[(0)]),(arguments[(1)]),(arguments[(2)]),(arguments[(3)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$2 = (function (came,cur){
return cljs.core.deref(editscript.diff.a_star.trace_STAR_(came,cur,cljs.core.volatile_BANG_(cljs.core.List.EMPTY)));
}));

(editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$4 = (function (came,cur,script,opts){
return editscript.diff.a_star.write_script(editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$2(came,cur),cur.a,script,opts);
}));

(editscript.diff.a_star.trace.cljs$lang$maxFixedArity = 4);

/**
 * Create an EditScript that represents the minimal difference between `b` and `a`
 */
editscript.diff.a_star.diff = (function editscript$diff$a_star$diff(var_args){
var G__47247 = arguments.length;
switch (G__47247) {
case 2:
return editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
case 3:
return editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$3((arguments[(0)]),(arguments[(1)]),(arguments[(2)]));

break;
default:
throw (new Error(["Invalid arity: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(arguments.length)].join('')));

}
});

(editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$2 = (function (a,b){
return editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$3(a,b,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"str-diff?","str-diff?",865254760),false], null));
}));

(editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$3 = (function (a,b,opts){
var script = editscript.edit.edits__GT_script(cljs.core.PersistentVector.EMPTY);
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(a,b)){
} else {
var roota_47418 = editscript.diff.a_star.index(a);
var rootb_47419 = editscript.diff.a_star.index(b);
var came_47420 = cljs.core.volatile_BANG_(cljs.core.PersistentArrayMap.EMPTY);
var cost_47421 = editscript.diff.a_star.diff_STAR_(roota_47418,rootb_47419,came_47420);
editscript.diff.a_star.trace.cljs$core$IFn$_invoke$arity$4(cljs.core.deref(came_47420),editscript.diff.a_star.__GT_Coord(roota_47418,rootb_47419),script,opts);

}

return script;
}));

(editscript.diff.a_star.diff.cljs$lang$maxFixedArity = 3);

editscript.util.common.diff_algo.cljs$core$IMultiFn$_add_method$arity$3(null,new cljs.core.Keyword(null,"a-star","a-star",-171330865),(function (a,b,opts){
return editscript.diff.a_star.diff.cljs$core$IFn$_invoke$arity$3(a,b,opts);
}));

//# sourceMappingURL=editscript.diff.a_star.js.map
