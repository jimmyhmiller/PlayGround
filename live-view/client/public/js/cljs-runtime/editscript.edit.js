goog.provide('editscript.edit');

/**
 * @interface
 */
editscript.edit.IEdit = function(){};

var editscript$edit$IEdit$auto_sizing$dyn_45607 = (function (this$,path,value){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.auto_sizing[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4429__auto__.call(null,this$,path,value));
} else {
var m__4426__auto__ = (editscript.edit.auto_sizing["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4426__auto__.call(null,this$,path,value));
} else {
throw cljs.core.missing_protocol("IEdit.auto-sizing",this$);
}
}
});
editscript.edit.auto_sizing = (function editscript$edit$auto_sizing(this$,path,value){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEdit$auto_sizing$arity$3 == null)))))){
return this$.editscript$edit$IEdit$auto_sizing$arity$3(this$,path,value);
} else {
return editscript$edit$IEdit$auto_sizing$dyn_45607(this$,path,value);
}
});

var editscript$edit$IEdit$add_data$dyn_45608 = (function (this$,path,value){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.add_data[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4429__auto__.call(null,this$,path,value));
} else {
var m__4426__auto__ = (editscript.edit.add_data["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4426__auto__.call(null,this$,path,value));
} else {
throw cljs.core.missing_protocol("IEdit.add-data",this$);
}
}
});
editscript.edit.add_data = (function editscript$edit$add_data(this$,path,value){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEdit$add_data$arity$3 == null)))))){
return this$.editscript$edit$IEdit$add_data$arity$3(this$,path,value);
} else {
return editscript$edit$IEdit$add_data$dyn_45608(this$,path,value);
}
});

var editscript$edit$IEdit$delete_data$dyn_45609 = (function (this$,path){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.delete_data[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,path) : m__4429__auto__.call(null,this$,path));
} else {
var m__4426__auto__ = (editscript.edit.delete_data["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,path) : m__4426__auto__.call(null,this$,path));
} else {
throw cljs.core.missing_protocol("IEdit.delete-data",this$);
}
}
});
editscript.edit.delete_data = (function editscript$edit$delete_data(this$,path){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEdit$delete_data$arity$2 == null)))))){
return this$.editscript$edit$IEdit$delete_data$arity$2(this$,path);
} else {
return editscript$edit$IEdit$delete_data$dyn_45609(this$,path);
}
});

var editscript$edit$IEdit$replace_data$dyn_45610 = (function (this$,path,value){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.replace_data[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4429__auto__.call(null,this$,path,value));
} else {
var m__4426__auto__ = (editscript.edit.replace_data["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,value) : m__4426__auto__.call(null,this$,path,value));
} else {
throw cljs.core.missing_protocol("IEdit.replace-data",this$);
}
}
});
editscript.edit.replace_data = (function editscript$edit$replace_data(this$,path,value){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEdit$replace_data$arity$3 == null)))))){
return this$.editscript$edit$IEdit$replace_data$arity$3(this$,path,value);
} else {
return editscript$edit$IEdit$replace_data$dyn_45610(this$,path,value);
}
});

var editscript$edit$IEdit$replace_str$dyn_45611 = (function (this$,path,ops){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.replace_str[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,ops) : m__4429__auto__.call(null,this$,path,ops));
} else {
var m__4426__auto__ = (editscript.edit.replace_str["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$3 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$3(this$,path,ops) : m__4426__auto__.call(null,this$,path,ops));
} else {
throw cljs.core.missing_protocol("IEdit.replace-str",this$);
}
}
});
editscript.edit.replace_str = (function editscript$edit$replace_str(this$,path,ops){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEdit$replace_str$arity$3 == null)))))){
return this$.editscript$edit$IEdit$replace_str$arity$3(this$,path,ops);
} else {
return editscript$edit$IEdit$replace_str$dyn_45611(this$,path,ops);
}
});


/**
 * @interface
 */
editscript.edit.IEditScript = function(){};

var editscript$edit$IEditScript$combine$dyn_45616 = (function (this$,that){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.combine[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,that) : m__4429__auto__.call(null,this$,that));
} else {
var m__4426__auto__ = (editscript.edit.combine["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,that) : m__4426__auto__.call(null,this$,that));
} else {
throw cljs.core.missing_protocol("IEditScript.combine",this$);
}
}
});
/**
 * Concate that editscript onto this editscript, return the new editscript
 */
editscript.edit.combine = (function editscript$edit$combine(this$,that){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$combine$arity$2 == null)))))){
return this$.editscript$edit$IEditScript$combine$arity$2(this$,that);
} else {
return editscript$edit$IEditScript$combine$dyn_45616(this$,that);
}
});

var editscript$edit$IEditScript$get_size$dyn_45617 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_size[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_size["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.get-size",this$);
}
}
});
/**
 * Report the size of the editscript
 */
editscript.edit.get_size = (function editscript$edit$get_size(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$get_size$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$get_size$arity$1(this$);
} else {
return editscript$edit$IEditScript$get_size$dyn_45617(this$);
}
});

var editscript$edit$IEditScript$set_size$dyn_45619 = (function (this$,size){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.set_size[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$2(this$,size) : m__4429__auto__.call(null,this$,size));
} else {
var m__4426__auto__ = (editscript.edit.set_size["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$2 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$2(this$,size) : m__4426__auto__.call(null,this$,size));
} else {
throw cljs.core.missing_protocol("IEditScript.set-size",this$);
}
}
});
/**
 * Set the size, return the script
 */
editscript.edit.set_size = (function editscript$edit$set_size(this$,size){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$set_size$arity$2 == null)))))){
return this$.editscript$edit$IEditScript$set_size$arity$2(this$,size);
} else {
return editscript$edit$IEditScript$set_size$dyn_45619(this$,size);
}
});

var editscript$edit$IEditScript$edit_distance$dyn_45622 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.edit_distance[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.edit_distance["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.edit-distance",this$);
}
}
});
/**
 * Report the edit distance, i.e number of operations
 */
editscript.edit.edit_distance = (function editscript$edit$edit_distance(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$edit_distance$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$edit_distance$arity$1(this$);
} else {
return editscript$edit$IEditScript$edit_distance$dyn_45622(this$);
}
});

var editscript$edit$IEditScript$get_edits$dyn_45624 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_edits[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_edits["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.get-edits",this$);
}
}
});
/**
 * Report the edits as a vector
 */
editscript.edit.get_edits = (function editscript$edit$get_edits(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$get_edits$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$get_edits$arity$1(this$);
} else {
return editscript$edit$IEditScript$get_edits$dyn_45624(this$);
}
});

var editscript$edit$IEditScript$get_adds_num$dyn_45630 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_adds_num[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_adds_num["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.get-adds-num",this$);
}
}
});
/**
 * Report the number of additions
 */
editscript.edit.get_adds_num = (function editscript$edit$get_adds_num(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$get_adds_num$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$get_adds_num$arity$1(this$);
} else {
return editscript$edit$IEditScript$get_adds_num$dyn_45630(this$);
}
});

var editscript$edit$IEditScript$get_dels_num$dyn_45634 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_dels_num[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_dels_num["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.get-dels-num",this$);
}
}
});
/**
 * Report the number of deletions
 */
editscript.edit.get_dels_num = (function editscript$edit$get_dels_num(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$get_dels_num$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$get_dels_num$arity$1(this$);
} else {
return editscript$edit$IEditScript$get_dels_num$dyn_45634(this$);
}
});

var editscript$edit$IEditScript$get_reps_num$dyn_45636 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_reps_num[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_reps_num["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IEditScript.get-reps-num",this$);
}
}
});
/**
 * Report the number of replacements
 */
editscript.edit.get_reps_num = (function editscript$edit$get_reps_num(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IEditScript$get_reps_num$arity$1 == null)))))){
return this$.editscript$edit$IEditScript$get_reps_num$arity$1(this$);
} else {
return editscript$edit$IEditScript$get_reps_num$dyn_45636(this$);
}
});


/**
 * @interface
 */
editscript.edit.IType = function(){};

var editscript$edit$IType$get_type$dyn_45658 = (function (this$){
var x__4428__auto__ = (((this$ == null))?null:this$);
var m__4429__auto__ = (editscript.edit.get_type[goog.typeOf(x__4428__auto__)]);
if((!((m__4429__auto__ == null)))){
return (m__4429__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4429__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4429__auto__.call(null,this$));
} else {
var m__4426__auto__ = (editscript.edit.get_type["_"]);
if((!((m__4426__auto__ == null)))){
return (m__4426__auto__.cljs$core$IFn$_invoke$arity$1 ? m__4426__auto__.cljs$core$IFn$_invoke$arity$1(this$) : m__4426__auto__.call(null,this$));
} else {
throw cljs.core.missing_protocol("IType.get-type",this$);
}
}
});
/**
 * Return a type keyword, :val, :map, :lst, etc.
 */
editscript.edit.get_type = (function editscript$edit$get_type(this$){
if((((!((this$ == null)))) && ((!((this$.editscript$edit$IType$get_type$arity$1 == null)))))){
return this$.editscript$edit$IType$get_type$arity$1(this$);
} else {
return editscript$edit$IType$get_type$dyn_45658(this$);
}
});

/**
 * A special type means 'not present'
 */
editscript.edit.nada = (function editscript$edit$nada(){
if((typeof editscript !== 'undefined') && (typeof editscript.edit !== 'undefined') && (typeof editscript.edit.t_editscript$edit45365 !== 'undefined')){
} else {

/**
* @constructor
 * @implements {editscript.edit.IType}
 * @implements {cljs.core.IMeta}
 * @implements {cljs.core.IWithMeta}
*/
editscript.edit.t_editscript$edit45365 = (function (meta45366){
this.meta45366 = meta45366;
this.cljs$lang$protocol_mask$partition0$ = 393216;
this.cljs$lang$protocol_mask$partition1$ = 0;
});
(editscript.edit.t_editscript$edit45365.prototype.cljs$core$IWithMeta$_with_meta$arity$2 = (function (_45367,meta45366__$1){
var self__ = this;
var _45367__$1 = this;
return (new editscript.edit.t_editscript$edit45365(meta45366__$1));
}));

(editscript.edit.t_editscript$edit45365.prototype.cljs$core$IMeta$_meta$arity$1 = (function (_45367){
var self__ = this;
var _45367__$1 = this;
return self__.meta45366;
}));

(editscript.edit.t_editscript$edit45365.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.edit.t_editscript$edit45365.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return new cljs.core.Keyword(null,"nil","nil",99600501);
}));

(editscript.edit.t_editscript$edit45365.getBasis = (function (){
return new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Symbol(null,"meta45366","meta45366",-695652626,null)], null);
}));

(editscript.edit.t_editscript$edit45365.cljs$lang$type = true);

(editscript.edit.t_editscript$edit45365.cljs$lang$ctorStr = "editscript.edit/t_editscript$edit45365");

(editscript.edit.t_editscript$edit45365.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.edit/t_editscript$edit45365");
}));

/**
 * Positional factory function for editscript.edit/t_editscript$edit45365.
 */
editscript.edit.__GT_t_editscript$edit45365 = (function editscript$edit$nada_$___GT_t_editscript$edit45365(meta45366){
return (new editscript.edit.t_editscript$edit45365(meta45366));
});

}

return (new editscript.edit.t_editscript$edit45365(cljs.core.PersistentArrayMap.EMPTY));
});
goog.object.set(editscript.edit.IType,"null",true);

goog.object.set(editscript.edit.get_type,"null",(function (_){
return new cljs.core.Keyword(null,"val","val",128701612);
}));

(cljs.core.MapEntry.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.MapEntry.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"vec","vec",-657847931);
}));

(cljs.core.PersistentTreeSet.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentTreeSet.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"set","set",304602554);
}));

(cljs.core.Cons.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.Cons.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"lst","lst",269745987);
}));

(cljs.core.PersistentHashMap.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentHashMap.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"map","map",1371690461);
}));

(cljs.core.Subvec.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.Subvec.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"vec","vec",-657847931);
}));

goog.object.set(editscript.edit.IType,"_",true);

goog.object.set(editscript.edit.get_type,"_",(function (_){
return new cljs.core.Keyword(null,"val","val",128701612);
}));

(cljs.core.PersistentTreeMap.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentTreeMap.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"map","map",1371690461);
}));

(cljs.core.PersistentHashSet.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentHashSet.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"set","set",304602554);
}));

(cljs.core.PersistentVector.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentVector.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"vec","vec",-657847931);
}));

goog.object.set(editscript.edit.IType,"string",true);

goog.object.set(editscript.edit.get_type,"string",(function (_){
return new cljs.core.Keyword(null,"str","str",1089608819);
}));

(cljs.core.EmptyList.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.EmptyList.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"lst","lst",269745987);
}));

(cljs.core.PersistentArrayMap.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.PersistentArrayMap.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"map","map",1371690461);
}));

(cljs.core.List.prototype.editscript$edit$IType$ = cljs.core.PROTOCOL_SENTINEL);

(cljs.core.List.prototype.editscript$edit$IType$get_type$arity$1 = (function (_){
var ___$1 = this;
return new cljs.core.Keyword(null,"lst","lst",269745987);
}));
editscript.edit.sizing_STAR_ = (function editscript$edit$sizing_STAR_(data,size){
var up = (function (s){
return (s + (1));
});
if(cljs.core.truth_((function (){var G__45386 = editscript.edit.get_type(data);
var fexpr__45385 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"lst","lst",269745987),null,new cljs.core.Keyword(null,"vec","vec",-657847931),null,new cljs.core.Keyword(null,"set","set",304602554),null,new cljs.core.Keyword(null,"map","map",1371690461),null], null), null);
return (fexpr__45385.cljs$core$IFn$_invoke$arity$1 ? fexpr__45385.cljs$core$IFn$_invoke$arity$1(G__45386) : fexpr__45385.call(null,G__45386));
})())){
cljs.core._vreset_BANG_(size,up(cljs.core._deref(size)));

var seq__45387 = cljs.core.seq(data);
var chunk__45388 = null;
var count__45389 = (0);
var i__45390 = (0);
while(true){
if((i__45390 < count__45389)){
var child = chunk__45388.cljs$core$IIndexed$_nth$arity$2(null,i__45390);
(editscript.edit.sizing_STAR_.cljs$core$IFn$_invoke$arity$2 ? editscript.edit.sizing_STAR_.cljs$core$IFn$_invoke$arity$2(child,size) : editscript.edit.sizing_STAR_.call(null,child,size));


var G__45684 = seq__45387;
var G__45685 = chunk__45388;
var G__45686 = count__45389;
var G__45687 = (i__45390 + (1));
seq__45387 = G__45684;
chunk__45388 = G__45685;
count__45389 = G__45686;
i__45390 = G__45687;
continue;
} else {
var temp__5735__auto__ = cljs.core.seq(seq__45387);
if(temp__5735__auto__){
var seq__45387__$1 = temp__5735__auto__;
if(cljs.core.chunked_seq_QMARK_(seq__45387__$1)){
var c__4556__auto__ = cljs.core.chunk_first(seq__45387__$1);
var G__45692 = cljs.core.chunk_rest(seq__45387__$1);
var G__45693 = c__4556__auto__;
var G__45694 = cljs.core.count(c__4556__auto__);
var G__45695 = (0);
seq__45387 = G__45692;
chunk__45388 = G__45693;
count__45389 = G__45694;
i__45390 = G__45695;
continue;
} else {
var child = cljs.core.first(seq__45387__$1);
(editscript.edit.sizing_STAR_.cljs$core$IFn$_invoke$arity$2 ? editscript.edit.sizing_STAR_.cljs$core$IFn$_invoke$arity$2(child,size) : editscript.edit.sizing_STAR_.call(null,child,size));


var G__45696 = cljs.core.next(seq__45387__$1);
var G__45697 = null;
var G__45698 = (0);
var G__45699 = (0);
seq__45387 = G__45696;
chunk__45388 = G__45697;
count__45389 = G__45698;
i__45390 = G__45699;
continue;
}
} else {
return null;
}
}
break;
}
} else {
return cljs.core._vreset_BANG_(size,up(cljs.core._deref(size)));
}
});
editscript.edit.sizing = (function editscript$edit$sizing(data){
var size = cljs.core.volatile_BANG_((0));
editscript.edit.sizing_STAR_(data,size);

return cljs.core.deref(size);
});

/**
* @constructor
 * @implements {editscript.edit.IEditScript}
 * @implements {editscript.edit.IEdit}
*/
editscript.edit.EditScript = (function (edits,auto_sizing_QMARK_,size,adds_num,dels_num,reps_num){
this.edits = edits;
this.auto_sizing_QMARK_ = auto_sizing_QMARK_;
this.size = size;
this.adds_num = adds_num;
this.dels_num = dels_num;
this.reps_num = reps_num;
});
(editscript.edit.EditScript.prototype.editscript$edit$IEdit$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.edit.EditScript.prototype.editscript$edit$IEdit$auto_sizing$arity$3 = (function (this$,path,value){
var self__ = this;
var this$__$1 = this;
if(self__.auto_sizing_QMARK_){
(self__.size = cljs.core.long$(((((2) + self__.size) + editscript.edit.sizing(path)) + (cljs.core.truth_(value)?editscript.edit.sizing(value):(0)))));
} else {
}

return this$__$1;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEdit$add_data$arity$3 = (function (this$,path,value){
var self__ = this;
var this$__$1 = this;
(self__.adds_num = (self__.adds_num + (1)));

(self__.edits = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(self__.edits,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [path,new cljs.core.Keyword(null,"+","+",1913524883),value], null)));

return this$__$1.editscript$edit$IEdit$auto_sizing$arity$3(null,path,value);
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEdit$delete_data$arity$2 = (function (this$,path){
var self__ = this;
var this$__$1 = this;
(self__.dels_num = (self__.dels_num + (1)));

(self__.edits = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(self__.edits,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [path,new cljs.core.Keyword(null,"-","-",-2112348439)], null)));

return this$__$1.editscript$edit$IEdit$auto_sizing$arity$3(null,path,null);
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEdit$replace_data$arity$3 = (function (this$,path,value){
var self__ = this;
var this$__$1 = this;
(self__.reps_num = (self__.reps_num + (1)));

(self__.edits = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(self__.edits,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [path,new cljs.core.Keyword(null,"r","r",-471384190),value], null)));

return this$__$1.editscript$edit$IEdit$auto_sizing$arity$3(null,path,value);
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEdit$replace_str$arity$3 = (function (this$,path,ops){
var self__ = this;
var this$__$1 = this;
(self__.reps_num = (self__.reps_num + (1)));

(self__.edits = cljs.core.conj.cljs$core$IFn$_invoke$arity$2(self__.edits,new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [path,new cljs.core.Keyword(null,"s","s",1705939918),ops], null)));

return this$__$1.editscript$edit$IEdit$auto_sizing$arity$3(null,path,"");
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$combine$arity$2 = (function (_,that){
var self__ = this;
var ___$1 = this;
return (new editscript.edit.EditScript(cljs.core.into.cljs$core$IFn$_invoke$arity$2(self__.edits,editscript.edit.get_edits(that)),self__.auto_sizing_QMARK_,(self__.size + editscript.edit.get_size(that)),(self__.adds_num + editscript.edit.get_adds_num(that)),(self__.dels_num + editscript.edit.get_dels_num(that)),(self__.reps_num + editscript.edit.get_reps_num(that))));
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$get_size$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.size;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$set_size$arity$2 = (function (this$,s){
var self__ = this;
var this$__$1 = this;
(self__.size = cljs.core.long$(s));

return this$__$1;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$get_edits$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.edits;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$get_adds_num$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.adds_num;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$get_dels_num$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.dels_num;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$get_reps_num$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return self__.reps_num;
}));

(editscript.edit.EditScript.prototype.editscript$edit$IEditScript$edit_distance$arity$1 = (function (_){
var self__ = this;
var ___$1 = this;
return ((self__.adds_num + self__.dels_num) + self__.reps_num);
}));

(editscript.edit.EditScript.getBasis = (function (){
return new cljs.core.PersistentVector(null, 6, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.with_meta(new cljs.core.Symbol(null,"edits","edits",-599366147,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"PersistentVector","PersistentVector",-837570443,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"auto-sizing?","auto-sizing?",-1705546383,null),new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"boolean","boolean",-278886877,null)], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"size","size",-1555742762,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"adds-num","adds-num",-1467287693,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"dels-num","dels-num",819513451,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null)),cljs.core.with_meta(new cljs.core.Symbol(null,"reps-num","reps-num",582946575,null),new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"tag","tag",-1290361223),new cljs.core.Symbol(null,"long","long",1469079434,null),new cljs.core.Keyword(null,"unsynchronized-mutable","unsynchronized-mutable",-164143950),true], null))], null);
}));

(editscript.edit.EditScript.cljs$lang$type = true);

(editscript.edit.EditScript.cljs$lang$ctorStr = "editscript.edit/EditScript");

(editscript.edit.EditScript.cljs$lang$ctorPrWriter = (function (this__4369__auto__,writer__4370__auto__,opt__4371__auto__){
return cljs.core._write(writer__4370__auto__,"editscript.edit/EditScript");
}));

/**
 * Positional factory function for editscript.edit/EditScript.
 */
editscript.edit.__GT_EditScript = (function editscript$edit$__GT_EditScript(edits,auto_sizing_QMARK_,size,adds_num,dels_num,reps_num){
return (new editscript.edit.EditScript(edits,auto_sizing_QMARK_,size,adds_num,dels_num,reps_num));
});

editscript.edit.valid_str_edits_QMARK_ = (function editscript$edit$valid_str_edits_QMARK_(data){
return ((cljs.core.vector_QMARK_(data)) && (cljs.core.every_QMARK_((function (x){
var or__4126__auto__ = cljs.core.nat_int_QMARK_(x);
if(or__4126__auto__){
return or__4126__auto__;
} else {
if(cljs.core.vector_QMARK_(x)){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2((2),cljs.core.count(x))){
var vec__45558 = x;
var op = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45558,(0),null);
var y = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45558,(1),null);
var and__4115__auto__ = (function (){var fexpr__45563 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 3, [new cljs.core.Keyword(null,"r","r",-471384190),null,new cljs.core.Keyword(null,"-","-",-2112348439),null,new cljs.core.Keyword(null,"+","+",1913524883),null], null), null);
return (fexpr__45563.cljs$core$IFn$_invoke$arity$1 ? fexpr__45563.cljs$core$IFn$_invoke$arity$1(op) : fexpr__45563.call(null,op));
})();
if(cljs.core.truth_(and__4115__auto__)){
var G__45564 = op;
var G__45564__$1 = (((G__45564 instanceof cljs.core.Keyword))?G__45564.fqn:null);
switch (G__45564__$1) {
case "-":
return cljs.core.nat_int_QMARK_(y);

break;
case "+":
case "r":
return typeof y === 'string';

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__45564__$1)].join('')));

}
} else {
return and__4115__auto__;
}
} else {
return false;
}
} else {
return false;
}
}
}),data)));
});
editscript.edit.valid_edit_QMARK_ = (function editscript$edit$valid_edit_QMARK_(edit){
if(cljs.core.vector_QMARK_(edit)){
var c = cljs.core.count(edit);
if(((((1) < c)) && ((c < (4))))){
var vec__45565 = edit;
var path = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45565,(0),null);
var op = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45565,(1),null);
var data = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45565,(2),null);
if(cljs.core.vector_QMARK_(path)){
var and__4115__auto__ = (function (){var fexpr__45574 = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"r","r",-471384190),null,new cljs.core.Keyword(null,"-","-",-2112348439),null,new cljs.core.Keyword(null,"s","s",1705939918),null,new cljs.core.Keyword(null,"+","+",1913524883),null], null), null);
return (fexpr__45574.cljs$core$IFn$_invoke$arity$1 ? fexpr__45574.cljs$core$IFn$_invoke$arity$1(op) : fexpr__45574.call(null,op));
})();
if(cljs.core.truth_(and__4115__auto__)){
if(((cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"-","-",-2112348439),op))?(data == null):cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(c,(3)))){
if(cljs.core._EQ_.cljs$core$IFn$_invoke$arity$2(new cljs.core.Keyword(null,"s","s",1705939918),op)){
return editscript.edit.valid_str_edits_QMARK_(data);
} else {
return true;
}
} else {
return false;
}
} else {
return and__4115__auto__;
}
} else {
return false;
}
} else {
return null;
}
} else {
return null;
}
});
/**
 * Check if the given vector represents valid edits that can be turned into an
 *   EditScript
 */
editscript.edit.valid_edits_QMARK_ = (function editscript$edit$valid_edits_QMARK_(edits){
if(cljs.core.vector_QMARK_(edits)){
if(cljs.core.seq(edits)){
return cljs.core.every_QMARK_(editscript.edit.valid_edit_QMARK_,edits);
} else {
return true;
}
} else {
return null;
}
});
editscript.edit.count_ops = (function editscript$edit$count_ops(edits){
var adds = cljs.core.volatile_BANG_((0));
var dels = cljs.core.volatile_BANG_((0));
var reps = cljs.core.volatile_BANG_((0));
var seq__45576_45728 = cljs.core.seq(edits);
var chunk__45577_45729 = null;
var count__45578_45730 = (0);
var i__45579_45731 = (0);
while(true){
if((i__45579_45731 < count__45578_45730)){
var vec__45593_45732 = chunk__45577_45729.cljs$core$IIndexed$_nth$arity$2(null,i__45579_45731);
var __45733 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45593_45732,(0),null);
var op_45734 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45593_45732,(1),null);
var __45735__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45593_45732,(2),null);
var G__45597_45738 = op_45734;
var G__45597_45739__$1 = (((G__45597_45738 instanceof cljs.core.Keyword))?G__45597_45738.fqn:null);
switch (G__45597_45739__$1) {
case "+":
adds.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(adds.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
case "-":
dels.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(dels.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
case "r":
reps.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(reps.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__45597_45739__$1)].join('')));

}


var G__45742 = seq__45576_45728;
var G__45743 = chunk__45577_45729;
var G__45744 = count__45578_45730;
var G__45745 = (i__45579_45731 + (1));
seq__45576_45728 = G__45742;
chunk__45577_45729 = G__45743;
count__45578_45730 = G__45744;
i__45579_45731 = G__45745;
continue;
} else {
var temp__5735__auto___45746 = cljs.core.seq(seq__45576_45728);
if(temp__5735__auto___45746){
var seq__45576_45747__$1 = temp__5735__auto___45746;
if(cljs.core.chunked_seq_QMARK_(seq__45576_45747__$1)){
var c__4556__auto___45748 = cljs.core.chunk_first(seq__45576_45747__$1);
var G__45749 = cljs.core.chunk_rest(seq__45576_45747__$1);
var G__45750 = c__4556__auto___45748;
var G__45751 = cljs.core.count(c__4556__auto___45748);
var G__45752 = (0);
seq__45576_45728 = G__45749;
chunk__45577_45729 = G__45750;
count__45578_45730 = G__45751;
i__45579_45731 = G__45752;
continue;
} else {
var vec__45598_45753 = cljs.core.first(seq__45576_45747__$1);
var __45754 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45598_45753,(0),null);
var op_45755 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45598_45753,(1),null);
var __45756__$1 = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45598_45753,(2),null);
var G__45601_45760 = op_45755;
var G__45601_45761__$1 = (((G__45601_45760 instanceof cljs.core.Keyword))?G__45601_45760.fqn:null);
switch (G__45601_45761__$1) {
case "+":
adds.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(adds.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
case "-":
dels.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(dels.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
case "r":
reps.cljs$core$IVolatile$_vreset_BANG_$arity$2(null,(reps.cljs$core$IDeref$_deref$arity$1(null) + (1)));

break;
default:
throw (new Error(["No matching clause: ",cljs.core.str.cljs$core$IFn$_invoke$arity$1(G__45601_45761__$1)].join('')));

}


var G__45763 = cljs.core.next(seq__45576_45747__$1);
var G__45764 = null;
var G__45765 = (0);
var G__45766 = (0);
seq__45576_45728 = G__45763;
chunk__45577_45729 = G__45764;
count__45578_45730 = G__45765;
i__45579_45731 = G__45766;
continue;
}
} else {
}
}
break;
}

return new cljs.core.PersistentVector(null, 3, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.deref(adds),cljs.core.deref(dels),cljs.core.deref(reps)], null);
});
/**
 * Create an EditScript instance from a vector of edits, like those obtained
 *   through calling `get-edits` on an EditScript
 */
editscript.edit.edits__GT_script = (function editscript$edit$edits__GT_script(edits){
if(cljs.core.truth_(editscript.edit.valid_edits_QMARK_(edits))){
} else {
throw (new Error(["Assert failed: ","Not a vector of valid edits","\n","(valid-edits? edits)"].join('')));
}

var vec__45604 = editscript.edit.count_ops(edits);
var adds = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45604,(0),null);
var dels = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45604,(1),null);
var reps = cljs.core.nth.cljs$core$IFn$_invoke$arity$3(vec__45604,(2),null);
return editscript.edit.__GT_EditScript(edits,true,editscript.edit.sizing(edits),adds,dels,reps);
});
(editscript.edit.EditScript.prototype.cljs$core$IPrintWithWriter$ = cljs.core.PROTOCOL_SENTINEL);

(editscript.edit.EditScript.prototype.cljs$core$IPrintWithWriter$_pr_writer$arity$3 = (function (o,writer,opts){
var o__$1 = this;
return cljs.core.write_all.cljs$core$IFn$_invoke$arity$variadic(writer,cljs.core.prim_seq.cljs$core$IFn$_invoke$arity$2([cljs.core.str.cljs$core$IFn$_invoke$arity$1(o__$1.editscript$edit$IEditScript$get_edits$arity$1(null))], 0));
}));

//# sourceMappingURL=editscript.edit.js.map
