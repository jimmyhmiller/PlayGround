// Compiled by ClojureScript 1.7.170 {}
goog.provide('figwheel.client.file_reloading');
goog.require('cljs.core');
goog.require('goog.string');
goog.require('goog.Uri');
goog.require('goog.net.jsloader');
goog.require('cljs.core.async');
goog.require('goog.object');
goog.require('clojure.set');
goog.require('clojure.string');
goog.require('figwheel.client.utils');
figwheel.client.file_reloading.queued_file_reload;
if(typeof figwheel.client.file_reloading.figwheel_meta_pragmas !== 'undefined'){
} else {
figwheel.client.file_reloading.figwheel_meta_pragmas = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
}
figwheel.client.file_reloading.on_jsload_custom_event = (function figwheel$client$file_reloading$on_jsload_custom_event(url){
return figwheel.client.utils.dispatch_custom_event.call(null,"figwheel.js-reload",url);
});
figwheel.client.file_reloading.before_jsload_custom_event = (function figwheel$client$file_reloading$before_jsload_custom_event(files){
return figwheel.client.utils.dispatch_custom_event.call(null,"figwheel.before-js-reload",files);
});
figwheel.client.file_reloading.namespace_file_map_QMARK_ = (function figwheel$client$file_reloading$namespace_file_map_QMARK_(m){
var or__16766__auto__ = (cljs.core.map_QMARK_.call(null,m)) && (typeof new cljs.core.Keyword(null,"namespace","namespace",-377510372).cljs$core$IFn$_invoke$arity$1(m) === 'string') && (((new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(m) == null)) || (typeof new cljs.core.Keyword(null,"file","file",-1269645878).cljs$core$IFn$_invoke$arity$1(m) === 'string')) && (cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"type","type",1174270348).cljs$core$IFn$_invoke$arity$1(m),new cljs.core.Keyword(null,"namespace","namespace",-377510372)));
if(or__16766__auto__){
return or__16766__auto__;
} else {
cljs.core.println.call(null,"Error not namespace-file-map",cljs.core.pr_str.call(null,m));

return false;
}
});
figwheel.client.file_reloading.add_cache_buster = (function figwheel$client$file_reloading$add_cache_buster(url){

return goog.Uri.parse(url).makeUnique();
});
figwheel.client.file_reloading.name__GT_path = (function figwheel$client$file_reloading$name__GT_path(ns){

return (goog.dependencies_.nameToPath[ns]);
});
figwheel.client.file_reloading.provided_QMARK_ = (function figwheel$client$file_reloading$provided_QMARK_(ns){
return (goog.dependencies_.written[figwheel.client.file_reloading.name__GT_path.call(null,ns)]);
});
figwheel.client.file_reloading.fix_node_request_url = (function figwheel$client$file_reloading$fix_node_request_url(url){

if(cljs.core.truth_(goog.string.startsWith(url,"../"))){
return clojure.string.replace.call(null,url,"../","");
} else {
return [cljs.core.str("goog/"),cljs.core.str(url)].join('');
}
});
figwheel.client.file_reloading.immutable_ns_QMARK_ = (function figwheel$client$file_reloading$immutable_ns_QMARK_(name){
var or__16766__auto__ = new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 9, ["svgpan.SvgPan",null,"far.out",null,"testDep.bar",null,"someprotopackage.TestPackageTypes",null,"goog",null,"an.existing.path",null,"cljs.core",null,"ns",null,"dup.base",null], null), null).call(null,name);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return cljs.core.some.call(null,cljs.core.partial.call(null,goog.string.startsWith,name),new cljs.core.PersistentVector(null, 5, 5, cljs.core.PersistentVector.EMPTY_NODE, ["goog.","cljs.","clojure.","fake.","proto2."], null));
}
});
figwheel.client.file_reloading.get_requires = (function figwheel$client$file_reloading$get_requires(ns){
return cljs.core.set.call(null,cljs.core.filter.call(null,(function (p1__25335_SHARP_){
return cljs.core.not.call(null,figwheel.client.file_reloading.immutable_ns_QMARK_.call(null,p1__25335_SHARP_));
}),goog.object.getKeys((goog.dependencies_.requires[figwheel.client.file_reloading.name__GT_path.call(null,ns)]))));
});
if(typeof figwheel.client.file_reloading.dependency_data !== 'undefined'){
} else {
figwheel.client.file_reloading.dependency_data = cljs.core.atom.call(null,new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"pathToName","pathToName",-1236616181),cljs.core.PersistentArrayMap.EMPTY,new cljs.core.Keyword(null,"dependents","dependents",136812837),cljs.core.PersistentArrayMap.EMPTY], null));
}
figwheel.client.file_reloading.path_to_name_BANG_ = (function figwheel$client$file_reloading$path_to_name_BANG_(path,name){
return cljs.core.swap_BANG_.call(null,figwheel.client.file_reloading.dependency_data,cljs.core.update_in,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"pathToName","pathToName",-1236616181),path], null),cljs.core.fnil.call(null,clojure.set.union,cljs.core.PersistentHashSet.EMPTY),cljs.core.PersistentHashSet.fromArray([name], true));
});
/**
 * Setup a path to name dependencies map.
 * That goes from path -> #{ ns-names }
 */
figwheel.client.file_reloading.setup_path__GT_name_BANG_ = (function figwheel$client$file_reloading$setup_path__GT_name_BANG_(){
var nameToPath = goog.object.filter(goog.dependencies_.nameToPath,(function (v,k,o){
return goog.string.startsWith(v,"../");
}));
return goog.object.forEach(nameToPath,((function (nameToPath){
return (function (v,k,o){
return figwheel.client.file_reloading.path_to_name_BANG_.call(null,v,k);
});})(nameToPath))
);
});
/**
 * returns a set of namespaces defined by a path
 */
figwheel.client.file_reloading.path__GT_name = (function figwheel$client$file_reloading$path__GT_name(path){
return cljs.core.get_in.call(null,cljs.core.deref.call(null,figwheel.client.file_reloading.dependency_data),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"pathToName","pathToName",-1236616181),path], null));
});
figwheel.client.file_reloading.name_to_parent_BANG_ = (function figwheel$client$file_reloading$name_to_parent_BANG_(ns,parent_ns){
return cljs.core.swap_BANG_.call(null,figwheel.client.file_reloading.dependency_data,cljs.core.update_in,new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"dependents","dependents",136812837),ns], null),cljs.core.fnil.call(null,clojure.set.union,cljs.core.PersistentHashSet.EMPTY),cljs.core.PersistentHashSet.fromArray([parent_ns], true));
});
/**
 * This reverses the goog.dependencies_.requires for looking up ns-dependents.
 */
figwheel.client.file_reloading.setup_ns__GT_dependents_BANG_ = (function figwheel$client$file_reloading$setup_ns__GT_dependents_BANG_(){
var requires = goog.object.filter(goog.dependencies_.requires,(function (v,k,o){
return goog.string.startsWith(k,"../");
}));
return goog.object.forEach(requires,((function (requires){
return (function (v,k,_){
return goog.object.forEach(v,((function (requires){
return (function (v_SINGLEQUOTE_,k_SINGLEQUOTE_,___$1){
var seq__25340 = cljs.core.seq.call(null,figwheel.client.file_reloading.path__GT_name.call(null,k));
var chunk__25341 = null;
var count__25342 = (0);
var i__25343 = (0);
while(true){
if((i__25343 < count__25342)){
var n = cljs.core._nth.call(null,chunk__25341,i__25343);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,k_SINGLEQUOTE_,n);

var G__25344 = seq__25340;
var G__25345 = chunk__25341;
var G__25346 = count__25342;
var G__25347 = (i__25343 + (1));
seq__25340 = G__25344;
chunk__25341 = G__25345;
count__25342 = G__25346;
i__25343 = G__25347;
continue;
} else {
var temp__4425__auto__ = cljs.core.seq.call(null,seq__25340);
if(temp__4425__auto__){
var seq__25340__$1 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25340__$1)){
var c__17569__auto__ = cljs.core.chunk_first.call(null,seq__25340__$1);
var G__25348 = cljs.core.chunk_rest.call(null,seq__25340__$1);
var G__25349 = c__17569__auto__;
var G__25350 = cljs.core.count.call(null,c__17569__auto__);
var G__25351 = (0);
seq__25340 = G__25348;
chunk__25341 = G__25349;
count__25342 = G__25350;
i__25343 = G__25351;
continue;
} else {
var n = cljs.core.first.call(null,seq__25340__$1);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,k_SINGLEQUOTE_,n);

var G__25352 = cljs.core.next.call(null,seq__25340__$1);
var G__25353 = null;
var G__25354 = (0);
var G__25355 = (0);
seq__25340 = G__25352;
chunk__25341 = G__25353;
count__25342 = G__25354;
i__25343 = G__25355;
continue;
}
} else {
return null;
}
}
break;
}
});})(requires))
);
});})(requires))
);
});
figwheel.client.file_reloading.ns__GT_dependents = (function figwheel$client$file_reloading$ns__GT_dependents(ns){
return cljs.core.get_in.call(null,cljs.core.deref.call(null,figwheel.client.file_reloading.dependency_data),new cljs.core.PersistentVector(null, 2, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"dependents","dependents",136812837),ns], null));
});
figwheel.client.file_reloading.build_topo_sort = (function figwheel$client$file_reloading$build_topo_sort(get_deps){
var get_deps__$1 = cljs.core.memoize.call(null,get_deps);
var topo_sort_helper_STAR_ = ((function (get_deps__$1){
return (function figwheel$client$file_reloading$build_topo_sort_$_topo_sort_helper_STAR_(x,depth,state){
var deps = get_deps__$1.call(null,x);
if(cljs.core.empty_QMARK_.call(null,deps)){
return null;
} else {
return topo_sort_STAR_.call(null,deps,depth,state);
}
});})(get_deps__$1))
;
var topo_sort_STAR_ = ((function (get_deps__$1){
return (function() {
var figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_ = null;
var figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___1 = (function (deps){
return figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_.call(null,deps,(0),cljs.core.atom.call(null,cljs.core.sorted_map.call(null)));
});
var figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___3 = (function (deps,depth,state){
cljs.core.swap_BANG_.call(null,state,cljs.core.update_in,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [depth], null),cljs.core.fnil.call(null,cljs.core.into,cljs.core.PersistentHashSet.EMPTY),deps);

var seq__25394_25401 = cljs.core.seq.call(null,deps);
var chunk__25395_25402 = null;
var count__25396_25403 = (0);
var i__25397_25404 = (0);
while(true){
if((i__25397_25404 < count__25396_25403)){
var dep_25405 = cljs.core._nth.call(null,chunk__25395_25402,i__25397_25404);
topo_sort_helper_STAR_.call(null,dep_25405,(depth + (1)),state);

var G__25406 = seq__25394_25401;
var G__25407 = chunk__25395_25402;
var G__25408 = count__25396_25403;
var G__25409 = (i__25397_25404 + (1));
seq__25394_25401 = G__25406;
chunk__25395_25402 = G__25407;
count__25396_25403 = G__25408;
i__25397_25404 = G__25409;
continue;
} else {
var temp__4425__auto___25410 = cljs.core.seq.call(null,seq__25394_25401);
if(temp__4425__auto___25410){
var seq__25394_25411__$1 = temp__4425__auto___25410;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25394_25411__$1)){
var c__17569__auto___25412 = cljs.core.chunk_first.call(null,seq__25394_25411__$1);
var G__25413 = cljs.core.chunk_rest.call(null,seq__25394_25411__$1);
var G__25414 = c__17569__auto___25412;
var G__25415 = cljs.core.count.call(null,c__17569__auto___25412);
var G__25416 = (0);
seq__25394_25401 = G__25413;
chunk__25395_25402 = G__25414;
count__25396_25403 = G__25415;
i__25397_25404 = G__25416;
continue;
} else {
var dep_25417 = cljs.core.first.call(null,seq__25394_25411__$1);
topo_sort_helper_STAR_.call(null,dep_25417,(depth + (1)),state);

var G__25418 = cljs.core.next.call(null,seq__25394_25411__$1);
var G__25419 = null;
var G__25420 = (0);
var G__25421 = (0);
seq__25394_25401 = G__25418;
chunk__25395_25402 = G__25419;
count__25396_25403 = G__25420;
i__25397_25404 = G__25421;
continue;
}
} else {
}
}
break;
}

if(cljs.core._EQ_.call(null,depth,(0))){
return elim_dups_STAR_.call(null,cljs.core.reverse.call(null,cljs.core.vals.call(null,cljs.core.deref.call(null,state))));
} else {
return null;
}
});
figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_ = function(deps,depth,state){
switch(arguments.length){
case 1:
return figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___1.call(this,deps);
case 3:
return figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___3.call(this,deps,depth,state);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_.cljs$core$IFn$_invoke$arity$1 = figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___1;
figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_.cljs$core$IFn$_invoke$arity$3 = figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR___3;
return figwheel$client$file_reloading$build_topo_sort_$_topo_sort_STAR_;
})()
;})(get_deps__$1))
;
var elim_dups_STAR_ = ((function (get_deps__$1){
return (function figwheel$client$file_reloading$build_topo_sort_$_elim_dups_STAR_(p__25398){
var vec__25400 = p__25398;
var x = cljs.core.nth.call(null,vec__25400,(0),null);
var xs = cljs.core.nthnext.call(null,vec__25400,(1));
if((x == null)){
return cljs.core.List.EMPTY;
} else {
return cljs.core.cons.call(null,x,figwheel$client$file_reloading$build_topo_sort_$_elim_dups_STAR_.call(null,cljs.core.map.call(null,((function (vec__25400,x,xs,get_deps__$1){
return (function (p1__25356_SHARP_){
return clojure.set.difference.call(null,p1__25356_SHARP_,x);
});})(vec__25400,x,xs,get_deps__$1))
,xs)));
}
});})(get_deps__$1))
;
return topo_sort_STAR_;
});
figwheel.client.file_reloading.get_all_dependencies = (function figwheel$client$file_reloading$get_all_dependencies(ns){
var topo_sort_SINGLEQUOTE_ = figwheel.client.file_reloading.build_topo_sort.call(null,figwheel.client.file_reloading.get_requires);
return cljs.core.apply.call(null,cljs.core.concat,topo_sort_SINGLEQUOTE_.call(null,cljs.core.set.call(null,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [ns], null))));
});
figwheel.client.file_reloading.get_all_dependents = (function figwheel$client$file_reloading$get_all_dependents(nss){
var topo_sort_SINGLEQUOTE_ = figwheel.client.file_reloading.build_topo_sort.call(null,figwheel.client.file_reloading.ns__GT_dependents);
return cljs.core.reverse.call(null,cljs.core.apply.call(null,cljs.core.concat,topo_sort_SINGLEQUOTE_.call(null,cljs.core.set.call(null,nss))));
});
figwheel.client.file_reloading.unprovide_BANG_ = (function figwheel$client$file_reloading$unprovide_BANG_(ns){
var path = figwheel.client.file_reloading.name__GT_path.call(null,ns);
goog.object.remove(goog.dependencies_.visited,path);

goog.object.remove(goog.dependencies_.written,path);

return goog.object.remove(goog.dependencies_.written,[cljs.core.str(goog.basePath),cljs.core.str(path)].join(''));
});
figwheel.client.file_reloading.resolve_ns = (function figwheel$client$file_reloading$resolve_ns(ns){
return [cljs.core.str(goog.basePath),cljs.core.str(figwheel.client.file_reloading.name__GT_path.call(null,ns))].join('');
});
figwheel.client.file_reloading.addDependency = (function figwheel$client$file_reloading$addDependency(path,provides,requires){
var seq__25434 = cljs.core.seq.call(null,provides);
var chunk__25435 = null;
var count__25436 = (0);
var i__25437 = (0);
while(true){
if((i__25437 < count__25436)){
var prov = cljs.core._nth.call(null,chunk__25435,i__25437);
figwheel.client.file_reloading.path_to_name_BANG_.call(null,path,prov);

var seq__25438_25446 = cljs.core.seq.call(null,requires);
var chunk__25439_25447 = null;
var count__25440_25448 = (0);
var i__25441_25449 = (0);
while(true){
if((i__25441_25449 < count__25440_25448)){
var req_25450 = cljs.core._nth.call(null,chunk__25439_25447,i__25441_25449);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,req_25450,prov);

var G__25451 = seq__25438_25446;
var G__25452 = chunk__25439_25447;
var G__25453 = count__25440_25448;
var G__25454 = (i__25441_25449 + (1));
seq__25438_25446 = G__25451;
chunk__25439_25447 = G__25452;
count__25440_25448 = G__25453;
i__25441_25449 = G__25454;
continue;
} else {
var temp__4425__auto___25455 = cljs.core.seq.call(null,seq__25438_25446);
if(temp__4425__auto___25455){
var seq__25438_25456__$1 = temp__4425__auto___25455;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25438_25456__$1)){
var c__17569__auto___25457 = cljs.core.chunk_first.call(null,seq__25438_25456__$1);
var G__25458 = cljs.core.chunk_rest.call(null,seq__25438_25456__$1);
var G__25459 = c__17569__auto___25457;
var G__25460 = cljs.core.count.call(null,c__17569__auto___25457);
var G__25461 = (0);
seq__25438_25446 = G__25458;
chunk__25439_25447 = G__25459;
count__25440_25448 = G__25460;
i__25441_25449 = G__25461;
continue;
} else {
var req_25462 = cljs.core.first.call(null,seq__25438_25456__$1);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,req_25462,prov);

var G__25463 = cljs.core.next.call(null,seq__25438_25456__$1);
var G__25464 = null;
var G__25465 = (0);
var G__25466 = (0);
seq__25438_25446 = G__25463;
chunk__25439_25447 = G__25464;
count__25440_25448 = G__25465;
i__25441_25449 = G__25466;
continue;
}
} else {
}
}
break;
}

var G__25467 = seq__25434;
var G__25468 = chunk__25435;
var G__25469 = count__25436;
var G__25470 = (i__25437 + (1));
seq__25434 = G__25467;
chunk__25435 = G__25468;
count__25436 = G__25469;
i__25437 = G__25470;
continue;
} else {
var temp__4425__auto__ = cljs.core.seq.call(null,seq__25434);
if(temp__4425__auto__){
var seq__25434__$1 = temp__4425__auto__;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25434__$1)){
var c__17569__auto__ = cljs.core.chunk_first.call(null,seq__25434__$1);
var G__25471 = cljs.core.chunk_rest.call(null,seq__25434__$1);
var G__25472 = c__17569__auto__;
var G__25473 = cljs.core.count.call(null,c__17569__auto__);
var G__25474 = (0);
seq__25434 = G__25471;
chunk__25435 = G__25472;
count__25436 = G__25473;
i__25437 = G__25474;
continue;
} else {
var prov = cljs.core.first.call(null,seq__25434__$1);
figwheel.client.file_reloading.path_to_name_BANG_.call(null,path,prov);

var seq__25442_25475 = cljs.core.seq.call(null,requires);
var chunk__25443_25476 = null;
var count__25444_25477 = (0);
var i__25445_25478 = (0);
while(true){
if((i__25445_25478 < count__25444_25477)){
var req_25479 = cljs.core._nth.call(null,chunk__25443_25476,i__25445_25478);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,req_25479,prov);

var G__25480 = seq__25442_25475;
var G__25481 = chunk__25443_25476;
var G__25482 = count__25444_25477;
var G__25483 = (i__25445_25478 + (1));
seq__25442_25475 = G__25480;
chunk__25443_25476 = G__25481;
count__25444_25477 = G__25482;
i__25445_25478 = G__25483;
continue;
} else {
var temp__4425__auto___25484__$1 = cljs.core.seq.call(null,seq__25442_25475);
if(temp__4425__auto___25484__$1){
var seq__25442_25485__$1 = temp__4425__auto___25484__$1;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25442_25485__$1)){
var c__17569__auto___25486 = cljs.core.chunk_first.call(null,seq__25442_25485__$1);
var G__25487 = cljs.core.chunk_rest.call(null,seq__25442_25485__$1);
var G__25488 = c__17569__auto___25486;
var G__25489 = cljs.core.count.call(null,c__17569__auto___25486);
var G__25490 = (0);
seq__25442_25475 = G__25487;
chunk__25443_25476 = G__25488;
count__25444_25477 = G__25489;
i__25445_25478 = G__25490;
continue;
} else {
var req_25491 = cljs.core.first.call(null,seq__25442_25485__$1);
figwheel.client.file_reloading.name_to_parent_BANG_.call(null,req_25491,prov);

var G__25492 = cljs.core.next.call(null,seq__25442_25485__$1);
var G__25493 = null;
var G__25494 = (0);
var G__25495 = (0);
seq__25442_25475 = G__25492;
chunk__25443_25476 = G__25493;
count__25444_25477 = G__25494;
i__25445_25478 = G__25495;
continue;
}
} else {
}
}
break;
}

var G__25496 = cljs.core.next.call(null,seq__25434__$1);
var G__25497 = null;
var G__25498 = (0);
var G__25499 = (0);
seq__25434 = G__25496;
chunk__25435 = G__25497;
count__25436 = G__25498;
i__25437 = G__25499;
continue;
}
} else {
return null;
}
}
break;
}
});
figwheel.client.file_reloading.figwheel_require = (function figwheel$client$file_reloading$figwheel_require(src,reload){
goog.require = figwheel$client$file_reloading$figwheel_require;

if(cljs.core._EQ_.call(null,reload,"reload-all")){
var seq__25504_25508 = cljs.core.seq.call(null,figwheel.client.file_reloading.get_all_dependencies.call(null,src));
var chunk__25505_25509 = null;
var count__25506_25510 = (0);
var i__25507_25511 = (0);
while(true){
if((i__25507_25511 < count__25506_25510)){
var ns_25512 = cljs.core._nth.call(null,chunk__25505_25509,i__25507_25511);
figwheel.client.file_reloading.unprovide_BANG_.call(null,ns_25512);

var G__25513 = seq__25504_25508;
var G__25514 = chunk__25505_25509;
var G__25515 = count__25506_25510;
var G__25516 = (i__25507_25511 + (1));
seq__25504_25508 = G__25513;
chunk__25505_25509 = G__25514;
count__25506_25510 = G__25515;
i__25507_25511 = G__25516;
continue;
} else {
var temp__4425__auto___25517 = cljs.core.seq.call(null,seq__25504_25508);
if(temp__4425__auto___25517){
var seq__25504_25518__$1 = temp__4425__auto___25517;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__25504_25518__$1)){
var c__17569__auto___25519 = cljs.core.chunk_first.call(null,seq__25504_25518__$1);
var G__25520 = cljs.core.chunk_rest.call(null,seq__25504_25518__$1);
var G__25521 = c__17569__auto___25519;
var G__25522 = cljs.core.count.call(null,c__17569__auto___25519);
var G__25523 = (0);
seq__25504_25508 = G__25520;
chunk__25505_25509 = G__25521;
count__25506_25510 = G__25522;
i__25507_25511 = G__25523;
continue;
} else {
var ns_25524 = cljs.core.first.call(null,seq__25504_25518__$1);
figwheel.client.file_reloading.unprovide_BANG_.call(null,ns_25524);

var G__25525 = cljs.core.next.call(null,seq__25504_25518__$1);
var G__25526 = null;
var G__25527 = (0);
var G__25528 = (0);
seq__25504_25508 = G__25525;
chunk__25505_25509 = G__25526;
count__25506_25510 = G__25527;
i__25507_25511 = G__25528;
continue;
}
} else {
}
}
break;
}
} else {
}

if(cljs.core.truth_(reload)){
figwheel.client.file_reloading.unprovide_BANG_.call(null,src);
} else {
}

return goog.require_figwheel_backup_(src);
});
/**
 * Reusable browser REPL bootstrapping. Patches the essential functions
 *   in goog.base to support re-loading of namespaces after page load.
 */
figwheel.client.file_reloading.bootstrap_goog_base = (function figwheel$client$file_reloading$bootstrap_goog_base(){
if(cljs.core.truth_(COMPILED)){
return null;
} else {
goog.require_figwheel_backup_ = (function (){var or__16766__auto__ = goog.require__;
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
return goog.require;
}
})();

goog.isProvided_ = (function (name){
return false;
});

figwheel.client.file_reloading.setup_path__GT_name_BANG_.call(null);

figwheel.client.file_reloading.setup_ns__GT_dependents_BANG_.call(null);

goog.addDependency_figwheel_backup_ = goog.addDependency;

goog.addDependency = (function() { 
var G__25529__delegate = function (args){
cljs.core.apply.call(null,figwheel.client.file_reloading.addDependency,args);

return cljs.core.apply.call(null,goog.addDependency_figwheel_backup_,args);
};
var G__25529 = function (var_args){
var args = null;
if (arguments.length > 0) {
var G__25530__i = 0, G__25530__a = new Array(arguments.length -  0);
while (G__25530__i < G__25530__a.length) {G__25530__a[G__25530__i] = arguments[G__25530__i + 0]; ++G__25530__i;}
  args = new cljs.core.IndexedSeq(G__25530__a,0);
} 
return G__25529__delegate.call(this,args);};
G__25529.cljs$lang$maxFixedArity = 0;
G__25529.cljs$lang$applyTo = (function (arglist__25531){
var args = cljs.core.seq(arglist__25531);
return G__25529__delegate(args);
});
G__25529.cljs$core$IFn$_invoke$arity$variadic = G__25529__delegate;
return G__25529;
})()
;

goog.constructNamespace_("cljs.user");

goog.global.CLOSURE_IMPORT_SCRIPT = figwheel.client.file_reloading.queued_file_reload;

return goog.require = figwheel.client.file_reloading.figwheel_require;
}
});
figwheel.client.file_reloading.patch_goog_base = (function figwheel$client$file_reloading$patch_goog_base(){
if(typeof figwheel.client.file_reloading.bootstrapped_cljs !== 'undefined'){
return null;
} else {
figwheel.client.file_reloading.bootstrapped_cljs = (function (){
figwheel.client.file_reloading.bootstrap_goog_base.call(null);

return true;
})()
;
}
});
figwheel.client.file_reloading.reload_file_STAR_ = (function (){var pred__25533 = cljs.core._EQ_;
var expr__25534 = figwheel.client.utils.host_env_QMARK_.call(null);
if(cljs.core.truth_(pred__25533.call(null,new cljs.core.Keyword(null,"node","node",581201198),expr__25534))){
var path_parts = ((function (pred__25533,expr__25534){
return (function (p1__25532_SHARP_){
return clojure.string.split.call(null,p1__25532_SHARP_,/[\/\\]/);
});})(pred__25533,expr__25534))
;
var sep = (cljs.core.truth_(cljs.core.re_matches.call(null,/win.*/,process.platform))?"\\":"/");
var root = clojure.string.join.call(null,sep,cljs.core.pop.call(null,cljs.core.pop.call(null,path_parts.call(null,__dirname))));
return ((function (path_parts,sep,root,pred__25533,expr__25534){
return (function (request_url,callback){

var cache_path = clojure.string.join.call(null,sep,cljs.core.cons.call(null,root,path_parts.call(null,figwheel.client.file_reloading.fix_node_request_url.call(null,request_url))));
(require.cache[cache_path] = null);

return callback.call(null,(function (){try{return require(cache_path);
}catch (e25536){if((e25536 instanceof Error)){
var e = e25536;
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"error","error",-978969032),[cljs.core.str("Figwheel: Error loading file "),cljs.core.str(cache_path)].join(''));

figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"error","error",-978969032),e.stack);

return false;
} else {
throw e25536;

}
}})());
});
;})(path_parts,sep,root,pred__25533,expr__25534))
} else {
if(cljs.core.truth_(pred__25533.call(null,new cljs.core.Keyword(null,"html","html",-998796897),expr__25534))){
return ((function (pred__25533,expr__25534){
return (function (request_url,callback){

var deferred = goog.net.jsloader.load(figwheel.client.file_reloading.add_cache_buster.call(null,request_url),{"cleanupWhenDone": true});
deferred.addCallback(((function (deferred,pred__25533,expr__25534){
return (function (){
return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [true], null));
});})(deferred,pred__25533,expr__25534))
);

return deferred.addErrback(((function (deferred,pred__25533,expr__25534){
return (function (){
return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [false], null));
});})(deferred,pred__25533,expr__25534))
);
});
;})(pred__25533,expr__25534))
} else {
return ((function (pred__25533,expr__25534){
return (function (a,b){
throw "Reload not defined for this platform";
});
;})(pred__25533,expr__25534))
}
}
})();
figwheel.client.file_reloading.reload_file = (function figwheel$client$file_reloading$reload_file(p__25537,callback){
var map__25540 = p__25537;
var map__25540__$1 = ((((!((map__25540 == null)))?((((map__25540.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25540.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25540):map__25540);
var file_msg = map__25540__$1;
var request_url = cljs.core.get.call(null,map__25540__$1,new cljs.core.Keyword(null,"request-url","request-url",2100346596));

figwheel.client.utils.debug_prn.call(null,[cljs.core.str("FigWheel: Attempting to load "),cljs.core.str(request_url)].join(''));

return figwheel.client.file_reloading.reload_file_STAR_.call(null,request_url,((function (map__25540,map__25540__$1,file_msg,request_url){
return (function (success_QMARK_){
if(cljs.core.truth_(success_QMARK_)){
figwheel.client.utils.debug_prn.call(null,[cljs.core.str("FigWheel: Successfully loaded "),cljs.core.str(request_url)].join(''));

return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.assoc.call(null,file_msg,new cljs.core.Keyword(null,"loaded-file","loaded-file",-168399375),true)], null));
} else {
figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"error","error",-978969032),[cljs.core.str("Figwheel: Error loading file "),cljs.core.str(request_url)].join(''));

return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [file_msg], null));
}
});})(map__25540,map__25540__$1,file_msg,request_url))
);
});
if(typeof figwheel.client.file_reloading.reload_chan !== 'undefined'){
} else {
figwheel.client.file_reloading.reload_chan = cljs.core.async.chan.call(null);
}
if(typeof figwheel.client.file_reloading.on_load_callbacks !== 'undefined'){
} else {
figwheel.client.file_reloading.on_load_callbacks = cljs.core.atom.call(null,cljs.core.PersistentArrayMap.EMPTY);
}
if(typeof figwheel.client.file_reloading.dependencies_loaded !== 'undefined'){
} else {
figwheel.client.file_reloading.dependencies_loaded = cljs.core.atom.call(null,cljs.core.PersistentVector.EMPTY);
}
figwheel.client.file_reloading.blocking_load = (function figwheel$client$file_reloading$blocking_load(url){
var out = cljs.core.async.chan.call(null);
figwheel.client.file_reloading.reload_file.call(null,new cljs.core.PersistentArrayMap(null, 1, [new cljs.core.Keyword(null,"request-url","request-url",2100346596),url], null),((function (out){
return (function (file_msg){
cljs.core.async.put_BANG_.call(null,out,file_msg);

return cljs.core.async.close_BANG_.call(null,out);
});})(out))
);

return out;
});
if(typeof figwheel.client.file_reloading.reloader_loop !== 'undefined'){
} else {
figwheel.client.file_reloading.reloader_loop = (function (){var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__){
return (function (state_25564){
var state_val_25565 = (state_25564[(1)]);
if((state_val_25565 === (7))){
var inst_25560 = (state_25564[(2)]);
var state_25564__$1 = state_25564;
var statearr_25566_25586 = state_25564__$1;
(statearr_25566_25586[(2)] = inst_25560);

(statearr_25566_25586[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (1))){
var state_25564__$1 = state_25564;
var statearr_25567_25587 = state_25564__$1;
(statearr_25567_25587[(2)] = null);

(statearr_25567_25587[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (4))){
var inst_25544 = (state_25564[(7)]);
var inst_25544__$1 = (state_25564[(2)]);
var state_25564__$1 = (function (){var statearr_25568 = state_25564;
(statearr_25568[(7)] = inst_25544__$1);

return statearr_25568;
})();
if(cljs.core.truth_(inst_25544__$1)){
var statearr_25569_25588 = state_25564__$1;
(statearr_25569_25588[(1)] = (5));

} else {
var statearr_25570_25589 = state_25564__$1;
(statearr_25570_25589[(1)] = (6));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (6))){
var state_25564__$1 = state_25564;
var statearr_25571_25590 = state_25564__$1;
(statearr_25571_25590[(2)] = null);

(statearr_25571_25590[(1)] = (7));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (3))){
var inst_25562 = (state_25564[(2)]);
var state_25564__$1 = state_25564;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_25564__$1,inst_25562);
} else {
if((state_val_25565 === (2))){
var state_25564__$1 = state_25564;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_25564__$1,(4),figwheel.client.file_reloading.reload_chan);
} else {
if((state_val_25565 === (11))){
var inst_25556 = (state_25564[(2)]);
var state_25564__$1 = (function (){var statearr_25572 = state_25564;
(statearr_25572[(8)] = inst_25556);

return statearr_25572;
})();
var statearr_25573_25591 = state_25564__$1;
(statearr_25573_25591[(2)] = null);

(statearr_25573_25591[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (9))){
var inst_25550 = (state_25564[(9)]);
var inst_25548 = (state_25564[(10)]);
var inst_25552 = inst_25550.call(null,inst_25548);
var state_25564__$1 = state_25564;
var statearr_25574_25592 = state_25564__$1;
(statearr_25574_25592[(2)] = inst_25552);

(statearr_25574_25592[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (5))){
var inst_25544 = (state_25564[(7)]);
var inst_25546 = figwheel.client.file_reloading.blocking_load.call(null,inst_25544);
var state_25564__$1 = state_25564;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_25564__$1,(8),inst_25546);
} else {
if((state_val_25565 === (10))){
var inst_25548 = (state_25564[(10)]);
var inst_25554 = cljs.core.swap_BANG_.call(null,figwheel.client.file_reloading.dependencies_loaded,cljs.core.conj,inst_25548);
var state_25564__$1 = state_25564;
var statearr_25575_25593 = state_25564__$1;
(statearr_25575_25593[(2)] = inst_25554);

(statearr_25575_25593[(1)] = (11));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25565 === (8))){
var inst_25544 = (state_25564[(7)]);
var inst_25550 = (state_25564[(9)]);
var inst_25548 = (state_25564[(2)]);
var inst_25549 = cljs.core.deref.call(null,figwheel.client.file_reloading.on_load_callbacks);
var inst_25550__$1 = cljs.core.get.call(null,inst_25549,inst_25544);
var state_25564__$1 = (function (){var statearr_25576 = state_25564;
(statearr_25576[(9)] = inst_25550__$1);

(statearr_25576[(10)] = inst_25548);

return statearr_25576;
})();
if(cljs.core.truth_(inst_25550__$1)){
var statearr_25577_25594 = state_25564__$1;
(statearr_25577_25594[(1)] = (9));

} else {
var statearr_25578_25595 = state_25564__$1;
(statearr_25578_25595[(1)] = (10));

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
});})(c__18933__auto__))
;
return ((function (switch__18821__auto__,c__18933__auto__){
return (function() {
var figwheel$client$file_reloading$state_machine__18822__auto__ = null;
var figwheel$client$file_reloading$state_machine__18822__auto____0 = (function (){
var statearr_25582 = [null,null,null,null,null,null,null,null,null,null,null];
(statearr_25582[(0)] = figwheel$client$file_reloading$state_machine__18822__auto__);

(statearr_25582[(1)] = (1));

return statearr_25582;
});
var figwheel$client$file_reloading$state_machine__18822__auto____1 = (function (state_25564){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_25564);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e25583){if((e25583 instanceof Object)){
var ex__18825__auto__ = e25583;
var statearr_25584_25596 = state_25564;
(statearr_25584_25596[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_25564);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e25583;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__25597 = state_25564;
state_25564 = G__25597;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$file_reloading$state_machine__18822__auto__ = function(state_25564){
switch(arguments.length){
case 0:
return figwheel$client$file_reloading$state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$file_reloading$state_machine__18822__auto____1.call(this,state_25564);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$file_reloading$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$file_reloading$state_machine__18822__auto____0;
figwheel$client$file_reloading$state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$file_reloading$state_machine__18822__auto____1;
return figwheel$client$file_reloading$state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__))
})();
var state__18935__auto__ = (function (){var statearr_25585 = f__18934__auto__.call(null);
(statearr_25585[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_25585;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__))
);

return c__18933__auto__;
})();
}
figwheel.client.file_reloading.queued_file_reload = (function figwheel$client$file_reloading$queued_file_reload(url){
return cljs.core.async.put_BANG_.call(null,figwheel.client.file_reloading.reload_chan,url);
});
figwheel.client.file_reloading.require_with_callback = (function figwheel$client$file_reloading$require_with_callback(p__25598,callback){
var map__25601 = p__25598;
var map__25601__$1 = ((((!((map__25601 == null)))?((((map__25601.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25601.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25601):map__25601);
var file_msg = map__25601__$1;
var namespace = cljs.core.get.call(null,map__25601__$1,new cljs.core.Keyword(null,"namespace","namespace",-377510372));
var request_url = figwheel.client.file_reloading.resolve_ns.call(null,namespace);
cljs.core.swap_BANG_.call(null,figwheel.client.file_reloading.on_load_callbacks,cljs.core.assoc,request_url,((function (request_url,map__25601,map__25601__$1,file_msg,namespace){
return (function (file_msg_SINGLEQUOTE_){
cljs.core.swap_BANG_.call(null,figwheel.client.file_reloading.on_load_callbacks,cljs.core.dissoc,request_url);

return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [cljs.core.merge.call(null,file_msg,cljs.core.select_keys.call(null,file_msg_SINGLEQUOTE_,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [new cljs.core.Keyword(null,"loaded-file","loaded-file",-168399375)], null)))], null));
});})(request_url,map__25601,map__25601__$1,file_msg,namespace))
);

return figwheel.client.file_reloading.figwheel_require.call(null,cljs.core.name.call(null,namespace),true);
});
figwheel.client.file_reloading.reload_file_QMARK_ = (function figwheel$client$file_reloading$reload_file_QMARK_(p__25603){
var map__25606 = p__25603;
var map__25606__$1 = ((((!((map__25606 == null)))?((((map__25606.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25606.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25606):map__25606);
var file_msg = map__25606__$1;
var namespace = cljs.core.get.call(null,map__25606__$1,new cljs.core.Keyword(null,"namespace","namespace",-377510372));

var meta_pragmas = cljs.core.get.call(null,cljs.core.deref.call(null,figwheel.client.file_reloading.figwheel_meta_pragmas),cljs.core.name.call(null,namespace));
var and__16754__auto__ = cljs.core.not.call(null,new cljs.core.Keyword(null,"figwheel-no-load","figwheel-no-load",-555840179).cljs$core$IFn$_invoke$arity$1(meta_pragmas));
if(and__16754__auto__){
var or__16766__auto__ = new cljs.core.Keyword(null,"figwheel-always","figwheel-always",799819691).cljs$core$IFn$_invoke$arity$1(meta_pragmas);
if(cljs.core.truth_(or__16766__auto__)){
return or__16766__auto__;
} else {
var or__16766__auto____$1 = new cljs.core.Keyword(null,"figwheel-load","figwheel-load",1316089175).cljs$core$IFn$_invoke$arity$1(meta_pragmas);
if(cljs.core.truth_(or__16766__auto____$1)){
return or__16766__auto____$1;
} else {
return figwheel.client.file_reloading.provided_QMARK_.call(null,cljs.core.name.call(null,namespace));
}
}
} else {
return and__16754__auto__;
}
});
figwheel.client.file_reloading.js_reload = (function figwheel$client$file_reloading$js_reload(p__25608,callback){
var map__25611 = p__25608;
var map__25611__$1 = ((((!((map__25611 == null)))?((((map__25611.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25611.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25611):map__25611);
var file_msg = map__25611__$1;
var request_url = cljs.core.get.call(null,map__25611__$1,new cljs.core.Keyword(null,"request-url","request-url",2100346596));
var namespace = cljs.core.get.call(null,map__25611__$1,new cljs.core.Keyword(null,"namespace","namespace",-377510372));

if(cljs.core.truth_(figwheel.client.file_reloading.reload_file_QMARK_.call(null,file_msg))){
return figwheel.client.file_reloading.require_with_callback.call(null,file_msg,callback);
} else {
figwheel.client.utils.debug_prn.call(null,[cljs.core.str("Figwheel: Not trying to load file "),cljs.core.str(request_url)].join(''));

return cljs.core.apply.call(null,callback,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [file_msg], null));
}
});
figwheel.client.file_reloading.reload_js_file = (function figwheel$client$file_reloading$reload_js_file(file_msg){
var out = cljs.core.async.chan.call(null);
figwheel.client.file_reloading.js_reload.call(null,file_msg,((function (out){
return (function (url){
cljs.core.async.put_BANG_.call(null,out,url);

return cljs.core.async.close_BANG_.call(null,out);
});})(out))
);

return out;
});
/**
 * Returns a chanel with one collection of loaded filenames on it.
 */
figwheel.client.file_reloading.load_all_js_files = (function figwheel$client$file_reloading$load_all_js_files(files){
var out = cljs.core.async.chan.call(null);
var c__18933__auto___25699 = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto___25699,out){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto___25699,out){
return (function (state_25681){
var state_val_25682 = (state_25681[(1)]);
if((state_val_25682 === (1))){
var inst_25659 = cljs.core.nth.call(null,files,(0),null);
var inst_25660 = cljs.core.nthnext.call(null,files,(1));
var inst_25661 = files;
var state_25681__$1 = (function (){var statearr_25683 = state_25681;
(statearr_25683[(7)] = inst_25660);

(statearr_25683[(8)] = inst_25661);

(statearr_25683[(9)] = inst_25659);

return statearr_25683;
})();
var statearr_25684_25700 = state_25681__$1;
(statearr_25684_25700[(2)] = null);

(statearr_25684_25700[(1)] = (2));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25682 === (2))){
var inst_25664 = (state_25681[(10)]);
var inst_25661 = (state_25681[(8)]);
var inst_25664__$1 = cljs.core.nth.call(null,inst_25661,(0),null);
var inst_25665 = cljs.core.nthnext.call(null,inst_25661,(1));
var inst_25666 = (inst_25664__$1 == null);
var inst_25667 = cljs.core.not.call(null,inst_25666);
var state_25681__$1 = (function (){var statearr_25685 = state_25681;
(statearr_25685[(11)] = inst_25665);

(statearr_25685[(10)] = inst_25664__$1);

return statearr_25685;
})();
if(inst_25667){
var statearr_25686_25701 = state_25681__$1;
(statearr_25686_25701[(1)] = (4));

} else {
var statearr_25687_25702 = state_25681__$1;
(statearr_25687_25702[(1)] = (5));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25682 === (3))){
var inst_25679 = (state_25681[(2)]);
var state_25681__$1 = state_25681;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_25681__$1,inst_25679);
} else {
if((state_val_25682 === (4))){
var inst_25664 = (state_25681[(10)]);
var inst_25669 = figwheel.client.file_reloading.reload_js_file.call(null,inst_25664);
var state_25681__$1 = state_25681;
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_25681__$1,(7),inst_25669);
} else {
if((state_val_25682 === (5))){
var inst_25675 = cljs.core.async.close_BANG_.call(null,out);
var state_25681__$1 = state_25681;
var statearr_25688_25703 = state_25681__$1;
(statearr_25688_25703[(2)] = inst_25675);

(statearr_25688_25703[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25682 === (6))){
var inst_25677 = (state_25681[(2)]);
var state_25681__$1 = state_25681;
var statearr_25689_25704 = state_25681__$1;
(statearr_25689_25704[(2)] = inst_25677);

(statearr_25689_25704[(1)] = (3));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_25682 === (7))){
var inst_25665 = (state_25681[(11)]);
var inst_25671 = (state_25681[(2)]);
var inst_25672 = cljs.core.async.put_BANG_.call(null,out,inst_25671);
var inst_25661 = inst_25665;
var state_25681__$1 = (function (){var statearr_25690 = state_25681;
(statearr_25690[(8)] = inst_25661);

(statearr_25690[(12)] = inst_25672);

return statearr_25690;
})();
var statearr_25691_25705 = state_25681__$1;
(statearr_25691_25705[(2)] = null);

(statearr_25691_25705[(1)] = (2));


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
});})(c__18933__auto___25699,out))
;
return ((function (switch__18821__auto__,c__18933__auto___25699,out){
return (function() {
var figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__ = null;
var figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____0 = (function (){
var statearr_25695 = [null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_25695[(0)] = figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__);

(statearr_25695[(1)] = (1));

return statearr_25695;
});
var figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____1 = (function (state_25681){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_25681);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e25696){if((e25696 instanceof Object)){
var ex__18825__auto__ = e25696;
var statearr_25697_25706 = state_25681;
(statearr_25697_25706[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_25681);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e25696;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__25707 = state_25681;
state_25681 = G__25707;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__ = function(state_25681){
switch(arguments.length){
case 0:
return figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____1.call(this,state_25681);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____0;
figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto____1;
return figwheel$client$file_reloading$load_all_js_files_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto___25699,out))
})();
var state__18935__auto__ = (function (){var statearr_25698 = f__18934__auto__.call(null);
(statearr_25698[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto___25699);

return statearr_25698;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto___25699,out))
);


return cljs.core.async.into.call(null,cljs.core.PersistentVector.EMPTY,out);
});
figwheel.client.file_reloading.eval_body = (function figwheel$client$file_reloading$eval_body(p__25708,opts){
var map__25712 = p__25708;
var map__25712__$1 = ((((!((map__25712 == null)))?((((map__25712.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25712.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25712):map__25712);
var eval_body__$1 = cljs.core.get.call(null,map__25712__$1,new cljs.core.Keyword(null,"eval-body","eval-body",-907279883));
var file = cljs.core.get.call(null,map__25712__$1,new cljs.core.Keyword(null,"file","file",-1269645878));
if(cljs.core.truth_((function (){var and__16754__auto__ = eval_body__$1;
if(cljs.core.truth_(and__16754__auto__)){
return typeof eval_body__$1 === 'string';
} else {
return and__16754__auto__;
}
})())){
var code = eval_body__$1;
try{figwheel.client.utils.debug_prn.call(null,[cljs.core.str("Evaling file "),cljs.core.str(file)].join(''));

return figwheel.client.utils.eval_helper.call(null,code,opts);
}catch (e25714){var e = e25714;
return figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"error","error",-978969032),[cljs.core.str("Unable to evaluate "),cljs.core.str(file)].join(''));
}} else {
return null;
}
});
figwheel.client.file_reloading.expand_files = (function figwheel$client$file_reloading$expand_files(files){
var deps = figwheel.client.file_reloading.get_all_dependents.call(null,cljs.core.map.call(null,new cljs.core.Keyword(null,"namespace","namespace",-377510372),files));
return cljs.core.filter.call(null,cljs.core.comp.call(null,cljs.core.not,new cljs.core.PersistentHashSet(null, new cljs.core.PersistentArrayMap(null, 1, ["figwheel.connect",null], null), null),new cljs.core.Keyword(null,"namespace","namespace",-377510372)),cljs.core.map.call(null,((function (deps){
return (function (n){
var temp__4423__auto__ = cljs.core.first.call(null,cljs.core.filter.call(null,((function (deps){
return (function (p1__25715_SHARP_){
return cljs.core._EQ_.call(null,new cljs.core.Keyword(null,"namespace","namespace",-377510372).cljs$core$IFn$_invoke$arity$1(p1__25715_SHARP_),n);
});})(deps))
,files));
if(cljs.core.truth_(temp__4423__auto__)){
var file_msg = temp__4423__auto__;
return file_msg;
} else {
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"namespace","namespace",-377510372),new cljs.core.Keyword(null,"namespace","namespace",-377510372),n], null);
}
});})(deps))
,deps));
});
figwheel.client.file_reloading.sort_files = (function figwheel$client$file_reloading$sort_files(files){
if((cljs.core.count.call(null,files) <= (1))){
return files;
} else {
var keep_files = cljs.core.set.call(null,cljs.core.keep.call(null,new cljs.core.Keyword(null,"namespace","namespace",-377510372),files));
return cljs.core.filter.call(null,cljs.core.comp.call(null,keep_files,new cljs.core.Keyword(null,"namespace","namespace",-377510372)),figwheel.client.file_reloading.expand_files.call(null,files));
}
});
figwheel.client.file_reloading.get_figwheel_always = (function figwheel$client$file_reloading$get_figwheel_always(){
return cljs.core.map.call(null,(function (p__25720){
var vec__25721 = p__25720;
var k = cljs.core.nth.call(null,vec__25721,(0),null);
var v = cljs.core.nth.call(null,vec__25721,(1),null);
return new cljs.core.PersistentArrayMap(null, 2, [new cljs.core.Keyword(null,"namespace","namespace",-377510372),k,new cljs.core.Keyword(null,"type","type",1174270348),new cljs.core.Keyword(null,"namespace","namespace",-377510372)], null);
}),cljs.core.filter.call(null,(function (p__25722){
var vec__25723 = p__25722;
var k = cljs.core.nth.call(null,vec__25723,(0),null);
var v = cljs.core.nth.call(null,vec__25723,(1),null);
return new cljs.core.Keyword(null,"figwheel-always","figwheel-always",799819691).cljs$core$IFn$_invoke$arity$1(v);
}),cljs.core.deref.call(null,figwheel.client.file_reloading.figwheel_meta_pragmas)));
});
figwheel.client.file_reloading.reload_js_files = (function figwheel$client$file_reloading$reload_js_files(p__25727,p__25728){
var map__25975 = p__25727;
var map__25975__$1 = ((((!((map__25975 == null)))?((((map__25975.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25975.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25975):map__25975);
var opts = map__25975__$1;
var before_jsload = cljs.core.get.call(null,map__25975__$1,new cljs.core.Keyword(null,"before-jsload","before-jsload",-847513128));
var on_jsload = cljs.core.get.call(null,map__25975__$1,new cljs.core.Keyword(null,"on-jsload","on-jsload",-395756602));
var reload_dependents = cljs.core.get.call(null,map__25975__$1,new cljs.core.Keyword(null,"reload-dependents","reload-dependents",-956865430));
var map__25976 = p__25728;
var map__25976__$1 = ((((!((map__25976 == null)))?((((map__25976.cljs$lang$protocol_mask$partition0$ & (64))) || (map__25976.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__25976):map__25976);
var msg = map__25976__$1;
var files = cljs.core.get.call(null,map__25976__$1,new cljs.core.Keyword(null,"files","files",-472457450));
var figwheel_meta = cljs.core.get.call(null,map__25976__$1,new cljs.core.Keyword(null,"figwheel-meta","figwheel-meta",-225970237));
var recompile_dependents = cljs.core.get.call(null,map__25976__$1,new cljs.core.Keyword(null,"recompile-dependents","recompile-dependents",523804171));
if(cljs.core.empty_QMARK_.call(null,figwheel_meta)){
} else {
cljs.core.reset_BANG_.call(null,figwheel.client.file_reloading.figwheel_meta_pragmas,figwheel_meta);
}

var c__18933__auto__ = cljs.core.async.chan.call(null,(1));
cljs.core.async.impl.dispatch.run.call(null,((function (c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (){
var f__18934__auto__ = (function (){var switch__18821__auto__ = ((function (c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (state_26129){
var state_val_26130 = (state_26129[(1)]);
if((state_val_26130 === (7))){
var inst_25992 = (state_26129[(7)]);
var inst_25991 = (state_26129[(8)]);
var inst_25990 = (state_26129[(9)]);
var inst_25993 = (state_26129[(10)]);
var inst_25998 = cljs.core._nth.call(null,inst_25991,inst_25993);
var inst_25999 = figwheel.client.file_reloading.eval_body.call(null,inst_25998,opts);
var inst_26000 = (inst_25993 + (1));
var tmp26131 = inst_25992;
var tmp26132 = inst_25991;
var tmp26133 = inst_25990;
var inst_25990__$1 = tmp26133;
var inst_25991__$1 = tmp26132;
var inst_25992__$1 = tmp26131;
var inst_25993__$1 = inst_26000;
var state_26129__$1 = (function (){var statearr_26134 = state_26129;
(statearr_26134[(7)] = inst_25992__$1);

(statearr_26134[(11)] = inst_25999);

(statearr_26134[(8)] = inst_25991__$1);

(statearr_26134[(9)] = inst_25990__$1);

(statearr_26134[(10)] = inst_25993__$1);

return statearr_26134;
})();
var statearr_26135_26221 = state_26129__$1;
(statearr_26135_26221[(2)] = null);

(statearr_26135_26221[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (20))){
var inst_26033 = (state_26129[(12)]);
var inst_26041 = figwheel.client.file_reloading.sort_files.call(null,inst_26033);
var state_26129__$1 = state_26129;
var statearr_26136_26222 = state_26129__$1;
(statearr_26136_26222[(2)] = inst_26041);

(statearr_26136_26222[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (27))){
var state_26129__$1 = state_26129;
var statearr_26137_26223 = state_26129__$1;
(statearr_26137_26223[(2)] = null);

(statearr_26137_26223[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (1))){
var inst_25982 = (state_26129[(13)]);
var inst_25979 = before_jsload.call(null,files);
var inst_25980 = figwheel.client.file_reloading.before_jsload_custom_event.call(null,files);
var inst_25981 = (function (){return ((function (inst_25982,inst_25979,inst_25980,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p1__25724_SHARP_){
return new cljs.core.Keyword(null,"eval-body","eval-body",-907279883).cljs$core$IFn$_invoke$arity$1(p1__25724_SHARP_);
});
;})(inst_25982,inst_25979,inst_25980,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_25982__$1 = cljs.core.filter.call(null,inst_25981,files);
var inst_25983 = cljs.core.not_empty.call(null,inst_25982__$1);
var state_26129__$1 = (function (){var statearr_26138 = state_26129;
(statearr_26138[(13)] = inst_25982__$1);

(statearr_26138[(14)] = inst_25980);

(statearr_26138[(15)] = inst_25979);

return statearr_26138;
})();
if(cljs.core.truth_(inst_25983)){
var statearr_26139_26224 = state_26129__$1;
(statearr_26139_26224[(1)] = (2));

} else {
var statearr_26140_26225 = state_26129__$1;
(statearr_26140_26225[(1)] = (3));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (24))){
var state_26129__$1 = state_26129;
var statearr_26141_26226 = state_26129__$1;
(statearr_26141_26226[(2)] = null);

(statearr_26141_26226[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (39))){
var inst_26083 = (state_26129[(16)]);
var state_26129__$1 = state_26129;
var statearr_26142_26227 = state_26129__$1;
(statearr_26142_26227[(2)] = inst_26083);

(statearr_26142_26227[(1)] = (40));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (46))){
var inst_26124 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26143_26228 = state_26129__$1;
(statearr_26143_26228[(2)] = inst_26124);

(statearr_26143_26228[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (4))){
var inst_26027 = (state_26129[(2)]);
var inst_26028 = cljs.core.List.EMPTY;
var inst_26029 = cljs.core.reset_BANG_.call(null,figwheel.client.file_reloading.dependencies_loaded,inst_26028);
var inst_26030 = (function (){return ((function (inst_26027,inst_26028,inst_26029,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p1__25725_SHARP_){
var and__16754__auto__ = new cljs.core.Keyword(null,"namespace","namespace",-377510372).cljs$core$IFn$_invoke$arity$1(p1__25725_SHARP_);
if(cljs.core.truth_(and__16754__auto__)){
return cljs.core.not.call(null,new cljs.core.Keyword(null,"eval-body","eval-body",-907279883).cljs$core$IFn$_invoke$arity$1(p1__25725_SHARP_));
} else {
return and__16754__auto__;
}
});
;})(inst_26027,inst_26028,inst_26029,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26031 = cljs.core.filter.call(null,inst_26030,files);
var inst_26032 = figwheel.client.file_reloading.get_figwheel_always.call(null);
var inst_26033 = cljs.core.concat.call(null,inst_26031,inst_26032);
var state_26129__$1 = (function (){var statearr_26144 = state_26129;
(statearr_26144[(17)] = inst_26027);

(statearr_26144[(18)] = inst_26029);

(statearr_26144[(12)] = inst_26033);

return statearr_26144;
})();
if(cljs.core.truth_(reload_dependents)){
var statearr_26145_26229 = state_26129__$1;
(statearr_26145_26229[(1)] = (16));

} else {
var statearr_26146_26230 = state_26129__$1;
(statearr_26146_26230[(1)] = (17));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (15))){
var inst_26017 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26147_26231 = state_26129__$1;
(statearr_26147_26231[(2)] = inst_26017);

(statearr_26147_26231[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (21))){
var inst_26043 = (state_26129[(19)]);
var inst_26043__$1 = (state_26129[(2)]);
var inst_26044 = figwheel.client.file_reloading.load_all_js_files.call(null,inst_26043__$1);
var state_26129__$1 = (function (){var statearr_26148 = state_26129;
(statearr_26148[(19)] = inst_26043__$1);

return statearr_26148;
})();
return cljs.core.async.impl.ioc_helpers.take_BANG_.call(null,state_26129__$1,(22),inst_26044);
} else {
if((state_val_26130 === (31))){
var inst_26127 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
return cljs.core.async.impl.ioc_helpers.return_chan.call(null,state_26129__$1,inst_26127);
} else {
if((state_val_26130 === (32))){
var inst_26083 = (state_26129[(16)]);
var inst_26088 = inst_26083.cljs$lang$protocol_mask$partition0$;
var inst_26089 = (inst_26088 & (64));
var inst_26090 = inst_26083.cljs$core$ISeq$;
var inst_26091 = (inst_26089) || (inst_26090);
var state_26129__$1 = state_26129;
if(cljs.core.truth_(inst_26091)){
var statearr_26149_26232 = state_26129__$1;
(statearr_26149_26232[(1)] = (35));

} else {
var statearr_26150_26233 = state_26129__$1;
(statearr_26150_26233[(1)] = (36));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (40))){
var inst_26104 = (state_26129[(20)]);
var inst_26103 = (state_26129[(2)]);
var inst_26104__$1 = cljs.core.get.call(null,inst_26103,new cljs.core.Keyword(null,"figwheel-no-load","figwheel-no-load",-555840179));
var inst_26105 = cljs.core.get.call(null,inst_26103,new cljs.core.Keyword(null,"not-required","not-required",-950359114));
var inst_26106 = cljs.core.not_empty.call(null,inst_26104__$1);
var state_26129__$1 = (function (){var statearr_26151 = state_26129;
(statearr_26151[(21)] = inst_26105);

(statearr_26151[(20)] = inst_26104__$1);

return statearr_26151;
})();
if(cljs.core.truth_(inst_26106)){
var statearr_26152_26234 = state_26129__$1;
(statearr_26152_26234[(1)] = (41));

} else {
var statearr_26153_26235 = state_26129__$1;
(statearr_26153_26235[(1)] = (42));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (33))){
var state_26129__$1 = state_26129;
var statearr_26154_26236 = state_26129__$1;
(statearr_26154_26236[(2)] = false);

(statearr_26154_26236[(1)] = (34));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (13))){
var inst_26003 = (state_26129[(22)]);
var inst_26007 = cljs.core.chunk_first.call(null,inst_26003);
var inst_26008 = cljs.core.chunk_rest.call(null,inst_26003);
var inst_26009 = cljs.core.count.call(null,inst_26007);
var inst_25990 = inst_26008;
var inst_25991 = inst_26007;
var inst_25992 = inst_26009;
var inst_25993 = (0);
var state_26129__$1 = (function (){var statearr_26155 = state_26129;
(statearr_26155[(7)] = inst_25992);

(statearr_26155[(8)] = inst_25991);

(statearr_26155[(9)] = inst_25990);

(statearr_26155[(10)] = inst_25993);

return statearr_26155;
})();
var statearr_26156_26237 = state_26129__$1;
(statearr_26156_26237[(2)] = null);

(statearr_26156_26237[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (22))){
var inst_26051 = (state_26129[(23)]);
var inst_26043 = (state_26129[(19)]);
var inst_26047 = (state_26129[(24)]);
var inst_26046 = (state_26129[(25)]);
var inst_26046__$1 = (state_26129[(2)]);
var inst_26047__$1 = cljs.core.filter.call(null,new cljs.core.Keyword(null,"loaded-file","loaded-file",-168399375),inst_26046__$1);
var inst_26048 = (function (){var all_files = inst_26043;
var res_SINGLEQUOTE_ = inst_26046__$1;
var res = inst_26047__$1;
return ((function (all_files,res_SINGLEQUOTE_,res,inst_26051,inst_26043,inst_26047,inst_26046,inst_26046__$1,inst_26047__$1,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p1__25726_SHARP_){
return cljs.core.not.call(null,new cljs.core.Keyword(null,"loaded-file","loaded-file",-168399375).cljs$core$IFn$_invoke$arity$1(p1__25726_SHARP_));
});
;})(all_files,res_SINGLEQUOTE_,res,inst_26051,inst_26043,inst_26047,inst_26046,inst_26046__$1,inst_26047__$1,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26049 = cljs.core.filter.call(null,inst_26048,inst_26046__$1);
var inst_26050 = cljs.core.deref.call(null,figwheel.client.file_reloading.dependencies_loaded);
var inst_26051__$1 = cljs.core.filter.call(null,new cljs.core.Keyword(null,"loaded-file","loaded-file",-168399375),inst_26050);
var inst_26052 = cljs.core.not_empty.call(null,inst_26051__$1);
var state_26129__$1 = (function (){var statearr_26157 = state_26129;
(statearr_26157[(26)] = inst_26049);

(statearr_26157[(23)] = inst_26051__$1);

(statearr_26157[(24)] = inst_26047__$1);

(statearr_26157[(25)] = inst_26046__$1);

return statearr_26157;
})();
if(cljs.core.truth_(inst_26052)){
var statearr_26158_26238 = state_26129__$1;
(statearr_26158_26238[(1)] = (23));

} else {
var statearr_26159_26239 = state_26129__$1;
(statearr_26159_26239[(1)] = (24));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (36))){
var state_26129__$1 = state_26129;
var statearr_26160_26240 = state_26129__$1;
(statearr_26160_26240[(2)] = false);

(statearr_26160_26240[(1)] = (37));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (41))){
var inst_26104 = (state_26129[(20)]);
var inst_26108 = cljs.core.comp.call(null,figwheel.client.file_reloading.name__GT_path,new cljs.core.Keyword(null,"namespace","namespace",-377510372));
var inst_26109 = cljs.core.map.call(null,inst_26108,inst_26104);
var inst_26110 = cljs.core.pr_str.call(null,inst_26109);
var inst_26111 = [cljs.core.str("figwheel-no-load meta-data: "),cljs.core.str(inst_26110)].join('');
var inst_26112 = figwheel.client.utils.log.call(null,inst_26111);
var state_26129__$1 = state_26129;
var statearr_26161_26241 = state_26129__$1;
(statearr_26161_26241[(2)] = inst_26112);

(statearr_26161_26241[(1)] = (43));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (43))){
var inst_26105 = (state_26129[(21)]);
var inst_26115 = (state_26129[(2)]);
var inst_26116 = cljs.core.not_empty.call(null,inst_26105);
var state_26129__$1 = (function (){var statearr_26162 = state_26129;
(statearr_26162[(27)] = inst_26115);

return statearr_26162;
})();
if(cljs.core.truth_(inst_26116)){
var statearr_26163_26242 = state_26129__$1;
(statearr_26163_26242[(1)] = (44));

} else {
var statearr_26164_26243 = state_26129__$1;
(statearr_26164_26243[(1)] = (45));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (29))){
var inst_26049 = (state_26129[(26)]);
var inst_26051 = (state_26129[(23)]);
var inst_26043 = (state_26129[(19)]);
var inst_26083 = (state_26129[(16)]);
var inst_26047 = (state_26129[(24)]);
var inst_26046 = (state_26129[(25)]);
var inst_26079 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: NOT loading these files ");
var inst_26082 = (function (){var all_files = inst_26043;
var res_SINGLEQUOTE_ = inst_26046;
var res = inst_26047;
var files_not_loaded = inst_26049;
var dependencies_that_loaded = inst_26051;
return ((function (all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26083,inst_26047,inst_26046,inst_26079,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p__26081){
var map__26165 = p__26081;
var map__26165__$1 = ((((!((map__26165 == null)))?((((map__26165.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26165.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26165):map__26165);
var namespace = cljs.core.get.call(null,map__26165__$1,new cljs.core.Keyword(null,"namespace","namespace",-377510372));
var meta_data = cljs.core.get.call(null,cljs.core.deref.call(null,figwheel.client.file_reloading.figwheel_meta_pragmas),cljs.core.name.call(null,namespace));
if((meta_data == null)){
return new cljs.core.Keyword(null,"not-required","not-required",-950359114);
} else {
if(cljs.core.truth_(meta_data.call(null,new cljs.core.Keyword(null,"figwheel-no-load","figwheel-no-load",-555840179)))){
return new cljs.core.Keyword(null,"figwheel-no-load","figwheel-no-load",-555840179);
} else {
return new cljs.core.Keyword(null,"not-required","not-required",-950359114);

}
}
});
;})(all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26083,inst_26047,inst_26046,inst_26079,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26083__$1 = cljs.core.group_by.call(null,inst_26082,inst_26049);
var inst_26085 = (inst_26083__$1 == null);
var inst_26086 = cljs.core.not.call(null,inst_26085);
var state_26129__$1 = (function (){var statearr_26167 = state_26129;
(statearr_26167[(16)] = inst_26083__$1);

(statearr_26167[(28)] = inst_26079);

return statearr_26167;
})();
if(inst_26086){
var statearr_26168_26244 = state_26129__$1;
(statearr_26168_26244[(1)] = (32));

} else {
var statearr_26169_26245 = state_26129__$1;
(statearr_26169_26245[(1)] = (33));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (44))){
var inst_26105 = (state_26129[(21)]);
var inst_26118 = cljs.core.map.call(null,new cljs.core.Keyword(null,"file","file",-1269645878),inst_26105);
var inst_26119 = cljs.core.pr_str.call(null,inst_26118);
var inst_26120 = [cljs.core.str("not required: "),cljs.core.str(inst_26119)].join('');
var inst_26121 = figwheel.client.utils.log.call(null,inst_26120);
var state_26129__$1 = state_26129;
var statearr_26170_26246 = state_26129__$1;
(statearr_26170_26246[(2)] = inst_26121);

(statearr_26170_26246[(1)] = (46));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (6))){
var inst_26024 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26171_26247 = state_26129__$1;
(statearr_26171_26247[(2)] = inst_26024);

(statearr_26171_26247[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (28))){
var inst_26049 = (state_26129[(26)]);
var inst_26076 = (state_26129[(2)]);
var inst_26077 = cljs.core.not_empty.call(null,inst_26049);
var state_26129__$1 = (function (){var statearr_26172 = state_26129;
(statearr_26172[(29)] = inst_26076);

return statearr_26172;
})();
if(cljs.core.truth_(inst_26077)){
var statearr_26173_26248 = state_26129__$1;
(statearr_26173_26248[(1)] = (29));

} else {
var statearr_26174_26249 = state_26129__$1;
(statearr_26174_26249[(1)] = (30));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (25))){
var inst_26047 = (state_26129[(24)]);
var inst_26063 = (state_26129[(2)]);
var inst_26064 = cljs.core.not_empty.call(null,inst_26047);
var state_26129__$1 = (function (){var statearr_26175 = state_26129;
(statearr_26175[(30)] = inst_26063);

return statearr_26175;
})();
if(cljs.core.truth_(inst_26064)){
var statearr_26176_26250 = state_26129__$1;
(statearr_26176_26250[(1)] = (26));

} else {
var statearr_26177_26251 = state_26129__$1;
(statearr_26177_26251[(1)] = (27));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (34))){
var inst_26098 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
if(cljs.core.truth_(inst_26098)){
var statearr_26178_26252 = state_26129__$1;
(statearr_26178_26252[(1)] = (38));

} else {
var statearr_26179_26253 = state_26129__$1;
(statearr_26179_26253[(1)] = (39));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (17))){
var state_26129__$1 = state_26129;
var statearr_26180_26254 = state_26129__$1;
(statearr_26180_26254[(2)] = recompile_dependents);

(statearr_26180_26254[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (3))){
var state_26129__$1 = state_26129;
var statearr_26181_26255 = state_26129__$1;
(statearr_26181_26255[(2)] = null);

(statearr_26181_26255[(1)] = (4));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (12))){
var inst_26020 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26182_26256 = state_26129__$1;
(statearr_26182_26256[(2)] = inst_26020);

(statearr_26182_26256[(1)] = (9));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (2))){
var inst_25982 = (state_26129[(13)]);
var inst_25989 = cljs.core.seq.call(null,inst_25982);
var inst_25990 = inst_25989;
var inst_25991 = null;
var inst_25992 = (0);
var inst_25993 = (0);
var state_26129__$1 = (function (){var statearr_26183 = state_26129;
(statearr_26183[(7)] = inst_25992);

(statearr_26183[(8)] = inst_25991);

(statearr_26183[(9)] = inst_25990);

(statearr_26183[(10)] = inst_25993);

return statearr_26183;
})();
var statearr_26184_26257 = state_26129__$1;
(statearr_26184_26257[(2)] = null);

(statearr_26184_26257[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (23))){
var inst_26049 = (state_26129[(26)]);
var inst_26051 = (state_26129[(23)]);
var inst_26043 = (state_26129[(19)]);
var inst_26047 = (state_26129[(24)]);
var inst_26046 = (state_26129[(25)]);
var inst_26054 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: loaded these dependencies");
var inst_26056 = (function (){var all_files = inst_26043;
var res_SINGLEQUOTE_ = inst_26046;
var res = inst_26047;
var files_not_loaded = inst_26049;
var dependencies_that_loaded = inst_26051;
return ((function (all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26054,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p__26055){
var map__26185 = p__26055;
var map__26185__$1 = ((((!((map__26185 == null)))?((((map__26185.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26185.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26185):map__26185);
var request_url = cljs.core.get.call(null,map__26185__$1,new cljs.core.Keyword(null,"request-url","request-url",2100346596));
return clojure.string.replace.call(null,request_url,goog.basePath,"");
});
;})(all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26054,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26057 = cljs.core.reverse.call(null,inst_26051);
var inst_26058 = cljs.core.map.call(null,inst_26056,inst_26057);
var inst_26059 = cljs.core.pr_str.call(null,inst_26058);
var inst_26060 = figwheel.client.utils.log.call(null,inst_26059);
var state_26129__$1 = (function (){var statearr_26187 = state_26129;
(statearr_26187[(31)] = inst_26054);

return statearr_26187;
})();
var statearr_26188_26258 = state_26129__$1;
(statearr_26188_26258[(2)] = inst_26060);

(statearr_26188_26258[(1)] = (25));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (35))){
var state_26129__$1 = state_26129;
var statearr_26189_26259 = state_26129__$1;
(statearr_26189_26259[(2)] = true);

(statearr_26189_26259[(1)] = (37));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (19))){
var inst_26033 = (state_26129[(12)]);
var inst_26039 = figwheel.client.file_reloading.expand_files.call(null,inst_26033);
var state_26129__$1 = state_26129;
var statearr_26190_26260 = state_26129__$1;
(statearr_26190_26260[(2)] = inst_26039);

(statearr_26190_26260[(1)] = (21));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (11))){
var state_26129__$1 = state_26129;
var statearr_26191_26261 = state_26129__$1;
(statearr_26191_26261[(2)] = null);

(statearr_26191_26261[(1)] = (12));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (9))){
var inst_26022 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26192_26262 = state_26129__$1;
(statearr_26192_26262[(2)] = inst_26022);

(statearr_26192_26262[(1)] = (6));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (5))){
var inst_25992 = (state_26129[(7)]);
var inst_25993 = (state_26129[(10)]);
var inst_25995 = (inst_25993 < inst_25992);
var inst_25996 = inst_25995;
var state_26129__$1 = state_26129;
if(cljs.core.truth_(inst_25996)){
var statearr_26193_26263 = state_26129__$1;
(statearr_26193_26263[(1)] = (7));

} else {
var statearr_26194_26264 = state_26129__$1;
(statearr_26194_26264[(1)] = (8));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (14))){
var inst_26003 = (state_26129[(22)]);
var inst_26012 = cljs.core.first.call(null,inst_26003);
var inst_26013 = figwheel.client.file_reloading.eval_body.call(null,inst_26012,opts);
var inst_26014 = cljs.core.next.call(null,inst_26003);
var inst_25990 = inst_26014;
var inst_25991 = null;
var inst_25992 = (0);
var inst_25993 = (0);
var state_26129__$1 = (function (){var statearr_26195 = state_26129;
(statearr_26195[(7)] = inst_25992);

(statearr_26195[(32)] = inst_26013);

(statearr_26195[(8)] = inst_25991);

(statearr_26195[(9)] = inst_25990);

(statearr_26195[(10)] = inst_25993);

return statearr_26195;
})();
var statearr_26196_26265 = state_26129__$1;
(statearr_26196_26265[(2)] = null);

(statearr_26196_26265[(1)] = (5));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (45))){
var state_26129__$1 = state_26129;
var statearr_26197_26266 = state_26129__$1;
(statearr_26197_26266[(2)] = null);

(statearr_26197_26266[(1)] = (46));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (26))){
var inst_26049 = (state_26129[(26)]);
var inst_26051 = (state_26129[(23)]);
var inst_26043 = (state_26129[(19)]);
var inst_26047 = (state_26129[(24)]);
var inst_26046 = (state_26129[(25)]);
var inst_26066 = figwheel.client.utils.log.call(null,new cljs.core.Keyword(null,"debug","debug",-1608172596),"Figwheel: loaded these files");
var inst_26068 = (function (){var all_files = inst_26043;
var res_SINGLEQUOTE_ = inst_26046;
var res = inst_26047;
var files_not_loaded = inst_26049;
var dependencies_that_loaded = inst_26051;
return ((function (all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26066,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (p__26067){
var map__26198 = p__26067;
var map__26198__$1 = ((((!((map__26198 == null)))?((((map__26198.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26198.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26198):map__26198);
var namespace = cljs.core.get.call(null,map__26198__$1,new cljs.core.Keyword(null,"namespace","namespace",-377510372));
var file = cljs.core.get.call(null,map__26198__$1,new cljs.core.Keyword(null,"file","file",-1269645878));
if(cljs.core.truth_(namespace)){
return figwheel.client.file_reloading.name__GT_path.call(null,cljs.core.name.call(null,namespace));
} else {
return file;
}
});
;})(all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26066,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26069 = cljs.core.map.call(null,inst_26068,inst_26047);
var inst_26070 = cljs.core.pr_str.call(null,inst_26069);
var inst_26071 = figwheel.client.utils.log.call(null,inst_26070);
var inst_26072 = (function (){var all_files = inst_26043;
var res_SINGLEQUOTE_ = inst_26046;
var res = inst_26047;
var files_not_loaded = inst_26049;
var dependencies_that_loaded = inst_26051;
return ((function (all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26066,inst_26068,inst_26069,inst_26070,inst_26071,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function (){
figwheel.client.file_reloading.on_jsload_custom_event.call(null,res);

return cljs.core.apply.call(null,on_jsload,new cljs.core.PersistentVector(null, 1, 5, cljs.core.PersistentVector.EMPTY_NODE, [res], null));
});
;})(all_files,res_SINGLEQUOTE_,res,files_not_loaded,dependencies_that_loaded,inst_26049,inst_26051,inst_26043,inst_26047,inst_26046,inst_26066,inst_26068,inst_26069,inst_26070,inst_26071,state_val_26130,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var inst_26073 = setTimeout(inst_26072,(10));
var state_26129__$1 = (function (){var statearr_26200 = state_26129;
(statearr_26200[(33)] = inst_26066);

(statearr_26200[(34)] = inst_26071);

return statearr_26200;
})();
var statearr_26201_26267 = state_26129__$1;
(statearr_26201_26267[(2)] = inst_26073);

(statearr_26201_26267[(1)] = (28));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (16))){
var state_26129__$1 = state_26129;
var statearr_26202_26268 = state_26129__$1;
(statearr_26202_26268[(2)] = reload_dependents);

(statearr_26202_26268[(1)] = (18));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (38))){
var inst_26083 = (state_26129[(16)]);
var inst_26100 = cljs.core.apply.call(null,cljs.core.hash_map,inst_26083);
var state_26129__$1 = state_26129;
var statearr_26203_26269 = state_26129__$1;
(statearr_26203_26269[(2)] = inst_26100);

(statearr_26203_26269[(1)] = (40));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (30))){
var state_26129__$1 = state_26129;
var statearr_26204_26270 = state_26129__$1;
(statearr_26204_26270[(2)] = null);

(statearr_26204_26270[(1)] = (31));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (10))){
var inst_26003 = (state_26129[(22)]);
var inst_26005 = cljs.core.chunked_seq_QMARK_.call(null,inst_26003);
var state_26129__$1 = state_26129;
if(inst_26005){
var statearr_26205_26271 = state_26129__$1;
(statearr_26205_26271[(1)] = (13));

} else {
var statearr_26206_26272 = state_26129__$1;
(statearr_26206_26272[(1)] = (14));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (18))){
var inst_26037 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
if(cljs.core.truth_(inst_26037)){
var statearr_26207_26273 = state_26129__$1;
(statearr_26207_26273[(1)] = (19));

} else {
var statearr_26208_26274 = state_26129__$1;
(statearr_26208_26274[(1)] = (20));

}

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (42))){
var state_26129__$1 = state_26129;
var statearr_26209_26275 = state_26129__$1;
(statearr_26209_26275[(2)] = null);

(statearr_26209_26275[(1)] = (43));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (37))){
var inst_26095 = (state_26129[(2)]);
var state_26129__$1 = state_26129;
var statearr_26210_26276 = state_26129__$1;
(statearr_26210_26276[(2)] = inst_26095);

(statearr_26210_26276[(1)] = (34));


return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
if((state_val_26130 === (8))){
var inst_25990 = (state_26129[(9)]);
var inst_26003 = (state_26129[(22)]);
var inst_26003__$1 = cljs.core.seq.call(null,inst_25990);
var state_26129__$1 = (function (){var statearr_26211 = state_26129;
(statearr_26211[(22)] = inst_26003__$1);

return statearr_26211;
})();
if(inst_26003__$1){
var statearr_26212_26277 = state_26129__$1;
(statearr_26212_26277[(1)] = (10));

} else {
var statearr_26213_26278 = state_26129__$1;
(statearr_26213_26278[(1)] = (11));

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
});})(c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
;
return ((function (switch__18821__auto__,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents){
return (function() {
var figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__ = null;
var figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____0 = (function (){
var statearr_26217 = [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null];
(statearr_26217[(0)] = figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__);

(statearr_26217[(1)] = (1));

return statearr_26217;
});
var figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____1 = (function (state_26129){
while(true){
var ret_value__18823__auto__ = (function (){try{while(true){
var result__18824__auto__ = switch__18821__auto__.call(null,state_26129);
if(cljs.core.keyword_identical_QMARK_.call(null,result__18824__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
continue;
} else {
return result__18824__auto__;
}
break;
}
}catch (e26218){if((e26218 instanceof Object)){
var ex__18825__auto__ = e26218;
var statearr_26219_26279 = state_26129;
(statearr_26219_26279[(5)] = ex__18825__auto__);


cljs.core.async.impl.ioc_helpers.process_exception.call(null,state_26129);

return new cljs.core.Keyword(null,"recur","recur",-437573268);
} else {
throw e26218;

}
}})();
if(cljs.core.keyword_identical_QMARK_.call(null,ret_value__18823__auto__,new cljs.core.Keyword(null,"recur","recur",-437573268))){
var G__26280 = state_26129;
state_26129 = G__26280;
continue;
} else {
return ret_value__18823__auto__;
}
break;
}
});
figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__ = function(state_26129){
switch(arguments.length){
case 0:
return figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____0.call(this);
case 1:
return figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____1.call(this,state_26129);
}
throw(new Error('Invalid arity: ' + arguments.length));
};
figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$0 = figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____0;
figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__.cljs$core$IFn$_invoke$arity$1 = figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto____1;
return figwheel$client$file_reloading$reload_js_files_$_state_machine__18822__auto__;
})()
;})(switch__18821__auto__,c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
})();
var state__18935__auto__ = (function (){var statearr_26220 = f__18934__auto__.call(null);
(statearr_26220[cljs.core.async.impl.ioc_helpers.USER_START_IDX] = c__18933__auto__);

return statearr_26220;
})();
return cljs.core.async.impl.ioc_helpers.run_state_machine_wrapped.call(null,state__18935__auto__);
});})(c__18933__auto__,map__25975,map__25975__$1,opts,before_jsload,on_jsload,reload_dependents,map__25976,map__25976__$1,msg,files,figwheel_meta,recompile_dependents))
);

return c__18933__auto__;
});
figwheel.client.file_reloading.current_links = (function figwheel$client$file_reloading$current_links(){
return Array.prototype.slice.call(document.getElementsByTagName("link"));
});
figwheel.client.file_reloading.truncate_url = (function figwheel$client$file_reloading$truncate_url(url){
return clojure.string.replace_first.call(null,clojure.string.replace_first.call(null,clojure.string.replace_first.call(null,clojure.string.replace_first.call(null,cljs.core.first.call(null,clojure.string.split.call(null,url,/\?/)),[cljs.core.str(location.protocol),cljs.core.str("//")].join(''),""),".*://",""),/^\/\//,""),/[^\\/]*/,"");
});
figwheel.client.file_reloading.matches_file_QMARK_ = (function figwheel$client$file_reloading$matches_file_QMARK_(p__26283,link){
var map__26286 = p__26283;
var map__26286__$1 = ((((!((map__26286 == null)))?((((map__26286.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26286.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26286):map__26286);
var file = cljs.core.get.call(null,map__26286__$1,new cljs.core.Keyword(null,"file","file",-1269645878));
var temp__4425__auto__ = link.href;
if(cljs.core.truth_(temp__4425__auto__)){
var link_href = temp__4425__auto__;
var match = clojure.string.join.call(null,"/",cljs.core.take_while.call(null,cljs.core.identity,cljs.core.map.call(null,((function (link_href,temp__4425__auto__,map__26286,map__26286__$1,file){
return (function (p1__26281_SHARP_,p2__26282_SHARP_){
if(cljs.core._EQ_.call(null,p1__26281_SHARP_,p2__26282_SHARP_)){
return p1__26281_SHARP_;
} else {
return false;
}
});})(link_href,temp__4425__auto__,map__26286,map__26286__$1,file))
,cljs.core.reverse.call(null,clojure.string.split.call(null,file,"/")),cljs.core.reverse.call(null,clojure.string.split.call(null,figwheel.client.file_reloading.truncate_url.call(null,link_href),"/")))));
var match_length = cljs.core.count.call(null,match);
var file_name_length = cljs.core.count.call(null,cljs.core.last.call(null,clojure.string.split.call(null,file,"/")));
if((match_length >= file_name_length)){
return new cljs.core.PersistentArrayMap(null, 4, [new cljs.core.Keyword(null,"link","link",-1769163468),link,new cljs.core.Keyword(null,"link-href","link-href",-250644450),link_href,new cljs.core.Keyword(null,"match-length","match-length",1101537310),match_length,new cljs.core.Keyword(null,"current-url-length","current-url-length",380404083),cljs.core.count.call(null,figwheel.client.file_reloading.truncate_url.call(null,link_href))], null);
} else {
return null;
}
} else {
return null;
}
});
figwheel.client.file_reloading.get_correct_link = (function figwheel$client$file_reloading$get_correct_link(f_data){
var temp__4425__auto__ = cljs.core.first.call(null,cljs.core.sort_by.call(null,(function (p__26292){
var map__26293 = p__26292;
var map__26293__$1 = ((((!((map__26293 == null)))?((((map__26293.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26293.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26293):map__26293);
var match_length = cljs.core.get.call(null,map__26293__$1,new cljs.core.Keyword(null,"match-length","match-length",1101537310));
var current_url_length = cljs.core.get.call(null,map__26293__$1,new cljs.core.Keyword(null,"current-url-length","current-url-length",380404083));
return (current_url_length - match_length);
}),cljs.core.keep.call(null,(function (p1__26288_SHARP_){
return figwheel.client.file_reloading.matches_file_QMARK_.call(null,f_data,p1__26288_SHARP_);
}),figwheel.client.file_reloading.current_links.call(null))));
if(cljs.core.truth_(temp__4425__auto__)){
var res = temp__4425__auto__;
return new cljs.core.Keyword(null,"link","link",-1769163468).cljs$core$IFn$_invoke$arity$1(res);
} else {
return null;
}
});
figwheel.client.file_reloading.clone_link = (function figwheel$client$file_reloading$clone_link(link,url){
var clone = document.createElement("link");
clone.rel = "stylesheet";

clone.media = link.media;

clone.disabled = link.disabled;

clone.href = figwheel.client.file_reloading.add_cache_buster.call(null,url);

return clone;
});
figwheel.client.file_reloading.create_link = (function figwheel$client$file_reloading$create_link(url){
var link = document.createElement("link");
link.rel = "stylesheet";

link.href = figwheel.client.file_reloading.add_cache_buster.call(null,url);

return link;
});
figwheel.client.file_reloading.add_link_to_doc = (function figwheel$client$file_reloading$add_link_to_doc(var_args){
var args26295 = [];
var len__17824__auto___26298 = arguments.length;
var i__17825__auto___26299 = (0);
while(true){
if((i__17825__auto___26299 < len__17824__auto___26298)){
args26295.push((arguments[i__17825__auto___26299]));

var G__26300 = (i__17825__auto___26299 + (1));
i__17825__auto___26299 = G__26300;
continue;
} else {
}
break;
}

var G__26297 = args26295.length;
switch (G__26297) {
case 1:
return figwheel.client.file_reloading.add_link_to_doc.cljs$core$IFn$_invoke$arity$1((arguments[(0)]));

break;
case 2:
return figwheel.client.file_reloading.add_link_to_doc.cljs$core$IFn$_invoke$arity$2((arguments[(0)]),(arguments[(1)]));

break;
default:
throw (new Error([cljs.core.str("Invalid arity: "),cljs.core.str(args26295.length)].join('')));

}
});

figwheel.client.file_reloading.add_link_to_doc.cljs$core$IFn$_invoke$arity$1 = (function (new_link){
return (document.getElementsByTagName("head")[(0)]).appendChild(new_link);
});

figwheel.client.file_reloading.add_link_to_doc.cljs$core$IFn$_invoke$arity$2 = (function (orig_link,klone){
var parent = orig_link.parentNode;
if(cljs.core._EQ_.call(null,orig_link,parent.lastChild)){
parent.appendChild(klone);
} else {
parent.insertBefore(klone,orig_link.nextSibling);
}

return setTimeout(((function (parent){
return (function (){
return parent.removeChild(orig_link);
});})(parent))
,(300));
});

figwheel.client.file_reloading.add_link_to_doc.cljs$lang$maxFixedArity = 2;
figwheel.client.file_reloading.distictify = (function figwheel$client$file_reloading$distictify(key,seqq){
return cljs.core.vals.call(null,cljs.core.reduce.call(null,(function (p1__26302_SHARP_,p2__26303_SHARP_){
return cljs.core.assoc.call(null,p1__26302_SHARP_,cljs.core.get.call(null,p2__26303_SHARP_,key),p2__26303_SHARP_);
}),cljs.core.PersistentArrayMap.EMPTY,seqq));
});
figwheel.client.file_reloading.reload_css_file = (function figwheel$client$file_reloading$reload_css_file(p__26304){
var map__26307 = p__26304;
var map__26307__$1 = ((((!((map__26307 == null)))?((((map__26307.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26307.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26307):map__26307);
var f_data = map__26307__$1;
var file = cljs.core.get.call(null,map__26307__$1,new cljs.core.Keyword(null,"file","file",-1269645878));
var temp__4425__auto__ = figwheel.client.file_reloading.get_correct_link.call(null,f_data);
if(cljs.core.truth_(temp__4425__auto__)){
var link = temp__4425__auto__;
return figwheel.client.file_reloading.add_link_to_doc.call(null,link,figwheel.client.file_reloading.clone_link.call(null,link,link.href));
} else {
return null;
}
});
figwheel.client.file_reloading.reload_css_files = (function figwheel$client$file_reloading$reload_css_files(p__26309,files_msg){
var map__26316 = p__26309;
var map__26316__$1 = ((((!((map__26316 == null)))?((((map__26316.cljs$lang$protocol_mask$partition0$ & (64))) || (map__26316.cljs$core$ISeq$))?true:false):false))?cljs.core.apply.call(null,cljs.core.hash_map,map__26316):map__26316);
var opts = map__26316__$1;
var on_cssload = cljs.core.get.call(null,map__26316__$1,new cljs.core.Keyword(null,"on-cssload","on-cssload",1825432318));
if(cljs.core.truth_(figwheel.client.utils.html_env_QMARK_.call(null))){
var seq__26318_26322 = cljs.core.seq.call(null,figwheel.client.file_reloading.distictify.call(null,new cljs.core.Keyword(null,"file","file",-1269645878),new cljs.core.Keyword(null,"files","files",-472457450).cljs$core$IFn$_invoke$arity$1(files_msg)));
var chunk__26319_26323 = null;
var count__26320_26324 = (0);
var i__26321_26325 = (0);
while(true){
if((i__26321_26325 < count__26320_26324)){
var f_26326 = cljs.core._nth.call(null,chunk__26319_26323,i__26321_26325);
figwheel.client.file_reloading.reload_css_file.call(null,f_26326);

var G__26327 = seq__26318_26322;
var G__26328 = chunk__26319_26323;
var G__26329 = count__26320_26324;
var G__26330 = (i__26321_26325 + (1));
seq__26318_26322 = G__26327;
chunk__26319_26323 = G__26328;
count__26320_26324 = G__26329;
i__26321_26325 = G__26330;
continue;
} else {
var temp__4425__auto___26331 = cljs.core.seq.call(null,seq__26318_26322);
if(temp__4425__auto___26331){
var seq__26318_26332__$1 = temp__4425__auto___26331;
if(cljs.core.chunked_seq_QMARK_.call(null,seq__26318_26332__$1)){
var c__17569__auto___26333 = cljs.core.chunk_first.call(null,seq__26318_26332__$1);
var G__26334 = cljs.core.chunk_rest.call(null,seq__26318_26332__$1);
var G__26335 = c__17569__auto___26333;
var G__26336 = cljs.core.count.call(null,c__17569__auto___26333);
var G__26337 = (0);
seq__26318_26322 = G__26334;
chunk__26319_26323 = G__26335;
count__26320_26324 = G__26336;
i__26321_26325 = G__26337;
continue;
} else {
var f_26338 = cljs.core.first.call(null,seq__26318_26332__$1);
figwheel.client.file_reloading.reload_css_file.call(null,f_26338);

var G__26339 = cljs.core.next.call(null,seq__26318_26332__$1);
var G__26340 = null;
var G__26341 = (0);
var G__26342 = (0);
seq__26318_26322 = G__26339;
chunk__26319_26323 = G__26340;
count__26320_26324 = G__26341;
i__26321_26325 = G__26342;
continue;
}
} else {
}
}
break;
}

return setTimeout(((function (map__26316,map__26316__$1,opts,on_cssload){
return (function (){
return on_cssload.call(null,new cljs.core.Keyword(null,"files","files",-472457450).cljs$core$IFn$_invoke$arity$1(files_msg));
});})(map__26316,map__26316__$1,opts,on_cssload))
,(100));
} else {
return null;
}
});

//# sourceMappingURL=file_reloading.js.map?rel=1454621293518