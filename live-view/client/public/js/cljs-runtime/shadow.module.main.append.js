
shadow.cljs.devtools.client.env.module_loaded('main');

try { live_view.core.init(); } catch (e) { console.error("An error occurred when calling (live-view.core/init)"); throw(e); }