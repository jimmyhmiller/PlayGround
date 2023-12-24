
use cacao::foundation::NSURL;
use cocoa::{
    appkit::{NSModalResponse, NSOpenPanel, NSSavePanel},
    base::nil,
};

use cocoa::base::id;
use objc::{runtime::Class, msg_send, sel, sel_impl};


pub fn open_file_dialog() -> Option<String> {
    // make an NsOpenPanel
    unsafe {
        let panel = NSOpenPanel::openPanel(nil);
        panel.setCanChooseFiles_(true);
        panel.setCanChooseDirectories_(false);
        panel.setAllowsMultipleSelection_(false);
        let response = panel.runModal();
        match response {
            NSModalResponse::NSModalResponseOk => {
                let url = panel.URL();
                let url = NSURL::retain(url);
                let path = url.absolute_string();
                Some(path)
            }
            NSModalResponse::NSModalResponseCancel => None,
        }
    }
}

pub fn feedback() {
    unsafe {
        let cls = Class::get("NSHapticFeedbackManager").unwrap();
        let performer:id = msg_send![cls, defaultPerformer];
        let _: () = msg_send![performer, performFeedbackPattern:1 performanceTime:0];
    }
}

