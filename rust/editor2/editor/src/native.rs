use cacao::foundation::NSURL;
use cocoa::{appkit::{NSOpenPanel, NSSavePanel, NSModalResponse}, base::nil};



pub fn open_file_dialog() -> Option<String> {
    // make an NsOpenPanel
    unsafe {
        let panel =  NSOpenPanel::openPanel(nil);
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
            NSModalResponse::NSModalResponseCancel => None
        }
    }

    
}