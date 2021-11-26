use std::fs;

use native_dialog::FileDialog;

pub fn set_smooth_scroll() {
    unsafe {
        use cocoa_foundation::base::nil;
        use cocoa_foundation::foundation::NSString;
        use cocoa_foundation::foundation::NSUserDefaults;
        let defaults = cocoa_foundation::base::id::standardUserDefaults();
        let key = NSString::alloc(nil).init_str("AppleMomentumScrollSupported");
        defaults.setBool_forKey_(cocoa_foundation::base::YES, key)
    }
}

pub fn open_file_dialog() -> Option<String> {
    let path = FileDialog::new()
        .set_location("~/Documents")
        .show_open_single_file()
        .unwrap();

    if path.is_none() {
        return None;
    }
    let path = &path.unwrap().to_str().unwrap().replace("file://", "");
    // Need to refactor into reusable function instead of just repeating here.
    fs::read_to_string(path).ok()
}
