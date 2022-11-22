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

