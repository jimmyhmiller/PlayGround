# List tab titles in Google Chrome windows

set find to "React"
set titleString to "
"

tell application "Google Chrome"
    set window_list to every window # get the windows
    
    repeat with the_window in window_list # for every window
        set tab_list to every tab in the_window # get the tabs
        
        set i to 1
        repeat with the_tab in tab_list # for every tab
            if the title of the_tab contains find then
                tell application "Google Chrome" to set active tab index of the_window to i
                tell application "Google Chrome" to activate tab i of the_window
                return
            end if
            set the_title to the title of the_tab # grab the title
            set titleString to titleString & the_title & return # concatenate
            set i to i + 1
        end repeat
    end repeat
end tell