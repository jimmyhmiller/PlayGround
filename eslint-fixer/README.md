```bash
vim \"+call cursor(%line, %column)\" %file
nano +%line,%column %file
subl -w %file:%line:%column
emacs +%line:%column %file
atom -w %file:%line:%column
code -w -g %file:%line:%column --waits for app to close, not tab
```