# We could of course not use cargo here and just run the executable
# out of the target directory.
# But don't you hate scripts like that? Ones where you have to remember
# to run the build script first and then the run script?
# You always end up doing build.sh && run.sh. It's annoying.
cargo run --release $1