use std::{fs::{File, self}, io::Write};


fn epoch_as_string() -> String {
    let now = std::time::SystemTime::now();
    let since_epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    let since_epoch_ms = since_epoch.as_secs() * 1000 + since_epoch.subsec_nanos() as u64 / 1_000_000;
    since_epoch_ms.to_string()
}

fn write_first(file: &mut File) -> std::io::Result<usize> {
    file.write(format!("Hello, world! {}\n", epoch_as_string()).as_bytes())
}

fn write_second(file: &mut File) -> std::io::Result<usize> {
    file.write(format!("Hello, others! {}\n", epoch_as_string()).as_bytes())
}

fn delete(path: &str) -> std::io::Result<()> {
    fs::remove_file(path)
}



fn main() -> std::io::Result<()> {
    let path = "hello.txt";
    if let Ok(_) = delete(path) {
        println!("File deleted");
    }
    let mut file = File::create(path).unwrap();
    write_first(&mut file)?;
    write_second(&mut file)?;
    Ok(())
}
