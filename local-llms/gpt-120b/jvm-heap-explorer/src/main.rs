use clap::Parser;
use anyhow::{Result, Context};
use std::fs::File;
use std::io::{BufReader, Read, Seek, Cursor};
use binread::BinRead;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the JVM heap dump (.hprof)
    #[arg(short, long)]
    dump: String,
}

#[derive(Debug, BinRead)]
#[br(little)]
struct HprofHeader {
    magic: [u8; 4],
    version: u32,
    timestamp: u64,
    header_len: u32,
    // actual header string follows
}

#[derive(Debug, BinRead)]
#[br(big)]
struct RecordHeader {
    tag: u8,
    timestamp: u32,
    length: u32,
}

fn parse_hprof<R: Read + Seek>(mut reader: R) -> anyhow::Result<()> {
    // read header
    let header: HprofHeader = HprofHeader::read(&mut reader)?;
    let mut desc_bytes = vec![0u8; header.header_len as usize];
    reader.read_exact(&mut desc_bytes)?;
    let description = String::from_utf8_lossy(&desc_bytes);
    println!("Header magic: {}", std::str::from_utf8(&header.magic).unwrap_or("<non‑utf8>"));
    println!("Version: {}", header.version);
    println!("Timestamp: {}", header.timestamp);
    println!("Description: {}", description);

    // collect objects
    let mut objects: Vec<(u64, Option<u64>, String)> = Vec::new();

    // loop records
    loop {
        let header_res = RecordHeader::read(&mut reader);
        let header = match header_res {
            Ok(h) => h,
            Err(_) => break,
        };
        let mut data = vec![0u8; header.length as usize];
        if reader.read_exact(&mut data).is_err() {
            break;
        }
        match header.tag {
            0x21 => {
                #[derive(Debug, BinRead)]
                #[br(big)]
                struct InstanceDump {
                    obj_id: u64,
                    class_id: u64,
                }
                let mut cur = Cursor::new(&data);
                let dump: InstanceDump = InstanceDump::read(&mut cur)?;
                objects.push((dump.obj_id, Some(dump.class_id), "Instance".to_string()));
            }
            0x22 => {
                let mut cur = Cursor::new(&data);
                let array_id: u64 = BinRead::read(&mut cur)?;
                let array_len: u32 = BinRead::read(&mut cur)?;
                objects.push((array_id, None, format!("ObjectArray len {}", array_len)));
            }
            _ => {}
        }
    }

    // list objects
    for (obj_id, class_opt, typ) in &objects {
        match class_opt {
            Some(class_id) => println!("{} {} of class {}", typ, obj_id, class_id),
            None => println!("{} {}", typ, obj_id),
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = File::open(&args.dump)
        .with_context(|| format!("Failed to open dump {}", args.dump))?;
    let mut reader = BufReader::new(file);
    parse_hprof(&mut reader)?;
    Ok(())
}
