extern crate byteorder;

use std::io::{BufReader};
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;
use std::io::SeekFrom;


use byteorder::{ReadBytesExt, BigEndian};


fn seek_to_first_tag(rdr: &mut BufReader<File>) -> () {
	let mut buf = vec![];
	rdr.read_until(0, &mut buf).unwrap();
	rdr.seek(SeekFrom::Current(4)).unwrap();
	rdr.seek(SeekFrom::Current(8)).unwrap();
}


fn seek_to_next_tag(rdr: &mut BufReader<File>) -> u64 {
	rdr.seek(SeekFrom::Current(4)).unwrap() // read_exact?
	let length = rdr.read_u32::<BigEndian>().unwrap();

	rdr.seek(SeekFrom::Current(length as i64)).unwrap();
	(length + 4 + 4) as u64
}


fn main() {
	let mut stats = HashMap::new();

    let f = File::open("/Users/jimmy.miller/Desktop/largedump.hprof").unwrap();
	let mut rdr = BufReader::new(f);

	seek_to_first_tag(&mut rdr);

	let mut length : u64 = 30;
	loop {
		let mut tag_holder = &mut [0];

		let tag_result = rdr.read_exact(tag_holder);
		if tag_result.is_err() {
			break;
		}
		let tag = tag_holder.first().cloned().unwrap();
		length += 1;
		if stats.contains_key(&tag) {
			let mut current_value : &mut Vec<u64> = stats.get_mut(&tag).unwrap();
			current_value.push(length);
		} else {
			stats.insert(tag, vec![length]);
		}

		length += seek_to_next_tag(&mut rdr);
	}

	
	// println!("{:?}", stats);


}
