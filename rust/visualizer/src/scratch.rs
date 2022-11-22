//     let mut block_to_method: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();

//     let mut record_by_method : HashMap<CodeLocation, Vec<Block>> = HashMap::new();

//     let mut record_by_start_addr : HashMap<usize, Block> = HashMap::new();
//     for record in records.iter() {
//         let start_addr = record.start_addr.unwrap();
//         if record_by_start_addr.contains_key(&start_addr) {
//             let already_recorded = record_by_start_addr.get(&start_addr).unwrap();
//             if already_recorded.epoch > record.epoch {
//                 continue;
//             }
//         }
//         record_by_start_addr.insert(start_addr, record.clone());
//     }

//     let mut all_methods : Vec<CodeLocation> = records.iter().map(|x| x.location.clone()).collect();
//     all_methods.sort_by_key(|x| (x.method_name.clone(), x.file.clone()));
//     all_methods.dedup();

//     // for record in records.iter() {
//     //     if record.start_addr.is_none() {
//     //         continue;
//     //     }
//     //     if record.start_addr == Some(4843271288) {
//     //         println!("Found it");
//     //         panic!("Done")
//     //     } else if (record.start_addr.unwrap().abs_diff(4843271288)) < 10000 {
//     //         println!("Found close {:?}", record.start_addr );
//     //     }
//     // }

//     // // println!("{:?} records", record_by_start_addr.contains_key(4843271288));


//     for record in records.iter() {
//         let key = record.location.clone();
//         let entry = record_by_method.entry(key).or_insert_with(|| vec![]);
//         entry.push(record.clone());
//     }

//     for record in records.iter() {
//         let key = record.block_id.name();
//         let value = (record.location.file.clone(), record.location.method_name.clone());
//         block_to_method.insert(key, value);
//     }

//     // for record in records.iter() {
//     //     let method = (record.location.file.clone(), record.location.method_name.clone());
//     //     for outgoing in record.outgoing.iter() {
//     //         for targets in outgoing.targets.iter() {
//     //             if let Some(target) = targets {
//     //                 if let Some(connecting_method) = block_to_method.get(&target.name()){
//     //                     method_to_method.push((method.clone(), connecting_method.clone()));
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }

//     // println!("Method to Method: {:?}", method_to_method.len());
//     // // sort and dedup
//     // method_to_method.sort();
//     // method_to_method.dedup();
//     // println!("Method to Method: {:?}", method_to_method.len());

//     // let mut all_methods = method_to_method
//     //     .iter()
//     //     .flat_map(|(x, y)| vec![x, y])
//     //     .collect::<Vec<_>>();

//     // all_methods.sort();
//     // all_methods.dedup();


//     let mut nodes: Vec<String> = vec![];

//     // for method in all_methods {
//     //     // make a graphviz node
//     //     nodes.push(format!("\"{}\" [label=\"{}\", shape=\"square\"];", method_name(method.clone()), method_name(method.clone())));
//     // }


//     let method = (Some(
//         "/Users/jimmyhmiller/.gem/ruby/3.2.0/gems/activerecord-6.0.4/lib/active_record/railties/controller_runtime.rb".to_string()),
//         Some("cleanup_view_runtime".to_string())
//     );
//     let location = all_methods.iter().find(|x| x.file == method.0 && x.method_name == method.1).unwrap();

//     // let method = (Some("/Users/jimmyhmiller/.gem/ruby/3.2.0/gems/activerecord-6.0.4/lib/active_record/result.rb".to_string()), Some("hash_rows".to_string()));
//     let mut method_records = record_by_method.get(location).unwrap().clone();
//     // let mut method_records = records.clone();


//     // let mut method_records_by_start_addr : HashMap<usize, Vec<Block>> = HashMap::new();


//     // for record in method_records.iter() {
//     //     let key = record.start_addr.unwrap();
//     //     let entry = method_records_by_start_addr.entry(key).or_insert_with(|| vec![]);
//     //     entry.push(record.clone());
//     // }

//     // let mut method_records = vec![];
//     // for records in method_records_by_start_addr.values() {
//     //     let mut records = records.clone();
//     //     records.sort_by_key(|x| x.epoch);
//     //     method_records.push(records.last().unwrap().clone());
//     // }


//     for record in method_records.iter() {
//         // make a graphviz node
//         let hash = calculate_hash(record);
//         nodes.push(format!("\"{}\" [label=\"{:?}\n{}\", shape=\"square\"];", hash, record.block_id, record.disasm));
//     }

//     // // // make graphviz edges
//     let mut edges: Vec<(String, String)> = vec![];

//     // edges.extend(method_to_method.iter()
//     //     .map(|(x, y)|
//     //         (method_name(x.clone()), method_name(y.clone()))));

//     for record in method_records.iter() {
//         for incoming in record.incoming.iter() {
//             if incoming.start_addr == Some(5100273920) {
//                 println!("Found it");
//             }
//             for other_record in method_records.iter() {
//                 for outgoing in other_record.outgoing.iter() {
//                     if incoming.start_addr == outgoing.start_addr {
//                         println!("NEW FIND!, {:?}", incoming.block_id);
//                         let hash = calculate_hash(record);
//                         let other_hash = calculate_hash(other_record);
//                         edges.push((other_hash, hash));
//                     }
//                 }
//             }
//         }
//     }

//     for record in method_records.iter() {

//         if let Some(end_addr) = record.end_addr {
//             if record_by_start_addr.contains_key(&end_addr) {
//                 println!("Found it {:?}", record.block_id);
//                 let next_record = record_by_start_addr.get(&end_addr).unwrap();
//                 let hash = calculate_hash(record);
//                 let next_hash = calculate_hash(next_record);
//                 edges.push((hash, next_hash));
//             }
//         }


//         for outgoing in record.outgoing.iter() {
//             let start_addr = outgoing.start_addr.unwrap();
//             if record_by_start_addr.contains_key(&start_addr) {
//                 println!("Found it {:?}", record.block_id);
//                 let next_record = record_by_start_addr.get(&start_addr).unwrap();
//                 let hash = calculate_hash(record);
//                 let next_hash = calculate_hash(next_record);
//                 edges.push((hash, next_hash));
//             }
//             for dst_addr in outgoing.dst_addrs.iter() {
//                 if let Some(dst_addr) = dst_addr {
//                     // println!("dst_addr: {:?}", dst_addr);
//                     if let Some(target) = record_by_start_addr.get(dst_addr) {
//                         println!("target: {:?}", target.block_id);
//                         let record_hash = calculate_hash(record);
//                         let target_hash = calculate_hash(target);

//                         edges.push((record_hash, target_hash));
//                     }
//                 }
//             }
//         }



//         for incoming in record.incoming.iter() {
//             let end_addr = incoming.end_addr.unwrap();
//             if record_by_start_addr.contains_key(&end_addr) {
//                 println!("Found it {:?}", record.block_id);
//                 let next_record = record_by_start_addr.get(&end_addr).unwrap();
//                 let hash = calculate_hash(record);
//                 let next_hash = calculate_hash(next_record);
//                 edges.push((hash, next_hash));
//             }
//             for dst_addr in incoming.dst_addrs.iter() {
//                 if let Some(dst_addr) = dst_addr {
//                     // println!("dst_addr: {:?}", dst_addr);
//                     if let Some(target) = record_by_start_addr.get(dst_addr) {
//                         println!("target: {:?}", target.block_id);
//                         let record_hash = calculate_hash(record);
//                         let target_hash = calculate_hash(target);

//                         edges.push((record_hash, target_hash));
//                     }
//                 }
//             }
//         }
//     }
//     // edges.sort();
//     // edges.dedup();
//     let edges = edges.iter()
//         // .filter(|(x, y)| x != y)
//         .map(|(x, y)| format!("\"{}\" -> \"{}\";", x, y))
//         .collect::<Vec<_>>();


//     let mut output = String::new();
//     output.push_str("digraph {\n");
//     output.push_str(&nodes.join("\n"));
//     output.push_str("\n");
//     output.push_str(&edges.join("\n"));
//     output.push_str("\n");
//     output.push_str("}\n");



//     // for entry in recordByMethod {
//     //     println!("{:?}: {}", entry.0, entry.1.len());
//     // }

//     // // write output to file named file.dot
//     let mut file = File::create("single_method.dot").unwrap();
//     file.write_all(output.as_bytes()).unwrap();


//     // serialize as json and write to file
//     let mut file = File::create("single_method.json").unwrap();
//     file.write_all(serde_json::to_string_pretty(&method_records).unwrap().as_bytes()).unwrap();


//    let methods_as_text = all_methods.iter()
//         .filter(|x| x.method_name.is_some())
//         .map(|x| format!("{}", x.method_name.as_ref().unwrap()))
//         .collect::<Vec<_>>()
//         .join("\n");
