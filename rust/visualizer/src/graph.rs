use std::collections::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;


use crate::Method;
use crate::block::Block;

fn calculate_hash<T: Hash>(t: &T) -> String {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish().to_string()
}


pub fn make_method_graph(all_records: &Vec<Block>, method: &Method) -> String {
    let method_records = &method.blocks;
    let mut nodes = Vec::new();

    let normal_node_color = "#5a8a5e";
    let exit_node_color = "#fd5e53";

    for record in method_records.iter() {
        // make a graphviz node

        if record.block_id.idx == 6 {
            println!("Got it!");
        }
        let color = if record.is_exit {
            exit_node_color
         } else {
            normal_node_color
        };
        nodes.push(format!("\"{}\" [label=\"{:?}\n{}\", shape=\"rectangle\", fontcolor=\"{}\", color=\"{}\", fontsize=\"20pt\", fontname=\"Ubuntu Mono\"];", record.id, record.block_id, record.disasm.replace("\n", "\\l"), color, color));
    }

    // // TODO: We almost certinaly have incoming nodes from elsewhere. We need to think about how we deal with that.
    // let mut record_by_start_addr : HashMap<usize, Block> = HashMap::new();
    // for record in method_records.iter() {
    //     let start_addr = record.start_addr.unwrap();
    //     if record_by_start_addr.contains_key(&start_addr) {
    //         let already_recorded = record_by_start_addr.get(&start_addr).unwrap();
    //         if already_recorded.epoch > record.epoch {
    //             continue;
    //         }
    //     }
    //     record_by_start_addr.insert(start_addr, record.clone());
    // }

    #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
    struct EdgeInfo {
        from: usize,
        to: usize,
        attributes: String,
    }


    // make graphviz edges
    let mut edges: Vec<EdgeInfo> = vec![];

    fn add_edge(edges: &mut Vec<EdgeInfo>, from: usize, to: usize, attributes: String) {
        edges.push(EdgeInfo {
            from,
            to,
            attributes,
        });

    }


    for record in method_records.iter() {
        for outgoing in record.outgoing.iter() {
            for block in outgoing.blocks.iter().flatten() {
                add_edge(&mut edges, record.id, *block, format!("label=\"{}\"", "outgoing"));
            }
        }
    }

    for record in method_records.iter() {
        for incoming in record.incoming.iter() {
            for block in incoming.blocks.iter().flatten() {
                add_edge(&mut edges, *block, record.id, format!("label=\"{}\"", "incoming"));
            }
        }
    }

    // for record in method_records.iter() {
    //     for incoming in record.incoming.iter() {
    //         for other_record in method_records.iter() {
    //             if record == other_record {
    //                 continue;
    //             }
    //             for outgoing in other_record.outgoing.iter() {
    //                 if incoming.start_addr == outgoing.start_addr {
    //                     add_edge(&mut edges, other_record.id, record.id, format!("label=\"{}\"", "fallthrough"));
    //                 }
    //             }
    //         }
    //     }
    // }

    for record in method_records.iter() {
        for other_record in method_records.iter() {
            if record == other_record {
                continue;
            }
            if record.end_addr == other_record.start_addr {
                add_edge(&mut edges, record.id, other_record.id, format!("label=\"{}\"", "fallthrough"));
            }

            if other_record.end_addr == record.start_addr {
                add_edge(&mut edges, other_record.id, record.id, format!("label=\"{}\"", "fallthrough"));
            }
        }
    }

    // for record in method_records.iter() {

    //     if let Some(end_addr) = record.end_addr {
    //         if record_by_start_addr.contains_key(&end_addr) {
    //             let next_record = record_by_start_addr.get(&end_addr).unwrap();
    //             add_edge(&mut edges, record.id, next_record.id, format!("label=\"{}\"", record.id.shape));
    //         }
    //     }

    //     for outgoing in record.outgoing.iter() {
    //         let start_addr = outgoing.start_addr.unwrap();
    //         if record_by_start_addr.contains_key(&start_addr) {
    //             let next_record = record_by_start_addr.get(&start_addr).unwrap();
    //             add_edge(&mut edges, record.id, next_record.id, format!("label=\"{}\"", record.id.shape));
    //         }
    //         for dst_addr in outgoing.dst_addrs.iter() {
    //             if let Some(dst_addr) = dst_addr {
    //                 if let Some(target) = record_by_start_addr.get(dst_addr) {

    //                     add_edge(&mut edges, record.id, target.id, format!("label=\"{}\"", record.id.shape));
    //                 }
    //             }
    //         }
    //     }



    //     for incoming in record.incoming.iter() {
    //         let end_addr = incoming.end_addr.unwrap();
    //         if record_by_start_addr.contains_key(&end_addr) {
    //             let next_record = record_by_start_addr.get(&end_addr).unwrap();
    //             add_edge(&mut edges, record.id, next_record.id, format!("label=\"{}\"", record.id.shape));
    //         }
    //         for dst_addr in incoming.dst_addrs.iter() {
    //             if let Some(dst_addr) = dst_addr {
    //                 // println!("dst_addr: {:?}", dst_addr);
    //                 if let Some(target) = record_by_start_addr.get(dst_addr) {
    //                     add_edge(&mut edges, target.id, record.id, format!("label=\"{}\"", target.id.shape));
    //                 }
    //             }
    //         }
    //     }
    // }
    edges.sort();
    edges.dedup();

    let edge_nodes = edges.iter().map(|e| e.from).chain(edges.iter().map(|e| e.to)).collect::<HashSet<usize>>();
    let node_set = method_records.iter().map(|r| r.id).collect::<HashSet<usize>>();
    let missing_nodes = node_set.difference(&edge_nodes).collect::<Vec<&usize>>();

    for record in all_records.iter().filter(|r| missing_nodes.contains(&&r.id)) {
        // make a graphviz node
        nodes.push(format!("\"{}\" [label=\"{:?}-{}\n{}\", shape=\"rectangle\", fontcolor=\"#5a8a5e\", color=\"#5a8a5e\", fontsize=\"20pt\", fontname=\"Ubuntu Mono\"];", record.id, record.block_id, record.id, record.disasm.replace("\n", "\\l")));
    }
    nodes.sort();
    nodes.dedup();


    let edges = edges.iter()
        .map(|EdgeInfo { from, to, attributes }| format!("\"{}\" -> \"{}\" [color=\"#5a8a5e\", fontcolor=\"#5a8a5e\", {}];", from, to, attributes))
        .collect::<Vec<_>>();


    let mut output = String::new();
    output.push_str("digraph {\n");
    output.push_str("bgcolor=\"#210522\"\n");
    output.push_str(&nodes.join("\n"));
    output.push_str("\n");
    output.push_str(&edges.join("\n"));
    output.push_str("\n");
    output.push_str("}\n");


    output

}


pub fn call_graphviz_command_line(graph: &str) -> Vec<u8> {
    use std::process::Command;

    let mut child = Command::new("dot")
        .arg("-Tpng")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to execute process");

    let stdin = child.stdin.as_mut().expect("failed to open stdin");
    stdin.write_all(graph.as_bytes()).expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    output.stdout
}


// TODO:
// Why do more exits show up in the blocks than in the graphs?
