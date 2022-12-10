use std::collections::{HashSet};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::mpsc::Receiver;


use crate::{Method, Style};
use crate::block::Block;

fn calculate_hash<T: Hash>(t: &T) -> String {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish().to_string()
}


pub fn make_method_graph(style: &Style, all_records: &Vec<Block>, method: &Method) -> String {
    let method_records = &method.blocks;
    let mut nodes = Vec::new();

    let normal_node_color = &style.primary_text_color.to_hex();
    let exit_node_color = &style.exit_text_color.to_hex();
    let outer_node_color = &style.outer_block_color.to_hex();

    for record in method_records.iter() {
        // make a graphviz node

        let color = if record.is_exit {
            exit_node_color
         } else {
            normal_node_color
        };
        nodes.push(format!("\"{}\" [label=\"{:?}\n{}\", shape=\"rectangle\", fontcolor=\"{}\", color=\"{}\", fontsize=\"20pt\", fontname=\"Ubuntu Mono\"];", format!("block-{}", record.id), record.block_id, record.disasm.replace('\n', "\\l"), color, color));
    }

    let mut all_branches = method_records.iter().flat_map(|r| r.incoming.iter()).collect::<Vec<_>>();
    all_branches.extend(method_records.iter().flat_map(|r| r.outgoing.iter()));


    all_branches.sort_by_key(|b| b.id);
    all_branches.dedup();

    for record in all_branches.iter() {
        let label = if record.disasm.is_empty() {
            "branch".to_string()
        } else {
            record.disasm.replace('\n', "\\l")
        };
        let color = normal_node_color;
        nodes.push(format!("\"{}\" [label=\"{:?}\n{}\", shape=\"rectangle\", fontcolor=\"{}\", color=\"{}\", fontsize=\"20pt\", fontname=\"Ubuntu Mono\"];", format!("branch-{}", record.id), record.id, label, color, color));
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
        from: String,
        to: String,
        attributes: String,
    }


    // make graphviz edges
    let mut edges: Vec<EdgeInfo> = vec![];

    fn add_edge(edges: &mut Vec<EdgeInfo>, from: String, to: String, attributes: String) {
        edges.push(EdgeInfo {
            from,
            to,
            attributes,
        });

    }


    for record in method_records.iter() {
        for outgoing in record.outgoing.iter() {
            add_edge(&mut edges, format!("block-{}", record.id), format!("branch-{}", outgoing.id), format!("label=\"{}\"", "outgoing"));
            for target in outgoing.targets.iter().flatten() {
                if let Some(block) = target.block {
                    add_edge(&mut edges, format!("branch-{}", outgoing.id), format!("block-{}", block), format!("label=\"{}\"", "outgoing"));
                }
            }
        }
    }

    for record in method_records.iter() {
        for incoming in record.incoming.iter() {
            add_edge(&mut edges, format!("branch-{}", incoming.id), format!("block-{}", record.id), format!("label=\"{}\"", "incoming"));
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

    // for record in method_records.iter() {
    //     for other_record in method_records.iter() {
    //         if record == other_record {
    //             continue;
    //         }
    //         if record.end_addr == other_record.start_addr {
    //             add_edge(&mut edges, record.id, other_record.id, format!("label=\"{}\"", "fallthrough"));
    //         }

    //         if other_record.end_addr == record.start_addr {
    //             add_edge(&mut edges, other_record.id, record.id, format!("label=\"{}\"", "fallthrough"));
    //         }
    //     }
    // }

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


    let edge_nodes = edges.iter().map(|e| e.from.clone()).chain(edges.iter().map(|e| e.to.clone())).filter(|e| e.starts_with("block")).collect::<HashSet<String>>();
    let node_set = method_records.iter().map(|r| format!("block-{}", r.id),).collect::<HashSet<String>>();
    let missing_nodes = edge_nodes.difference(&node_set).collect::<Vec<&String>>();

    for record in all_records.iter().filter(|r| missing_nodes.contains(&&format!("block-{}", r.id),)) {
        // make a graphviz node
        nodes.push(format!("\"{}\" [label=\"{:?}\n{}\", shape=\"rectangle\", fontcolor=\"{}\", color=\"{}\", fontsize=\"20pt\", fontname=\"Ubuntu Mono\"];", format!("block-{}", record.id), record.block_id, record.disasm.replace('\n', "\\l"), outer_node_color, outer_node_color));
    }
    nodes.sort();
    nodes.dedup();


    let edges = edges.iter()
        .map(|EdgeInfo { from, to, attributes }| format!("\"{}\" -> \"{}\" [color=\"{}\", fontcolor=\"{}\", {}];", from, to, normal_node_color, normal_node_color, attributes))
        .collect::<Vec<_>>();


    let mut output = String::new();
    output.push_str("digraph {\n");
    output.push_str("bgcolor=\"#210522\"\n");
    output.push_str(&nodes.join("\n"));
    output.push('\n');
    output.push_str(&edges.join("\n"));
    output.push('\n');
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

pub struct Promise<T> {
    value: Option<T>,
    receiver: Option<Receiver<T>>,
}

impl<T> Promise<T> {
    pub fn new(receiver: Receiver<T>) -> Self {
        Promise {
            value: None,
            receiver: Some(receiver),
        }
    }

    pub fn ready(&mut self) -> bool {
        if let Some(ref receiver) = self.receiver {
            if let Ok(value) = receiver.try_recv() {
                self.value = Some(value);
                self.receiver = None;
                return true;
            }
        }
        false
    }

    pub fn get(&mut self) -> Option<&T> {
        self.value.as_ref()
    }
}


pub fn call_graphviz_in_new_thread(graph: &str) -> Promise<Vec<u8>> {

    let (sender, receiver) = std::sync::mpsc::channel::<Vec<u8>>();
    let promise = Promise::new(receiver);

    let graph = graph.to_string();
    std::thread::spawn(move || {
        let output = call_graphviz_command_line(&graph);
        sender.send(output).unwrap();
    });

    promise

}
