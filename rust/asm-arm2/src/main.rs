use std::{fs, error::Error, str::from_utf8};

use roxmltree::{Node, Document};


#[derive(Debug, Clone)]
struct FileInfo {
    name: String,
    asm: Vec<String>,
    desc: String,
    regdiagram: Vec<String>,
}

#[derive(Debug, Clone)]
struct Field {
    name: String,
    shift: u32,
    bits: String,
    width: u32,
    required: bool,
}





fn main() -> Result<(), Box<dyn Error>> {
    let xml_file_path = "resources/onebigfile.xml";

    let xml_file_bytes = fs::read(xml_file_path)?;
    let xml_file_text = from_utf8(&xml_file_bytes)?;
    let xml = roxmltree::Document::parse(xml_file_text.clone())?;


    let file_names = xml
        .descendants()
        .filter(|x| x.has_tag_name("iforms"))
        .find(|x| {
            x.attribute("title")
                .unwrap_or("")
                .contains("Base Instructions")
        })
        .unwrap()
        .descendants()
        .filter(|x| x.has_tag_name("iform"))
        .filter_map(|x| x.attribute("iformfile"));

    let mut found_file_nodes = vec![];
    for file_name in file_names {
        let file_ndoe = xml
            .descendants()
            .find(|x| x.attribute("file") == Some(file_name));
        if let Some(file_node) = file_ndoe {
            found_file_nodes.push(file_node);
        }
    }

    let file_info : Vec<FileInfo> = found_file_nodes
        .iter()
        .flat_map(|x| {
            x.descendants()
                .filter(|x| x.has_tag_name("instructionsection"))
        })
        .map(|x| {
            let name = x
                .attribute("id")
                .unwrap_or("No file found")
                .to_ascii_lowercase();
            let asm = x
                .descendants()
                .filter(|x| x.has_tag_name("asmtemplate"))
                .map(|x| xml_file_text[x.range()].to_string())
                .collect();
            let desc = x
                .descendants()
                .find(|x| x.has_tag_name("desc"))
                .and_then(|x| x.descendants().find(|x| x.has_tag_name("brief")))
                .and_then(|x| x.descendants().find(|x| x.has_tag_name("para")))
                .map(|x| x.text().unwrap_or(""))
                .unwrap_or("")
                .to_string();
            let regdiagram = x
                .descendants()
                .find(|x| x.has_tag_name("regdiagram"))
                .map(|x| {
                    let boxes: Vec<Node> =
                        x.descendants().filter(|x| x.has_tag_name("box")).collect();
                    boxes
                })
                .unwrap_or_default()
                .iter()
                .map(|x| xml_file_text[x.range()].to_string())
                .collect();
            FileInfo {
                name,
                asm,
                desc,
                regdiagram,
            }
        })
        .collect();





    for file_info in file_info.iter() {

        let mut fields = vec![];
        let name = file_info.name.to_ascii_uppercase();

        let mut argument_comments = vec![];

        for asm in file_info.asm.iter() {
            let asm = Document::parse(asm).unwrap();
            let texts = asm
                .descendants()
                .filter(|x| x.has_tag_name("a"))
                .map(|x| x.text().unwrap_or("").to_string())
                .collect::<Vec<String>>();
            for text in texts {
                argument_comments.push(text);
            }
        }


        for regdiagram in file_info.regdiagram.iter() {
            let document = roxmltree::Document::parse(&regdiagram).unwrap();
            let root = document.root_element();
            let name = root.attribute("name").unwrap_or("");
            // let use_name = root.attribute("usename");

            let bits : Vec<String> = root.descendants()
                .filter(|child| !child.is_text())
                .filter_map(|child| {
                    let text = child.text().unwrap_or_default().trim();
                    if !text.is_empty() {
                        Some(text.replace("(", "").replace(")", ""))
                    } else {
                        None
                    }
                })
                .collect();

            let hibit = root.attribute("hibit").unwrap().parse::<u32>().unwrap() + 1;
            let width = root.attribute("width").unwrap_or("1").parse::<u32>().unwrap();
            let shift = hibit - width;


            fields.push(Field {
                name: name.to_string(),
                shift,
                bits: if bits.is_empty() {
                    "0".repeat(width as usize).to_string()
                } else {
                    bits.join("")
                },
                width,
                required: bits.is_empty(),
            })
        }
        println!("{:?}", name);
        println!("{:?}", file_info.desc);
        println!("{:?}", argument_comments);

        for field in fields {
            println!("{:?}", field);
        }
        println!("==================================================\n\n");
    }


    


    Ok(())
}
