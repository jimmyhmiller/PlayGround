use simple_websockets::{Event, Responder};
use std::{collections::HashMap, sync::mpsc::{Receiver, Sender}};

use crate::{Message, Serialize};

pub fn start_websocket_thread() -> Result<(Sender<Message>, Receiver<()>), Box<dyn std::error::Error>> {
    
    let (sender, receiver) = std::sync::mpsc::channel::<Message>();
    let (output_sender, output_receiver) = std::sync::mpsc::channel::<()>();

    let _thread = std::thread::spawn(move || handler(receiver, output_sender));

    Ok((sender, output_receiver))
}
    
fn handler(receiver: Receiver<Message>, output_sender: Sender<()>) {
    let mut any_client_has_connected = false;
    // attempt to listen for WebSockets on port 3030
    let event_hub = match simple_websockets::launch(3030) {
        Ok(hub) => hub,
        Err(e) => {
            eprintln!("Failed to listen on websocket port 3030: {:?}", e);
            return; // exit thread gracefully
        }
    };
    // map between client ids and the client's `Responder`:
    let mut clients: HashMap<u64, Responder> = HashMap::new();
    
    
    loop {
    
        match event_hub.next_event() {
            Some(Event::Connect(client_id, responder)) => {
                if !any_client_has_connected {
                    output_sender.send(()).unwrap();
                    any_client_has_connected = true;
                }
                println!("A client connected with id #{}", client_id);
                // add their Responder to our `clients` map:
                clients.insert(client_id, responder);
            },
            Some(Event::Disconnect(client_id)) => {
                println!("Client #{} disconnected.", client_id);
                // remove the disconnected client from the clients map:
                clients.remove(&client_id);
            },
            Some(Event::Message(client_id, message)) => {
                println!("Received a message from client #{}: {:?}", client_id, message);
                // retrieve this client's `Responder`:
                let responder = clients.get(&client_id).unwrap();
                // echo the message back:
                responder.send(message);
            },
            None => {
                // no new events
            }
        }
    
        // check for new messages from the main thread:
        while let Ok(message) = receiver.try_recv() {
            // broadcast the message to all connected clients:
            for responder in clients.values() {
                println!("{:?}", message);
                responder.send(simple_websockets::Message::Binary(message.to_binary()));
            }
        }
    
    }
}