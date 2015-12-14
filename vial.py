from kafka import SimpleProducer, KafkaClient, KafkaConsumer
import json
from itertools import imap, ifilter
from functools import wraps


class Vial(object):
    """docstring for Vial"""
    def __init__(self, name, host):
        super(Vial, self).__init__()
        self.name = name
        self.host = host
        self.listening = {}
        self.default_events()
        kafka = KafkaClient(self.host)
        self.producer = SimpleProducer(kafka)
        self.consumer = KafkaConsumer("need",
                         group_id=self.name,
                         bootstrap_servers=[self.host])

    def default_events(self):
        self.register_listener("apps", lambda m: self.name)
        self.register_listener("events", lambda m: self.listening.keys())

    def register_listener(self, type, fn):
        self.listening[type] = fn

    def handle_event(self, message):
        handler = self.listening[message["type"]]
        result = handler(message)
        self.supply(message, result)
    
    def supply(self, message, data):
        response = self.build_response(message, data)
        print(response)
        self.producer.send_messages("supply", response)

    def build_response(self, message, data):
        return json.dumps({
            "app": self.name,
            "request_id": message["request_id"],
            "data": data
        })

    def filter_event(self, event):
        print(set(self.listening.keys()))
        return (event.has_key("request_id") and 
               event.has_key("type") and 
               event["type"] in set(self.listening.keys()))
    
    def map_event(self, event):
       return json.loads(event.value)

    def process_events(self, events):
        events = imap(self.map_event, events)
        events = ifilter(self.filter_event, events)
        return events

    def listen(self):
        consumer = self.process_events(self.consumer)
        for event in consumer:
            self.handle_event(event)

    def event(self, name):
        def event_decorator(f):
            self.register_listener(name, f)
            return f
        return event_decorator

    def run(self):
        self.listen()




