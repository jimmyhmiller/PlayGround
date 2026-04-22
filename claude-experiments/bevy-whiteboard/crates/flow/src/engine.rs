use std::cmp::Reverse;
use std::collections::VecDeque;

use crate::event::Event;
use crate::expr::{EvalCtx, Expr};
use crate::rule::{Effect, EmitTo, Rule, When};
use crate::scenario::Action;
use crate::sim::{EdgeId, Node, NodeId, Packet, Scheduled, Sim, Time};
use crate::template::EdgeEnd;
use crate::value::{Bindings, Value, match_pattern};

impl Sim {
    /// Run the simulation forward until either `deadline_ns` is reached
    /// or the world becomes quiescent (no fireable rules, no in-flight
    /// packets). Returns the number of rule firings that occurred.
    pub fn run_until(&mut self, deadline_ns: Time) -> usize {
        let mut firings = 0usize;
        loop {
            // 1. Fire every rule that can fire at the current instant.
            let mut instant_steps = 0usize;
            while self.try_fire_one() {
                firings += 1;
                instant_steps += 1;
                if instant_steps > self.max_steps_per_instant {
                    panic!(
                        "Sim: exceeded {} rule firings at instant {} ns \
                         (runaway rules? cycle with zero-latency edges?)",
                        self.max_steps_per_instant, self.now_ns
                    );
                }
            }

            // 2. Nothing more to fire. Next event is the earliest of:
            //    (a) a scheduled scenario action
            //    (b) an in-flight delivery
            //    (c) the deadline.
            let next_delivery = self.in_flight.peek().map(|r| r.0.arrives_at_ns);
            let next_action = self.pending_actions.peek().map(|r| r.0.at_ns);

            let next_event = match (next_delivery, next_action) {
                (Some(d), Some(a)) => Some(d.min(a)),
                (Some(d), None) => Some(d),
                (None, Some(a)) => Some(a),
                (None, None) => None,
            };

            let Some(t) = next_event else {
                if self.now_ns < deadline_ns {
                    self.log.push(Event::ClockAdvanced {
                        from_ns: self.now_ns,
                        to_ns: deadline_ns,
                    });
                    self.now_ns = deadline_ns;
                }
                return firings;
            };

            if t > deadline_ns {
                if self.now_ns < deadline_ns {
                    self.log.push(Event::ClockAdvanced {
                        from_ns: self.now_ns,
                        to_ns: deadline_ns,
                    });
                    self.now_ns = deadline_ns;
                }
                return firings;
            }

            if t > self.now_ns {
                self.log.push(Event::ClockAdvanced {
                    from_ns: self.now_ns,
                    to_ns: t,
                });
                self.now_ns = t;
            }

            // 3a. Apply scheduled scenario actions for this instant, in order.
            while let Some(head) = self.pending_actions.peek() {
                if head.0.at_ns != self.now_ns { break; }
                let p = self.pending_actions.pop().unwrap().0;
                self.apply_action(p.action);
            }

            // 3b. Deliver all packets scheduled for this instant.
            while let Some(head) = self.in_flight.peek() {
                if head.0.arrives_at_ns != self.now_ns { break; }
                let sched = self.in_flight.pop().unwrap().0;
                let mut pkt = sched.packet;
                pkt.from_edge = Some(sched.edge);
                if !self.nodes.contains_key(&sched.deliver_to) {
                    continue;  // despawned while in flight
                }
                // Resolve compound → inner leaf via in_ports.
                let port = self.edges.get(&sched.edge).and_then(|e| e.to_port.clone());
                let leaf = self.resolve_in_port_chain(sched.deliver_to, port);
                let Some(leaf) = leaf else { continue; };
                self.log.push(Event::PacketDelivered {
                    packet: pkt.id,
                    to: leaf,
                    at_ns: self.now_ns,
                });
                if let Some(node) = self.nodes.get_mut(&leaf) {
                    node.inbox.push_back(pkt);
                }
            }
        }
    }

    /// Attempt to fire exactly one rule. Returns true if one fired.
    ///
    /// Deterministic firing order: iterate nodes by id, rules by
    /// definition order, first fireable wins.
    fn try_fire_one(&mut self) -> bool {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for nid in node_ids {
            let rule_count = self.nodes.get(&nid).map(|n| n.rules.len()).unwrap_or(0);
            for ri in 0..rule_count {
                if self.try_fire(nid, ri) {
                    return true;
                }
            }
        }
        false
    }

    fn try_fire(&mut self, nid: NodeId, rule_idx: usize) -> bool {
        let rule = {
            let node = self.nodes.get(&nid).expect("try_fire: node gone");
            node.rules.get(rule_idx).cloned()
        };
        let Some(rule) = rule else { return false; };
        let Some(fire) = self.match_rule(nid, &rule) else { return false; };
        self.execute_fire(nid, &rule, fire);
        true
    }

    fn match_rule(&mut self, nid: NodeId, rule: &Rule) -> Option<MatchResult> {
        let (inbox_len, slot_snapshot) = {
            let node = self.nodes.get(&nid)?;
            (node.inbox.len(), node.slots.clone())
        };

        // Find the (at most one) Input pattern.
        let input_when: Option<When> = rule.when.iter()
            .find(|w| matches!(w, When::Input { .. }))
            .cloned();

        // If there's an Input pattern, try each inbox entry until one matches.
        // Otherwise try just once with no packet consumed.
        let candidates: Vec<Option<usize>> = if input_when.is_some() {
            (0..inbox_len).map(Some).collect()
        } else {
            vec![None]
        };

        for pkt_idx in candidates {
            let mut bindings = Bindings::new();

            // Match the Input pattern if present.
            if let (Some(w), Some(idx)) = (input_when.as_ref(), pkt_idx) {
                let (pattern, from) = match w {
                    When::Input { pattern, from } => (pattern.clone(), from.clone()),
                    _ => unreachable!(),
                };
                let pkt = &self.nodes[&nid].inbox[idx];
                if let Some(from_name) = from {
                    let ok = pkt.from_edge
                        .and_then(|eid| self.edges.get(&eid))
                        .and_then(|e| self.nodes.get(&e.from))
                        .map(|n| n.name == from_name)
                        .unwrap_or(false);
                    if !ok { continue; }
                }
                if !match_pattern(&pkt.payload, &pattern, &mut bindings) {
                    continue;
                }
            }

            // Match all SlotMatch patterns.
            let mut slot_ok = true;
            for w in &rule.when {
                if let When::SlotMatch { slot, pattern } = w {
                    let Some(sv) = slot_snapshot.get(slot) else { slot_ok = false; break; };
                    if !match_pattern(sv, pattern, &mut bindings) {
                        slot_ok = false;
                        break;
                    }
                }
            }
            if !slot_ok { continue; }

            // Evaluate guard, if present. IMPORTANT: guard evaluation can consume
            // RNG (e.g. Bernoulli). To preserve determinism and avoid "phantom"
            // RNG draws from rules that fail to match, we only evaluate the
            // guard on rules where when-patterns already matched (this branch).
            if let Some(ref g) = rule.guard {
                let mut pstack: Vec<String> = Vec::new();
                // Field-disjoint borrows: rng is &mut, everything else is &.
                let nodes = &self.nodes;
                let edges = &self.edges;
                let params = &self.params;
                let now_ns = self.now_ns;
                let rng = &mut self.rng;
                let mut ctx = EvalCtx {
                    bindings: &bindings,
                    slots: &slot_snapshot,
                    now_ns,
                    rng,
                    current_node: Some(nid),
                    params,
                    param_stack: &mut pstack,
                    nodes,
                    edges,
                };
                let v = g.eval(&mut ctx);
                if !v.as_bool().expect("guard must evaluate to Bool") {
                    continue;
                }
            }

            return Some(MatchResult { bindings, consumed_pkt_idx: pkt_idx });
        }

        None
    }

    fn execute_fire(&mut self, nid: NodeId, rule: &Rule, fire: MatchResult) {
        let MatchResult { bindings, consumed_pkt_idx } = fire;
        let now = self.now_ns;

        self.log.push(Event::RuleFired {
            node: nid,
            rule: rule.name.clone(),
            at_ns: now,
        });

        let consumed_pkt: Option<Packet> = if let Some(idx) = consumed_pkt_idx {
            let node = self.nodes.get_mut(&nid).unwrap();
            let pkt = node.inbox.remove(idx).expect("consume: index valid");
            self.log.push(Event::PacketConsumed {
                packet: pkt.id,
                by: nid,
                rule: rule.name.clone(),
                at_ns: now,
            });
            Some(pkt)
        } else {
            None
        };

        let mut bindings = bindings;
        for eff in &rule.effects {
            self.execute_effect(nid, &rule.name, eff, &mut bindings, consumed_pkt.as_ref());
        }
    }

    fn execute_effect(
        &mut self,
        nid: NodeId,
        rule_name: &str,
        eff: &Effect,
        bindings: &mut Bindings,
        consumed_pkt: Option<&Packet>,
    ) {
        let now = self.now_ns;

        match eff {
            Effect::SetSlot { slot, value } => {
                let v = self.eval_at_node(nid, bindings, value);
                self.write_slot(nid, slot, v, now);
            }
            Effect::SamplesPush { slot, value } => {
                let v = self.eval_at_node(nid, bindings, value);
                let node = self.nodes.get_mut(&nid).unwrap();
                let entry = node.slots.get_mut(slot)
                    .unwrap_or_else(|| panic!("SamplesPush: slot `{}` missing", slot));
                match entry {
                    Value::Samples(s) => s.push(v),
                    other => panic!("SamplesPush: slot `{}` is not Samples: {:?}", slot, other),
                }
            }
            Effect::SamplesPopOldestInto { slot, into_var } => {
                let node = self.nodes.get_mut(&nid).unwrap();
                let entry = node.slots.get_mut(slot)
                    .unwrap_or_else(|| panic!("SamplesPopOldestInto: slot `{}` missing", slot));
                let v = match entry {
                    Value::Samples(s) => s.pop_oldest()
                        .unwrap_or_else(|| panic!("SamplesPopOldestInto: slot `{}` empty", slot)),
                    other => panic!("SamplesPopOldestInto: slot `{}` not Samples: {:?}", slot, other),
                };
                bindings.insert(into_var.clone(), v);
            }
            Effect::SamplesDropOldest { slot, n } => {
                let n_val = self.eval_at_node(nid, bindings, n).as_int()
                    .expect("SamplesDropOldest: n must be Int");
                let node = self.nodes.get_mut(&nid).unwrap();
                let entry = node.slots.get_mut(slot)
                    .unwrap_or_else(|| panic!("SamplesDropOldest: slot `{}` missing", slot));
                match entry {
                    Value::Samples(s) => {
                        for _ in 0..n_val.max(0) {
                            if s.pop_oldest().is_none() { break; }
                        }
                    }
                    other => panic!("SamplesDropOldest: slot `{}` not Samples: {:?}", slot, other),
                }
            }
            Effect::Emit { payload, to } => {
                let v = self.eval_at_node(nid, bindings, payload);
                self.schedule_emit(nid, to.clone(), v, Some(nid), bindings);
            }
            Effect::EmitToEach { payload, targets } => {
                let targets_v = self.eval_at_node(nid, bindings, targets);
                let items = match targets_v {
                    Value::List(v) => v,
                    other => panic!(
                        "EmitToEach: targets must be a List of NodeRef, got {:?}",
                        other
                    ),
                };
                let payload_v = self.eval_at_node(nid, bindings, payload);
                for t in items {
                    let nref = match t {
                        Value::NodeRef(id) => id,
                        other => panic!(
                            "EmitToEach: each target must be NodeRef, got {:?}",
                            other
                        ),
                    };
                    // Routes through the same engine path as
                    // ToTargetExpr — silent-drops when no outbound
                    // edge exists from `nid` to `nref`.
                    self.schedule_emit(
                        nid,
                        crate::rule::EmitTo::ToTargetExpr(
                            crate::expr::Expr::Lit(Value::NodeRef(nref))
                        ),
                        payload_v.clone(),
                        Some(nid),
                        bindings,
                    );
                }
            }
            Effect::Respond { payload } => {
                let Some(pkt) = consumed_pkt else {
                    panic!("Respond effect requires a consumed inbound packet (rule `{}`)", rule_name);
                };
                let Some(reply_to) = pkt.reply_to else {
                    // Nowhere to reply — silently skip. (Modeling choice: you could panic.)
                    return;
                };
                // The reply travels on the outbound edge from this node (the responder)
                // to the requester. That edge is where the response latency expression
                // lives — NOT the inbound edge, which is evaluated in the requester's
                // slot context and would panic here.
                // Silent-drop when no reverse edge exists. Interactive
                // graphs often have a request edge but not the matching
                // response edge — panicking here cascades through every
                // subsequent rule firing. Same contract as unwired
                // `DefaultOut` / stale `ToTargetExpr`.
                let Some(reverse_edge) = self
                    .outbound(nid)
                    .into_iter()
                    .find(|eid| self.edges[eid].to == reply_to)
                else {
                    return;
                };
                let edge = self.edges[&reverse_edge].clone();
                let v = self.eval_at_node(nid, bindings, payload);
                let latency = self.eval_latency_expr(&edge.latency_ns, nid, bindings, &v);
                let arrives_at = self.now_ns.saturating_add(latency);
                let pid = self.next_packet_id();
                self.log.push(Event::PacketEmitted {
                    packet: pid,
                    from: nid,
                    to: reply_to,
                    at_ns: self.now_ns,
                    arrives_at_ns: arrives_at,
                    payload: v.clone(),
                });
                self.in_flight.push(Reverse(Scheduled {
                    arrives_at_ns: arrives_at,
                    packet: Packet {
                        id: pid,
                        payload: v,
                        from_edge: None,
                        reply_to: Some(nid),
                        emitted_at_ns: self.now_ns,
                    },
                    edge: reverse_edge,
                    deliver_to: reply_to,
                }));
            }
            Effect::RecordMetric { name, value } => {
                let v = self.eval_at_node(nid, bindings, value);
                self.log.push(Event::MetricRecorded {
                    node: nid,
                    name: name.clone(),
                    value: v,
                    at_ns: now,
                });
            }
            Effect::Spawn { template, into_var } => {
                let new_id = self.spawn_from_template(template, Some(nid));
                if let Some(var) = into_var {
                    bindings.insert(var.clone(), Value::NodeRef(new_id));
                }
            }
            Effect::Despawn { node } => {
                let v = self.eval_at_node(nid, bindings, node);
                let target = match v {
                    Value::NodeRef(id) => id,
                    other => panic!("Despawn: expected NodeRef, got {:?}", other),
                };
                self.despawn_node(target);
            }
        }
    }

    /// Walk up from `node` through its `parent` chain to find the
    /// nearest compound whose `out_ports[port_name]` equals the
    /// starting descendant. This is how an inner leaf's
    /// `ToOutPort("name")` is resolved to a specific compound
    /// boundary.
    fn find_enclosing_compound_for_out_port(&self, start: NodeId, port_name: &str) -> Option<NodeId> {
        let mut cur = self.nodes.get(&start)?.parent;
        let seeker = start;
        while let Some(pid) = cur {
            if let Some(p_node) = self.nodes.get(&pid) {
                if let Some(body) = &p_node.compound {
                    if body.out_ports.get(port_name) == Some(&seeker) {
                        return Some(pid);
                    }
                    // Port exists but isn't mapped to us? Keep walking
                    // upward — outer compounds may also have a same-named
                    // port and we might be their port via transitive
                    // inner-node identity.
                }
                cur = p_node.parent;
            } else {
                break;
            }
        }
        None
    }

    /// Walk a compound → inner leaf chain using the given port name at
    /// each level (same name re-used for nested compounds). Returns
    /// `None` if a port is missing or a compound has no inner node
    /// mapped for this port.
    fn resolve_in_port_chain(&self, start: NodeId, port: Option<String>) -> Option<NodeId> {
        let mut cur = start;
        loop {
            let node = self.nodes.get(&cur)?;
            match &node.compound {
                None => return Some(cur),
                Some(body) => {
                    let port_name = port.as_ref()?;
                    let inner = body.in_ports.get(port_name).copied()?;
                    cur = inner;
                }
            }
        }
    }

    /// Apply one scripted scenario action at the current clock.
    fn apply_action(&mut self, action: Action) {
        match action {
            Action::Inject { node, payload, reply_to } => {
                self.inject(node, payload, reply_to);
            }
            Action::SetSlot { node, slot, value } => {
                let now = self.now_ns;
                if let Some(n) = self.nodes.get_mut(&node) {
                    n.slots.insert(slot.clone(), value.clone());
                    self.log.push(Event::SlotWritten {
                        node, slot, value, at_ns: now,
                    });
                }
            }
            Action::SetEdgeLatency { edge, latency } => {
                if let Some(e) = self.edges.get_mut(&edge) {
                    e.latency_ns = latency;
                }
            }
            Action::KillNode { node } => {
                self.despawn_node(node);
            }
            Action::SetParam { name, value } => {
                self.params.insert(name, value);
            }
        }
    }

    /// Instantiate a template. Returns the new node's id.
    ///
    /// Template edges with `EdgeEnd::Parent` require a parent to be
    /// supplied. Panics otherwise.
    pub fn spawn_from_template(&mut self, template_name: &str, parent: Option<NodeId>) -> NodeId {
        let t = self.templates.get(template_name).cloned()
            .unwrap_or_else(|| panic!("spawn: no template named `{}`", template_name));

        self.next_instance_seq += 1;
        let seq = self.next_instance_seq;
        let instance_name = format!("{}_{}", t.node_name_prefix, seq);

        let new_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node {
            id: new_id,
            name: instance_name.clone(),
            slots: t.slots.clone(),
            rules: t.rules.clone(),
            inbox: VecDeque::new(),
            parent,
            compound: None,
        };
        self.nodes.insert(new_id, node);
        self.log.push(Event::NodeSpawned {
            node: new_id,
            template: t.name.clone(),
            parent,
            at_ns: self.now_ns,
        });

        // Materialize template edges.
        for spec in &t.edges {
            let resolve = |end: EdgeEnd| match end {
                EdgeEnd::ThisInstance => new_id,
                EdgeEnd::Parent => parent.unwrap_or_else(|| panic!(
                    "spawn: template `{}` has an edge with EdgeEnd::Parent \
                     but no parent was supplied",
                    template_name
                )),
            };
            let from = resolve(spec.from);
            let to = resolve(spec.to);
            self.add_edge(from, to, spec.latency.clone());
        }

        new_id
    }

    fn schedule_emit(
        &mut self,
        from_nid: NodeId,
        to: EmitTo,
        payload: Value,
        reply_to: Option<NodeId>,
        bindings: &Bindings,
    ) {
        // Resolve outbound edge.
        let outs = self.outbound(from_nid);
        let target_edge_id = match to {
            EmitTo::DefaultOut => {
                // Prefer the first NON-self outbound. A node's self-loop is
                // usually a timing mechanism, not its intended downstream.
                // Fall back to first outbound if only self-loops exist. If
                // the node has NO outbound at all, silently drop (matches
                // "unwired output" semantics; common during interactive
                // authoring).
                let pick = outs.iter().find(|&&eid| self.edges[&eid].to != from_nid)
                    .or_else(|| outs.first())
                    .copied();
                match pick {
                    Some(eid) => eid,
                    None => return,
                }
            },
            EmitTo::ToTarget(name) => outs.into_iter().find(|eid| {
                let to_id = self.edges[eid].to;
                self.nodes.get(&to_id).map(|n| n.name == name).unwrap_or(false)
            }).unwrap_or_else(|| {
                panic!("Emit::ToTarget(`{}`): no outbound edge with that target", name)
            }),
            EmitTo::ToTargetExpr(e) => {
                let v = self.eval_at_node(from_nid, bindings, &e);
                let target_id = match v {
                    Value::NodeRef(id) => id,
                    Value::Str(name) => self.node_by_name(&name).unwrap_or_else(|| {
                        panic!("EmitTo::ToTargetExpr: no node named `{}`", name)
                    }),
                    // Nil is the "no match" signal from expressions like
                    // `OutNeighborWithSlot` that may resolve to nothing.
                    // Silently drop the emit — same contract as unwired
                    // `DefaultOut`.
                    Value::Nil => return,
                    other => panic!(
                        "EmitTo::ToTargetExpr must yield Str or NodeRef, got {:?}",
                        other
                    ),
                };
                // Silent-drop when no outbound edge reaches the resolved
                // target. Interactive graphs churn: a user deletes the
                // node that was a node's current reply target, and stale
                // NodeRefs sit in slots like `pending_pull` until the
                // next write. Panicking here would cascade through every
                // subsequent rule firing; dropping matches the "unwired
                // output" contract DefaultOut already uses.
                match outs.into_iter().find(|eid| self.edges[eid].to == target_id) {
                    Some(eid) => eid,
                    None => return,
                }
            }
            EmitTo::ToOutPort(port_name) => {
                // Find the nearest ancestor compound that has this out-port
                // mapped to `from_nid`. Fan onto ALL outgoing edges whose
                // from=that compound and from_port=port_name. (Multiple
                // edges = broadcast.) Return early without setting a
                // single target edge — handled below.
                let compound_nid = self.find_enclosing_compound_for_out_port(from_nid, &port_name)
                    .unwrap_or_else(|| panic!(
                        "ToOutPort(`{}`): node {:?} has no enclosing compound with that out-port mapping it",
                        port_name, from_nid
                    ));
                let edge_ids: Vec<EdgeId> = self.edges.values()
                    .filter(|e| e.from == compound_nid && e.from_port.as_deref() == Some(port_name.as_str()))
                    .map(|e| e.id)
                    .collect();
                if edge_ids.is_empty() {
                    // Port is mapped but nothing is wired to it — silently drop.
                    // (Could alternatively panic; dropping matches "unwired output" semantics.)
                    return;
                }
                for eid in edge_ids {
                    let edge = self.edges[&eid].clone();
                    let latency = self.eval_latency_expr(&edge.latency_ns, compound_nid, bindings, &payload);
                    let arrives_at = self.now_ns.saturating_add(latency);
                    let pid = self.next_packet_id();
                    self.log.push(Event::PacketEmitted {
                        packet: pid,
                        from: compound_nid,
                        to: edge.to,
                        at_ns: self.now_ns,
                        arrives_at_ns: arrives_at,
                        payload: payload.clone(),
                    });
                    self.in_flight.push(Reverse(Scheduled {
                        arrives_at_ns: arrives_at,
                        packet: Packet {
                            id: pid,
                            payload: payload.clone(),
                            from_edge: None,
                            reply_to,
                            emitted_at_ns: self.now_ns,
                        },
                        edge: eid,
                        deliver_to: edge.to,
                    }));
                }
                return;
            }
        };
        let edge = self.edges[&target_edge_id].clone();
        let latency = self.eval_latency_expr(&edge.latency_ns, from_nid, &Bindings::new(), &payload);
        let arrives_at = self.now_ns.saturating_add(latency);
        let pid = self.next_packet_id();
        self.log.push(Event::PacketEmitted {
            packet: pid,
            from: from_nid,
            to: edge.to,
            at_ns: self.now_ns,
            arrives_at_ns: arrives_at,
            payload: payload.clone(),
        });
        self.in_flight.push(Reverse(Scheduled {
            arrives_at_ns: arrives_at,
            packet: Packet {
                id: pid,
                payload,
                from_edge: None,
                reply_to,
                emitted_at_ns: self.now_ns,
            },
            edge: target_edge_id,
            deliver_to: edge.to,
        }));
    }

    fn eval_at_node(&mut self, nid: NodeId, bindings: &Bindings, e: &Expr) -> Value {
        let slots = self.nodes.get(&nid).unwrap().slots.clone();
        let mut pstack: Vec<String> = Vec::new();
        let nodes = &self.nodes;
        let edges = &self.edges;
        let params = &self.params;
        let now_ns = self.now_ns;
        let rng = &mut self.rng;
        let mut ctx = EvalCtx {
            bindings,
            slots: &slots,
            now_ns,
            rng,
            current_node: Some(nid),
            params,
            param_stack: &mut pstack,
            nodes,
            edges,
        };
        e.eval(&mut ctx)
    }

    fn eval_latency_expr(
        &mut self,
        latency_ns: &Expr,
        src_nid: NodeId,
        bindings: &Bindings,
        packet_payload: &Value,
    ) -> Time {
        // Edge latency is evaluated with: source node's slots + `packet` = payload.
        let mut b = bindings.clone();
        b.insert("packet".to_string(), packet_payload.clone());
        let slots = self.nodes.get(&src_nid).unwrap().slots.clone();
        let mut pstack: Vec<String> = Vec::new();
        let nodes = &self.nodes;
        let edges = &self.edges;
        let params = &self.params;
        let now_ns = self.now_ns;
        let rng = &mut self.rng;
        let mut ctx = EvalCtx {
            bindings: &b,
            slots: &slots,
            now_ns,
            rng,
            current_node: Some(src_nid),
            params,
            param_stack: &mut pstack,
            nodes,
            edges,
        };
        let v = latency_ns.eval(&mut ctx);
        let n = v.as_int().unwrap_or_else(|| panic!("edge latency must be Int, got {:?}", v));
        assert!(n >= 0, "edge latency was negative: {}", n);
        n as Time
    }

    fn write_slot(&mut self, nid: NodeId, slot: &str, value: Value, at_ns: Time) {
        let node = self.nodes.get_mut(&nid).expect("write_slot: node gone");
        if let Some(existing) = node.slots.get(slot) {
            match (existing, &value) {
                (Value::Samples(_), Value::Samples(_)) => {}
                (Value::Samples(_), other) => panic!(
                    "SetSlot: slot `{}` is Samples but tried to write {:?}", slot, other
                ),
                _ => {}
            }
        }
        node.slots.insert(slot.to_string(), value.clone());
        self.log.push(Event::SlotWritten {
            node: nid,
            slot: slot.to_string(),
            value,
            at_ns,
        });
    }
}

struct MatchResult {
    bindings: Bindings,
    consumed_pkt_idx: Option<usize>,
}
