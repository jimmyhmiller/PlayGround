use midly::{Smf, TrackEventKind, MidiMessage};
use std::collections::BTreeSet;

/// Parse a MIDI file into a sequence of events with absolute tick timing.
#[derive(Debug, Clone)]
pub struct MidiEvent {
    pub tick: u64,
    pub kind: MidiEventKind,
}

#[derive(Debug, Clone)]
pub enum MidiEventKind {
    NoteOn { pitch: u8, velocity: u8 },
    NoteOff { pitch: u8 },
}

pub fn parse_midi_file(path: &str) -> Vec<MidiEvent> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
    let smf = Smf::parse(&data).unwrap();

    let mut events = Vec::new();

    for track in &smf.tracks {
        let mut tick: u64 = 0;
        for event in track {
            tick += event.delta.as_int() as u64;
            match event.kind {
                TrackEventKind::Midi { message, .. } => match message {
                    MidiMessage::NoteOn { key, vel } => {
                        if vel.as_int() == 0 {
                            events.push(MidiEvent {
                                tick,
                                kind: MidiEventKind::NoteOff { pitch: key.as_int() },
                            });
                        } else {
                            events.push(MidiEvent {
                                tick,
                                kind: MidiEventKind::NoteOn {
                                    pitch: key.as_int(),
                                    velocity: vel.as_int(),
                                },
                            });
                        }
                    }
                    MidiMessage::NoteOff { key, .. } => {
                        events.push(MidiEvent {
                            tick,
                            kind: MidiEventKind::NoteOff { pitch: key.as_int() },
                        });
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // Sort by tick (stable sort preserves order within same tick)
    events.sort_by_key(|e| e.tick);
    events
}

/// Token vocabulary for MIDI event sequences.
///
/// Layout:
///   0                = PAD
///   1..=n_pitches    = NOTE_ON for each pitch (mapped from compact pitch set)
///   n_pitches+1..=2*n_pitches = NOTE_OFF for each pitch
///   2*n_pitches+1..  = TIME_SHIFT(1), TIME_SHIFT(2), ..., TIME_SHIFT(max_shift)
///   last             = (total vocab size)
pub struct Vocab {
    /// Sorted unique pitches found in the piece
    pub pitches: Vec<u8>,
    pub max_time_shift: usize,
}

impl Vocab {
    pub fn size(&self) -> usize {
        1 + self.pitches.len() * 2 + self.max_time_shift
    }

    pub fn pad(&self) -> usize {
        0
    }

    pub fn note_on(&self, pitch: u8) -> usize {
        let idx = self.pitches.iter().position(|&p| p == pitch)
            .unwrap_or_else(|| panic!("pitch {pitch} not in vocab"));
        1 + idx
    }

    pub fn note_off(&self, pitch: u8) -> usize {
        let idx = self.pitches.iter().position(|&p| p == pitch)
            .unwrap_or_else(|| panic!("pitch {pitch} not in vocab"));
        1 + self.pitches.len() + idx
    }

    /// Time shift token for `steps` quantized time steps (1-based).
    pub fn time_shift(&self, steps: usize) -> usize {
        assert!(steps >= 1 && steps <= self.max_time_shift,
            "time shift {steps} out of range 1..={}", self.max_time_shift);
        1 + self.pitches.len() * 2 + (steps - 1)
    }

    pub fn is_note_on(&self, token: usize) -> Option<u8> {
        if token >= 1 && token <= self.pitches.len() {
            Some(self.pitches[token - 1])
        } else {
            None
        }
    }

    pub fn is_note_off(&self, token: usize) -> Option<u8> {
        let off_start = 1 + self.pitches.len();
        if token >= off_start && token < off_start + self.pitches.len() {
            Some(self.pitches[token - off_start])
        } else {
            None
        }
    }

    pub fn is_time_shift(&self, token: usize) -> Option<usize> {
        let ts_start = 1 + self.pitches.len() * 2;
        if token >= ts_start && token < ts_start + self.max_time_shift {
            Some(token - ts_start + 1)
        } else {
            None
        }
    }

    pub fn token_name(&self, token: usize) -> String {
        if token == 0 {
            "PAD".to_string()
        } else if let Some(p) = self.is_note_on(token) {
            format!("ON({})", pitch_name(p))
        } else if let Some(p) = self.is_note_off(token) {
            format!("OFF({})", pitch_name(p))
        } else if let Some(s) = self.is_time_shift(token) {
            format!("T+{s}")
        } else {
            format!("?{token}")
        }
    }
}

pub fn pitch_name(pitch: u8) -> String {
    let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let octave = (pitch / 12) as i32 - 1;
    format!("{}{}", names[(pitch % 12) as usize], octave)
}

/// Quantize MIDI events into a token sequence.
/// `ticks_per_step` controls time quantization granularity.
pub fn tokenize(events: &[MidiEvent], ticks_per_step: u64, max_time_shift: usize) -> (Vec<usize>, Vocab) {
    // Collect unique pitches
    let mut pitch_set = BTreeSet::new();
    for e in events {
        match &e.kind {
            MidiEventKind::NoteOn { pitch, .. } | MidiEventKind::NoteOff { pitch } => {
                pitch_set.insert(*pitch);
            }
        }
    }
    let pitches: Vec<u8> = pitch_set.into_iter().collect();

    let vocab = Vocab { pitches, max_time_shift };

    let mut tokens = Vec::new();
    let mut last_step: u64 = 0;

    for e in events {
        let step = e.tick / ticks_per_step;

        // Emit time shifts
        if step > last_step {
            let mut delta = (step - last_step) as usize;
            while delta > 0 {
                let shift = delta.min(max_time_shift);
                tokens.push(vocab.time_shift(shift));
                delta -= shift;
            }
            last_step = step;
        }

        match &e.kind {
            MidiEventKind::NoteOn { pitch, velocity } => {
                if *velocity > 0 {
                    tokens.push(vocab.note_on(*pitch));
                } else {
                    tokens.push(vocab.note_off(*pitch));
                }
            }
            MidiEventKind::NoteOff { pitch } => {
                tokens.push(vocab.note_off(*pitch));
            }
        }
    }

    (tokens, vocab)
}

/// Tokenize events using an existing vocab (for multi-file training).
pub fn tokenize_with_vocab(events: &[MidiEvent], vocab: &Vocab, ticks_per_step: u64) -> Vec<usize> {
    let mut tokens = Vec::new();
    let mut last_step: u64 = 0;

    for e in events {
        let step = e.tick / ticks_per_step;
        if step > last_step {
            let mut delta = (step - last_step) as usize;
            while delta > 0 {
                let shift = delta.min(vocab.max_time_shift);
                tokens.push(vocab.time_shift(shift));
                delta -= shift;
            }
            last_step = step;
        }
        match &e.kind {
            MidiEventKind::NoteOn { pitch, velocity } => {
                if *velocity > 0 {
                    tokens.push(vocab.note_on(*pitch));
                } else {
                    tokens.push(vocab.note_off(*pitch));
                }
            }
            MidiEventKind::NoteOff { pitch } => {
                tokens.push(vocab.note_off(*pitch));
            }
        }
    }
    tokens
}

/// Parse multiple MIDI files, build a shared vocabulary, return concatenated tokens
/// with a separator gap between pieces.
pub fn tokenize_multi(paths: &[String], ticks_per_step: u64, max_time_shift: usize) -> (Vec<usize>, Vocab) {
    // First pass: collect all pitches
    let mut all_events: Vec<Vec<MidiEvent>> = Vec::new();
    let mut pitch_set = BTreeSet::new();
    for path in paths {
        let events = parse_midi_file(path);
        for e in &events {
            match &e.kind {
                MidiEventKind::NoteOn { pitch, .. } | MidiEventKind::NoteOff { pitch } => {
                    pitch_set.insert(*pitch);
                }
            }
        }
        all_events.push(events);
    }

    let pitches: Vec<u8> = pitch_set.into_iter().collect();
    let vocab = Vocab { pitches, max_time_shift };

    // Second pass: tokenize each file with shared vocab, concatenate
    let mut all_tokens = Vec::new();
    for (i, events) in all_events.iter().enumerate() {
        if i > 0 {
            // Add a large time gap between pieces
            all_tokens.push(vocab.time_shift(max_time_shift));
            all_tokens.push(vocab.time_shift(max_time_shift));
        }
        let tokens = tokenize_with_vocab(events, &vocab, ticks_per_step);
        all_tokens.extend(tokens);
    }

    (all_tokens, vocab)
}

/// Print summary of parsed MIDI
pub fn print_summary(events: &[MidiEvent], tokens: &[usize], vocab: &Vocab) {
    eprintln!("  Notes used: {} unique pitches", vocab.pitches.len());
    eprint!("    ");
    for p in &vocab.pitches {
        eprint!("{} ", pitch_name(*p));
    }
    eprintln!();
    eprintln!("  Total events: {}", events.len());
    eprintln!("  Token sequence length: {}", tokens.len());
    eprintln!("  Vocab size: {}", vocab.size());
    eprintln!("  First 40 tokens: ");
    eprint!("    ");
    for (i, &t) in tokens.iter().take(40).enumerate() {
        if i > 0 { eprint!(" "); }
        eprint!("{}", vocab.token_name(t));
    }
    eprintln!();
}
