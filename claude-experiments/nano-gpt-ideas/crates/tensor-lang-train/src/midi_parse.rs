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

// ─── Satie-style reward functions ──────────────────────────────────────────
//
// Satie's Gymnopédies have a very specific structure:
//   1. Single bass note (octave 1-2) followed by a time gap
//   2. Then exactly 3 chord tones (octave 3-5) together
//   3. Bass alternates between ~2 pitches (I-V pendulum)
//   4. Smooth voice leading between consecutive chords
//   5. Regular ~12-step measures (3/4 waltz feel)
//   6. Clean releases — notes don't pile up

const BASS_CEILING: u8 = 50; // D3 — anything below is "bass"

/// Extract "cycles" from a token sequence: each cycle starts at a bass NOTE_ON
/// and contains the bass pitch, the chord pitches that follow, the gap between
/// bass and chord, and the total time steps in the cycle.
#[derive(Debug)]
pub struct Cycle {
    pub bass_pitch: u8,
    pub bass_gap: usize,       // time steps between bass ON and first chord ON
    pub chord_pitches: Vec<u8>, // pitches of the chord burst after the gap
    pub total_steps: usize,     // total time steps from this bass to next bass
}

pub fn extract_cycles(tokens: &[usize], vocab: &Vocab) -> Vec<Cycle> {
    // First pass: find bass ON positions and their pitches
    struct BassEvent { token_idx: usize, pitch: u8 }
    let mut bass_events: Vec<BassEvent> = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        if let Some(pitch) = vocab.is_note_on(tok) {
            if pitch < BASS_CEILING {
                bass_events.push(BassEvent { token_idx: i, pitch });
            }
        }
    }

    let mut cycles = Vec::new();
    for (bi, bass) in bass_events.iter().enumerate() {
        let end = if bi + 1 < bass_events.len() {
            bass_events[bi + 1].token_idx
        } else {
            tokens.len()
        };

        // Walk from bass ON to end, collecting gap + chord pitches + total steps
        let mut bass_gap = 0usize;
        let mut chord_pitches: Vec<u8> = Vec::new();
        let mut total_steps = 0usize;
        let mut chord_collecting = false;
        let mut chord_done = false;

        for &tok in &tokens[bass.token_idx + 1..end] {
            if let Some(s) = vocab.is_time_shift(tok) {
                total_steps += s;
                if !chord_collecting && !chord_done { bass_gap += s; }
                // A T+1 within a chord burst is OK (MIDI quantization artifact)
                // But T+2 or more ends the burst
                if chord_collecting && s >= 2 { chord_done = true; chord_collecting = false; }
            } else if let Some(pitch) = vocab.is_note_on(tok) {
                if pitch >= BASS_CEILING && !chord_done {
                    chord_collecting = true;
                    chord_pitches.push(pitch);
                }
            }
            // NOTE_OFFs are ignored for cycle extraction
        }

        cycles.push(Cycle {
            bass_pitch: bass.pitch,
            bass_gap,
            chord_pitches,
            total_steps,
        });
    }
    cycles
}

/// Rule 1: Bass note followed by time gap before chord.
/// Every bass ON must have at least 1 time step before the next chord ON.
pub fn reward_bass_chord_separation(cycles: &[Cycle]) -> bool {
    if cycles.is_empty() { return false; }
    let with_chords: Vec<_> = cycles.iter().filter(|c| !c.chord_pitches.is_empty()).collect();
    if with_chords.is_empty() { return false; }
    let good = with_chords.iter().filter(|c| c.bass_gap >= 1).count();
    good * 2 >= with_chords.len() // at least half have separation
}

/// Rule 2: Chord bursts have 2-4 notes (ideal is 3).
pub fn reward_chord_size(cycles: &[Cycle]) -> bool {
    if cycles.is_empty() { return false; }
    let with_chords: Vec<_> = cycles.iter().filter(|c| !c.chord_pitches.is_empty()).collect();
    if with_chords.is_empty() { return false; }
    let good = with_chords.iter().filter(|c| {
        c.chord_pitches.len() >= 2 && c.chord_pitches.len() <= 4
    }).count();
    good * 2 >= with_chords.len() // at least half in range
}

/// Rule 3: Bass alternates between <= 2 unique pitches.
pub fn reward_bass_alternation(cycles: &[Cycle]) -> bool {
    if cycles.len() < 2 { return false; }
    let mut bass_pitches: Vec<u8> = cycles.iter().map(|c| c.bass_pitch).collect();
    bass_pitches.sort();
    bass_pitches.dedup();
    bass_pitches.len() <= 3 // allow up to 3 unique bass notes (2 ideal + 1 variation)
}

/// Rule 4: Smooth voice leading — consecutive chords move by small intervals.
/// Sorts each chord ascending and compares element-wise.
pub fn reward_smooth_voice_leading(cycles: &[Cycle]) -> bool {
    let chords: Vec<Vec<u8>> = cycles.iter()
        .filter(|c| c.chord_pitches.len() >= 2 && c.chord_pitches.len() <= 4)
        .map(|c| { let mut p = c.chord_pitches.clone(); p.sort(); p })
        .collect();
    if chords.len() < 2 { return true; } // not enough to judge
    let mut good = 0usize;
    let total = chords.len() - 1;
    for i in 0..total {
        let max_move: i32 = chords[i].iter().zip(chords[i + 1].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max().unwrap_or(0);
        if max_move <= 5 { good += 1; } // within a perfect 4th
    }
    good * 2 >= total // at least half have smooth motion
}

/// Rule 5: Measure length roughly consistent (waltz feel).
/// Total steps per cycle should be in [8, 20] range.
pub fn reward_waltz_rhythm(cycles: &[Cycle]) -> bool {
    if cycles.len() < 2 { return false; }
    // Only check cycles that have a "next" (total_steps is meaningful)
    let measurable: Vec<_> = cycles.iter().take(cycles.len() - 1).collect();
    if measurable.is_empty() { return false; }
    let good = measurable.iter().filter(|c| c.total_steps >= 8 && c.total_steps <= 20).count();
    good * 2 >= measurable.len()
}

/// Rule 6: Clean releases — at each bass ON, at most 2 notes should be held.
pub fn reward_clean_releases(tokens: &[usize], vocab: &Vocab) -> bool {
    use std::collections::HashSet;
    let mut held: HashSet<u8> = HashSet::new();
    let mut violations = 0usize;
    let mut bass_count = 0usize;

    for &tok in tokens {
        if let Some(pitch) = vocab.is_note_on(tok) {
            if pitch < BASS_CEILING {
                bass_count += 1;
                if held.len() > 2 { violations += 1; }
            }
            held.insert(pitch);
        } else if let Some(pitch) = vocab.is_note_off(tok) {
            held.remove(&pitch);
        }
    }

    if bass_count == 0 { return false; }
    violations * 3 <= bass_count // at most 1/3 violations
}

/// Rule 7: Chord span between 5 and 16 semitones.
pub fn reward_chord_span(cycles: &[Cycle]) -> bool {
    let with_chords: Vec<_> = cycles.iter()
        .filter(|c| c.chord_pitches.len() >= 2).collect();
    if with_chords.is_empty() { return false; }
    let good = with_chords.iter().filter(|c| {
        let min = *c.chord_pitches.iter().min().unwrap();
        let max = *c.chord_pitches.iter().max().unwrap();
        let span = max - min;
        span >= 5 && span <= 16
    }).count();
    good * 2 >= with_chords.len()
}

/// Rule 8: No dead space — the largest gap between any two note-on events
/// should not exceed 12 time steps (~3 beats). Corpus max is 12.
pub fn reward_no_dead_space(tokens: &[usize], vocab: &Vocab) -> bool {
    let mut last_on_time = 0usize;
    let mut time_pos = 0usize;
    let mut max_gap = 0usize;
    let mut note_count = 0usize;

    for &tok in tokens {
        if let Some(s) = vocab.is_time_shift(tok) {
            time_pos += s;
        } else if vocab.is_note_on(tok).is_some() {
            if note_count > 0 {
                let gap = time_pos - last_on_time;
                max_gap = max_gap.max(gap);
            }
            last_on_time = time_pos;
            note_count += 1;
        }
    }
    note_count >= 8 && max_gap <= 14 // generous but prevents long silences
}

/// Rule 9: Rhythmic commitment — the sequence should use decisive time shifts,
/// not hedge with tons of T+1. In Satie, most gaps between events are T+1..T+5
/// but at least some are T+4 or longer (the structural beats).
/// Penalize if more than 70% of time shifts are T+1.
pub fn reward_rhythmic_commitment(tokens: &[usize], vocab: &Vocab) -> bool {
    let mut t1_count = 0usize;
    let mut total_shifts = 0usize;
    for &tok in tokens {
        if let Some(s) = vocab.is_time_shift(tok) {
            total_shifts += 1;
            if s == 1 { t1_count += 1; }
        }
    }
    if total_shifts < 5 { return false; }
    // T+1 should not dominate — max 70% of all shifts
    t1_count * 10 <= total_shifts * 7
}

/// Rule 10: Melodic contour — the top voice should trace a melody with
/// clear directional movement, not random wandering.
/// Measures this as: the melody has at least one "run" of 3+ notes going
/// in the same direction (up or down).
pub fn reward_melodic_direction(cycles: &[Cycle]) -> bool {
    let tops: Vec<i32> = cycles.iter()
        .filter(|c| !c.chord_pitches.is_empty())
        .map(|c| *c.chord_pitches.iter().max().unwrap() as i32)
        .collect();
    if tops.len() < 4 { return false; }

    // Check for directional runs
    let mut best_run = 1usize;
    let mut cur_run = 1usize;
    let mut cur_dir: i32 = 0; // +1 = up, -1 = down
    for i in 1..tops.len() {
        let diff = tops[i] - tops[i - 1];
        let dir = if diff > 0 { 1 } else if diff < 0 { -1 } else { 0 };
        if dir != 0 && dir == cur_dir {
            cur_run += 1;
            best_run = best_run.max(cur_run);
        } else if dir != 0 {
            cur_dir = dir;
            cur_run = 2; // this note + previous
        }
        // dir == 0 (repeat) doesn't extend or break
    }

    // Also check: melody range spans at least 5 semitones
    let min_top = tops.iter().min().unwrap();
    let max_top = tops.iter().max().unwrap();
    let range = max_top - min_top;

    // Must have a 3+ note directional run AND span at least 5 semitones
    best_run >= 3 && range >= 5
}

/// Rule 11: Harmonic progression — consecutive chords should change.
/// No more than 2 identical chords in a row.
pub fn reward_harmonic_motion(cycles: &[Cycle]) -> bool {
    let chords: Vec<Vec<u8>> = cycles.iter()
        .filter(|c| c.chord_pitches.len() >= 2)
        .map(|c| { let mut p = c.chord_pitches.clone(); p.sort(); p })
        .collect();
    if chords.len() < 3 { return false; }

    // Check max consecutive identical chords
    let mut max_repeat = 1usize;
    let mut cur_repeat = 1usize;
    for i in 1..chords.len() {
        if chords[i] == chords[i - 1] {
            cur_repeat += 1;
            max_repeat = max_repeat.max(cur_repeat);
        } else {
            cur_repeat = 1;
        }
    }

    // Also: at least 3 distinct chords total
    let mut unique = chords.clone();
    unique.sort();
    unique.dedup();

    max_repeat <= 2 && unique.len() >= 3
}

/// Rule 12: Minimum density — must have at least 4 note-ons per 16-step measure.
/// Satie averages 5.4-7.7. This prevents sparse drifting.
pub fn reward_density(tokens: &[usize], vocab: &Vocab) -> bool {
    let mut time_pos = 0usize;
    let mut measure_start = 0usize;
    let mut measure_notes = 0usize;
    let mut measures = 0usize;
    let mut sparse_measures = 0usize;

    for &tok in tokens {
        if let Some(s) = vocab.is_time_shift(tok) {
            time_pos += s;
            while time_pos >= measure_start + 16 {
                measures += 1;
                if measure_notes < 4 { sparse_measures += 1; }
                measure_start += 16;
                measure_notes = 0;
            }
        } else if vocab.is_note_on(tok).is_some() {
            measure_notes += 1;
        }
    }
    if measures < 2 { return false; }
    // At most 1/3 of measures can be sparse
    sparse_measures * 3 <= measures
}

/// Compute the overall Satie-style reward for a token sequence.
pub fn compute_satie_reward(tokens: &[usize], vocab: &Vocab) -> (f32, SatieRewardBreakdown) {
    let cycles = extract_cycles(tokens, vocab);

    // Structure
    let separation = reward_bass_chord_separation(&cycles);
    let chords = reward_chord_size(&cycles);
    let alternation = reward_bass_alternation(&cycles);
    let voice_leading = reward_smooth_voice_leading(&cycles);

    // Drive / forward motion
    let no_dead = reward_no_dead_space(tokens, vocab);
    let rhythm = reward_rhythmic_commitment(tokens, vocab);
    let melody = reward_melodic_direction(&cycles);
    let harmony = reward_harmonic_motion(&cycles);
    let density = reward_density(tokens, vocab);

    let breakdown = SatieRewardBreakdown {
        n_cycles: cycles.len(),
        separation, chords, alternation, voice_leading,
        no_dead, rhythm, melody, harmony, density,
    };

    // Must have Satie texture AND forward motion
    let texture = separation && chords && alternation;
    let motion = melody && harmony;
    let drive = no_dead && density;
    let quality = voice_leading || rhythm;
    let reward = if texture && motion && drive && quality { 1.0 } else { 0.0 };

    (reward, breakdown)
}

pub struct SatieRewardBreakdown {
    pub n_cycles: usize,
    pub separation: bool,
    pub chords: bool,
    pub alternation: bool,
    pub voice_leading: bool,
    pub no_dead: bool,
    pub rhythm: bool,
    pub melody: bool,
    pub harmony: bool,
    pub density: bool,
}

impl std::fmt::Display for SatieRewardBreakdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = |b: bool| if b { "✓" } else { "✗" };
        write!(f, "{}cyc tx={}{}{}{} mv={}{}{} dr={}{}",
            self.n_cycles,
            c(self.separation), c(self.chords), c(self.alternation), c(self.voice_leading),
            c(self.melody), c(self.harmony), c(self.rhythm),
            c(self.no_dead), c(self.density),
        )
    }
}
