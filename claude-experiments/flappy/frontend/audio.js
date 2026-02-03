// Simple audio system using Web Audio API
class AudioSystem {
    constructor() {
        this.enabled = true;
        this.audioContext = null;
        this.masterGain = null;
        this.init();
    }
    
    init() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.connect(this.audioContext.destination);
            this.masterGain.gain.value = 0.3; // Master volume
        } catch (e) {
            console.warn('Web Audio API not supported:', e);
            this.enabled = false;
        }
    }
    
    // Generate jump sound
    playJump() {
        if (!this.enabled) return;
        
        const now = this.audioContext.currentTime;
        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        // Jump sound: quick pitch rise
        osc.frequency.setValueAtTime(400, now);
        osc.frequency.exponentialRampToValueAtTime(800, now + 0.1);
        
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
        
        osc.start(now);
        osc.stop(now + 0.15);
    }
    
    // Generate score sound
    playScore() {
        if (!this.enabled) return;
        
        const now = this.audioContext.currentTime;
        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        // Score sound: pleasant chime
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.setValueAtTime(1000, now + 0.05);
        osc.type = 'sine';
        
        gain.gain.setValueAtTime(0.4, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
        
        osc.start(now);
        osc.stop(now + 0.3);
    }
    
    // Generate death sound
    playDeath() {
        if (!this.enabled) return;
        
        const now = this.audioContext.currentTime;
        
        // Death sound: descending explosion
        const osc1 = this.audioContext.createOscillator();
        const osc2 = this.audioContext.createOscillator();
        const noise = this.createNoiseBuffer();
        const noiseSource = this.audioContext.createBufferSource();
        const gain = this.audioContext.createGain();
        
        noiseSource.buffer = noise;
        
        osc1.connect(gain);
        osc2.connect(gain);
        noiseSource.connect(gain);
        gain.connect(this.masterGain);
        
        osc1.frequency.setValueAtTime(200, now);
        osc1.frequency.exponentialRampToValueAtTime(50, now + 0.5);
        osc1.type = 'sawtooth';
        
        osc2.frequency.setValueAtTime(100, now);
        osc2.frequency.exponentialRampToValueAtTime(25, now + 0.5);
        osc2.type = 'square';
        
        gain.gain.setValueAtTime(0.5, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
        
        osc1.start(now);
        osc2.start(now);
        noiseSource.start(now);
        
        osc1.stop(now + 0.5);
        osc2.stop(now + 0.5);
        noiseSource.stop(now + 0.5);
    }
    
    // Create white noise for explosion effect
    createNoiseBuffer() {
        const bufferSize = this.audioContext.sampleRate * 0.5;
        const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const output = buffer.getChannelData(0);
        
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
        
        return buffer;
    }
    
    toggle() {
        this.enabled = !this.enabled;
        return this.enabled;
    }
    
    setVolume(volume) {
        if (this.masterGain) {
            this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
        }
    }
}

// Export global audio instance
window.audioSystem = new AudioSystem();
