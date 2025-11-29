/**
 * Animation and easing utilities
 */

/**
 * Easing functions for smooth animations
 */
export const Easing = {
    // No easing
    linear: (t) => t,

    // Quadratic
    easeInQuad: (t) => t * t,
    easeOutQuad: (t) => t * (2 - t),
    easeInOutQuad: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

    // Cubic
    easeInCubic: (t) => t * t * t,
    easeOutCubic: (t) => (--t) * t * t + 1,
    easeInOutCubic: (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,

    // Quartic
    easeInQuart: (t) => t * t * t * t,
    easeOutQuart: (t) => 1 - (--t) * t * t * t,
    easeInOutQuart: (t) => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t,

    // Quintic
    easeInQuint: (t) => t * t * t * t * t,
    easeOutQuint: (t) => 1 + (--t) * t * t * t * t,
    easeInOutQuint: (t) => t < 0.5 ? 16 * t * t * t * t * t : 1 + 16 * (--t) * t * t * t * t,

    // Sine
    easeInSine: (t) => 1 - Math.cos(t * Math.PI / 2),
    easeOutSine: (t) => Math.sin(t * Math.PI / 2),
    easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,

    // Exponential
    easeInExpo: (t) => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
    easeOutExpo: (t) => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
    easeInOutExpo: (t) => {
        if (t === 0 || t === 1) return t;
        t *= 2;
        if (t < 1) return 0.5 * Math.pow(2, 10 * (t - 1));
        return 0.5 * (2 - Math.pow(2, -10 * (t - 1)));
    },

    // Circular
    easeInCirc: (t) => 1 - Math.sqrt(1 - t * t),
    easeOutCirc: (t) => Math.sqrt(1 - (--t) * t),
    easeInOutCirc: (t) => {
        t *= 2;
        if (t < 1) return -(Math.sqrt(1 - t * t) - 1) / 2;
        t -= 2;
        return (Math.sqrt(1 - t * t) + 1) / 2;
    },

    // Elastic
    easeInElastic: (t) => {
        if (t === 0 || t === 1) return t;
        const p = 0.3;
        return -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1 - p / 4) * (2 * Math.PI) / p);
    },
    easeOutElastic: (t) => {
        if (t === 0 || t === 1) return t;
        const p = 0.3;
        return Math.pow(2, -10 * t) * Math.sin((t - p / 4) * (2 * Math.PI) / p) + 1;
    },
    easeInOutElastic: (t) => {
        if (t === 0 || t === 1) return t;
        const p = 0.3 * 1.5;
        t *= 2;
        if (t < 1) {
            return -0.5 * Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1 - p / 4) * (2 * Math.PI) / p);
        }
        return Math.pow(2, -10 * (t - 1)) * Math.sin((t - 1 - p / 4) * (2 * Math.PI) / p) * 0.5 + 1;
    },

    // Back
    easeInBack: (t) => {
        const s = 1.70158;
        return t * t * ((s + 1) * t - s);
    },
    easeOutBack: (t) => {
        const s = 1.70158;
        t--;
        return t * t * ((s + 1) * t + s) + 1;
    },
    easeInOutBack: (t) => {
        const s = 1.70158 * 1.525;
        t *= 2;
        if (t < 1) return 0.5 * (t * t * ((s + 1) * t - s));
        t -= 2;
        return 0.5 * (t * t * ((s + 1) * t + s) + 2);
    },

    // Bounce
    easeInBounce: (t) => 1 - Easing.easeOutBounce(1 - t),
    easeOutBounce: (t) => {
        if (t < 1 / 2.75) {
            return 7.5625 * t * t;
        } else if (t < 2 / 2.75) {
            t -= 1.5 / 2.75;
            return 7.5625 * t * t + 0.75;
        } else if (t < 2.5 / 2.75) {
            t -= 2.25 / 2.75;
            return 7.5625 * t * t + 0.9375;
        } else {
            t -= 2.625 / 2.75;
            return 7.5625 * t * t + 0.984375;
        }
    },
    easeInOutBounce: (t) => {
        if (t < 0.5) return Easing.easeInBounce(t * 2) * 0.5;
        return Easing.easeOutBounce(t * 2 - 1) * 0.5 + 0.5;
    },
};

/**
 * Interpolate between two values
 */
export function lerp(start, end, t) {
    return start + (end - start) * t;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

/**
 * Map a value from one range to another
 */
export function map(value, inMin, inMax, outMin, outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

/**
 * Simple animation class
 */
export class Animation {
    constructor(duration, easing = Easing.linear) {
        this.duration = duration; // milliseconds
        this.easing = easing;
        this.startTime = null;
        this.running = false;
        this.onUpdate = null;
        this.onComplete = null;
        this.loop = false;
        this.yoyo = false;
        this.reverse = false;
    }

    start(currentTime = performance.now()) {
        this.startTime = currentTime;
        this.running = true;
        this.reverse = false;
        return this;
    }

    stop() {
        this.running = false;
        return this;
    }

    reset() {
        this.startTime = null;
        this.running = false;
        this.reverse = false;
        return this;
    }

    update(currentTime = performance.now()) {
        if (!this.running || this.startTime === null) {
            return 0;
        }

        const elapsed = currentTime - this.startTime;
        let t = clamp(elapsed / this.duration, 0, 1);

        if (this.reverse) {
            t = 1 - t;
        }

        const easedT = this.easing(t);

        if (this.onUpdate) {
            this.onUpdate(easedT, t);
        }

        if (t >= 1) {
            if (this.yoyo) {
                this.reverse = !this.reverse;
                this.startTime = currentTime;
                if (this.reverse && this.onComplete) {
                    this.onComplete();
                }
            } else if (this.loop) {
                this.startTime = currentTime;
            } else {
                this.running = false;
                if (this.onComplete) {
                    this.onComplete();
                }
            }
        }

        return easedT;
    }

    setLoop(loop) {
        this.loop = loop;
        return this;
    }

    setYoyo(yoyo) {
        this.yoyo = yoyo;
        return this;
    }
}

/**
 * Animation sequence manager
 */
export class AnimationSequence {
    constructor() {
        this.animations = [];
        this.currentIndex = 0;
        this.running = false;
    }

    add(animation) {
        this.animations.push(animation);
        return this;
    }

    start() {
        if (this.animations.length === 0) return;
        this.currentIndex = 0;
        this.running = true;
        const anim = this.animations[this.currentIndex];
        anim.onComplete = () => this.next();
        anim.start();
        return this;
    }

    next() {
        this.currentIndex++;
        if (this.currentIndex < this.animations.length) {
            const anim = this.animations[this.currentIndex];
            anim.onComplete = () => this.next();
            anim.start();
        } else {
            this.running = false;
        }
    }

    stop() {
        if (this.currentIndex < this.animations.length) {
            this.animations[this.currentIndex].stop();
        }
        this.running = false;
        return this;
    }

    reset() {
        this.animations.forEach(anim => anim.reset());
        this.currentIndex = 0;
        this.running = false;
        return this;
    }

    update(currentTime) {
        if (this.running && this.currentIndex < this.animations.length) {
            this.animations[this.currentIndex].update(currentTime);
        }
    }
}

/**
 * Spring physics for smooth animations
 */
export class Spring {
    constructor(stiffness = 170, damping = 26) {
        this.stiffness = stiffness;
        this.damping = damping;
        this.mass = 1;
        this.value = 0;
        this.velocity = 0;
        this.target = 0;
    }

    setTarget(target) {
        this.target = target;
    }

    setValue(value) {
        this.value = value;
        this.velocity = 0;
    }

    update(dt) {
        const force = -this.stiffness * (this.value - this.target);
        const dampingForce = -this.damping * this.velocity;
        const acceleration = (force + dampingForce) / this.mass;

        this.velocity += acceleration * dt;
        this.value += this.velocity * dt;

        return this.value;
    }

    isAtRest(threshold = 0.001) {
        return Math.abs(this.value - this.target) < threshold &&
               Math.abs(this.velocity) < threshold;
    }
}

/**
 * Oscillator for periodic animations
 */
export class Oscillator {
    constructor(frequency = 1.0, amplitude = 1.0, phase = 0) {
        this.frequency = frequency;
        this.amplitude = amplitude;
        this.phase = phase;
    }

    sine(t) {
        return this.amplitude * Math.sin(2 * Math.PI * this.frequency * t + this.phase);
    }

    cosine(t) {
        return this.amplitude * Math.cos(2 * Math.PI * this.frequency * t + this.phase);
    }

    triangle(t) {
        const period = 1 / this.frequency;
        const phaseShift = this.phase / (2 * Math.PI);
        const x = ((t + phaseShift) % period) / period;
        return this.amplitude * (4 * Math.abs(x - 0.5) - 1);
    }

    square(t) {
        const period = 1 / this.frequency;
        const phaseShift = this.phase / (2 * Math.PI);
        const x = ((t + phaseShift) % period) / period;
        return this.amplitude * (x < 0.5 ? 1 : -1);
    }

    sawtooth(t) {
        const period = 1 / this.frequency;
        const phaseShift = this.phase / (2 * Math.PI);
        const x = ((t + phaseShift) % period) / period;
        return this.amplitude * (2 * x - 1);
    }
}
