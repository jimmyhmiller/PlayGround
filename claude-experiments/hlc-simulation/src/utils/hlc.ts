export interface HLCTimestamp {
  physical: number;
  logical: number;
}

export class HybridLogicalClock {
  private timestamp: HLCTimestamp;
  private clockSkew: number = 0;

  constructor() {
    this.timestamp = {
      physical: this.getPhysicalTime(),
      logical: 0
    };
  }

  getPhysicalTime(): number {
    return Date.now() + this.clockSkew;
  }

  setClockSkew(skewMs: number): void {
    this.clockSkew = skewMs;
  }

  getTimestamp(): HLCTimestamp {
    return { ...this.timestamp };
  }

  tick(): HLCTimestamp {
    const physicalTime = this.getPhysicalTime();
    
    if (physicalTime > this.timestamp.physical) {
      this.timestamp.physical = physicalTime;
      this.timestamp.logical = 0;
    } else {
      this.timestamp.logical++;
    }
    
    return this.getTimestamp();
  }

  update(remoteTimestamp: HLCTimestamp): HLCTimestamp {
    const localPhysical = this.getPhysicalTime();
    const maxPhysical = Math.max(localPhysical, this.timestamp.physical, remoteTimestamp.physical);
    
    if (maxPhysical === localPhysical && maxPhysical > this.timestamp.physical && maxPhysical > remoteTimestamp.physical) {
      this.timestamp.physical = maxPhysical;
      this.timestamp.logical = 0;
    } else if (maxPhysical === this.timestamp.physical && maxPhysical === remoteTimestamp.physical) {
      this.timestamp.physical = maxPhysical;
      this.timestamp.logical = Math.max(this.timestamp.logical, remoteTimestamp.logical) + 1;
    } else if (maxPhysical === this.timestamp.physical) {
      this.timestamp.logical++;
    } else if (maxPhysical === remoteTimestamp.physical) {
      this.timestamp.physical = maxPhysical;
      this.timestamp.logical = remoteTimestamp.logical + 1;
    } else {
      this.timestamp.physical = maxPhysical;
      this.timestamp.logical = 0;
    }
    
    return this.getTimestamp();
  }

  toString(): string {
    return `${this.timestamp.physical}:${this.timestamp.logical}`;
  }
}