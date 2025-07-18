import { HLCTimestamp } from '../utils/hlc';

export interface TimestampedValue<T> {
  value: T;
  timestamp: HLCTimestamp;
  author: string;
}

export interface Entity {
  name: TimestampedValue<string>;
  favoriteFood: TimestampedValue<string>;
}

export class LWWEntity {
  private entity: Entity;

  constructor(clientId: string, initialTimestamp: HLCTimestamp) {
    this.entity = {
      name: {
        value: '',
        timestamp: initialTimestamp,
        author: clientId
      },
      favoriteFood: {
        value: '',
        timestamp: initialTimestamp,
        author: clientId
      }
    };
  }

  getEntity(): Entity {
    return { ...this.entity };
  }

  updateAttribute<K extends keyof Entity>(
    attribute: K, 
    value: string, 
    timestamp: HLCTimestamp,
    author: string
  ): boolean {
    const current = this.entity[attribute];
    
    // Last write wins: compare timestamps
    if (timestamp.physical > current.timestamp.physical ||
        (timestamp.physical === current.timestamp.physical && 
         timestamp.logical > current.timestamp.logical)) {
      this.entity[attribute] = {
        value,
        timestamp,
        author
      };
      return true;
    }
    
    return false;
  }

  merge(updates: Partial<Entity>): void {
    Object.entries(updates).forEach(([key, update]) => {
      const attr = key as keyof Entity;
      if (update) {
        this.updateAttribute(attr, update.value, update.timestamp, update.author);
      }
    });
  }
}