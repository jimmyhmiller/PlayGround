// Shared test data for all benchmarks
export const SMALL_FUNCTION = `function add(a, b) {
    return a + b;
}`;

export const SMALL_CLASS = `function Calculator() {
    this.result = 0;
}

Calculator.prototype.add = function(a, b) {
    return a + b;
};

Calculator.prototype.subtract = function(a, b) {
    return a - b;
};`;

export const MEDIUM_ASYNC_MODULE = `function UserDataFetcher() {
    this.cache = {};
}

UserDataFetcher.prototype.fetchUserData = function(userId, callback) {
    var self = this;

    if (this.cache[userId]) {
        callback(null, this.cache[userId]);
        return;
    }

    fetch('/api/users/' + userId).then(function(response) {
        if (!response.ok) {
            throw new Error('Failed to fetch user');
        }
        return response.json();
    }).then(function(data) {
        self.cache[userId] = data;
        callback(null, data);
    }).catch(function(error) {
        console.error('Error fetching user:', error);
        callback(error);
    });
};

UserDataFetcher.prototype.processUserBatch = function(userIds, callback) {
    var results = [];
    var completed = 0;
    var hasError = false;

    for (var i = 0; i < userIds.length; i++) {
        this.fetchUserData(userIds[i], function(err, data) {
            if (hasError) return;
            if (err) {
                hasError = true;
                callback(err);
                return;
            }
            results.push(data);
            completed++;
            if (completed === userIds.length) {
                callback(null, results);
            }
        });
    }
};

function debounce(func, wait) {
    var timeout;
    return function() {
        var context = this;
        var args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            func.apply(context, args);
        }, wait);
    };
}`;

export const LARGE_MODULE = `function DataProcessor(options) {
    options = options || {};
    this.batchSize = options.batchSize || 100;
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 1000;
    this.queue = [];
    this.processing = false;
    this.listeners = {};
}

DataProcessor.prototype.on = function(event, handler) {
    if (!this.listeners[event]) {
        this.listeners[event] = [];
    }
    this.listeners[event].push(handler);
};

DataProcessor.prototype.emit = function(event, data) {
    var handlers = this.listeners[event] || [];
    for (var i = 0; i < handlers.length; i++) {
        handlers[i](data);
    }
};

DataProcessor.prototype.process = function(data, callback) {
    this.queue.push(data);
    this.emit('queued', { queueSize: this.queue.length });

    if (!this.processing) {
        this._processBatch(callback);
    }
};

DataProcessor.prototype._processBatch = function(callback) {
    var self = this;

    if (this.queue.length === 0) {
        this.processing = false;
        if (callback) callback();
        return;
    }

    this.processing = true;
    var batch = this.queue.splice(0, this.batchSize);
    var results = [];
    var completed = 0;

    for (var i = 0; i < batch.length; i++) {
        this._processItem(batch[i], function(err, result) {
            if (err) {
                self.emit('error', err);
                return;
            }
            results.push(result);
            completed++;
            if (completed === batch.length) {
                self.emit('batch-complete', { count: results.length });
                self._processBatch(callback);
            }
        });
    }
};

DataProcessor.prototype._processItem = function(item, callback) {
    var self = this;
    var attempt = 1;

    function tryProcess() {
        try {
            var hashValue = JSON.stringify(item);
            self._delay(10, function() {
                callback(null, {
                    id: item.id,
                    data: item.data,
                    hash: hashValue,
                    processed: true
                });
            });
        } catch (error) {
            if (attempt >= self.retryAttempts) {
                callback(error);
            } else {
                attempt++;
                self._delay(self.retryDelay * attempt, tryProcess);
            }
        }
    }

    tryProcess();
};

DataProcessor.prototype._delay = function(ms, callback) {
    setTimeout(callback, ms);
};

DataProcessor.prototype.getQueueSize = function() {
    return this.queue.length;
};

DataProcessor.prototype.isProcessing = function() {
    return this.processing;
};`;
