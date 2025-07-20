# Ping Utility TODO

## Current Issues

### High Priority
- [ ] Add mechanism to ensure only one instance is running (prevent multiple daemons)

### Medium Priority
- [ ] Improve error handling for network disconnections
- [ ] Add configuration file support for target SSID and ping settings
- [ ] Add ping statistics summary (average, min, max response times)

### Low Priority
- [ ] Add web interface for viewing ping logs
- [ ] Support for multiple target networks
- [ ] Email notifications for network issues
- [ ] Export logs to CSV format

## Completed Items
- [x] Fix WiFi detection (replaced deprecated airport command with system_profiler)
- [x] Add comprehensive debug logging
- [x] Fix daemon logging after daemonization
- [x] Fix Client::new file descriptor issue in daemon mode
- [x] Improve efficiency (replaced 1-second polling with proper intervals)
- [x] Fix tokio intervals hanging in daemon mode (replaced fork-based daemon with background mode)
- [x] Fix daemon hanging after pinger initialization (resolved tokio+fork incompatibility)