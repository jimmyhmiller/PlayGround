import CoreLocation
import Foundation

/// Service that uses location tracking to keep the app alive in background
/// This is the technique used by SSH apps like Blink shell
class LocationKeepAliveService: NSObject, CLLocationManagerDelegate {
    static let shared = LocationKeepAliveService()

    private let locationManager = CLLocationManager()
    private var isTracking = false

    private override init() {
        super.init()
        locationManager.delegate = self
        // Use high accuracy to force iOS to keep the app running
        // Lower accuracy allows iOS to suspend the app between updates
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone // Get all updates
        locationManager.allowsBackgroundLocationUpdates = true
        locationManager.pausesLocationUpdatesAutomatically = false
        // Hint that this is for navigation-like continuous tracking
        locationManager.activityType = .otherNavigation
        // Show indicator so user knows location is being used
        locationManager.showsBackgroundLocationIndicator = true
    }

    /// Start location tracking to keep app alive
    func startTracking() {
        guard !isTracking else { return }

        let status = locationManager.authorizationStatus

        switch status {
        case .notDetermined:
            appLog("Requesting location authorization", category: "Location")
            locationManager.requestAlwaysAuthorization()
        case .authorizedAlways:
            startLocationUpdates()
        case .authorizedWhenInUse:
            appLog("Have 'when in use' authorization, requesting 'always'", category: "Location")
            locationManager.requestAlwaysAuthorization()
            // Start anyway - will work while app is in foreground
            startLocationUpdates()
        case .denied, .restricted:
            appLog("Location access denied - background connections will drop", category: "Location")
        @unknown default:
            appLog("Unknown location authorization status", category: "Location")
        }
    }

    private func startLocationUpdates() {
        locationManager.startUpdatingLocation()
        isTracking = true
        appLog("Started location tracking for background keep-alive", category: "Location")
    }

    /// Stop location tracking
    func stopTracking() {
        guard isTracking else { return }

        locationManager.stopUpdatingLocation()
        isTracking = false
        appLog("Stopped location tracking", category: "Location")
    }

    // MARK: - CLLocationManagerDelegate

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        appLog("Location authorization changed: \(status.rawValue)", category: "Location")

        if status == .authorizedAlways || status == .authorizedWhenInUse {
            if !isTracking {
                startLocationUpdates()
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        // We don't actually need the location, just the background execution time
        // Log occasionally to confirm it's working
        if let location = locations.last {
            appLog("Location update (keeping connection alive): \(location.coordinate.latitude), \(location.coordinate.longitude)", category: "Location")
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        appLog("Location error: \(error.localizedDescription)", category: "Location")
    }
}
