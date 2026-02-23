import AVFoundation
import Accelerate
import Combine
import CoreAudio
import SwiftUI

class AudioMonitor: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var pollTimer: Timer?
    private var smoothedLevel: Float = -80

    @Published var currentLevel: Float = -80
    @Published var isMonitoring = false
    @Published var micInUseByOtherApp = false
    @Published var permissionDenied = false
    @Published var manualOverride = false

    @Published var thresholdDB: Double = -20.0 {
        didSet { UserDefaults.standard.set(thresholdDB, forKey: "thresholdDB") }
    }

    var isAboveThreshold: Bool {
        isMonitoring && currentLevel > Float(thresholdDB)
    }

    var isNearThreshold: Bool {
        isMonitoring && currentLevel > Float(thresholdDB) - 6 && !isAboveThreshold
    }

    init() {
        if let saved = UserDefaults.standard.object(forKey: "thresholdDB") as? Double {
            self.thresholdDB = saved
        }
        startPolling()
    }

    func startPolling() {
        // Fire immediately, then every second
        tick()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    private func isMicInUseByCoreAudio() -> Bool {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID) == noErr,
              deviceID != kAudioDeviceUnknown else { return false }

        var isRunning: UInt32 = 0
        size = UInt32(MemoryLayout<UInt32>.size)
        address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &isRunning) == noErr else { return false }
        return isRunning != 0
    }

    private func tick() {
        let inUse = isMicInUseByCoreAudio()

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.micInUseByOtherApp = inUse

            let shouldMonitor = self.manualOverride || inUse
            if shouldMonitor && !self.isMonitoring {
                self.requestAndStart()
            } else if !shouldMonitor && self.isMonitoring {
                self.stopEngine()
            }
        }
    }

    func toggleManual() {
        manualOverride.toggle()
        if manualOverride && !isMonitoring {
            requestAndStart()
        } else if !manualOverride && !micInUseByOtherApp && isMonitoring {
            stopEngine()
        }
    }

    private func requestAndStart() {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if granted {
                    self.permissionDenied = false
                    self.launchEngine()
                } else {
                    self.permissionDenied = true
                }
            }
        }
    }

    private func launchEngine() {
        guard !isMonitoring else { return }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }
            let db = self.calculateDB(buffer: buffer)
            DispatchQueue.main.async {
                self.smoothedLevel = 0.3 * db + 0.7 * self.smoothedLevel
                self.currentLevel = self.smoothedLevel
            }
        }

        do {
            try engine.start()
            self.audioEngine = engine
            self.isMonitoring = true
        } catch {
            print("Audio engine failed: \(error)")
        }
    }

    func stopEngine() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isMonitoring = false
        currentLevel = -80
        smoothedLevel = -80
    }

    private func calculateDB(buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData?[0] else { return -80 }
        let count = UInt(buffer.frameLength)
        var rms: Float = 0
        vDSP_rmsqv(data, 1, &rms, vDSP_Length(count))
        return 20 * log10(max(rms, 1e-10))
    }

    deinit {
        pollTimer?.invalidate()
        stopEngine()
    }
}
