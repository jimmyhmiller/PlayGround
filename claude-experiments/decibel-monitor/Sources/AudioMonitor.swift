import AVFoundation
import Accelerate
import Combine
import CoreAudio
import SwiftUI

class AudioMonitor: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var pollTimer: Timer?
    private var stopCheckTimer: Timer?
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

    // Runs every 1s. Only checks when NOT monitoring (avoids detecting our own engine).
    private func tick() {
        guard !isMonitoring else { return }

        let inUse = isMicInUseByCoreAudio()
        micInUseByOtherApp = inUse

        if inUse || manualOverride {
            requestAndStart()
        }
    }

    // Runs every 5s while auto-monitoring. Synchronously stops engine, checks if
    // others are still using the mic, and restarts immediately if so. The entire
    // stop→check→restart takes <1ms — no visible UI flicker, no audible gap.
    private func performStopCheck() {
        guard isMonitoring, !manualOverride, let engine = audioEngine else {
            stopCheckTimer?.invalidate()
            stopCheckTimer = nil
            return
        }

        // Synchronous: stop releases our IOProc so isRunningSomewhere
        // reflects only other processes.
        engine.stop()
        let othersRunning = isMicInUseByCoreAudio()

        if othersRunning {
            // Others still active — restart immediately. Tap survives stop/start.
            // No @Published properties change, so zero UI flicker.
            try? engine.start()
        } else {
            // Nobody else using mic — actually tear down.
            engine.inputNode.removeTap(onBus: 0)
            audioEngine = nil
            isMonitoring = false
            currentLevel = -80
            smoothedLevel = -80
            micInUseByOtherApp = false
            stopCheckTimer?.invalidate()
            stopCheckTimer = nil
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

            // Only schedule stop-checks for auto-started monitoring.
            // When manualOverride is on, user controls stop explicitly.
            if !manualOverride {
                stopCheckTimer?.invalidate()
                stopCheckTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
                    self?.performStopCheck()
                }
            }
        } catch {
            print("Audio engine failed: \(error)")
        }
    }

    func stopEngine() {
        stopCheckTimer?.invalidate()
        stopCheckTimer = nil
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
