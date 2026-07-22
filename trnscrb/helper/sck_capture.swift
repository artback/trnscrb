// sck-capture — system-audio capture helper for trnscrb.
//
// Lives inside Trnscrb.app and is the ONLY process that touches
// ScreenCaptureKit, so macOS attributes the Screen Recording permission to
// Trnscrb rather than to the Python interpreter (which is Homebrew's
// Python.app, identified as org.python.python and impossible to re-sign as
// ours — see trnscrb/app_bundle.py).
//
// Writes raw 16 kHz mono float32 PCM to stdout; status lines go to stderr.
//
//   sck-capture           capture until SIGTERM/SIGINT or stdin closes
//   sck-capture --check   exit 0 if Screen Recording is permitted, else 1

import AVFoundation
import CoreGraphics
import Foundation
import ScreenCaptureKit

let sampleRate = 16_000
let channelCount = 1

@available(macOS 13.0, *)
final class AudioCapture: NSObject, SCStreamDelegate, SCStreamOutput {
    private var stream: SCStream?
    private let stdout = FileHandle.standardOutput

    func start() async throws {
        // Fails fast with a clear message rather than hanging when the user
        // has not granted (or has revoked) Screen Recording.
        guard CGPreflightScreenCaptureAccess() else {
            throw NSError(
                domain: "sck-capture", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Screen Recording permission not granted"])
        }

        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(
                domain: "sck-capture", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "No display available"])
        }

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = sampleRate
        config.channelCount = channelCount
        config.excludesCurrentProcessAudio = true
        // A video stream is mandatory; keep it as small and slow as possible
        // since we never read the frames.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let newStream = SCStream(filter: filter, configuration: config, delegate: self)
        try newStream.addStreamOutput(
            self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "trnscrb.sck"))
        try await newStream.startCapture()
        stream = newStream
        FileHandle.standardError.write("READY\n".data(using: .utf8)!)
    }

    func stop() async {
        guard let stream else { return }
        self.stream = nil
        try? await stream.stopCapture()
    }

    // MARK: SCStreamOutput

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            block, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length,
            dataPointerOut: &pointer)
        guard status == kCMBlockBufferNoErr, let bytes = pointer, length > 0 else { return }

        // A closed pipe (Python exited) surfaces as EPIPE — exit quietly
        // rather than dying on an uncaught Objective-C exception.
        do {
            try stdout.write(contentsOf: Data(bytes: bytes, count: length))
        } catch {
            exit(0)
        }
    }

    // MARK: SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write(
            "ERROR: stream stopped: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(3)
    }
}

// MARK: - Entry point

if CommandLine.arguments.contains("--check") {
    exit(CGPreflightScreenCaptureAccess() ? 0 : 1)
}

if CommandLine.arguments.contains("--request") {
    // Triggers the system prompt on first use; afterwards the user must
    // grant it in System Settings.
    exit(CGRequestScreenCaptureAccess() ? 0 : 1)
}

guard #available(macOS 13.0, *) else {
    FileHandle.standardError.write("ERROR: requires macOS 13+\n".data(using: .utf8)!)
    exit(2)
}

let capture = AudioCapture()
// Dispatch sources must stay referenced or they are cancelled immediately.
var signalSources: [DispatchSourceSignal] = []

for sig in [SIGTERM, SIGINT] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler {
        Task {
            await capture.stop()
            exit(0)
        }
    }
    source.resume()
    signalSources.append(source)
}

Task {
    do {
        try await capture.start()
    } catch {
        FileHandle.standardError.write(
            "ERROR: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(1)
    }
}

RunLoop.main.run()
