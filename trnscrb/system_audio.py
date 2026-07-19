"""Native system-audio capture via ScreenCaptureKit (macOS 13+).

Replaces the old BlackHole loopback-driver approach: captures everything the
Mac plays (the other meeting participants) directly through ScreenCaptureKit,
so no virtual audio driver or Multi-Output Device setup is needed. Requires
the Screen Recording permission (macOS 15+ surfaces it as "System Audio
Recording Only").

Chunks are delivered as 16 kHz mono float32 numpy arrays via a callback,
matching the Recorder's capture format.
"""

import platform
import threading

import numpy as np

from trnscrb.log import get_logger

_log = get_logger("trnscrb.system_audio")

TARGET_RATE = 16_000  # matches trnscrb.recorder.SAMPLE_RATE; SCK supports it natively

_NON_INTERLEAVED_FLAG = 1 << 5  # kAudioFormatFlagIsNonInterleaved

try:  # pragma: no cover - depends on macOS + pyobjc at runtime
    import libdispatch
    import objc
    from CoreMedia import (
        CMAudioFormatDescriptionGetStreamBasicDescription,
        CMBlockBufferCopyDataBytes,
        CMBlockBufferGetDataLength,
        CMSampleBufferGetDataBuffer,
        CMSampleBufferGetFormatDescription,
        CMTimeMake,
    )
    from Foundation import NSObject
    from ScreenCaptureKit import (
        SCContentFilter,
        SCShareableContent,
        SCStream,
        SCStreamConfiguration,
        SCStreamOutputTypeAudio,
    )

    _SCK_AVAILABLE = True
except Exception:  # pragma: no cover
    _SCK_AVAILABLE = False


if _SCK_AVAILABLE:  # pragma: no cover - exercised only with real ScreenCaptureKit
    # Formal protocol adoption is required — without it PyObjC exposes the
    # methods with guessed type signatures and SCStream never delivers buffers.

    class _StreamOutput(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
        """SCStreamOutput conformer that forwards audio sample buffers."""

        def initWithHandler_(self, handler):
            self = objc.super(_StreamOutput, self).init()
            if self is None:
                return None
            self._handler = handler
            return self

        def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
            if output_type != SCStreamOutputTypeAudio:
                return
            try:
                self._handler(sample_buffer)
            except Exception:
                _log.debug("Audio sample handling failed", exc_info=True)

    class _StreamDelegate(NSObject, protocols=[objc.protocolNamed("SCStreamDelegate")]):
        """SCStreamDelegate conformer that reports unexpected stream stops."""

        def initWithHandler_(self, handler):
            self = objc.super(_StreamDelegate, self).init()
            if self is None:
                return None
            self._handler = handler
            return self

        def stream_didStopWithError_(self, stream, error):
            try:
                self._handler(error)
            except Exception:
                _log.debug("Stream-stop handling failed", exc_info=True)


class SystemAudioCapture:
    """Streams macOS system audio as 16 kHz mono float32 chunks.

    ``on_chunk`` is invoked on a background dispatch queue with a 1-D numpy
    array per audio buffer. Use :meth:`is_supported` / :meth:`has_permission`
    to check availability before :meth:`start`.
    """

    def __init__(self, on_chunk):
        self._on_chunk = on_chunk
        self._stream = None
        self._output = None
        self._delegate = None
        self._queue = None
        self._resample = None

    # ── availability ────────────────────────────────────────────────────────

    @staticmethod
    def is_supported() -> bool:
        """True on macOS 13+ with the ScreenCaptureKit bindings importable."""
        if not _SCK_AVAILABLE:
            return False
        major = platform.mac_ver()[0].split(".")[0]
        return major.isdigit() and int(major) >= 13

    @staticmethod
    def has_permission() -> bool:
        """True if the Screen Recording permission is already granted."""
        try:
            from Quartz import CGPreflightScreenCaptureAccess

            return bool(CGPreflightScreenCaptureAccess())
        except Exception:
            return False

    @staticmethod
    def request_permission() -> bool:
        """Prompt for Screen Recording permission; True if already/now granted.

        macOS shows the system dialog only on the first request — afterwards
        the user must enable it manually in System Settings.
        """
        try:
            from Quartz import CGRequestScreenCaptureAccess

            return bool(CGRequestScreenCaptureAccess())
        except Exception:
            return False

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self, timeout: float = 10.0) -> None:
        """Begin capture. Raises if unsupported, unauthorized, or SCK errors."""
        if not self.is_supported():
            raise RuntimeError("ScreenCaptureKit requires macOS 13+")

        # Pre-import here (caller's thread): scipy must never be first-imported
        # on the GCD sample-handler thread, where it can deadlock the queue.
        from scipy.signal import resample_poly

        self._resample = resample_poly

        display = self._primary_display(timeout)
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

        cfg = SCStreamConfiguration.alloc().init()
        cfg.setCapturesAudio_(True)
        cfg.setExcludesCurrentProcessAudio_(True)
        cfg.setSampleRate_(TARGET_RATE)
        cfg.setChannelCount_(1)
        # SCK streams always include video; keep it as cheap as possible
        # since we never attach a video output.
        cfg.setWidth_(2)
        cfg.setHeight_(2)
        cfg.setMinimumFrameInterval_(CMTimeMake(1, 1))
        cfg.setShowsCursor_(False)

        self._delegate = _StreamDelegate.alloc().initWithHandler_(self._on_stream_stopped)
        self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, cfg, self._delegate
        )
        self._output = _StreamOutput.alloc().initWithHandler_(self._handle_sample_buffer)
        self._queue = libdispatch.dispatch_queue_create(b"trnscrb.system-audio", None)
        ok, err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._output, SCStreamOutputTypeAudio, self._queue, None
        )
        if not ok:
            self._stream = None
            raise RuntimeError(f"Could not attach audio output: {err}")

        done = threading.Event()
        result: dict = {}

        def _completion(error):
            result["error"] = error
            done.set()

        self._stream.startCaptureWithCompletionHandler_(_completion)
        if not done.wait(timeout):
            self._stream = None
            raise TimeoutError("ScreenCaptureKit capture start timed out")
        if result.get("error") is not None:
            self._stream = None
            raise RuntimeError(f"ScreenCaptureKit capture failed to start: {result['error']}")
        _log.info("System audio capture started (ScreenCaptureKit)")

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        done = threading.Event()
        try:
            stream.stopCaptureWithCompletionHandler_(lambda error: done.set())
            done.wait(5.0)
        except Exception:
            _log.warning("System audio capture stop failed", exc_info=True)
        self._output = None
        self._delegate = None
        self._queue = None
        _log.info("System audio capture stopped")

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _primary_display(timeout: float):
        done = threading.Event()
        result: dict = {}

        def _handler(content, error):
            result["content"] = content
            result["error"] = error
            done.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(_handler)
        if not done.wait(timeout):
            raise TimeoutError("Querying shareable content timed out")
        if result.get("error") is not None or result.get("content") is None:
            # The usual cause: Screen Recording permission not granted.
            raise RuntimeError(
                f"Cannot access shareable content (Screen Recording permission?): "
                f"{result.get('error')}"
            )
        displays = result["content"].displays()
        if not displays:
            raise RuntimeError("No display available for system audio capture")
        return displays[0]

    def _on_stream_stopped(self, error) -> None:
        # Stream died underneath us (display sleep, permission revoked, …).
        # The Recorder keeps going mic-only; system audio just stops arriving.
        if self._stream is not None:
            _log.warning("System audio stream stopped unexpectedly: %s", error)
            self._stream = None

    def _handle_sample_buffer(self, sample_buffer) -> None:
        block = CMSampleBufferGetDataBuffer(sample_buffer)
        if block is None:
            return
        length = CMBlockBufferGetDataLength(block)
        if length < 4:
            return
        status, data = CMBlockBufferCopyDataBytes(block, 0, length, None)
        if status != 0 or not data:
            return

        samples = np.frombuffer(data, dtype=np.float32)
        rate, channels, interleaved = self._buffer_format(sample_buffer)
        if channels > 1 and samples.size % channels == 0:
            if interleaved:
                samples = samples.reshape(-1, channels).mean(axis=1)
            else:
                samples = samples.reshape(channels, -1).mean(axis=0)
        if rate != TARGET_RATE and self._resample is not None:
            from math import gcd

            g = gcd(rate, TARGET_RATE)
            samples = self._resample(samples, TARGET_RATE // g, rate // g)
        self._on_chunk(np.ascontiguousarray(samples, dtype=np.float32))

    @staticmethod
    def _buffer_format(sample_buffer) -> tuple[int, int, bool]:
        """Return (sample_rate, channels, interleaved) for a sample buffer."""
        try:
            desc = CMSampleBufferGetFormatDescription(sample_buffer)
            # PyObjC returns the AudioStreamBasicDescription as a plain tuple:
            # (mSampleRate, mFormatID, mFormatFlags, mBytesPerPacket,
            #  mFramesPerPacket, mBytesPerFrame, mChannelsPerFrame,
            #  mBitsPerChannel, mReserved)
            asbd = tuple(CMAudioFormatDescriptionGetStreamBasicDescription(desc))
            rate = int(asbd[0])
            flags = int(asbd[2])
            channels = int(asbd[6])
            interleaved = not (flags & _NON_INTERLEAVED_FLAG)
            if rate > 0 and channels > 0:
                return rate, channels, interleaved
        except Exception:
            _log.debug("Could not read sample buffer format", exc_info=True)
        return TARGET_RATE, 1, True
