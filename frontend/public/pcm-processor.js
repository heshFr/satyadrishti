// pcm-processor.js — AudioWorklet processor for reliable PCM capture
// This replaces the deprecated ScriptProcessorNode which Chrome throttles
// in background tabs, causing silent audio drops.

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._sampleCount = 0;
    // Buffer ~128ms of audio before posting (2048 samples at 16kHz)
    this._flushThreshold = 2048;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Take channel 0 (mono)
    const channelData = input[0];
    if (!channelData || channelData.length === 0) return true;

    // Copy the samples (they're recycled by the engine)
    this._buffer.push(new Float32Array(channelData));
    this._sampleCount += channelData.length;

    if (this._sampleCount >= this._flushThreshold) {
      // Concatenate and send
      const combined = new Float32Array(this._sampleCount);
      let offset = 0;
      for (const chunk of this._buffer) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      this.port.postMessage({ type: "pcm", samples: combined });
      this._buffer = [];
      this._sampleCount = 0;
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
