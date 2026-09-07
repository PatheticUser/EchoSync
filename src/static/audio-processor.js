/**
 * AudioWorkletProcessor for EchoSync Voice Ingestion.
 * Downsamples native hardware audio to 16,000 Hz, quantizes to 16-bit Linear PCM (little-endian),
 * and frames into exact 512-sample (1,024 byte) buffers for zero-copy transmission.
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSize = 512;
    this.buffer = new Int16Array(this.frameSize);
    this.bufferIndex = 0;
    this.ratio = sampleRate / this.targetSampleRate;
    this.phase = 0.0;
    this.prevSample = 0.0;
    this.gain = 2.0; // 2x gain boost for quiet hardware mic inputs
    this.framesProcessed = 0;
    this.framesEmitted = 0;
    this.peakSample = 0.0;
    this.hasLoggedInitial = false;
  }

  process(inputs, outputs, parameters) {
    this.framesProcessed++;
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }
    const channel = input[0];
    if (!channel || channel.length === 0) {
      return true;
    }

    // Diagnostic check for non-zero audio input on startup
    if (!this.hasLoggedInitial) {
      let nonZeroCount = 0;
      for (let i = 0; i < channel.length; i++) {
        if (Math.abs(channel[i]) > 1e-6) nonZeroCount++;
      }
      if (nonZeroCount > 0) {
        console.log(
          `[AudioProcessor] Non-zero input detected: ${nonZeroCount}/${channel.length} samples active. Native sampleRate: ${sampleRate} Hz`
        );
        this.hasLoggedInitial = true;
      }
    }

    const len = channel.length;
    let phase = this.phase;
    const prev = this.prevSample;

    while (phase < len) {
      const idx = Math.floor(phase);
      const frac = phase - idx;
      const s0 = idx < 0 ? prev : channel[idx];
      const s1 = idx + 1 < len ? channel[idx + 1] : channel[len - 1];
      const rawSample = s0 + frac * (s1 - s0);

      // Apply gain boost
      const sample = rawSample * this.gain;
      const absSample = Math.abs(sample);
      if (absSample > this.peakSample) {
        this.peakSample = absSample;
      }

      // Clamp to [-1.0, 1.0] and quantize to 16-bit signed PCM
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      const pcm16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this.buffer[this.bufferIndex++] = Math.round(pcm16);

      if (this.bufferIndex >= this.frameSize) {
        this.framesEmitted++;
        if (this.framesEmitted % 100 === 0) {
          console.log(
            `[AudioProcessor] Frame #${this.framesEmitted} emitted, peak amp: ${this.peakSample.toFixed(4)}, gain: ${this.gain}x`
          );
          this.peakSample = 0.0;
        }

        const transferBuffer = this.buffer.buffer;
        try {
          this.port.postMessage(transferBuffer, [transferBuffer]);
        } catch (postErr) {
          console.error("[AudioProcessor] Error posting PCM buffer:", postErr);
        }
        this.buffer = new Int16Array(this.frameSize);
        this.bufferIndex = 0;
      }

      phase += this.ratio;
    }

    this.phase = phase - len;
    this.prevSample = channel[len - 1];
    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
