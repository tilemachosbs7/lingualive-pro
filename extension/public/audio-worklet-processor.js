// AudioWorkletProcessor for processing PCM16 audio chunks
// Optimized for Deepgram with 20-40ms chunking for low latency
// This replaces the deprecated ScriptProcessorNode
class AudioWorkletProcessorImpl extends AudioWorkletProcessor {
  constructor() {
    super();
    // AAA: Buffer for accumulating samples to optimal chunk size
    // At 16kHz, 128 samples = 8ms. Accumulate to ~320-640 samples (20-40ms)
    this.sampleBuffer = new Float32Array(0);
    // Target chunk size: 512 samples (~32ms at 16kHz)
    this.targetChunkSize = 512;
    
    // AAA: Silence gating - skip silent chunks to reduce WS traffic
    // x2: Relaxed threshold (0.002) and longer hangover (6) to avoid cutting word endings
    this.silenceThresholdRMS = 0.002; // RMS threshold (lower = less aggressive)
    this.silenceHangover = 6; // Send N chunks after voice stops (avoid cutoff)
    this.hangoverCounter = 0;
    
    this.port.onmessage = (event) => {
      // Handle config from main thread
      if (event.data.type === 'config') {
        // Allow dynamic chunk size configuration
        if (event.data.chunkSize) {
          this.targetChunkSize = event.data.chunkSize;
        }
        if (event.data.silenceThreshold !== undefined) {
          this.silenceThresholdRMS = event.data.silenceThreshold;
        }
      }
    };
  }

  // AAA: Calculate RMS (root mean square) of audio chunk
  calculateRMS(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  process(inputs, outputs, parameters) {
    // Get input buffer (mono channel)
    const input = inputs[0];
    if (input.length > 0 && input[0].length > 0) {
      const inputData = input[0];
      
      // AAA: Accumulate samples into buffer for optimal chunking
      const newBuffer = new Float32Array(this.sampleBuffer.length + inputData.length);
      newBuffer.set(this.sampleBuffer);
      newBuffer.set(inputData, this.sampleBuffer.length);
      this.sampleBuffer = newBuffer;
      
      // AAA: Send chunks when we have enough samples
      while (this.sampleBuffer.length >= this.targetChunkSize) {
        // Extract a chunk
        const chunk = this.sampleBuffer.slice(0, this.targetChunkSize);
        this.sampleBuffer = this.sampleBuffer.slice(this.targetChunkSize);
        
        // AAA: Silence gating - check RMS level
        const rms = this.calculateRMS(chunk);
        const isSilent = rms < this.silenceThresholdRMS;
        
        if (isSilent) {
          if (this.hangoverCounter > 0) {
            // Still in hangover period - send the chunk
            this.hangoverCounter--;
          } else {
            // Skip silent chunk
            continue;
          }
        } else {
          // Voice detected - reset hangover
          this.hangoverCounter = this.silenceHangover;
        }
        
        // Convert Float32 [-1, 1] to PCM16 Int16Array
        const pcm16 = new Int16Array(chunk.length);
        for (let i = 0; i < chunk.length; i++) {
          const s = Math.max(-1, Math.min(1, chunk[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        
        // AAA: Send as raw ArrayBuffer (binary) for better performance
        // The main thread will send this directly to WebSocket as binary
        const arrayBuffer = pcm16.buffer.slice(0);
        
        this.port.postMessage({
          type: 'audio',
          data: arrayBuffer,
          length: chunk.length,
          binary: true  // AAA: Flag for main thread to send as binary
        }, [arrayBuffer]);
      }
    }
    
    // Return true to keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('audio-worklet-processor', AudioWorkletProcessorImpl);