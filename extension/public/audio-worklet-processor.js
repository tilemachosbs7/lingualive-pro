// AudioWorkletProcessor for processing PCM16 audio chunks
// SIMPLIFIED: No silence gating - send ALL audio to Deepgram
// This ensures no audio is lost during quiet passages or music
class AudioWorkletProcessorImpl extends AudioWorkletProcessor {
  constructor() {
    super();
    // Buffer for accumulating samples to optimal chunk size
    this.sampleBuffer = new Float32Array(0);
    // Target chunk size: 512 samples (~32ms at 16kHz)
    this.targetChunkSize = 512;
    // Debug: count chunks sent
    this.chunksSent = 0;
    
    this.port.onmessage = (event) => {
      if (event.data.type === 'config') {
        if (event.data.chunkSize) {
          this.targetChunkSize = event.data.chunkSize;
        }
      }
    };
  }

  process(inputs, outputs, parameters) {
    // Get input buffer (mono channel)
    const input = inputs[0];
    if (input.length > 0 && input[0].length > 0) {
      const inputData = input[0];
      
      // Accumulate samples into buffer
      const newBuffer = new Float32Array(this.sampleBuffer.length + inputData.length);
      newBuffer.set(this.sampleBuffer);
      newBuffer.set(inputData, this.sampleBuffer.length);
      this.sampleBuffer = newBuffer;
      
      // Send ALL chunks - NO SILENCE GATING
      while (this.sampleBuffer.length >= this.targetChunkSize) {
        // Extract a chunk
        const chunk = this.sampleBuffer.slice(0, this.targetChunkSize);
        this.sampleBuffer = this.sampleBuffer.slice(this.targetChunkSize);
        
        // Convert Float32 [-1, 1] to PCM16 Int16Array
        const pcm16 = new Int16Array(chunk.length);
        for (let i = 0; i < chunk.length; i++) {
          const s = Math.max(-1, Math.min(1, chunk[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        
        // Send as raw ArrayBuffer (binary)
        const arrayBuffer = pcm16.buffer.slice(0);
        
        this.chunksSent++;
        this.port.postMessage({
          type: 'audio',
          data: arrayBuffer,
          length: chunk.length,
          binary: true,
          chunkNum: this.chunksSent
        }, [arrayBuffer]);
      }
    }
    
    // Return true to keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('audio-worklet-processor', AudioWorkletProcessorImpl);