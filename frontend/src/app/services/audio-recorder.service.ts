import { Injectable, signal } from '@angular/core';

export interface RecordingState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioBlob: Blob | null;
}

@Injectable({
  providedIn: 'root'
})
export class AudioRecorderService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private startTime: number = 0;
  private timerInterval: any = null;

  // Signals for reactive state
  readonly isRecording = signal(false);
  readonly isPaused = signal(false);
  readonly duration = signal(0);
  readonly audioBlob = signal<Blob | null>(null);
  readonly error = signal<string | null>(null);

  async startRecording(): Promise<void> {
    try {
      this.error.set(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      // Create MediaRecorder
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.audioChunks = [];
      this.startTime = Date.now();

      // Handle data available
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      // Handle recording stop
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
        this.audioBlob.set(blob);
        this.stopTimer();

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      // Start recording
      this.mediaRecorder.start();
      this.isRecording.set(true);
      this.startTimer();

    } catch (err) {
      console.error('Error starting recording:', err);
      this.error.set('Failed to access microphone. Please check permissions.');
      throw err;
    }
  }

  stopRecording(): void {
    if (this.mediaRecorder && this.isRecording()) {
      this.mediaRecorder.stop();
      this.isRecording.set(false);
      this.isPaused.set(false);
    }
  }

  pauseRecording(): void {
    if (this.mediaRecorder && this.isRecording() && !this.isPaused()) {
      this.mediaRecorder.pause();
      this.isPaused.set(true);
      this.stopTimer();
    }
  }

  resumeRecording(): void {
    if (this.mediaRecorder && this.isRecording() && this.isPaused()) {
      this.mediaRecorder.resume();
      this.isPaused.set(false);
      this.startTimer();
    }
  }

  reset(): void {
    this.stopRecording();
    this.audioChunks = [];
    this.audioBlob.set(null);
    this.duration.set(0);
    this.error.set(null);
  }

  private startTimer(): void {
    this.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      this.duration.set(elapsed);
    }, 100);
  }

  private stopTimer(): void {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  getFormattedDuration(): string {
    const seconds = this.duration();
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
}
