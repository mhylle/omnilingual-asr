import { Component, signal, inject, computed } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AudioRecorderService } from './services/audio-recorder.service';
import { AsrApiService } from './services/asr-api.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  private readonly recorderService = inject(AudioRecorderService);
  private readonly apiService = inject(AsrApiService);

  // State signals
  readonly title = signal('Omnilingual ASR');
  readonly transcription = signal<string>('');
  readonly isTranscribing = signal(false);
  readonly apiError = signal<string | null>(null);
  readonly selectedLanguage = signal<string>('english');
  readonly selectedModel = signal<string>('ctc_1b');

  // Expose recorder state
  readonly isRecording = this.recorderService.isRecording;
  readonly isPaused = this.recorderService.isPaused;
  readonly duration = this.recorderService.duration;
  readonly recordingError = this.recorderService.error;

  // Computed values
  readonly canTranscribe = computed(() =>
    !this.isRecording() && this.recorderService.audioBlob() !== null
  );

  readonly isProcessing = computed(() =>
    this.isRecording() || this.isTranscribing()
  );

  readonly formattedDuration = computed(() =>
    this.recorderService.getFormattedDuration()
  );

  readonly durationWarning = computed(() =>
    this.duration() > 35 // Warning at 35 seconds
  );

  readonly durationCritical = computed(() =>
    this.duration() >= 40 // Critical at 40 seconds
  );

  // Available languages (subset from API)
  readonly languages = [
    { code: 'english', name: 'English' },
    { code: 'spanish', name: 'Spanish' },
    { code: 'french', name: 'French' },
    { code: 'german', name: 'German' },
    { code: 'italian', name: 'Italian' },
    { code: 'portuguese', name: 'Portuguese' },
    { code: 'russian', name: 'Russian' },
    { code: 'chinese', name: 'Chinese (Simplified)' },
    { code: 'japanese', name: 'Japanese' },
    { code: 'korean', name: 'Korean' },
    { code: 'arabic', name: 'Arabic' },
    { code: 'hindi', name: 'Hindi' },
  ];

  // Available models
  readonly models = [
    { code: 'ctc_1b', name: 'CTC 1B (Fast, Recommended)' },
    { code: 'ctc_300m', name: 'CTC 300M (Fastest)' },
    { code: 'ctc_3b', name: 'CTC 3B (Better)' },
    { code: 'llm_1b', name: 'LLM 1B (Language-aware)' },
    { code: 'llm_3b', name: 'LLM 3B (Best)' },
  ];

  async startRecording(): Promise<void> {
    try {
      this.transcription.set('');
      this.apiError.set(null);
      await this.recorderService.startRecording();
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  }

  stopRecording(): void {
    this.recorderService.stopRecording();
  }

  pauseRecording(): void {
    this.recorderService.pauseRecording();
  }

  resumeRecording(): void {
    this.recorderService.resumeRecording();
  }

  async transcribe(): Promise<void> {
    const audioBlob = this.recorderService.audioBlob();
    if (!audioBlob) {
      return;
    }

    this.isTranscribing.set(true);
    this.apiError.set(null);
    this.transcription.set('');

    try {
      const result = await this.apiService.transcribe(
        audioBlob,
        this.selectedModel(),
        this.selectedLanguage()
      ).toPromise();

      if (result) {
        this.transcription.set(result.transcription);
      }
    } catch (err: any) {
      console.error('Transcription failed:', err);
      this.apiError.set(
        err.error?.detail || 'Failed to transcribe audio. Please try again.'
      );
    } finally {
      this.isTranscribing.set(false);
    }
  }

  reset(): void {
    this.recorderService.reset();
    this.transcription.set('');
    this.apiError.set(null);
  }

  copyTranscription(): void {
    const text = this.transcription();
    if (text) {
      navigator.clipboard.writeText(text);
    }
  }
}
