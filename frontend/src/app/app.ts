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
  readonly lastProcessingTime = signal<string | null>(null);
  readonly lastMetadata = signal<any>(null);
  readonly lastTimings = signal<any>(null);
  readonly uploadedFile = signal<File | null>(null);
  readonly uploadedBlob = signal<Blob | null>(null);
  readonly audioSource = signal<'recording' | 'upload' | null>(null);

  // Expose recorder state
  readonly isRecording = this.recorderService.isRecording;
  readonly isPaused = this.recorderService.isPaused;
  readonly duration = this.recorderService.duration;
  readonly recordingError = this.recorderService.error;

  // Computed values
  readonly canTranscribe = computed(() =>
    !this.isRecording() && (this.recorderService.audioBlob() !== null || this.uploadedBlob() !== null)
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
    { code: 'danish', name: 'Danish' },
    { code: 'russian', name: 'Russian' },
    { code: 'chinese', name: 'Chinese (Simplified)' },
    { code: 'japanese', name: 'Japanese' },
    { code: 'korean', name: 'Korean' },
    { code: 'arabic', name: 'Arabic' },
    { code: 'hindi', name: 'Hindi' },
  ];

  // Available models
  readonly models = [
    { code: 'ctc_300m', name: 'CTC 300M (Fastest)' },
    { code: 'ctc_1b', name: 'CTC 1B (Fast, Recommended)' },
    { code: 'ctc_3b', name: 'CTC 3B (Better)' },
    { code: 'ctc_7b', name: 'CTC 7B (Best CTC)' },
    { code: 'llm_300m', name: 'LLM 300M (Compact, Language-aware)' },
    { code: 'llm_1b', name: 'LLM 1B (Balanced, Language-aware)' },
    { code: 'llm_3b', name: 'LLM 3B (High-quality, Language-aware)' },
    { code: 'llm_7b', name: 'LLM 7B (Best quality, ~17GB VRAM)' },
    { code: 'llm_7b_zs', name: 'LLM 7B Zero-shot (Multilingual)' },
  ];

  async startRecording(): Promise<void> {
    try {
      this.transcription.set('');
      this.apiError.set(null);
      this.clearUpload();
      this.audioSource.set('recording');
      await this.recorderService.startRecording();
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];

      // Validate file type
      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/webm', 'audio/m4a', 'audio/x-m4a'];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|flac|ogg|webm|m4a)$/i)) {
        this.apiError.set('Invalid file type. Please upload an audio file (WAV, MP3, FLAC, OGG, WebM, M4A).');
        return;
      }

      // Validate file size (50MB limit)
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (file.size > maxSize) {
        this.apiError.set('File too large. Maximum size is 50MB.');
        return;
      }

      this.uploadedFile.set(file);
      this.uploadedBlob.set(file);
      this.audioSource.set('upload');
      this.transcription.set('');
      this.apiError.set(null);
      this.recorderService.reset();
    }
  }

  clearUpload(): void {
    this.uploadedFile.set(null);
    this.uploadedBlob.set(null);
    if (this.audioSource() === 'upload') {
      this.audioSource.set(null);
    }
  }

  switchToUpload(): void {
    this.recorderService.reset();
    this.audioSource.set('upload');
    this.transcription.set('');
    this.apiError.set(null);
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
    // Get audio from either recording or upload
    const audioBlob = this.audioSource() === 'upload'
      ? this.uploadedBlob()
      : this.recorderService.audioBlob();

    if (!audioBlob) {
      return;
    }

    this.isTranscribing.set(true);
    this.apiError.set(null);
    this.transcription.set('');
    this.lastProcessingTime.set(null);
    this.lastMetadata.set(null);
    this.lastTimings.set(null);

    // Track client-side timings
    const clientTimings: any = {};
    const totalStart = performance.now();

    try {
      // Track upload time
      const uploadStart = performance.now();

      const result = await this.apiService.transcribe(
        audioBlob,
        this.selectedModel(),
        this.selectedLanguage()
      ).toPromise();

      const uploadEnd = performance.now();
      clientTimings.network_upload = ((uploadEnd - uploadStart) / 1000).toFixed(3) + 's';

      if (result) {
        const totalEnd = performance.now();
        clientTimings.total_client = ((totalEnd - totalStart) / 1000).toFixed(3) + 's';

        this.transcription.set(result.transcription);
        this.lastProcessingTime.set(result.metadata.processing_time);
        this.lastMetadata.set(result.metadata);

        // Combine client and server timings
        this.lastTimings.set({
          ...clientTimings,
          ...result.timings
        });
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
    this.clearUpload();
    this.audioSource.set(null);
    this.transcription.set('');
    this.apiError.set(null);
    this.lastProcessingTime.set(null);
    this.lastMetadata.set(null);
    this.lastTimings.set(null);
  }

  copyTranscription(): void {
    const text = this.transcription();
    if (text) {
      navigator.clipboard.writeText(text);
    }
  }
}
