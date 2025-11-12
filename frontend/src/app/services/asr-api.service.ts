import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface TranscriptionResponse {
  success: boolean;
  transcription: string;
  metadata: {
    filename: string;
    model: string;
    language: string | null;
    processing_time: string;
    timestamp: string;
  };
  timings: {
    upload_and_save: string;
    audio_conversion: string;
    model_loading: string;
    transcription: string;
    total_backend: string;
    model_was_cached: boolean;
  };
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model_loaded: boolean;
  current_model: string | null;
}

export interface ModelInfo {
  models: string[];
  details: {
    ctc: Record<string, any>;
    llm: Record<string, any>;
  };
  default: string;
}

@Injectable({
  providedIn: 'root'
})
export class AsrApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = 'http://localhost:8000';

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  getModels(): Observable<ModelInfo> {
    return this.http.get<ModelInfo>(`${this.baseUrl}/models`);
  }

  transcribe(
    audioBlob: Blob,
    model: string = 'ctc_1b',
    language: string | null = null
  ): Observable<TranscriptionResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    let params: any = { model };
    if (language) {
      params.language = language;
    }

    return this.http.post<TranscriptionResponse>(
      `${this.baseUrl}/transcribe`,
      formData,
      { params }
    );
  }
}
