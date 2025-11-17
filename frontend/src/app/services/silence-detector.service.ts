import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SilenceDetectorService {
  silenceDetected$ = new Subject<void>();

  private monitoringInterval: any;
  private silenceDuration = 0;
  private isMonitoring = false;

  constructor() { }

  /**
   * Start monitoring audio analyser for silence.
   *
   * @param analyser Web Audio API AnalyserNode
   * @param silenceThreshold Audio level below which is considered silence (0-255)
   * @param silenceDurationMs Duration of silence before triggering event (ms)
   */
  startMonitoring(
    analyser: AnalyserNode,
    silenceThreshold: number = 20,
    silenceDurationMs: number = 500
  ): void {
    if (this.isMonitoring) {
      this.stopMonitoring();
    }

    this.isMonitoring = true;
    this.silenceDuration = 0;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    this.monitoringInterval = setInterval(() => {
      if (!this.isMonitoring) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate average audio level
      const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;

      if (average < silenceThreshold) {
        this.silenceDuration += 100;

        if (this.silenceDuration >= silenceDurationMs) {
          this.silenceDetected$.next();
          this.silenceDuration = 0; // Reset after detection
        }
      } else {
        this.silenceDuration = 0; // Reset on sound
      }
    }, 100); // Check every 100ms
  }

  /**
   * Stop monitoring for silence.
   */
  stopMonitoring(): void {
    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.silenceDuration = 0;
  }

  /**
   * Get current silence duration in ms.
   */
  getCurrentSilenceDuration(): number {
    return this.silenceDuration;
  }
}
