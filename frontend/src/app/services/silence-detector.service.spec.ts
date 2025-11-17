import { TestBed } from '@angular/core/testing';
import { SilenceDetectorService } from './silence-detector.service';

describe('SilenceDetectorService', () => {
  let service: SilenceDetectorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SilenceDetectorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should detect silence after threshold', (done) => {
    const silenceThreshold = 10; // Low threshold for testing
    const silenceDuration = 200; // 200ms

    service.silenceDetected$.subscribe(() => {
      expect(true).toBe(true);
      done();
    });

    // Simulate low audio levels
    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: (array: Uint8Array) => {
        array.fill(5); // Below threshold
      }
    };

    service.startMonitoring(mockAnalyser as any, silenceThreshold, silenceDuration);
  });

  it('should reset silence timer on sound', () => {
    const silenceThreshold = 10;
    const silenceDuration = 500;

    let silenceDetected = false;
    service.silenceDetected$.subscribe(() => {
      silenceDetected = true;
    });

    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: jasmine.createSpy('getByteFrequencyData')
    };

    // First call: silence
    (mockAnalyser.getByteFrequencyData as jasmine.Spy).and.callFake((array: Uint8Array) => {
      array.fill(5);
    });

    service.startMonitoring(mockAnalyser as any, silenceThreshold, silenceDuration);

    // Second call: sound (should reset timer)
    setTimeout(() => {
      (mockAnalyser.getByteFrequencyData as jasmine.Spy).and.callFake((array: Uint8Array) => {
        array.fill(50); // Above threshold
      });
    }, 100);

    setTimeout(() => {
      expect(silenceDetected).toBe(false);
    }, 600);
  });

  it('should stop monitoring', () => {
    const mockAnalyser = {
      frequencyBinCount: 1024,
      getByteFrequencyData: jasmine.createSpy('getByteFrequencyData')
    };

    service.startMonitoring(mockAnalyser as any, 10, 500);
    service.stopMonitoring();

    expect((mockAnalyser.getByteFrequencyData as jasmine.Spy).calls.count()).toBeLessThan(10);
  });
});
