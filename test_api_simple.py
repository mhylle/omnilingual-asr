#!/usr/bin/env python3
"""
Simple API Test

Quick test to verify the API is working correctly.
Run this after starting the API server.
"""

import requests
import sys
from pathlib import Path


def test_api():
    """Test basic API functionality."""
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("API Quick Test")
    print("=" * 60)

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Make sure the server is running: python api.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test 2: List models
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint passed")
            print(f"   Available models: {len(data['models'])}")
            print(f"   Default: {data['default']}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

    # Test 3: List languages
    print("\n3. Testing languages endpoint...")
    try:
        response = requests.get(f"{base_url}/languages")
        if response.status_code == 200:
            data = response.json()
            print("✅ Languages endpoint passed")
            print(f"   Total supported: {data['total_supported']}")
        else:
            print(f"❌ Languages endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Languages endpoint error: {e}")
        return False

    # Test 4: API info
    print("\n4. Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Info endpoint passed")
            print(f"   API: {data['api']['name']}")
            print(f"   Version: {data['api']['version']}")
        else:
            print(f"❌ Info endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Info endpoint error: {e}")
        return False

    # Test 5: Transcription (if test file exists)
    print("\n5. Testing transcription endpoint...")
    test_file = Path("recording.wav")
    if test_file.exists():
        try:
            with open(test_file, "rb") as f:
                files = {"file": f}
                params = {"model": "ctc_1b", "language": "english"}
                response = requests.post(
                    f"{base_url}/transcribe",
                    files=files,
                    params=params,
                    timeout=60
                )

            if response.status_code == 200:
                data = response.json()
                print("✅ Transcription endpoint passed")
                print(f"   Transcription: {data['transcription'][:50]}...")
                print(f"   Processing time: {data['metadata']['processing_time']}")
            else:
                print(f"❌ Transcription failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return False
    else:
        print("⚠️  Skipping transcription test (no test file)")
        print("   Create one with: python main.py record --duration 5")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
