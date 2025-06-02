#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Quality Analyzer
This script analyzes audio files to extract quality metrics that can be used
for descriptive caption generation. It computes MOS (Mean Opinion Score) and
sub-dimensions of audio quality: noise level, coloration, discontinuity, and loudness.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, List, Optional

def load_audio(audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        audio_path: Path to the audio file
        sr: Target sample rate (if None, uses the native sample rate)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        raise

def estimate_noise_level(audio_data: np.ndarray, sr: int) -> float:
    """
    Estimate the noise level in the audio (1-5 scale, higher is better/cleaner).
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        
    Returns:
        Noise level score (1-5)
    """
    # Simple noise estimation using signal-to-noise ratio
    # This is a placeholder implementation - a real system would use a more sophisticated method
    
    # Compute signal power
    signal_power = np.mean(audio_data**2)
    
    # Estimate noise using a simple voice activity detection
    # and measuring power in non-speech segments
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Extract frames
    frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sum(frames**2, axis=0)
    
    # Normalize energies to 0-1
    norm_energies = (frame_energies - np.min(frame_energies)) / (np.max(frame_energies) - np.min(frame_energies) + 1e-10)
    
    # Simple VAD - frames with energy below threshold are considered non-speech
    threshold = 0.2
    noise_frames = frames[:, norm_energies < threshold]
    
    if noise_frames.size > 0:
        noise_power = np.mean(noise_frames**2)
    else:
        # If no noise frames detected, assume low noise
        noise_power = signal_power * 0.01
    
    # Calculate SNR
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = 100  # Very high SNR if no noise detected
    
    # Map SNR to 1-5 scale
    # These thresholds are arbitrary and should be calibrated on real data
    if snr < 10:
        return 1.0  # Very noisy
    elif snr < 15:
        return 2.0  # Somewhat noisy
    elif snr < 20:
        return 3.0  # Average
    elif snr < 30:
        return 4.0  # Somewhat clean
    else:
        return 5.0  # Very clean

def estimate_coloration(audio_data: np.ndarray, sr: int) -> float:
    """
    Estimate the coloration/distortion level (1-5 scale, higher is better/less distorted).
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        
    Returns:
        Coloration score (1-5)
    """
    # Measure spectral flatness as a proxy for coloration
    # Higher spectral flatness often indicates less coloration
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
    mean_flatness = np.mean(spectral_flatness)
    
    # Measure harmonic-to-noise ratio
    # Higher harmonic ratio often indicates less distortion
    harmonics = librosa.effects.harmonic(audio_data)
    hnr = np.mean(harmonics**2) / (np.mean(audio_data**2) + 1e-10)
    
    # Combine metrics
    # This approach is simplified and should be calibrated with real data
    coloration_score = 2.5 + (mean_flatness * 5) + (hnr * 10)
    
    # Clip to 1-5 range
    return max(1.0, min(5.0, coloration_score))

def estimate_discontinuity(audio_data: np.ndarray, sr: int) -> float:
    """
    Estimate the discontinuity in the audio (1-5 scale, higher is better/more continuous).
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        
    Returns:
        Discontinuity score (1-5)
    """
    # Look for sudden changes in energy as a signal for discontinuities
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Compute energy per frame
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Compute energy derivative
    energy_diff = np.abs(np.diff(energy))
    
    # Normalize differences
    if len(energy_diff) > 0 and np.max(energy_diff) > 0:
        norm_diff = energy_diff / np.max(energy_diff)
    else:
        # No discontinuities detected
        return 5.0
    
    # Count significant jumps
    significant_jumps = np.sum(norm_diff > 0.5)
    
    # Calculate discontinuity score based on jump density
    jump_density = significant_jumps / len(energy_diff) if len(energy_diff) > 0 else 0
    
    # Map jump density to 1-5 scale (inverted, fewer jumps = higher score)
    if jump_density > 0.1:
        return 1.0  # Severe discontinuities
    elif jump_density > 0.05:
        return 2.0  # Significant discontinuities
    elif jump_density > 0.02:
        return 3.0  # Moderate discontinuities
    elif jump_density > 0.01:
        return 4.0  # Minor discontinuities
    else:
        return 5.0  # No discontinuities

def estimate_loudness(audio_data: np.ndarray, sr: int) -> float:
    """
    Estimate the perceived loudness (1-5 scale, higher is better/optimal loudness).
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        
    Returns:
        Loudness score (1-5)
    """
    # Compute RMS energy as a proxy for loudness
    rms = np.sqrt(np.mean(audio_data**2))
    
    # Map RMS to dB
    if rms > 0:
        db = 20 * np.log10(rms)
    else:
        db = -100  # Very quiet
    
    # Normalized loudness score (higher values = better)
    # These thresholds are arbitrary and should be calibrated
    if db < -40:
        return 1.0  # Extremely quiet
    elif db < -30:
        return 2.0  # Significantly quiet
    elif db < -20:
        return 3.0  # Moderate volume
    elif db < -10:
        return 4.0  # Good volume
    elif db < -3:
        return 5.0  # Optimal volume
    else:
        # Penalty for being too loud/potential clipping
        return max(1.0, 5.0 - (db + 3) * 0.5)

def estimate_overall_quality(noi: float, col: float, dis: float, loud: float) -> float:
    """
    Estimate the overall MOS (Mean Opinion Score) based on sub-dimensions.
    
    Args:
        noi: Noise level score (1-5)
        col: Coloration score (1-5)
        dis: Discontinuity score (1-5)
        loud: Loudness score (1-5)
        
    Returns:
        MOS score (1-5)
    """
    # Simple weighted average
    # These weights should be calibrated on real data
    weights = {
        'noi': 0.3,
        'col': 0.3,
        'dis': 0.3,
        'loud': 0.1
    }
    
    mos = (weights['noi'] * noi + 
           weights['col'] * col + 
           weights['dis'] * dis + 
           weights['loud'] * loud)
    
    # Round to one decimal place
    return round(mos, 1)

def analyze_audio(audio_path: str) -> Dict[str, float]:
    """
    Analyze an audio file and return quality metrics.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary with quality metrics: 'mos', 'noi', 'col', 'dis', 'loud'
    """
    # Load the audio
    audio_data, sr = load_audio(audio_path)
    
    # Extract metrics
    noi = estimate_noise_level(audio_data, sr)
    col = estimate_coloration(audio_data, sr)
    dis = estimate_discontinuity(audio_data, sr)
    loud = estimate_loudness(audio_data, sr)
    
    # Calculate overall MOS
    mos = estimate_overall_quality(noi, col, dis, loud)
    
    return {
        'mos': mos,
        'noi': noi,
        'col': col,
        'dis': dis,
        'loud': loud
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze audio quality metrics for speech files")
    parser.add_argument("audio_path", help="Path to the audio file to analyze")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file {args.audio_path} not found")
        return
    
    metrics = analyze_audio(args.audio_path)
    
    print("Audio Quality Metrics:")
    print(f"Overall MOS:     {metrics['mos']:.1f}")
    print(f"Noise Level:     {metrics['noi']:.1f}")
    print(f"Coloration:      {metrics['col']:.1f}")
    print(f"Discontinuity:   {metrics['dis']:.1f}")
    print(f"Loudness:        {metrics['loud']:.1f}")

if __name__ == "__main__":
    main() 
