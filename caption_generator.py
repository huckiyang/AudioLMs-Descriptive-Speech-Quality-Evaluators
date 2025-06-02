#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech Quality Descriptive Caption Generator
This script generates descriptive captions for speech quality evaluation using LLaMA-3.1 70B model,
following the paper "Audio Large Language Models Can Be Descriptive Speech Quality Evaluators".
It supports both individual speech quality evaluation (MOS prediction) and A/B testing between two speech samples.
"""

import os
import argparse
import json
from typing import Dict, List, Tuple, Union, Optional

# Import the audio analyzer
from audio_analyzer import analyze_audio

# You'll need to replace this with your actual LLaMA API implementation
# This is a placeholder for the LLaMA API call
def call_llama_api(prompt: str, temperature: float = 1.0, top_p: float = 0.9) -> str:
    """
    Call the LLaMA-3.1 70B model with the given prompt.
    
    Args:
        prompt: The input prompt for the model
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        
    Returns:
        The model's response as a string
    """
    # Replace with actual API call to LLaMA-3.1 70B
    print(f"[DEBUG] Sending prompt to LLaMA-3.1 70B with temperature={temperature}, top_p={top_p}")
    print(f"[DEBUG] Prompt: {prompt}")
    
    # This is where you'd implement the actual API call
    # For example:
    # from llama_api import generate_text
    # response = generate_text(prompt, temperature=temperature, top_p=top_p)
    # return response
    
    return "Placeholder LLaMA-3.1 70B response"

def generate_mos_prediction_prompt(metadata: Dict[str, float], example_data: Optional[Dict] = None, example_response: Optional[str] = None) -> str:
    """
    Generate a prompt for MOS prediction based on the metadata.
    
    Args:
        metadata: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values
        example_data: Optional example data point to include in the prompt
        example_response: Optional example response to include in the prompt
        
    Returns:
        The formatted prompt string
    """
    prompt = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are rating from 1 to 5. For all these factors, higher is better.
(1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
(2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
(3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
(4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
(5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
I need you to generate a descriptive evaluation for this speech, including a description according to the score from (2) to (5), analyze how they influence the overall quality, and add the mos in the end."""

    # Add example if provided
    if example_data and example_response:
        prompt += f"\nFor example, input is {json.dumps(example_data)}, then you should output: {example_response}"
    
    # Add current data point
    prompt += f"\nNow the input is {json.dumps(metadata)}. Please only output the evaluation:"
    
    return prompt

def generate_ab_test_prompt(metadata_a: Dict[str, float], metadata_b: Dict[str, float]) -> str:
    """
    Generate a prompt for A/B testing based on the metadata of two speech samples.
    
    Args:
        metadata_a: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values for Speech A
        metadata_b: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values for Speech B
        
    Returns:
        The formatted prompt string
    """
    prompt = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are rating from 1 to 5. For all these factors, higher is better.
(1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
(2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
(3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
(4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
(5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
I need you to perform A/B test according to their mos (mos higher means winner). You can flexibly select 1~3 aspects from (2)~(5) with an obvious gap (usually score difference more than 0.5), then compare them according to these distinctions. Finally, please give your preference with a reasonable analysis."""

    # Add metadata for both speech samples
    prompt += f"\nSpeechA: {json.dumps(metadata_a)}"
    prompt += f"\nSpeechB: {json.dumps(metadata_b)}"
    prompt += "\nPlease provide your comparison and determine which speech is better:"
    
    return prompt

def summarize_ab_test(llm_output: str) -> str:
    """
    Summarize the A/B test result using LLaMA-3.1 70B.
    
    Args:
        llm_output: The output from the A/B test generation
        
    Returns:
        A string with either "[SpeechA]" or "[SpeechB]"
    """
    prompt = f"""According to the context, please judge if SpeechA is better or SpeechB is better. Only output '[SpeechA]' or '[SpeechB]', do not give any analysis.
Context:
{llm_output}"""
    
    result = call_llama_api(prompt, temperature=0.7, top_p=1.0)
    return result.strip()

def generate_captions(audio_path_1: str, audio_path_2: str, output_dir: str, run_ab_test: bool = True):
    """
    Generate captions for two audio files, including individual MOS predictions and optionally an A/B test.
    
    Args:
        audio_path_1: Path to the first audio file
        audio_path_2: Path to the second audio file
        output_dir: Directory to save the generated captions
        run_ab_test: Whether to run an A/B test comparing the two audio files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze the audio files to extract quality metrics
    print(f"Analyzing audio file 1: {audio_path_1}")
    metadata_1 = analyze_audio(audio_path_1)
    
    print(f"Analyzing audio file 2: {audio_path_2}")
    metadata_2 = analyze_audio(audio_path_2)
    
    # Save the extracted metrics
    with open(os.path.join(output_dir, "audio1_metrics.json"), "w") as f:
        json.dump(metadata_1, f, indent=2)
    
    with open(os.path.join(output_dir, "audio2_metrics.json"), "w") as f:
        json.dump(metadata_2, f, indent=2)
    
    # Generate MOS prediction for audio 1
    print("Generating MOS prediction for audio file 1...")
    mos_prompt_1 = generate_mos_prediction_prompt(metadata_1)
    mos_result_1 = call_llama_api(mos_prompt_1)
    
    # Generate MOS prediction with higher diversity for audio 1
    print("Generating diverse MOS prediction for audio file 1...")
    mos_result_1_diverse = call_llama_api(mos_prompt_1, temperature=1.1, top_p=0.9)
    
    # Generate MOS prediction for audio 2
    print("Generating MOS prediction for audio file 2...")
    mos_prompt_2 = generate_mos_prediction_prompt(metadata_2)
    mos_result_2 = call_llama_api(mos_prompt_2)
    
    # Generate MOS prediction with higher diversity for audio 2
    print("Generating diverse MOS prediction for audio file 2...")
    mos_result_2_diverse = call_llama_api(mos_prompt_2, temperature=1.1, top_p=0.9)
    
    # Save individual results
    with open(os.path.join(output_dir, "audio1_mos.txt"), "w") as f:
        f.write(mos_result_1)
    
    with open(os.path.join(output_dir, "audio1_mos_diverse.txt"), "w") as f:
        f.write(mos_result_1_diverse)
    
    with open(os.path.join(output_dir, "audio2_mos.txt"), "w") as f:
        f.write(mos_result_2)
    
    with open(os.path.join(output_dir, "audio2_mos_diverse.txt"), "w") as f:
        f.write(mos_result_2_diverse)
    
    # Run A/B test if requested
    if run_ab_test:
        print("Running A/B test comparing both audio files...")
        ab_prompt = generate_ab_test_prompt(metadata_1, metadata_2)
        ab_result = call_llama_api(ab_prompt)
        
        # Summarize A/B test
        print("Summarizing A/B test result...")
        summary = summarize_ab_test(ab_result)
        
        # Save A/B test results
        with open(os.path.join(output_dir, "ab_test.txt"), "w") as f:
            f.write(ab_result)
        
        with open(os.path.join(output_dir, "ab_test_summary.txt"), "w") as f:
            f.write(summary)
    
    print(f"Caption generation complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate speech quality captions using LLaMA-3.1 70B")
    parser.add_argument("--audio1", required=True, help="Path to the first audio file")
    parser.add_argument("--audio2", required=True, help="Path to the second audio file")
    parser.add_argument("--output", default="./output", help="Output directory for captions")
    parser.add_argument("--skip-ab-test", action="store_true", help="Skip A/B test")
    parser.add_argument("--example-data", help="Path to a JSON file with example data point")
    parser.add_argument("--example-response", help="Path to a file with example response")
    
    args = parser.parse_args()
    
    generate_captions(args.audio1, args.audio2, args.output, not args.skip_ab_test)

if __name__ == "__main__":
    main() 
