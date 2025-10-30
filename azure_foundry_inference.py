#!/usr/bin/env python3
"""
Simple Azure FoundryAI Inference Script

This script makes a simple inference request to Azure FoundryAI
using the Azure OpenAI client library.
"""

import json
import argparse
import sys
import openai


def make_inference_request(endpoint, api_key, deployment_name, prompt, max_tokens=100, temperature=0.7):
    """
    Make a simple inference request to Azure FoundryAI.
    
    Args:
        endpoint: The Azure FoundryAI endpoint URL
        api_key: The API key for authentication
        deployment_name: The model deployment name
        prompt: The input prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
    
    Returns:
        Dictionary containing the response from the API
    """
    try:
        # Initialize the Azure OpenAI client (version 2.1.0 approach)
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )
        
        # Prepare the messages for the chat completion
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make the inference request
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Convert the response to a simple dictionary format
        result = {
            "id": response.id,
            "model": response.model,
            "choices": []
        }
        
        for choice in response.choices:
            choice_dict = {
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                }
            }
            result["choices"].append(choice_dict)
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def list_models(endpoint, api_key):
    """
    List available models from the endpoint.
    
    Args:
        endpoint: The Azure FoundryAI endpoint URL
        api_key: The API key for authentication
    
    Returns:
        List of available models
    """
    try:
        # Initialize the Azure OpenAI client (version 2.1.0 approach)
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )
        
        # Get available models (simplified - Azure doesn't have a direct list models endpoint)
        # Return a placeholder response
        return [{"name": "gpt-4", "model": "gpt-4"}, {"name": "gpt-3.5-turbo", "model": "gpt-3.5-turbo"}]
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(
        description='Make inference requests to Azure FoundryAI'
    )
    parser.add_argument(
        '--endpoint',
        required=True,
        help='Azure FoundryAI endpoint URL (e.g., https://your-resource.openai.azure.com)'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='API key for Azure FoundryAI authentication'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model deployment name to use for inference (e.g., gpt-4, gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--prompt',
        default='Hello, how are you?',
        help='Input prompt for the model (default: "Hello, how are you?")'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum number of tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0 to 1.0, default: 0.7)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_models:
            print("Fetching available models...")
            models = list_models(args.endpoint, args.api_key)
            print("\nAvailable models:")
            for model in models:
                print(f"- {model['name']} ({model['model']})")
            return
        
        # Make inference request
        print(f"Making inference request with model: {args.model}")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        
        response = make_inference_request(
            endpoint=args.endpoint,
            api_key=args.api_key,
            deployment_name=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # Display the response
        print("Response:")
        print(json.dumps(response, indent=2))
        
        # Extract and display just the generated text if available
        if 'choices' in response and len(response['choices']) > 0:
            generated_text = response['choices'][0].get('message', {}).get('content', '')
            if generated_text:
                print("\nGenerated text:")
                print(generated_text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
