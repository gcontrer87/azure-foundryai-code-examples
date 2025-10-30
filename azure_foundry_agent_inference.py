#!/usr/bin/env python3
"""
Azure FoundryAI Agent Inference Script

This script invokes existing Azure AI Foundry agents to perform inference
through the AIProjectClient with Azure-managed identity authentication.
"""

import json
import argparse
import sys
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


def create_agent_client(endpoint):
    """
    Create an Azure FoundryAI agent client.
    
    Args:
        endpoint: The Azure FoundryAI project endpoint URL
    
    Returns:
        AIProjectClient for making agent requests
    """
    try:
        credential = DefaultAzureCredential()
        client = AIProjectClient(
            endpoint=endpoint,
            credential=credential
        )

        return client
        
    except Exception as e:
        print(f"Error creating client: {e}")
        raise


def make_agent_inference_request(endpoint, agent_id, prompt):
    """
    Invoke an existing Azure FoundryAI agent to make an inference request.
    
    Args:
        endpoint: The Azure FoundryAI project endpoint URL
        agent_id: The ID of the agent to invoke
        prompt: The input prompt for the agent
    
    Returns:
        Dictionary containing the response from the agent
    """
    try:
        # Create the agent client
        client = create_agent_client(endpoint)
        
        # Create a new thread for this conversation
        thread = client.agents.threads.create()
        
        # Add the user message to the thread
        client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=[{"type": "text", "text": prompt}]
        )
        
        # Run the thread with the specified agent
        run = client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent_id
        )
        
        # Wait for the run to complete and retrieve messages
        # The run should process automatically, but we'll list messages to get the response
        messages = client.agents.messages.list(thread_id=thread.id)
        
        # Build result structure
        result = {
            "thread_id": thread.id,
            "run_id": run.id if hasattr(run, 'id') else None,
            "messages": []
        }
        
        # Extract agent responses
        agent_responses = []
        for msg in messages:
            message_dict = {
                "role": msg.role,
                "content": []
            }
            
            if hasattr(msg, 'content') and msg.content:
                for content in msg.content:
                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                        message_dict["content"].append(content.text.value)
                
                # If we found agent responses, add them
                if message_dict["role"] == "assistant" and message_dict["content"]:
                    agent_responses.extend(message_dict["content"])
            
            result["messages"].append(message_dict)
        
        # Add the combined agent response for easy access
        result["agent_response"] = "\n".join(agent_responses) if agent_responses else ""
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def list_agents(endpoint):
    """
    List available agents in the project.
    
    Args:
        endpoint: The Azure FoundryAI project endpoint URL
    
    Returns:
        List of available agents
    """
    try:
        # Create the agent client
        client = create_agent_client(endpoint)
        
        # List all agents in the project
        agents_list = client.agents.list_agents()
        
        agents = []
        for agent in agents_list:
            agent_info = {
                "id": agent.id,
                "name": agent.name if hasattr(agent, 'name') else "Unknown",
                "description": agent.description if hasattr(agent, 'description') else ""
            }
            agents.append(agent_info)
        
        return agents
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    """Main function to run the agent inference script."""
    parser = argparse.ArgumentParser(
        description='Invoke existing Azure FoundryAI agents for inference'
    )
    parser.add_argument(
        '--endpoint',
        required=True,
        help='Azure FoundryAI project endpoint URL (e.g., https://your-project.services.ai.azure.com/api/projects/your-project)'
    )
    parser.add_argument(
        '--agent-id',
        required=True,
        help='Agent ID to invoke'
    )
    parser.add_argument(
        '--prompt',
        default='Hello, how are you?',
        help='Input prompt for the agent (default: "Hello, how are you?")'
    )
    # Authentication is via Azure AD using DefaultAzureCredential
    parser.add_argument(
        '--list-agents',
        action='store_true',
        help='List available agents and exit'
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_agents:
            print("Fetching available agents...")
            agents = list_agents(args.endpoint)
            print("\nAvailable agents:")
            for agent in agents:
                print(f"- {agent['name']} (ID: {agent['id']})")
                if agent['description']:
                    print(f"  Description: {agent['description']}")
            return
        
        # Make inference request
        print(f"Invoking agent: {args.agent_id}")
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        
        response = make_agent_inference_request(
            endpoint=args.endpoint,
            agent_id=args.agent_id,
            prompt=args.prompt
        )
        
        # Display the response
        print("Response:")
        print(json.dumps(response, indent=2))
        
        # Extract and display the agent's response
        if 'agent_response' in response and response['agent_response']:
            print("\nAgent Response:")
            print(response['agent_response'])
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
