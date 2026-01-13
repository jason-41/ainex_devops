#!/usr/bin/env python3
#!/home/wbo/hrs_vision_env/bin/python3
"""
File: openai_client.py

Purpose:
This file provides a lightweight wrapper around the OpenAI Responses API,
encapsulating model selection and message-based chat interaction.

The client is designed to be used by higher-level ROS2 nodes to send structured
conversation histories to an LLM backend and retrieve plain text responses.

Structure:
- OpenAIChatClient class:
  - Initializes the OpenAI API client
  - Sends chat-style message lists to the LLM
  - Extracts and returns assistant text from the API response
"""

from openai import OpenAI


class OpenAIChatClient:
    """
    Client wrapper for interacting with the OpenAI chat completion interface.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the OpenAI chat client.

        Purpose:
        - Create an OpenAI API client instance
        - Store the target language model name for later requests

        Inputs:
        - model (str): Name of the LLM model to use

        Outputs:
        - None
        """
        self.client = OpenAI()
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        """
        Send a list of chat messages to the LLM and return the assistant response.

        Purpose:
        - Forward a structured conversation history to the OpenAI Responses API
        - Collect and concatenate all assistant output text segments

        Inputs:
        - messages (list[dict]): Conversation messages in the following format:
          [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]

        Outputs:
        - str: Combined assistant response text
        """

        # Send request to OpenAI Responses API
        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        # Extract assistant text content from the response structure
        texts = []
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        texts.append(c.text)

        # Return concatenated assistant response
        return "\n".join(texts).strip()



