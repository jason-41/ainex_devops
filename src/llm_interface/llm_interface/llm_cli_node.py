#!/usr/bin/env python3
"""
File: llm_cli_node.py

Purpose:
This file implements a command-line interface (CLI) ROS2 node that allows
a user to interact with an LLM service via text input from the terminal.

Structure:
- LLMCLINode class:
  - Initializes a ROS2 node and service client for the /llm/chat service
  - Sends user text requests to the LLM service and receives responses
- main function:
  - Handles CLI input loop
  - Parses CLI-specific commands (reset, session switch, exit)
  - Forwards normal text input to the LLM service
"""

import rclpy
from rclpy.node import Node
from llm_msgs.srv import Chat


class LLMCLINode(Node):
    """
    ROS2 node providing a terminal-based interface to the LLM chat service.
    """

    def __init__(self):
        """
        Initialize the LLM CLI node.

        Purpose:
        - Create a ROS2 node
        - Initialize a service client for the /llm/chat service
        - Wait until the LLM service becomes available
        - Set default session information and print usage instructions

        Inputs:
        - None

        Outputs:
        - None
        """
        super().__init__("llm_cli_node")

        # Create a client for the LLM chat service
        self.client = self.create_client(Chat, "/llm/chat")

        # Wait for the LLM service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /llm/chat service...")

        # Initialize default session ID for CLI interaction
        self.session_id = "terminal"

        # Log basic CLI usage information
        self.get_logger().info(f"[LLM CLI] session={self.session_id}")
        self.get_logger().info("Type text and press Enter.")
        self.get_logger().info("Commands: /reset, /session <name>, /exit")

    def call_llm(self, text: str, reset: bool = False) -> str:
        """
        Send a request to the LLM service and return the assistant response.

        Purpose:
        - Package user input into a Chat service request
        - Call the LLM service asynchronously
        - Wait for and return the LLM response

        Inputs:
        - text (str): User input text to send to the LLM
        - reset (bool): Whether to reset the conversation session

        Outputs:
        - str: Assistant response text, or an error message if the call fails
        """

        # Construct service request
        req = Chat.Request()
        req.session_id = self.session_id
        req.user_text = text
        req.reset = reset

        # Call the service asynchronously
        future = self.client.call_async(req)

        # Block until the service call completes
        rclpy.spin_until_future_complete(self, future)

        # Handle failure case
        if future.result() is None:
            return "[ERROR] No response from LLM service"

        # Return assistant reply text
        return future.result().assistant_text


def main():
    """
    Entry point for the LLM CLI node.

    Purpose:
    - Initialize ROS2
    - Create and run the LLM CLI node
    - Handle user input from the terminal
    - Parse CLI-specific commands and normal chat messages

    Inputs:
    - None

    Outputs:
    - None
    """

    # Initialize ROS2 context
    rclpy.init()

    # Create CLI node instance
    node = LLMCLINode()

    try:
        # Main CLI interaction loop
        while rclpy.ok():
            user_input = input("> ").strip()

            # Ignore empty input
            if not user_input:
                continue

            # ---------- CLI-only commands ----------

            # Exit command
            if user_input == "/exit":
                print("[LLM CLI] exit")
                break

            # Reset current LLM session
            if user_input == "/reset":
                node.call_llm("", reset=True)
                print("[LLM CLI] session reset")
                continue

            # Switch session ID
            if user_input.startswith("/session"):
                parts = user_input.split(maxsplit=1)
                if len(parts) != 2:
                    print("[LLM CLI] usage: /session <name>")
                    continue
                node.session_id = parts[1]
                print(f"[LLM CLI] switched to session={node.session_id}")
                continue

            # ---------- normal chat ----------

            # Send user input to LLM service
            reply = node.call_llm(user_input)
            # print(f"[DEBUG] raw reply: {repr(reply)}")

            # ---- lock-aware behavior ----
            # If LLM service reports a locked state, notify the user
            if "LLM is locked" in reply or "locked" in reply.lower():
                print("[LLM CLI]: Waiting for face authentication, no access granted")
                continue

            # Print assistant reply
            print(f"LLM: {reply}")

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print("\n[LLM CLI] interrupted")

    # Clean up ROS2 resources
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
