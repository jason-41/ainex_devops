#!/usr/bin/env python3
"""
File: llm_server_node.py

Purpose:
This file implements a ROS2-based LLM server node that provides a chat service
for both terminal-based and speech-based human–robot interaction.

The node integrates:
- Authorization control based on face authentication
- Session-based conversation memory management
- Interaction with an OpenAI-compatible LLM backend
- Structured task extraction and persistent task saving
- Speech input and text-to-speech output interfaces

Structure:
- LLMServerNode class:
  - Initializes ROS2 services, subscriptions, and publishers
  - Manages authorization and session memory
  - Handles chat requests and speech input
  - Parses and saves structured task outputs
- main function:
  - Initializes and spins the ROS2 node
"""

import rclpy
from rclpy.node import Node
from llm_msgs.srv import Chat
from llm_interface.openai_client import OpenAIChatClient
from pathlib import Path
from auth_msgs.msg import AuthState
from std_msgs.msg import String
import re


class LLMServerNode(Node):
    """
    ROS2 node providing an LLM-backed chat server with authorization control,
    speech interaction, and task extraction capabilities.
    """

    def __init__(self):
        """
        Initialize the LLM server node.

        Purpose:
        - Create a ROS2 node and expose the /llm/chat service
        - Initialize authorization state and face authentication subscription
        - Load system prompt for LLM interaction
        - Initialize conversation memory and LLM client
        - Enable speech input and text output interfaces

        Inputs:
        - None

        Outputs:
        - None
        """
        super().__init__("llm_server_node")

        # Create LLM chat service
        self.srv = self.create_service(Chat, "/llm/chat", self.on_chat)
        self.get_logger().info("LLM Server ready at /llm/chat")

        # ---- authorization state ----
        self.authorized = False
        self.authorized_user = "Unknown"
        self.locked = True

        # Subscribe to face authentication state
        self.create_subscription(
            AuthState,
            "/auth/face_state",
            self.auth_callback,
            10
        )
        self.get_logger().info("Waiting for face authentication...")

        # Conversation memory: session_id -> list of message dicts
        self.memory = {}

        # Initialize LLM client
        self.client = OpenAIChatClient(model="gpt-4.1-mini")

        # Load system prompt from file
        prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {prompt_path}")

        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.get_logger().info("System prompt loaded successfully")

        # ---- speech interface ----

        # Subscribe to speech-to-text input
        self.speech_sub = self.create_subscription(
            String,
            "/speech/text_input",
            self.on_speech_input,
            10
        )

        # Publish text output for TTS
        self.speech_pub = self.create_publisher(
            String,
            "/speech/text_output",
            10
        )

        self.get_logger().info("Speech interface enabled.")

    def save_final_task(self, text: str):
        """
        Save the final structured task description to a file.

        Purpose:
        - Persist the extracted task information for later execution
        - Overwrite the latest task file with the most recent task

        Inputs:
        - text (str): Clean, structured task description

        Outputs:
        - None
        """

        # Create task directory if it does not exist
        task_dir = Path.home() / "ainex_tasks"
        task_dir.mkdir(exist_ok=True)

        # Define task file path
        task_file = task_dir / "latest_task.txt"

        # Write task content to file
        task_file.write_text(text, encoding="utf-8")

        self.get_logger().info(f"[TASK SAVED] {task_file}")

    def is_final_task_output(self, text: str) -> bool:
        """
        Check whether the LLM output contains all required task fields.

        Purpose:
        - Determine whether an LLM response represents a final structured task

        Inputs:
        - text (str): LLM output text

        Outputs:
        - bool: True if all required fields are present, otherwise False
        """
        required_fields = [
            "Object_color:",
            "Object_shape:",
            "Pickup_location:",
            "Destination_location:"
        ]
        return all(field in text for field in required_fields)

    def extract_task_fields(self, text: str):
        """
        Parse structured task fields from LLM output using regular expressions.

        Purpose:
        - Extract object attributes and locations from LLM-generated text
        - Ensure all required fields are present before accepting the task

        Inputs:
        - text (str): LLM output text

        Outputs:
        - dict or None: Dictionary of extracted fields if successful,
          otherwise None
        """

        patterns = {
            "Object_color": r"Object_color:\s*(\w+)",
            "Object_shape": r"Object_shape:\s*(\w+)",
            "Pickup_location": r"Pickup_location:\s*(\w+)",
            "Destination_location": r"Destination_location:\s*(\w+)",
        }

        result = {}

        # Extract each required field using regex
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if not match:
                return None
            result[key] = match.group(1)

        return result

    def build_clean_task_text(self, fields: dict) -> str:
        """
        Construct a clean and executable task description from parsed fields.

        Purpose:
        - Reformat extracted task fields into a standardized text layout

        Inputs:
        - fields (dict): Parsed task fields

        Outputs:
        - str: Clean task description string
        """
        return (
            f"Object_color: {fields['Object_color']}\n"
            f"Object_shape: {fields['Object_shape']}\n"
            f"Pickup_location: {fields['Pickup_location']}\n"
            f"Destination_location: {fields['Destination_location']}"
        )

    def auth_callback(self, msg: AuthState):
        """
        Handle face authentication state updates.

        Purpose:
        - Update authorization and lock state based on authentication results

        Inputs:
        - msg (AuthState): Face authentication status message

        Outputs:
        - None
        """

        # Only update internal state when authorization changes
        if self.authorized != msg.authorized:
            self.authorized = msg.authorized
            self.authorized_user = msg.user_name

            if self.authorized:
                self.locked = False
                self.get_logger().info(
                    f"LLM unlocked for user: {self.authorized_user}"
                )
            else:
                self.get_logger().info("Authentication lost or not authorized")

    def on_speech_input(self, msg: String):
        """
        Handle speech-to-text input and generate an LLM response.

        Purpose:
        - Receive transcribed speech input
        - Forward input to the LLM if authorized
        - Publish LLM response for text-to-speech
        - Extract and save structured task information if present

        Inputs:
        - msg (String): Speech-to-text input message

        Outputs:
        - None
        """
        user_text = msg.data.strip()
        if not user_text:
            return

        self.get_logger().info(f"[Speech Input] {user_text}")

        # ---- authorization gate ----
        if self.locked:
            reply = "I am locked. Please complete authentication first."
        else:
            # Default session name for speech interaction
            sid = "speech"

            if sid not in self.memory:
                self.memory[sid] = []

            messages = []

            # Add system prompt
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

            # Add conversation history
            for m in self.memory[sid]:
                messages.append(m)

            # Add current user input
            messages.append({
                "role": "user",
                "content": user_text
            })

            try:
                reply = self.client.chat(messages)
            except Exception as e:
                self.get_logger().error(f"OpenAI API error: {e}")
                reply = "Sorry, I encountered an error."

            # Write interaction back to memory
            self.memory[sid].append({"role": "user", "content": user_text})
            self.memory[sid].append({"role": "assistant", "content": reply})

        # Publish response to TTS
        out = String()
        out.data = reply
        self.speech_pub.publish(out)

        # ---- FINAL TASK HARD PARSE & SAVE ----
        fields = self.extract_task_fields(reply)

        if fields:
            clean_task = self.build_clean_task_text(fields)
            self.save_final_task(clean_task)

        self.get_logger().info(f"[LLM Reply] {reply}")

    def on_chat(self, req: Chat.Request, res: Chat.Response):
        """
        Handle incoming LLM chat service requests.

        Purpose:
        - Process text-based chat requests from external clients
        - Enforce authorization
        - Maintain session-based conversation history
        - Query the LLM and return its response
        - Extract and save structured tasks if detected

        Inputs:
        - req (Chat.Request): Incoming chat request
        - res (Chat.Response): Outgoing chat response

        Outputs:
        - Chat.Response: Filled response message
        """

        # ---- authorization gate ----
        if self.locked:
            res.assistant_text = (
                "LLM is locked. Please complete authentication first."
            )
            return res

        # Determine session ID
        sid = req.session_id.strip() or "default"

        # Reset session memory if requested
        if req.reset or sid not in self.memory:
            self.memory[sid] = []

        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Add conversation history
        for m in self.memory[sid]:
            messages.append(m)

        # Add current user input
        messages.append({
            "role": "user",
            "content": req.user_text
        })

        try:
            reply = self.client.chat(messages)
        except Exception as e:
            self.get_logger().error(f"OpenAI API error: {e}")
            reply = f"[ERROR] {e}"

        # Write interaction back to memory
        self.memory[sid].append({"role": "user", "content": req.user_text})
        self.memory[sid].append({"role": "assistant", "content": reply})

        # ---- FINAL TASK HARD PARSE & SAVE ----
        fields = self.extract_task_fields(reply)

        if fields:
            clean_task = self.build_clean_task_text(fields)
            self.save_final_task(clean_task)

        res.assistant_text = reply
        return res


def main():
    """
    Entry point for the LLM server node.

    Purpose:
    - Initialize ROS2
    - Create and spin the LLM server node
    - Cleanly shut down on exit

    Inputs:
    - None

    Outputs:
    - None
    """
    rclpy.init()
    node = LLMServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()



