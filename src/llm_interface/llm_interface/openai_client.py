#!/home/wbo/hrs_vision_env/bin/python3
from openai import OpenAI


class OpenAIChatClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model

    def chat(self, messages: list[dict]) -> str:
        """
        messages:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
        ]
        """
        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        # extract text from response
        texts = []
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        texts.append(c.text)

        return "\n".join(texts).strip()
