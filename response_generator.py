from openai import OpenAI
from abc import ABC, abstractmethod

class IResponseGenerator(ABC):
    @abstractmethod
    def query(self, user_prompt):
        pass

class OpenaiResponseGenerator(IResponseGenerator):
    def __init__(self, base_url, api_key, model, hystory_length=20):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = "You are a helpful AI assistant"
        self.chat_history = []
        self.hystory_length = hystory_length

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def add_to_chat_history(self, message):
        sys_prompt_not_in_history = not any(
            msg['role'] == 'system' and msg['content'] == self.system_prompt for msg in self.chat_history
        )
        if self.system_prompt and sys_prompt_not_in_history:
            self.chat_history.insert(0, {'role': 'system', 'content': self.system_prompt})

        self.chat_history.append({'role': 'user', 'content': message})

        if len(self.chat_history) > self.hystory_length:
            self.chat_history = self.chat_history[-(self.hystory_length):]

    def query(self, user_prompt):
        try:
            self.add_to_chat_history(user_prompt)

            response = self.client.chat.completions.create(
                messages=self.chat_history,
                model=self.model
            )

            self.chat_history.append({'role': 'assistant', 'content': response.choices[0].message.content})

            if len(self.chat_history) > self.hystory_length:
                self.chat_history = self.chat_history[-(self.hystory_length):]

            return response.choices[0].message.content
        except Exception as e:
            return f"Exception occured: {str(e)}"
