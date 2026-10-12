from dataclasses import dataclass

@dataclass
class Turn:
    user: str
    assistant: str
    
class ConversationMemory:
    def __init__(self):
        self.turns: list[Turn] = []
        
    def add(self, user: str, assistant: str) -> None:
        self.turns.append(Turn(user, assistant))
        
    def as_input(self) -> list[dict]:
        return [m for t in self.turns for m in ({"role": "user", "content": t.user}, {"role": "assistant", "content": t.assistant})]
    