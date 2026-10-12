from dataclasses import dataclass
import tiktoken
import re

CITATION_RE = re.compile(r"\s*\[S\d+\]")

@dataclass
class Turn:
    user: str
    assistant: str
    
class ConversationMemory:
    def __init__(self, budget_tokens: int = 3000):
        self.turns = []
        self.budget = budget_tokens
        self.dropped = 0
        self._encoding = tiktoken.get_encoding("o200k_base")
        
    def add(self, user: str, assistant: str) -> None:
        self.turns.append(Turn(user, CITATION_RE.sub("", assistant)))
        self._trim()
        
    def as_input(self) -> list[dict]:
        return [m for t in self.turns for m in ({"role": "user", "content": t.user}, {"role": "assistant", "content": t.assistant})]
    
    def tokens(self) -> int:
        return sum(len(self._encoding.encode(t.user + t.assistant)) for t in self.turns)
    
    def _trim(self) -> None:
        while len(self.turns) > 1 and self.tokens() > self.budget:
            self.turns.pop(0)
            self.dropped += 1
            
    