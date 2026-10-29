"""Software execution of the intended device queue semantics, for contract testing."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Request:
    id: str
    prompt: list
    limit: int
    capacity: int
    eos: int
    prompt_position: int = 0
    generated: int = 0
    next_token: int = 0
    status: str = "active"
    cancel: bool = False
    output: deque = field(default_factory=deque)

class Scheduler:
    def __init__(self, engine, lanes=1):
        if lanes < 1: raise ValueError("No lanes")
        self.engine, self.lanes = engine, [None]*lanes
        self.state = engine.initial_state(lanes)
        self.requests = {}
        self.counters = dict(token_completions=0,active_lane_steps=0,available_lane_steps=0)

    def submit(self, request_id, prompt, limit, output_capacity=16, eos=None):
        if request_id in self.requests: raise ValueError("Duplicate request ID")
        if not prompt or limit<0 or output_capacity<1: raise ValueError("Invalid request")
        if any(not isinstance(x,(int,np.integer)) or not 0<=x<self.engine.cfg.vocab for x in prompt):
            raise ValueError("Invalid prompt tokens")
        eos = self.engine.cfg.eos if eos is None else eos
        if not 0<=eos<self.engine.cfg.vocab: raise ValueError("Invalid EOS")
        try: lane = self.lanes.index(None)
        except ValueError: raise RuntimeError("No free lane; advance scheduler to a boundary")
        r = Request(str(request_id),list(prompt),limit,output_capacity,eos)
        self.requests[request_id] = r
        self.lanes[lane] = r
        self.state[lane].fill(0)
        return lane

    def cancel(self, request_id): self.requests[request_id].cancel = True

    def collect(self, request_id):
        r = self.requests[request_id]
        result = list(r.output)
        r.output.clear()
        return result, r.status

    def advance(self, quantum=1):
        if quantum<1: raise ValueError("Quantum must be positive")
        for _ in range(quantum):
            active, tokens = [], []
            for lane,r in enumerate(self.lanes):
                if r is None: continue
                if r.cancel or r.generated >= r.limit:
                    r.status = "cancelled" if r.cancel else "complete"
                    self.lanes[lane] = None
                    continue
                emits = r.prompt_position >= len(r.prompt)-1
                if emits and len(r.output) >= r.capacity: continue
                active.append(lane)
                tokens.append(r.prompt[r.prompt_position] if r.prompt_position<len(r.prompt) else r.next_token)
            self.counters["available_lane_steps"] += len(self.lanes)
            if not active: continue
            logits,state = self.engine.step(np.asarray(tokens,dtype=np.int64), self.state[active])
            self.state[active] = state
            self.counters["active_lane_steps"] += len(active)
            for j,lane in enumerate(active):
                r = self.lanes[lane]
                r.prompt_position += int(r.prompt_position<len(r.prompt))
                if r.prompt_position < len(r.prompt): continue
                token = int(np.argmax(logits[j]))
                r.next_token = token
                r.output.append(token)
                r.generated += 1
                self.counters["token_completions"] += 1
                if token == r.eos or r.generated == r.limit:
                    r.status = "complete"
                    self.lanes[lane] = None

    def reset(self):
        for r in self.lanes:
            if r is not None: r.status = "reset"
        self.lanes = [None]*len(self.lanes)
        self.state.fill(0)
