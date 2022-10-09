from dataclasses import dataclass


@dataclass
class Case:
    runmode: int = 0,
    retain: int = 1,
    resume: int = 0,
    retain_operators: int = 0,
    resume_operators: int = 0,
    
     