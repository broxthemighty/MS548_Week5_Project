"""
domain.py
Author: Matt Lindborg
Course: MS548 - Advanced Programming Concepts and AI
Assignment: Week 4
Date: 10/01/2025

Purpose:
This file defines the core domain model for the Learnflow Base application.
It contains the enumerations and data structures used to represent user
learning activity.
    - Introducing a LearningLog class (objects instead of raw strings).
    - Storing multiple entries per type (lists of logs).
    - Adding timestamp and optional mood fields for future analysis.
This separation allows the GUI (ui.py) and business logic (service.py)
to be improved while still sharing a consistent data model.
"""

# import dataclass to simplify object creation
from dataclasses import dataclass, field

# import Enum to define fixed categories of entries
from enum import Enum

# import typing helpers
from typing import Dict, List

# import datetime so we can automatically timestamp log entries
from datetime import datetime

class EntryType(str, Enum):
    """
    Supported entry types.
    Each entry represents a category of user activity.
    Stored as strings for readability in save/load and gui.
    """
    Goal = "Goal"       # represents a learning goal
    Skill = "Skill"     # represents a tracked skill or topic
    Session = "Session" # represents a daily learning session
    Notes = "Notes"     # represents reflections or simply notes

@dataclass
class LearningLog:
    """
    Base log entry class for all types of learning records.
    Each log is timestamped, stores user text, and includes mood.
    """
    entry_type: EntryType   # which type of entry this belongs to
    text: str               # the actual content the user entered
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )                       # when the entry was created, default = now
    mood: str = ""          # optional mood

    def summary(self) -> str:
        """
        Return a one-line summary for display in gui or logs.
        Example: "Notes: Felt stuck [Mood: frustrated]"
        """
        mood_str = f" [Mood: {self.mood}]" if self.mood else ""
        return f"{self.entry_type.value}: {self.text}{mood_str}"

@dataclass
class GoalLog(LearningLog):
    """
    Class representing a learning goal.
    Unique attributes:
        - status: current state of the goal (planned, in-progress, done, etc).
    Unique methods:
        - update_status(): change status string.
    """
    status: str = "planned"  # default status when created

    def update_status(self, new_status: str) -> str:
        """
        Update goal status and return a confirmation string.
        """
        self.status = new_status
        return f"Goal '{self.text}' updated to status: {new_status}"
    

@dataclass
class ReflectionLog(LearningLog):
    """
    Class representing a reflective note.
    Unique attributes:
        - mood: sentiment analysis result (inherited from base).
    Unique methods:
        - analyze_mood(): run sentiment analysis (TextBlob).
    """

@dataclass
class LearnflowState:
    """
    The overall state of the application.
    Stores lists of LearningLog objects for each EntryType.
    This makes it easy to:
      - Append new logs instead of overwriting.
      - Retrieve full history later (CSV export, history viewer).
    """
    entries: Dict[EntryType, List[LearningLog]] = field(
        default_factory=lambda: {e: [] for e in EntryType} # initialize dict with empty lists
    )
